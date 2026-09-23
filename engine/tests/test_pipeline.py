from __future__ import annotations

from copy import deepcopy
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from crosscut.auto_anchor import AutoAnchorResult
from crosscut.acquisition import _build_acquisition_artifact
from crosscut.cut_local_sync import (
    MEMBER_ACTIVATION as MEMBER_CUT_LOCAL_SYNC_ACTIVATION,
)
from crosscut.direct_member_visual import (
    DirectMemberVisualConfig,
    DirectMemberVisualSelectionError,
)
from crosscut.direct_member_transition import (
    OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT,
)
from crosscut.direct_member_rendered_proxy import (
    DirectMemberRenderedProxyPreflight,
)
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from crosscut.draft import DraftSegment, GroupDraftTimeline
from crosscut.editorial_motion import EditorialMotionPlan, EditorialMotionSegment
from crosscut.group_transition import GroupTransitionPlan
from crosscut.ive import AspectRatio, EditMode, IVEMember, SourceMode
from crosscut.member_draft import (
    MemberDraftSegment,
    MemberDraftSelectionDiagnostics,
    MemberDraftTimeline,
    MemberSegmentKind,
    SegmentCropSample,
    SegmentSubjectSample,
)
from crosscut.member_post_render_qa import (
    AUTOMATIC_IVE_MEMBER_APP_RESULT_EXPOSURE_CONTRACT,
    AUTOMATIC_IVE_MEMBER_POST_RENDER_QA_CONTRACT,
    RENDER_RECEIPT_CONTRACT as MEMBER_RENDER_RECEIPT_CONTRACT,
)
from crosscut.pipeline import (
    _BoundarySearchBudget,
    DEFAULT_IVE_PROFESSIONAL_MONTAGE_PROFILE_PATH,
    DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_FILE_SHA256,
    DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_PATH,
    ELIGIBLE_SOURCE_INDEX_DOMAIN,
    OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY,
    REQUEST_SOURCE_INDEX_DOMAIN,
    _DirectFullFrameSpan,
    _GroupRenderCandidateBundle,
    _apply_bounded_human_target_proxy_rank,
    _alignment_eligibility,
    _attempt_proxy_triggered_partial_reedit,
    _attach_direct_member_editorial_motion_tracks,
    _attach_member_editorial_motion_tracks,
    _automatic_member_direct_full_frame_ratios,
    _capabilities_payload,
    _common_master_range,
    _direct_full_frame_base_partition_profile,
    _direct_full_frame_proof_directed_child_profiles,
    _direct_full_frame_shifted_partition_profiles,
    _direct_full_frame_visual_frontier_split_child_profiles,
    _direct_partition_failure_bottlenecks,
    _direct_partition_future_operation_state_key,
    _direct_partition_queue_signature_digest,
    _direct_strict_extra_quota_assignment_candidates,
    _direct_strict_extra_quota_assignment_failure_is_proven,
    _DIRECT_MEMBER_FAILURE_REPORT_MAX_BYTES,
    _DIRECT_MEMBER_MIN_SWITCHES_PER_MINUTE,
    _DIRECT_MEMBER_MAX_PARTITION_PROFILE_UPPER_BOUND,
    _derive_complete_group_candidate,
    _enforce_automatic_stage_edit_post_render_qa,
    _official_target_fancam_direct_full_frame_segments,
    _member_anchor_trim,
    _member_render_segments,
    _member_render_selection_basis,
    _materialize_ranked_visual_proxy_candidates,
    _load_learned_reviewer_session,
    _load_stage_edit_motion_cut_profile,
    _load_professional_reference_profile,
    _load_professional_reference_cadence_profile,
    _review_visual_timeline_for_render,
    _review_complete_group_candidates,
    _report_direct_member_selection_failure,
    _safe_slug,
    _source_index_mapping,
    _stage_edit_montage_analysis_payload,
    _stage_edit_motion_cut_profile_manifest_payload,
    _stage_edit_semimarkov_config_from_montage_policy,
    _report_stage_edit_motion_cut_profile,
    _rendered_montage_config_from_policy,
    _strip_superseded_exact_stage_edit_shot_clocks,
    _verify_automatic_member_borderline_alignments,
    _with_index_domain,
    analyze,
)
from crosscut.post_render_qa import (
    PostRenderArtifactSet,
    fingerprint_post_render_artifacts,
)
from crosscut.professional_montage_policy import (
    load_professional_montage_policy,
)
from crosscut.professional_motion_cut_profile import (
    load_professional_motion_cut_profile,
)
from crosscut.stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
)
from crosscut.stage_edit_semimarkov_beam import (
    HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
    HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
)
from crosscut.stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
)
from crosscut.source_derivative_diversity import (
    SourceDerivativeConfig,
    SourceDerivativeDiversityResult,
)
from crosscut.audio import AlignmentResult, OffsetEstimate, SourceAlignment
from crosscut.media import AudioStreamInfo, MediaInfo, VideoStreamInfo
from crosscut.render import (
    RENDER_PLAN_SOURCE_INDEX_DOMAIN,
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    SourceLookAdjustment,
    TransitionKind,
)
from crosscut.rendered_montage_grammar_review import (
    RenderedMontageGrammarConfig,
)
from crosscut.rendered_candidate_selection import RenderedCandidate, RenderedCandidateReview
from crosscut.rendered_proxy_review import RenderedProxyReview
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.request import AnalysisRequest, MemberAnchorRequest
from crosscut.tracking import (
    MemberTrackObservation,
    MemberTrackingResult,
    TrackingFailure,
    TrackingState,
)
from crosscut.timeline import NoCompleteTimelinePath, ShotType
from crosscut.timewarp import (
    TimeWarpConfig,
    TimeWarpControlPoint,
    TimeWarpMap,
    TimeWarpResidualDiagnostics,
)
from crosscut.vision import SceneChange
from crosscut.visual_boundary_repair import VisualBoundaryRepairPlan
from crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan


def _media(path: str, duration: float) -> MediaInfo:
    return MediaInfo(
        path=path,
        format_name="mov,mp4",
        duration_seconds=duration,
        start_time_seconds=0.0,
        size_bytes=1,
        bit_rate=1,
        audio_streams=(),
        video_streams=(),
        tags={},
    )


def _source_derivative_diversity_fixture(
    source_count: int,
) -> SourceDerivativeDiversityResult:
    """Return verified-distinct synthetic metadata for pipeline unit tests."""

    return SourceDerivativeDiversityResult(
        canonical_sample_times_seconds=(),
        source_decode_audit=tuple(
            {
                "source_index": index,
                "decoded_sample_count": 0,
                "clock_contract_complete": True,
            }
            for index in range(source_count)
        ),
        pairs=(),
        family_ids=tuple(range(source_count)),
        family_consistency_conflict_count=0,
        clock_contract_complete=True,
        production_ready=True,
        audit_sha256="a" * 64,
        config=SourceDerivativeConfig(),
    )


def _professional_direct_partition_profile(
    timing: GroupDraftTimeline,
):
    duration = timing.common_master_end - timing.common_master_start
    minimum_span_count = (
        math.ceil(
            duration * _DIRECT_MEMBER_MIN_SWITCHES_PER_MINUTE / 60.0 - 1e-9
        )
        + 1
    )
    return _direct_full_frame_base_partition_profile(
        timing.segments,
        minimum_span_count=minimum_span_count,
    )


def _current_motion_cut_profile_fixture(
    root: Path,
    montage_policy,
):
    """Write and strict-load a current-schema profile for pipeline tests only."""

    from engine.tests.test_professional_motion_cut_profile import (
        _reseal,
        _valid_payload,
    )

    payload = _valid_payload()
    payload["provenance"]["source_montage_profile"] = dict(
        montage_policy.profile_identity
    )
    _reseal(payload)
    from crosscut.professional_motion_cut_profile import (
        ProfessionalMotionCutProfile,
    )

    path = ProfessionalMotionCutProfile.from_mapping(payload).write_json(
        root / "current-motion-cut-profile.json"
    )
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    profile = load_professional_motion_cut_profile(
        path,
        expected_file_sha256=file_sha256,
        verify_source_inputs=False,
    )
    return profile, path, file_sha256


def _current_montage_policy_fixture(root: Path):
    """Write and strict-load a hermetic montage profile for unit tests."""

    from engine.tests.test_professional_montage_policy import (
        _install_positive_duration_role,
        _profile,
        _write,
    )

    profile = _profile()
    _install_positive_duration_role(
        profile,
        role_name="build",
        close_values=[0.0, 0.0, 0.0]
        + [0.18 + 0.01 * index for index in range(17)],
        nonwide_values=[0.0, 0.0, 0.0]
        + [0.50 + 0.01 * index for index in range(17)],
    )
    _install_positive_duration_role(
        profile,
        role_name="bridge_like",
        close_values=[0.0, 0.0, 0.0] + [0.32] * 17,
        nonwide_values=[0.60] * 20,
    )
    path = root / "current-montage-profile.json"
    _write(path, profile)
    return load_professional_montage_policy(path), path


def _tracking(source: str, source_local_time: float) -> MemberTrackingResult:
    return MemberTrackingResult(
        member=IVEMember.GAEUL,
        source_id=source,
        fps=30.0,
        backend_name="scripted",
        observations=(
            MemberTrackObservation(
                frame_index=round(source_local_time * 30.0),
                timestamp_seconds=source_local_time,
                bbox_xyxy=(0.1, 0.2, 0.3, 0.8),
                confidence=1.0,
                state=TrackingState.TRACKED,
                member=IVEMember.GAEUL,
                source_id=source,
            ),
        ),
        terminal_failure=None,
    )


def _analysis_media(path: Path, duration: float = 30.0) -> MediaInfo:
    return MediaInfo(
        path=str(path),
        format_name="mov,mp4",
        duration_seconds=duration,
        start_time_seconds=0.0,
        size_bytes=1,
        bit_rate=1,
        audio_streams=(
            AudioStreamInfo(
                index=1,
                codec_name="aac",
                sample_rate=48_000,
                channels=2,
                channel_layout="stereo",
                duration_seconds=duration,
                bit_rate=128_000,
            ),
        ),
        video_streams=(
            VideoStreamInfo(
                index=0,
                codec_name="h264",
                width=1920,
                height=1080,
                frame_rate=30.0,
                duration_seconds=duration,
                bit_rate=4_000_000,
            ),
        ),
        tags={},
    )


def _portrait_analysis_media(path: Path, duration: float = 40.0) -> MediaInfo:
    info = _analysis_media(path, duration=duration)
    stream = info.video_streams[0]
    return MediaInfo(
        path=info.path,
        format_name=info.format_name,
        duration_seconds=info.duration_seconds,
        start_time_seconds=info.start_time_seconds,
        size_bytes=info.size_bytes,
        bit_rate=info.bit_rate,
        audio_streams=info.audio_streams,
        video_streams=(
            VideoStreamInfo(
                index=stream.index,
                codec_name=stream.codec_name,
                width=1080,
                height=1920,
                frame_rate=stream.frame_rate,
                duration_seconds=stream.duration_seconds,
                bit_rate=stream.bit_rate,
            ),
        ),
        tags=info.tags,
    )


def _linear_timewarp(
    *,
    duration: float = 30.0,
    global_offset: float = 0.0,
    rate: float = 1.004,
) -> TimeWarpMap:
    return TimeWarpMap(
        global_offset_seconds=global_offset,
        master_duration_seconds=duration,
        source_duration_seconds=global_offset + rate * duration,
        control_points=(
            TimeWarpControlPoint(0.0, global_offset),
            TimeWarpControlPoint(duration, global_offset + rate * duration),
        ),
        anchors=(),
        residuals=TimeWarpResidualDiagnostics(None, None, None, 0),
        config=TimeWarpConfig(maximum_drift_rate=0.01),
        scalar_fallback=False,
        fallback_reason=None,
    )


def _rotating_timing_draft(
    *,
    source_count: int = 10,
    common_master_start: float = 0.0,
    duration_seconds: float = 40.0,
    unit_seconds: float = 1.0,
) -> GroupDraftTimeline:
    segment_count = int(round(duration_seconds / unit_seconds))
    if not math.isclose(
        segment_count * unit_seconds,
        duration_seconds,
        abs_tol=1e-9,
    ):
        raise ValueError("synthetic timing duration must contain whole units")
    segments = tuple(
        DraftSegment(
            segment_index=index,
            source_index=index % source_count,
            canonical_master_start=(
                common_master_start + index * unit_seconds
            ),
            canonical_master_end=(
                common_master_start + (index + 1) * unit_seconds
            ),
            output_start_seconds=index * unit_seconds,
            source_local_start=(
                common_master_start + index * unit_seconds
            ),
            source_local_end=(
                common_master_start + (index + 1) * unit_seconds
            ),
            audio_offset_seconds=0.0,
            sync_score=1.0,
            estimated_beat_count=unit_seconds * 2.0,
            start_boundary_reasons=(
                ("range_edge",)
                if index == 0
                else ("phrase",)
                if index % 4 == 0
                else ("strong_beat",)
                if index % 2 == 0
                else ("beat",)
            ),
            end_boundary_reasons=(
                ("range_edge",)
                if index == segment_count - 1
                else ("phrase",)
                if (index + 1) % 4 == 0
                else ("strong_beat",)
                if (index + 1) % 2 == 0
                else ("beat",)
            ),
        )
        for index in range(segment_count)
    )
    return GroupDraftTimeline(
        source_count=source_count,
        common_master_start=common_master_start,
        common_master_end=common_master_start + duration_seconds,
        beat_period_seconds=0.5,
        audio_offsets=(0.0,) * source_count,
        sync_scores=(1.0,) * source_count,
        segments=segments,
    )


def _direct_partition_failure_fixture(
    *,
    span_count: int = 50,
    failed_segment_index: int,
    frontier_boundary_index: int,
    zero_local_boundary_indices: tuple[int, ...] = (),
    empty_valid_side_boundary_index: int | None = None,
) -> dict[str, object]:
    boundaries = []
    for boundary_index in range(span_count - 1):
        left_valid = list(range(10))
        right_valid = list(range(10))
        if boundary_index == empty_valid_side_boundary_index:
            right_valid = []
        boundaries.append(
            {
                "boundary_index": boundary_index,
                "left_valid_source_indices": left_valid,
                "right_valid_source_indices": right_valid,
                "forward_reachable_left_source_indices": (
                    [2] if boundary_index <= frontier_boundary_index else []
                ),
                "backward_reachable_right_source_indices": [],
                "content_valid_cross_source_edge_count": (
                    90 if left_valid and right_valid else 0
                ),
                "nonsevere_cross_source_edge_count": (
                    0
                    if boundary_index in zero_local_boundary_indices
                    else 2
                ),
                "reachable_nonsevere_transition_count": 0,
            }
        )
    return {
        "failure": "validity_quota_path_exhausted",
        "constraint_infeasibility_proven": True,
        "failed_at_segment_index": failed_segment_index,
        "hard_boundary_connectivity": {
            "analysis_basis": (
                "content_valid_nonsevere_no_repeat_reachability_v1"
            ),
            "quota_and_duration_considered": False,
            "boundaries": boundaries,
            "zero_local_nonsevere_transition_boundary_indices": list(
                zero_local_boundary_indices
            ),
        },
    }


def _visual_frontier_span_diagnostic(
    span: object,
    *,
    segment_index: int,
    source_count: int = 10,
    native_cut_master_seconds: float | None = None,
    sample_safety_change: bool = False,
) -> dict[str, object]:
    start = float(getattr(span, "canonical_master_start"))
    end = float(getattr(span, "canonical_master_end"))
    duration = end - start
    sample_times = (
        start + min(1.0 / 29.97, duration * 0.08),
        start + min(0.20, duration * 0.18),
        start + duration * 0.25,
        start + duration * 0.50,
        start + duration * 0.75,
        end - min(0.18, duration * 0.20),
    )
    sources: list[dict[str, object]] = []
    for source_index in range(source_count):
        changing = sample_safety_change and source_index == 0
        quality = [0.10, 0.10, 0.10, 0.90, 0.90, 0.90] if changing else [0.90] * 6
        samples = []
        for sample_index, master_seconds in enumerate(sample_times):
            safe = not changing or sample_index >= 3
            samples.append(
                {
                    "canonical_master_seconds": master_seconds,
                    "descriptor_available": True,
                    "attention_basis": "aggregate_human_silhouette" if safe else None,
                    "face_count": 1 if safe else 0,
                    "detected_person_count": 1 if safe else 0,
                    "formation_person_count": 1 if safe else 0,
                    "silhouette_confidence": 0.95 if safe else 0.0,
                    "foreground_obstruction_score": 0.0,
                }
            )
        sources.append(
            {
                "eligible_source_index": source_index,
                "valid": not changing and not (
                    source_index == 0 and native_cut_master_seconds is not None
                ),
                "rejection_reasons": (
                    ["persistent_low_visual_quality_or_foreground_obstruction"]
                    if changing
                    else ["source_native_scene_cut_inside_rendered_span"]
                    if source_index == 0 and native_cut_master_seconds is not None
                    else []
                ),
                "native_scene_cut_master_seconds": (
                    [native_cut_master_seconds]
                    if source_index == 0 and native_cut_master_seconds is not None
                    else []
                ),
                "quality_sample_scores": quality,
                "portrait_fancam_lighting_quality_rescue_applied": False,
                "samples": samples,
            }
        )
    return {
        "segment_index": segment_index,
        "canonical_master_start": start,
        "canonical_master_end": end,
        "sources": sources,
    }


def _direct_partition_exact_capacity_failure_fixture(
    *,
    state_limit: int,
) -> dict[str, object]:
    return {
        "failure": "search_capacity_exhausted",
        "constraint_infeasibility_proven": False,
        "visual_gates_relaxed": False,
        "search": {
            "exact_lifeline": {
                "invoked": True,
                "explored_state_count": state_limit,
                "memoized_failed_state_count": state_limit - 1,
                "state_capacity_exhausted": True,
                "duration_state_quantization_applied": False,
                "root_candidate_diagnostics": [
                    {
                        "eligible_source_index": 3,
                        "search_outcome": "search_capacity_exhausted",
                    }
                ],
            }
        },
    }


def _direct_partition_exact_hard_quota_failure_fixture(
    *,
    span_count: int,
    valid_counts: tuple[int, ...],
    explored_state_count: int = 275_414,
) -> dict[str, object]:
    source_count = len(valid_counts)
    base_quota, extra_quota_count = divmod(span_count, source_count)
    return {
        "failure": "validity_quota_path_exhausted",
        "constraint_infeasibility_proven": True,
        "failed_at_segment_index": span_count - 2,
        "source_quotas": [
            base_quota + int(index < extra_quota_count)
            for index in range(source_count)
        ],
        "valid_count_by_source": [
            {
                "eligible_source_index": index,
                "quota": base_quota + int(index < extra_quota_count),
                "valid_span_count": count,
                "invalid_span_count": span_count - count,
            }
            for index, count in enumerate(valid_counts)
        ],
        "visual_gates_relaxed": False,
        "search": {
            "exact_lifeline": {
                "invoked": True,
                "explored_state_count": explored_state_count,
                "memoized_failed_state_count": explored_state_count,
                "state_capacity_exhausted": False,
                "duration_state_quantization_applied": False,
                "root_candidate_diagnostics": [
                    {
                        "eligible_source_index": index,
                        "search_outcome": "subtree_infeasible",
                    }
                    for index in range(source_count)
                ],
            }
        },
    }


class _MemoryReporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, event_type: str, **payload: object) -> None:
        self.events.append((event_type, payload))

    def status(self, message: str) -> None:
        self.emit("status", message=message)

    def progress(self, value: float, message: str) -> None:
        self.emit("progress", value=value, message=message)

    def log(self, message: str) -> None:
        self.emit("log", message=message)


def _persist_mock_bound_qa_report(
    artifact_set: object,
    *,
    verdict: str,
    output_path: Path | None = None,
) -> tuple[SimpleNamespace, Path]:
    passed = verdict == "PASS"
    fingerprints = fingerprint_post_render_artifacts(artifact_set)  # type: ignore[arg-type]
    from engine.tests.test_review_first_runtime import _group_integrity_metrics
    payload = {
        "metrics": _group_integrity_metrics(),
        "issue_count": 0 if passed else 1,
        "fail_issue_count": 0,
        "conditional_issue_count": 0 if passed else 1,
        "issues": (
            []
            if passed
            else [
                {
                    "code": "fixture_conditional_review_failure",
                    "severity": "conditional",
                }
            ]
        ),
        "hard_failures": [],
        "warnings": [],
        "passed": passed,
        "input_artifact_fingerprints": fingerprints,
    }
    report = SimpleNamespace(
        verdict=SimpleNamespace(value=verdict),
        passed=passed,
        analysis_basis="fixture_post_render_qa",
        to_dict=lambda: dict(payload),
    )
    destination = Path(
        output_path or artifact_set.default_report_path  # type: ignore[attr-defined]
    )
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return report, destination


def _persist_post_render_artifact_shell(
    manifest_path: Path,
    manifest: dict[str, object],
) -> None:
    base = manifest_path.name[: -len("-analysis.json")]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (manifest_path.parent / f"{base}-render-plan.json").write_text(
        "{}", encoding="utf-8"
    )
    (
        manifest_path.parent
        / f"{base}-rendered-candidate-selection.json"
    ).write_text("{}", encoding="utf-8")
    (manifest_path.parent / f"{base}-group-crosscut.mp4").write_bytes(
        b"fixture final mp4"
    )


class _VisualSelectionStub:
    def __init__(
        self,
        source_indices: tuple[int, ...],
        source_count: int,
        *,
        usage_policy: str = "heterogeneous_role_aware_safe_hold_v1",
    ) -> None:
        self.source_indices = source_indices
        self.usage_policy = usage_policy
        self.optimizer_search_diagnostics: dict[str, object] = {}
        self.source_counts = tuple(
            source_indices.count(index) for index in range(source_count)
        )

    def bind(
        self,
        canonical_spans: tuple[tuple[float, float], ...],
        config: DirectMemberVisualConfig,
    ) -> "_VisualSelectionStub":
        self.selected_boundary_master_seconds = tuple(
            float(end) for _start, end in canonical_spans[:-1]
        )
        self.config = config
        source_durations = [0.0] * len(self.source_counts)
        for source_index, (start, end) in zip(
            self.source_indices,
            canonical_spans,
            strict=True,
        ):
            source_durations[source_index] += float(end) - float(start)
        self.source_durations_seconds = tuple(source_durations)
        self.source_target_durations_seconds = tuple(source_durations)
        self.source_duration_relative_deviations = tuple(
            0.0 for _ in source_durations
        )
        self.source_count_minimums = self.source_counts
        self.source_count_maximums = self.source_counts
        self.source_duration_minimums_seconds = tuple(source_durations)
        self.source_duration_maximums_seconds = tuple(source_durations)
        source_geometry = direct_member_source_frame_geometry((1080, 1920))
        self.source_frame_geometries = tuple(
            source_geometry for _ in self.source_counts
        )
        validity_sources = tuple(
            SimpleNamespace(
                eligible_source_index=source_index,
                native_scene_cut_source_seconds=(),
                samples=tuple(
                    SimpleNamespace(
                        descriptor=SimpleNamespace(
                            color_embedding=(
                                0.5,
                                128.0 / 255.0,
                                128.0 / 255.0,
                                0.0,
                                0.0,
                                0.0,
                                0.45,
                                0.15,
                            )
                        )
                    )
                    for _ in range(3)
                ),
            )
            for source_index in range(len(self.source_counts))
        )
        self.span_validity = tuple(
            SimpleNamespace(sources=validity_sources)
            for _ in canonical_spans
        )
        self.boundaries = tuple(
            SimpleNamespace(
                boundary_index=boundary_index,
                cut_applied=(outgoing != incoming),
                mismatch=SimpleNamespace(
                    cinematic_bridge=None,
                    transform_repair=None,
                    dense_attention_validation_required=False,
                    warning_flags=(),
                    severe_flags=(),
                    evidence_component_count=0,
                    motion_connection=SimpleNamespace(
                        both_moving=True,
                        timestamp_bound_adjacent=False,
                    ),
                ),
            )
            for boundary_index, (outgoing, incoming) in enumerate(
                zip(
                    self.source_indices,
                    self.source_indices[1:],
                )
            )
        )
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "analysis_basis": "synthetic_non_identity_geometry_test",
            "selected_source_indices": list(self.source_indices),
            "distinct_source_count": len(set(self.source_indices)),
            "source_usage": [
                {
                    "eligible_source_index": index,
                    "quota": count,
                    "segment_count": count,
                }
                for index, count in enumerate(self.source_counts)
            ],
            "counts_differ_by_at_most_one": (
                max(self.source_counts) - min(self.source_counts) <= 1
            ),
            "same_source_adjacent_count": 0,
            "aggregate": {
                "selected_boundary_count": len(self.source_indices) - 1,
                "warning_boundary_count": 0,
                "content_validity": {"invalid_candidate_count": 0},
                "raw_embeddings_persisted": False,
                "face_identity_inference_used": False,
            },
            "span_content_validity": [
                {
                    "segment_index": segment_index,
                    "sources": [
                        {
                            "eligible_source_index": source_index,
                            "valid": True,
                            "rejection_reasons": [],
                        }
                        for source_index in range(len(self.source_counts))
                    ],
                }
                for segment_index in range(len(self.source_indices))
            ],
            "boundaries": [
                {
                    "boundary_index": index,
                    "outgoing_eligible_source_index": outgoing,
                    "incoming_eligible_source_index": incoming,
                }
                for index, (outgoing, incoming) in enumerate(
                    zip(self.source_indices, self.source_indices[1:])
                )
            ],
            "privacy": {
                "face_identity_inference_used": False,
                "raw_pose_silhouette_layout_embeddings_persisted": False,
            },
        }


def _bound_visual_selection(
    selection: _VisualSelectionStub,
):
    def choose(**kwargs: object) -> _VisualSelectionStub:
        spans = kwargs["canonical_spans"]
        config = kwargs["config"]
        assert isinstance(spans, tuple)
        assert isinstance(config, DirectMemberVisualConfig)
        return selection.bind(spans, config)

    return choose


def _visual_choice(
    index: int,
    source: int,
    start: float,
    end: float,
) -> VisualTimelineChoice:
    return VisualTimelineChoice(
        segment_index=index,
        source_index=source,
        canonical_master_start=start,
        canonical_master_end=end,
        source_local_start=start,
        source_local_end=end,
        quality_score=0.8,
        composition_score=0.8,
        boundary_score=0.8,
        shot_scale=0.4,
        shot_type=ShotType.MEDIUM,
        shot_id=f"{source}:shot-{index}",
    )


class PipelineTests(unittest.TestCase):
    def test_default_ive_montage_profile_is_release_v2(self) -> None:
        self.assertEqual(
            DEFAULT_IVE_PROFESSIONAL_MONTAGE_PROFILE_PATH,
            Path(
                "/Volumes/CrossCutMedia/CrossCut-Training/release/profiles/"
                "ive-professional-montage-grammar-v2.json"
            ),
        )

    def test_rendered_montage_config_binds_hash_bound_role_policy_and_cadence(
        self,
    ) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        policy, _profile_path = _current_montage_policy_fixture(
            Path(temporary.name)
        )
        roles = {
            item.role: item
            for item in policy.role_conditioned_duration_policies
        }
        self.assertEqual(
            {"build", "chorus_like", "bridge_like"},
            set(roles),
        )
        config = _rendered_montage_config_from_policy(
            policy,
            require_complete_role_duration_policy=True,
        )

        field_bindings = {
            "minimum_closeup_fraction": (
                "chorus_like",
                "local_minimum_close_duration_fraction",
            ),
            "minimum_nonwide_fraction": (
                "chorus_like",
                "local_minimum_nonwide_duration_fraction",
            ),
            "maximum_consecutive_wide_run_seconds": (
                "chorus_like",
                "maximum_consecutive_wide_run_seconds",
            ),
            "chorus_role_pooled_product_minimum_closeup_fraction": (
                "chorus_like",
                "pooled_product_minimum_close_duration_fraction",
            ),
            "chorus_role_pooled_product_minimum_nonwide_fraction": (
                "chorus_like",
                "pooled_product_minimum_nonwide_duration_fraction",
            ),
            "chorus_role_preferred_closeup_fraction": (
                "chorus_like",
                "preferred_close_duration_fraction",
            ),
            "chorus_role_preferred_nonwide_fraction": (
                "chorus_like",
                "preferred_nonwide_duration_fraction",
            ),
            "chorus_role_preferred_maximum_consecutive_wide_run_seconds": (
                "chorus_like",
                "preferred_maximum_consecutive_wide_run_seconds",
            ),
            "build_minimum_closeup_fraction": (
                "build",
                "local_minimum_close_duration_fraction",
            ),
            "build_minimum_nonwide_fraction": (
                "build",
                "local_minimum_nonwide_duration_fraction",
            ),
            "build_pooled_product_minimum_closeup_fraction": (
                "build",
                "pooled_product_minimum_close_duration_fraction",
            ),
            "build_pooled_product_minimum_nonwide_fraction": (
                "build",
                "pooled_product_minimum_nonwide_duration_fraction",
            ),
            "build_preferred_closeup_fraction": (
                "build",
                "preferred_close_duration_fraction",
            ),
            "build_preferred_nonwide_fraction": (
                "build",
                "preferred_nonwide_duration_fraction",
            ),
            "build_maximum_consecutive_wide_run_seconds": (
                "build",
                "maximum_consecutive_wide_run_seconds",
            ),
            "build_preferred_maximum_consecutive_wide_run_seconds": (
                "build",
                "preferred_maximum_consecutive_wide_run_seconds",
            ),
            "bridge_minimum_closeup_fraction": (
                "bridge_like",
                "local_minimum_close_duration_fraction",
            ),
            "bridge_minimum_nonwide_fraction": (
                "bridge_like",
                "local_minimum_nonwide_duration_fraction",
            ),
            "bridge_pooled_product_minimum_closeup_fraction": (
                "bridge_like",
                "pooled_product_minimum_close_duration_fraction",
            ),
            "bridge_pooled_product_minimum_nonwide_fraction": (
                "bridge_like",
                "pooled_product_minimum_nonwide_duration_fraction",
            ),
            "bridge_preferred_closeup_fraction": (
                "bridge_like",
                "preferred_close_duration_fraction",
            ),
            "bridge_preferred_nonwide_fraction": (
                "bridge_like",
                "preferred_nonwide_duration_fraction",
            ),
            "bridge_maximum_consecutive_wide_run_seconds": (
                "bridge_like",
                "maximum_consecutive_wide_run_seconds",
            ),
            "bridge_preferred_maximum_consecutive_wide_run_seconds": (
                "bridge_like",
                "preferred_maximum_consecutive_wide_run_seconds",
            ),
        }
        for config_field, (role_name, policy_field) in field_bindings.items():
            with self.subTest(config_field=config_field):
                threshold = getattr(roles[role_name], policy_field)
                expected = None if threshold is None else threshold.value
                self.assertEqual(expected, getattr(config, config_field))

        self.assertIsNone(config.build_minimum_closeup_fraction)
        self.assertIsNone(config.build_minimum_nonwide_fraction)
        self.assertIsNone(config.bridge_minimum_closeup_fraction)
        self.assertGreater(
            config.build_pooled_product_minimum_nonwide_fraction,
            0.0,
        )
        self.assertGreater(
            config.bridge_pooled_product_minimum_nonwide_fraction,
            0.0,
        )
        self.assertEqual(
            roles["build"].reference_count,
            config.build_duration_reference_count,
        )
        self.assertEqual(
            roles["bridge_like"].reference_count,
            config.bridge_duration_reference_count,
        )
        self.assertEqual(
            policy.highlight_product_minimum_source_switches_per_minute.value,
            config.highlight_product_minimum_source_switches_per_minute,
        )
        self.assertEqual(
            policy.highlight_preferred_source_switches_per_minute.value,
            config.highlight_preferred_source_switches_per_minute,
        )
        self.assertEqual(
            policy.highlight_cadence_reference_count,
            config.highlight_cadence_reference_count,
        )
        self.assertAlmostEqual(
            18.642084765823732,
            config.highlight_product_minimum_source_switches_per_minute,
        )
        self.assertAlmostEqual(
            23.505554096533405,
            config.highlight_preferred_source_switches_per_minute,
        )
        self.assertEqual(
            policy.minimum_close_burst_shot_count.value,
            config.minimum_close_burst_shot_count,
        )
        self.assertEqual(
            policy.minimum_close_burst_duration_seconds.value,
            config.minimum_close_burst_duration_seconds,
        )
        self.assertEqual(
            policy.minimum_close_burst_source_switch_fraction.value,
            config.minimum_close_burst_source_switch_fraction,
        )
        self.assertEqual(6, config.close_burst_event_reference_count)
        self.assertEqual(64, len(str(config.close_burst_policy_binding_sha256)))

        build = roles["build"]
        assert build.pooled_product_minimum_close_duration_fraction is not None
        sentinel = (
            build.pooled_product_minimum_close_duration_fraction.value + 0.01
        )
        changed_build = replace(
            build,
            pooled_product_minimum_close_duration_fraction=replace(
                build.pooled_product_minimum_close_duration_fraction,
                value=sentinel,
            ),
        )
        changed_policy = replace(
            policy,
            role_conditioned_duration_policies=tuple(
                changed_build if item.role == "build" else item
                for item in policy.role_conditioned_duration_policies
            ),
        )
        changed_config = _rendered_montage_config_from_policy(
            changed_policy,
            require_complete_role_duration_policy=True,
        )
        self.assertEqual(
            sentinel,
            changed_config.build_pooled_product_minimum_closeup_fraction,
        )
        constructor_default = RenderedMontageGrammarConfig()
        self.assertNotEqual(
            constructor_default.build_pooled_product_minimum_closeup_fraction,
            changed_config.build_pooled_product_minimum_closeup_fraction,
        )

        missing_bridge_policy = replace(
            policy,
            role_conditioned_duration_policies=tuple(
                item
                for item in policy.role_conditioned_duration_policies
                if item.role != "bridge_like"
            ),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "requires complete build, chorus_like, and bridge_like",
        ):
            _rendered_montage_config_from_policy(
                missing_bridge_policy,
                require_complete_role_duration_policy=True,
            )

    def test_default_ive_motion_cut_profile_is_fixed_release_v1(self) -> None:
        self.assertEqual(
            DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_PATH,
            Path(
                "/Volumes/CrossCutMedia/CrossCut-Training/release/profiles/"
                "ive-professional-motion-cut-profile-v1.json"
            ),
        )
        self.assertEqual(
            DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_FILE_SHA256,
            "005a21f79ab1e0b5de76d3299ca2f61b54eb460b88cfb2f88dd392d93e9419b6",
        )

    def test_stage_edit_motion_cut_profile_is_inactive_only_without_stageedit(
        self,
    ) -> None:
        with patch(
            "crosscut.pipeline.load_professional_motion_cut_profile"
        ) as loader:
            self.assertIsNone(
                _load_stage_edit_motion_cut_profile(
                    registered_bundle=None,
                    explicit_preflight=None,
                    montage_policy=None,
                )
            )
        loader.assert_not_called()

    def test_stage_edit_motion_cut_profile_binds_soft_runtime_and_manifest(
        self,
    ) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        montage_policy, _montage_path = _current_montage_policy_fixture(
            Path(temporary.name)
        )
        expected_profile, profile_path, profile_file_sha256 = (
            _current_motion_cut_profile_fixture(
                Path(temporary.name),
                montage_policy,
            )
        )
        path_patch = patch(
            "crosscut.pipeline.DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_PATH",
            profile_path,
        )
        sha_patch = patch(
            "crosscut.pipeline."
            "DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_FILE_SHA256",
            profile_file_sha256,
        )
        path_patch.start()
        sha_patch.start()
        self.addCleanup(path_patch.stop)
        self.addCleanup(sha_patch.stop)
        with patch(
            "crosscut.pipeline.load_professional_motion_cut_profile",
            wraps=load_professional_motion_cut_profile,
        ) as loader:
            profile = _load_stage_edit_motion_cut_profile(
                registered_bundle=object(),  # type: ignore[arg-type]
                explicit_preflight=None,
                montage_policy=montage_policy,
            )
        loader.assert_called_once_with(
            profile_path,
            expected_file_sha256=profile_file_sha256,
            verify_source_inputs=False,
        )
        self.assertIsNotNone(profile)
        assert profile is not None
        payload = _stage_edit_motion_cut_profile_manifest_payload(
            profile,
            montage_policy,
        )
        self.assertEqual(
            str(profile_path.resolve()),
            payload["path"],
        )
        self.assertEqual(
            profile_file_sha256,
            payload["file_sha256"],
        )
        self.assertEqual(
            expected_profile.profile_content_sha256,
            payload["content_sha256"],
        )
        self.assertEqual(
            montage_policy.profile_identity,
            payload["source_montage_profile_identity"],
        )
        self.assertEqual(
            montage_policy.profile_identity,
            payload["active_montage_policy_identity"],
        )
        self.assertTrue(payload["source_identity_binding_passed"])
        self.assertFalse(payload["direction_or_velocity_proven"])
        runtime = payload["runtime_binding"]
        expected_recommendation = expected_profile.recommendation
        self.assertAlmostEqual(
            expected_recommendation.both_moving_minimum_pose_motion,
            runtime["both_motion_minimum_pose_motion_p10"],
        )
        self.assertAlmostEqual(
            expected_recommendation.continuity_compatible_minimum_score,
            runtime["continuity_minimum_score_p10"],
        )
        self.assertAlmostEqual(
            expected_recommendation.pose_phase_compatible_preferred_cosine,
            runtime["same_time_pose_cosine_soft_bonus_qualifier_p50"],
        )
        self.assertTrue(runtime["soft_bonus_only"])
        self.assertFalse(runtime["hard_gate"])
        self.assertTrue(
            runtime["independent_from_primary_boundary_continuity"]
        )
        self.assertFalse(runtime["primary_boundary_score_reused"])
        self.assertEqual(
            HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
            runtime["continuity_scorer_fingerprint"],
        )
        self.assertIsNone(runtime["minimum_required_switches_per_section"])
        self.assertEqual(
            0,
            runtime["highlight_minimum_motion_closeup_switch_boundaries"],
        )
        self.assertFalse(runtime["direction_or_velocity_proven"])

        config = _stage_edit_semimarkov_config_from_montage_policy(
            montage_policy,
            motion_cut_profile=profile,
            require_motion_cut_calibration=True,
        )
        self.assertIsNotNone(config.highlight_motion_calibration)
        self.assertAlmostEqual(
            expected_recommendation.both_moving_minimum_pose_motion,
            config.highlight_motion_boundary_minimum_pose_motion,
        )
        self.assertAlmostEqual(
            expected_recommendation.continuity_compatible_minimum_score,
            config.highlight_motion_boundary_minimum_continuity_score,
        )
        self.assertAlmostEqual(
            expected_recommendation.pose_phase_compatible_preferred_cosine,
            config.highlight_motion_boundary_minimum_pose_phase_cosine,
        )
        self.assertIsNone(
            config.highlight_motion_boundary_minimum_sharpness
        )
        self.assertEqual(
            0,
            config.highlight_minimum_motion_closeup_switch_boundaries,
        )

        reporter = _MemoryReporter()
        _report_stage_edit_motion_cut_profile(reporter, payload)
        self.assertEqual(1, len(reporter.events))
        event_type, event = reporter.events[0]
        self.assertEqual("log", event_type)
        self.assertEqual(
            payload,
            event["professional_motion_cut_profile"],
        )
        self.assertIn(str(payload["path"]), str(event["message"]))
        self.assertIn(str(payload["file_sha256"]), str(event["message"]))
        self.assertIn(str(payload["content_sha256"]), str(event["message"]))

    def test_stage_edit_motion_cut_profile_fails_closed_on_file_or_source_change(
        self,
    ) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        montage_policy, _montage_path = _current_montage_policy_fixture(
            Path(temporary.name)
        )
        for error in (
            FileNotFoundError("profile missing"),
            ValueError("motion-cut profile file SHA-256 mismatch"),
        ):
            with self.subTest(error=type(error).__name__), patch(
                "crosscut.pipeline.load_professional_motion_cut_profile",
                side_effect=error,
            ):
                with self.assertRaises(type(error)):
                    _load_stage_edit_motion_cut_profile(
                        registered_bundle=object(),  # type: ignore[arg-type]
                        explicit_preflight=None,
                        montage_policy=montage_policy,
                    )

        profile, profile_path, profile_file_sha256 = (
            _current_motion_cut_profile_fixture(
                Path(temporary.name),
                montage_policy,
            )
        )
        path_patch = patch(
            "crosscut.pipeline.DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_PATH",
            profile_path,
        )
        sha_patch = patch(
            "crosscut.pipeline."
            "DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_FILE_SHA256",
            profile_file_sha256,
        )
        path_patch.start()
        sha_patch.start()
        self.addCleanup(path_patch.stop)
        self.addCleanup(sha_patch.stop)
        changed_policy = replace(
            montage_policy,
            identity=replace(
                montage_policy.identity,
                content_sha256="0" * 64,
            ),
        )
        with patch(
            "crosscut.pipeline.load_professional_motion_cut_profile",
            return_value=profile,
        ), self.assertRaisesRegex(
            RuntimeError,
            "source montage identity differs",
        ):
            _load_stage_edit_motion_cut_profile(
                registered_bundle=object(),  # type: ignore[arg-type]
                explicit_preflight=None,
                montage_policy=changed_policy,
            )

    def test_active_learned_continuity_keeps_independent_handcrafted_motion_qualifier(
        self,
    ) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        montage_policy, _montage_path = _current_montage_policy_fixture(
            Path(temporary.name)
        )
        profile, profile_path, profile_file_sha256 = (
            _current_motion_cut_profile_fixture(
                Path(temporary.name),
                montage_policy,
            )
        )
        path_patch = patch(
            "crosscut.pipeline.DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_PATH",
            profile_path,
        )
        sha_patch = patch(
            "crosscut.pipeline."
            "DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_FILE_SHA256",
            profile_file_sha256,
        )
        path_patch.start()
        sha_patch.start()
        self.addCleanup(path_patch.stop)
        self.addCleanup(sha_patch.stop)
        config = _stage_edit_semimarkov_config_from_montage_policy(
            montage_policy,
            motion_cut_profile=profile,
            require_motion_cut_calibration=True,
            continuity_authority=(
                PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
            ),
            continuity_scorer_contract=(
                STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT
            ),
            continuity_scorer_fingerprint="a" * 64,
        )

        self.assertEqual(
            PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
            config.continuity_authority,
        )
        self.assertEqual(
            STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
            config.continuity_scorer_contract,
        )
        self.assertEqual("a" * 64, config.continuity_scorer_fingerprint)
        self.assertIsNotNone(config.highlight_motion_calibration)
        assert config.highlight_motion_calibration is not None
        self.assertEqual(
            HANDCRAFTED_CONTINUITY_AUTHORITY,
            config.highlight_motion_calibration.continuity_authority,
        )
        self.assertEqual(
            HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
            config.highlight_motion_calibration.continuity_scorer_contract,
        )

    def test_stage_edit_request_is_forwarded_to_top32_proxy_rerank_only_when_explicit(
        self,
    ) -> None:
        records = tuple(
            SimpleNamespace(
                candidate_index=index,
                review=SimpleNamespace(passed=index != 8),
            )
            for index in range(10)
        )
        candidate_set = SimpleNamespace(candidates=records)
        checkpoint = Path("/tmp/stage-edit-production-checkpoint")
        report = checkpoint / "training-report.json"
        request = SimpleNamespace(
            stage_edit_checkpoint_directory=checkpoint,
            stage_edit_production_training_report=report,
        )
        rhythm = object()
        section_plan = object()
        rerank_result = SimpleNamespace(
            ordered_records=lambda supplied: (
                records[::-1] if supplied is candidate_set else ()
            )
        )
        reporter = _MemoryReporter()
        with patch(
            "crosscut.pipeline.rerank_visual_timeline_candidates_with_stage_edit",
            return_value=rerank_result,
        ) as rerank, patch(
            "crosscut.pipeline.materialize_visual_timeline_candidate_record",
            side_effect=lambda _candidate_set, record: record.candidate_index,
        ) as materialize_record, patch(
            "crosscut.pipeline.materialize_visual_timeline_candidates",
        ) as legacy_materialize:
            materialized, result = _materialize_ranked_visual_proxy_candidates(
                candidate_set,  # type: ignore[arg-type]
                request=request,  # type: ignore[arg-type]
                rhythm_analysis=rhythm,
                section_plan=section_plan,  # type: ignore[arg-type]
                reporter=reporter,  # type: ignore[arg-type]
            )

        self.assertIs(result, rerank_result)
        self.assertEqual(materialized, (9, 7, 6, 5, 4, 3, 2, 1))
        rerank.assert_called_once_with(
            candidate_set,
            checkpoint_directory=checkpoint,
            production_training_report=report,
            rhythm=rhythm,
            section_plan=section_plan,
        )
        self.assertEqual(materialize_record.call_count, 8)
        legacy_materialize.assert_not_called()
        self.assertTrue(any(kind == "status" for kind, _ in reporter.events))

        legacy_request = SimpleNamespace(
            stage_edit_checkpoint_directory=None,
            stage_edit_production_training_report=None,
        )
        with patch(
            "crosscut.pipeline.rerank_visual_timeline_candidates_with_stage_edit"
        ) as rerank, patch(
            "crosscut.pipeline.materialize_visual_timeline_candidates",
            return_value=("legacy-order",),
        ) as legacy_materialize:
            materialized, result = _materialize_ranked_visual_proxy_candidates(
                candidate_set,  # type: ignore[arg-type]
                request=legacy_request,  # type: ignore[arg-type]
                rhythm_analysis=rhythm,
                section_plan=None,
                reporter=reporter,  # type: ignore[arg-type]
            )

        self.assertEqual(materialized, ("legacy-order",))
        self.assertIsNone(result)
        rerank.assert_not_called()
        legacy_materialize.assert_called_once_with(
            candidate_set,
            maximum_candidates=8,
            passed_only=True,
        )
        with self.assertRaisesRegex(RuntimeError, "requires decoded proxy review"):
            analyze(
                request,  # type: ignore[arg-type]
                _MemoryReporter(),  # type: ignore[arg-type]
                review_rendered_proxies=False,
            )

    def test_complete_group_candidate_derives_every_production_layer_and_global_indices(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = tuple(root / f"source-{index}.mp4" for index in range(6))
            visual = VisualTimelinePlan(
                (
                    _visual_choice(0, 0, 10.0, 11.0),
                    _visual_choice(1, 1, 11.0, 12.0),
                ),
                9.0,
                selected_candidate_index=7,
            )
            materialized = SimpleNamespace(
                visual_plan=visual,
                candidate_record=SimpleNamespace(optimizer_score=9.0),
                candidate_index=7,
                review_rank=2,
                cadence_profile="synthetic-cadence",
            )
            repaired = VisualBoundaryRepairPlan(visual, (visual,), (), 0, ())
            timeline_review = SimpleNamespace(
                weighted_score=81.5,
                passed=True,
                hard_failures=(),
                warnings=(),
            )
            face_match = SimpleNamespace(
                pan_crop_for=lambda _segment_index: None,
            )
            transitions = GroupTransitionPlan(
                (RenderTransition(tier="C", reason="synthetic"),),
                analysis_basis="synthetic_complete_candidate",
            )
            affine = SimpleNamespace()
            boundary_subject = SimpleNamespace()
            editorial = SimpleNamespace(
                track_for=lambda _segment_index: None,
            )
            request = SimpleNamespace(
                sources=sources,
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                source_mode=SourceMode.AUTOMATIC,
            )
            look_adjustments = (
                SourceLookAdjustment(brightness=0.01),
                SourceLookAdjustment(saturation=0.97),
            )
            source_look = SimpleNamespace(
                adjustment_for_source=lambda index: look_adjustments[index],
            )
            boundary_look = SimpleNamespace(
                apply=lambda plan: plan,
                to_dict=lambda: {"enabled": True},
            )

            with patch(
                "crosscut.pipeline.repair_visual_timeline_boundaries",
                return_value=repaired,
            ) as repair, patch(
                "crosscut.pipeline.review_timeline",
                return_value=timeline_review,
            ) as review, patch(
                "crosscut.pipeline.build_face_match_plan",
                return_value=face_match,
            ) as face, patch(
                "crosscut.pipeline.build_group_transition_plan",
                return_value=transitions,
            ) as transition, patch(
                "crosscut.pipeline.build_boundary_affine_plan",
                return_value=affine,
            ) as boundary_affine, patch(
                "crosscut.pipeline.build_boundary_subject_alignment_plan",
                return_value=boundary_subject,
            ) as subject_alignment, patch(
                "crosscut.pipeline.build_editorial_motion_plan",
                return_value=editorial,
            ) as motion:
                with patch(
                    "crosscut.pipeline.analyze_boundary_look_match",
                    return_value=boundary_look,
                ) as boundary_color, patch(
                    "crosscut.pipeline.verify_boundary_look_match_render_domain",
                    side_effect=lambda _plan, result: result,
                ) as boundary_color_verify:
                    bundle = _derive_complete_group_candidate(
                        materialized,
                        request=request,  # type: ignore[arg-type]
                        eligible_indices=(2, 5),
                        audio_offsets=(0.0, 0.0),
                        scene_cut_times=((), ()),
                        source_timewarps=(None, None),
                        source_aspect_ratios=(16.0 / 9.0, 16.0 / 9.0),
                        source_durations_seconds=(30.0, 30.0),
                        beat_period_seconds=0.5,
                        phrase_times=(10.0, 12.0),
                        section_plan=None,
                        common_master_start=10.0,
                        width=1920,
                        height=1080,
                        output=root / "candidate.mp4",
                        source_look_match=source_look,  # type: ignore[arg-type]
                    )

            repair.assert_called_once()
            review.assert_called_once()
            face.assert_called_once()
            transition.assert_called_once()
            boundary_affine.assert_called_once()
            subject_alignment.assert_called_once()
            motion.assert_called_once()
            boundary_color.assert_called_once()
            boundary_color_verify.assert_called_once()
            self.assertEqual(bundle.candidate_index, 7)
            self.assertEqual(bundle.pre_render_score, 81.5)
            self.assertIs(bundle.boundary_repairs, repaired)
            self.assertIs(bundle.timeline_review, timeline_review)
            self.assertIs(bundle.face_match_plan, face_match)
            self.assertIs(bundle.group_transition_plan, transitions)
            self.assertIs(bundle.boundary_affine_plan, affine)
            self.assertIs(bundle.editorial_motion_plan, editorial)
            self.assertIs(bundle.boundary_look_match, boundary_look)
            self.assertEqual(
                tuple(item.source_index for item in bundle.render_plan.segments),
                (2, 5),
            )
            self.assertEqual(bundle.render_plan.sources, sources)
            self.assertEqual(bundle.render_plan.master_audio_start, 10.0)
            self.assertEqual(
                tuple(
                    item.look_adjustment
                    for item in bundle.render_plan.segments
                ),
                look_adjustments,
            )

    def test_complete_group_candidate_batch_forwards_eight_atomic_bundles(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            bundles = tuple(
                SimpleNamespace(
                    candidate_index=index,
                    pre_render_score=float(100 - index),
                    render_plan=RenderPlan(
                        sources=(source,),
                        segments=(
                            RenderSegment(
                                source_index=0,
                                source_start=0.0,
                                duration=1.0,
                                master_start=0.0,
                            ),
                        ),
                        reference_source=0,
                        aspect_ratio=AspectRatio.LANDSCAPE,
                        output=root / f"candidate-{index}.mp4",
                    ),
                    timeline_review=SimpleNamespace(
                        source_switches_per_minute=13.5 + index * 0.1
                    ),
                )
                for index in range(8)
            )
            sentinel = object()
            learned_session = object()

            with patch(
                "crosscut.pipeline.review_rendered_candidates",
                return_value=sentinel,
            ) as review:
                result = _review_complete_group_candidates(
                    bundles,  # type: ignore[arg-type]
                    scratch_directory=root,
                    learned_reviewer_session=learned_session,  # type: ignore[arg-type]
                )

            self.assertIs(result, sentinel)
            review.assert_called_once()
            args = review.call_args.args
            kwargs = review.call_args.kwargs
            forwarded = args[0]
            self.assertEqual(len(forwarded), 8)
            self.assertEqual(
                tuple(item.candidate_index for item in forwarded),
                tuple(range(8)),
            )
            self.assertEqual(
                tuple(item.payload for item in forwarded),
                bundles,
            )
            self.assertEqual(kwargs["scratch_directory"], root)
            self.assertEqual(kwargs["maximum_review_count"], 8)
            self.assertEqual(kwargs["minimum_required_candidate_count"], 8)
            self.assertIs(kwargs["learned_reviewer_session"], learned_session)

    def test_stage_edit_candidate_batch_hard_gates_every_decoded_proxy_with_montage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            source.touch()
            bundles = tuple(
                SimpleNamespace(
                    candidate_index=index,
                    pre_render_score=float(100 - index),
                    render_plan=RenderPlan(
                        sources=(source,),
                        segments=(RenderSegment(0, 0.0, 6.0, 0.0),),
                        reference_source=0,
                        aspect_ratio=AspectRatio.LANDSCAPE,
                        output=root / f"candidate-{index}.mp4",
                    ),
                    timeline_review=SimpleNamespace(
                        source_switches_per_minute=15.0
                    ),
                    visual_timeline=SimpleNamespace(
                        to_dict=lambda: {"choices": []}
                    ),
                    editorial_motion_plan=None,
                    candidate_origin="explicit_stage_edit_generation",
                    stage_edit_dynamic_crop_version=2,
                    stage_edit_dynamic_crop_v3_decodes=None,
                    dynamic_crop_motion_timeline_v3=None,
                    candidate_lineage=(
                        {
                            "exact_music_structure_shot_clock": {
                                "contract": (
                                    "serialized_inference_token_section_and_"
                                    "phrase_progress_per_selected_shot_v1"
                                ),
                                "source_index_domain": (
                                    "eligible_source_index"
                                ),
                                "shots": [],
                            }
                        },
                    ),
                    stage_edit_exact_shot_clock={
                        "contract": (
                            "serialized_inference_token_section_and_phrase_"
                            "progress_per_selected_shot_v1"
                        ),
                        "source_index_domain": "eligible_source_index",
                        "shots": [],
                    },
                )
                for index in range(8)
            )
            section_plan = SimpleNamespace(
                to_dict=lambda: {"sections": []}
            )
            diversity = SimpleNamespace(
                to_dict=lambda: {
                    "source_count": 1,
                    "family_ids": [0],
                    "production_ready": True,
                }
            )
            identity = {
                "analysis_basis": "ive_human_reference_visual_montage_grammar_v1",
                "schema_version": 1,
                "path": str(root / "profile.json"),
                "file_sha256": "a" * 64,
                "content_sha256": "b" * 64,
            }
            reviewed_payloads: list[dict[str, object]] = []
            sentinel = object()
            boundary_evidence = SimpleNamespace(
                to_dict=lambda: {
                    "schema_version": 1,
                    "evidence_sha256": "c" * 64,
                }
            )
            boundary_assessment_passed = SimpleNamespace(
                passed=True,
                to_dict=lambda: {
                    "schema_version": 1,
                    "passed": True,
                    "failure_reasons": [],
                },
            )
            boundary_assessment_failed = SimpleNamespace(
                passed=False,
                to_dict=lambda: {
                    "schema_version": 1,
                    "passed": False,
                    "failure_reasons": [
                        "highlight_no_switch_gap_above_maximum"
                    ],
                },
            )

            def review_batch(rendered, **kwargs):
                for item in rendered:
                    _proxy, review = kwargs["render_review"](
                        item.render_plan,
                        root / f"proxy-{item.candidate_index}.mp4",
                    )
                    reviewed_payloads.append(review.payload)
                return sentinel

            with patch(
                "crosscut.pipeline.review_rendered_candidates",
                side_effect=review_batch,
            ), patch(
                "crosscut.pipeline.render_and_review_proxy",
                side_effect=lambda plan, _output, **_kwargs: (
                    plan,
                    RenderedProxyReview(
                        {"passed": True, "weighted_score": 91.0}
                    ),
                ),
            ), patch(
                "crosscut.pipeline.review_rendered_montage_grammar",
                side_effect=lambda *_args, **_kwargs: {
                    "schema_version": 1,
                    "analysis_basis": "test_montage",
                    "passed": True,
                    "failure_codes": [],
                    "configuration": {},
                    "highlight_aggregate": {
                        "soft_targets": {
                            "aggregate_alignment_score": 0.5,
                        }
                    },
                },
            ) as montage_review, patch(
                "crosscut.pipeline.adapt_group_boundary_grammar_evidence",
                return_value=boundary_evidence,
            ) as boundary_adapter, patch(
                "crosscut.pipeline.assess_professional_boundary_grammar",
                side_effect=(
                    boundary_assessment_failed,
                    *([boundary_assessment_passed] * 7),
                ),
            ) as boundary_gate:
                result = _review_complete_group_candidates(
                    bundles,  # type: ignore[arg-type]
                    scratch_directory=root,
                    stage_edit_montage_section_plan=section_plan,  # type: ignore[arg-type]
                    stage_edit_source_derivative_diversity=diversity,  # type: ignore[arg-type]
                    stage_edit_source_index_mapping={"bound": True},
                    stage_edit_source_scene_cut_times=((),),
                    stage_edit_montage_config=RenderedMontageGrammarConfig(),
                    stage_edit_montage_reference_profile=identity,
                )

            self.assertIs(result, sentinel)
            self.assertEqual(montage_review.call_count, 8)
            self.assertEqual(boundary_adapter.call_count, 8)
            self.assertEqual(boundary_gate.call_count, 8)
            self.assertEqual(len(reviewed_payloads), 8)
            self.assertFalse(reviewed_payloads[0]["passed"])
            self.assertTrue(
                all(item["passed"] for item in reviewed_payloads[1:])
            )
            self.assertTrue(
                all(
                    item["rendered_montage_grammar_review"][
                        "reference_profile"
                    ]
                    == identity
                    for item in reviewed_payloads
                )
            )
            self.assertFalse(
                reviewed_payloads[0]["rendered_montage_grammar_review"][
                    "professional_boundary_grammar_assessment"
                ]["passed"]
            )
            self.assertTrue(
                all(
                    item["rendered_montage_grammar_review"][
                        "professional_boundary_grammar_assessment"
                    ]["passed"]
                    for item in reviewed_payloads[1:]
                )
            )

    def test_hard_pass_proxy_rank_uses_bounded_symmetric_human_target(self) -> None:
        payload = {
            "passed": True,
            "weighted_score": 80.0,
            "weighted_score_before_perceptual_source_admission": 80.0,
            "perceptual_source_admission_penalty_points": 0.0,
        }
        montage = {
            "passed": True,
            "highlight_aggregate": {
                "soft_targets": {
                    "aggregate_alignment_score": 1.0,
                    "closeup_p50": {"alignment_score": 1.0},
                    "nonwide_p50": {"alignment_score": 1.0},
                    "maximum_wide_run_p75": {"alignment_score": 1.0},
                }
            },
        }

        _apply_bounded_human_target_proxy_rank(
            payload,
            montage,
            technical_passed=True,
        )

        self.assertEqual(
            payload["technical_proxy_weighted_score_before_human_target"],
            80.0,
        )
        self.assertEqual(payload["weighted_score"], 80.0)
        self.assertEqual(payload["ranking_weighted_score"], 81.0)
        self.assertEqual(
            payload["weighted_score"],
            payload["weighted_score_before_perceptual_source_admission"]
            - payload["perceptual_source_admission_penalty_points"],
        )
        evidence = payload["human_target_bounded_ranking"]
        self.assertEqual(evidence["maximum_absolute_adjustment_points"], 1.0)
        self.assertFalse(evidence["over_close_past_target_is_rewarded"])

        over_close = {"passed": True, "weighted_score": 80.0}
        montage["highlight_aggregate"]["soft_targets"][
            "aggregate_alignment_score"
        ] = 0.2
        _apply_bounded_human_target_proxy_rank(
            over_close,
            montage,
            technical_passed=True,
        )
        self.assertEqual(over_close["weighted_score"], 80.0)
        self.assertAlmostEqual(over_close["ranking_weighted_score"], 79.4)
        self.assertGreaterEqual(over_close["ranking_weighted_score"], 79.0)

        failed = {"passed": False, "weighted_score": 80.0}
        _apply_bounded_human_target_proxy_rank(
            failed,
            montage,
            technical_passed=False,
        )
        self.assertEqual(failed["weighted_score"], 80.0)
        self.assertEqual(failed["ranking_weighted_score"], 80.0)
        self.assertFalse(failed["human_target_bounded_ranking"]["eligible"])

        with self.assertRaisesRegex(
            RuntimeError,
            "lacks bounded human-target alignment evidence",
        ):
            _apply_bounded_human_target_proxy_rank(
                {"passed": True, "weighted_score": 80.0},
                {"passed": True},
                technical_passed=True,
            )

    def test_qualified_close_burst_adds_only_evidence_weighted_soft_rank(self) -> None:
        base_montage = {
            "passed": True,
            "highlight_aggregate": {
                "soft_targets": {
                    "aggregate_alignment_score": 0.5,
                    "closeup_p50": {"alignment_score": 0.5},
                    "nonwide_p50": {"alignment_score": 0.5},
                    "maximum_wide_run_p75": {"alignment_score": 0.5},
                }
            },
        }
        absent = {"passed": True, "weighted_score": 80.0}
        _apply_bounded_human_target_proxy_rank(
            absent,
            base_montage,
            technical_passed=True,
        )
        self.assertEqual(80.0, absent["weighted_score"])
        self.assertEqual(80.0, absent["ranking_weighted_score"])

        qualified_montage = deepcopy(base_montage)
        qualified_montage["highlight_aggregate"]["soft_targets"][
            "close_only_multi_source_burst_event_p10"
        ] = {
            "policy_binding_active": True,
            "policy_binding": {"binding_sha256": "a" * 64},
            "alignment_score": 1.0,
            "event_occurrence_support_weight": 6.0 / 20.0,
            "evidence_weighted_alignment_score": 6.0 / 20.0,
            "observed_event_count": 6,
            "human_reference_count": 20,
            "qualified_event_count": 1,
            "absence_is_hard_failure": False,
        }
        qualified = {"passed": True, "weighted_score": 80.0}
        _apply_bounded_human_target_proxy_rank(
            qualified,
            qualified_montage,
            technical_passed=True,
        )
        self.assertEqual(80.0, qualified["weighted_score"])
        self.assertAlmostEqual(80.3, qualified["ranking_weighted_score"])
        ranking = qualified["human_target_bounded_ranking"]
        self.assertAlmostEqual(
            0.3,
            ranking["close_burst_zero_or_additive_bonus_points"],
        )
        self.assertFalse(ranking["close_burst_absence_lowers_existing_alignment"])

        tampered_montage = deepcopy(qualified_montage)
        tampered_montage["highlight_aggregate"]["soft_targets"][
            "close_only_multi_source_burst_event_p10"
        ]["event_occurrence_support_weight"] = 0.9
        with self.assertRaisesRegex(RuntimeError, "support weight is malformed"):
            _apply_bounded_human_target_proxy_rank(
                {"passed": True, "weighted_score": 80.0},
                tampered_montage,
                technical_passed=True,
            )

    def test_final_repaired_proxy_rank_replaces_stale_pre_repair_evidence(
        self,
    ) -> None:
        repaired_payload = {
            "passed": True,
            "weighted_score": 80.0,
            "technical_proxy_passed_before_montage_grammar": True,
            "ranking_weighted_score": 81.0,
            "human_target_bounded_ranking": {
                "aggregate_alignment_score": 1.0,
                "applied_adjustment_points": 1.0,
                "stale_pre_repair_evidence": True,
            },
        }
        final_montage = {
            "passed": True,
            "highlight_aggregate": {
                "soft_targets": {
                    "aggregate_alignment_score": 0.0,
                    "final_repaired_timeline": True,
                }
            },
        }

        _apply_bounded_human_target_proxy_rank(
            repaired_payload,
            final_montage,
            technical_passed=bool(
                repaired_payload[
                    "technical_proxy_passed_before_montage_grammar"
                ]
            ),
        )

        self.assertEqual(repaired_payload["weighted_score"], 80.0)
        self.assertEqual(repaired_payload["ranking_weighted_score"], 79.0)
        final_evidence = repaired_payload["human_target_bounded_ranking"]
        self.assertNotIn("stale_pre_repair_evidence", final_evidence)
        self.assertEqual(final_evidence["aggregate_alignment_score"], 0.0)
        self.assertEqual(final_evidence["applied_adjustment_points"], -1.0)
        self.assertTrue(
            final_evidence["soft_target_evidence"][
                "final_repaired_timeline"
            ]
        )

        hard_failed_payload = {
            "passed": False,
            "weighted_score": 80.0,
            "technical_proxy_passed_before_montage_grammar": False,
            "ranking_weighted_score": 81.0,
            "human_target_bounded_ranking": {
                "aggregate_alignment_score": 1.0,
                "applied_adjustment_points": 1.0,
                "stale_pre_repair_evidence": True,
            },
        }
        _apply_bounded_human_target_proxy_rank(
            hard_failed_payload,
            {
                "passed": True,
                "highlight_aggregate": {
                    "soft_targets": {
                        "aggregate_alignment_score": 1.0,
                    }
                },
            },
            technical_passed=False,
        )

        self.assertFalse(hard_failed_payload["passed"])
        self.assertEqual(hard_failed_payload["weighted_score"], 80.0)
        self.assertEqual(hard_failed_payload["ranking_weighted_score"], 80.0)
        hard_fail_evidence = hard_failed_payload[
            "human_target_bounded_ranking"
        ]
        self.assertFalse(hard_fail_evidence["eligible"])
        self.assertEqual(hard_fail_evidence["applied_adjustment_points"], 0.0)
        self.assertNotIn(
            "stale_pre_repair_evidence",
            hard_fail_evidence,
        )

    def test_windowed_repair_keeps_only_final_exact_shot_clock_for_montage(
        self,
    ) -> None:
        base_clock = {
            "contract": (
                "serialized_inference_token_section_and_phrase_progress_"
                "per_selected_shot_v1"
            ),
            "source_index_domain": "eligible_source_index",
            "shots": [{"shot_index": 0, "source_index": 0}],
        }
        final_clock = {
            **base_clock,
            "shots": [{"shot_index": 0, "source_index": 1}],
        }
        stripped, hashes = _strip_superseded_exact_stage_edit_shot_clocks(
            ({"exact_music_structure_shot_clock": base_clock},)
        )
        self.assertEqual(1, len(hashes))
        lineage = stripped + (
            {"exact_music_structure_shot_clock": final_clock},
        )
        plan = RenderPlan(
            sources=(Path("/tmp/a.mp4"), Path("/tmp/b.mp4")),
            segments=(RenderSegment(1, 0.0, 1.0, 0.0),),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("/tmp/out.mp4"),
        )
        bundle = SimpleNamespace(
            render_plan=plan,
            candidate_lineage=lineage,
            visual_timeline=SimpleNamespace(to_dict=lambda: {"choices": []}),
            editorial_motion_plan=None,
            candidate_origin="ordinary_visual_baseline",
            stage_edit_dynamic_crop_version=None,
            stage_edit_dynamic_crop_v3_decodes=None,
            dynamic_crop_motion_timeline_v3=None,
            stage_edit_exact_shot_clock=final_clock,
        )

        payload = _stage_edit_montage_analysis_payload(
            bundle=bundle,
            proxy_review_payload={"passed": True},
            section_plan=SimpleNamespace(to_dict=lambda: {"sections": []}),
            source_derivative_diversity=SimpleNamespace(
                to_dict=lambda: {"source_count": 2}
            ),
            source_index_mapping={"bound": True},
        )

        self.assertEqual(final_clock, payload["stage_edit_exact_shot_clock"])
        self.assertNotEqual(base_clock, payload["stage_edit_exact_shot_clock"])

    def test_learned_reviewer_session_is_opt_in_and_configured_failure_is_fatal(
        self,
    ) -> None:
        absent = SimpleNamespace(
            stage_edit_learned_reviewer_checkpoint_directory=None,
            stage_edit_learned_reviewer_production_training_report=None,
            song="BANG BANG",
        )
        with patch(
            "crosscut.pipeline.LearnedReviewerProductionSession.load"
        ) as load:
            self.assertIsNone(
                _load_learned_reviewer_session(absent)  # type: ignore[arg-type]
            )
        load.assert_not_called()

        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "reviewer"
            checkpoint.mkdir()
            report = checkpoint / "training-report.json"
            report.write_text("{}", encoding="utf-8")
            configured = SimpleNamespace(
                stage_edit_learned_reviewer_checkpoint_directory=checkpoint,
                stage_edit_learned_reviewer_production_training_report=report,
                song="BANG BANG",
            )
            with patch(
                "crosscut.pipeline.LearnedReviewerProductionSession.load",
                side_effect=RuntimeError("production evidence rejected"),
            ) as strict_load, self.assertRaisesRegex(
                RuntimeError,
                "production evidence rejected",
            ):
                _load_learned_reviewer_session(  # type: ignore[arg-type]
                    configured
                )
        strict_load.assert_called_once()
        config = strict_load.call_args.args[0]
        self.assertEqual(config.checkpoint_directory, checkpoint.resolve())
        self.assertEqual(config.production_training_report, report.resolve())
        self.assertEqual(config.song_id, "BANG BANG")

    def test_complete_group_candidate_batch_reuses_one_loaded_reference_profile(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            bundles = tuple(
                SimpleNamespace(
                    candidate_index=index,
                    pre_render_score=float(100 - index),
                    render_plan=RenderPlan(
                        sources=(source,),
                        segments=(RenderSegment(0, 0.0, 1.0, 0.0),),
                        reference_source=0,
                        aspect_ratio=AspectRatio.LANDSCAPE,
                        output=root / f"candidate-{index}.mp4",
                    ),
                    timeline_review=SimpleNamespace(
                        source_switches_per_minute=13.5 + index * 0.1
                    ),
                )
                for index in range(8)
            )
            loaded_profile = object()
            sentinel = object()

            def review_batch(rendered, **kwargs):
                self.assertIs(
                    kwargs["professional_reference_profile"],
                    loaded_profile,
                )
                self.assertEqual(
                    tuple(
                        item.rank_metadata.source_switches_per_minute
                        for item in rendered
                    ),
                    tuple(13.5 + index * 0.1 for index in range(8)),
                )
                for item in rendered:
                    kwargs["render_review"](
                        item.render_plan,
                        root / f"proxy-{item.candidate_index}.mp4",
                    )
                return sentinel

            with patch(
                "crosscut.pipeline.review_rendered_candidates",
                side_effect=review_batch,
            ), patch(
                "crosscut.pipeline.render_and_review_proxy",
                return_value=(object(), object()),
            ) as render_review:
                result = _review_complete_group_candidates(
                    bundles,  # type: ignore[arg-type]
                    scratch_directory=root,
                    professional_reference_profile=loaded_profile,  # type: ignore[arg-type]
                )

            self.assertIs(result, sentinel)
            self.assertEqual(render_review.call_count, 8)
            self.assertTrue(
                all(
                    call.kwargs["professional_reference_profile"]
                    is loaded_profile
                    for call in render_review.call_args_list
                )
            )

    def test_professional_reference_profile_is_strict_loaded_once_and_song_scoped(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            profile_path = Path(temp) / "profile.json"
            profile_path.write_text("{}", encoding="utf-8")
            request = SimpleNamespace(
                professional_reference_profile=profile_path,
                song="BANG-BANG",
            )
            loaded_profile = SimpleNamespace(song="BANG BANG")
            with patch(
                "crosscut.pipeline.ProfessionalReferenceProfile.from_file",
                return_value=loaded_profile,
            ) as load:
                result = _load_professional_reference_profile(
                    request  # type: ignore[arg-type]
                )
            self.assertIs(result, loaded_profile)
            load.assert_called_once_with(profile_path)

            loaded_profile.song = "I AM"
            with patch(
                "crosscut.pipeline.ProfessionalReferenceProfile.from_file",
                return_value=loaded_profile,
            ), self.assertRaisesRegex(ValueError, "does not match"):
                _load_professional_reference_profile(
                    request  # type: ignore[arg-type]
                )

    def test_professional_cadence_profile_is_loaded_once_and_hash_bound(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cadence_path = root / "cadence.json"
            cadence_path.write_text("{}", encoding="utf-8")
            master = root / "master.mp4"
            media = root / "human.mp4"
            audit = root / "human-audit.json"
            for path in (master, media, audit):
                path.touch()
            request = SimpleNamespace(
                professional_reference_cadence_profile=cadence_path,
                song="BANG-BANG",
                sources=(master,),
                reference_source=0,
            )
            quality = SimpleNamespace(
                artist="IVE",
                song="BANG BANG",
                references=(
                    SimpleNamespace(
                        video_id="human-short-id",
                        path=media,
                        sha256="a" * 64,
                        audit_path=audit,
                        audit_sha256="b" * 64,
                    ),
                ),
            )
            cadence = SimpleNamespace(
                artist="IVE",
                song="BANG BANG",
                master_media_path=master.resolve(),
                reference_provenance=lambda: (
                    {
                        "video_id": "human-full-id",
                        "media_path": str(media.resolve()),
                        "media_sha256": "a" * 64,
                        "audit_path": str(audit.resolve()),
                        "audit_sha256": "b" * 64,
                    },
                ),
            )
            with patch(
                "crosscut.pipeline.ProfessionalReferenceCadenceProfile.from_file",
                return_value=cadence,
            ) as load:
                result = _load_professional_reference_cadence_profile(
                    request,  # type: ignore[arg-type]
                    quality,  # type: ignore[arg-type]
                )
            self.assertIs(result, cadence)
            load.assert_called_once_with(cadence_path)

            cadence.reference_provenance = lambda: (
                {
                    "video_id": "human-full-id",
                    "media_path": str(media.resolve()),
                    "media_sha256": "c" * 64,
                    "audit_path": str(audit.resolve()),
                    "audit_sha256": "b" * 64,
                },
            )
            with patch(
                "crosscut.pipeline.ProfessionalReferenceCadenceProfile.from_file",
                return_value=cadence,
            ), self.assertRaisesRegex(ValueError, "provenance/hashes"):
                _load_professional_reference_cadence_profile(
                    request,  # type: ignore[arg-type]
                    quality,  # type: ignore[arg-type]
                )

    def test_failed_proxy_gets_one_targeted_eight_frame_reedit_and_only_improvement_wins(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = (root / "source-a.mp4", root / "source-b.mp4")
            original_visual = VisualTimelinePlan(
                (
                    _visual_choice(0, 0, 0.0, 1.0),
                    _visual_choice(1, 1, 1.0, 2.0),
                ),
                8.0,
            )
            repaired_visual = VisualTimelinePlan(
                (
                    _visual_choice(0, 0, 0.0, 1.1),
                    _visual_choice(1, 1, 1.1, 2.0),
                ),
                8.0,
            )
            transitions = GroupTransitionPlan(
                (RenderTransition(tier="C", reason="synthetic"),)
            )
            original_plan = RenderPlan(
                sources=sources,
                segments=(
                    RenderSegment(0, 0.0, 1.0, 0.0),
                    RenderSegment(1, 1.0, 1.0, 1.0),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=root / "original.mp4",
                transitions=transitions.transitions,
            )
            partial_plan = RenderPlan(
                sources=sources,
                segments=(
                    RenderSegment(0, 0.0, 1.1, 0.0),
                    RenderSegment(1, 1.1, 0.9, 1.1),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=root / "partial.mp4",
                transitions=transitions.transitions,
            )
            original_bundle = _GroupRenderCandidateBundle(
                candidate_index=3,
                review_rank=0,
                cadence_profile="synthetic",
                pre_render_score=90.0,
                visual_timeline=original_visual,
                boundary_repairs=SimpleNamespace(),
                timeline_review=None,
                face_match_plan=None,
                group_transition_plan=transitions,
                boundary_affine_plan=None,
                editorial_motion_plan=None,
                render_plan=original_plan,
                splice_graph_authorization={"proxy_eligible": True},
            )
            partial_bundle = _GroupRenderCandidateBundle(
                candidate_index=3,
                review_rank=0,
                cadence_profile="synthetic",
                pre_render_score=89.0,
                visual_timeline=repaired_visual,
                boundary_repairs=None,
                timeline_review=SimpleNamespace(passed=True),
                face_match_plan=None,
                group_transition_plan=transitions,
                boundary_affine_plan=None,
                editorial_motion_plan=None,
                render_plan=partial_plan,
            )
            polluted_visual = VisualTimelinePlan(
                (
                    _visual_choice(0, 0, 0.0, 1.1),
                    _visual_choice(1, 0, 1.1, 2.0),
                ),
                999.0,
            )
            polluted_plan = RenderPlan(
                sources=sources,
                segments=(
                    RenderSegment(0, 0.0, 1.1, 0.0),
                    RenderSegment(0, 1.1, 0.9, 1.1),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=root / "polluted.mp4",
                transitions=transitions.transitions,
            )
            polluted_bundle = _GroupRenderCandidateBundle(
                candidate_index=3,
                review_rank=0,
                cadence_profile="synthetic",
                pre_render_score=999.0,
                visual_timeline=polluted_visual,
                boundary_repairs=None,
                timeline_review=SimpleNamespace(passed=True),
                face_match_plan=None,
                group_transition_plan=transitions,
                boundary_affine_plan=None,
                editorial_motion_plan=None,
                render_plan=polluted_plan,
            )
            targeted = SimpleNamespace(
                timeline=repaired_visual,
                repairs=(
                    SimpleNamespace(
                        boundary_index=0,
                        result=SimpleNamespace(changed=True),
                    ),
                ),
                changed_count=1,
                to_dict=lambda: {
                    "requested_boundary_indices": [0],
                    "changed_count": 1,
                },
            )
            original_review = {
                "passed": False,
                "weighted_score": 70.0,
                "boundary_issue_indices": [0],
                "boundary_score": {"issue_fraction": 0.5, "p10": 50.0},
            }
            after_review = {
                "passed": False,
                "weighted_score": 80.0,
                "boundary_issue_indices": [],
                "boundary_score": {"issue_fraction": 0.2, "p10": 60.0},
            }
            request = SimpleNamespace(
                sources=sources,
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
            )
            materialized = SimpleNamespace(candidate_index=3)

            with patch(
                "crosscut.pipeline._budgeted_repair_visual_timeline_boundaries",
                return_value=targeted,
            ) as repair, patch(
                "crosscut.pipeline._record_boundary_repair_assessments",
            ), patch(
                "crosscut.pipeline._authorize_partial_reedit_splice_lineage",
                return_value=(
                    targeted,
                    {
                        "proxy_eligible": True,
                        "geometry_proof_boundary_indices": [],
                    },
                ),
            ), patch(
                "crosscut.pipeline._derive_complete_group_candidate",
                return_value=partial_bundle,
            ) as derive, patch(
                "crosscut.pipeline.review_rendered_candidates",
                return_value=SimpleNamespace(
                    selected=SimpleNamespace(review_payload=after_review)
                ),
            ) as proxy_review:
                result = _attempt_proxy_triggered_partial_reedit(
                    original_bundle,
                    materialized,  # type: ignore[arg-type]
                    original_review,
                    request=request,  # type: ignore[arg-type]
                    eligible_indices=(0, 1),
                    audio_offsets=(0.0, 0.0),
                    scene_cut_times=((), ()),
                    source_timewarps=(None, None),
                    source_aspect_ratios=(16.0 / 9.0, 16.0 / 9.0),
                    source_durations_seconds=(10.0, 10.0),
                    beat_period_seconds=0.5,
                    phrase_times=(),
                    section_plan=None,
                    common_master_start=0.0,
                    width=1920,
                    height=1080,
                    scratch_directory=root,
                    boundary_descriptor_cache=SimpleNamespace(),
                    splice_feasibility_graph=SimpleNamespace(),
                    boundary_search_budget=_BoundarySearchBudget(
                        structural_edge_bound=4
                    ),
                )
                after_review["weighted_score"] = 60.0
                after_review["boundary_score"] = {
                    "issue_fraction": 0.6,
                    "p10": 40.0,
                }
                not_improved = _attempt_proxy_triggered_partial_reedit(
                    original_bundle,
                    materialized,  # type: ignore[arg-type]
                    original_review,
                    request=request,  # type: ignore[arg-type]
                    eligible_indices=(0, 1),
                    audio_offsets=(0.0, 0.0),
                    scene_cut_times=((), ()),
                    source_timewarps=(None, None),
                    source_aspect_ratios=(16.0 / 9.0, 16.0 / 9.0),
                    source_durations_seconds=(10.0, 10.0),
                    beat_period_seconds=0.5,
                    phrase_times=(),
                    section_plan=None,
                    common_master_start=0.0,
                    width=1920,
                    height=1080,
                    scratch_directory=root,
                    boundary_descriptor_cache=SimpleNamespace(),
                    splice_feasibility_graph=SimpleNamespace(),
                    boundary_search_budget=_BoundarySearchBudget(
                        structural_edge_bound=4
                    ),
                )
                successful_review_kwargs = dict(proxy_review.call_args.kwargs)
                self.assertEqual(proxy_review.call_count, 2)
                derive.return_value = polluted_bundle
                proxy_review.reset_mock()
                contaminated = _attempt_proxy_triggered_partial_reedit(
                    original_bundle,
                    materialized,  # type: ignore[arg-type]
                    original_review,
                    request=request,  # type: ignore[arg-type]
                    eligible_indices=(0, 1),
                    audio_offsets=(0.0, 0.0),
                    scene_cut_times=((), ()),
                    source_timewarps=(None, None),
                    source_aspect_ratios=(16.0 / 9.0, 16.0 / 9.0),
                    source_durations_seconds=(10.0, 10.0),
                    beat_period_seconds=0.5,
                    phrase_times=(),
                    section_plan=None,
                    common_master_start=0.0,
                    width=1920,
                    height=1080,
                    scratch_directory=root,
                    boundary_descriptor_cache=SimpleNamespace(),
                    splice_feasibility_graph=SimpleNamespace(),
                    boundary_search_budget=_BoundarySearchBudget(
                        structural_edge_bound=4
                    ),
                )
                proxy_review.assert_not_called()
                derive.return_value = partial_bundle
                proxy_review.return_value = SimpleNamespace(
                    selected=SimpleNamespace(
                        review_payload=None,
                        error="RuntimeError: synthetic partial decode failure",
                    )
                )
                review_failed = _attempt_proxy_triggered_partial_reedit(
                    original_bundle,
                    materialized,  # type: ignore[arg-type]
                    original_review,
                    request=request,  # type: ignore[arg-type]
                    eligible_indices=(0, 1),
                    audio_offsets=(0.0, 0.0),
                    scene_cut_times=((), ()),
                    source_timewarps=(None, None),
                    source_aspect_ratios=(16.0 / 9.0, 16.0 / 9.0),
                    source_durations_seconds=(10.0, 10.0),
                    beat_period_seconds=0.5,
                    phrase_times=(),
                    section_plan=None,
                    common_master_start=0.0,
                    width=1920,
                    height=1080,
                    scratch_directory=root,
                    boundary_descriptor_cache=SimpleNamespace(),
                    splice_feasibility_graph=SimpleNamespace(),
                    boundary_search_budget=_BoundarySearchBudget(
                        structural_edge_bound=4
                    ),
                )

            self.assertEqual(
                result.selected_bundle.render_plan,
                partial_bundle.render_plan,
            )
            self.assertIs(result.selected_bundle.boundary_repairs, targeted)
            self.assertEqual(
                result.selected_bundle.splice_graph_authorization,
                {
                    "proxy_eligible": True,
                    "geometry_proof_boundary_indices": [],
                },
            )
            self.assertEqual(
                result.selected_review_payload["weighted_score"],
                80.0,
            )
            self.assertTrue(result.audit_payload["attempted"])
            self.assertNotIn(
                "learned_reviewer_issue_suggestions",
                result.audit_payload,
            )
            self.assertTrue(result.audit_payload["improved"])
            self.assertTrue(
                result.audit_payload["outside_issue_boundaries_preserved"]
            )
            self.assertTrue(
                all(result.audit_payload["invariant_checks"].values())
            )
            self.assertEqual(result.audit_payload["issue_boundary_indices"], [0])
            self.assertEqual(result.audit_payload["changed_boundary_indices"], [0])
            self.assertEqual(result.audit_payload["selected_version"], "partial_reedit")
            self.assertIs(not_improved.selected_bundle, original_bundle)
            self.assertEqual(
                not_improved.selected_review_payload,
                original_review,
            )
            self.assertFalse(not_improved.audit_payload["improved"])
            self.assertEqual(
                not_improved.audit_payload["selected_version"],
                "original",
            )
            self.assertIs(contaminated.selected_bundle, original_bundle)
            self.assertEqual(
                contaminated.selected_review_payload,
                original_review,
            )
            self.assertFalse(
                contaminated.audit_payload[
                    "outside_issue_boundaries_preserved"
                ]
            )
            self.assertEqual(
                contaminated.audit_payload["reason"],
                "partial_reedit_invariant_violation_original_retained",
            )
            self.assertIn(
                "source_sequence_changed",
                contaminated.audit_payload["invariant_errors"],
            )
            self.assertIn(
                "source_sequence_changed",
                contaminated.audit_payload["error"],
            )
            self.assertIs(review_failed.selected_bundle, original_bundle)
            self.assertEqual(
                review_failed.selected_review_payload,
                original_review,
            )
            self.assertEqual(
                review_failed.audit_payload["reason"],
                "partial_reedit_proxy_review_failed_original_retained",
            )
            self.assertEqual(
                review_failed.audit_payload["partial_review_error"],
                "RuntimeError: synthetic partial decode failure",
            )
            self.assertIsNone(review_failed.audit_payload["after_review"])
            self.assertFalse(review_failed.audit_payload["improved"])
            self.assertEqual(
                review_failed.audit_payload["selected_version"],
                "original",
            )
            self.assertEqual(repair.call_count, 4)
            repair_kwargs = repair.call_args.kwargs
            self.assertEqual(repair_kwargs["boundary_indices"], (0,))
            self.assertEqual(repair_kwargs["config"].search_radius_frames, 8)
            self.assertEqual(
                repair_kwargs["config"].interior_scene_cut_guard_frames,
                9,
            )
            self.assertEqual(derive.call_count, 4)
            self.assertIs(
                derive.call_args.kwargs["precomputed_boundary_repairs"],
                targeted,
            )
            self.assertEqual(proxy_review.call_count, 1)
            self.assertEqual(
                successful_review_kwargs["maximum_review_count"],
                1,
            )

    def test_partial_reedit_targets_only_introduced_cross_source_issues(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = (root / "source-a.mp4", root / "source-b.mp4")
            visual = VisualTimelinePlan(
                (
                    _visual_choice(0, 0, 0.0, 1.0),
                    _visual_choice(1, 0, 1.0, 2.0),
                    _visual_choice(2, 1, 2.0, 3.0),
                ),
                8.0,
            )
            transitions = GroupTransitionPlan(
                (
                    RenderTransition(
                        tier="D",
                        reason="same_source_hold_no_visible_switch",
                    ),
                    RenderTransition(tier="C", reason="synthetic_cross"),
                )
            )
            plan = RenderPlan(
                sources=sources,
                segments=(
                    RenderSegment(0, 0.0, 1.0, 0.0),
                    RenderSegment(0, 1.0, 1.0, 1.0),
                    RenderSegment(1, 0.0, 1.0, 2.0),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=root / "original.mp4",
                transitions=transitions.transitions,
            )
            bundle = _GroupRenderCandidateBundle(
                candidate_index=4,
                review_rank=0,
                cadence_profile="synthetic",
                pre_render_score=90.0,
                visual_timeline=visual,
                boundary_repairs=SimpleNamespace(),
                timeline_review=None,
                face_match_plan=None,
                group_transition_plan=transitions,
                boundary_affine_plan=None,
                editorial_motion_plan=None,
                render_plan=plan,
                splice_graph_authorization={"proxy_eligible": True},
            )
            targeted = SimpleNamespace(
                timeline=visual,
                repairs=(),
                changed_count=0,
                to_dict=lambda: {
                    "requested_boundary_indices": [1],
                    "changed_count": 0,
                },
            )
            review = {
                "passed": False,
                "weighted_score": 70.0,
                # Schema-v3 field must win over the legacy all-issue list.
                "partial_reedit_boundary_issue_indices": [0],
                "boundary_issue_indices": [0],
                "visible_boundary_score": {
                    "issue_fraction": 1.0,
                    "p10": 50.0,
                },
            }
            request = SimpleNamespace(
                sources=sources,
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
            )

            with patch(
                "crosscut.pipeline._budgeted_repair_visual_timeline_boundaries",
                return_value=targeted,
            ) as repair, patch(
                "crosscut.pipeline._record_boundary_repair_assessments",
            ):
                result = _attempt_proxy_triggered_partial_reedit(
                    bundle,
                    SimpleNamespace(candidate_index=4),  # type: ignore[arg-type]
                    review,
                    request=request,  # type: ignore[arg-type]
                    eligible_indices=(0, 1),
                    audio_offsets=(0.0, 0.0),
                    scene_cut_times=((), ()),
                    source_timewarps=(None, None),
                    source_aspect_ratios=(16.0 / 9.0, 16.0 / 9.0),
                    source_durations_seconds=(10.0, 10.0),
                    beat_period_seconds=0.5,
                    phrase_times=(),
                    section_plan=None,
                    common_master_start=0.0,
                    width=1920,
                    height=1080,
                    scratch_directory=root,
                    learned_issue_boundary_indices=(1,),
                    boundary_descriptor_cache=SimpleNamespace(),
                    splice_feasibility_graph=SimpleNamespace(),
                    boundary_search_budget=_BoundarySearchBudget(
                        structural_edge_bound=1
                    ),
                )

            self.assertIs(result.selected_bundle, bundle)
            self.assertEqual(repair.call_args.kwargs["boundary_indices"], (1,))
            audit = result.audit_payload
            self.assertEqual(
                audit["issue_index_source_field"],
                (
                    "partial_reedit_boundary_issue_indices_plus_"
                    "learned_reviewer_suggestions"
                ),
            )
            self.assertEqual(audit["issue_boundary_indices"], [1])
            self.assertEqual(audit["non_introduced_issue_boundary_indices"], [0])
            self.assertIn(0, audit["ignored_issue_boundary_indices"])
            self.assertEqual(
                audit["learned_reviewer_issue_suggestions"][
                    "used_boundary_indices"
                ],
                [1],
            )
            self.assertTrue(
                audit["learned_reviewer_issue_suggestions"][
                    "deterministic_commit_authority_unchanged"
                ]
            )
            self.assertEqual(
                audit["reason"],
                "targeted_repair_made_no_timing_change",
            )

    def test_automatic_group_joint_stream_delivers_first_pass_and_blocks_exhaustion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            montage_policy, _montage_path = _current_montage_policy_fixture(
                root
            )
            sources = tuple(root / f"source-{index}.mp4" for index in range(9))
            for source in sources:
                source.touch()
            request = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "review_first_enabled": False,
                    "sources": [str(source) for source in sources],
                    "output_directory": str(root / "out"),
                    "source_cache_directory": str(root / "source-cache"),
                }
            )
            alignment = AlignmentResult(
                reference=str(sources[0]),
                sample_rate=4_000,
                alignments=tuple(
                    SourceAlignment(
                        source=str(source),
                        is_reference=index == 0,
                        offset_samples=0,
                        offset_seconds=0.0,
                        score=1.0 if index == 0 else 0.9,
                        overlap_samples=48_000,
                        overlap_seconds=12.0,
                        score_margin=1.0 if index == 0 else 0.2,
                        overlap_ratio=1.0,
                    )
                    for index, source in enumerate(sources)
                ),
            )
            group_draft = GroupDraftTimeline(
                source_count=9,
                common_master_start=0.0,
                common_master_end=8.0,
                beat_period_seconds=0.5,
                audio_offsets=(0.0,) * 9,
                sync_scores=(1.0,) * 9,
                segments=(
                    DraftSegment(
                        segment_index=0,
                        source_index=0,
                        canonical_master_start=0.0,
                        canonical_master_end=8.0,
                        output_start_seconds=0.0,
                        source_local_start=0.0,
                        source_local_end=8.0,
                        audio_offset_seconds=0.0,
                        sync_score=1.0,
                        estimated_beat_count=16.0,
                        start_boundary_reasons=("range_edge",),
                        end_boundary_reasons=("range_edge",),
                    ),
                ),
            )

            class RhythmStub:
                bpm = 120.0
                beats = (object(),)
                phrase_candidates: tuple[object, ...] = ()
                beat_times_seconds = (0.0, 8.0)

                @staticmethod
                def to_dict() -> dict[str, object]:
                    return {}

            class ReviewStub:
                passed = True
                weighted_score = 90.0
                hard_failures: tuple[object, ...] = ()

                @staticmethod
                def to_dict() -> dict[str, object]:
                    return {"passed": True, "weighted_score": 90.0}

            transitions = GroupTransitionPlan(())
            bundles: list[_GroupRenderCandidateBundle] = []
            for index in range(8):
                visual = VisualTimelinePlan(
                    (_visual_choice(0, index, 0.0, 8.0),),
                    float(100 - index),
                    selected_candidate_index=index,
                )
                plan = RenderPlan(
                    sources=sources,
                    segments=(
                        RenderSegment(
                            source_index=index,
                            source_start=0.0,
                            duration=8.0,
                            master_start=0.0,
                        ),
                    ),
                    reference_source=0,
                    aspect_ratio=AspectRatio.LANDSCAPE,
                    output=root / f"candidate-{index}.mp4",
                    transitions=(),
                )
                bundles.append(
                    _GroupRenderCandidateBundle(
                        candidate_index=index,
                        review_rank=index,
                        cadence_profile=f"profile-{index}",
                        pre_render_score=float(100 - index),
                        visual_timeline=visual,
                        boundary_repairs=None,
                        timeline_review=ReviewStub(),  # type: ignore[arg-type]
                        face_match_plan=None,
                        group_transition_plan=transitions,
                        boundary_affine_plan=None,
                        editorial_motion_plan=None,
                        render_plan=plan,
                    )
                )

            section_plan = SectionPlan((SectionDirective(
                section_index=0,
                role=SectionRole.CHORUS_LIKE,
                start_seconds=0.0,
                end_seconds=8.0,
                energy_score=0.8,
                confidence=0.9,
                phrase_count=1,
                preferred_source_hold_beats=2,
                target_source_switches_per_minute=20.0,
                camera_motion_budget=0.5,
                target_roll_events_per_minute=0.0,
                establish_with_wide_shot=False,
                protect_last_beat_from_extra_cut=False,
            ),))
            candidate_set = SimpleNamespace(
                optimizer_workspace=SimpleNamespace(
                    candidates_by_segment=((SimpleNamespace(),),)
                ),
                candidates=(),
                ranked_candidates=(),
                to_dict=lambda: {"candidate_count": 8},
            )
            materialized = tuple(
                SimpleNamespace(
                    candidate_index=index,
                    review_rank=index,
                    candidate_history_entry={"candidate_index": index},
                )
                for index in range(8)
            )
            records = tuple(
                SimpleNamespace(
                    candidate_index=index,
                    cadence_profile=f"profile-{index}",
                    timeline_result=SimpleNamespace(
                        candidates=(SimpleNamespace(source_id=str(index)),)
                    ),
                    review=SimpleNamespace(passed=True, boundaries=()),
                )
                for index in range(8)
            )
            proposal_pool = SimpleNamespace(
                paths=tuple(range(8)),
                scalar_audit=lambda: {"path_count": 8},
            )
            look_result = SimpleNamespace(
                adjustments=tuple(SourceLookAdjustment() for _ in sources),
                target_mode="explicit_reference",
                adjustment_for_source=lambda index: SourceLookAdjustment(),
                to_dict=lambda: {"target_mode": "explicit_reference", "evidence": []},
            )

            def materialize_joint_batch(**kwargs):
                kwargs["provider_audit_sink"]({"closed": True})
                batch_index = kwargs["batch_index"]
                if batch_index >= 2:
                    raise NoCompleteTimelinePath(
                        "synthetic complete path pool exhausted",
                        reason_code="joint_visual_complete_path_no_good_exhausted",
                        rejection_counts={"proof_materialization_call_count": 0},
                    )
                start = batch_index * 4
                selection_set = SimpleNamespace(
                    analysis_basis="synthetic_joint_visual_selection",
                    candidate_records=records[start:start + 4],
                    unique_source_shot_path_count=4,
                    proof_variant_supplement_count=0,
                    selected_candidate_index=0,
                    role_control_audits=(),
                    shared_bridge_evidence_cache_audit={},
                    shared_option_evidence_cache_audit={},
                    profile_failures=(),
                )
                return selection_set, materialized[start:start + 4]

            def globally_index_joint_batch(candidates, *, first_candidate_index, **_kwargs):
                return tuple(
                    SimpleNamespace(
                        materialized=item,
                        candidate_index=first_candidate_index + offset,
                        review_rank=item.review_rank,
                    )
                    for offset, item in enumerate(candidates)
                )

            # The orchestration test substitutes pixel/identity producers;
            # their exact evidence contracts have dedicated integration tests.
            # The actual stream reducer, budgets, cohort selection, and final
            # render handoff below must still enforce a passing delivery.
            cases = (
                ("later_pass", 6, None, None),
                ("montage_failed", 6, None, None),
                ("all_failed", None, None, None),
                ("derivation_failed", 6, 6, None),
                ("decode_failed", 6, None, 6),
            )
            for name, winner, derivation_failure, decode_failure in cases:
                with self.subTest(case=name):
                    case_request = replace(request, output_directory=root / name)

                    def derive_candidate(item, **_kwargs):
                        if item.candidate_index == derivation_failure:
                            raise ValueError("synthetic complete derivation failure")
                        return bundles[item.candidate_index]

                    def review_one(candidate_bundles, **_kwargs):
                        self.assertEqual(len(candidate_bundles), 1)
                        bundle = candidate_bundles[0]
                        candidate = RenderedCandidate(
                            candidate_index=bundle.candidate_index,
                            render_plan=bundle.render_plan,
                            pre_render_score=bundle.pre_render_score,
                            payload=bundle,
                        )
                        if bundle.candidate_index == decode_failure:
                            review = RenderedCandidateReview(
                                candidate, None, error="synthetic decoder failure",
                            )
                        else:
                            review = RenderedCandidateReview(
                                candidate,
                                {
                                    "passed": bundle.candidate_index == winner,
                                    "weighted_score": 96.0,
                                    "partial_reedit_boundary_issue_indices": [],
                                    "visible_boundary_score": {
                                        "issue_fraction": 0.0,
                                        "p10": 96.0,
                                    },
                                },
                            )
                        return SimpleNamespace(reviews=(review,))

                    materialize = Mock(side_effect=materialize_joint_batch)
                    derive = Mock(side_effect=derive_candidate)
                    review = Mock(side_effect=review_one)
                    render_video = Mock(return_value="h264_videotoolbox")

                    def qa_handoff(*, manifest, automatic_ive_group_delivery, **_kwargs):
                        render_video.assert_called_once()
                        self.assertTrue(automatic_ive_group_delivery)
                        self.assertEqual(manifest["artifact_role"], "pending_qa")
                        self.assertFalse(manifest["app_result_exposure"]["video_output_exposed"])
                        manifest["artifact_role"] = "delivery"
                        manifest["app_result_exposure"].update({
                            "video_output_exposed": True,
                            "delivery_completion_message_allowed": True,
                        })

                    post_render_qa = Mock(side_effect=qa_handoff)
                    look = Mock(return_value=look_result)
                    seed = Mock(return_value=candidate_set)
                    with patch.multiple(
                        "crosscut.pipeline",
                        _load_stage_edit_montage_policy=Mock(return_value=montage_policy),
                        _enforce_automatic_group_stage_edit_requirement=Mock(return_value=None),
                        probe_media=Mock(side_effect=lambda source: _analysis_media(source, duration=12.0)),
                        _align_sequentially=Mock(return_value=(alignment, None)),
                        analyze_rhythm=Mock(return_value=RhythmStub()),
                        build_section_plan=Mock(return_value=section_plan),
                        analyze_source_derivative_diversity=Mock(
                            return_value=_source_derivative_diversity_fixture(len(sources))
                        ),
                        detect_scene_changes_in_video_cached=Mock(
                            return_value=((), {"detection_policy": "synthetic_test"})
                        ),
                        analyze_source_look_match=look,
                        estimate_shared_performance_range=Mock(
                            side_effect=ValueError("synthetic range unavailable")
                        ),
                        build_group_draft=Mock(return_value=group_draft),
                        build_visual_timeline_optimizer_seed=seed,
                        build_joint_proposal_pool=Mock(return_value=proposal_pool),
                        _materialize_joint_feedback_batch_with_fresh_provider=materialize,
                        _joint_selection_proof_materialization_call_count=(
                            lambda selection, **_kwargs: len(selection.candidate_records)
                        ),
                        joint_candidate_review_summaries=lambda *_args, **_kwargs: (),
                        _joint_selection_retry_audit_payload=lambda *_args, **_kwargs: {},
                        _exact_source_shot_path=lambda item: ((str(item.candidate_index), "shot"),),
                        _pre_review_hard_failure_codes=lambda _item: (),
                        _globally_index_joint_visual_candidate_batch=globally_index_joint_batch,
                        _joint_complete_path_signature=lambda item: ("synthetic", item.candidate_index),
                        _record_bound_joint_decoded_feedback=Mock(
                            return_value=({"feedback_binding_valid": True}, None)
                        ),
                        _derive_complete_group_candidate=derive,
                        _review_complete_group_candidates=review,
                        _review_stage_edit_bundle_montage=Mock(return_value=(
                            {
                                "passed": name != "montage_failed",
                                "failures": [],
                                "highlight_aggregate": {
                                    "soft_targets": {"aggregate_alignment_score": 0.80}
                                },
                            },
                            {
                                "stage_edit_exact_shot_clock": None,
                                "dynamic_crop_version": None,
                                "dynamic_crop_v3_decodes": None,
                                "dynamic_crop_motion_v3": None,
                            },
                        )),
                        _enforce_automatic_stage_edit_post_render_qa=post_render_qa,
                        write_plan=Mock(),
                        render=render_video,
                    ):
                        if name == "later_pass":
                            artifact = analyze(
                                case_request, _MemoryReporter(),
                                review_rendered_proxies=True,
                            )
                            report = artifact.manifest["rendered_candidate_selection"]
                            self.assertEqual(report["selected_candidate_index"], 6)
                            self.assertEqual(report["complete_candidate_count"], 1)
                            self.assertEqual(report["successful_review_count"], 1)
                            self.assertTrue(report["exact_complete_candidate_count_satisfied"])
                            self.assertEqual(materialize.call_count, 2)
                            self.assertEqual(derive.call_count, 7)
                            self.assertEqual(review.call_count, 7)
                            render_video.assert_called_once()
                            post_render_qa.assert_called_once()
                            self.assertEqual(render_video.call_args.args[0].segments[0].source_index, 6)
                            self.assertEqual(
                                artifact.manifest["render"]["rendered_proxy_review"]["weighted_score"],
                                96.0,
                            )
                        elif name == "montage_failed":
                            with self.assertRaisesRegex(
                                RuntimeError, "blocked by decoded proxy review",
                            ):
                                analyze(
                                    case_request, _MemoryReporter(),
                                    review_rendered_proxies=True,
                                )
                            render_video.assert_not_called()
                            post_render_qa.assert_not_called()
                            report_path, = case_request.output_directory.glob(
                                "*-rendered-candidate-selection.json"
                            )
                            report = json.loads(report_path.read_text(encoding="utf-8"))
                            self.assertFalse(report["passed"])
                            self.assertEqual(report["complete_candidate_count"], 1)
                            self.assertEqual(report["selected_candidate_index"], 6)
                            self.assertEqual(materialize.call_count, 2)
                            self.assertEqual(review.call_count, 7)
                        else:
                            with self.assertRaisesRegex(
                                RuntimeError,
                                "requires 1 successfully decoded candidates; got 0",
                            ):
                                analyze(
                                    case_request, _MemoryReporter(),
                                    review_rendered_proxies=True,
                                )
                            render_video.assert_not_called()
                            post_render_qa.assert_not_called()
                            report_paths = tuple(case_request.output_directory.glob(
                                "*-rendered-candidate-selection.json"
                            ))
                            self.assertEqual(len(report_paths), 1)
                            report = json.loads(report_paths[0].read_text(encoding="utf-8"))
                            self.assertFalse(report["passed"])
                            self.assertEqual(report["complete_candidate_count"], 0)
                            self.assertEqual(report["successful_review_count"], 0)
                            self.assertEqual(
                                report["blocking_reason"],
                                "joint_decoded_exact_complete_path_exhausted",
                            )
                            self.assertEqual(materialize.call_count, 3)
                            self.assertEqual(derive.call_count, 8)
                            self.assertEqual(review.call_count, 7 if derivation_failure is not None else 8)
                            if derivation_failure is not None:
                                self.assertEqual(
                                    report["candidate_derivation_failures"][0]["candidate_index"], 6
                                )
                            if decode_failure is not None:
                                self.assertTrue(any(
                                    failure.get("error") == "synthetic decoder failure"
                                    for failure in report["candidate_decoded_review_failures"]
                                ))
                    self.assertEqual(
                        report["source_index_mapping"]["eligible_to_request"], list(range(9))
                    )
                    self.assertEqual(
                        report["nested_index_domains"][
                            "complete_candidates[].final_render_segments[].request_source_index"
                        ],
                        REQUEST_SOURCE_INDEX_DOMAIN,
                    )
                    look.assert_called_once()
                    self.assertIsNone(look.call_args.kwargs["reference_source"])
                    self.assertEqual(
                        seed.call_args.kwargs["feature_cache_directory"],
                        (root / "source-cache").resolve(),
                    )
                    self.assertTrue(all(
                        call.kwargs["source_look_match"] is look_result
                        for call in derive.call_args_list
                    ))
                    self.assertTrue(all(
                        call.kwargs["maximum_review_count"] == 1
                        and call.kwargs["minimum_required_candidate_count"] == 1
                        for call in review.call_args_list
                    ))

    def test_timeline_review_helper_logs_concise_reference_metrics(self) -> None:
        sources = tuple(
            source
            for source, count in enumerate((3, 3, 3, 3, 2, 2, 2, 2, 2, 2))
            for _ in range(count)
        )
        choices = tuple(
            _visual_choice(
                index,
                source,
                index * 1.4,
                (index + 1) * 1.4,
            )
            for index, source in enumerate(sources)
        )
        reporter = _MemoryReporter()
        warnings: list[str] = []

        review = _review_visual_timeline_for_render(
            VisualTimelinePlan(choices, 10.0),
            available_source_count=10,
            reporter=reporter,  # type: ignore[arg-type]
            warnings=warnings,
        )

        self.assertIsNotNone(review)
        assert review is not None
        self.assertTrue(review.passed)
        self.assertEqual(warnings, [])
        logs = [
            str(payload["message"])
            for event_type, payload in reporter.events
            if event_type == "log"
        ]
        self.assertEqual(len(logs), 1)
        self.assertIn("타임라인 사전 QA 통과", logs[0])
        self.assertIn("소스 10/10", logs[0])
        self.assertIn("교차 전환 9개 (39.1%, 16.07회/분)", logs[0])
        self.assertIn("시각 숏 중앙값 1.40초", logs[0])
        self.assertIn("소스 런 중앙값/최장 2.80/4.20초", logs[0])

    def test_group_review_first_retains_incumbent_at_search_deadline(self):
        for elapsed in (359.0, 360.0, 361.0, 599.0, 600.0):
            with self.subTest(elapsed=elapsed):
                self.test_automatic_group_blocks_failed_review_but_comparison_opt_in_and_local_render(
                    _cutoff_elapsed=elapsed
                )

    def test_automatic_group_optional_models_cannot_erase_complete_baseline(self):
        for case in ("missing", "discovery_error", "motion_error", "all_rejected",
                     "late_apply_error", "assembly_error", "malformed_boundary", "late_clock_error",
                     "cutoff", "explicit_registry_error"):
            with self.subTest(case=case):
                self.test_automatic_group_blocks_failed_review_but_comparison_opt_in_and_local_render(
                    _model_case=case)

    def test_automatic_group_blocks_failed_review_but_comparison_opt_in_and_local_render(
        self, _cutoff_elapsed=None, _model_case=None,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            montage_policy, _montage_path = _current_montage_policy_fixture(
                root
            )
            sources = tuple(root / f"source-{index}.mp4" for index in range(9))
            for source in sources:
                source.touch()
            request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "review_first_enabled": False,
                    "sources": [str(source) for source in sources],
                    "output_directory": str(root / "out"),
                }
            )
            automatic_request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "review_first_enabled": False,
                    "sources": [str(source) for source in sources],
                    "output_directory": str(root / "automatic-out"),
                }
            )
            alignment = AlignmentResult(
                reference=str(sources[0]),
                sample_rate=4_000,
                alignments=tuple(
                    SourceAlignment(
                        source=str(source),
                        is_reference=index == 0,
                        offset_samples=0,
                        offset_seconds=0.0,
                        score=1.0 if index == 0 else 0.9,
                        overlap_samples=48_000,
                        overlap_seconds=12.0,
                        score_margin=1.0 if index == 0 else 0.2,
                        overlap_ratio=1.0,
                    )
                    for index, source in enumerate(sources)
                ),
            )
            draft_segments = (
                DraftSegment(
                    segment_index=0,
                    source_index=0,
                    canonical_master_start=0.0,
                    canonical_master_end=4.0,
                    output_start_seconds=0.0,
                    source_local_start=0.0,
                    source_local_end=4.0,
                    audio_offset_seconds=0.0,
                    sync_score=1.0,
                    estimated_beat_count=8.0,
                    start_boundary_reasons=("range_edge",),
                    end_boundary_reasons=("phrase",),
                ),
                DraftSegment(
                    segment_index=1,
                    source_index=0,
                    canonical_master_start=4.0,
                    canonical_master_end=8.0,
                    output_start_seconds=4.0,
                    source_local_start=4.0,
                    source_local_end=8.0,
                    audio_offset_seconds=0.0,
                    sync_score=1.0,
                    estimated_beat_count=8.0,
                    start_boundary_reasons=("phrase",),
                    end_boundary_reasons=("range_edge",),
                ),
            )
            group_draft = GroupDraftTimeline(
                source_count=9,
                common_master_start=0.0,
                common_master_end=8.0,
                beat_period_seconds=0.5,
                audio_offsets=(0.0,) * 9,
                sync_scores=(1.0,) * 9,
                segments=draft_segments,
            )
            visual_timeline = VisualTimelinePlan(
                (
                    _visual_choice(0, 0, 0.0, 4.0),
                    _visual_choice(1, 0, 4.0, 8.0),
                ),
                2.0,
            )
            repairs = VisualBoundaryRepairPlan(
                timeline=visual_timeline,
                attempt_input_timelines=(visual_timeline,),
                repairs=(),
                skipped_same_source_boundaries=1,
                failures=(),
            )
            transitions = GroupTransitionPlan(
                (
                    RenderTransition(
                        tier="D",
                        reason="same_source_hold",
                    ),
                )
            )
            section_plan = SectionPlan((SectionDirective(
                section_index=0,
                role=SectionRole.VERSE_LIKE,
                start_seconds=0.0,
                end_seconds=8.0,
                energy_score=0.5,
                confidence=0.9,
                phrase_count=1,
                preferred_source_hold_beats=4,
                target_source_switches_per_minute=13.5,
                camera_motion_budget=0.34,
                target_roll_events_per_minute=0.0,
                establish_with_wide_shot=False,
                protect_last_beat_from_extra_cut=False,
            ),))

            class RhythmStub:
                bpm = 120.0
                beats = (object(),)
                phrase_candidates: tuple[object, ...] = ()
                beat_times_seconds = (0.0, 4.0, 8.0)

                @staticmethod
                def to_dict() -> dict[str, object]:
                    return {}

            reporter = _MemoryReporter()
            with patch(
                "crosscut.pipeline._load_stage_edit_montage_policy",
                return_value=montage_policy,
            ), patch(
                "crosscut.pipeline._enforce_automatic_group_stage_edit_requirement",
                return_value=None,
            ), patch(
                "crosscut.pipeline.probe_media",
                side_effect=lambda source: _analysis_media(source, duration=12.0),
            ), patch(
                "crosscut.pipeline._align_sequentially",
                return_value=(alignment, None),
            ), patch(
                "crosscut.pipeline.analyze_rhythm",
                return_value=RhythmStub(),
            ), patch(
                "crosscut.pipeline.build_section_plan",
                return_value=section_plan,
            ), patch(
                "crosscut.pipeline.analyze_source_derivative_diversity",
                return_value=_source_derivative_diversity_fixture(len(sources)),
            ), patch(
                "crosscut.pipeline.detect_scene_changes_in_video_cached",
                return_value=((), {"detection_policy": "synthetic_test"}),
            ), patch(
                "crosscut.pipeline.build_group_draft",
                return_value=group_draft,
            ), patch(
                "crosscut.pipeline.build_visual_timeline",
                return_value=visual_timeline,
            ) as baseline_builder, patch(
                "crosscut.pipeline.repair_visual_timeline_boundaries",
                return_value=repairs,
            ), patch(
                "crosscut.pipeline.build_face_match_plan",
                side_effect=ValueError("test keeps raw frames"),
            ), patch(
                "crosscut.pipeline.build_group_transition_plan",
                return_value=transitions,
            ), patch("crosscut.pipeline.write_plan"), patch(
                "crosscut.pipeline.render",
                return_value="h264_videotoolbox",
            ) as render_video:
                if _model_case is not None:
                    import crosscut.pipeline as pipeline
                    from crosscut.review_first_runtime import ReviewFirstRuntime, ReviewFirstSearchCutoff
                    from crosscut.stage_edit_production_registry import STAGE_EDIT_PRODUCTION_REGISTRY_ENV
                    from time import monotonic
                    class ReachedDecodedProxy(BaseException):
                        pass
                    now = [monotonic()]
                    runtime = ReviewFirstRuntime(enabled=True, _clock=lambda: now[0])
                    def discover(_request):
                        if _model_case == "explicit_registry_error":
                            baseline_builder.assert_not_called()
                            raise ValueError("explicit registry invalid")
                        baseline_builder.assert_called_once()
                        if _model_case == "discovery_error":
                            raise ValueError("optional registry invalid")
                        return None if _model_case in {"missing", "assembly_error", "malformed_boundary", "late_clock_error"} else SimpleNamespace()
                    generation = SimpleNamespace(
                        retained_candidates=tuple(SimpleNamespace(rank=index) for index in range(2)),
                        learned_section_director=SimpleNamespace(section_plan=section_plan))
                    context = dict(bundle=SimpleNamespace(to_dict=lambda: {"fixture": True}),
                        motion=None, policy=montage_policy, motion_profile=None, profile_payload=None,
                        montage_config=pipeline._rendered_montage_config_from_policy(
                            montage_policy, require_complete_role_duration_policy=True),
                        snapshot=None, causal_audit=None, generation=generation,
                        audit_path=root / "synthetic-generation-audit.json")
                    def rejected_candidate(*args, **kwargs):
                        if _model_case == "cutoff":
                            now[0] += 361.
                            raise ReviewFirstSearchCutoff("injected candidate cutoff")
                        if _model_case == "late_apply_error":
                            return SimpleNamespace(visual_plan=visual_timeline)
                        raise ValueError("injected visual candidate rejection")
                    def motion(bundle):
                        if bundle is not None:
                            baseline_builder.assert_called_once()
                            raise RuntimeError("optional motion authority invalid")
                        return None
                    with ExitStack() as stack:
                        stack.enter_context(patch.dict("os.environ", {}, clear=True))
                        if _model_case == "explicit_registry_error":
                            stack.enter_context(patch.dict("os.environ", {STAGE_EDIT_PRODUCTION_REGISTRY_ENV: str(root / 'invalid-registry')}))
                        stack.enter_context(patch('crosscut.pipeline.ReviewFirstRuntime', return_value=runtime))
                        stack.enter_context(patch('crosscut.pipeline._discover_automatic_stage_edit_production_bundle', side_effect=discover))
                        stack.enter_context(patch('crosscut.pipeline._registered_dynamic_crop_motion_authority_v3', side_effect=motion))
                        stack.enter_context(patch('crosscut.pipeline.analyze_source_look_match', return_value=None))
                        if _model_case == 'malformed_boundary':
                            bad_timeline = replace(visual_timeline, choices=(replace(visual_timeline.choices[0], segment_index=99),)+visual_timeline.choices[1:])
                            repair = VisualBoundaryRepairPlan(bad_timeline, (visual_timeline,), (), 0, ())
                            stack.enter_context(patch('crosscut.pipeline.repair_visual_timeline_boundaries', return_value=repair))
                        if _model_case == 'late_clock_error':
                            original_clock = pipeline._deterministic_exact_shot_clock_from_timeline
                            clock_calls = []
                            def late_clock(*args, **kwargs):
                                clock_calls.append(True)
                                if len(clock_calls) > 1:
                                    kwargs['phrase_times'] = (float('nan'),)
                                return original_clock(*args, **kwargs)
                            stack.enter_context(patch('crosscut.pipeline._deterministic_exact_shot_clock_from_timeline', side_effect=late_clock))
                        if _model_case == 'assembly_error':
                            bad_face = SimpleNamespace(applied_count=1, fallback_count=0,
                                pan_crop_for=Mock(side_effect=ValueError('invalid optional pan crop')))
                            stack.enter_context(patch('crosscut.pipeline.build_face_match_plan', return_value=bad_face))
                            stack.enter_context(patch('crosscut.pipeline.rehydrate_visual_timeline_dissolve_evidence', return_value=visual_timeline))
                        candidate = stack.enter_context(patch('crosscut.pipeline.materialize_stage_edit_visual_candidate', side_effect=rejected_candidate))
                        if _model_case in {"all_rejected", "late_apply_error", "cutoff"}:
                            def model_context(**kwargs):
                                baseline_builder.assert_called_once()
                                return context
                            stack.enter_context(patch('crosscut.pipeline._review_first_registered_group_generation', side_effect=model_context))
                        stack.enter_context(patch('crosscut.pipeline._apply_stage_edit_candidate_render_contract', side_effect=ValueError('injected late geometry rejection')))
                        proxy = stack.enter_context(patch('crosscut.pipeline.render_and_review_proxy', side_effect=ReachedDecodedProxy))
                        if _model_case == "explicit_registry_error":
                            with self.assertRaisesRegex(ValueError, 'explicit registry invalid'):
                                analyze(replace(automatic_request, review_first_enabled=True), reporter, review_rendered_proxies=True)
                            proxy.assert_not_called()
                        else:
                            with self.assertRaises(ReachedDecodedProxy):
                                analyze(replace(automatic_request, review_first_enabled=True), reporter, review_rendered_proxies=True)
                            plan = proxy.call_args.args[0]
                            self.assertIsInstance(plan, RenderPlan)
                            self.assertEqual(plan.sources, tuple(path.resolve() for path in sources))
                            self.assertEqual(plan.output_frame_count, round(8 * plan.fps))
                            audit_paths = list(automatic_request.output_directory.glob('*-review-first-selection-audit.json'))
                            self.assertEqual(len(audit_paths), 1)
                            audit = json.loads(audit_paths[0].read_text())
                            self.assertEqual(audit['authority'], 'deterministic_full_song_visual_selector')
                            self.assertTrue(audit['first_result_controller']['baseline_retained'])
                            if _model_case in {"all_rejected", "late_apply_error"}:
                                self.assertEqual(len(audit['stage_edit_rejections']), 2)
                                self.assertEqual(candidate.call_count, 2)
                                self.assertFalse(audit['unselected_generation_attempt']['selected_for_render'])
                            if _model_case == 'cutoff':
                                self.assertEqual(candidate.call_count, 1)
                                self.assertIn('search_cutoff', audit)
                            if _model_case in {'discovery_error', 'motion_error'}:
                                self.assertTrue(any(row['stage'] == 'default_registered_models' and row['status'] == 'rejected'
                                    for row in audit['first_result_controller']['attempts']))
                            if _model_case == 'missing':
                                self.assertTrue(any(row['stage'] == 'default_registered_models' and row['status'] == 'unavailable'
                                    for row in audit['first_result_controller']['attempts']))
                            if _model_case in {'assembly_error', 'late_clock_error'}:
                                self.assertTrue(any(row['stage'] == 'final_geometry_plan' and row['status'] == 'rejected'
                                    for row in audit['first_result_controller']['attempts']))
                            if _model_case == 'malformed_boundary':
                                self.assertTrue(any(row['stage'] == 'boundary_repair' and row['status'] == 'rejected'
                                    for row in audit['first_result_controller']['attempts']))
                        render_video.assert_not_called()  # Sentinel stops at real decoded-proxy entry.
                    return
                if _cutoff_elapsed is not None:
                    from crosscut.review_first_runtime import ReviewFirstRuntime, ReviewFirstDeadlineExceeded
                    now = [0.]
                    runtime = ReviewFirstRuntime(enabled=True, _clock=lambda:now[0])
                    now[0] = _cutoff_elapsed
                    with patch('crosscut.pipeline.ReviewFirstRuntime', return_value=runtime):
                        if _cutoff_elapsed >= 600.:
                            with self.assertRaises(ReviewFirstDeadlineExceeded):
                                analyze(replace(request,review_first_enabled=True), reporter)
                            render_video.assert_not_called()
                        else:
                            bounded_artifact = analyze(replace(request,review_first_enabled=True), reporter)
                            baseline_builder.assert_called_once()
                            render_video.assert_called_once()
                            self.assertEqual(baseline_builder.call_args.kwargs['maximum_results'],1)
                            self.assertEqual(len(render_video.call_args.args[0].segments),2)
                            self.assertIsNotNone(bounded_artifact.manifest['visual_timeline'])
                            if _cutoff_elapsed >= 360.:
                                self.assertTrue(any('확보한 전체 경로' in item for item in bounded_artifact.manifest['warnings']))
                    return
                with self.assertRaisesRegex(
                    RuntimeError,
                    "automatic GROUP render blocked by timeline review failures",
                ):
                    analyze(
                        automatic_request,
                        reporter,  # type: ignore[arg-type]
                    )
                render_video.assert_not_called()
                automatic_artifact = analyze(
                    automatic_request,
                    reporter,  # type: ignore[arg-type]
                    allow_failed_comparison_render=True,
                )
                artifact = analyze(
                    request,
                    reporter,  # type: ignore[arg-type]
                )

            self.assertEqual(render_video.call_count, 2)
            self.assertTrue(
                automatic_artifact.manifest["render"][
                    "allow_failed_comparison_render"
                ]
            )
            self.assertEqual(
                automatic_artifact.manifest["artifact_role"],
                "comparison",
            )
            self.assertFalse(
                automatic_artifact.manifest["app_result_exposure"][
                    "video_output_exposed"
                ]
            )
            self.assertFalse(
                any(
                    path.suffix.lower() == ".mp4"
                    for path in automatic_artifact.output_paths
                )
            )
            performance_range_trim = automatic_artifact.manifest[
                "performance_range_trim"
            ]
            self.assertIsNotNone(performance_range_trim)
            self.assertFalse(performance_range_trim["applied"])
            self.assertFalse(performance_range_trim["evidence_strong"])
            self.assertEqual(
                performance_range_trim["reason"],
                "insufficient_decoded_peer_sources",
            )
            self.assertFalse(
                performance_range_trim["privacy"]["raw_pcm_persisted"]
            )
            self.assertIsNone(artifact.manifest["performance_range_trim"])
            self.assertFalse(
                artifact.manifest["render"]["allow_failed_comparison_render"]
            )
            review_payload = artifact.manifest["timeline_review"]
            self.assertIsNotNone(review_payload)
            self.assertFalse(review_payload["passed"])
            self.assertEqual(
                review_payload["index_domain"],
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            self.assertEqual(review_payload["unique_source_count"], 1)
            hard_codes = {
                item["code"] for item in review_payload["hard_failures"]
            }
            self.assertIn("insufficient_source_diversity", hard_codes)
            self.assertIn("dominant_source_share_hard", hard_codes)
            self.assertEqual(
                artifact.manifest["render"]["timeline_review"],
                review_payload,
            )
            self.assertTrue(
                any(
                    "비교용 렌더를 계속했습니다" in warning
                    for warning in artifact.manifest["warnings"]
                )
            )
            self.assertTrue(
                any(
                    event_type == "log"
                    and "타임라인 사전 QA 실패" in str(payload.get("message"))
                    for event_type, payload in reporter.events
                )
            )

    def test_automatic_group_visual_failure_requires_comparison_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            montage_policy, _montage_path = _current_montage_policy_fixture(
                root
            )
            sources = tuple(root / f"source-{index}.mp4" for index in range(9))
            for source in sources:
                source.touch()
            request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "review_first_enabled": False,
                    "sources": [str(source) for source in sources],
                    "output_directory": str(root / "out"),
                }
            )
            alignment = AlignmentResult(
                reference=str(sources[0]),
                sample_rate=4_000,
                alignments=tuple(
                    SourceAlignment(
                        str(source),
                        index == 0,
                        0,
                        0.0,
                        1.0 if index == 0 else 0.9,
                        48_000,
                        12.0,
                        score_margin=1.0 if index == 0 else 0.2,
                    )
                    for index, source in enumerate(sources)
                ),
            )
            mapped = _linear_timewarp(duration=12.0)
            timewarps = (mapped,) * len(sources)
            draft = GroupDraftTimeline(
                source_count=len(sources),
                common_master_start=0.0,
                common_master_end=8.0,
                beat_period_seconds=0.5,
                audio_offsets=(0.0,) * len(sources),
                sync_scores=(1.0,) * len(sources),
                segments=(
                    DraftSegment(
                        segment_index=0,
                        source_index=0,
                        canonical_master_start=0.0,
                        canonical_master_end=8.0,
                        output_start_seconds=0.0,
                        source_local_start=0.0,
                        source_local_end=8.0,
                        audio_offset_seconds=0.0,
                        sync_score=1.0,
                        estimated_beat_count=16.0,
                        start_boundary_reasons=("range_edge",),
                        end_boundary_reasons=("range_edge",),
                    ),
                ),
            )

            class RhythmStub:
                bpm = 120.0
                beats = (object(),)
                phrase_candidates: tuple[object, ...] = ()
                beat_times_seconds = (0.0, 8.0)

                @staticmethod
                def to_dict() -> dict[str, object]:
                    return {}

            class PerformanceRangeStub:
                final_start_seconds = 0.0
                final_end_seconds = 8.0
                applied = True
                reason = "trim_applied"

                @staticmethod
                def to_dict() -> dict[str, object]:
                    return {
                        "analysis_basis": "synthetic_pipeline_order_test",
                        "scope": "automatic_group_only",
                        "applied": True,
                        "evidence_strong": True,
                        "reason": "trim_applied",
                        "final_range": {
                            "start_seconds": 0.0,
                            "end_seconds": 8.0,
                            "duration_seconds": 8.0,
                        },
                        "privacy": {"raw_pcm_persisted": False},
                    }

            with patch(
                "crosscut.pipeline._load_stage_edit_montage_policy",
                return_value=montage_policy,
            ), patch(
                "crosscut.pipeline._enforce_automatic_group_stage_edit_requirement",
                return_value=None,
            ), patch(
                "crosscut.pipeline.probe_media",
                side_effect=lambda source: _analysis_media(source, duration=12.0),
            ), patch(
                "crosscut.pipeline._align_sequentially",
                return_value=(alignment, None),
            ), patch(
                "crosscut.pipeline._estimate_source_timewarps",
                return_value=(timewarps, ()),
            ), patch(
                "crosscut.pipeline.analyze_rhythm",
                return_value=RhythmStub(),
            ), patch(
                "crosscut.pipeline.analyze_source_derivative_diversity",
                return_value=_source_derivative_diversity_fixture(len(sources)),
            ), patch(
                "crosscut.pipeline.detect_scene_changes_in_video_cached",
                return_value=((), {"detection_policy": "synthetic_test"}),
            ), patch(
                "crosscut.pipeline.estimate_shared_performance_range",
                return_value=PerformanceRangeStub(),
            ) as estimate_performance_range, patch(
                "crosscut.pipeline.build_group_draft",
                return_value=draft,
            ) as build_draft, patch(
                "crosscut.pipeline.build_visual_timeline",
                side_effect=ValueError("synthetic visual failure"),
            ), patch("crosscut.pipeline.write_plan"), patch(
                "crosscut.pipeline.render",
                return_value="h264_videotoolbox",
            ) as render_video:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "automatic GROUP render requires a visual timeline",
                ):
                    analyze(
                        request,
                        _MemoryReporter(),  # type: ignore[arg-type]
                    )
                render_video.assert_not_called()
                artifact = analyze(
                    request,
                    _MemoryReporter(),  # type: ignore[arg-type]
                    allow_failed_comparison_render=True,
                )

            render_video.assert_called_once()
            self.assertEqual(estimate_performance_range.call_count, 2)
            self.assertEqual(build_draft.call_count, 2)
            self.assertEqual(
                build_draft.call_args.kwargs["common_master_start"],
                0.0,
            )
            self.assertEqual(
                build_draft.call_args.kwargs["common_master_end"],
                8.0,
            )
            plan = render_video.call_args.args[0]
            self.assertEqual(plan.master_audio_start, 0.0)
            self.assertAlmostEqual(plan.segments[0].source_start, 0.0)
            self.assertAlmostEqual(
                plan.segments[0].effective_source_duration,
                8.0 * 1.004,
            )
            self.assertEqual(
                artifact.manifest["render"]["source_clock_basis"],
                "local_monotonic_timewarp_with_per_source_scalar_fallback",
            )
            self.assertTrue(
                artifact.manifest["render"]["allow_failed_comparison_render"]
            )
            self.assertEqual(artifact.manifest["artifact_role"], "comparison")
            self.assertFalse(
                artifact.manifest["app_result_exposure"][
                    "video_output_exposed"
                ]
            )
            self.assertFalse(
                any(path.suffix.lower() == ".mp4" for path in artifact.output_paths)
            )
            self.assertEqual(
                artifact.manifest["common_master_range"]["end_seconds"],
                8.0,
            )
            self.assertTrue(
                artifact.manifest["performance_range_trim"]["applied"]
            )

    def test_performance_range_uses_nonzero_reference_index_and_clock(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            montage_policy, _montage_path = _current_montage_policy_fixture(
                root
            )
            sources = tuple(root / f"source-{index}.mp4" for index in range(9))
            for source in sources:
                source.touch()
            reference_index = 2
            request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "review_first_enabled": False,
                    "sources": [str(source) for source in sources],
                    "reference_source": reference_index,
                    "output_directory": str(root / "out"),
                }
            )
            alignment = AlignmentResult(
                reference=str(sources[reference_index]),
                sample_rate=4_000,
                alignments=tuple(
                    SourceAlignment(
                        str(source),
                        index == reference_index,
                        0,
                        0.0,
                        1.0 if index == reference_index else 0.9,
                        48_000,
                        12.0,
                        score_margin=(
                            1.0 if index == reference_index else 0.2
                        ),
                    )
                    for index, source in enumerate(sources)
                ),
            )
            timewarps = tuple(
                _linear_timewarp(duration=12.0, rate=1.0 + index * 0.0005)
                for index in range(len(sources))
            )
            draft = GroupDraftTimeline(
                source_count=len(sources),
                common_master_start=0.0,
                common_master_end=8.0,
                beat_period_seconds=0.5,
                audio_offsets=(0.0,) * len(sources),
                sync_scores=(1.0,) * len(sources),
                segments=(
                    DraftSegment(
                        segment_index=0,
                        source_index=reference_index,
                        canonical_master_start=0.0,
                        canonical_master_end=8.0,
                        output_start_seconds=0.0,
                        source_local_start=0.0,
                        source_local_end=8.0,
                        audio_offset_seconds=0.0,
                        sync_score=1.0,
                        estimated_beat_count=16.0,
                        start_boundary_reasons=("range_edge",),
                        end_boundary_reasons=("range_edge",),
                    ),
                ),
            )

            class RhythmStub:
                bpm = 120.0
                beats = (object(),)
                phrase_candidates: tuple[object, ...] = ()
                beat_times_seconds = (0.0, 8.0)

                @staticmethod
                def to_dict() -> dict[str, object]:
                    return {}

            class PerformanceRangeStub:
                final_start_seconds = 0.0
                final_end_seconds = 8.0
                applied = True
                reason = "trim_applied"

                @staticmethod
                def to_dict() -> dict[str, object]:
                    return {
                        "analysis_basis": "synthetic_nonzero_reference_test",
                        "scope": "automatic_group_only",
                        "applied": True,
                        "evidence_strong": True,
                        "reason": "trim_applied",
                        "final_range": {
                            "start_seconds": 0.0,
                            "end_seconds": 8.0,
                            "duration_seconds": 8.0,
                        },
                        "privacy": {"raw_pcm_persisted": False},
                    }

            performance_call: dict[str, object] = {}

            def populate_performance_pcm(*args, **kwargs):  # type: ignore[no-untyped-def]
                del args
                decoded = kwargs["decoded_pcm_by_source"]
                for index in range(len(sources)):
                    decoded[index] = (float(index),)
                return (timewarps, ())

            def estimate_performance(pcm_by_source, **kwargs):  # type: ignore[no-untyped-def]
                performance_call["pcm_source_indices"] = tuple(
                    sorted(pcm_by_source)
                )
                performance_call.update(kwargs)
                return PerformanceRangeStub()

            def scene_changes(source: Path, **kwargs):  # type: ignore[no-untyped-def]
                del kwargs
                index = int(source.stem.rsplit("-", 1)[-1])
                frame_index = 300 if index == reference_index else 30 + index
                return (SceneChange(frame_index, 0.9, 0.8, 0.7),)

            with patch(
                "crosscut.pipeline._load_stage_edit_montage_policy",
                return_value=montage_policy,
            ), patch(
                "crosscut.pipeline._enforce_automatic_group_stage_edit_requirement",
                return_value=None,
            ), patch(
                "crosscut.pipeline.probe_media",
                side_effect=lambda source: _analysis_media(source, duration=12.0),
            ), patch(
                "crosscut.pipeline._align_sequentially",
                return_value=(alignment, None),
            ), patch(
                "crosscut.pipeline._estimate_source_timewarps",
                side_effect=populate_performance_pcm,
            ), patch(
                "crosscut.pipeline.analyze_rhythm",
                return_value=RhythmStub(),
            ), patch(
                "crosscut.pipeline.analyze_source_derivative_diversity",
                return_value=_source_derivative_diversity_fixture(len(sources)),
            ), patch(
                "crosscut.pipeline.detect_scene_changes_in_video_cached",
                side_effect=lambda source, **kwargs: (
                    scene_changes(source, **kwargs),
                    {"detection_policy": "synthetic_test"},
                ),
            ), patch(
                "crosscut.pipeline.estimate_shared_performance_range",
                side_effect=estimate_performance,
            ) as estimate_performance_range, patch(
                "crosscut.pipeline.build_group_draft",
                return_value=draft,
            ), patch(
                "crosscut.pipeline.build_visual_timeline",
                side_effect=ValueError("synthetic visual stop"),
            ), patch("crosscut.pipeline.render") as render_video:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "automatic GROUP render requires a visual timeline",
                ):
                    analyze(
                        request,
                        _MemoryReporter(),  # type: ignore[arg-type]
                    )

            render_video.assert_not_called()
            estimate_performance_range.assert_called_once()
            self.assertEqual(
                performance_call["reference_source_index"],
                reference_index,
            )
            self.assertEqual(
                tuple(performance_call["source_indices"]),
                tuple(range(len(sources))),
            )
            self.assertEqual(
                performance_call["pcm_source_indices"],
                tuple(range(len(sources))),
            )
            self.assertIs(
                performance_call["source_timewarps"][reference_index],
                timewarps[reference_index],
            )
            expected_reference_cut = timewarps[
                reference_index
            ].source_to_master(10.0)
            self.assertEqual(
                len(performance_call["reference_scene_cut_master_seconds"]),
                1,
            )
            self.assertAlmostEqual(
                performance_call["reference_scene_cut_master_seconds"][0],
                expected_reference_cut,
            )

    def test_local_member_crop_failure_never_falls_back_to_full_frame(self) -> None:
        timing_segments = (
            DraftSegment(
                segment_index=0,
                source_index=0,
                canonical_master_start=0.0,
                canonical_master_end=1.0,
                output_start_seconds=0.0,
                source_local_start=0.0,
                source_local_end=1.0,
                audio_offset_seconds=0.0,
                sync_score=1.0,
                estimated_beat_count=2.0,
                start_boundary_reasons=("range_edge",),
                end_boundary_reasons=("beat",),
            ),
            DraftSegment(
                segment_index=1,
                source_index=1,
                canonical_master_start=1.0,
                canonical_master_end=2.0,
                output_start_seconds=1.0,
                source_local_start=1.0,
                source_local_end=2.0,
                audio_offset_seconds=0.0,
                sync_score=0.9,
                estimated_beat_count=2.0,
                start_boundary_reasons=("beat",),
                end_boundary_reasons=("beat",),
            ),
            DraftSegment(
                segment_index=2,
                source_index=0,
                canonical_master_start=2.0,
                canonical_master_end=3.0,
                output_start_seconds=2.0,
                source_local_start=2.0,
                source_local_end=3.0,
                audio_offset_seconds=0.0,
                sync_score=1.0,
                estimated_beat_count=2.0,
                start_boundary_reasons=("beat",),
                end_boundary_reasons=("range_edge",),
            ),
        )
        crop = (
            SegmentCropSample(
                time_seconds=0.0,
                x=0.2,
                y=0.1,
                width=0.4,
                height=0.4,
            ),
        )
        member_segments = (
            MemberDraftSegment(
                segment_index=0,
                canonical_master_start=0.0,
                canonical_master_end=1.0,
                output_start_seconds=0.0,
                kind=MemberSegmentKind.TARGET_CROP,
                source_index=0,
                source_local_start=0.0,
                source_local_end=1.0,
                audio_offset_seconds=0.0,
                crop_samples=crop,
                interval_coverage=1.0,
                mean_confidence=0.9,
                stability=0.9,
                selection_score=0.9,
            ),
            MemberDraftSegment(
                segment_index=1,
                canonical_master_start=1.0,
                canonical_master_end=2.0,
                output_start_seconds=1.0,
                kind=MemberSegmentKind.UNAVAILABLE,
                source_index=None,
                source_local_start=None,
                source_local_end=None,
                audio_offset_seconds=None,
                crop_samples=(),
                interval_coverage=0.0,
                mean_confidence=0.0,
                stability=0.0,
                selection_score=None,
            ),
            MemberDraftSegment(
                segment_index=2,
                canonical_master_start=2.0,
                canonical_master_end=3.0,
                output_start_seconds=2.0,
                kind=MemberSegmentKind.TARGET_CROP,
                source_index=0,
                source_local_start=2.0,
                source_local_end=3.0,
                audio_offset_seconds=0.0,
                crop_samples=crop,
                interval_coverage=1.0,
                mean_confidence=0.9,
                stability=0.9,
                selection_score=0.9,
            ),
        )
        with self.assertRaisesRegex(ValueError, "cannot fall back"):
            _member_render_segments(
                member_segments=member_segments,
                timing_segments=timing_segments,
                eligible_indices=(4, 7),
                common_master_start=0.0,
            )

    def test_member_editorial_tracks_attach_only_to_confirmed_subject_crop(self) -> None:
        member = MemberDraftSegment(
            segment_index=0,
            canonical_master_start=0.0,
            canonical_master_end=2.0,
            output_start_seconds=0.0,
            kind=MemberSegmentKind.TARGET_CROP,
            source_index=0,
            source_local_start=0.0,
            source_local_end=2.0,
            audio_offset_seconds=0.0,
            crop_samples=(
                SegmentCropSample(0.0, 0.30, 0.10, 0.25, 0.80),
            ),
            subject_samples=(
                SegmentSubjectSample(0.0, (0.36, 0.18, 0.49, 0.86), 0.95),
            ),
            interval_coverage=1.0,
            mean_confidence=0.95,
            stability=0.95,
            selection_score=0.95,
        )
        track = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.0),
                EditorialTransformKeyframe(0.75, 1.06),
            ),
            reason="confirmed_member_boundary_match",
        )
        plan = EditorialMotionPlan(
            segments=(EditorialMotionSegment(0, 0, track),),
            boundary_match_count=1,
            zoom_event_count=1,
            pan_event_count=0,
            roll_event_count=0,
            blocked_dissolve_segment_count=0,
        )
        base = RenderSegment(
            source_index=4,
            source_start=0.0,
            duration=2.0,
            master_start=0.0,
            pan_crop=member.to_pan_crop_track(),
        )

        attached = _attach_member_editorial_motion_tracks(
            member_segments=(member,),
            render_segments=(base,),
            motion_plan=plan,
        )

        self.assertIs(attached[0].editorial_transform, track)
        self.assertIs(attached[0].pan_crop, base.pan_crop)

    def test_member_editorial_no_track_preserves_existing_render_segment(self) -> None:
        unavailable = MemberDraftSegment(
            segment_index=0,
            canonical_master_start=0.0,
            canonical_master_end=2.0,
            output_start_seconds=0.0,
            kind=MemberSegmentKind.UNAVAILABLE,
            source_index=None,
            source_local_start=None,
            source_local_end=None,
            audio_offset_seconds=None,
            crop_samples=(),
            interval_coverage=0.0,
            mean_confidence=0.0,
            stability=0.0,
            selection_score=None,
        )
        plan = EditorialMotionPlan(
            segments=(EditorialMotionSegment(0, 0, None),),
            boundary_match_count=0,
            zoom_event_count=0,
            pan_event_count=0,
            roll_event_count=0,
            blocked_dissolve_segment_count=0,
        )
        base = RenderSegment(0, 0.0, 2.0, 0.0)

        attached = _attach_member_editorial_motion_tracks(
            member_segments=(unavailable,),
            render_segments=(base,),
            motion_plan=plan,
        )

        self.assertIs(attached[0], base)

    def test_member_editorial_track_rejects_unconfirmed_segment(self) -> None:
        unavailable = MemberDraftSegment(
            segment_index=0,
            canonical_master_start=0.0,
            canonical_master_end=2.0,
            output_start_seconds=0.0,
            kind=MemberSegmentKind.UNAVAILABLE,
            source_index=None,
            source_local_start=None,
            source_local_end=None,
            audio_offset_seconds=None,
            crop_samples=(),
            interval_coverage=0.0,
            mean_confidence=0.0,
            stability=0.0,
            selection_score=None,
        )
        track = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.0),
                EditorialTransformKeyframe(0.75, 1.06),
            )
        )
        plan = EditorialMotionPlan(
            segments=(EditorialMotionSegment(0, 0, track),),
            boundary_match_count=1,
            zoom_event_count=1,
            pan_event_count=0,
            roll_event_count=0,
            blocked_dissolve_segment_count=0,
        )

        with self.assertRaisesRegex(ValueError, "confirmed target geometry"):
            _attach_member_editorial_motion_tracks(
                member_segments=(unavailable,),
                render_segments=(RenderSegment(0, 0.0, 2.0, 0.0),),
                motion_plan=plan,
            )

    def test_direct_member_editorial_attach_rechecks_both_source_domains(self) -> None:
        track = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.0),
                EditorialTransformKeyframe(0.75, 1.06),
            ),
            reason="official_fancam_dense_face_scale_position_match",
        )
        selection = SimpleNamespace(source_indices=(1, 0))
        plan = EditorialMotionPlan(
            segments=(
                EditorialMotionSegment(0, 1, track),
                EditorialMotionSegment(1, 0, None),
            ),
            boundary_match_count=1,
            zoom_event_count=1,
            pan_event_count=0,
            roll_event_count=0,
            blocked_dissolve_segment_count=0,
            analysis_basis=(
                "official_target_fancam_unambiguous_face_"
                "dense_output_frame_review_v1"
            ),
        )
        renders = (
            RenderSegment(7, 0.0, 2.0, 0.0),
            RenderSegment(4, 2.0, 2.0, 2.0),
        )

        attached = _attach_direct_member_editorial_motion_tracks(
            visual_selection=selection,  # type: ignore[arg-type]
            eligible_indices=(4, 7),
            render_segments=renders,
            motion_plan=plan,
        )

        self.assertIs(attached[0].editorial_transform, track)
        self.assertIs(attached[1], renders[1])
        malformed = EditorialMotionPlan(
            segments=(
                EditorialMotionSegment(0, 0, track),
                EditorialMotionSegment(1, 0, None),
            ),
            boundary_match_count=1,
            zoom_event_count=1,
            pan_event_count=0,
            roll_event_count=0,
            blocked_dissolve_segment_count=0,
        )
        with self.assertRaisesRegex(ValueError, "source disagrees"):
            _attach_direct_member_editorial_motion_tracks(
                visual_selection=selection,  # type: ignore[arg-type]
                eligible_indices=(4, 7),
                render_segments=renders,
                motion_plan=malformed,
            )
        self.assertTrue(
            all(item.editorial_transform is None for item in renders)
        )

    def test_local_member_pipeline_calls_motion_planner_and_serializes_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = (root / "member-0.mp4", root / "member-1.mp4")
            for source in sources:
                source.touch()
            request = AnalysisRequest.from_dict(
                {
                    "song": "Synthetic",
                    "edit_mode": "member",
                    "aspect_ratio": "9:16",
                    "target_member": "가을",
                    "source_mode": "local",
                    "sources": [str(source) for source in sources],
                    "member_anchors": [
                        {
                            "source_index": index,
                            "time_seconds": 1.0,
                            "bbox_xyxy": [0.34, 0.16, 0.50, 0.86],
                        }
                        for index in range(2)
                    ],
                    "output_directory": str(root / "out"),
                }
            )
            alignment = AlignmentResult(
                reference=str(sources[0]),
                sample_rate=4_000,
                alignments=tuple(
                    SourceAlignment(
                        source=str(source),
                        is_reference=index == 0,
                        offset_samples=0,
                        offset_seconds=0.0,
                        score=1.0 if index == 0 else 0.9,
                        overlap_samples=16_000,
                        overlap_seconds=4.0,
                        score_margin=1.0 if index == 0 else 0.2,
                        overlap_ratio=1.0,
                    )
                    for index, source in enumerate(sources)
                ),
            )
            timing_segments = tuple(
                DraftSegment(
                    segment_index=index,
                    source_index=index,
                    canonical_master_start=2.0 * index,
                    canonical_master_end=2.0 * (index + 1),
                    output_start_seconds=2.0 * index,
                    source_local_start=2.0 * index,
                    source_local_end=2.0 * (index + 1),
                    audio_offset_seconds=0.0,
                    sync_score=1.0,
                    estimated_beat_count=4.0,
                    start_boundary_reasons=(
                        ("range_edge",) if index == 0 else ("beat",)
                    ),
                    end_boundary_reasons=(
                        ("range_edge",) if index == 1 else ("beat",)
                    ),
                )
                for index in range(2)
            )
            timing_draft = GroupDraftTimeline(
                source_count=2,
                common_master_start=0.0,
                common_master_end=4.0,
                beat_period_seconds=0.5,
                audio_offsets=(0.0, 0.0),
                sync_scores=(1.0, 1.0),
                segments=timing_segments,
            )
            member_segments = tuple(
                MemberDraftSegment(
                    segment_index=index,
                    canonical_master_start=2.0 * index,
                    canonical_master_end=2.0 * (index + 1),
                    output_start_seconds=2.0 * index,
                    kind=MemberSegmentKind.TARGET_CROP,
                    source_index=index,
                    source_local_start=2.0 * index,
                    source_local_end=2.0 * (index + 1),
                    audio_offset_seconds=0.0,
                    crop_samples=(
                        SegmentCropSample(0.0, 0.30, 0.10, 0.253125, 0.80),
                    ),
                    subject_samples=(
                        SegmentSubjectSample(
                            0.0,
                            (0.34, 0.16, 0.50, 0.86),
                            0.95,
                        ),
                    ),
                    interval_coverage=1.0,
                    mean_confidence=0.95,
                    stability=0.95,
                    selection_score=0.95,
                )
                for index in range(2)
            )
            member_draft = MemberDraftTimeline(
                target_member=IVEMember.GAEUL,
                output_aspect=AspectRatio.PORTRAIT,
                source_aspect_ratio=16.0 / 9.0,
                common_master_start=0.0,
                common_master_end=4.0,
                audio_offsets=(0.0, 0.0),
                group_wide_fallback_source=None,
                segments=member_segments,
                selection_diagnostics=MemberDraftSelectionDiagnostics(
                    policy="confirmed_subject_balanced_path_v1",
                    optimizer="synthetic",
                    quality_score_drop_limit=0.04,
                    maximum_distinct_quality_source_count=2,
                    safe_candidate_counts_by_segment=(2, 2),
                    quality_candidate_counts_by_segment=(2, 2),
                    safe_candidate_counts_by_source=(2, 2),
                    quality_candidate_counts_by_source=(2, 2),
                    selected_segment_counts_by_source=(1, 1),
                ),
            )
            track = EditorialTransformTrack(
                (
                    EditorialTransformKeyframe(0.0, 1.0),
                    EditorialTransformKeyframe(0.75, 1.06),
                ),
                reason="confirmed_member_boundary_match",
            )
            motion_plan = EditorialMotionPlan(
                segments=(
                    EditorialMotionSegment(0, 0, None, ("outgoing_hold",)),
                    EditorialMotionSegment(1, 1, track, ("boundary_match",)),
                ),
                boundary_match_count=1,
                zoom_event_count=1,
                pan_event_count=0,
                roll_event_count=0,
                blocked_dissolve_segment_count=0,
                skipped_boundary_reasons=(("already_scale_matched", 1),),
                analysis_basis="confirmed_member_subject_geometry_v1",
            )

            class RhythmStub:
                bpm = 120.0
                beats = (object(),)
                phrase_candidates: tuple[object, ...] = ()
                beat_times_seconds = (0.0, 2.0, 4.0)

                @staticmethod
                def to_dict() -> dict[str, object]:
                    return {}

            def scene_changes(source: Path, **_kwargs: object) -> tuple[SceneChange, ...]:
                source_index = int(source.stem.rsplit("-", 1)[-1])
                return (
                    SceneChange(
                        frame_index=30 * (source_index + 1),
                        score=0.9,
                        frame_difference=0.8,
                        histogram_distance=0.7,
                    ),
                )

            reporter = _MemoryReporter()
            with patch(
                "crosscut.pipeline.probe_media",
                side_effect=lambda source: _analysis_media(source, duration=4.0),
            ), patch(
                "crosscut.pipeline._align_sequentially",
                return_value=(alignment, None),
            ), patch(
                "crosscut.pipeline._estimate_source_timewarps",
                return_value=((None, None), ()),
            ), patch(
                "crosscut.pipeline.analyze_rhythm",
                return_value=RhythmStub(),
            ), patch(
                "crosscut.pipeline.detect_scene_changes_in_video_cached",
                side_effect=lambda source, **kwargs: (
                    scene_changes(source, **kwargs),
                    {"detection_policy": "synthetic_test"},
                ),
            ), patch(
                "crosscut.pipeline.track_anchored_member",
                return_value=_tracking("synthetic", 1.0),
            ), patch(
                "crosscut.pipeline._member_anchor_trim",
                return_value=(
                    0.0,
                    {
                        "usable_for_render": True,
                        "applied": False,
                        "full_frame_only_source_indices": [],
                    },
                ),
            ), patch(
                "crosscut.pipeline.build_group_draft",
                return_value=timing_draft,
            ), patch(
                "crosscut.pipeline.build_member_draft",
                return_value=member_draft,
            ), patch(
                "crosscut.pipeline.build_member_editorial_motion_plan",
                return_value=motion_plan,
            ) as build_motion, patch(
                "crosscut.pipeline.write_plan"
            ), patch(
                "crosscut.pipeline.render",
                return_value="h264_videotoolbox",
            ) as render_video:
                artifact = analyze(request, reporter)  # type: ignore[arg-type]

            build_motion.assert_called_once()
            call = build_motion.call_args.kwargs
            self.assertIs(call["member_segments"], member_draft.segments)
            self.assertTrue(
                all(
                    segment.editorial_transform is None
                    for segment in call["render_segments"]
                )
            )
            self.assertEqual(call["source_scene_cut_times"], ((1.0,), (2.0,)))
            self.assertEqual(call["fps"], 29.97)
            plan = render_video.call_args.args[0]
            self.assertIsNone(plan.segments[0].editorial_transform)
            self.assertIs(plan.segments[1].editorial_transform, track)
            self.assertEqual(
                artifact.manifest["editorial_motion"]["analysis_basis"],
                "confirmed_member_subject_geometry_v1",
            )
            self.assertEqual(
                artifact.manifest["editorial_motion"]["index_domain"],
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            self.assertEqual(
                artifact.manifest["source_index_mapping"][
                    "payload_index_domains"
                ][
                    "render.editorial_motion.segments[].source_index"
                ],
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            self.assertEqual(
                artifact.manifest["render"]["editorial_motion"][
                    "applied_segment_count"
                ],
                1,
            )
            self.assertEqual(
                artifact.manifest["render"]["final_segments"][1]["crop_mode"],
                "dynamic_pan_crop_plus_editorial_camera",
            )
            log_messages = [
                str(payload["message"])
                for event, payload in reporter.events
                if event == "log" and "message" in payload
            ]
            self.assertTrue(any("멤버 편집 카메라 1개" in item for item in log_messages))
            self.assertTrue(any("already_scale_matched 1개" in item for item in log_messages))

    def test_source_index_mapping_documents_gap_after_excluded_source(self) -> None:
        sources = tuple(Path(f"/tmp/source-{index}.mp4") for index in range(4))

        mapping = _source_index_mapping(sources, (0, 2, 3))

        self.assertEqual(mapping["request_index_domain"], REQUEST_SOURCE_INDEX_DOMAIN)
        self.assertEqual(mapping["eligible_index_domain"], ELIGIBLE_SOURCE_INDEX_DOMAIN)
        self.assertEqual(
            mapping["render_plan_index_domain"],
            RENDER_PLAN_SOURCE_INDEX_DOMAIN,
        )
        self.assertEqual(
            mapping["domain_equivalences"],
            {RENDER_PLAN_SOURCE_INDEX_DOMAIN: REQUEST_SOURCE_INDEX_DOMAIN},
        )
        self.assertEqual(mapping["eligible_to_request"], [0, 2, 3])
        self.assertEqual(mapping["request_to_eligible"], [0, None, 1, 2])
        self.assertEqual(
            [
                (item["eligible_source_index"], item["request_source_index"])
                for item in mapping["eligible_sources"]
            ],
            [(0, 0), (1, 2), (2, 3)],
        )
        domains = mapping["payload_index_domains"]
        for payload_path in (
            "request.sources[]",
            "request.reference_source",
            "request.member_anchors[].source_index",
            "media[]",
            "alignment.alignments[]",
            "alignment_eligibility[].source_index",
            "local_timewarps[].source_index",
            "scene_analysis[].source_index",
            "member_auto_anchors[].source_index",
            "member_tracking[].source_index",
            "member_anchor_trim.anchors[].source_index",
            "member_anchor_trim.limiting_source_indices[]",
            "member_anchor_trim.full_frame_only_source_indices[]",
            "member_direct_full_frame.eligible_request_source_indices[]",
            "member_direct_full_frame.segments[].request_source_index",
            "member_direct_full_frame.visual_selection.source_usage[].request_source_index",
            "member_direct_full_frame.visual_selection.span_content_validity[].sources[].request_source_index",
            "member_direct_full_frame.visual_selection.boundaries[].outgoing_request_source_index",
            "member_direct_full_frame.visual_selection.boundaries[].incoming_request_source_index",
            "render.automatic_full_frame_fallbacks[].request_source_index",
            "render.final_segments[].request_source_index",
            "rendered_proxy_review.source_quality.per_source[].source_index",
            "rendered_proxy_review.boundaries[].outgoing_source_index",
            "rendered_proxy_review.boundaries[].incoming_source_index",
            (
                "render.rendered_proxy_review.source_quality.per_source[]"
                ".source_index"
            ),
            "render.rendered_proxy_review.boundaries[].outgoing_source_index",
            "render.rendered_proxy_review.boundaries[].incoming_source_index",
            (
                "rendered_candidate_selection.complete_candidates[]"
                ".final_render_segments[].request_source_index"
            ),
            (
                "rendered_candidate_selection.candidates[].review"
                ".boundaries[].outgoing_source_index"
            ),
            (
                "rendered_candidate_selection.final_selected_review"
                ".source_quality.per_source[].source_index"
            ),
            (
                "rendered_candidate_selection.proxy_triggered_partial_reedit"
                ".after_review.boundaries[].incoming_source_index"
            ),
        ):
            self.assertEqual(domains[payload_path], REQUEST_SOURCE_INDEX_DOMAIN)
        for payload_path in (
            "render_plan.sources[]",
            "render_plan.reference_source",
            "render_plan.segments[].source_index",
            "member_direct_full_frame.final_source_usage.used_source_indices[]",
            "member_direct_full_frame.final_source_usage.sources[].source_index",
            "render.final_source_usage.used_source_indices[]",
            "render.final_source_usage.sources[].source_index",
            "render.cut_local_sync.used_source_indices[]",
            "render.cut_local_sync.source_decode_audit[].source_index",
            "render.cut_local_sync.per_boundary[].outgoing_source_index",
            "render.cut_local_sync.per_boundary[].incoming_source_index",
            "render.cut_local_sync.per_boundary[].renderer_frame_selection.outgoing.source_index",
            "render.cut_local_sync.per_boundary[].renderer_frame_selection.incoming.source_index",
        ):
            self.assertEqual(
                domains[payload_path],
                RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            )
        for payload_path in (
            "group_draft.segments[].source_index",
            "group_draft.audio_offsets[]",
            "group_draft.sync_scores[]",
            "visual_timeline.choices[].source_index",
            "face_match.segments[].source_index",
            "member_draft.segments[].source_index",
            "member_draft.audio_offsets[]",
            "member_draft.group_wide_fallback_source",
            "member_draft.selection_diagnostics.safe_source_indices[]",
            "member_draft.selection_diagnostics.quality_eligible_source_indices[]",
            "member_draft.selection_diagnostics.selected_source_indices[]",
            "member_draft.selection_diagnostics.no_safe_candidate_source_indices[]",
            "member_draft.selection_diagnostics.quality_guard_excluded_source_indices[]",
            "member_draft.selection_diagnostics.unused_quality_source_indices[]",
            "member_draft.selection_diagnostics.sources[].source_index",
            "member_direct_full_frame.segments[].eligible_source_index",
            "member_direct_full_frame.visual_selection.selected_source_indices[]",
            "member_direct_full_frame.visual_selection.source_usage[].eligible_source_index",
            "member_direct_full_frame.visual_selection.span_content_validity[].sources[].eligible_source_index",
            "member_direct_full_frame.visual_selection.boundaries[].outgoing_eligible_source_index",
            "member_direct_full_frame.visual_selection.boundaries[].incoming_eligible_source_index",
            "member_direct_full_frame.editorial_motion.segments[].source_index",
            "render.automatic_full_frame_fallbacks[].eligible_source_index",
            "source_look_match.target_source_index",
            "render.source_look_match.target_source_index",
            (
                "rendered_candidate_selection.visual_candidate_set"
                ".candidates[].source_sequence[]"
            ),
            (
                "rendered_candidate_selection.complete_candidates[]"
                ".visual_timeline.choices[].source_index"
            ),
        ):
            self.assertEqual(domains[payload_path], ELIGIBLE_SOURCE_INDEX_DOMAIN)
        for nonexistent_path in (
            "media[].position",
            "alignment.alignments[].position",
            "group_draft.audio_offsets[].position",
            "member_draft.audio_offsets[].position",
        ):
            self.assertNotIn(nonexistent_path, domains)

    def test_registered_post_render_enforcement_passes_trusted_activation_authority(
        self,
    ) -> None:
        reporter = _MemoryReporter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "IVE-fixture-20260824-120000-analysis.json"
            manifest: dict[str, object] = {
                "request": {
                    "group": "IVE",
                    "edit_mode": "group",
                    "source_mode": "automatic",
                },
                "artifact_role": "pending_qa",
                "app_result_exposure": {
                    "contract": "automatic_ive_group_app_result_exposure_v1",
                    "video_output_exposed": False,
                    "delivery_completion_message_allowed": False,
                },
                "render": {"artifact_role": "pending_qa"},
                "stage_edit_generation": {},
            }
            _persist_post_render_artifact_shell(manifest_path, manifest)
            qa_path = PostRenderArtifactSet.from_analysis_path(
                manifest_path
            ).default_report_path
            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa",
                side_effect=lambda artifact_set, **_kwargs: (
                    _persist_mock_bound_qa_report(
                        artifact_set,
                        verdict="PASS",
                    )
                ),
            ) as evaluate:
                returned = _enforce_automatic_stage_edit_post_render_qa(
                    registered_stage_edit_bundle=object(),  # type: ignore[arg-type]
                    manifest=manifest,  # type: ignore[arg-type]
                    manifest_path=manifest_path,
                    reporter=reporter,  # type: ignore[arg-type]
                )

        self.assertEqual(returned, qa_path.resolve())
        self.assertTrue(
            evaluate.call_args.kwargs["trusted_registered_stage_edit"]
        )
        self.assertEqual(manifest["artifact_role"], "delivery")
        self.assertTrue(
            manifest["app_result_exposure"]["video_output_exposed"]
        )
        self.assertEqual(manifest["render"]["artifact_role"], "delivery")

    def test_explicit_automatic_ive_group_also_runs_mandatory_post_render_qa(
        self,
    ) -> None:
        reporter = _MemoryReporter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "IVE-fixture-20260824-120000-analysis.json"
            manifest: dict[str, object] = {
                "request": {
                    "group": "IVE",
                    "edit_mode": "group",
                    "source_mode": "automatic",
                },
                "artifact_role": "pending_qa",
                "app_result_exposure": {
                    "contract": "automatic_ive_group_app_result_exposure_v1",
                    "video_output_exposed": False,
                    "delivery_completion_message_allowed": False,
                },
                "render": {"artifact_role": "pending_qa"},
                "stage_edit_generation": {},
            }
            _persist_post_render_artifact_shell(manifest_path, manifest)
            qa_path = PostRenderArtifactSet.from_analysis_path(
                manifest_path
            ).default_report_path
            montage_config = RenderedMontageGrammarConfig()
            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa",
                side_effect=lambda artifact_set, **_kwargs: (
                    _persist_mock_bound_qa_report(
                        artifact_set,
                        verdict="PASS",
                    )
                ),
            ) as evaluate:
                returned = _enforce_automatic_stage_edit_post_render_qa(
                    registered_stage_edit_bundle=None,
                    manifest=manifest,  # type: ignore[arg-type]
                    manifest_path=manifest_path,
                    reporter=reporter,  # type: ignore[arg-type]
                    automatic_ive_group_delivery=True,
                    rendered_montage_config=montage_config,
                )

        self.assertEqual(returned, qa_path.resolve())
        self.assertIsNone(
            evaluate.call_args.kwargs["trusted_registered_stage_edit"]
        )
        self.assertIs(
            evaluate.call_args.kwargs["config"].rendered_montage_grammar,
            montage_config,
        )
        self.assertEqual(
            manifest["post_render_qa"]["activation"],
            "automatic_request_scoped_ive_group_delivery",
        )
        self.assertTrue(
            manifest["post_render_qa"][
                "ive_product_floor_required_independent_of_activation"
            ]
        )
        self.assertEqual(manifest["artifact_role"], "delivery")
        self.assertTrue(
            manifest["app_result_exposure"][
                "delivery_completion_message_allowed"
            ]
        )

    def test_automatic_ive_delivery_blocks_conditional_post_render_report(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "IVE-fixture-20260824-120000-analysis.json"
            manifest: dict[str, object] = {
                "request": {
                    "group": "IVE",
                    "edit_mode": "group",
                    "source_mode": "automatic",
                },
                "artifact_role": "pending_qa",
                "app_result_exposure": {
                    "contract": "automatic_ive_group_app_result_exposure_v1",
                    "video_output_exposed": False,
                    "delivery_completion_message_allowed": False,
                },
                "render": {"artifact_role": "pending_qa"},
            }
            _persist_post_render_artifact_shell(manifest_path, manifest)
            qa_path = PostRenderArtifactSet.from_analysis_path(
                manifest_path
            ).default_report_path
            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa",
                side_effect=lambda artifact_set, **_kwargs: (
                    _persist_mock_bound_qa_report(
                        artifact_set,
                        verdict="CONDITIONAL",
                    )
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "blocked by mandatory post-render QA verdict CONDITIONAL",
                ):
                    _enforce_automatic_stage_edit_post_render_qa(
                        registered_stage_edit_bundle=None,
                        manifest=manifest,  # type: ignore[arg-type]
                        manifest_path=manifest_path,
                        reporter=_MemoryReporter(),  # type: ignore[arg-type]
                        automatic_ive_group_delivery=True,
                    )

        self.assertFalse(manifest["post_render_qa"]["passed"])
        self.assertEqual(manifest["artifact_role"], "rejected")
        self.assertFalse(
            manifest["app_result_exposure"]["video_output_exposed"]
        )
        self.assertEqual(manifest["render"]["artifact_role"], "rejected")

    def test_mandatory_qa_refuses_preexposed_or_nonpending_product_media(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "IVE-fixture-20260824-120000-analysis.json"
            manifest: dict[str, object] = {
                "request": {
                    "group": "IVE",
                    "edit_mode": "group",
                    "source_mode": "automatic",
                },
                "artifact_role": "delivery",
                "app_result_exposure": {
                    "contract": "automatic_ive_group_app_result_exposure_v1",
                    "video_output_exposed": True,
                    "delivery_completion_message_allowed": True,
                },
                "render": {"artifact_role": "delivery"},
            }
            manifest_path.write_text("{}", encoding="utf-8")
            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa"
            ) as evaluate:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "requires an unexposed pending_qa artifact",
                ):
                    _enforce_automatic_stage_edit_post_render_qa(
                        registered_stage_edit_bundle=None,
                        manifest=manifest,  # type: ignore[arg-type]
                        manifest_path=manifest_path,
                        reporter=_MemoryReporter(),  # type: ignore[arg-type]
                        automatic_ive_group_delivery=True,
                    )
            evaluate.assert_not_called()

    def test_index_domain_helpers_reject_ambiguous_mappings(self) -> None:
        sources = (Path("/tmp/a.mp4"), Path("/tmp/b.mp4"))
        with self.assertRaises(ValueError):
            _source_index_mapping(sources, (0, 0))
        with self.assertRaises(ValueError):
            _source_index_mapping(sources, (0, 2))
        with self.assertRaises(ValueError):
            _with_index_domain({}, "unknown")

        original = {"segments": []}
        annotated = _with_index_domain(original, ELIGIBLE_SOURCE_INDEX_DOMAIN)
        self.assertNotIn("index_domain", original)
        self.assertEqual(annotated["index_domain"], ELIGIBLE_SOURCE_INDEX_DOMAIN)

    def test_alignment_eligibility_uses_request_index_domain(self) -> None:
        alignment = AlignmentResult(
            reference="a",
            sample_rate=4_000,
            alignments=(
                SourceAlignment("a", True, 0, 0.0, 1.0, 1, 1.0),
                SourceAlignment("b", False, 0, 0.0, 0.2, 1, 1.0),
            ),
        )

        payload = _alignment_eligibility(alignment)

        self.assertEqual([item["source_index"] for item in payload], [0, 1])
        self.assertTrue(
            all(
                item["index_domain"] == REQUEST_SOURCE_INDEX_DOMAIN
                for item in payload
            )
        )
        self.assertTrue(payload[0]["render_eligible"])
        self.assertFalse(payload[1]["render_eligible"])

    def test_member_fancam_alignment_allows_lower_mix_correlation_with_margin(self) -> None:
        alignment = AlignmentResult(
            reference="reference",
            sample_rate=4_000,
            alignments=(
                SourceAlignment(
                    "reference", True, 0, 0.0, 1.0, 800_000, 200.0,
                    backend="reference",
                ),
                SourceAlignment(
                    "fancam", False, 12_000, 3.0, 0.33, 780_000, 195.0,
                    second_best_score=0.25,
                    score_margin=0.08,
                    overlap_ratio=0.95,
                    backend="spectral_flux_ncc",
                ),
            ),
        )

        group = _alignment_eligibility(alignment, edit_mode=EditMode.GROUP)
        member = _alignment_eligibility(alignment, edit_mode=EditMode.MEMBER)

        self.assertFalse(group[1]["render_eligible"])
        self.assertTrue(member[1]["render_eligible"])

    def test_member_fancam_alignment_rejects_unsealed_peer_claim(self) -> None:
        alignment = AlignmentResult(
            reference="reference",
            sample_rate=4_000,
            alignments=(
                SourceAlignment(
                    "reference", True, 0, 0.0, 1.0, 876_000, 219.0,
                    backend="reference",
                ),
                SourceAlignment(
                    "mbc-gaeul", False, 7_218, 1.8045977011494252,
                    0.2977856872655474, 875_724, 218.93103448275863,
                    second_best_score=0.19159423832693445,
                    score_margin=0.10619144893861293,
                    overlap_ratio=1.0,
                    backend="spectral_flux_ncc",
                ),
            ),
        )

        peer_consensus = {
            "contract": (
                "automatic_member_borderline_multi_peer_"
                "alignment_consensus_v1"
            ),
            "sources": [
                {
                    "source_index": 1,
                    "authorized": True,
                    "direct_alignment": {
                        "offset_seconds": 1.8045977011494252,
                        "score": 0.2977856872655474,
                        "score_margin": 0.10619144893861293,
                        "overlap_ratio": 1.0,
                        "backend": "spectral_flux_ncc",
                    },
                }
            ],
        }
        group = _alignment_eligibility(alignment, edit_mode=EditMode.GROUP)
        member_without_peers = _alignment_eligibility(
            alignment,
            edit_mode=EditMode.MEMBER,
        )
        member = _alignment_eligibility(
            alignment,
            edit_mode=EditMode.MEMBER,
            member_peer_consensus=peer_consensus,
        )

        self.assertFalse(group[1]["render_eligible"])
        self.assertFalse(member_without_peers[1]["render_eligible"])
        self.assertFalse(member[1]["render_eligible"])
        self.assertNotEqual(member[1]["reasons"], [])
        self.assertEqual(
            member[1]["policy"],
            "member_fancam_multi_peer_consensus_v3",
        )
        self.assertFalse(
            member[1]["measured"]["peer_consensus_acceptance_applied"]
        )
        self.assertEqual(
            member[1]["thresholds"][
                "peer_consensus_candidate_score_floor"
            ],
            0.18,
        )

    def test_member_fancam_near_floor_still_rejects_ambiguous_peak(self) -> None:
        alignment = AlignmentResult(
            reference="reference",
            sample_rate=4_000,
            alignments=(
                SourceAlignment(
                    "reference", True, 0, 0.0, 1.0, 800_000, 200.0,
                    backend="reference",
                ),
                SourceAlignment(
                    "ambiguous-fancam", False, 7_000, 1.75, 0.295,
                    780_000, 195.0,
                    second_best_score=0.215,
                    score_margin=0.08,
                    overlap_ratio=1.0,
                    backend="spectral_flux_ncc",
                ),
            ),
        )

        member = _alignment_eligibility(alignment, edit_mode=EditMode.MEMBER)

        self.assertFalse(member[1]["render_eligible"])
        self.assertFalse(
            member[1]["measured"]["peer_consensus_acceptance_applied"]
        )

    def test_member_peer_candidate_covers_margin_only_baseline_failure(self) -> None:
        alignment = AlignmentResult(
            reference="reference",
            sample_rate=4_000,
            alignments=(
                SourceAlignment(
                    "reference", True, 0, 0.0, 1.0, 800_000, 200.0,
                    backend="reference",
                ),
                SourceAlignment(
                    "low-margin-fancam", False, 7_000, 1.75, 0.40,
                    780_000, 195.0, second_best_score=0.37,
                    score_margin=0.03, overlap_ratio=1.0,
                    backend="spectral_flux_ncc",
                ),
            ),
        )

        member = _alignment_eligibility(alignment, edit_mode=EditMode.MEMBER)

        self.assertFalse(member[1]["render_eligible"])
        self.assertTrue(member[1]["measured"]["peer_consensus_candidate"])

    def test_member_borderline_alignment_uses_two_independent_peer_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            sources = tuple(root / f"source-{index}.mp4" for index in range(4))
            selected = []
            for index, source in enumerate(sources):
                content = f"verified member source {index}\n".encode()
                source.write_bytes(content)
                selected.append(
                    {
                        "source_index": index,
                        "video_id": f"video{index:06d}",
                        "title": (
                            "IVE GAEUL fancam Music Bank "
                            f"2026-03-{index + 1:02d}"
                        ),
                        "local_path": str(source),
                        "broadcast_date": f"2026-03-{index + 1:02d}",
                        "broadcast_program": "music_bank",
                        "channel_id": f"channel-{index}",
                        "media_validation": {"passed": True},
                        "aspect_validation": {"passed": True},
                        "content_integrity": {
                            "contract_version": 1,
                            "algorithm": "sha256",
                            "size_bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(),
                        },
                    }
                )
            manifest = {
                "schema_version": 1,
                "selection_policy_version": 4,
                "content_integrity_contract_version": 1,
                "automatic_media_quality_contract_version": 1,
                "member_portrait_aspect_contract_version": 2,
                "group": "IVE",
                "song": "BLACKHOLE",
                "edit_mode": "member",
                "target_member": "가을",
                "selected": selected,
                "summary": {"status": "complete"},
            }
            manifest_path = cache / "acquisition-manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            acquisition = _build_acquisition_artifact(
                source_paths=sources,
                manifest_path=manifest_path,
                manifest=manifest,
            )
            request = AnalysisRequest(
                song="BLACKHOLE",
                edit_mode=EditMode.MEMBER,
                aspect_ratio=AspectRatio.PORTRAIT,
                sources=sources,
                output_directory=root / "output",
                target_member=IVEMember.GAEUL,
                source_mode=SourceMode.AUTOMATIC,
                source_cache_directory=cache,
            )
            alignment = AlignmentResult(
                reference=str(sources[0]),
                sample_rate=4_000,
                alignments=(
                    SourceAlignment(
                        str(sources[0]), True, 0, 0.0, 1.0, 880_000, 220.0,
                        backend="reference",
                    ),
                    SourceAlignment(
                        str(sources[1]), False, 8_000, 2.0, 0.42,
                        876_000, 219.0, second_best_score=0.20,
                        score_margin=0.22, overlap_ratio=1.0,
                        backend="spectral_flux_ncc",
                    ),
                    SourceAlignment(
                        str(sources[2]), False, 4_000, 1.0, 0.40,
                        876_000, 219.0, second_best_score=0.19,
                        score_margin=0.21, overlap_ratio=1.0,
                        backend="spectral_flux_ncc",
                    ),
                    SourceAlignment(
                        str(sources[3]), False, 7_200, 1.8, 0.2978,
                        876_000, 219.0, second_best_score=0.19,
                        score_margin=0.1078, overlap_ratio=1.0,
                        backend="spectral_flux_ncc",
                    ),
                ),
            )
            estimates = iter(
                (
                    OffsetEstimate(
                        -800, -0.2, 0.31, 870_000, 217.5, 4_000,
                        0.19, 0.12, 0.99, "spectral_flux_ncc",
                    ),
                    OffsetEstimate(
                        3_200, 0.8, 0.35, 870_000, 217.5, 4_000,
                        0.20, 0.15, 0.99, "spectral_flux_ncc",
                    ),
                )
            )

            with patch(
                "crosscut.pipeline.extract_mono_pcm",
                side_effect=lambda source, sample_rate: [
                    float(sources.index(Path(source)) + 1)
                ]
                * 8_000,
            ), patch(
                "crosscut.pipeline.spectral_flux_alignment_offset",
                side_effect=lambda *args, **kwargs: next(estimates),
            ):
                evidence = _verify_automatic_member_borderline_alignments(
                    request,
                    alignment,
                    reporter=_MemoryReporter(),  # type: ignore[arg-type]
                    acquisition_identity_authority=(
                        acquisition.member_identity_authority
                    ),
                )
                original_manifest_bytes = manifest_path.read_bytes()
                same_slot_manifest = deepcopy(manifest)
                same_slot_manifest["selected"][2]["broadcast_date"] = (
                    same_slot_manifest["selected"][1]["broadcast_date"]
                )
                same_slot_manifest["selected"][2]["title"] = (
                    "IVE GAEUL fancam Music Bank "
                    + same_slot_manifest["selected"][1]["broadcast_date"]
                )
                same_slot_manifest["selected"][2]["broadcast_program"] = (
                    same_slot_manifest["selected"][1]["broadcast_program"]
                )
                manifest_path.write_text(
                    json.dumps(
                        same_slot_manifest,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                same_slot_acquisition = _build_acquisition_artifact(
                    source_paths=sources,
                    manifest_path=manifest_path,
                    manifest=same_slot_manifest,
                )
                estimates = iter(
                    (
                        OffsetEstimate(
                            -800, -0.2, 0.31, 870_000, 217.5, 4_000,
                            0.19, 0.12, 0.99, "spectral_flux_ncc",
                        ),
                    )
                )
                same_slot_evidence = (
                    _verify_automatic_member_borderline_alignments(
                        request,
                        alignment,
                        reporter=_MemoryReporter(),  # type: ignore[arg-type]
                        acquisition_identity_authority=(
                            same_slot_acquisition.member_identity_authority
                        ),
                    )
                )
                manifest_path.write_bytes(original_manifest_bytes)
                manifest["selected"][1]["broadcast_date"] = "2026-04-01"
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                manifest_tamper = (
                    _verify_automatic_member_borderline_alignments(
                        request,
                        alignment,
                        reporter=_MemoryReporter(),  # type: ignore[arg-type]
                        acquisition_identity_authority=(
                            acquisition.member_identity_authority
                        ),
                    )
                )
                manifest_path.write_bytes(original_manifest_bytes)
                sources[1].write_bytes(b"mutated after acquisition authority\n")
                source_tamper = (
                    _verify_automatic_member_borderline_alignments(
                        request,
                        alignment,
                        reporter=_MemoryReporter(),  # type: ignore[arg-type]
                        acquisition_identity_authority=(
                            acquisition.member_identity_authority
                        ),
                    )
                )

        self.assertEqual(evidence["attempted_source_count"], 1)
        self.assertEqual(evidence["authorized_source_count"], 1)
        self.assertEqual(evidence["sources"][0]["source_index"], 3)
        self.assertTrue(evidence["sources"][0]["authorized"])
        self.assertAlmostEqual(
            evidence["sources"][0]["consensus_offset_seconds"],
            1.8,
        )
        eligible = _alignment_eligibility(
            alignment,
            edit_mode=EditMode.MEMBER,
            member_peer_consensus=evidence,
        )
        self.assertTrue(all(item["render_eligible"] for item in eligible))
        self.assertEqual(manifest_tamper["authorized_source_count"], 0)
        self.assertFalse(
            manifest_tamper["broadcast_identity_verification"]["verified"]
        )
        self.assertEqual(source_tamper["authorized_source_count"], 0)
        self.assertFalse(
            source_tamper["broadcast_identity_verification"]["verified"]
        )
        self.assertEqual(same_slot_evidence["authorized_source_count"], 0)
        self.assertEqual(
            same_slot_evidence["sources"][0]["reason"],
            "fewer_than_two_reliable_peer_predictions",
        )
        changed_alignments = list(alignment.alignments)
        changed_alignments[3] = replace(
            changed_alignments[3],
            score=changed_alignments[3].score + 1e-6,
        )
        stale_replay = _alignment_eligibility(
            replace(alignment, alignments=tuple(changed_alignments)),
            edit_mode=EditMode.MEMBER,
            member_peer_consensus=evidence,
        )
        self.assertFalse(stale_replay[3]["render_eligible"])
        self.assertFalse(
            stale_replay[3]["measured"][
                "peer_consensus_acceptance_applied"
            ]
        )

    def test_common_range_respects_positive_and_negative_offsets(self) -> None:
        media = [_media("a", 10.0), _media("b", 12.0), _media("c", 9.0)]
        alignment = AlignmentResult(
            reference="a",
            sample_rate=4_000,
            alignments=(
                SourceAlignment("a", True, 0, 0.0, 1.0, 1, 1.0),
                SourceAlignment("b", False, 0, 2.0, 0.9, 1, 1.0),
                SourceAlignment("c", False, 0, -1.0, 0.9, 1, 1.0),
            ),
        )
        self.assertEqual(_common_master_range(media, alignment), (1.0, 10.0))

    def test_slug_keeps_korean_song_names(self) -> None:
        self.assertEqual(_safe_slug(" 해야 (HEYA) "), "해야-HEYA")

    def test_member_render_selection_basis_includes_global_path_policy(self) -> None:
        policy = (
            "safe_gate_then_local_best_quality_guard_then_"
            "maximum_distinct_balanced_global_path_v2"
        )

        self.assertEqual(
            _member_render_selection_basis(policy),
            "anchored_target_tracking_virtual_camera_" + policy,
        )
        with self.assertRaises(ValueError):
            _member_render_selection_basis("  ")

    def test_capabilities_describe_anchored_tracking_without_identity_claim(self) -> None:
        payload = _capabilities_payload(
            group_ready=True,
            edit_mode=EditMode.MEMBER,
            member_draft=None,
            rendered_video_path=None,
            used_automatic_member_anchors=False,
        )

        self.assertFalse(payload["member_tracking_ready"])
        self.assertIn("앵커", payload["member_tracking_note"])
        self.assertNotIn("member_identity_ready", payload)
        self.assertNotIn("member_identity_note", payload)

    def test_direct_full_frame_coarsens_to_reference_cadence_and_timewarps(self) -> None:
        timing = _rotating_timing_draft()
        timewarps = tuple(
            _linear_timewarp(duration=40.0, rate=1.004) for _ in range(10)
        )
        scene_cut_times = tuple(
            (1.0 + source_index * 0.1,)
            for source_index in range(10)
        )

        selection = _VisualSelectionStub(
            (*tuple(range(10)), 0, 1),
            10,
        )
        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=_bound_visual_selection(selection),
        ) as select_visual:
            segments, transitions, details, cadence, visual_selection = (
                _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing,
                    eligible_indices=tuple(range(10)),
                    eligible_sources=tuple(
                        Path(f"official-gaeul-{index}.mp4")
                        for index in range(10)
                    ),
                    common_master_start=0.0,
                    source_timewarps=timewarps,
                    scene_cut_times=scene_cut_times,
                    descriptor_provider=lambda *_args: None,
                    eligible_source_durations_seconds=(120.0,) * 10,
                    source_metadata_contract=(
                        OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                    ),
                    member_pose_dissolve_enabled=False,
                )
            )
        self.assertEqual(
            select_visual.call_args.kwargs["source_timewarps"],
            timewarps,
        )
        self.assertEqual(
            select_visual.call_args.kwargs["source_scene_cut_times"],
            scene_cut_times,
        )
        self.assertTrue(
            select_visual.call_args.kwargs[
                "continuous_official_portrait_fancam_contract"
            ]
        )
        self.assertEqual(
            len(select_visual.call_args.kwargs["canonical_spans"]),
            12,
        )

        self.assertEqual(len(segments), 12)
        self.assertEqual(
            [item.source_index for item in segments],
            [*range(10), 0, 1],
        )
        self.assertEqual(
            [item.duration for item in segments],
            [2.0, 2.0, 2.0, 2.0, *([4.0] * 8)],
        )
        self.assertTrue(all(item.crop is None for item in segments))
        self.assertTrue(all(item.pan_crop is None for item in segments))
        self.assertTrue(all(item.source_duration is not None for item in segments))
        self.assertAlmostEqual(segments[0].source_start, 0.0)
        self.assertAlmostEqual(segments[0].source_end, 2.008)
        self.assertEqual(len(transitions), 11)
        self.assertTrue(
            all(
                item.kind is TransitionKind.HARD_CUT
                and item.reason
                == OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY
                for item in transitions
            )
        )
        self.assertEqual(details[0]["timing_segment_count"], 2)
        self.assertEqual(details[0]["source_clock_basis"], "local_monotonic_timewarp")
        self.assertAlmostEqual(details[0]["source_local_end"], 2.008)
        self.assertTrue(cadence["passed"])
        self.assertAlmostEqual(cadence["source_switches_per_minute"], 16.5)
        self.assertAlmostEqual(cadence["source_run_median_seconds"], 4.0)
        self.assertAlmostEqual(cadence["longest_source_run_seconds"], 4.0)
        self.assertFalse(cadence["member_pose_dissolve_requested"])
        self.assertFalse(cadence["member_pose_dissolve_effective"])
        self.assertEqual(cadence["member_pose_dissolve_count"], 0)
        self.assertEqual(
            cadence["pose_matched_dissolves"]["rejection_reason_counts"][
                "effect_disabled_by_request"
            ],
            11,
        )
        self.assertEqual(
            visual_selection.source_counts,
            (2, 2, 1, 1, 1, 1, 1, 1, 1, 1),
        )

    def test_direct_full_frame_reference_length_avoids_the_200_cut_path(self) -> None:
        duration = 196.4835616941711
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=duration / 200.0,
        )

        selection = _VisualSelectionStub(
            tuple(index % 10 for index in range(51)),
            10,
        )
        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=_bound_visual_selection(selection),
        ):
            segments, _transitions, _details, cadence, visual_selection = (
                _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing,
                    eligible_indices=tuple(range(10)),
                    eligible_sources=tuple(
                        Path(f"official-gaeul-{index}.mp4")
                        for index in range(10)
                    ),
                    common_master_start=0.0,
                    source_timewarps=None,
                    scene_cut_times=((),) * 10,
                    descriptor_provider=lambda *_args: None,
                )
            )

        self.assertEqual(len(segments), 51)
        self.assertEqual(
            [sum(item.source_index == source for item in segments) for source in range(10)],
            [6, *([5] * 9)],
        )
        self.assertAlmostEqual(
            cadence["source_switches_per_minute"],
            50 * 60.0 / duration,
            places=3,
        )
        self.assertAlmostEqual(
            cadence["source_run_median_seconds"],
            duration / 50.0,
        )
        self.assertLessEqual(cadence["longest_source_run_seconds"], 7.2)
        self.assertEqual(visual_selection.source_counts, (6, *([5] * 9)))

    def test_direct_full_frame_fifty_spans_supports_eight_balanced_sources(
        self,
    ) -> None:
        timing = _rotating_timing_draft(
            source_count=8,
            duration_seconds=200.0,
            unit_seconds=1.0,
        )
        path = tuple(range(8)) * 6 + (0, 1, 2)
        selection = _VisualSelectionStub(path, 8)

        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=_bound_visual_selection(selection),
        ):
            segments, _transitions, _details, cadence, visual_selection = (
                _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing,
                    eligible_indices=tuple(range(8)),
                    eligible_sources=tuple(
                        Path(f"official-gaeul-{index}.mp4")
                        for index in range(8)
                    ),
                    common_master_start=0.0,
                    source_timewarps=None,
                    scene_cut_times=((),) * 8,
                    descriptor_provider=lambda *_args: None,
                )
            )

        self.assertEqual(len(segments), 51)
        self.assertEqual(
            visual_selection.source_counts,
            (7, 7, 7, 6, 6, 6, 6, 6),
        )
        self.assertEqual(len({segment.source_index for segment in segments}), 8)
        self.assertTrue(cadence["passed"])
        self.assertLessEqual(
            cadence["source_duration_balance"]["maximum_relative_deviation"],
            0.08,
        )

    def test_direct_full_frame_four_sources_preserve_fifty_span_cut_cadence(
        self,
    ) -> None:
        duration = 200.0
        timing = _rotating_timing_draft(
            source_count=4,
            duration_seconds=duration,
            unit_seconds=1.0,
        )
        path = tuple(range(4)) * 12 + (0, 1, 2)
        selection = _VisualSelectionStub(path, 4)

        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=_bound_visual_selection(selection),
        ):
            segments, _transitions, _details, cadence, visual_selection = (
                _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing,
                    eligible_indices=tuple(range(4)),
                    eligible_sources=tuple(
                        Path(f"official-gaeul-{index}.mp4")
                        for index in range(4)
                    ),
                    common_master_start=0.0,
                    source_timewarps=None,
                    scene_cut_times=((),) * 4,
                    descriptor_provider=lambda *_args: None,
                )
            )

        self.assertEqual(len(segments), 51)
        self.assertEqual(cadence["source_switch_count"], 50)
        self.assertAlmostEqual(
            cadence["source_switches_per_minute"],
            50 * 60.0 / duration,
        )
        self.assertTrue(cadence["passed"])
        self.assertEqual(
            visual_selection.source_counts,
            (13, 13, 13, 12),
        )
        self.assertTrue(
            all(
                left.source_index != right.source_index
                for left, right in zip(segments, segments[1:])
            )
        )

    def test_direct_partition_fallback_moves_proven_final_boundary_to_five_three(
        self,
    ) -> None:
        duration = 196.4835616941711
        unit_seconds = duration / 200.0
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=unit_seconds,
        )
        base_profile = _professional_direct_partition_profile(timing)
        selection = _VisualSelectionStub(
            tuple(index % 10 for index in range(len(base_profile.spans))),
            10,
        )
        base_failure = {
            "failure": "validity_quota_path_exhausted",
            "constraint_infeasibility_proven": True,
            "failed_at_segment_index": 49,
            "hard_boundary_connectivity": {
                "zero_local_nonsevere_transition_boundary_indices": [48],
            },
            "search": {
                "exact_lifeline": {
                    "path_found": False,
                    "state_capacity_exhausted": False,
                    "root_candidate_diagnostics": [
                        {
                            "eligible_source_index": source_index,
                            "search_outcome": "root_predecessor_exhausted",
                        }
                        for source_index in (3, 8, 9)
                    ],
                }
            },
        }
        calls: list[tuple[int, ...]] = []

        def choose_profile(**kwargs: object) -> _VisualSelectionStub:
            spans = kwargs["canonical_spans"]
            assert isinstance(spans, tuple)
            unit_counts = tuple(
                int(round((end - start) / unit_seconds))
                for start, end in spans
            )
            calls.append(unit_counts)
            if unit_counts[-2:] == (5, 3):
                return selection.bind(spans, kwargs["config"])
            raise DirectMemberVisualSelectionError(
                "synthetic hard-infeasible partition",
                base_failure,
            )

        shared_provider = lambda *_args: None
        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=choose_profile,
        ) as select_visual:
            segments, _transitions, details, cadence, visual_selection = (
                _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing,
                    eligible_indices=tuple(range(10)),
                    eligible_sources=tuple(
                        Path(f"official-gaeul-{index}.mp4")
                        for index in range(10)
                    ),
                    common_master_start=0.0,
                    source_timewarps=None,
                    scene_cut_times=((),) * 10,
                    descriptor_provider=shared_provider,
                )
            )

        self.assertEqual(len(calls), 5)
        self.assertEqual(calls[0], base_profile.unit_counts)
        self.assertEqual(calls[1][-2:], (5, 3))
        self.assertEqual(calls[2][-2:], (3, 5))
        self.assertEqual([item["timing_segment_count"] for item in details[-2:]], [5, 3])
        self.assertEqual(len(segments), len(base_profile.spans))
        self.assertEqual(visual_selection.source_counts, (6, *([5] * 9)))
        self.assertFalse(
            any(
                left.source_index == right.source_index
                for left, right in zip(segments, segments[1:])
            )
        )
        self.assertTrue(cadence["passed"])
        self.assertLessEqual(
            cadence["source_duration_balance"]["maximum_relative_deviation"],
            0.08,
        )
        partition = cadence["partition_search"]
        self.assertEqual(
            partition["selected_profile_id"],
            "boundary-49-5-plus-3",
        )
        self.assertTrue(partition["fallback_used"])
        self.assertFalse(partition["visual_gates_relaxed"])
        self.assertEqual(len(partition["profiles_attempted"]), 5)
        self.assertTrue(
            all(
                call.kwargs["descriptor_provider"] is shared_provider
                for call in select_visual.call_args_list
            )
        )
        frozen_config = select_visual.call_args_list[0].kwargs["config"]
        self.assertTrue(
            all(
                call.kwargs["config"] == frozen_config
                for call in select_visual.call_args_list
            )
        )

    def test_official_portrait_partition_repairs_one_proven_forward_frontier(
        self,
    ) -> None:
        duration = 196.4835616941711
        unit_seconds = duration / 200.0
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=unit_seconds,
        )
        base_profile = _professional_direct_partition_profile(timing)
        selection = _VisualSelectionStub(
            tuple(index % 10 for index in range(len(base_profile.spans))),
            10,
        )
        base_failure = _direct_partition_failure_fixture(
            span_count=len(base_profile.spans),
            failed_segment_index=4,
            frontier_boundary_index=3,
            zero_local_boundary_indices=(3,),
        )
        primary_failure = _direct_partition_failure_fixture(
            span_count=len(base_profile.spans),
            failed_segment_index=12,
            frontier_boundary_index=11,
        )
        calls: list[tuple[int, ...]] = []
        progress_events: list[dict[str, object]] = []

        def choose_profile(**kwargs: object) -> _VisualSelectionStub:
            spans = kwargs["canonical_spans"]
            assert isinstance(spans, tuple)
            unit_counts = tuple(
                int(round((end - start) / unit_seconds))
                for start, end in spans
            )
            calls.append(unit_counts)
            if unit_counts == base_profile.unit_counts:
                raise DirectMemberVisualSelectionError(
                    "synthetic local hard bottleneck",
                    base_failure,
                )
            if unit_counts[3:5] not in ((5, 3), (3, 5)):
                self.fail("primary repair did not stay at the proven boundary")
            if (
                unit_counts[3:5] == (5, 3)
                and unit_counts[11:13] == (5, 3)
            ):
                return selection.bind(spans, kwargs["config"])
            raise DirectMemberVisualSelectionError(
                "synthetic global forward-frontier bottleneck",
                primary_failure,
            )

        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=choose_profile,
        ) as select_visual:
            segments, _transitions, details, cadence, visual_selection = (
                _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing,
                    eligible_indices=tuple(range(10)),
                    eligible_sources=tuple(
                        Path(f"official-gaeul-{index}.mp4")
                        for index in range(10)
                    ),
                    common_master_start=0.0,
                    source_timewarps=None,
                    scene_cut_times=((),) * 10,
                    source_metadata_contract=(
                        OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                    ),
                    descriptor_provider=lambda *_args: None,
                    partition_progress_callback=progress_events.append,
                )
            )

        self.assertEqual(select_visual.call_count, 7)
        self.assertEqual(calls[0], base_profile.unit_counts)
        self.assertEqual(
            {calls[1][3:5], calls[3][3:5]},
            {(5, 3), (3, 5)},
        )
        self.assertEqual(
            {
                (counts[3:5], counts[11:13])
                for counts in calls[2:]
                if counts[11:13] != (4, 4)
            },
            {
                ((5, 3), (5, 3)),
                ((5, 3), (3, 5)),
                ((3, 5), (5, 3)),
                ((3, 5), (3, 5)),
            },
        )
        self.assertEqual(
            [details[index]["timing_segment_count"] for index in (3, 4, 11, 12)],
            [5, 3, 5, 3],
        )
        self.assertEqual(len(segments), len(base_profile.spans))
        self.assertEqual(visual_selection.source_counts, (6, *([5] * 9)))
        self.assertTrue(cadence["passed"])
        self.assertLessEqual(cadence["longest_source_run_seconds"], 7.2)
        partition = cadence["partition_search"]
        self.assertEqual(
            partition["selected_profile_id"],
            "boundary-3-5-plus-3__frontier-11-5-plus-3",
        )
        selected_profile = next(
            item
            for item in partition["profiles_attempted"]
            if item["status"] == "feasible"
        )
        self.assertEqual(selected_profile["changed_span_count"], 4)
        self.assertEqual(selected_profile["shifted_boundary_count"], 2)
        self.assertEqual(selected_profile["secondary_shifted_boundary_index"], 11)
        self.assertEqual(selected_profile["secondary_boundary_shift_units"], 1)
        self.assertEqual(
            selected_profile["secondary_shift_trigger"],
            "proven_forward_connectivity_frontier_exhaustion",
        )
        self.assertFalse(partition["visual_gates_relaxed"])
        self.assertFalse(partition["profile_cap_reached"])
        self.assertEqual(partition["generated_profile_count"], 11)
        shifts = selected_profile["proof_directed_boundary_shifts"]
        self.assertEqual([item["boundary_index"] for item in shifts], [3, 11])
        self.assertEqual(shifts[1]["parent_profile_id"], "boundary-3-5-plus-3")
        parent_profile = next(
            item
            for item in partition["profiles_attempted"]
            if item["profile_id"] == shifts[1]["parent_profile_id"]
        )
        self.assertEqual(
            shifts[1]["parent_partition_signature_sha256"],
            parent_profile["partition_signature_sha256"],
        )
        self.assertEqual(shifts[1]["proof"]["failure"], "validity_quota_path_exhausted")
        self.assertEqual(shifts[1]["proof"]["failed_segment_index"], 12)
        self.assertEqual(
            shifts[1]["proof"]["local_nonsevere_cross_source_edge_count"],
            2,
        )
        self.assertTrue(
            all(
                call.kwargs["continuous_official_portrait_fancam_contract"]
                for call in select_visual.call_args_list
            )
        )
        profile_events = [
            item
            for item in progress_events
            if item["event"] == "direct_member_partition_profile_evaluation"
        ]
        self.assertEqual(len(profile_events), 14)
        self.assertEqual(
            [
                item["evaluation_index"]
                for item in profile_events
                if item["phase"] == "started"
            ],
            list(range(1, 8)),
        )
        self.assertTrue(
            all(item["evaluation_limit"] == 64 for item in profile_events)
        )
        final_progress = progress_events[-1]
        self.assertEqual(
            final_progress["event"],
            "direct_member_partition_search_completed",
        )
        self.assertEqual(final_progress["result"], "feasible")
        self.assertEqual(final_progress["evaluated_profile_count"], 7)
        self.assertEqual(final_progress["generated_profile_count"], 11)
        self.assertFalse(final_progress["profile_cap_reached"])

    def test_official_portrait_partition_chains_three_independent_proofs(
        self,
    ) -> None:
        duration = 196.4835616941711
        unit_seconds = duration / 200.0
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=unit_seconds,
        )
        base_profile = _professional_direct_partition_profile(timing)
        selection = _VisualSelectionStub(
            tuple(index % 10 for index in range(len(base_profile.spans))),
            10,
        )
        failures = {
            0: _direct_partition_failure_fixture(
                span_count=len(base_profile.spans),
                failed_segment_index=4,
                frontier_boundary_index=3,
                zero_local_boundary_indices=(3,),
            ),
            1: _direct_partition_failure_fixture(
                span_count=len(base_profile.spans),
                failed_segment_index=12,
                frontier_boundary_index=11,
            ),
            2: _direct_partition_failure_fixture(
                span_count=len(base_profile.spans),
                failed_segment_index=28,
                frontier_boundary_index=27,
            ),
        }
        calls: list[tuple[int, ...]] = []

        def choose_profile(**kwargs: object) -> _VisualSelectionStub:
            spans = kwargs["canonical_spans"]
            assert isinstance(spans, tuple)
            unit_counts = tuple(
                int(round((end - start) / unit_seconds))
                for start, end in spans
            )
            calls.append(unit_counts)
            shifted = sum(
                unit_counts[index : index + 2] != (4, 4)
                for index in (3, 11, 27)
            )
            if shifted >= 3:
                return selection.bind(spans, kwargs["config"])
            raise DirectMemberVisualSelectionError(
                "synthetic sequential hard bottleneck",
                failures[shifted],
            )

        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=choose_profile,
        ):
            segments, _transitions, _details, cadence, _visual_selection = (
                _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing,
                    eligible_indices=tuple(range(10)),
                    eligible_sources=tuple(
                        Path(f"official-gaeul-{index}.mp4")
                        for index in range(10)
                    ),
                    common_master_start=0.0,
                    source_timewarps=None,
                    scene_cut_times=((),) * 10,
                    source_metadata_contract=(
                        OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                    ),
                    descriptor_provider=lambda *_args: None,
                )
            )

        self.assertEqual(len(calls), 19)
        self.assertEqual(
            sorted(
                sum(
                    counts[index : index + 2] != (4, 4)
                    for index in (3, 11, 27)
                )
                for counts in calls
            ),
            [0] + [1] * 2 + [2] * 8 + [3] * 8,
        )
        partition = cadence["partition_search"]
        attempted_ids = [item["profile_id"] for item in partition["profiles_attempted"]]
        attempted_signatures = [
            item["partition_signature_sha256"]
            for item in partition["profiles_attempted"]
        ]
        self.assertEqual(
            len(attempted_signatures),
            len(set(attempted_signatures)),
        )
        self.assertEqual(attempted_ids[0], base_profile.profile_id)
        self.assertTrue(any("frontier-27" in item for item in attempted_ids))
        selected = next(
            item
            for item in partition["profiles_attempted"]
            if item["profile_id"] == partition["selected_profile_id"]
        )
        self.assertEqual(selected["shifted_boundary_count"], 3)
        self.assertEqual(
            [
                item["boundary_index"]
                for item in selected["proof_directed_boundary_shifts"]
            ],
            [3, 11, 27],
        )
        self.assertEqual(partition["generated_profile_count"], 43)
        self.assertFalse(partition["profile_cap_reached"])
        self.assertEqual(len(segments), len(base_profile.spans))
        self.assertTrue(cadence["passed"])

    def test_official_partition_feasible_pruning_ignores_proof_progress_tie_breaker(
        self,
    ) -> None:
        duration = 192.0
        unit_seconds = duration / 200.0
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=unit_seconds,
        )
        base_profile = _professional_direct_partition_profile(timing)
        source_path = tuple(
            index % 10 for index in range(len(base_profile.spans))
        )
        high_cost_selection = _VisualSelectionStub(source_path, 10)
        high_cost_selection.optimizer_cost = 10.0
        low_cost_selection = _VisualSelectionStub(source_path, 10)
        low_cost_selection.optimizer_cost = 1.0
        base_failure = _direct_partition_failure_fixture(
            span_count=len(base_profile.spans),
            failed_segment_index=4,
            frontier_boundary_index=3,
            zero_local_boundary_indices=(3,),
        )
        farther_failure = _direct_partition_failure_fixture(
            span_count=len(base_profile.spans),
            failed_segment_index=28,
            frontier_boundary_index=27,
        )
        nearer_failure = _direct_partition_failure_fixture(
            span_count=len(base_profile.spans),
            failed_segment_index=12,
            frontier_boundary_index=11,
        )
        calls: list[tuple[int, ...]] = []

        def choose_profile(**kwargs: object) -> _VisualSelectionStub:
            spans = kwargs["canonical_spans"]
            assert isinstance(spans, tuple)
            counts = tuple(
                int(round((end - start) / unit_seconds))
                for start, end in spans
            )
            calls.append(counts)
            if counts == base_profile.unit_counts:
                raise DirectMemberVisualSelectionError("base", base_failure)
            if counts[27:29] != (4, 4):
                return high_cost_selection.bind(spans, kwargs["config"])
            if counts[11:13] != (4, 4):
                return low_cost_selection.bind(spans, kwargs["config"])
            if counts[3:5] == (5, 3):
                raise DirectMemberVisualSelectionError(
                    "farther proof progress",
                    farther_failure,
                )
            raise DirectMemberVisualSelectionError(
                "nearer proof progress",
                nearer_failure,
            )

        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=choose_profile,
        ):
            _segments, _transitions, _details, cadence, selection = (
                _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing,
                    eligible_indices=tuple(range(10)),
                    eligible_sources=tuple(
                        Path(f"official-gaeul-{index}.mp4")
                        for index in range(10)
                    ),
                    common_master_start=0.0,
                    source_timewarps=None,
                    scene_cut_times=((),) * 10,
                    source_metadata_contract=(
                        OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                    ),
                    descriptor_provider=lambda *_args: None,
                )
            )

        # Both depth-two structural peers are evaluated even though the b27
        # parent carries a farther proof-progress tie-breaker.  Selection then
        # uses the real optimizer rank and chooses the lower-cost b11 repair.
        attempted = cadence["partition_search"]["profiles_attempted"]
        self.assertEqual(len(calls), 7)
        self.assertEqual(len(attempted), 7)
        self.assertTrue(any(counts[27:29] != (4, 4) for counts in calls))
        self.assertTrue(any(counts[11:13] != (4, 4) for counts in calls))
        self.assertIn(
            "frontier-11",
            cadence["partition_search"]["selected_profile_id"],
        )
        self.assertIs(selection, low_cost_selection)

    def test_official_partition_allows_exact_adjacent_forward_compensation(
        self,
    ) -> None:
        duration = 192.0
        unit_seconds = duration / 200.0
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=unit_seconds,
        )
        selection = _VisualSelectionStub(tuple(range(10)) * 5, 10)
        base_failure = _direct_partition_failure_fixture(
            failed_segment_index=4,
            frontier_boundary_index=3,
            zero_local_boundary_indices=(3,),
        )
        adjacent_failure = _direct_partition_failure_fixture(
            failed_segment_index=5,
            frontier_boundary_index=4,
        )

        def choose_profile(**kwargs: object) -> _VisualSelectionStub:
            spans = kwargs["canonical_spans"]
            assert isinstance(spans, tuple)
            counts = tuple(
                int(round((end - start) / unit_seconds))
                for start, end in spans
            )
            if counts == (4,) * 50:
                raise DirectMemberVisualSelectionError(
                    "synthetic local bottleneck",
                    base_failure,
                )
            if counts[3:6] == (5, 4, 3):
                return selection.bind(spans, kwargs["config"])
            raise DirectMemberVisualSelectionError(
                "synthetic immediately adjacent frontier",
                adjacent_failure,
            )

        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=choose_profile,
        ):
            _segments, _transitions, _details, cadence, _selection = (
                _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing,
                    eligible_indices=tuple(range(10)),
                    eligible_sources=tuple(
                        Path(f"official-gaeul-{index}.mp4")
                        for index in range(10)
                    ),
                    common_master_start=0.0,
                    source_timewarps=None,
                    scene_cut_times=((),) * 10,
                    source_metadata_contract=(
                        OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                    ),
                    descriptor_provider=lambda *_args: None,
                )
            )

        partition = cadence["partition_search"]
        selected = next(
            item
            for item in partition["profiles_attempted"]
            if item["profile_id"] == partition["selected_profile_id"]
        )
        self.assertEqual(
            selected["selected_partition_unit_counts"]
            if "selected_partition_unit_counts" in selected
            else selected["partition_unit_counts"][3:6],
            [5, 4, 3],
        )
        shifts = selected["proof_directed_boundary_shifts"]
        self.assertEqual(
            [(item["boundary_index"], item["shift_units"]) for item in shifts],
            [(3, 1), (4, 1)],
        )
        parent_profile = next(
            item
            for item in partition["profiles_attempted"]
            if item["profile_id"] == shifts[1]["parent_profile_id"]
        )
        self.assertEqual(
            shifts[1]["parent_partition_signature_sha256"],
            parent_profile["partition_signature_sha256"],
        )
        self.assertEqual(shifts[1]["proof"]["failed_segment_index"], 5)
        self.assertEqual(
            shifts[1]["trigger"],
            "proven_forward_connectivity_frontier_exhaustion",
        )
        self.assertTrue(cadence["passed"])

    def test_official_partition_uses_proven_two_unit_single_boundary_move(
        self,
    ) -> None:
        duration = 192.0
        unit_seconds = duration / 200.0
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=unit_seconds,
        )
        # Keep the synthetic selector faithful to the production duration
        # quota gate.  The +2/-2 move at boundary 11 is offset by the already
        # proven +1/-1 move at boundary 3 while every source still owns five
        # non-adjacent spans.
        source_indices = list(tuple(range(10)) * 5)
        source_indices[2], source_indices[23] = (
            source_indices[23],
            source_indices[2],
        )
        source_indices[3], source_indices[22] = (
            source_indices[22],
            source_indices[3],
        )
        source_indices[4], source_indices[21] = (
            source_indices[21],
            source_indices[4],
        )
        selection = _VisualSelectionStub(tuple(source_indices), 10)
        base_failure = _direct_partition_failure_fixture(
            failed_segment_index=4,
            frontier_boundary_index=3,
            zero_local_boundary_indices=(3,),
        )
        frontier_failure = _direct_partition_failure_fixture(
            failed_segment_index=12,
            frontier_boundary_index=11,
        )

        def choose_profile(**kwargs: object) -> _VisualSelectionStub:
            spans = kwargs["canonical_spans"]
            assert isinstance(spans, tuple)
            counts = tuple(
                int(round((end - start) / unit_seconds))
                for start, end in spans
            )
            if counts == (4,) * 50:
                raise DirectMemberVisualSelectionError("base", base_failure)
            if counts[3:5] == (5, 3) and counts[11:13] == (6, 2):
                return selection.bind(spans, kwargs["config"])
            raise DirectMemberVisualSelectionError(
                "needs a two-unit compensation",
                frontier_failure,
            )

        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=choose_profile,
        ):
            _segments, _transitions, _details, cadence, _selection = (
                _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing,
                    eligible_indices=tuple(range(10)),
                    eligible_sources=tuple(
                        Path(f"official-gaeul-{index}.mp4")
                        for index in range(10)
                    ),
                    common_master_start=0.0,
                    source_timewarps=None,
                    scene_cut_times=((),) * 10,
                    source_metadata_contract=(
                        OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                    ),
                    descriptor_provider=lambda *_args: None,
                )
            )

        partition = cadence["partition_search"]
        selected = next(
            item
            for item in partition["profiles_attempted"]
            if item["profile_id"] == partition["selected_profile_id"]
        )
        self.assertEqual(selected["partition_unit_counts"][11:13], [6, 2])
        shifts = selected["proof_directed_boundary_shifts"]
        self.assertEqual(shifts[-1]["boundary_index"], 11)
        self.assertEqual(shifts[-1]["shift_units"], 2)
        parent_profile = next(
            item
            for item in partition["profiles_attempted"]
            if item["profile_id"] == shifts[-1]["parent_profile_id"]
        )
        self.assertEqual(
            shifts[-1]["parent_partition_signature_sha256"],
            parent_profile["partition_signature_sha256"],
        )
        self.assertEqual(shifts[-1]["proof"]["failed_segment_index"], 12)
        self.assertLessEqual(selected["maximum_span_seconds"], 7.2)
        self.assertTrue(cadence["passed"])

    def test_official_partition_does_not_shift_an_empty_content_side(self) -> None:
        timing = _rotating_timing_draft(
            duration_seconds=192.0,
            unit_seconds=0.96,
        )
        failure = DirectMemberVisualSelectionError(
            "synthetic empty right content side",
            _direct_partition_failure_fixture(
                failed_segment_index=4,
                frontier_boundary_index=3,
                zero_local_boundary_indices=(3,),
                empty_valid_side_boundary_index=3,
            ),
        )
        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=failure,
        ) as select_visual, self.assertRaises(
            DirectMemberVisualSelectionError
        ) as raised:
            _official_target_fancam_direct_full_frame_segments(
                timing_draft=timing,
                eligible_indices=tuple(range(10)),
                eligible_sources=tuple(
                    Path(f"official-gaeul-{index}.mp4") for index in range(10)
                ),
                common_master_start=0.0,
                source_timewarps=None,
                scene_cut_times=((),) * 10,
                source_metadata_contract=(
                    OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                ),
                descriptor_provider=lambda *_args: None,
            )

        self.assertEqual(select_visual.call_count, 2)
        search = raised.exception.diagnostics["partition_search"]
        self.assertEqual(search["generated_profile_count"], 1)
        self.assertFalse(search["profile_cap_reached"])
        self.assertEqual(len(search["profiles_attempted"]), 1)

    def test_official_partition_profile_cap_is_atomic_fail_closed(self) -> None:
        timing = _rotating_timing_draft(
            duration_seconds=192.0,
            unit_seconds=0.96,
        )
        failures = [
            _direct_partition_failure_fixture(
                failed_segment_index=4,
                frontier_boundary_index=3,
                zero_local_boundary_indices=(3,),
            ),
            _direct_partition_failure_fixture(
                failed_segment_index=12,
                frontier_boundary_index=11,
            ),
        ]
        call_count = 0
        progress_events: list[dict[str, object]] = []

        def always_fail(**_kwargs: object) -> _VisualSelectionStub:
            nonlocal call_count
            diagnostics = failures[min(call_count, 1)]
            call_count += 1
            raise DirectMemberVisualSelectionError(
                "synthetic bounded search failure",
                diagnostics,
            )

        with patch(
            "crosscut.pipeline._DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT",
            2,
        ), patch(
            "crosscut.pipeline._DIRECT_MEMBER_MAX_PARTITION_PROFILE_UPPER_BOUND",
            2,
        ), patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=always_fail,
        ) as select_visual, self.assertRaises(
            DirectMemberVisualSelectionError
        ) as raised:
            _official_target_fancam_direct_full_frame_segments(
                timing_draft=timing,
                eligible_indices=tuple(range(10)),
                eligible_sources=tuple(
                    Path(f"official-gaeul-{index}.mp4") for index in range(10)
                ),
                common_master_start=0.0,
                source_timewarps=None,
                scene_cut_times=((),) * 10,
                source_metadata_contract=(
                    OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                ),
                descriptor_provider=lambda *_args: None,
                partition_progress_callback=progress_events.append,
            )

        self.assertEqual(select_visual.call_count, 2)
        self.assertEqual(
            raised.exception.diagnostics["failure"],
            "direct_partition_search_capacity_exhausted",
        )
        self.assertFalse(
            raised.exception.diagnostics["constraint_infeasibility_proven"]
        )
        search = raised.exception.diagnostics["partition_search"]
        self.assertTrue(search["profile_cap_reached"])
        self.assertEqual(search["maximum_profile_count"], 2)
        self.assertEqual(search["profile_limit_expansions"], [])
        self.assertFalse(search["render_started"])
        final_progress = progress_events[-1]
        self.assertEqual(final_progress["result"], "profile_cap_reached")
        self.assertEqual(final_progress["evaluated_profile_count"], 2)
        self.assertEqual(final_progress["evaluation_limit"], 2)
        self.assertTrue(final_progress["profile_cap_reached"])

    def test_official_partition_static_upper_bound_is_atomic_fail_closed(
        self,
    ) -> None:
        self.assertEqual(
            _DIRECT_MEMBER_MAX_PARTITION_PROFILE_UPPER_BOUND,
            4_163,
        )
        timing = _rotating_timing_draft(
            duration_seconds=192.0,
            unit_seconds=0.96,
        )
        failures = (
            _direct_partition_failure_fixture(
                failed_segment_index=4,
                frontier_boundary_index=3,
                zero_local_boundary_indices=(3,),
            ),
            _direct_partition_failure_fixture(
                failed_segment_index=12,
                frontier_boundary_index=11,
            ),
        )
        call_count = 0

        def always_fail(**_kwargs: object) -> _VisualSelectionStub:
            nonlocal call_count
            diagnostics = failures[min(call_count, 1)]
            call_count += 1
            raise DirectMemberVisualSelectionError(
                "synthetic static-bound failure",
                diagnostics,
            )

        with patch(
            "crosscut.pipeline._DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT",
            2,
        ), patch(
            "crosscut.pipeline._DIRECT_MEMBER_MAX_PARTITION_PROFILE_UPPER_BOUND",
            3,
        ), patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=always_fail,
        ), self.assertRaises(DirectMemberVisualSelectionError) as raised:
            _official_target_fancam_direct_full_frame_segments(
                timing_draft=timing,
                eligible_indices=tuple(range(10)),
                eligible_sources=tuple(
                    Path(f"official-gaeul-{index}.mp4") for index in range(10)
                ),
                common_master_start=0.0,
                source_timewarps=None,
                scene_cut_times=((),) * 10,
                source_metadata_contract=(
                    OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                ),
                descriptor_provider=lambda *_args: None,
            )

        self.assertEqual(call_count, 3)
        search = raised.exception.diagnostics["partition_search"]
        self.assertTrue(search["profile_cap_reached"])
        self.assertEqual(search["maximum_profile_count"], 3)
        self.assertEqual(search["static_profile_upper_bound"], 3)
        self.assertEqual(len(search["profile_limit_expansions"]), 1)
        expansion = search["profile_limit_expansions"][0]
        self.assertGreater(
            expansion["exact_frontier_continuation_upper_bound"],
            3,
        )
        self.assertEqual(expansion["new_limit"], 3)
        self.assertTrue(expansion["truncated_by_static_upper_bound"])

    def test_official_partition_limit_expands_to_exact_queue_snapshot(
        self,
    ) -> None:
        duration = 192.0
        unit_seconds = duration / 200.0
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=unit_seconds,
        )
        selection = _VisualSelectionStub(tuple(range(10)) * 5, 10)
        base_failure = _direct_partition_failure_fixture(
            failed_segment_index=4,
            frontier_boundary_index=3,
            zero_local_boundary_indices=(3,),
        )
        frontier_failure = _direct_partition_failure_fixture(
            failed_segment_index=12,
            frontier_boundary_index=11,
        )
        calls: list[tuple[int, ...]] = []
        progress_events: list[dict[str, object]] = []

        def choose_profile(**kwargs: object) -> _VisualSelectionStub:
            spans = kwargs["canonical_spans"]
            assert isinstance(spans, tuple)
            counts = tuple(
                int(round((end - start) / unit_seconds))
                for start, end in spans
            )
            calls.append(counts)
            if counts == (4,) * 50:
                raise DirectMemberVisualSelectionError("base", base_failure)
            if counts[11:13] != (4, 4):
                return selection.bind(spans, kwargs["config"])
            raise DirectMemberVisualSelectionError(
                "proof-bound frontier",
                frontier_failure,
            )

        with patch(
            "crosscut.pipeline._DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT",
            3,
        ), patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=choose_profile,
        ):
            _segments, _transitions, _details, cadence, _selection = (
                _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing,
                    eligible_indices=tuple(range(10)),
                    eligible_sources=tuple(
                        Path(f"official-gaeul-{index}.mp4")
                        for index in range(10)
                    ),
                    common_master_start=0.0,
                    source_timewarps=None,
                    scene_cut_times=((),) * 10,
                    source_metadata_contract=(
                        OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                    ),
                    descriptor_provider=lambda *_args: None,
                    partition_progress_callback=progress_events.append,
                )
            )

        self.assertEqual(len(calls), 7)
        search = cadence["partition_search"]
        self.assertEqual(search["initial_profile_count"], 3)
        self.assertEqual(search["maximum_profile_count"], 7)
        self.assertEqual(search["generated_profile_count"], 11)
        self.assertFalse(search["profile_cap_reached"])
        self.assertEqual(len(search["profile_limit_expansions"]), 2)
        expansion = search["profile_limit_expansions"][0]
        self.assertEqual(expansion["epoch"], 1)
        self.assertEqual(expansion["evaluated_at_expansion"], 3)
        self.assertEqual(expansion["queued_unique_profile_count"], 2)
        self.assertEqual(expansion["exact_frontier_continuation_upper_bound"], 5)
        self.assertEqual(len(expansion["queue_signature_digest_sha256"]), 64)
        progress_expansion = next(
            item
            for item in progress_events
            if item["event"]
            == "direct_member_partition_search_limit_extended"
        )
        self.assertEqual(progress_expansion["new_limit"], 5)
        self.assertEqual(progress_expansion["queued_unique_profile_count"], 2)
        self.assertEqual(progress_events[-1]["evaluation_limit"], 7)
        self.assertEqual(progress_events[-1]["result"], "feasible")

    def test_direct_partition_future_domain_and_queue_digest_are_deterministic(
        self,
    ) -> None:
        timing = _rotating_timing_draft(
            duration_seconds=192.0,
            unit_seconds=0.96,
        )
        base = _direct_full_frame_base_partition_profile(timing.segments)
        base_failure = _direct_partition_failure_fixture(
            failed_segment_index=4,
            frontier_boundary_index=3,
            zero_local_boundary_indices=(3,),
        )
        children = _direct_full_frame_proof_directed_child_profiles(
            timing.segments,
            base,
            base_failure,
        )
        self.assertEqual(len(children), 2)
        self.assertEqual(
            _direct_partition_queue_signature_digest(children),
            _direct_partition_queue_signature_digest(tuple(reversed(children))),
        )
        key = _direct_partition_future_operation_state_key(
            base,
            base_failure,
            children,
        )
        permuted_key = _direct_partition_future_operation_state_key(
            base,
            base_failure,
            tuple(reversed(children)),
        )
        self.assertEqual(key, permuted_key)
        self.assertEqual(key[1], ())

        frontier_failure = _direct_partition_failure_fixture(
            failed_segment_index=12,
            frontier_boundary_index=11,
        )
        grandchildren = _direct_full_frame_proof_directed_child_profiles(
            timing.segments,
            children[0],
            frontier_failure,
        )
        child_key = _direct_partition_future_operation_state_key(
            children[0],
            frontier_failure,
            grandchildren,
        )
        self.assertEqual(child_key[1], (3,))
        self.assertEqual(child_key[2], 3)
        self.assertNotEqual(key, child_key)

    def test_official_partition_allows_exact_seventh_proof_then_fails_closed(
        self,
    ) -> None:
        timing = _rotating_timing_draft(
            duration_seconds=192.0,
            unit_seconds=0.96,
        )
        profile = _direct_full_frame_base_partition_profile(timing.segments)
        proof_boundaries = (3, 11, 27, 34, 39, 40, 45)
        for depth, boundary_index in enumerate(proof_boundaries, start=1):
            diagnostics = _direct_partition_failure_fixture(
                failed_segment_index=boundary_index + 1,
                frontier_boundary_index=boundary_index,
                zero_local_boundary_indices=(boundary_index,) if depth == 1 else (),
            )
            children = _direct_full_frame_proof_directed_child_profiles(
                timing.segments,
                profile,
                diagnostics,
            )
            matching = tuple(
                child
                for child in children
                if child.boundary_shifts[-1].boundary_index == boundary_index
                and child.boundary_shifts[-1].shift_units == 1
            )
            self.assertEqual(len(matching), 1)
            child = matching[0]
            last_shift = child.boundary_shifts[-1]
            self.assertEqual(
                last_shift.parent_partition_signature_sha256,
                hashlib.sha256(
                    json.dumps(
                        list(profile.unit_counts),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            )
            self.assertEqual(
                last_shift.proof_failed_segment_index,
                boundary_index + 1,
            )
            self.assertEqual(len(child.boundary_shifts), depth)
            profile = child

        self.assertEqual(
            [shift.boundary_index for shift in profile.boundary_shifts],
            list(proof_boundaries),
        )
        self.assertEqual(profile.unit_counts[39:42], (5, 4, 3))
        exhausted = _direct_partition_failure_fixture(
            failed_segment_index=48,
            frontier_boundary_index=47,
        )
        self.assertEqual(
            _direct_full_frame_proof_directed_child_profiles(
                timing.segments,
                profile,
                exhausted,
            ),
            (),
        )

    def test_official_partition_adaptive_frontier_reaches_adjacent_fifth_proof(
        self,
    ) -> None:
        duration = 192.0
        unit_seconds = duration / 200.0
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=unit_seconds,
        )
        selection = _VisualSelectionStub(tuple(range(10)) * 5, 10)
        failures = (
            _direct_partition_failure_fixture(
                failed_segment_index=4,
                frontier_boundary_index=3,
                zero_local_boundary_indices=(3,),
            ),
            _direct_partition_failure_fixture(
                failed_segment_index=12,
                frontier_boundary_index=11,
            ),
            _direct_partition_failure_fixture(
                failed_segment_index=28,
                frontier_boundary_index=27,
            ),
            _direct_partition_failure_fixture(
                failed_segment_index=35,
                frontier_boundary_index=34,
            ),
            _direct_partition_failure_fixture(
                failed_segment_index=36,
                frontier_boundary_index=35,
            ),
        )
        calls: list[tuple[int, ...]] = []

        def choose_profile(**kwargs: object) -> _VisualSelectionStub:
            spans = kwargs["canonical_spans"]
            assert isinstance(spans, tuple)
            counts = tuple(
                int(round((end - start) / unit_seconds))
                for start, end in spans
            )
            calls.append(counts)
            if counts[3:5] == (4, 4):
                failure = failures[0]
            elif counts[11:13] == (4, 4):
                failure = failures[1]
            elif counts[27:29] == (4, 4):
                failure = failures[2]
            elif counts[34:36] == (4, 4):
                failure = failures[3]
            elif (
                counts[3:5] == (5, 3)
                and counts[11:13] == (5, 3)
                and counts[27:29] == (5, 3)
                and counts[34:37] == (5, 4, 3)
            ):
                return selection.bind(spans, kwargs["config"])
            else:
                failure = failures[4]
            raise DirectMemberVisualSelectionError(
                "synthetic five-proof frontier",
                failure,
            )

        with patch(
            "crosscut.pipeline._DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT",
            16,
        ), patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=choose_profile,
        ):
            _segments, _transitions, _details, cadence, _selection = (
                _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing,
                    eligible_indices=tuple(range(10)),
                    eligible_sources=tuple(
                        Path(f"official-gaeul-{index}.mp4")
                        for index in range(10)
                    ),
                    common_master_start=0.0,
                    source_timewarps=None,
                    scene_cut_times=((),) * 10,
                    source_metadata_contract=(
                        OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                    ),
                    descriptor_provider=lambda *_args: None,
                )
            )

        search = cadence["partition_search"]
        self.assertGreater(len(calls), 16)
        self.assertGreater(len(search["profile_limit_expansions"]), 0)
        selected = next(
            item
            for item in search["profiles_attempted"]
            if item["profile_id"] == search["selected_profile_id"]
        )
        shifts = selected["proof_directed_boundary_shifts"]
        self.assertEqual(
            [item["boundary_index"] for item in shifts],
            [3, 11, 27, 34, 35],
        )
        self.assertEqual(len({item["boundary_index"] for item in shifts}), 5)
        self.assertEqual(selected["partition_unit_counts"][34:37], [5, 4, 3])
        self.assertFalse(search["profile_cap_reached"])
        self.assertTrue(cadence["passed"])

    def test_visual_frontier_split_uses_one_hard_valid_bridge_boundary(
        self,
    ) -> None:
        duration = 196.4835616941711
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=duration / 200.0,
        )
        base = _direct_full_frame_base_partition_profile(timing.segments)
        diagnostics = _direct_partition_failure_fixture(
            span_count=len(base.spans),
            failed_segment_index=46,
            frontier_boundary_index=45,
        )
        diagnostics["span_content_validity"] = [
            _visual_frontier_span_diagnostic(
                base.spans[46],
                segment_index=46,
            )
        ]

        children = _direct_full_frame_visual_frontier_split_child_profiles(
            timing.segments,
            base,
            diagnostics,
            config=DirectMemberVisualConfig(),
        )

        self.assertEqual(len(children), 1)
        child = children[0]
        self.assertEqual(sum(child.unit_counts), sum(base.unit_counts))
        self.assertEqual(len(child.unit_counts), len(base.unit_counts) + 1)
        self.assertEqual(child.unit_counts[46:48], (2, 2))
        self.assertEqual(len(child.visual_frontier_splits), 1)
        split = child.visual_frontier_splits[0]
        self.assertEqual(split.parent_span_index, 46)
        self.assertEqual(
            split.evidence_contract,
            "visual_bracket_bound_existing_timing_boundary_v1",
        )
        self.assertEqual(
            split.evidence_kind,
            "hard_valid_bridge_source_entry_exit_bracket",
        )
        self.assertEqual(split.proof_frontier_boundary_index, 45)
        self.assertEqual(split.proof_failed_segment_index, 46)
        self.assertEqual(
            len(child.visual_frontier_splits),
            1,
            "one profile may receive only one micro split",
        )
        self.assertEqual(
            _direct_full_frame_visual_frontier_split_child_profiles(
                timing.segments,
                child,
                diagnostics,
                config=DirectMemberVisualConfig(),
            ),
            (),
        )

    def test_visual_frontier_split_prefers_exact_native_cut_snap(self) -> None:
        duration = 196.4835616941711
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=duration / 200.0,
        )
        base = _direct_full_frame_base_partition_profile(timing.segments)
        target = base.spans[46]
        target_unit_cursor = sum(base.unit_counts[:46])
        exact_timing_boundary = timing.segments[
            target_unit_cursor + 1
        ].canonical_master_end
        native_cut = exact_timing_boundary + 0.10
        diagnostics = _direct_partition_failure_fixture(
            span_count=len(base.spans),
            failed_segment_index=46,
            frontier_boundary_index=45,
        )
        diagnostics["span_content_validity"] = [
            _visual_frontier_span_diagnostic(
                target,
                segment_index=46,
                native_cut_master_seconds=native_cut,
            )
        ]

        child = _direct_full_frame_visual_frontier_split_child_profiles(
            timing.segments,
            base,
            diagnostics,
            config=DirectMemberVisualConfig(),
        )[0]
        split = child.visual_frontier_splits[0]
        self.assertEqual(
            split.evidence_kind,
            "confirmed_source_native_cut_master_time_snap",
        )
        self.assertAlmostEqual(split.canonical_master_seconds, exact_timing_boundary)
        self.assertEqual(split.evidence_master_seconds, (native_cut,))
        self.assertIsNone(split.evidence_bracket_master_seconds)

    def test_visual_frontier_split_records_sample_change_as_bracket_only(
        self,
    ) -> None:
        duration = 196.4835616941711
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=duration / 200.0,
        )
        base = _direct_full_frame_base_partition_profile(timing.segments)
        diagnostics = _direct_partition_failure_fixture(
            span_count=len(base.spans),
            failed_segment_index=46,
            frontier_boundary_index=45,
        )
        diagnostics["span_content_validity"] = [
            _visual_frontier_span_diagnostic(
                base.spans[46],
                segment_index=46,
                sample_safety_change=True,
            )
        ]

        child = _direct_full_frame_visual_frontier_split_child_profiles(
            timing.segments,
            base,
            diagnostics,
            config=DirectMemberVisualConfig(),
        )[0]
        split = child.visual_frontier_splits[0]
        self.assertEqual(
            split.evidence_kind,
            "sample_hard_safety_change_bracket",
        )
        self.assertEqual(split.evidence_master_seconds, ())
        self.assertIsNotNone(split.evidence_bracket_master_seconds)
        assert split.evidence_bracket_master_seconds is not None
        self.assertLessEqual(
            split.evidence_bracket_master_seconds[0],
            split.canonical_master_seconds,
        )
        self.assertLessEqual(
            split.canonical_master_seconds,
            split.evidence_bracket_master_seconds[1],
        )

    def test_visual_frontier_split_requires_exact_forward_exhaustion_proof(
        self,
    ) -> None:
        duration = 196.4835616941711
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=duration / 200.0,
        )
        base = _direct_full_frame_base_partition_profile(timing.segments)
        diagnostics = _direct_partition_failure_fixture(
            span_count=len(base.spans),
            failed_segment_index=46,
            frontier_boundary_index=45,
            zero_local_boundary_indices=(45,),
        )
        diagnostics["span_content_validity"] = [
            _visual_frontier_span_diagnostic(
                base.spans[46],
                segment_index=46,
            )
        ]

        self.assertEqual(
            _direct_full_frame_visual_frontier_split_child_profiles(
                timing.segments,
                base,
                diagnostics,
                config=DirectMemberVisualConfig(),
            ),
            (),
        )

    def test_official_partition_visual_frontier_split_is_fully_reevaluated(
        self,
    ) -> None:
        duration = 196.4835616941711
        timing = _rotating_timing_draft(
            source_count=6,
            duration_seconds=duration,
            unit_seconds=duration / 200.0,
        )
        selection = _VisualSelectionStub(
            tuple(index % 6 for index in range(52)),
            6,
        )
        calls: list[tuple[tuple[float, float], ...]] = []

        def choose_profile(**kwargs: object) -> _VisualSelectionStub:
            spans = kwargs["canonical_spans"]
            assert isinstance(spans, tuple)
            calls.append(spans)
            if len(spans) == 52:
                return selection.bind(spans, kwargs["config"])
            diagnostics = _direct_partition_failure_fixture(
                span_count=len(spans),
                failed_segment_index=46,
                frontier_boundary_index=45,
            )
            span = SimpleNamespace(
                canonical_master_start=spans[46][0],
                canonical_master_end=spans[46][1],
            )
            diagnostics["span_content_validity"] = [
                _visual_frontier_span_diagnostic(
                    span,
                    segment_index=46,
                    source_count=6,
                )
            ]
            raise DirectMemberVisualSelectionError(
                "synthetic frontier requiring an intermediate bridge",
                diagnostics,
            )

        with patch(
            "crosscut.pipeline._DIRECT_MEMBER_MAX_SHIFTED_BOUNDARY_COUNT",
            0,
        ), patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=choose_profile,
        ) as select_visual:
            _segments, _transitions, _details, cadence, _selection = (
                _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing,
                    eligible_indices=tuple(range(6)),
                    eligible_sources=tuple(
                        Path(f"official-gaeul-{index}.mp4")
                        for index in range(6)
                    ),
                    common_master_start=0.0,
                    source_timewarps=None,
                    scene_cut_times=((),) * 6,
                    source_metadata_contract=(
                        OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                    ),
                    descriptor_provider=lambda *_args: None,
                )
            )

        self.assertEqual(select_visual.call_count, 2)
        self.assertEqual([len(spans) for spans in calls], [51, 52])
        search = cadence["partition_search"]
        self.assertTrue(cadence["passed"])
        self.assertFalse(search["profile_cap_reached"])
        selected = next(
            item
            for item in search["profiles_attempted"]
            if item["profile_id"] == search["selected_profile_id"]
        )
        self.assertEqual(selected["visual_frontier_split_count"], 1)
        self.assertEqual(selected["structural_change_count"], 1)
        self.assertEqual(
            selected["proof_directed_visual_frontier_splits"][0][
                "proposal_evidence"
            ]["contract"],
            "visual_bracket_bound_existing_timing_boundary_v1",
        )

    def test_visual_split_exact_capacity_gets_one_gate_preserving_retry(
        self,
    ) -> None:
        duration = 196.4835616941711
        timing = _rotating_timing_draft(
            source_count=6,
            duration_seconds=duration,
            unit_seconds=duration / 200.0,
        )
        selection = _VisualSelectionStub(
            tuple(index % 6 for index in range(52)),
            6,
        )
        calls: list[tuple[int, DirectMemberVisualConfig]] = []
        progress_events: list[dict[str, object]] = []

        def choose_profile(**kwargs: object) -> _VisualSelectionStub:
            spans = kwargs["canonical_spans"]
            config = kwargs["config"]
            assert isinstance(spans, tuple)
            assert isinstance(config, DirectMemberVisualConfig)
            calls.append((len(spans), config))
            if len(spans) == 51:
                diagnostics = _direct_partition_failure_fixture(
                    span_count=51,
                    failed_segment_index=46,
                    frontier_boundary_index=45,
                )
                span = SimpleNamespace(
                    canonical_master_start=spans[46][0],
                    canonical_master_end=spans[46][1],
                )
                diagnostics["span_content_validity"] = [
                    _visual_frontier_span_diagnostic(
                        span,
                        segment_index=46,
                        source_count=6,
                    )
                ]
                raise DirectMemberVisualSelectionError("base", diagnostics)
            if (
                config.feasibility_lifeline_state_limit
                == DirectMemberVisualConfig().feasibility_lifeline_state_limit
            ):
                raise DirectMemberVisualSelectionError(
                    "exact lifeline initial capacity",
                    _direct_partition_exact_capacity_failure_fixture(
                        state_limit=config.feasibility_lifeline_state_limit,
                    ),
                )
            return selection.bind(spans, config)

        with patch(
            "crosscut.pipeline._DIRECT_MEMBER_MAX_SHIFTED_BOUNDARY_COUNT",
            0,
        ), patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=choose_profile,
        ) as select_visual:
            _segments, _transitions, _details, cadence, _selection = (
                _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing,
                    eligible_indices=tuple(range(6)),
                    eligible_sources=tuple(
                        Path(f"official-gaeul-{index}.mp4")
                        for index in range(6)
                    ),
                    common_master_start=0.0,
                    source_timewarps=None,
                    scene_cut_times=((),) * 6,
                    source_metadata_contract=(
                        OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                    ),
                    descriptor_provider=lambda *_args: None,
                    partition_progress_callback=progress_events.append,
                )
            )

        self.assertEqual(select_visual.call_count, 3)
        self.assertEqual([count for count, _config in calls], [51, 52, 52])
        self.assertEqual(
            [
                config.feasibility_lifeline_state_limit
                for _count, config in calls
            ],
            [250_000, 250_000, 1_000_000],
        )
        self.assertEqual(calls[1][1], calls[0][1])
        self.assertEqual(
            calls[2][1],
            replace(
                calls[1][1],
                feasibility_lifeline_state_limit=1_000_000,
            ),
        )
        search = cadence["partition_search"]
        retry = search["exact_lifeline_capacity_retry"]
        self.assertEqual(retry["retry_count"], 1)
        self.assertEqual(retry["success_count"], 1)
        self.assertEqual(retry["capacity_exhausted_count"], 0)
        self.assertFalse(retry["visual_gates_relaxed"])
        selected = next(
            item
            for item in search["profiles_attempted"]
            if item["profile_id"] == search["selected_profile_id"]
        )
        self.assertEqual(
            selected["exact_lifeline_retry"]["result"],
            "feasible",
        )
        retry_events = [
            item
            for item in progress_events
            if item["event"]
            == "direct_member_visual_split_exact_lifeline_retry"
        ]
        self.assertEqual(
            [item["phase"] for item in retry_events],
            ["exact_lifeline_retry_started", "exact_lifeline_retry_completed"],
        )
        self.assertEqual(retry_events[-1]["result"], "feasible")
        self.assertEqual(
            progress_events[-1]["exact_lifeline_capacity_retry_success_count"],
            1,
        )

    def test_extra_quota_candidates_are_complete_and_validity_slack_ordered(
        self,
    ) -> None:
        diagnostics = {
            "valid_count_by_source": [
                {
                    "eligible_source_index": index,
                    "valid_span_count": count,
                }
                for index, count in enumerate((49, 53, 48, 55, 48, 54))
            ]
        }

        self.assertEqual(
            _direct_strict_extra_quota_assignment_candidates(
                span_count=55,
                source_count=6,
                source_order=tuple(range(6)),
                diagnostics=diagnostics,
            ),
            ((3,), (5,), (1,), (2,), (4,)),
        )
        self.assertEqual(
            _direct_strict_extra_quota_assignment_candidates(
                span_count=55,
                source_count=6,
                source_order=(5, 4, 3, 2, 1, 0),
                diagnostics=diagnostics,
            ),
            ((3,), (1,), (0,), (4,), (2,)),
        )
        complete_domain = _direct_strict_extra_quota_assignment_candidates(
            span_count=12,
            source_count=8,
            source_order=tuple(range(8)),
            diagnostics={
                "valid_count_by_source": [
                    {
                        "eligible_source_index": index,
                        "valid_span_count": 12,
                    }
                    for index in range(8)
                ]
            },
        )
        self.assertEqual(len(complete_domain), math.comb(8, 4) - 1)
        self.assertNotIn((0, 1, 2, 3), complete_domain)
        self.assertEqual(len(set(complete_domain)), len(complete_domain))
        self.assertTrue(
            all(
                len(assignment) == 4
                and len(set(assignment)) == 4
                and set(assignment) <= set(range(8))
                for assignment in complete_domain
            )
        )

    def test_extra_quota_failure_certificate_rejects_capacity_or_quantization(
        self,
    ) -> None:
        proven = _direct_partition_exact_hard_quota_failure_fixture(
            span_count=55,
            valid_counts=(49, 53, 48, 55, 48, 54),
        )
        self.assertTrue(
            _direct_strict_extra_quota_assignment_failure_is_proven(proven)
        )

        capacity = deepcopy(proven)
        capacity["search"]["exact_lifeline"]["state_capacity_exhausted"] = True
        self.assertFalse(
            _direct_strict_extra_quota_assignment_failure_is_proven(capacity)
        )

        quantized = deepcopy(proven)
        quantized["search"]["exact_lifeline"][
            "duration_state_quantization_applied"
        ] = True
        self.assertFalse(
            _direct_strict_extra_quota_assignment_failure_is_proven(quantized)
        )

        unproven = deepcopy(proven)
        unproven["constraint_infeasibility_proven"] = False
        self.assertFalse(
            _direct_strict_extra_quota_assignment_failure_is_proven(unproven)
        )

    def test_retry_qualified_split_completely_evaluates_extra_quota_owners(
        self,
    ) -> None:
        duration = 196.4835616941711
        timing = _rotating_timing_draft(
            source_count=6,
            duration_seconds=duration,
            unit_seconds=duration / 216.0,
        )
        valid_counts = (49, 53, 48, 55, 48, 54)
        hard = _direct_partition_exact_hard_quota_failure_fixture(
            span_count=55,
            valid_counts=valid_counts,
        )
        calls: list[tuple[int, int, tuple[int, ...] | None, tuple[int, ...]]] = []
        progress_events: list[dict[str, object]] = []

        def selected_for(owner: int, cost: float) -> _VisualSelectionStub:
            selected = _VisualSelectionStub(
                tuple((owner + index) % 6 for index in range(55)),
                6,
            )
            selected.optimizer_cost = cost
            return selected

        def choose_profile(**kwargs: object) -> _VisualSelectionStub:
            spans = kwargs["canonical_spans"]
            config = kwargs["config"]
            assignment = kwargs.get("extra_quota_source_indices")
            order = kwargs["preferred_source_order"]
            assert isinstance(spans, tuple)
            assert isinstance(config, DirectMemberVisualConfig)
            assert assignment is None or isinstance(assignment, tuple)
            assert isinstance(order, tuple)
            calls.append(
                (
                    len(spans),
                    config.feasibility_lifeline_state_limit,
                    assignment,
                    order,
                )
            )
            if len(spans) == 54:
                diagnostics = _direct_partition_failure_fixture(
                    span_count=54,
                    failed_segment_index=50,
                    frontier_boundary_index=49,
                )
                span = SimpleNamespace(
                    canonical_master_start=spans[50][0],
                    canonical_master_end=spans[50][1],
                )
                diagnostics["span_content_validity"] = [
                    _visual_frontier_span_diagnostic(
                        span,
                        segment_index=50,
                        source_count=6,
                    )
                ]
                raise DirectMemberVisualSelectionError("base", diagnostics)
            if config.feasibility_lifeline_state_limit == 250_000:
                raise DirectMemberVisualSelectionError(
                    "initial exact capacity",
                    _direct_partition_exact_capacity_failure_fixture(
                        state_limit=250_000
                    ),
                )
            if assignment is None:
                raise DirectMemberVisualSelectionError("default quota", hard)
            if assignment == (3,):
                return selected_for(3, 2.0).bind(spans, config)
            if assignment == (5,):
                return selected_for(5, 1.0).bind(spans, config)
            alternate_hard = deepcopy(hard)
            alternate_hard["source_quotas"] = [
                10 if index in assignment else 9 for index in range(6)
            ]
            raise DirectMemberVisualSelectionError(
                "alternate quota",
                alternate_hard,
            )

        with patch(
            "crosscut.pipeline._DIRECT_MEMBER_MAX_SHIFTED_BOUNDARY_COUNT",
            0,
        ), patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=choose_profile,
        ):
            _segments, _transitions, _details, cadence, selection = (
                _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing,
                    eligible_indices=tuple(range(6)),
                    eligible_sources=tuple(
                        Path(f"official-gaeul-{index}.mp4")
                        for index in range(6)
                    ),
                    common_master_start=0.0,
                    source_timewarps=None,
                    scene_cut_times=((),) * 6,
                    source_metadata_contract=(
                        OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                    ),
                    descriptor_provider=lambda *_args: None,
                    partition_progress_callback=progress_events.append,
                )
            )

        self.assertEqual([item[0] for item in calls], [54] + [55] * 7)
        self.assertEqual([item[1] for item in calls], [250_000, 250_000] + [1_000_000] * 6)
        self.assertEqual(
            [item[2] for item in calls[3:]],
            [(3,), (5,), (1,), (2,), (4,)],
        )
        self.assertTrue(all(item[3] == tuple(range(6)) for item in calls))
        self.assertEqual(selection.source_counts, (9, 9, 9, 9, 9, 10))
        search = cadence["partition_search"]
        quota_search = search["strict_extra_quota_assignment_search"]
        self.assertEqual(quota_search["qualified_profile_count"], 1)
        self.assertEqual(quota_search["assignment_evaluation_count"], 5)
        self.assertEqual(quota_search["feasible_profile_count"], 1)
        self.assertFalse(quota_search["source_search_order_changed"])
        self.assertTrue(quota_search["counts_differ_by_at_most_one"])
        selected_attempt = next(
            item
            for item in search["profiles_attempted"]
            if item["profile_id"] == search["selected_profile_id"]
        )
        certificate = selected_attempt["strict_extra_quota_assignment_search"]
        self.assertTrue(certificate["complete_assignment_domain_evaluated"])
        self.assertEqual(certificate["total_assignment_domain_count"], 6)
        self.assertEqual(certificate["selected_extra_quota_source_indices"], [5])
        self.assertEqual(len(certificate["assignment_certificate_sha256"]), 64)
        evaluated_events = [
            item
            for item in progress_events
            if item["phase"] == "extra_quota_assignment_evaluated"
        ]
        self.assertEqual(len(evaluated_events), 5)
        self.assertEqual(
            progress_events[-1][
                "strict_extra_quota_assignment_evaluation_count"
            ],
            5,
        )

    def test_extra_quota_assignment_domain_exhaustion_fails_closed(self) -> None:
        duration = 196.4835616941711
        timing = _rotating_timing_draft(
            source_count=6,
            duration_seconds=duration,
            unit_seconds=duration / 216.0,
        )
        hard = _direct_partition_exact_hard_quota_failure_fixture(
            span_count=55,
            valid_counts=(49, 53, 48, 55, 48, 54),
        )

        def choose_profile(**kwargs: object) -> _VisualSelectionStub:
            spans = kwargs["canonical_spans"]
            config = kwargs["config"]
            assignment = kwargs.get("extra_quota_source_indices")
            assert isinstance(spans, tuple)
            assert isinstance(config, DirectMemberVisualConfig)
            if len(spans) == 54:
                diagnostics = _direct_partition_failure_fixture(
                    span_count=54,
                    failed_segment_index=50,
                    frontier_boundary_index=49,
                )
                span = SimpleNamespace(
                    canonical_master_start=spans[50][0],
                    canonical_master_end=spans[50][1],
                )
                diagnostics["span_content_validity"] = [
                    _visual_frontier_span_diagnostic(
                        span,
                        segment_index=50,
                        source_count=6,
                    )
                ]
                raise DirectMemberVisualSelectionError("base", diagnostics)
            if config.feasibility_lifeline_state_limit == 250_000:
                raise DirectMemberVisualSelectionError(
                    "initial exact capacity",
                    _direct_partition_exact_capacity_failure_fixture(
                        state_limit=250_000
                    ),
                )
            alternate_hard = deepcopy(hard)
            if isinstance(assignment, tuple):
                alternate_hard["source_quotas"] = [
                    10 if index in assignment else 9 for index in range(6)
                ]
            raise DirectMemberVisualSelectionError("hard quota", alternate_hard)

        with patch(
            "crosscut.pipeline._DIRECT_MEMBER_MAX_SHIFTED_BOUNDARY_COUNT",
            0,
        ), patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=choose_profile,
        ), self.assertRaises(DirectMemberVisualSelectionError) as raised:
            _official_target_fancam_direct_full_frame_segments(
                timing_draft=timing,
                eligible_indices=tuple(range(6)),
                eligible_sources=tuple(
                    Path(f"official-gaeul-{index}.mp4")
                    for index in range(6)
                ),
                common_master_start=0.0,
                source_timewarps=None,
                scene_cut_times=((),) * 6,
                source_metadata_contract=(
                    OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                ),
                descriptor_provider=lambda *_args: None,
            )

        diagnostics = raised.exception.diagnostics
        self.assertFalse(diagnostics["constraint_infeasibility_proven"])
        search = diagnostics["partition_search"]
        self.assertEqual(
            search["strict_extra_quota_assignment_search"][
                "exhausted_profile_count"
            ],
            1,
        )
        attempt = next(
            item
            for item in search["profiles_attempted"]
            if "strict_extra_quota_assignment_search" in item
        )
        certificate = attempt["strict_extra_quota_assignment_search"]
        self.assertTrue(certificate["complete_assignment_domain_evaluated"])
        self.assertTrue(certificate["all_alternatives_proven_hard_infeasible"])
        self.assertEqual(certificate["total_assignment_domain_count"], 6)
        self.assertEqual(
            [item["result"] for item in certificate["assignments_evaluated"]],
            ["hard_infeasible"] * 5,
        )

    def test_visual_split_exact_capacity_retry_is_once_then_fail_closed(
        self,
    ) -> None:
        duration = 192.0
        timing = _rotating_timing_draft(
            source_count=6,
            duration_seconds=duration,
            unit_seconds=duration / 200.0,
        )
        calls: list[tuple[int, int]] = []
        progress_events: list[dict[str, object]] = []

        def choose_profile(**kwargs: object) -> _VisualSelectionStub:
            spans = kwargs["canonical_spans"]
            config = kwargs["config"]
            assert isinstance(spans, tuple)
            assert isinstance(config, DirectMemberVisualConfig)
            calls.append(
                (len(spans), config.feasibility_lifeline_state_limit)
            )
            if len(spans) == 50:
                diagnostics = _direct_partition_failure_fixture(
                    span_count=50,
                    failed_segment_index=46,
                    frontier_boundary_index=45,
                )
                span = SimpleNamespace(
                    canonical_master_start=spans[46][0],
                    canonical_master_end=spans[46][1],
                )
                diagnostics["span_content_validity"] = [
                    _visual_frontier_span_diagnostic(
                        span,
                        segment_index=46,
                        source_count=6,
                    )
                ]
                raise DirectMemberVisualSelectionError("base", diagnostics)
            raise DirectMemberVisualSelectionError(
                "exact lifeline remains capacity bounded",
                _direct_partition_exact_capacity_failure_fixture(
                    state_limit=config.feasibility_lifeline_state_limit,
                ),
            )

        with patch(
            "crosscut.pipeline._DIRECT_MEMBER_MAX_SHIFTED_BOUNDARY_COUNT",
            0,
        ), patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=choose_profile,
        ) as select_visual, self.assertRaises(
            DirectMemberVisualSelectionError
        ) as raised:
            _official_target_fancam_direct_full_frame_segments(
                timing_draft=timing,
                eligible_indices=tuple(range(6)),
                eligible_sources=tuple(
                    Path(f"official-gaeul-{index}.mp4")
                    for index in range(6)
                ),
                common_master_start=0.0,
                source_timewarps=None,
                scene_cut_times=((),) * 6,
                source_metadata_contract=(
                    OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                ),
                descriptor_provider=lambda *_args: None,
                partition_progress_callback=progress_events.append,
            )

        self.assertEqual(select_visual.call_count, 4, calls)
        self.assertEqual(
            calls,
            [
                (50, 250_000),
                (51, 250_000),
                (51, 1_000_000),
                (50, 250_000),
            ],
        )
        diagnostics = raised.exception.diagnostics
        self.assertEqual(
            diagnostics["failure"],
            "direct_partition_selector_capacity_exhausted",
        )
        self.assertFalse(diagnostics["constraint_infeasibility_proven"])
        retry = diagnostics["partition_search"][
            "exact_lifeline_capacity_retry"
        ]
        self.assertEqual(retry["retry_count"], 1)
        self.assertEqual(retry["success_count"], 0)
        self.assertEqual(retry["capacity_exhausted_count"], 1)
        retried_attempt = next(
            item
            for item in diagnostics["partition_search"]["profiles_attempted"]
            if "exact_lifeline_retry" in item
        )
        self.assertEqual(
            retried_attempt["exact_lifeline_retry"]["result"],
            "capacity_exhausted",
        )
        retry_events = [
            item
            for item in progress_events
            if item["event"]
            == "direct_member_visual_split_exact_lifeline_retry"
        ]
        self.assertEqual(len(retry_events), 2)
        self.assertEqual(
            retry_events[-1]["final_explored_state_count"],
            1_000_000,
        )
        self.assertEqual(
            progress_events[-1][
                "exact_lifeline_capacity_retry_capacity_exhausted_count"
            ],
            1,
        )

    def test_direct_partition_fallback_all_fail_is_atomic_and_hash_compacted(
        self,
    ) -> None:
        duration = 192.0
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=duration / 200.0,
        )
        large_runtime_only_sentinel = "runtime-only-selector-evidence-" * 400
        failure = DirectMemberVisualSelectionError(
            "synthetic terminal hard conflict",
            {
                "failure": "validity_quota_path_exhausted",
                "constraint_infeasibility_proven": True,
                "failed_at_segment_index": 49,
                "hard_boundary_connectivity": {
                    "zero_local_nonsevere_transition_boundary_indices": [48],
                },
                "span_content_validity": [
                    {
                        "segment_index": 49,
                        "sources": [
                            {
                                "eligible_source_index": 3,
                                "valid": False,
                                "rejection_reasons": [
                                    "foreground_camera_equipment_or_subject_obstruction"
                                ],
                            }
                        ],
                    }
                ],
                "search": {
                    "exact_lifeline": {
                        "root_candidate_diagnostics": [
                            {
                                "eligible_source_index": 3,
                                "search_outcome": "root_predecessor_exhausted",
                            }
                        ]
                    }
                },
                "valid_count_by_source": [
                    {
                        "eligible_source_index": 3,
                        "quota": 5,
                        "valid_span_count": 44,
                        "invalid_span_count": 6,
                        "valid_segment_indices": list(range(44)),
                    }
                ],
                "large_runtime_only_sentinel": large_runtime_only_sentinel,
            },
        )
        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=failure,
        ) as select_visual, patch(
            "crosscut.pipeline.aligned_render_segment"
        ) as align_segment, self.assertRaises(
            DirectMemberVisualSelectionError
        ) as raised:
            _official_target_fancam_direct_full_frame_segments(
                timing_draft=timing,
                eligible_indices=tuple(range(10)),
                eligible_sources=tuple(
                    Path(f"official-gaeul-{index}.mp4") for index in range(10)
                ),
                common_master_start=0.0,
                source_timewarps=None,
                scene_cut_times=((),) * 10,
                descriptor_provider=lambda *_args: None,
            )

        self.assertEqual(select_visual.call_count, 3)
        align_segment.assert_not_called()
        diagnostics = raised.exception.diagnostics
        self.assertEqual(
            diagnostics["failure"],
            "direct_partition_profiles_exhausted",
        )
        self.assertTrue(diagnostics["constraint_infeasibility_proven"])
        self.assertFalse(diagnostics["visual_gates_relaxed"])
        search = diagnostics["partition_search"]
        self.assertFalse(search["render_started"])
        self.assertEqual(len(search["profiles_attempted"]), 3)
        compact = search["profiles_attempted"][0]["diagnostics"]
        self.assertEqual(
            compact["storage_contract"],
            "hash_bound_scalar_partition_failure_v1",
        )
        self.assertFalse(compact["full_diagnostics_persisted"])
        self.assertNotIn("span_content_validity", compact)
        self.assertEqual(
            compact["valid_count_by_source"],
            [
                {
                    "eligible_source_index": 3,
                    "quota": 5,
                    "valid_span_count": 44,
                    "invalid_span_count": 6,
                }
            ],
        )
        self.assertNotIn(
            "valid_segment_indices",
            json.dumps(compact, ensure_ascii=False),
        )
        self.assertEqual(
            compact["summary"]["rejection_reason_counts"],
            {"foreground_camera_equipment_or_subject_obstruction": 1},
        )
        fingerprint = compact["diagnostics_content_fingerprint"]
        self.assertEqual(len(fingerprint["sha256"]), 64)
        self.assertGreater(fingerprint["canonical_size_bytes"], 0)
        encoded_search = json.dumps(search, ensure_ascii=False)
        self.assertNotIn(large_runtime_only_sentinel, encoded_search)

    def test_direct_partition_does_not_retry_unproven_capacity_exhaustion(self) -> None:
        timing = _rotating_timing_draft(
            duration_seconds=200.0,
            unit_seconds=1.0,
        )
        failure = DirectMemberVisualSelectionError(
            "synthetic capacity exhaustion",
            {
                "failure": "search_capacity_exhausted",
                "constraint_infeasibility_proven": False,
                "visual_gates_relaxed": False,
            },
        )
        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            side_effect=failure,
        ) as select_visual, self.assertRaises(
            DirectMemberVisualSelectionError
        ) as raised:
            _official_target_fancam_direct_full_frame_segments(
                timing_draft=timing,
                eligible_indices=tuple(range(10)),
                eligible_sources=tuple(
                    Path(f"official-gaeul-{index}.mp4") for index in range(10)
                ),
                common_master_start=0.0,
                source_timewarps=None,
                scene_cut_times=((),) * 10,
                descriptor_provider=lambda *_args: None,
            )

        self.assertEqual(select_visual.call_count, 1)
        self.assertFalse(
            raised.exception.diagnostics["constraint_infeasibility_proven"]
        )
        self.assertFalse(raised.exception.diagnostics["visual_gates_relaxed"])

    def test_partition_profiles_are_bounded_final_first_and_never_combined(
        self,
    ) -> None:
        duration = 192.0
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=duration / 200.0,
        )
        base = _direct_full_frame_base_partition_profile(timing.segments)
        diagnostics = {
            "failed_at_segment_index": 31,
            "hard_boundary_connectivity": {
                "zero_local_nonsevere_transition_boundary_indices": [7, 42],
            },
            "search": {
                "exact_lifeline": {
                    "root_candidate_diagnostics": [
                        {
                            "final_candidate_source_index": 3,
                            "search_outcome": "root_predecessor_exhausted",
                        }
                    ]
                }
            },
        }
        bottlenecks = _direct_partition_failure_bottlenecks(
            diagnostics,
            span_count=len(base.spans),
        )
        profiles = _direct_full_frame_shifted_partition_profiles(
            timing.segments,
            base,
            bottlenecks,
        )

        self.assertEqual(bottlenecks, (48, 42))
        self.assertEqual(
            [profile.profile_id for profile in profiles],
            [
                "boundary-48-5-plus-3",
                "boundary-48-3-plus-5",
                "boundary-42-5-plus-3",
                "boundary-42-3-plus-5",
            ],
        )
        self.assertEqual(base.unit_counts, (4,) * 50)
        self.assertEqual(len(profiles) + 1, 5)
        for profile in profiles:
            changed = [
                index
                for index, (left, right) in enumerate(
                    zip(base.unit_counts, profile.unit_counts)
                )
                if left != right
            ]
            self.assertEqual(len(changed), 2)
            self.assertEqual(changed[1], changed[0] + 1)
            self.assertEqual(sum(profile.unit_counts), 200)
            self.assertEqual(len(profile.spans), 50)

    def test_direct_full_frame_rechecks_duration_balance_from_render_spans(self) -> None:
        timing = _rotating_timing_draft()
        durations = (6.0,) + (34.0 / 9.0,) * 9
        cursor = 0.0
        spans = []
        for index, duration in enumerate(durations):
            spans.append(
                _DirectFullFrameSpan(
                    canonical_master_start=cursor,
                    canonical_master_end=cursor + duration,
                    timing_segment_count=1,
                    estimated_beat_count=duration * 2.0,
                    start_boundary_reasons=(
                        "range_edge" if index == 0 else "strong_beat",
                    ),
                    end_boundary_reasons=(
                        "range_edge"
                        if index == len(durations) - 1
                        else "strong_beat",
                    ),
                )
            )
            cursor += duration
        selection = _VisualSelectionStub(
            tuple(range(10)),
            10,
            usage_policy="exact_balanced_quota_v1",
        )

        bound_selection = selection.bind(
            tuple(
                (
                    span.canonical_master_start,
                    span.canonical_master_end,
                )
                for span in spans
            ),
            DirectMemberVisualConfig(),
        )
        with patch(
            "crosscut.pipeline._select_direct_member_partition_profile",
            return_value=(
                SimpleNamespace(spans=tuple(spans)),
                bound_selection,
                {},
            ),
        ), self.assertRaisesRegex(RuntimeError, "source-duration balance"):
            _official_target_fancam_direct_full_frame_segments(
                timing_draft=timing,
                eligible_indices=tuple(range(10)),
                eligible_sources=tuple(
                    Path(f"official-gaeul-{index}.mp4")
                    for index in range(10)
                ),
                common_master_start=0.0,
                source_timewarps=None,
                scene_cut_times=((),) * 10,
                descriptor_provider=lambda *_args: None,
            )

    def test_direct_selection_failure_is_atomic_persisted_and_reported(self) -> None:
        diagnostics = {
            "failure": "content_validity_quota_precheck_failed",
            "quota_shortfalls": [
                {
                    "eligible_source_index": 2,
                    "quota": 5,
                    "valid_span_count": 4,
                }
            ],
            "empty_segment_indices": [],
            "span_content_validity": [
                {
                    "segment_index": 7,
                    "sources": [
                        {
                            "eligible_source_index": 1,
                            "valid": False,
                            "rejection_reasons": [
                                "source_native_scene_cut_inside_rendered_span"
                            ],
                        }
                    ],
                }
            ],
        }
        error = DirectMemberVisualSelectionError(
            "balanced direct source quotas are impossible",
            diagnostics,
        )
        reporter = _MemoryReporter()

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            reported = _report_direct_member_selection_failure(
                error,
                output_directory=output,
                base_filename="IVE-I-AM-test",
                target_member="가을",
                eligible_indices=(3, 7, 9),
                reporter=reporter,  # type: ignore[arg-type]
            )

            error_events = [
                payload
                for event_type, payload in reporter.events
                if event_type == "error"
            ]
            self.assertEqual(len(error_events), 1)
            diagnostics_path = Path(
                str(error_events[0]["diagnostics_path"])
            )
            self.assertTrue(diagnostics_path.is_file())
            self.assertEqual(tuple(output.glob(".*.tmp")), ())
            self.assertIn(str(diagnostics_path), str(reported))
            persisted = json.loads(diagnostics_path.read_text(encoding="utf-8"))

        self.assertEqual(persisted["diagnostics"], diagnostics)
        self.assertEqual(
            persisted["eligible_request_source_indices"],
            [3, 7, 9],
        )
        summary = persisted["summary"]
        self.assertEqual(summary["affected_segment_indices"], [7])
        self.assertEqual(summary["affected_eligible_source_indices"], [1, 2])
        self.assertEqual(
            summary["rejection_reason_counts"],
            {"source_native_scene_cut_inside_rendered_span": 1},
        )
        self.assertFalse(
            persisted["privacy"]["face_appearance_embeddings_persisted"]
        )

    def test_direct_selection_failure_report_hard_cap_uses_hash_bound_fallback(
        self,
    ) -> None:
        self.assertEqual(
            _DIRECT_MEMBER_FAILURE_REPORT_MAX_BYTES,
            64 * 1024 * 1024,
        )
        sentinel = "runtime-only-oversized-selector-diagnostics-" * 8_000
        diagnostics = {
            "failure": "validity_quota_path_exhausted",
            "constraint_infeasibility_proven": True,
            "failed_at_segment_index": 41,
            "span_content_validity": [
                {
                    "segment_index": 41,
                    "sources": [
                        {
                            "eligible_source_index": 2,
                            "valid": False,
                            "rejection_reasons": [
                                "foreground_camera_equipment_or_subject_obstruction"
                            ],
                        }
                    ],
                }
            ],
            "runtime_only_payload": sentinel,
        }
        error = DirectMemberVisualSelectionError(
            "synthetic oversized selector evidence",
            diagnostics,
        )
        reporter = _MemoryReporter()
        hard_cap = 32 * 1024

        with tempfile.TemporaryDirectory() as temp, patch(
            "crosscut.pipeline._DIRECT_MEMBER_FAILURE_REPORT_MAX_BYTES",
            hard_cap,
        ):
            output = Path(temp)
            _report_direct_member_selection_failure(
                error,
                output_directory=output,
                base_filename="IVE-BLACKHOLE-test",
                target_member="가을",
                eligible_indices=(1, 2, 3, 4, 5, 6),
                reporter=reporter,  # type: ignore[arg-type]
            )
            report_path = next(output.glob("*-direct-selection-failure.json"))
            report_bytes = report_path.read_bytes()
            persisted = json.loads(report_bytes)
            self.assertLessEqual(len(report_bytes), hard_cap)
            self.assertEqual(tuple(output.glob(".*.tmp")), ())

        self.assertNotIn(sentinel.encode("utf-8"), report_bytes)
        storage = persisted["report_storage"]
        self.assertTrue(storage["bounded_fallback_used"])
        self.assertEqual(storage["hard_cap_bytes"], hard_cap)
        oversized = storage["oversized_report_fingerprint"]
        self.assertGreater(oversized["serialized_size_bytes"], hard_cap)
        self.assertEqual(len(oversized["sha256"]), 64)
        persisted_diagnostics = persisted["diagnostics"]
        self.assertNotIn("span_content_validity", persisted_diagnostics)
        self.assertEqual(
            persisted_diagnostics["latest_failure_summary"][
                "rejection_reason_counts"
            ],
            {"foreground_camera_equipment_or_subject_obstruction": 1},
        )
        fingerprint = persisted_diagnostics[
            "original_diagnostics_content_fingerprint"
        ]
        self.assertEqual(len(fingerprint["sha256"]), 64)
        self.assertGreater(fingerprint["canonical_size_bytes"], hard_cap)

    def test_local_explicit_member_anchors_preserve_tracking_for_portrait_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = tuple(root / f"official-gaeul-{index}.mp4" for index in range(10))
            for source in sources:
                source.touch()
            anchors = tuple(
                MemberAnchorRequest(
                    source_index=index,
                    time_seconds=1.0,
                    bbox_xyxy=(0.2, 0.1, 0.6, 0.9),
                )
                for index in range(10)
            )
            request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "member",
                    "aspect_ratio": "9:16",
                    "target_member": "가을",
                    "source_mode": "local",
                    "sources": [str(source) for source in sources],
                    "member_anchors": [anchor.as_dict() for anchor in anchors],
                    "output_directory": str(root / "out"),
                }
            )

            self.assertIsNone(
                _automatic_member_direct_full_frame_ratios(
                    request,
                    tuple(_portrait_analysis_media(source) for source in sources),
                    tuple(range(10)),
                )
            )

    def test_automatic_portrait_member_allows_four_to_ten_aligned_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = tuple(root / f"official-gaeul-{index}.mp4" for index in range(10))
            for source in sources:
                source.touch()
            request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "member",
                    "aspect_ratio": "9:16",
                    "target_member": "가을",
                    "source_mode": "automatic",
                    "sources": [],
                    "output_directory": str(root / "out"),
                }
            ).with_materialized_sources(sources, member_anchors=())

            ratios = _automatic_member_direct_full_frame_ratios(
                request,
                tuple(_portrait_analysis_media(source) for source in sources),
                tuple(range(4)),
            )
            self.assertIsNotNone(ratios)
            self.assertEqual(len(ratios or ()), 4)

            with self.assertRaisesRegex(ValueError, "between 4 and 10"):
                _automatic_member_direct_full_frame_ratios(
                    request,
                    tuple(_portrait_analysis_media(source) for source in sources),
                    tuple(range(3)),
                )

    def test_automatic_portrait_member_bad_aspect_fails_closed_before_fallback(self) -> None:
        cases = (
            ("landscape", 1920, 1080, "lossless-fill contract failed"),
            ("dimensions", None, 1920, "dimensions unavailable"),
        )
        for label, bad_width, bad_height, expected_error in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                sources = tuple(
                    root / f"official-gaeul-{index}.mp4" for index in range(10)
                )
                for source in sources:
                    source.touch()
                request = AnalysisRequest.from_dict(
                    {
                        "song": "I AM",
                        "edit_mode": "member",
                        "aspect_ratio": "9:16",
                        "target_member": "가을",
                        "source_mode": "automatic",
                        "sources": [],
                        "output_directory": str(root / "out"),
                    }
                ).with_materialized_sources(sources, member_anchors=())

                def media_for(source: Path) -> MediaInfo:
                    info = _portrait_analysis_media(source)
                    if source.name != sources[4].name:
                        return info
                    stream = info.video_streams[0]
                    return MediaInfo(
                        path=info.path,
                        format_name=info.format_name,
                        duration_seconds=info.duration_seconds,
                        start_time_seconds=info.start_time_seconds,
                        size_bytes=info.size_bytes,
                        bit_rate=info.bit_rate,
                        audio_streams=info.audio_streams,
                        video_streams=(
                            VideoStreamInfo(
                                index=stream.index,
                                codec_name=stream.codec_name,
                                width=bad_width,
                                height=bad_height,
                                frame_rate=stream.frame_rate,
                                duration_seconds=stream.duration_seconds,
                                bit_rate=stream.bit_rate,
                            ),
                        ),
                        tags=info.tags,
                    )

                with patch(
                    "crosscut.pipeline.probe_media",
                    side_effect=media_for,
                ), patch(
                    "crosscut.pipeline._align_sequentially"
                ) as align, patch(
                    "crosscut.pipeline.track_anchored_member"
                ) as track_member, patch(
                    "crosscut.pipeline.build_member_draft"
                ) as build_cropped_member:
                    with self.assertRaisesRegex(
                        ValueError,
                        expected_error,
                    ):
                        analyze(request, _MemoryReporter())  # type: ignore[arg-type]

                align.assert_not_called()
                track_member.assert_not_called()
                build_cropped_member.assert_not_called()

    def test_member_review_first_quality_miss_renders_once_and_reaches_final_review_ready(self) -> None:
        self.test_automatic_portrait_member_uses_direct_full_frame_without_identity_anchor(_quality_miss=True)

    def test_member_default_dissolve_missing_camera_keeps_first_plan_without_reselection(self):
        self.test_automatic_portrait_member_uses_direct_full_frame_without_identity_anchor(
            _quality_miss=True, _camera_failure="joint", _dissolve_requested=True)

    def test_member_dense_camera_failure_keeps_first_plan_without_reselection(self):
        self.test_automatic_portrait_member_uses_direct_full_frame_without_identity_anchor(
            _quality_miss=True, _camera_failure="dense", _dissolve_requested=True)

    def test_member_rejection_journal_survives_technical_proxy_failure(self):
        self.test_automatic_portrait_member_uses_direct_full_frame_without_identity_anchor(
            _quality_miss=True, _camera_failure='dense', _dissolve_requested=True, _proxy_failure=True)

    def test_member_successful_rotation_reaches_final_render_and_exposure_with_dissolve_off(self):
        # Camera generation/pixels have separate real tests. This exercises the
        # complete analyze caller and actual final exposure controller.
        self.test_automatic_portrait_member_uses_direct_full_frame_without_identity_anchor(
            _quality_miss=True, _roll_degrees=1.2)

    def test_automatic_portrait_member_uses_direct_full_frame_without_identity_anchor(
            self, _quality_miss=False, _camera_failure=None, _dissolve_requested=False, _proxy_failure=False,
            _roll_degrees=0.) -> None:
        from crosscut.pipeline import _enforce_automatic_member_post_render_qa as actual_member_gate
        from crosscut.member_joint_camera_preflight import MemberJointCameraPreflight
        original_camera_review = MemberJointCameraPreflight.review
        def camera_review(preflight, **kwargs):
            if _camera_failure == "joint":
                return {"passed": False, "reason": "required_camera_audit_missing"}
            return original_camera_review(preflight, **kwargs)
        camera_patcher = patch("crosscut.member_joint_camera_preflight.MemberJointCameraPreflight.review", camera_review)
        camera_patcher.start()
        self.addCleanup(camera_patcher.stop)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = tuple(root / f"official-gaeul-{index}.mp4" for index in range(10))
            for source in sources:
                source.touch()
            request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "member",
                    "aspect_ratio": "9:16",
                    "target_member": "가을",
                    "source_mode": "automatic",
                    "sources": [],
                    "member_pose_dissolve_enabled": _dissolve_requested,
                    "output_directory": str(root / "out"),
                }
            ).with_materialized_sources(sources, member_anchors=())
            alignment = AlignmentResult(
                reference=str(sources[0]),
                sample_rate=4_000,
                alignments=tuple(
                    SourceAlignment(
                        source=str(source),
                        is_reference=index == 0,
                        offset_samples=0,
                        offset_seconds=0.0,
                        score=1.0 if index == 0 else 0.9,
                        overlap_samples=160_000,
                        overlap_seconds=40.0,
                        score_margin=1.0 if index == 0 else 0.2,
                        overlap_ratio=1.0,
                    )
                    for index, source in enumerate(sources)
                ),
            )
            timing = _rotating_timing_draft()
            timewarps = tuple(
                _linear_timewarp(duration=40.0, rate=0.996) for _ in sources
            )
            visual_selection = _VisualSelectionStub(
                (*tuple(range(10)), 0, 1),
                10,
            )

            class DescriptorProviderStub:
                def __call__(self, *_args: object) -> None:
                    return None

                @staticmethod
                def cache_diagnostics() -> dict[str, int]:
                    return {
                        "descriptor_request_count": 0,
                        "descriptor_cache_hit_count": 0,
                        "descriptor_decode_count": 0,
                        "descriptor_cache_entry_count": 0,
                    }

                def close(self) -> None:
                    return None

            class DenseFaceProviderStub:
                def __init__(self) -> None:
                    self.sample_calls: list[tuple[int, float]] = []
                    self.closed = False
                    self.pose = SimpleNamespace(model_path=root / "pose.onnx")
                    self.encoder = SimpleNamespace(model_path=root / "body.onnx")
                    self.body_execution_device = "cpu"

                def sample(
                    self,
                    source_index: int,
                    source_local_seconds: float,
                ) -> None:
                    self.sample_calls.append(
                        (source_index, source_local_seconds)
                    )
                    return None

                def paired_samples(
                    self,
                    outgoing_source_index: int,
                    outgoing_source_local_seconds: float,
                    incoming_source_index: int,
                    incoming_source_local_seconds: float,
                ) -> tuple[None, None]:
                    self.sample(
                        outgoing_source_index,
                        outgoing_source_local_seconds,
                    )
                    self.sample(
                        incoming_source_index,
                        incoming_source_local_seconds,
                    )
                    return None, None

                def begin_focal_boundary_reference_capture(self) -> None:
                    return None

                def focal_boundary_reference_capture(
                    self,
                ) -> tuple[tuple[int, int, None, None], ...]:
                    return tuple(
                        (left, right, None, None)
                        for left, right in zip(
                            visual_selection.source_indices,
                            visual_selection.source_indices[1:],
                        )
                        if left != right
                    )

                def recover_exact_focal_pair_from_registered_references(
                    self,
                    *_args: object,
                ) -> None:
                    return None

                def diagnostics(self) -> dict[str, int | bool | str]:
                    return {
                        "backend": "synthetic_dense_face_provider",
                        "request_count": len(self.sample_calls),
                        "cache_hit_count": 0,
                        "decode_count": len(self.sample_calls),
                        "cache_entry_count": len(self.sample_calls),
                        "pixels_persisted": False,
                        "face_embeddings_computed": False,
                    }

                def close(self) -> None:
                    self.closed = True

            direct_track = EditorialTransformTrack(
                (
                    EditorialTransformKeyframe(0.0, 1.0),
                    EditorialTransformKeyframe(0.75, 1.12 if _roll_degrees else 1.06,roll_degrees=_roll_degrees),
                ),
                reason="official_fancam_dense_face_scale_position_match",
            )
            direct_motion_plan = EditorialMotionPlan(
                segments=tuple(
                    EditorialMotionSegment(
                        segment_index,
                        source_index,
                        direct_track if segment_index == 0 else None,
                    )
                    for segment_index, source_index in enumerate(
                        visual_selection.source_indices
                    )
                ),
                boundary_match_count=1,
                zoom_event_count=1,
                pan_event_count=0,
                roll_event_count=int(bool(_roll_degrees)),
                blocked_dissolve_segment_count=0,
                analysis_basis=(
                    "official_target_fancam_unambiguous_face_"
                    "dense_output_frame_review_v1"
                ),
            )
            dense_face_provider = DenseFaceProviderStub()
            member_cut_local_sync_payload = {
                "activation": MEMBER_CUT_LOCAL_SYNC_ACTIVATION,
                "required": True,
                "retained_cross_source_boundary_count": 9,
                "measured_boundary_count": 9,
                "unknown_boundary_count": 0,
                "residual_p95_frames": 0.1,
                "passed": True,
            }

            def build_direct_motion(**kwargs: object) -> EditorialMotionPlan:
                geometry_provider = kwargs["geometry_provider"]
                assert callable(geometry_provider)
                geometry_provider(1, 4.0)
                return direct_motion_plan

            class RhythmStub:
                bpm = 120.0
                beats = (object(),)
                phrase_candidates: tuple[object, ...] = ()
                beat_times_seconds = tuple(index * 0.5 for index in range(81))

                @staticmethod
                def to_dict() -> dict[str, object]:
                    return {}

            def pass_member_delivery_gate(**kwargs: object) -> Path:
                manifest = kwargs["manifest"]
                manifest_path = kwargs["manifest_path"]
                assert isinstance(manifest, dict)
                assert isinstance(manifest_path, Path)
                render_payload = manifest["render"]
                direct_payload = manifest["member_direct_full_frame"]
                assert isinstance(render_payload, dict)
                assert isinstance(direct_payload, dict)
                assert manifest["artifact_role"] == "pending_qa"
                assert render_payload["artifact_role"] == "pending_qa"
                assert render_payload["direct_full_frame"] is True
                assert render_payload["cut_local_sync"] == (
                    member_cut_local_sync_payload
                )
                eligible = direct_payload["eligible_request_source_indices"]
                assert eligible == list(range(10))
                if _quality_miss:
                    # Full analyze() orchestration plus the actual final exposure
                    # controller. Media/model work is injected in this code test.
                    from crosscut.member_post_render_qa import AutomaticMemberPostRenderArtifactSet, fingerprint_automatic_member_artifacts
                    artifacts = AutomaticMemberPostRenderArtifactSet.from_manifest(manifest_path, manifest)
                    artifacts.render_plan_path.write_text('{}')
                    artifacts.video_path.write_bytes(b'synthetic member output')
                    def evaluate(bound, **unused):
                        fingerprints = fingerprint_automatic_member_artifacts(bound)
                        report_payload = {'passed': False, 'verdict': 'FAIL',
                            'issues': [{'code': 'member_rendered_continuity_not_pass'}],
                            'issue_count': 1, 'metrics': {'rendered_continuity': member_proxy_continuity()},
                            'input_artifact_fingerprints': fingerprints, 'final_artifact_fingerprints': fingerprints}
                        report = SimpleNamespace(passed=False, verdict='FAIL',
                            input_artifact_fingerprints=fingerprints, final_artifact_fingerprints=fingerprints,
                            render_receipt={'all_eligible_sources_pixel_proven': True}, to_dict=lambda: report_payload)
                        bound.default_report_path.write_text(json.dumps(report_payload))
                        return report, bound.default_report_path
                    with patch('crosscut.pipeline.evaluate_and_write_automatic_member_post_render_qa', side_effect=evaluate):
                        return actual_member_gate(**kwargs)
                receipt = {
                    "contract": MEMBER_RENDER_RECEIPT_CONTRACT,
                    "eligible_source_indices": eligible,
                    "sources": [
                        {
                            "request_source_index": source_index,
                            "safe_interior_structure_sample_count": 1,
                            "passed": True,
                        }
                        for source_index in eligible
                    ],
                    "all_eligible_sources_pixel_proven": True,
                    "artifact_content_stable": True,
                    "source_file_identities_stable": True,
                    "passed": True,
                }
                qa = {
                    "contract": AUTOMATIC_IVE_MEMBER_POST_RENDER_QA_CONTRACT,
                    "required": True,
                    "verdict": "PASS",
                    "report_passed": True,
                    "content_binding_passed": True,
                    "render_receipt": receipt,
                    "passed": True,
                }
                exposure = {
                    "contract": AUTOMATIC_IVE_MEMBER_APP_RESULT_EXPOSURE_CONTRACT,
                    "video_output_exposed": True,
                    "delivery_completion_message_allowed": True,
                }
                manifest["artifact_role"] = "delivery"
                manifest["app_result_exposure"] = exposure
                manifest["post_render_qa"] = qa
                render_payload["artifact_role"] = "delivery"
                render_payload["app_result_exposure"] = exposure
                render_payload["post_render_qa"] = qa
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
                )
                report_path = manifest_path.with_name(
                    manifest_path.name.replace(
                        "-analysis.json", "-gaeul-member-post-render-qa.json"
                    )
                )
                report_path.write_text(
                    json.dumps({"verdict": "PASS", "render_receipt": receipt}),
                    encoding="utf-8",
                )
                return report_path

            def member_proxy_continuity():
                from tests.member_continuity_fixture import member_continuity_report
                payload = member_continuity_report()
                payload['passed'] = not _quality_miss
                payload['failure_reasons'] = ['rendered_pose_p95_above_floor'] if _quality_miss else []
                payload['metrics']['pose_mismatch_p95'] = .27 if _quality_miss else 0.
                for boundary in payload['boundaries']:
                    boundary['pose_mismatch'] = .27 if _quality_miss else 0.
                return payload

            def pass_member_proxy(
                plan: RenderPlan,
                _output_path: Path,
                **_kwargs: object,
            ) -> tuple[RenderPlan, DirectMemberRenderedProxyPreflight]:
                if _proxy_failure:
                    journals = list(request.output_directory.glob('*-member-first-result-journal.json'))
                    self.assertEqual(len(journals), 1)
                    journal = json.loads(journals[0].read_text())
                    self.assertEqual(journal['rejected_attempt_count'], 1)
                    self.assertIn('dense focal geometry unavailable', journal['attempts'][0]['error'])
                    raise RuntimeError('synthetic technical proxy failure')
                preflight = DirectMemberRenderedProxyPreflight(
                    {
                        "schema_version": 1,
                        "passed": True,
                        "member_output_quality_passed": True,
                        "member_output_quality_failure_reasons": [],
                        "direct_member_rendered_continuity_passed": True,
                        "repair_target_boundary_indices": [],
                        "generic_repair_target_boundary_indices": [],
                        "continuity_repair_target_boundary_indices": [],
                        "rendered_proxy_review": {
                            "schema_version": 3, "geometry_provider_failure_count": 0,
                            "reviewed_boundary_count": 0, "boundaries": [],
                            "introduced_cross_source_technical_issue_reasons": [],
                            "passed": True,
                            "weighted_score": 1.0,
                        },
                        "rendered_continuity": {
                            "passed": True,
                            "failure_boundary_indices": {},
                            "boundaries": [],
                            "metrics": {
                                "attention_position_mismatch_p95": 0.0,
                                "attention_scale_mismatch_p95": 0.0,
                                "pose_mismatch_p95": 0.0,
                                "silhouette_mismatch_p95": 0.0,
                            },
                            "thresholds": {
                                "maximum_attention_position_mismatch_p95": 1.0,
                                "maximum_attention_scale_mismatch_p95": 1.0,
                                "maximum_pose_mismatch_p95": 1.0,
                                "maximum_silhouette_mismatch_p95": 1.0,
                            },
                        },
                    }
                )

                payload = dict(preflight.payload)
                payload['passed'] = not _quality_miss
                payload['direct_member_rendered_continuity_passed'] = not _quality_miss
                payload['rendered_continuity'] = member_proxy_continuity()
                return plan, DirectMemberRenderedProxyPreflight(payload)

            # This fixture verifies the full parent pipeline. The actual IPC
            # and worker-owned provider are exercised separately with real processes.
            spec_patch = patch('crosscut.member_camera_worker.MemberCameraProviderSpec.from_provider',
                return_value=SimpleNamespace(validate_files=lambda: None))
            worker_patch = patch('crosscut.member_camera_worker.propose_member_camera_in_worker',
                side_effect=lambda provider_spec, arguments: __import__('crosscut.pipeline', fromlist=['_build_member_camera_with_provider'])._build_member_camera_with_provider(
                    dense_face_provider=dense_face_provider, **arguments))
            spec_patch.start()
            worker_patch.start()
            self.addCleanup(spec_patch.stop)
            self.addCleanup(worker_patch.stop)
            professional_gate_metrics = {
                "visible_switches_per_minute": 16.5,
                "pose_mismatch_p95": 0.0,
                "silhouette_mismatch_p95": 0.0,
                "editorial_transform_segment_count": 1,
            }
            professional_gate = SimpleNamespace(
                passed=not _quality_miss,
                failure_reasons=('pose_mismatch_p95_above_professional_floor',) if _quality_miss else (),
                metrics=professional_gate_metrics,
                to_dict=lambda: {
                    "contract": "synthetic_direct_member_professional_gate",
                    "passed": not _quality_miss,
                    "failure_reasons": ["pose_mismatch_p95_above_professional_floor"] if _quality_miss else [],
                    "metrics": dict(professional_gate_metrics),
                },
            )

            with patch(
                "crosscut.pipeline.probe_media",
                side_effect=lambda source: _portrait_analysis_media(source),
            ), patch(
                "crosscut.pipeline._align_sequentially",
                return_value=(alignment, None),
            ), patch(
                "crosscut.pipeline._estimate_source_timewarps",
                return_value=(timewarps, ()),
            ), patch(
                "crosscut.pipeline.analyze_rhythm",
                return_value=RhythmStub(),
            ), patch(
                "crosscut.pipeline.detect_scene_changes_in_video_cached",
                return_value=((), {"detection_policy": "synthetic_test"}),
            ), patch(
                "crosscut.pipeline.build_group_draft",
                return_value=timing,
            ), patch(
                "crosscut.pipeline.select_balanced_direct_member_sources",
                side_effect=_bound_visual_selection(visual_selection),
            ) as select_member_mock, patch(
                "crosscut.pipeline._VideoDescriptorProvider",
                return_value=DescriptorProviderStub(),
            ), patch(
                "crosscut.source_first_member_session."
                "open_source_first_member_session",
                return_value=(dense_face_provider, DescriptorProviderStub()),
            ) as source_first_session, patch(
                "crosscut.pipeline.build_direct_member_editorial_motion_plan",
                side_effect=ValueError("dense focal geometry unavailable") if _camera_failure == "dense" else build_direct_motion,
            ) as build_direct_motion_mock, patch(
                "crosscut.pipeline.render_and_review_direct_member_proxy",
                side_effect=pass_member_proxy,
            ) as member_proxy_mock, patch(
                "crosscut.pipeline."
                "evaluate_direct_member_professional_pre_render_gate",
                return_value=professional_gate,
            ), patch(
                "crosscut.pipeline.build_member_editorial_motion_plan"
            ) as build_member_motion, patch(
                "crosscut.pipeline.track_anchored_member"
            ) as track_member, patch(
                "crosscut.pipeline.build_member_draft"
            ) as build_cropped_member, patch(
                "crosscut.pipeline.analyze_cut_local_sync",
                return_value=member_cut_local_sync_payload,
            ) as analyze_member_cut_local_sync, patch(
                "crosscut.pipeline.write_plan"
            ), patch(
                "crosscut.pipeline.render",
                return_value="h264_videotoolbox",
            ) as render_video, patch(
                "crosscut.pipeline._enforce_automatic_member_post_render_qa",
                side_effect=pass_member_delivery_gate,
            ) as member_delivery_gate:
                reporter = _MemoryReporter()
                if _proxy_failure:
                    self.assertRaisesRegex(RuntimeError, 'synthetic technical proxy failure', analyze, request, reporter)
                    member_delivery_gate.assert_not_called()
                    render_video.assert_not_called()
                    return
                artifact = analyze(request, reporter)  # type: ignore[arg-type]

            track_member.assert_not_called()
            build_cropped_member.assert_not_called()
            build_member_motion.assert_not_called()
            build_direct_motion_mock.assert_called_once()
            select_member_mock.assert_called_once()
            member_proxy_mock.assert_called_once()
            render_video.assert_called_once()
            source_first_session.assert_called_once()
            member_delivery_gate.assert_called_once()
            analyze_member_cut_local_sync.assert_called_once()
            if _camera_failure != "dense":
                self.assertEqual(len(dense_face_provider.sample_calls), 1)
                self.assertEqual(dense_face_provider.sample_calls[0][0], 1)
                self.assertAlmostEqual(dense_face_provider.sample_calls[0][1], timewarps[1].master_to_source(4.0))
            self.assertTrue(dense_face_provider.closed)
            partition_events = [
                payload
                for event_type, payload in reporter.events
                if event_type == "direct_member_partition_profile"
            ]
            self.assertEqual(
                [payload["phase"] for payload in partition_events],
                ["started", "completed", "search_completed"],
            )
            self.assertEqual(
                partition_events[-1]["event"],
                "direct_member_partition_search_completed",
            )
            self.assertEqual(partition_events[-1]["result"], "feasible")
            plan = render_video.call_args.args[0]
            cut_local_call = analyze_member_cut_local_sync.call_args
            self.assertIs(cut_local_call.args[0], plan)
            self.assertEqual(
                cut_local_call.kwargs["activation"],
                MEMBER_CUT_LOCAL_SYNC_ACTIVATION,
            )
            self.assertIs(cut_local_call.kwargs["source_timewarps"], timewarps)
            self.assertEqual(cut_local_call.kwargs["timewarp_payloads"], ())
            self.assertEqual(plan.member_pose_dissolve_requested, _dissolve_requested)
            self.assertFalse(plan.member_pose_dissolve_effective)
            self.assertEqual(plan.member_pose_dissolve_count, 0)
            self.assertEqual(len(plan.segments), 12)
            self.assertEqual(
                [item.source_index for item in plan.segments],
                [*range(10), 0, 1],
            )
            self.assertTrue(all(item.pan_crop is None for item in plan.segments))
            self.assertTrue(all(item.crop is None for item in plan.segments))
            self.assertIs(plan.segments[0].editorial_transform, None if _camera_failure else direct_track)
            self.assertTrue(
                all(
                    item.editorial_transform is None
                    for item in plan.segments[1:]
                )
            )
            self.assertAlmostEqual(plan.segments[0].source_end, 1.992)

            manifest = artifact.manifest
            self.assertEqual(
                manifest["render"]["cut_local_sync"],
                member_cut_local_sync_payload,
            )
            self.assertEqual(manifest["artifact_role"], "review_ready" if _quality_miss else "delivery")
            self.assertTrue(
                manifest["app_result_exposure"]["video_output_exposed"]
            )
            self.assertEqual(manifest["post_render_qa"]["passed"], not _quality_miss)
            if _quality_miss:
                self.assertEqual(manifest['post_render_qa']['review_first_disposition']['status'], 'REVIEW_READY')
                self.assertFalse(manifest['app_result_exposure']['delivery_completion_message_allowed'])
            self.assertTrue(
                manifest["post_render_qa"]["render_receipt"][
                    "all_eligible_sources_pixel_proven"
                ]
            )
            self.assertTrue(
                any(path.suffix.lower() == ".mp4" for path in artifact.output_paths)
            )
            self.assertTrue(
                any(
                    path.name.endswith("-member-post-render-qa.json")
                    for path in artifact.output_paths
                )
            )
            self.assertEqual(manifest["member_tracking"], [])
            self.assertEqual(manifest["member_auto_anchors"], [])
            self.assertIsNone(manifest["member_anchor_trim"])
            self.assertIsNone(manifest["member_draft"])
            direct = manifest["member_direct_full_frame"]
            self.assertTrue(direct["enabled"])
            self.assertEqual(
                direct["minimum_required_distinct_source_count"],
                10,
            )
            self.assertEqual(
                direct["required_used_request_source_indices"],
                list(range(10)),
            )
            self.assertEqual(direct["member_pose_dissolve_requested"], _dissolve_requested)
            if _camera_failure:
                controller = direct["cadence"]["first_result_controller"]
                self.assertTrue(controller["baseline_retained"])
                self.assertEqual(controller["optional_attempt_count"], 1)
                self.assertEqual(controller["rejected_attempt_count"], 1)
                self.assertEqual(controller["selected_authority"], "verified_member_source_primitive")
                self.assertEqual(controller["stage"], "REVIEW_READY")
                self.assertIn("TECHNICALLY_VERIFIED", controller["stage_history"])
                self.assertFalse(controller["quality_approved"])
            self.assertFalse(direct["member_pose_dissolve_effective"])
            self.assertEqual(direct["member_pose_dissolve_count"], 0)
            self.assertEqual(
                direct["pose_matched_dissolves"]["hard_cut_count"],
                11,
            )
            self.assertTrue(
                direct["pose_matched_dissolves"][
                    "source_metadata_contract_verified"
                ]
            )
            self.assertTrue(
                direct["pose_matched_dissolves"]["planner_completed"]
            )
            self.assertNotIn(
                "source_duration_evidence_missing_or_invalid",
                direct["pose_matched_dissolves"][
                    "rejection_reason_counts"
                ],
            )
            self.assertEqual(
                direct["policy"],
                OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY,
            )
            self.assertEqual(
                direct["identity_policy"],
                "official_target_member_provenance_no_identity_inference_"
                "with_lossless_portrait_fill",
            )
            self.assertEqual(
                direct["visual_selection"]["analysis_basis"],
                "synthetic_non_identity_geometry_test",
            )
            self.assertEqual(
                direct["visual_selection"]["source_usage"][0][
                    "request_source_index"
                ],
                0,
            )
            self.assertEqual(
                direct["visual_selection"]["boundaries"][0][
                    "incoming_request_source_index"
                ],
                1,
            )
            self.assertEqual(
                direct["visual_selection"]["span_content_validity"][0][
                    "sources"
                ][0]["request_source_index"],
                0,
            )
            self.assertFalse(
                direct["visual_selection"]["privacy"][
                    "face_identity_inference_used"
                ]
            )
            self.assertEqual(
                direct["editorial_motion"]["analysis_basis"],
                "official_target_fancam_verified_primitive_v1" if _camera_failure else direct_motion_plan.analysis_basis,
            )
            self.assertEqual(
                direct["dense_face_review"]["backend"],
                "synthetic_dense_face_provider",
            )
            self.assertEqual(direct["dense_face_review"]["completed"], not bool(_camera_failure))
            self.assertFalse(
                direct["dense_face_review"][
                    "fallback_to_untouched_full_frame"
                ]
            )
            self.assertEqual(
                direct["dense_face_review"]["applied_segment_count"],
                0 if _camera_failure else 1,
            )
            self.assertEqual(direct["final_source_usage"]["distinct_source_count"], 10)
            self.assertEqual(
                [item["segment_count"] for item in direct["final_source_usage"]["sources"]],
                [2, 2, *([1] * 8)],
            )
            self.assertEqual(
                [item["duration_seconds"] for item in direct["final_source_usage"]["sources"]],
                [6.0, 6.0, 2.0, 2.0, *([4.0] * 6)],
            )
            self.assertAlmostEqual(direct["segments"][0]["source_local_end"], 1.992)
            self.assertEqual(
                direct["segments"][0]["source_clock_basis"],
                "local_monotonic_timewarp",
            )
            render_payload = manifest["render"]
            self.assertEqual(
                render_payload["selection_basis"],
                OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY,
            )
            self.assertTrue(render_payload["direct_full_frame"])
            self.assertEqual(render_payload["automatic_full_frame_fallback_count"], 0)
            self.assertEqual(render_payload["automatic_full_frame_fallbacks"], [])
            self.assertEqual(render_payload["final_source_usage"]["distinct_source_count"], 10)
            self.assertAlmostEqual(
                render_payload["final_segments"][0]["source_local_end"],
                1.992,
            )
            self.assertTrue(render_payload["cadence"]["passed"])
            self.assertEqual(
                render_payload["editorial_motion"]["analysis_basis"],
                "official_target_fancam_verified_primitive_v1" if _camera_failure else direct_motion_plan.analysis_basis,
            )
            self.assertEqual(
                manifest["editorial_motion"]["analysis_basis"],
                "official_target_fancam_verified_primitive_v1" if _camera_failure else direct_motion_plan.analysis_basis,
            )
            self.assertEqual(
                manifest["editorial_motion"]["index_domain"],
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            capabilities = manifest["capabilities"]
            self.assertTrue(capabilities["member_direct_full_frame_ready"])
            self.assertFalse(capabilities["member_tracking_ready"])
            self.assertIn("자동 얼굴 신원 판정", capabilities["member_tracking_note"])
            self.assertIn("크롭·추적은 수행하지 않습니다", capabilities["member_tracking_note"])

    def test_member_start_trims_to_latest_canonical_anchor(self) -> None:
        alignment = AlignmentResult(
            reference="a",
            sample_rate=4_000,
            alignments=(
                SourceAlignment("a", True, 0, 0.0, 1.0, 1, 1.0),
                SourceAlignment("b", False, 0, 2.0, 0.9, 1, 1.0),
                SourceAlignment("c", False, 0, -1.0, 0.9, 1, 1.0),
            ),
        )

        effective_start, payload = _member_anchor_trim(
            common_master_start=1.0,
            common_master_end=10.0,
            alignment=alignment,
            eligible_indices=[0, 1, 2],
            tracking_results={
                0: _tracking("a", 2.0),
                1: _tracking("b", 4.5),
                2: _tracking("c", 0.5),
            },
        )

        self.assertAlmostEqual(effective_start, 2.5)
        self.assertEqual(payload["index_domain"], REQUEST_SOURCE_INDEX_DOMAIN)
        self.assertTrue(payload["applied"])
        self.assertFalse(payload["reverse_tracking_used"])
        self.assertAlmostEqual(payload["trimmed_seconds"], 1.5)
        self.assertEqual(payload["limiting_source_indices"], [1])
        self.assertEqual(
            [item["canonical_anchor_time_seconds"] for item in payload["anchors"]],
            [2.0, 2.5, 1.5],
        )

    def test_automatic_member_range_preserves_prefix_and_missing_anchor_source(self) -> None:
        alignment = AlignmentResult(
            reference="a",
            sample_rate=4_000,
            alignments=(
                SourceAlignment("a", True, 0, 0.0, 1.0, 1, 1.0),
                SourceAlignment("b", False, 0, 2.0, 0.9, 1, 1.0),
                SourceAlignment("c", False, 0, -1.0, 0.9, 1, 1.0),
            ),
        )

        effective_start, payload = _member_anchor_trim(
            common_master_start=1.0,
            common_master_end=10.0,
            alignment=alignment,
            eligible_indices=[0, 1, 2],
            tracking_results={
                0: _tracking("a", 4.0),
                1: _tracking("b", 7.0),
            },
            preserve_automatic_full_frame_prefix=True,
        )

        self.assertEqual(effective_start, 1.0)
        self.assertFalse(payload["applied"])
        self.assertEqual(payload["full_frame_only_source_indices"], [2])
        self.assertIn("full_frame_fallback", payload["policy"])

    def test_member_start_does_not_expand_an_already_later_common_range(self) -> None:
        alignment = AlignmentResult(
            reference="a",
            sample_rate=4_000,
            alignments=(
                SourceAlignment("a", True, 0, 0.0, 1.0, 1, 1.0),
                SourceAlignment("b", False, 0, 1.0, 0.9, 1, 1.0),
            ),
        )

        effective_start, payload = _member_anchor_trim(
            common_master_start=3.0,
            common_master_end=10.0,
            alignment=alignment,
            eligible_indices=[0, 1],
            tracking_results={
                0: _tracking("a", 1.0),
                1: _tracking("b", 2.0),
            },
        )

        self.assertEqual(effective_start, 3.0)
        self.assertFalse(payload["applied"])
        self.assertEqual(payload["trimmed_seconds"], 0.0)
        self.assertTrue(payload["usable_for_render"])

    def test_member_start_marks_too_short_post_anchor_range_unusable(self) -> None:
        alignment = AlignmentResult(
            reference="a",
            sample_rate=4_000,
            alignments=(
                SourceAlignment("a", True, 0, 0.0, 1.0, 1, 1.0),
                SourceAlignment("b", False, 0, 0.0, 0.9, 1, 1.0),
            ),
        )

        effective_start, payload = _member_anchor_trim(
            common_master_start=0.0,
            common_master_end=10.0,
            alignment=alignment,
            eligible_indices=[0, 1],
            tracking_results={
                0: _tracking("a", 1.0),
                1: _tracking("b", 9.95),
            },
        )

        self.assertEqual(effective_start, 9.95)
        self.assertFalse(payload["usable_for_render"])
        self.assertAlmostEqual(payload["usable_duration_seconds"], 0.05)


if __name__ == "__main__":
    unittest.main()
