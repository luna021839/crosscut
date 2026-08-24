from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.auto_anchor import AutoAnchorResult
from crosscut.direct_member_visual import DirectMemberVisualSelectionError
from crosscut.direct_member_transition import (
    OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT,
)
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
from crosscut.pipeline import (
    DEFAULT_IVE_PROFESSIONAL_MONTAGE_PROFILE_PATH,
    DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_FILE_SHA256,
    DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_PATH,
    ELIGIBLE_SOURCE_INDEX_DOMAIN,
    OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY,
    REQUEST_SOURCE_INDEX_DOMAIN,
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
    _direct_full_frame_shifted_partition_profiles,
    _direct_partition_failure_bottlenecks,
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
from crosscut.audio import AlignmentResult, SourceAlignment
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
from crosscut.rendered_proxy_review import RenderedProxyReview
from crosscut.request import AnalysisRequest, MemberAnchorRequest
from crosscut.tracking import (
    MemberTrackObservation,
    MemberTrackingResult,
    TrackingFailure,
    TrackingState,
)
from crosscut.timeline import ShotType
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
    payload = {
        "issue_count": 0 if passed else 1,
        "fail_issue_count": 0,
        "conditional_issue_count": 0 if passed else 1,
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
    def __init__(self, source_indices: tuple[int, ...], source_count: int) -> None:
        self.source_indices = source_indices
        self.source_counts = tuple(
            source_indices.count(index) for index in range(source_count)
        )

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
        policy = load_professional_montage_policy(
            DEFAULT_IVE_PROFESSIONAL_MONTAGE_PROFILE_PATH
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
        montage_policy = load_professional_montage_policy(
            DEFAULT_IVE_PROFESSIONAL_MONTAGE_PROFILE_PATH
        )
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
            DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_PATH,
            expected_file_sha256=(
                DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_FILE_SHA256
            ),
            verify_source_inputs=False,
        )
        self.assertIsNotNone(profile)
        assert profile is not None
        payload = _stage_edit_motion_cut_profile_manifest_payload(
            profile,
            montage_policy,
        )
        self.assertEqual(
            str(DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_PATH),
            payload["path"],
        )
        self.assertEqual(
            DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_FILE_SHA256,
            payload["file_sha256"],
        )
        self.assertEqual(
            "9f70c549acbda2e80f3497b57c189ddbb5df4e4bbd4f5eda9095c5af699ae680",
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
        self.assertAlmostEqual(
            0.0002705025925969218,
            runtime["both_motion_minimum_pose_motion_p10"],
        )
        self.assertAlmostEqual(
            -0.49213925079872173,
            runtime["continuity_minimum_score_p10"],
        )
        self.assertAlmostEqual(
            0.8486102174679628,
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
            0.0002705025925969218,
            config.highlight_motion_boundary_minimum_pose_motion,
        )
        self.assertAlmostEqual(
            -0.49213925079872173,
            config.highlight_motion_boundary_minimum_continuity_score,
        )
        self.assertAlmostEqual(
            0.8486102174679628,
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
        montage_policy = load_professional_montage_policy(
            DEFAULT_IVE_PROFESSIONAL_MONTAGE_PROFILE_PATH
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

        profile = load_professional_motion_cut_profile(
            DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_PATH,
            expected_file_sha256=(
                DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_FILE_SHA256
            ),
            verify_source_inputs=False,
        )
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
        montage_policy = load_professional_montage_policy(
            DEFAULT_IVE_PROFESSIONAL_MONTAGE_PROFILE_PATH
        )
        profile = load_professional_motion_cut_profile(
            DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_PATH,
            expected_file_sha256=(
                DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_FILE_SHA256
            ),
            verify_source_inputs=False,
        )
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
            repaired = VisualBoundaryRepairPlan(visual, (), 0, ())
            timeline_review = SimpleNamespace(weighted_score=81.5)
            face_match = SimpleNamespace(
                pan_crop_for=lambda _segment_index: None,
            )
            transitions = GroupTransitionPlan(
                (RenderTransition(tier="C", reason="synthetic"),),
                analysis_basis="synthetic_complete_candidate",
            )
            affine = SimpleNamespace()
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
            ) as montage_review:
                result = _review_complete_group_candidates(
                    bundles,  # type: ignore[arg-type]
                    scratch_directory=root,
                    stage_edit_montage_section_plan=section_plan,  # type: ignore[arg-type]
                    stage_edit_source_derivative_diversity=diversity,  # type: ignore[arg-type]
                    stage_edit_source_index_mapping={"bound": True},
                    stage_edit_montage_config=RenderedMontageGrammarConfig(),
                    stage_edit_montage_reference_profile=identity,
                )

            self.assertIs(result, sentinel)
            self.assertEqual(montage_review.call_count, 8)
            self.assertEqual(len(reviewed_payloads), 8)
            self.assertTrue(all(item["passed"] for item in reviewed_payloads))
            self.assertTrue(
                all(
                    item["rendered_montage_grammar_review"][
                        "reference_profile"
                    ]
                    == identity
                    for item in reviewed_payloads
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
                boundary_repairs=None,
                timeline_review=None,
                face_match_plan=None,
                group_transition_plan=transitions,
                boundary_affine_plan=None,
                editorial_motion_plan=None,
                render_plan=original_plan,
            )
            partial_bundle = _GroupRenderCandidateBundle(
                candidate_index=3,
                review_rank=0,
                cadence_profile="synthetic",
                pre_render_score=89.0,
                visual_timeline=repaired_visual,
                boundary_repairs=None,
                timeline_review=None,
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
                timeline_review=None,
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
                "crosscut.pipeline.repair_visual_timeline_boundaries",
                return_value=targeted,
            ) as repair, patch(
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
                )

            self.assertIs(result.selected_bundle, partial_bundle)
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
                boundary_repairs=None,
                timeline_review=None,
                face_match_plan=None,
                group_transition_plan=transitions,
                boundary_affine_plan=None,
                editorial_motion_plan=None,
                render_plan=plan,
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
                "crosscut.pipeline.repair_visual_timeline_boundaries",
                return_value=targeted,
            ) as repair:
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

    def test_automatic_group_proxy_rerank_applies_actual_winner_and_blocks_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = tuple(root / f"source-{index}.mp4" for index in range(9))
            for source in sources:
                source.touch()
            request = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
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

            class SelectionStub:
                selected_candidate_index = 6
                reviews = tuple(object() for _ in range(8))
                successful_review_count = 8
                minimum_candidate_count_satisfied = True
                passed = False
                selected = SimpleNamespace(
                    review_payload={
                        "passed": False,
                        "weighted_score": 96.0,
                        "boundary_issue_indices": [0],
                    },
                    weighted_score=96.0,
                )

                def to_dict(self) -> dict[str, object]:
                    return {
                        "selected_candidate_index": self.selected_candidate_index,
                        "reviewed_candidate_count": len(self.reviews),
                        "passed": self.passed,
                    }

            selection = SelectionStub()
            candidate_set = SimpleNamespace(
                to_dict=lambda: {
                    "candidate_count": 8,
                    "selected_candidate_index": 0,
                }
            )
            materialized = tuple(
                SimpleNamespace(candidate_index=index, review_rank=index)
                for index in range(8)
            )
            look_result = SimpleNamespace(
                adjustments=tuple(SourceLookAdjustment() for _ in sources),
                target_mode="explicit_reference",
                adjustment_for_source=lambda index: SourceLookAdjustment(),
                to_dict=lambda: {
                    "target_mode": "explicit_reference",
                    "evidence": [],
                },
            )

            with patch(
                "crosscut.pipeline._enforce_automatic_group_stage_edit_requirement"
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
                "crosscut.pipeline.detect_scene_changes_in_video",
                return_value=(),
            ), patch(
                "crosscut.pipeline.analyze_source_look_match",
                return_value=look_result,
            ) as analyze_look, patch(
                "crosscut.pipeline.estimate_shared_performance_range",
                side_effect=ValueError("synthetic range unavailable"),
            ), patch(
                "crosscut.pipeline.build_group_draft",
                return_value=group_draft,
            ), patch(
                "crosscut.pipeline.build_visual_timeline_candidate_set",
                return_value=candidate_set,
            ) as build_candidates, patch(
                "crosscut.pipeline.materialize_visual_timeline_candidates",
                return_value=materialized,
            ) as materialize_candidates, patch(
                "crosscut.pipeline.materialize_visual_timeline_candidate_record",
            ) as materialize_candidate_record, patch(
                "crosscut.pipeline._derive_complete_group_candidate",
                side_effect=lambda item, **_kwargs: bundles[
                    item.candidate_index
                ],
            ) as derive, patch(
                "crosscut.pipeline._review_complete_group_candidates",
                return_value=selection,
            ) as review_candidates, patch("crosscut.pipeline.write_plan"), patch(
                "crosscut.pipeline.render",
                return_value="h264_videotoolbox",
            ) as render_video:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "blocked by decoded proxy review",
                ):
                    analyze(
                        request,
                        _MemoryReporter(),  # type: ignore[arg-type]
                        review_rendered_proxies=True,
                    )
                render_video.assert_not_called()
                self.assertTrue(
                    any(
                        (root / "out").glob(
                            "*-rendered-candidate-selection.json"
                        )
                    )
                )

                selection.passed = True
                selection.selected.review_payload["passed"] = True
                artifact = analyze(
                    request,
                    _MemoryReporter(),  # type: ignore[arg-type]
                    allow_failed_comparison_render=True,
                    review_rendered_proxies=True,
                )

                def fail_last_candidate(item, **_kwargs):
                    if item.candidate_index == 7:
                        raise ValueError("synthetic complete derivation failure")
                    return bundles[item.candidate_index]

                derive.side_effect = fail_last_candidate
                selection.reviews = tuple(object() for _ in range(7))
                selection.minimum_candidate_count_satisfied = False
                with self.assertRaisesRegex(
                    RuntimeError,
                    "requires exactly 8 complete candidates; got 7",
                ):
                    analyze(
                        request,
                        _MemoryReporter(),  # type: ignore[arg-type]
                        review_rendered_proxies=True,
                    )
                selection_reports = tuple(
                    (root / "out").glob(
                        "*-rendered-candidate-selection.json"
                    )
                )
                self.assertTrue(selection_reports)
                blocked_report = json.loads(
                    max(
                        selection_reports,
                        key=lambda path: path.stat().st_mtime_ns,
                    ).read_text(encoding="utf-8")
                )

                derive.side_effect = lambda item, **_kwargs: bundles[
                    item.candidate_index
                ]
                selection.reviews = tuple(object() for _ in range(8))
                selection.successful_review_count = 7
                selection.minimum_candidate_count_satisfied = True
                with self.assertRaisesRegex(
                    RuntimeError,
                    "8 successful decoded reviews; got 8 complete / 7 decoded",
                ):
                    analyze(
                        request,
                        _MemoryReporter(),  # type: ignore[arg-type]
                        review_rendered_proxies=True,
                    )
                decoded_failure_report = json.loads(
                    max(
                        (root / "out").glob(
                            "*-rendered-candidate-selection.json"
                        ),
                        key=lambda path: path.stat().st_mtime_ns,
                    ).read_text(encoding="utf-8")
                )

                materialize_candidates.return_value = materialized[:7]
                selection.successful_review_count = 8
                review_call_count = review_candidates.call_count
                with self.assertRaisesRegex(
                    RuntimeError,
                    "requires exactly 8 complete candidates; got 7",
                ):
                    analyze(
                        request,
                        _MemoryReporter(),  # type: ignore[arg-type]
                        review_rendered_proxies=True,
                    )
                self.assertEqual(
                    review_candidates.call_count,
                    review_call_count,
                )
                structure_failure_report = json.loads(
                    max(
                        (root / "out").glob(
                            "*-rendered-candidate-selection.json"
                        ),
                        key=lambda path: path.stat().st_mtime_ns,
                    ).read_text(encoding="utf-8")
                )

                class FocusIssueStub:
                    code = "focus_mismatch_exceeded"

                class PostRepairIssueStub:
                    code = "focus_mismatch_exceeded"

                class PostRepairFailureReviewStub:
                    passed = False
                    weighted_score = 20.0
                    hard_failures = (PostRepairIssueStub(),)

                    @staticmethod
                    def to_dict() -> dict[str, object]:
                        return {
                            "passed": False,
                            "weighted_score": 20.0,
                            "hard_failures": [
                                {"code": "focus_mismatch_exceeded"}
                            ],
                        }

                passing_recovery_records = tuple(
                    SimpleNamespace(
                        candidate_index=index,
                        review_rank=index,
                        review=SimpleNamespace(
                            passed=True,
                            hard_failures=(),
                        ),
                    )
                    for index in range(4)
                )
                density_failure_record = SimpleNamespace(
                    candidate_index=9,
                    review_rank=4,
                    review=SimpleNamespace(
                        passed=False,
                        hard_failures=(
                            SimpleNamespace(
                                code=(
                                    "source_switch_density_below_hard_minimum"
                                )
                            ),
                        ),
                    ),
                )
                focus_recovery_records = tuple(
                    SimpleNamespace(
                        candidate_index=index,
                        review_rank=index + 1,
                        review=SimpleNamespace(
                            passed=False,
                            hard_failures=(FocusIssueStub(),),
                        ),
                    )
                    for index in range(4, 9)
                )
                candidate_set.ranked_candidates = (
                    passing_recovery_records
                    + (density_failure_record,)
                    + focus_recovery_records
                )
                recovery_materialized = materialized + (
                    SimpleNamespace(candidate_index=8, review_rank=8),
                )
                recovery_bundles = {
                    item.candidate_index: item for item in bundles
                }
                recovery_bundles[4] = replace(
                    recovery_bundles[4],
                    timeline_review=PostRepairFailureReviewStub(),  # type: ignore[arg-type]
                )
                recovery_bundles[8] = replace(
                    recovery_bundles[7],
                    candidate_index=8,
                    review_rank=8,
                )
                materialize_candidates.return_value = materialized[:4]
                materialize_candidate_record.side_effect = (
                    lambda _candidate_set, record: recovery_materialized[
                        record.candidate_index
                    ]
                )
                derive.side_effect = lambda item, **_kwargs: recovery_bundles[
                    item.candidate_index
                ]
                selection.reviews = tuple(object() for _ in range(8))
                selection.successful_review_count = 8
                selection.minimum_candidate_count_satisfied = True
                selection.passed = True
                selection.selected.review_payload["passed"] = True

                recovered_artifact = analyze(
                    request,
                    _MemoryReporter(),  # type: ignore[arg-type]
                    allow_failed_comparison_render=True,
                    review_rendered_proxies=True,
                )
                recovery_report = recovered_artifact.manifest[
                    "rendered_candidate_selection"
                ]

            build_candidates.assert_called()
            self.assertEqual(
                build_candidates.call_args.kwargs["maximum_results"],
                32,
            )
            self.assertEqual(
                build_candidates.call_args.kwargs["feature_cache_directory"],
                (root / "source-cache").resolve(),
            )
            self.assertEqual(materialize_candidates.call_count, 6)
            self.assertTrue(
                all(
                    call.kwargs["maximum_candidates"] == 8
                    and call.kwargs["passed_only"] is True
                    for call in materialize_candidates.call_args_list
                )
            )
            self.assertEqual(analyze_look.call_count, 6)
            self.assertEqual(
                analyze_look.call_args.kwargs["reference_source"],
                0,
            )
            self.assertEqual(
                analyze_look.call_args.kwargs["source_timewarps"],
                (None,) * len(sources),
            )
            self.assertEqual(derive.call_count, 48)
            self.assertTrue(
                all(
                    call.kwargs["source_look_match"] is look_result
                    for call in derive.call_args_list
                )
            )
            self.assertEqual(render_video.call_count, 2)
            self.assertEqual(materialize_candidate_record.call_count, 5)
            self.assertEqual(
                recovery_report["structure_passing_materialized_count"],
                4,
            )
            self.assertEqual(
                recovery_report["repair_eligible_candidate_count"],
                5,
            )
            self.assertEqual(
                recovery_report["repair_eligible_materialized_count"],
                5,
            )
            self.assertEqual(
                recovery_report["repair_eligible_attempted_count"],
                5,
            )
            self.assertEqual(
                recovery_report["total_materialized_candidate_count"],
                9,
            )
            self.assertEqual(
                recovery_report[
                    "post_repair_structure_passing_complete_count"
                ],
                8,
            )
            self.assertEqual(
                recovery_report["complete_candidate_count"],
                8,
            )
            self.assertEqual(
                recovery_report[
                    "candidate_post_repair_review_rejections"
                ][0]["candidate_index"],
                4,
            )
            self.assertTrue(
                recovery_report[
                    "exact_complete_candidate_count_satisfied"
                ]
            )
            self.assertEqual(blocked_report["complete_candidate_count"], 7)
            self.assertFalse(
                blocked_report["exact_complete_candidate_count_satisfied"]
            )
            self.assertFalse(blocked_report["passed"])
            self.assertEqual(
                blocked_report["blocking_reason"],
                "automatic_group_requires_exactly_eight_complete_candidates",
            )
            self.assertEqual(
                blocked_report["candidate_derivation_failures"][0][
                    "candidate_index"
                ],
                7,
            )
            self.assertIn(
                "synthetic complete derivation failure",
                blocked_report["candidate_derivation_failures"][0]["error"],
            )
            self.assertEqual(
                blocked_report["structure_passing_materialized_count"],
                8,
            )
            self.assertEqual(
                blocked_report["source_index_mapping"][
                    "eligible_to_request"
                ],
                list(range(9)),
            )
            self.assertEqual(
                blocked_report["nested_index_domains"][
                    "complete_candidates[].visual_timeline.choices[]"
                    ".source_index"
                ],
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            self.assertEqual(
                blocked_report["nested_index_domains"][
                    "complete_candidates[].final_render_segments[]"
                    ".request_source_index"
                ],
                REQUEST_SOURCE_INDEX_DOMAIN,
            )
            self.assertEqual(
                blocked_report["nested_index_domains"][
                    "final_selected_review.boundaries[]"
                    ".outgoing_source_index"
                ],
                REQUEST_SOURCE_INDEX_DOMAIN,
            )
            self.assertEqual(decoded_failure_report["complete_candidate_count"], 8)
            self.assertEqual(
                decoded_failure_report["successful_review_count"],
                7,
            )
            self.assertFalse(
                decoded_failure_report[
                    "exact_successful_decoded_review_count_satisfied"
                ]
            )
            self.assertFalse(
                decoded_failure_report[
                    "exact_complete_candidate_count_satisfied"
                ]
            )
            self.assertEqual(
                decoded_failure_report["blocking_reason"],
                (
                    "automatic_group_requires_exactly_eight_"
                    "successful_decoded_candidate_reviews"
                ),
            )
            self.assertFalse(decoded_failure_report["passed"])
            self.assertEqual(
                structure_failure_report[
                    "structure_passing_materialized_count"
                ],
                7,
            )
            self.assertEqual(
                structure_failure_report["complete_candidate_count"],
                7,
            )
            self.assertEqual(
                structure_failure_report["candidate_derivation_failures"],
                [],
            )
            self.assertEqual(
                structure_failure_report["reviewed_candidate_count"],
                0,
            )
            self.assertEqual(
                structure_failure_report["successful_review_count"],
                0,
            )
            self.assertFalse(structure_failure_report["passed"])
            final_plan = render_video.call_args.args[0]
            self.assertEqual(final_plan.segments[0].source_index, 6)
            self.assertEqual(
                artifact.manifest["rendered_candidate_selection"][
                    "selected_candidate_index"
                ],
                6,
            )
            self.assertEqual(
                artifact.manifest["render"]["rendered_proxy_review"][
                    "weighted_score"
                ],
                96.0,
            )
            self.assertEqual(
                artifact.manifest["source_look_match"]["index_domain"],
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            self.assertEqual(
                artifact.manifest["render"]["source_look_match"][
                    "target_mode"
                ],
                "explicit_reference",
            )
            selection_payload = artifact.manifest[
                "rendered_candidate_selection"
            ]
            self.assertEqual(
                selection_payload["source_index_mapping"][
                    "eligible_to_request"
                ],
                list(range(9)),
            )
            self.assertEqual(
                selection_payload["nested_index_domains"][
                    "visual_candidate_set.candidates[].source_sequence[]"
                ],
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            self.assertEqual(
                selection_payload["nested_index_domains"][
                    "candidates[].review.source_quality.per_source[]"
                    ".source_index"
                ],
                REQUEST_SOURCE_INDEX_DOMAIN,
            )
            manifest_domains = artifact.manifest["source_index_mapping"][
                "payload_index_domains"
            ]
            self.assertEqual(
                manifest_domains["source_look_match.target_source_index"],
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            self.assertEqual(
                manifest_domains[
                    "rendered_candidate_selection.complete_candidates[]"
                    ".final_render_segments[].request_source_index"
                ],
                REQUEST_SOURCE_INDEX_DOMAIN,
            )

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

    def test_automatic_group_blocks_failed_review_but_comparison_opt_in_and_local_render(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = tuple(root / f"source-{index}.mp4" for index in range(9))
            for source in sources:
                source.touch()
            request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
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
                "crosscut.pipeline._enforce_automatic_group_stage_edit_requirement"
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
                "crosscut.pipeline.detect_scene_changes_in_video",
                return_value=(),
            ), patch(
                "crosscut.pipeline.build_group_draft",
                return_value=group_draft,
            ), patch(
                "crosscut.pipeline.build_visual_timeline",
                return_value=visual_timeline,
            ), patch(
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
                with self.assertRaisesRegex(
                    RuntimeError,
                    "automatic GROUP render blocked by timeline review failures",
                ):
                    analyze(automatic_request, reporter)  # type: ignore[arg-type]
                render_video.assert_not_called()
                automatic_artifact = analyze(
                    automatic_request,
                    reporter,  # type: ignore[arg-type]
                    allow_failed_comparison_render=True,
                )
                artifact = analyze(request, reporter)  # type: ignore[arg-type]

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
            sources = tuple(root / f"source-{index}.mp4" for index in range(9))
            for source in sources:
                source.touch()
            request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
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
                "crosscut.pipeline._enforce_automatic_group_stage_edit_requirement"
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
                "crosscut.pipeline.detect_scene_changes_in_video",
                return_value=(),
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
                    analyze(request, _MemoryReporter())  # type: ignore[arg-type]
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
                "crosscut.pipeline._enforce_automatic_group_stage_edit_requirement"
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
                "crosscut.pipeline.detect_scene_changes_in_video",
                side_effect=scene_changes,
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
                    analyze(request, _MemoryReporter())  # type: ignore[arg-type]

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
                "crosscut.pipeline.detect_scene_changes_in_video",
                side_effect=scene_changes,
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

        selection = _VisualSelectionStub(tuple(range(10)), 10)
        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            return_value=selection,
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
        self.assertEqual(
            len(select_visual.call_args.kwargs["canonical_spans"]),
            10,
        )

        self.assertEqual(len(segments), 10)
        self.assertEqual([item.source_index for item in segments], list(range(10)))
        self.assertTrue(all(math.isclose(item.duration, 4.0) for item in segments))
        self.assertTrue(all(item.crop is None for item in segments))
        self.assertTrue(all(item.pan_crop is None for item in segments))
        self.assertTrue(all(item.source_duration is not None for item in segments))
        self.assertAlmostEqual(segments[0].source_start, 0.0)
        self.assertAlmostEqual(segments[0].source_end, 4.016)
        self.assertEqual(len(transitions), 9)
        self.assertTrue(
            all(
                item.kind is TransitionKind.HARD_CUT
                and item.reason
                == OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY
                for item in transitions
            )
        )
        self.assertEqual(details[0]["timing_segment_count"], 4)
        self.assertEqual(details[0]["source_clock_basis"], "local_monotonic_timewarp")
        self.assertAlmostEqual(details[0]["source_local_end"], 4.016)
        self.assertTrue(cadence["passed"])
        self.assertAlmostEqual(cadence["source_switches_per_minute"], 13.5)
        self.assertAlmostEqual(cadence["source_run_median_seconds"], 4.0)
        self.assertAlmostEqual(cadence["longest_source_run_seconds"], 4.0)
        self.assertFalse(cadence["member_pose_dissolve_requested"])
        self.assertFalse(cadence["member_pose_dissolve_effective"])
        self.assertEqual(cadence["member_pose_dissolve_count"], 0)
        self.assertEqual(
            cadence["pose_matched_dissolves"]["rejection_reason_counts"][
                "effect_disabled_by_request"
            ],
            9,
        )
        self.assertEqual(visual_selection.source_counts, (1,) * 10)

    def test_direct_full_frame_reference_length_avoids_the_200_cut_path(self) -> None:
        duration = 196.4835616941711
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=duration / 200.0,
        )

        selection = _VisualSelectionStub(
            tuple(index % 10 for index in range(50)),
            10,
        )
        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            return_value=selection,
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

        self.assertEqual(len(segments), 50)
        self.assertEqual(
            [sum(item.source_index == source for item in segments) for source in range(10)],
            [5] * 10,
        )
        self.assertAlmostEqual(cadence["source_switches_per_minute"], 14.9631, places=3)
        self.assertAlmostEqual(
            cadence["source_run_median_seconds"],
            duration / 50.0,
        )
        self.assertLessEqual(cadence["longest_source_run_seconds"], 7.2)
        self.assertEqual(visual_selection.source_counts, (5,) * 10)

    def test_direct_full_frame_fifty_spans_supports_eight_balanced_sources(
        self,
    ) -> None:
        timing = _rotating_timing_draft(
            source_count=8,
            duration_seconds=200.0,
            unit_seconds=1.0,
        )
        path = tuple(range(8)) * 6 + (0, 1)
        selection = _VisualSelectionStub(path, 8)

        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            return_value=selection,
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

        self.assertEqual(len(segments), 50)
        self.assertEqual(visual_selection.source_counts, (7, 7, 6, 6, 6, 6, 6, 6))
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
        path = tuple(range(4)) * 12 + (0, 1)
        selection = _VisualSelectionStub(path, 4)

        with patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            return_value=selection,
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

        self.assertEqual(len(segments), 50)
        self.assertEqual(cadence["source_switch_count"], 49)
        self.assertAlmostEqual(
            cadence["source_switches_per_minute"],
            49 * 60.0 / duration,
        )
        self.assertTrue(cadence["passed"])
        self.assertEqual(
            visual_selection.source_counts,
            (13, 13, 12, 12),
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
        selection = _VisualSelectionStub(
            tuple(index % 10 for index in range(50)),
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
                return selection
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

        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0], (4,) * 50)
        self.assertEqual(calls[1][-2:], (5, 3))
        self.assertEqual(calls[2][-2:], (3, 5))
        self.assertEqual([item["timing_segment_count"] for item in details[-2:]], [5, 3])
        self.assertEqual(len(segments), 50)
        self.assertEqual(visual_selection.source_counts, (5,) * 10)
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
            "boundary-48-5-plus-3",
        )
        self.assertTrue(partition["fallback_used"])
        self.assertFalse(partition["visual_gates_relaxed"])
        self.assertEqual(len(partition["profiles_attempted"]), 3)
        self.assertTrue(
            all(
                call.kwargs["descriptor_provider"] is shared_provider
                for call in select_visual.call_args_list
            )
        )
        frozen_config = select_visual.call_args_list[0].kwargs["config"]
        self.assertTrue(
            all(
                call.kwargs["config"] is frozen_config
                for call in select_visual.call_args_list
            )
        )

    def test_direct_partition_fallback_all_fail_is_atomic_and_keeps_full_evidence(
        self,
    ) -> None:
        duration = 196.4835616941711
        timing = _rotating_timing_draft(
            duration_seconds=duration,
            unit_seconds=duration / 200.0,
        )
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
        self.assertEqual(
            search["profiles_attempted"][0]["diagnostics"][
                "span_content_validity"
            ][0]["sources"][0]["rejection_reasons"],
            ["foreground_camera_equipment_or_subject_obstruction"],
        )

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
        duration = 196.4835616941711
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
        for duration in durations:
            spans.append(
                SimpleNamespace(
                    canonical_master_start=cursor,
                    canonical_master_end=cursor + duration,
                    duration_seconds=duration,
                    timing_segment_count=1,
                    estimated_beat_count=duration * 2.0,
                    start_boundary_reasons=("strong_beat",),
                    end_boundary_reasons=("strong_beat",),
                )
            )
            cursor += duration
        selection = _VisualSelectionStub(tuple(range(10)), 10)

        with patch(
            "crosscut.pipeline._coarsen_direct_full_frame_spans",
            return_value=tuple(spans),
        ), patch(
            "crosscut.pipeline.select_balanced_direct_member_sources",
            return_value=selection,
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
            ("landscape", 1920, 1080, "outside 9:16"),
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

    def test_automatic_portrait_member_uses_direct_full_frame_without_identity_anchor(self) -> None:
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
            visual_selection = _VisualSelectionStub(tuple(range(10)), 10)

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

                def sample(
                    self,
                    source_index: int,
                    source_local_seconds: float,
                ) -> None:
                    self.sample_calls.append(
                        (source_index, source_local_seconds)
                    )
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
                    EditorialTransformKeyframe(0.75, 1.06),
                ),
                reason="official_fancam_dense_face_scale_position_match",
            )
            direct_motion_plan = EditorialMotionPlan(
                segments=tuple(
                    EditorialMotionSegment(
                        index,
                        index,
                        direct_track if index == 0 else None,
                    )
                    for index in range(10)
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
            dense_face_provider = DenseFaceProviderStub()

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
                "crosscut.pipeline.detect_scene_changes_in_video",
                return_value=(),
            ), patch(
                "crosscut.pipeline.build_group_draft",
                return_value=timing,
            ), patch(
                "crosscut.pipeline.select_balanced_direct_member_sources",
                return_value=visual_selection,
            ), patch(
                "crosscut.pipeline._VideoDescriptorProvider",
                return_value=DescriptorProviderStub(),
            ), patch(
                "crosscut.pipeline.DirectMemberDenseFaceProvider",
                return_value=dense_face_provider,
            ) as dense_provider_factory, patch(
                "crosscut.pipeline.build_direct_member_editorial_motion_plan",
                side_effect=build_direct_motion,
            ) as build_direct_motion_mock, patch(
                "crosscut.pipeline.build_member_editorial_motion_plan"
            ) as build_member_motion, patch(
                "crosscut.pipeline.track_anchored_member"
            ) as track_member, patch(
                "crosscut.pipeline.build_member_draft"
            ) as build_cropped_member, patch(
                "crosscut.pipeline.write_plan"
            ), patch(
                "crosscut.pipeline.render",
                return_value="h264_videotoolbox",
            ) as render_video:
                artifact = analyze(request, _MemoryReporter())  # type: ignore[arg-type]

            track_member.assert_not_called()
            build_cropped_member.assert_not_called()
            build_member_motion.assert_not_called()
            build_direct_motion_mock.assert_called_once()
            dense_provider_factory.assert_called_once_with(request.sources)
            self.assertEqual(len(dense_face_provider.sample_calls), 1)
            self.assertEqual(dense_face_provider.sample_calls[0][0], 1)
            self.assertAlmostEqual(
                dense_face_provider.sample_calls[0][1],
                timewarps[1].master_to_source(4.0),
            )
            self.assertTrue(dense_face_provider.closed)
            plan = render_video.call_args.args[0]
            self.assertTrue(plan.member_pose_dissolve_requested)
            self.assertFalse(plan.member_pose_dissolve_effective)
            self.assertEqual(plan.member_pose_dissolve_count, 0)
            self.assertEqual(len(plan.segments), 10)
            self.assertEqual(
                [item.source_index for item in plan.segments],
                list(range(10)),
            )
            self.assertTrue(all(item.pan_crop is None for item in plan.segments))
            self.assertTrue(all(item.crop is None for item in plan.segments))
            self.assertIs(plan.segments[0].editorial_transform, direct_track)
            self.assertTrue(
                all(
                    item.editorial_transform is None
                    for item in plan.segments[1:]
                )
            )
            self.assertAlmostEqual(plan.segments[0].source_end, 3.984)

            manifest = artifact.manifest
            self.assertEqual(manifest["member_tracking"], [])
            self.assertEqual(manifest["member_auto_anchors"], [])
            self.assertIsNone(manifest["member_anchor_trim"])
            self.assertIsNone(manifest["member_draft"])
            direct = manifest["member_direct_full_frame"]
            self.assertTrue(direct["enabled"])
            self.assertTrue(direct["member_pose_dissolve_requested"])
            self.assertFalse(direct["member_pose_dissolve_effective"])
            self.assertEqual(direct["member_pose_dissolve_count"], 0)
            self.assertEqual(
                direct["pose_matched_dissolves"]["hard_cut_count"],
                9,
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
                "no_geometry_identity_claim_no_tracking_no_crop",
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
                direct_motion_plan.analysis_basis,
            )
            self.assertEqual(
                direct["dense_face_review"]["backend"],
                "synthetic_dense_face_provider",
            )
            self.assertTrue(direct["dense_face_review"]["completed"])
            self.assertFalse(
                direct["dense_face_review"][
                    "fallback_to_untouched_full_frame"
                ]
            )
            self.assertEqual(
                direct["dense_face_review"]["applied_segment_count"],
                1,
            )
            self.assertEqual(direct["final_source_usage"]["distinct_source_count"], 10)
            self.assertEqual(
                [item["segment_count"] for item in direct["final_source_usage"]["sources"]],
                [1] * 10,
            )
            self.assertEqual(
                [item["duration_seconds"] for item in direct["final_source_usage"]["sources"]],
                [4.0] * 10,
            )
            self.assertAlmostEqual(direct["segments"][0]["source_local_end"], 3.984)
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
                3.984,
            )
            self.assertTrue(render_payload["cadence"]["passed"])
            self.assertEqual(
                render_payload["editorial_motion"]["analysis_basis"],
                direct_motion_plan.analysis_basis,
            )
            self.assertEqual(
                manifest["editorial_motion"]["analysis_basis"],
                direct_motion_plan.analysis_basis,
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
