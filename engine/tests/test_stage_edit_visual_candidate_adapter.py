from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from crosscut.audio import AlignmentResult, SourceAlignment
from crosscut.draft import DraftSegment, GroupDraftTimeline
from crosscut.ive import AspectRatio
from crosscut.group_transition import GroupTransitionPlan
from crosscut.pipeline import (
    _GroupRenderCandidateBundle,
    _attempt_proxy_triggered_partial_reedit,
    analyze,
)
from crosscut.render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    NormalizedCrop,
    PanCropKeyframe,
    PanCropTrack,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    SourceLookAdjustment,
    TransitionKind,
)
from crosscut.request import AnalysisRequest
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.stage_edit import StageBoundary, StageEdit, StageShot
from crosscut.stage_edit_feature_materializer import StageEditFeatureSnapshot
from crosscut.stage_edit_pipeline_bridge import (
    RetainedStageEditCandidate,
    StageEditPipelineBridgePreflight,
    StageEditPipelineBridgeResult,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditReadinessReport,
    StageEditSequence,
    StageEditToken,
)
from crosscut.stage_edit_visual_candidate_adapter import (
    STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT,
    StageEditMaterializedVisualCandidate,
    StageEditVisualCandidateBatch,
    StageEditVisualCandidateError,
    apply_stage_edit_dynamic_crop_to_render_plan,
    materialize_stage_edit_visual_candidates,
)
from crosscut.timeline import ShotType
from crosscut.visual_score import (
    FrameVisualDescriptor,
    VisualTimelinePlan,
    build_prescribed_visual_timeline_candidate_set,
)
from engine.tests.test_pipeline import (
    _MemoryReporter,
    _analysis_media,
    _visual_choice,
)


def _section_plan() -> SectionPlan:
    return SectionPlan(
        (
            SectionDirective(
                section_index=0,
                role=SectionRole.CHORUS_LIKE,
                start_seconds=0.0,
                end_seconds=4.0,
                energy_score=0.8,
                confidence=0.9,
                phrase_count=1,
                preferred_source_hold_beats=2,
                target_source_switches_per_minute=20.0,
                camera_motion_budget=0.6,
                target_roll_events_per_minute=0.0,
                establish_with_wide_shot=False,
                protect_last_beat_from_extra_cut=False,
            ),
        )
    )


def _snapshot() -> StageEditFeatureSnapshot:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    source_candidates = tuple(
        SourceCandidateToken(
            source_id=source_id,
            features=(0.2 + index * 0.1,) * schema.raw_source_dimension,
            missing_mask=(False,) * schema.raw_source_dimension,
            available=True,
            hard_gate_passed=True,
        )
        for index, source_id in enumerate(("source-a", "source-b"))
    )
    return StageEditFeatureSnapshot.build(
        StageEditSequence(
            "ive:visual-promotion-test",
            (
                StageEditToken(
                    master_time_seconds=0.0,
                    music_features=(0.1,) * len(schema.music_feature_names),
                    section_features=(0.2,) * len(schema.section_feature_names),
                    source_candidates=source_candidates,
                    recent_edit_features=(0.0,)
                    * len(schema.recent_edit_feature_names),
                    style_features=(0.0,) * len(schema.style_feature_names),
                    missing_mask=(False,) * schema.raw_context_dimension,
                ),
            ),
        )
    )


def _dynamic_track(reason: str = "dynamic_crop_v2_test") -> EditorialTransformTrack:
    return EditorialTransformTrack(
        (
            EditorialTransformKeyframe(0.0, 1.04, 0.50, 0.50, 0.0),
            EditorialTransformKeyframe(2.0, 1.10, 0.52, 0.49, 0.0),
        ),
        reason=reason,
    )


def _edit(sources: tuple[Path, ...]) -> StageEdit:
    return StageEdit(
        sources=sources,
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        width=1920,
        height=1080,
        fps=30.0,
        master_audio_start=0.0,
        section_plan=_section_plan(),
        shots=(
            StageShot(
                0,
                0,
                RenderSegment(
                    source_index=0,
                    source_start=0.0,
                    duration=2.0,
                    master_start=0.0,
                    editorial_transform=_dynamic_track(),
                ),
                "learned-shot-0",
                0.9,
            ),
            StageShot(
                1,
                0,
                RenderSegment(1, 2.0, 2.0, 2.0),
                "learned-shot-1",
                0.8,
            ),
        ),
        boundaries=(
            StageBoundary(
                0,
                RenderTransition(tier="C", reason="learned_hard_cut"),
            ),
        ),
        output=Path("/tmp/stage-edit-visual-promotion.mp4"),
        candidate_history=({"decoder": "semi_markov_test"},),
    )


def _bridge_result(count: int = 8) -> StageEditPipelineBridgeResult:
    sources = (
        Path("/tmp/stage-edit-visual-a.mp4").resolve(),
        Path("/tmp/stage-edit-visual-b.mp4").resolve(),
    )
    snapshot = _snapshot()
    checkpoint = Path("/tmp/stage-edit-visual-checkpoint").resolve()
    preflight = StageEditPipelineBridgePreflight(
        snapshot_path=Path("/tmp/stage-edit-visual-snapshot.json").resolve(),
        checkpoint_directory=checkpoint,
        source_ids=("source-a", "source-b"),
        snapshot=snapshot,
        readiness=StageEditReadinessReport(
            architecture_available=True,
            learned_inference_ready=True,
            status="production_checkpoint_ready",
            reason="test",
            mlx_version="test",
            checkpoint_path=str(checkpoint),
            training_steps=10,
        ),
    )
    retained = tuple(
        RetainedStageEditCandidate(
            rank=index + 1,
            score=10.0 - index,
            edit=_edit(sources),
            exact_clock_reprojected=True,
            timewarped_segment_count=0,
        )
        for index in range(count)
    )
    return StageEditPipelineBridgeResult(
        preflight=preflight,
        head_materialization=SimpleNamespace(to_dict=lambda: {"test": True}),
        semimarkov=SimpleNamespace(to_dict=lambda: {"test": True}),
        retained_candidates=retained,
        selection_policy={
            "selected_for_render": False,
            "visual_baseline_replaced": False,
        },
    )


def _visual_fixture(*, passed: bool = True):
    review = SimpleNamespace(
        passed=passed,
        hard_failures=()
        if passed
        else (SimpleNamespace(code="focus_mismatch_exceeded"),),
        to_dict=lambda: {
            "passed": passed,
            "hard_failures": []
            if passed
            else [{"code": "focus_mismatch_exceeded"}],
        },
    )
    record = SimpleNamespace(
        optimizer_score=4.0,
        review=review,
        path_signature=("0:0", "1:1"),
        to_dict=lambda: {
            "candidate_index": 0,
            "review_rank": 0,
            "review": review.to_dict(),
        },
    )
    candidate_set = SimpleNamespace(candidates=(record,))
    base_materialized = SimpleNamespace(
        visual_plan=VisualTimelinePlan(choices=(), optimizer_score=4.0),
        segments=(
            RenderSegment(0, 0.0, 2.0, 0.0),
            RenderSegment(1, 2.0, 2.0, 2.0),
        ),
    )
    return candidate_set, base_materialized


class StageEditVisualCandidateAdapterTests(unittest.TestCase):
    def test_prescribed_builder_runs_real_visual_dp_and_never_changes_source_path(
        self,
    ) -> None:
        descriptor = FrameVisualDescriptor(
            quality_score=0.9,
            composition_score=0.9,
            shot_scale=0.35,
            shot_type=ShotType.GROUP_WIDE,
            face_count=0,
            structure_embedding=(0.1, 0.2, 0.3),
            motion=(0.0, 0.0),
            color_embedding=(0.2, 0.3, 0.4),
            detected_person_count=6,
            person_layout_embedding=(0.1, 0.2, 0.3, 0.4),
            silhouette_embedding=(0.1, 0.2),
            silhouette_bbox_xywh=(0.2, 0.1, 0.6, 0.8),
            silhouette_centroid_xy=(0.5, 0.5),
            silhouette_area_ratio=0.4,
            silhouette_confidence=0.9,
            silhouette_backend="synthetic",
        )

        class _Sampler:
            frame_rate = 30.0

            def __init__(self, *_args, **_kwargs):
                pass

            def feature_descriptor_at(self, *_args, **_kwargs):
                return descriptor

            def descriptor_at(self, *_args, **_kwargs):
                return descriptor

            def sface_embedding_at(self, *_args, **_kwargs):
                return ()

            def close(self):
                pass

        sources = tuple(Path(f"/tmp/prescribed-{index}.mp4") for index in range(8))
        spans = tuple((float(index), float(index + 1)) for index in range(60))
        safe_path = tuple((index // 4) % 8 for index in range(60))
        with patch("crosscut.visual_score._VideoSampler", _Sampler):
            candidate_set = build_prescribed_visual_timeline_candidate_set(
                sources=sources,
                canonical_spans=spans,
                prescribed_source_indices=safe_path,
                audio_offsets=(0.0,) * 8,
                sync_scores=(0.9,) * 8,
                scene_cut_times=((),) * 8,
                source_timewarps=(None,) * 8,
            )
        record = candidate_set.candidates[0]
        self.assertTrue(record.review.passed)
        self.assertEqual(
            tuple(str(value) for value in safe_path),
            tuple(item.source_id for item in record.timeline_result.candidates),
        )
        self.assertEqual(
            "prescribed_source_path_visual_hard_gates_then_reference_review_v1",
            candidate_set.candidate_generation_basis,
        )

        unsafe_path = (0,) * 60
        with (
            patch("crosscut.visual_score._VideoSampler", _Sampler),
            self.assertRaisesRegex(ValueError, "no cadence-safe candidate"),
        ):
            build_prescribed_visual_timeline_candidate_set(
                sources=sources,
                canonical_spans=spans,
                prescribed_source_indices=unsafe_path,
                audio_offsets=(0.0,) * 8,
                sync_scores=(0.9,) * 8,
                scene_cut_times=((),) * 8,
                source_timewarps=(None,) * 8,
            )

    def test_exact_eight_paths_use_prescribed_visual_gate_without_baseline_substitution(
        self,
    ) -> None:
        bridge = _bridge_result()
        candidate_set, base_materialized = _visual_fixture()
        with (
            patch(
                "crosscut.stage_edit_visual_candidate_adapter."
                "build_prescribed_visual_timeline_candidate_set",
                return_value=candidate_set,
            ) as prescribed,
            patch(
                "crosscut.stage_edit_visual_candidate_adapter."
                "materialize_visual_timeline_candidates",
                return_value=(base_materialized,),
            ) as materialize,
        ):
            batch = materialize_stage_edit_visual_candidates(
                bridge,
                sources=bridge.retained_candidates[0].edit.sources,
                audio_offsets=(0.0, 0.0),
                sync_scores=(0.9, 0.8),
                scene_cut_times=((), ()),
                source_timewarps=(None, None),
            )

        self.assertEqual(8, prescribed.call_count)
        self.assertEqual(8, materialize.call_count)
        self.assertEqual(list(range(8)), [item.candidate_index for item in batch.candidates])
        for call in prescribed.call_args_list:
            self.assertEqual((0, 1), call.kwargs["prescribed_source_indices"])
            self.assertEqual(
                ((0.0, 2.0), (2.0, 4.0)),
                call.kwargs["canonical_spans"],
            )
        payload = batch.to_dict()
        self.assertEqual("stage_edit_generation_only", payload["candidate_pool"])
        self.assertFalse(payload["ordinary_visual_baseline_candidate_pool_eligible"])
        self.assertFalse(payload["silent_baseline_fallback_allowed"])

    def test_short_batch_and_failed_visual_review_fail_closed(self) -> None:
        with self.assertRaisesRegex(StageEditVisualCandidateError, "exactly 8"):
            materialize_stage_edit_visual_candidates(
                _bridge_result(7),
                sources=_bridge_result(7).retained_candidates[0].edit.sources,
                audio_offsets=(0.0, 0.0),
                sync_scores=(0.9, 0.8),
                scene_cut_times=((), ()),
                source_timewarps=(None, None),
            )

        bridge = _bridge_result()
        failed_set, base_materialized = _visual_fixture(passed=False)
        with (
            patch(
                "crosscut.stage_edit_visual_candidate_adapter."
                "build_prescribed_visual_timeline_candidate_set",
                return_value=failed_set,
            ),
            patch(
                "crosscut.stage_edit_visual_candidate_adapter."
                "materialize_visual_timeline_candidates",
                return_value=(base_materialized,),
            ),
            self.assertRaisesRegex(
                StageEditVisualCandidateError,
                "deterministic visual timeline review",
            ),
        ):
            materialize_stage_edit_visual_candidates(
                bridge,
                sources=bridge.retained_candidates[0].edit.sources,
                audio_offsets=(0.0, 0.0),
                sync_scores=(0.9, 0.8),
                scene_cut_times=((), ()),
                source_timewarps=(None, None),
            )

    def test_dynamic_crop_track_survives_and_reparameterizes_targeted_repair(
        self,
    ) -> None:
        bridge = _bridge_result()
        candidate_set, base_materialized = _visual_fixture()
        with (
            patch(
                "crosscut.stage_edit_visual_candidate_adapter."
                "build_prescribed_visual_timeline_candidate_set",
                return_value=candidate_set,
            ),
            patch(
                "crosscut.stage_edit_visual_candidate_adapter."
                "materialize_visual_timeline_candidates",
                return_value=(base_materialized,),
            ),
        ):
            candidate = materialize_stage_edit_visual_candidates(
                bridge,
                sources=bridge.retained_candidates[0].edit.sources,
                audio_offsets=(0.0, 0.0),
                sync_scores=(0.9, 0.8),
                scene_cut_times=((), ()),
                source_timewarps=(None, None),
            ).candidates[0]

        baseline_track = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.01),
                EditorialTransformKeyframe(1.5, 1.02),
            ),
            reason="ordinary_editorial_motion",
        )
        render_plan = RenderPlan(
            sources=candidate.retained_candidate.edit.sources,
            segments=(
                RenderSegment(0, 0.0, 1.5, 0.0, editorial_transform=baseline_track),
                RenderSegment(1, 1.5, 2.5, 1.5, editorial_transform=baseline_track),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("/tmp/stage-edit-targeted-repair.mp4"),
            master_audio_start=0.0,
            width=1920,
            height=1080,
            fps=30.0,
            transitions=(RenderTransition(tier="C", reason="hard_cut"),),
        )
        promoted = apply_stage_edit_dynamic_crop_to_render_plan(candidate, render_plan)

        first = promoted.segments[0].editorial_transform
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(1.5, first.keyframes[-1].time_seconds)
        self.assertEqual(1.10, first.keyframes[-1].zoom)
        self.assertIn("targeted_repair_reparameterized", first.reason)
        self.assertIsNone(promoted.segments[1].editorial_transform)

    def test_verified_face_dissolve_alignment_precedes_dynamic_crop_overlay(
        self,
    ) -> None:
        bridge = _bridge_result()
        candidate_set, base_materialized = _visual_fixture()
        with (
            patch(
                "crosscut.stage_edit_visual_candidate_adapter."
                "build_prescribed_visual_timeline_candidate_set",
                return_value=candidate_set,
            ),
            patch(
                "crosscut.stage_edit_visual_candidate_adapter."
                "materialize_visual_timeline_candidates",
                return_value=(base_materialized,),
            ),
        ):
            candidate = materialize_stage_edit_visual_candidates(
                bridge,
                sources=bridge.retained_candidates[0].edit.sources,
                audio_offsets=(0.0, 0.0),
                sync_scores=(0.9, 0.8),
                scene_cut_times=((), ()),
                source_timewarps=(None, None),
            ).candidates[0]

        outgoing_alignment = PanCropTrack(
            (
                PanCropKeyframe(
                    0.0,
                    NormalizedCrop(0.08, 0.06, 0.84, 0.84),
                ),
                PanCropKeyframe(
                    2.0,
                    NormalizedCrop(0.10, 0.06, 0.84, 0.84),
                ),
            )
        )
        incoming_alignment = PanCropTrack(
            (
                PanCropKeyframe(
                    0.0,
                    NormalizedCrop(0.09, 0.05, 0.84, 0.84),
                ),
                PanCropKeyframe(
                    2.0,
                    NormalizedCrop(0.10, 0.05, 0.84, 0.84),
                ),
            )
        )
        render_plan = RenderPlan(
            sources=candidate.retained_candidate.edit.sources,
            segments=(
                RenderSegment(0, 0.0, 2.0, 0.0, pan_crop=outgoing_alignment),
                RenderSegment(1, 2.0, 2.0, 2.0, pan_crop=incoming_alignment),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("/tmp/stage-edit-face-dissolve-alignment.mp4"),
            master_audio_start=0.0,
            width=1920,
            height=1080,
            fps=30.0,
            transitions=(
                RenderTransition(
                    TransitionKind.FACE_DISSOLVE,
                    9,
                    "verified_face_alignment",
                    face_mismatch=0.08,
                    tier="A",
                ),
            ),
        )

        promoted = apply_stage_edit_dynamic_crop_to_render_plan(
            candidate,
            render_plan,
        )

        self.assertIs(promoted.segments[0].pan_crop, outgoing_alignment)
        self.assertIs(promoted.segments[1].pan_crop, incoming_alignment)
        self.assertIsNone(promoted.segments[0].editorial_transform)
        self.assertIsNone(promoted.segments[1].editorial_transform)
        self.assertIs(
            promoted.transitions[0].kind,
            TransitionKind.FACE_DISSOLVE,
        )

    def test_request_rejects_ambiguous_generated_and_baseline_rerank_mix(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sources = []
            for index in range(8):
                source = root / f"source-{index}.mp4"
                source.touch()
                sources.append(str(source))
            generation_checkpoint = root / "generation-checkpoint"
            generation_checkpoint.mkdir()
            generation_snapshot = root / "snapshot.json"
            generation_snapshot.write_text("{}", encoding="utf-8")
            rerank_checkpoint = root / "rerank-checkpoint"
            rerank_checkpoint.mkdir()
            report = root / "production-report.json"
            report.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "split": {
                            "training_reference_ids": ["a-train", "b-train"],
                            "validation_reference_ids": [
                                "a-validation",
                                "b-validation",
                            ],
                            "held_out_reference_ids": [
                                "c-heldout-1",
                                "c-heldout-2",
                            ],
                            "model_selection_validation_level": (
                                "whole_reference_within_training_songs"
                            ),
                            "held_out_used_for_model_selection": False,
                        },
                        "optimization": {
                            "model_selection": (
                                "minimum_whole_reference_validation_loss_"
                                "within_training_songs"
                            ),
                            "unseen_song_used_for_model_selection": False,
                        },
                        "production_supervision_gate": {
                            "scope": (
                                "training_songs_only_including_reference_validation"
                            ),
                            "held_out_references_included": False,
                        },
                        "production_feature_snapshot_gate": {
                            "passed": True,
                            "contract": (
                                "same_causal_media_feature_snapshot_"
                                "train_and_inference_v1"
                            ),
                            "reference_count": 6,
                            "feature_snapshot_sha256_by_reference": {
                                "a-train": "a" * 64,
                                "b-train": "b" * 64,
                                "a-validation": "c" * 64,
                                "b-validation": "d" * 64,
                                "c-heldout-1": "e" * 64,
                                "c-heldout-2": "f" * 64,
                            },
                            "reference_match_or_current_choice_features_present": (
                                False
                            ),
                            "missing_reference_ids": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cannot be enabled together"):
                AnalysisRequest.from_dict(
                    {
                        "song": "BANG BANG",
                        "edit_mode": "group",
                        "aspect_ratio": "16:9",
                        "source_mode": "automatic",
                        "desired_source_count": 8,
                        "sources": sources,
                        "output_directory": str(root / "output"),
                        "stage_edit_checkpoint_directory": str(rerank_checkpoint),
                        "stage_edit_production_training_report": str(report),
                        "stage_edit_generation_checkpoint_directory": str(
                            generation_checkpoint
                        ),
                        "stage_edit_generation_feature_snapshot": str(
                            generation_snapshot
                        ),
                        "stage_edit_generation_source_ids": [
                            f"source-{index}" for index in range(8)
                        ],
                    }
                )

    def test_targeted_proxy_repair_reapplies_candidate_render_contract(self) -> None:
        sources = (
            Path("/tmp/stage-edit-repair-a.mp4"),
            Path("/tmp/stage-edit-repair-b.mp4"),
        )
        original_visual = VisualTimelinePlan(
            choices=(
                SimpleNamespace(
                    source_index=0,
                    canonical_master_start=0.0,
                    canonical_master_end=1.0,
                ),
                SimpleNamespace(
                    source_index=1,
                    canonical_master_start=1.0,
                    canonical_master_end=2.0,
                ),
            ),
            optimizer_score=8.0,
        )
        repaired_visual = VisualTimelinePlan(
            choices=(
                SimpleNamespace(
                    source_index=0,
                    canonical_master_start=0.0,
                    canonical_master_end=1.1,
                ),
                SimpleNamespace(
                    source_index=1,
                    canonical_master_start=1.1,
                    canonical_master_end=2.0,
                ),
            ),
            optimizer_score=8.0,
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
            output=Path("/tmp/stage-edit-repair-original.mp4"),
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
            output=Path("/tmp/stage-edit-repair-partial.mp4"),
            transitions=transitions.transitions,
        )
        original_bundle = _GroupRenderCandidateBundle(
            candidate_index=2,
            review_rank=0,
            cadence_profile="stage_edit",
            pre_render_score=90.0,
            visual_timeline=original_visual,
            boundary_repairs=None,
            timeline_review=None,
            face_match_plan=None,
            group_transition_plan=transitions,
            boundary_affine_plan=None,
            editorial_motion_plan=None,
            render_plan=original_plan,
            candidate_origin="explicit_stage_edit_generation",
            candidate_lineage=({"stage": "initial"},),
        )
        partial_bundle = _GroupRenderCandidateBundle(
            candidate_index=2,
            review_rank=0,
            cadence_profile="stage_edit",
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
        promoted_bundle = _GroupRenderCandidateBundle(
            candidate_index=2,
            review_rank=0,
            cadence_profile="stage_edit",
            pre_render_score=89.0,
            visual_timeline=repaired_visual,
            boundary_repairs=None,
            timeline_review=None,
            face_match_plan=None,
            group_transition_plan=transitions,
            boundary_affine_plan=None,
            editorial_motion_plan=None,
            render_plan=partial_plan,
            candidate_origin="explicit_stage_edit_generation",
            candidate_lineage=(
                {"stage": "initial"},
                {"stage": "dynamic_crop_v2_render_plan_overlay"},
            ),
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
            to_dict=lambda: {"requested_boundary_indices": [0]},
        )
        original_review = {
            "passed": False,
            "weighted_score": 70.0,
            "partial_reedit_boundary_issue_indices": [],
            "visible_boundary_score": {"issue_fraction": 0.5, "p10": 50.0},
        }
        after_review = {
            "passed": True,
            "weighted_score": 80.0,
            "partial_reedit_boundary_issue_indices": [],
            "visible_boundary_score": {"issue_fraction": 0.0, "p10": 80.0},
        }
        postprocessor = Mock(return_value=promoted_bundle)
        with (
            patch(
                "crosscut.pipeline.repair_visual_timeline_boundaries",
                return_value=targeted,
            ),
            patch(
                "crosscut.pipeline._derive_complete_group_candidate",
                return_value=partial_bundle,
            ),
            patch(
                "crosscut.pipeline.review_rendered_candidates",
                return_value=SimpleNamespace(
                    selected=SimpleNamespace(review_payload=after_review)
                ),
            ),
        ):
            result = _attempt_proxy_triggered_partial_reedit(
                original_bundle,
                SimpleNamespace(candidate_index=2),
                original_review,
                request=SimpleNamespace(
                    sources=sources,
                    reference_source=0,
                    aspect_ratio=AspectRatio.LANDSCAPE,
                ),
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
                scratch_directory=Path("/tmp"),
                learned_issue_boundary_indices=(0,),
                candidate_bundle_postprocessor=postprocessor,
            )

        postprocessor.assert_called_once_with(
            partial_bundle,
            unittest.mock.ANY,
        )
        self.assertEqual(
            promoted_bundle.render_plan,
            result.selected_bundle.render_plan,
        )
        self.assertEqual(
            promoted_bundle.candidate_origin,
            result.selected_bundle.candidate_origin,
        )
        self.assertEqual("partial_reedit", result.audit_payload["selected_version"])
        self.assertEqual(
            [0],
            result.audit_payload["learned_reviewer_issue_suggestions"][
                "used_boundary_indices"
            ],
        )
        self.assertEqual(
            "partial_reedit_boundary_issue_indices_plus_learned_reviewer_suggestions",
            result.audit_payload["issue_index_source_field"],
        )
        self.assertEqual(
            "decoded_proxy_issue_targeted_boundary_repair",
            result.audit_payload["candidate_lineage_after"][-2]["stage"],
        )
        self.assertEqual(
            "dynamic_crop_v2_render_plan_overlay",
            result.audit_payload["candidate_lineage_after"][-1]["stage"],
        )

    def test_pipeline_promotes_only_stage_edit_batch_after_exact_proxy_review(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sources = tuple(root / f"source-{index}.mp4" for index in range(8))
            for source in sources:
                source.touch()
            checkpoint = root / "generation-checkpoint"
            checkpoint.mkdir()
            snapshot = root / "snapshot.json"
            snapshot.write_text("{}", encoding="utf-8")
            reviewer_checkpoint = root / "reviewer-checkpoint"
            reviewer_checkpoint.mkdir()
            reviewer_report = root / "reviewer-report.json"
            reviewer_report.write_text("{}", encoding="utf-8")
            request = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "desired_source_count": 8,
                    "sources": [str(source) for source in sources],
                    "output_directory": str(root / "out"),
                    "stage_edit_generation_checkpoint_directory": str(checkpoint),
                    "stage_edit_generation_feature_snapshot": str(snapshot),
                    "stage_edit_generation_source_ids": [
                        f"source-{index}" for index in range(8)
                    ],
                    "stage_edit_learned_reviewer_checkpoint_directory": str(
                        reviewer_checkpoint
                    ),
                    "stage_edit_learned_reviewer_production_training_report": str(
                        reviewer_report
                    ),
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
                source_count=8,
                common_master_start=0.0,
                common_master_end=8.0,
                beat_period_seconds=0.5,
                audio_offsets=(0.0,) * 8,
                sync_scores=(1.0,) * 8,
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
            section_plan = SectionPlan(
                (
                    SectionDirective(
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
                    ),
                )
            )

            class _Rhythm:
                bpm = 120.0
                beats = (object(),)
                phrase_candidates: tuple[object, ...] = ()
                beat_times_seconds = (0.0, 8.0)

                @staticmethod
                def to_dict():
                    return {}

            class _Review:
                passed = True
                weighted_score = 90.0
                hard_failures: tuple[object, ...] = ()
                source_switches_per_minute = 14.0

                @staticmethod
                def to_dict():
                    return {"passed": True, "weighted_score": 90.0}

            transitions = GroupTransitionPlan(())
            ordinary_bundles = []
            promoted_bundles = []
            candidate_mocks = []
            for index in range(8):
                visual = VisualTimelinePlan(
                    (_visual_choice(0, index, 0.0, 8.0),),
                    float(100 - index),
                    selected_candidate_index=index,
                    analysis_basis=STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT,
                )
                plan = RenderPlan(
                    sources=sources,
                    segments=(RenderSegment(index, 0.0, 8.0, 0.0),),
                    reference_source=0,
                    aspect_ratio=AspectRatio.LANDSCAPE,
                    output=root / f"candidate-{index}.mp4",
                    transitions=(),
                )
                ordinary = _GroupRenderCandidateBundle(
                    candidate_index=index,
                    review_rank=index,
                    cadence_profile="stage_edit",
                    pre_render_score=float(100 - index),
                    visual_timeline=visual,
                    boundary_repairs=None,
                    timeline_review=_Review(),
                    face_match_plan=None,
                    group_transition_plan=transitions,
                    boundary_affine_plan=None,
                    editorial_motion_plan=None,
                    render_plan=plan,
                )
                ordinary_bundles.append(ordinary)
                promoted_bundles.append(
                    _GroupRenderCandidateBundle(
                        candidate_index=index,
                        review_rank=index,
                        cadence_profile="stage_edit",
                        pre_render_score=float(100 - index),
                        visual_timeline=visual,
                        boundary_repairs=None,
                        timeline_review=_Review(),
                        face_match_plan=None,
                        group_transition_plan=transitions,
                        boundary_affine_plan=None,
                        editorial_motion_plan=None,
                        render_plan=plan,
                        candidate_origin="explicit_stage_edit_generation",
                        candidate_lineage=(
                            {
                                "stage": (
                                    "dynamic_crop_v2_render_plan_overlay"
                                )
                            },
                        ),
                    )
                )
                candidate = Mock(spec=StageEditMaterializedVisualCandidate)
                candidate.candidate_index = index
                candidate.review_rank = index
                candidate.candidate_record = SimpleNamespace(
                    review=SimpleNamespace(passed=True)
                )
                candidate.to_dict.return_value = {
                    "candidate_index": index,
                    "candidate_origin": "explicit_stage_edit_generation",
                }
                candidate_mocks.append(candidate)
            batch = StageEditVisualCandidateBatch(tuple(candidate_mocks))

            def _write_generation(path: Path):
                path.write_text("{}", encoding="utf-8")
                return path

            generation = SimpleNamespace(
                retained_candidates=tuple(object() for _ in range(8)),
                learned_section_director=SimpleNamespace(
                    evidence=(object(),),
                    to_dict=lambda: {"section_count": 1},
                ),
                write_json=_write_generation,
                to_dict=lambda: {
                    "retained_candidate_count": 8,
                    "selection_policy": {
                        "selected_for_render": False,
                        "visual_baseline_replaced": False,
                    },
                },
            )
            look_result = SimpleNamespace(
                adjustments=tuple(SourceLookAdjustment() for _ in sources),
                target_mode="explicit_reference",
                adjustment_for_source=lambda _index: SourceLookAdjustment(),
                to_dict=lambda: {"target_mode": "explicit_reference"},
            )

            class _Selection:
                selected_candidate_index = 6
                reviews = tuple(object() for _ in range(8))
                successful_review_count = 8
                minimum_candidate_count_satisfied = True
                learned_reviewer_rerank = {
                    "selected_repair_issue_boundary_indices": [0],
                    "selected_repair_issues": [
                        {
                            "issue_id": "learned-boundary-0",
                            "start_seconds": 3.8,
                            "end_seconds": 4.2,
                            "causes": ["learned_editorial_discontinuity"],
                            "severity": "error",
                        }
                    ],
                    "selection_mode": (
                        "learned_quality_within_deterministic_hard_safe_pool"
                    ),
                }
                selected = SimpleNamespace(
                    review_payload={
                        "passed": True,
                        "weighted_score": 96.0,
                        "partial_reedit_boundary_issue_indices": [],
                    }
                )

                @staticmethod
                def to_dict():
                    return {
                        "selected_candidate_index": 6,
                        "reviewed_candidate_count": 8,
                        "passed": True,
                    }

            reviewer_session = SimpleNamespace(
                metadata=SimpleNamespace(checkpoint_id="reviewer-v1"),
                evidence={"bundle_sha256": "b" * 64},
            )

            def _stage_beam_repair(bundle, _materialized, review, **kwargs):
                self.assertEqual(
                    "learned-boundary-0",
                    kwargs["learned_repair_issues"][0]["issue_id"],
                )
                self.assertIs(
                    reviewer_session,
                    kwargs["learned_reviewer_session"],
                )
                return SimpleNamespace(
                    selected_bundle=bundle,
                    selected_review_payload=review,
                    audit_payload={
                        "analysis_basis": (
                            "reviewer_issue_window_actual_semimarkov_redecode_v1"
                        ),
                        "attempted": True,
                        "actual_local_semimarkov_redecode": True,
                        "selected_version": "original",
                        "reason": "no_strict_improvement",
                    },
                )

            derivative_metadata = SimpleNamespace(
                production_ready=True,
                family_count=8,
                confirmed_derivative_pair_count=0,
                uncertain_pair_count=0,
                decoder_metadata=lambda: SimpleNamespace(
                    source_count=8,
                    family_ids=tuple(range(8)),
                    production_ready=True,
                ),
                to_dict=lambda: {
                    "analysis_basis": "synthetic_actual_pts_derivative_test",
                    "production_ready": True,
                    "family_ids": list(range(8)),
                },
            )

            with (
                patch(
                    "crosscut.pipeline._load_learned_reviewer_session",
                    return_value=reviewer_session,
                ),
                patch(
                    "crosscut.pipeline._load_stage_edit_generation_preflight",
                    return_value=SimpleNamespace(
                        snapshot=SimpleNamespace(snapshot_sha256="a" * 64),
                        source_ids=tuple(
                            f"source-{index}" for index in range(8)
                        ),
                    ),
                ),
                patch(
                    "crosscut.pipeline.probe_media",
                    side_effect=lambda source: _analysis_media(source, duration=12.0),
                ),
                patch(
                    "crosscut.pipeline._align_sequentially",
                    return_value=(alignment, None),
                ),
                patch("crosscut.pipeline.analyze_rhythm", return_value=_Rhythm()),
                patch(
                    "crosscut.pipeline.detect_scene_changes_in_video",
                    return_value=(),
                ),
                patch(
                    "crosscut.pipeline.estimate_shared_performance_range",
                    side_effect=ValueError("synthetic range unavailable"),
                ),
                patch("crosscut.pipeline.build_section_plan", return_value=section_plan),
                patch.multiple(
                    "crosscut.pipeline",
                    build_group_draft=lambda *_args, **_kwargs: group_draft,
                    analyze_source_derivative_diversity=(
                        lambda *_args, **_kwargs: derivative_metadata
                    ),
                ),
                patch(
                    "crosscut.pipeline.run_stage_edit_pipeline_bridge",
                    return_value=generation,
                ) as run_stage_bridge,
                patch(
                    "crosscut.pipeline.analyze_source_look_match",
                    return_value=look_result,
                ),
                patch(
                    "crosscut.pipeline.materialize_stage_edit_visual_candidates",
                    return_value=batch,
                ) as materialize_stage,
                patch(
                    "crosscut.pipeline.build_visual_timeline_candidate_set"
                ) as build_baseline,
                patch(
                    "crosscut.pipeline._derive_complete_group_candidate",
                    side_effect=lambda item, **_kwargs: ordinary_bundles[
                        item.candidate_index
                    ],
                ),
                patch(
                    "crosscut.pipeline._apply_stage_edit_candidate_render_contract",
                    side_effect=lambda _bundle, item: promoted_bundles[
                        item.candidate_index
                    ],
                ) as apply_contract,
                patch(
                    "crosscut.pipeline._review_complete_group_candidates",
                    return_value=_Selection(),
                ) as review_candidates,
                patch(
                    "crosscut.pipeline._attempt_stage_edit_windowed_beam_reedit",
                    side_effect=_stage_beam_repair,
                ) as stage_beam_repair,
                patch("crosscut.pipeline.write_plan"),
                patch(
                    "crosscut.pipeline.render",
                    return_value="h264_videotoolbox",
                ),
            ):
                artifact = analyze(
                    request,
                    _MemoryReporter(),
                    review_rendered_proxies=True,
                )

            materialize_stage.assert_called_once()
            run_stage_bridge.assert_called_once()
            self.assertTrue(
                run_stage_bridge.call_args.kwargs[
                    "require_verified_source_derivative_metadata"
                ]
            )
            self.assertIsNotNone(
                run_stage_bridge.call_args.kwargs["source_derivative_metadata"]
            )
            build_baseline.assert_not_called()
            self.assertEqual(8, apply_contract.call_count)
            review_candidates.assert_called_once()
            stage_beam_repair.assert_called_once()
            reviewed = review_candidates.call_args.args[0]
            self.assertEqual(8, len(reviewed))
            self.assertIs(
                reviewer_session,
                review_candidates.call_args.kwargs["learned_reviewer_session"],
            )
            self.assertTrue(
                all(
                    item.candidate_origin == "explicit_stage_edit_generation"
                    for item in reviewed
                )
            )
            selection = artifact.manifest["rendered_candidate_selection"]
            self.assertEqual("stage_edit_generation_only", selection["candidate_pool"])
            self.assertFalse(selection["silent_baseline_fallback_allowed"])
            self.assertEqual(
                "eligible_for_final_render",
                selection["stage_edit_render_promotion"]["promotion_state"],
            )
            self.assertTrue(
                selection["stage_edit_render_promotion"][
                    "all_promotion_hard_gates_satisfied"
                ]
            )
            self.assertTrue(
                selection["stage_edit_render_promotion"][
                    "learned_reviewer_configured"
                ]
            )
            selection_lineage = selection["stage_edit_render_promotion"][
                "selected_candidate_lineage"
            ]
            self.assertEqual(
                [0],
                next(
                    item
                    for item in selection_lineage
                    if item["stage"] == "decoded_proxy_candidate_selection"
                )["learned_reviewer_repair_issue_boundary_indices"],
            )
            self.assertEqual(
                "explicit_stage_edit_generation",
                artifact.manifest["render"]["candidate_origin"],
            )


if __name__ == "__main__":
    unittest.main()
