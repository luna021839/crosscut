from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from engine.tests.test_review_first_runtime import _group_integrity_metrics

from crosscut.audio import AlignmentResult, SourceAlignment
from crosscut.draft import DraftSegment, GroupDraftTimeline
from crosscut.ive import AspectRatio
from crosscut.ive_product_quality_floor import APP_RESULT_EXPOSURE_CONTRACT
from crosscut.group_transition import GroupTransitionPlan
from crosscut.pipeline import (
    AUTOMATIC_STAGE_EDIT_POST_RENDER_QA_CONTRACT,
    _GroupRenderCandidateBundle,
    _BoundarySearchBudget,
    _apply_stage_edit_candidate_render_contract,
    _attempt_proxy_triggered_partial_reedit,
    _enforce_automatic_stage_edit_post_render_qa,
    analyze,
)
from crosscut.post_render_qa import fingerprint_post_render_artifacts
from crosscut.rendered_candidate_selection import (
    RenderedCandidate,
    RenderedCandidateReview,
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
from crosscut.rendered_montage_grammar_review import (
    EXACT_STAGE_EDIT_SHOT_CLOCK_CONTRACT,
)
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.stage_edit import StageBoundary, StageEdit, StageShot
from crosscut.stage_edit_feature_materializer import StageEditFeatureSnapshot
from crosscut.stage_edit_pipeline_bridge import (
    STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT,
    RetainedStageEditCandidate,
    StageEditPipelineBridgePreflight,
    StageEditPipelineBridgeResult,
    load_stage_edit_production_snapshot_identity,
)
from crosscut.stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
)
from crosscut.stage_edit_dynamic_crop_motion_authority_v3 import (
    DynamicCropMotionAuthorityV3,
)
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
    DynamicCropTrackDecodeV3,
)
from crosscut.stage_edit_semimarkov_beam import (
    HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
    HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
    StageEditSemiMarkovConfig,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_CAUSAL_SENSING_SOURCE_FEATURE_NAMES,
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
    _retained_dynamic_crop_version,
    apply_stage_edit_dynamic_crop_to_render_plan,
    materialize_stage_edit_visual_candidate,
    materialize_stage_edit_visual_candidates,
)
from crosscut.timeline import ShotType
from crosscut.visual_score import (
    FrameVisualDescriptor,
    PersistentVisualFeatureCacheInitializationError,
    VisualTimelinePlan,
    build_prescribed_visual_timeline_candidate_set,
)
from engine.tests.test_pipeline import (
    _MemoryReporter,
    _analysis_media,
    _current_montage_policy_fixture,
    _current_motion_cut_profile_fixture,
    _visual_choice,
)
from engine.tests.test_stage_edit_pipeline_bridge import (
    _typed_decode_result,
    _typed_head_result,
    _typed_timelines,
)
from engine.tests.stage_edit_causal_sensing_test_support import (
    synthetic_production_snapshot,
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
    causal_sensing_names = set(STAGE_EDIT_CAUSAL_SENSING_SOURCE_FEATURE_NAMES)
    missing_mask = tuple(
        name in causal_sensing_names for name in schema.source_feature_names
    )
    source_candidates = tuple(
        SourceCandidateToken(
            source_id=source_id,
            features=tuple(
                0.0 if missing else 0.2 + index * 0.1
                for missing in missing_mask
            ),
            missing_mask=missing_mask,
            available=True,
            hard_gate_passed=True,
        )
        for index, source_id in enumerate(("source-a", "source-b"))
    )
    return synthetic_production_snapshot(
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
    snapshot_path = Path("/tmp/stage-edit-visual-snapshot.json").resolve()
    snapshot_path.write_text(
        json.dumps(snapshot.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    snapshot_identity = load_stage_edit_production_snapshot_identity(
        snapshot_path,
        edit_mode="group",
        target_member=None,
    )
    checkpoint = Path("/tmp/stage-edit-visual-checkpoint").resolve()
    preflight = StageEditPipelineBridgePreflight(
        snapshot_path=snapshot_path,
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
            checkpoint_format="dynamic_crop_v2",
            dynamic_crop_version=2,
            checkpoint_metadata_filename="dynamic-crop-v2-metadata.json",
        ),
        snapshot_identity=snapshot_identity,
        edit_mode="group",
        target_member=None,
    )
    retained = tuple(
        RetainedStageEditCandidate(
            rank=index + 1,
            score=10.0 - index,
            edit=replace(
                _edit(sources),
                candidate_history=(
                    *_edit(sources).candidate_history,
                    {
                        "bridge": STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT,
                        "dynamic_crop_version": 2,
                        "continuity_authority": (
                            HANDCRAFTED_CONTINUITY_AUTHORITY
                        ),
                    },
                ),
            ),
            exact_clock_reprojected=True,
            timewarped_segment_count=0,
        )
        for index in range(count)
    )
    config = StageEditSemiMarkovConfig(minimum_unique_sources=1)
    timelines = _typed_timelines(
        sources,
        count=count,
        authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
        scorer_contract=HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
        scorer_fingerprint=HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
    )
    decoded = _typed_decode_result(
        timelines,
        config=config,
        scorer_contract=HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
        scorer_fingerprint=HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
        request_fulfilled=(count == config.timeline_count),
    )
    return StageEditPipelineBridgeResult(
        preflight=preflight,
        head_materialization=replace(
            _typed_head_result(checkpoint),
            sequence_id=snapshot.sequence.sequence_id,
            token_schema_fingerprint=snapshot.token_schema_fingerprint,
            readiness=preflight.readiness,
            training_steps=preflight.readiness.training_steps,
        ),
        semimarkov=decoded,
        retained_candidates=retained,
        selection_policy={
            "selected_for_render": False,
            "visual_baseline_replaced": False,
            "dynamic_crop_version": 2,
            "continuity_authority": HANDCRAFTED_CONTINUITY_AUTHORITY,
            "continuity_scorer_contract": (
                HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
            ),
            "continuity_scorer_fingerprint": (
                HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
            ),
            "temporal_pose_sidecar": snapshot.temporal_pose_sidecar_audit(),
        },
    )


def _visual_fixture(*, passed: bool = True):
    from engine.tests.test_pipeline import _visual_choice
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
    candidate_set = SimpleNamespace(
        candidates=(record,),
        to_dict=lambda: {
            "candidate_count": 1,
            "selected_candidate_index": 0,
        },
    )
    base_materialized = SimpleNamespace(
        visual_plan=VisualTimelinePlan(choices=(_visual_choice(0,0,0.,2.),
            _visual_choice(1,1,2.,4.)), optimizer_score=4.0),
        segments=(
            RenderSegment(0, 0.0, 2.0, 0.0),
            RenderSegment(1, 2.0, 2.0, 2.0),
        ),
    )
    return candidate_set, base_materialized


def _typed_visual_batch(
    *,
    quality_gated_single_delivery: bool = False,
) -> StageEditVisualCandidateBatch:
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
        return materialize_stage_edit_visual_candidates(
            bridge,
            sources=bridge.retained_candidates[0].edit.sources,
            audio_offsets=(0.0, 0.0),
            sync_scores=(0.9, 0.8),
            scene_cut_times=((), ()),
            source_timewarps=(None, None),
            quality_gated_single_delivery=quality_gated_single_delivery,
        )


class StageEditVisualCandidateAdapterTests(unittest.TestCase):
    def test_visual_batch_direct_constructor_rejects_fake_records_and_versions(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "typed materialized candidates",
        ):
            StageEditVisualCandidateBatch(
                tuple(SimpleNamespace(candidate_index=index) for index in range(8))
            )

        valid = _typed_visual_batch()
        first = valid.candidates[0]
        with self.assertRaisesRegex(TypeError, "typed candidate records"):
            StageEditVisualCandidateBatch(
                (
                    replace(
                        first,
                        candidate_record=SimpleNamespace(
                            review=SimpleNamespace(passed=True)
                        ),
                    ),
                    *valid.candidates[1:],
                )
            )

        invalid_retained = replace(
            first.retained_candidate,
            edit=replace(
                first.retained_candidate.edit,
                candidate_history=(
                    {
                        "bridge": STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT,
                        "dynamic_crop_version": 4,
                        "continuity_authority": (
                            HANDCRAFTED_CONTINUITY_AUTHORITY
                        ),
                    },
                ),
            ),
        )
        invalid_candidate = replace(
            first,
            candidate_record=replace(
                first.candidate_record,
                retained_candidate=invalid_retained,
            ),
            retained_candidate=invalid_retained,
        )
        with self.assertRaisesRegex(
            StageEditVisualCandidateError,
            "invalid Dynamic Crop version",
        ):
            StageEditVisualCandidateBatch(
                (invalid_candidate, *valid.candidates[1:])
            )

    def test_bridge_result_rejects_v3_preflight_with_v2_candidate_history(
        self,
    ) -> None:
        bridge = _bridge_result()
        v3_readiness = replace(
            bridge.preflight.readiness,
            checkpoint_format="dynamic_crop_v3",
            dynamic_crop_version=3,
            checkpoint_metadata_filename="dynamic-crop-v3-metadata.json",
        )
        v3_preflight = replace(
            bridge.preflight,
            readiness=v3_readiness,
            dynamic_crop_version=3,
        )
        with self.assertRaisesRegex(
            ValueError, "retained candidate version differs"
        ):
            replace(
                bridge,
                preflight=v3_preflight,
                head_materialization=replace(
                    bridge.head_materialization,
                    readiness=v3_readiness,
                    checkpoint_format="dynamic_crop_v3",
                    dynamic_crop_version=3,
                    checkpoint_metadata_filename=(
                        "dynamic-crop-v3-metadata.json"
                    ),
                ),
                selection_policy={
                    **bridge.selection_policy,
                    "dynamic_crop_version": 3,
                },
            )

    def test_candidate_history_rejects_invalid_or_mixed_dynamic_crop_versions(
        self,
    ) -> None:
        retained = _bridge_result().retained_candidates[0]
        for history in (
            (*retained.edit.candidate_history, {"dynamic_crop_version": 4}),
            (
                *retained.edit.candidate_history,
                {"dynamic_crop_version": 2},
                {"dynamic_crop_version": 3},
            ),
        ):
            with self.subTest(history=history[-1]):
                invalid = replace(
                    retained,
                    edit=replace(retained.edit, candidate_history=history),
                )
                with self.assertRaisesRegex(
                    StageEditVisualCandidateError,
                    "invalid Dynamic Crop version|mixes Dynamic Crop versions",
                ):
                    _retained_dynamic_crop_version(invalid)

    @staticmethod
    def _post_render_report(
        verdict: str,
        *,
        input_artifact_fingerprints: dict[str, object] | None = None,
    ) -> SimpleNamespace:
        passed = verdict == "PASS"
        payload = {
            "passed": passed,
            "metrics": _group_integrity_metrics(),
            "issue_count": 0 if passed else 1,
            "fail_issue_count": 0,
            "conditional_issue_count": 0 if passed else 1,
            "issues": (
                []
                if passed
                else [
                    {
                        "code": "timeline_review_failed",
                        "severity": "conditional",
                    }
                ]
            ),
            "hard_failures": [],
            "warnings": [] if passed else [{"code": "fixture_conditional"}],
            "input_artifact_fingerprints": (
                input_artifact_fingerprints or {}
            ),
        }
        return SimpleNamespace(
            verdict=SimpleNamespace(value=verdict),
            passed=passed,
            analysis_basis="mandatory_stage_edit_post_render_test_v1",
            to_dict=lambda: dict(payload),
        )

    @staticmethod
    def _post_render_artifact_shell(
        root: Path,
    ) -> tuple[Path, Path, dict[str, object]]:
        base = "IVE-BANG-BANG-20260824-120000"
        analysis = root / f"{base}-analysis.json"
        video = root / f"{base}-group-crosscut.mp4"
        (root / f"{base}-render-plan.json").write_text("{}", encoding="utf-8")
        (root / f"{base}-rendered-candidate-selection.json").write_text(
            "{}", encoding="utf-8"
        )
        video.write_bytes(b"preserved final mp4")
        manifest: dict[str, object] = {
            "request": {
                "group": "IVE",
                "edit_mode": "group",
                "source_mode": "automatic",
            },
            "artifact_role": "pending_qa",
            "app_result_exposure": {
                "contract": APP_RESULT_EXPOSURE_CONTRACT,
                "video_output_exposed": False,
                "delivery_completion_message_allowed": False,
            },
            "render": {
                "video_path": str(video),
                "artifact_role": "pending_qa",
            },
            "stage_edit_generation": {
                "activation": {
                    "activation": "automatic_registered_production_bundle"
                }
            },
        }
        analysis.write_text(json.dumps(manifest), encoding="utf-8")
        return analysis, video, manifest

    def test_registered_automatic_post_render_pass_is_persisted_as_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            analysis, video, manifest = self._post_render_artifact_shell(root)

            def evaluate(
                artifact_set,
                *,
                output_path,
                declared_artifact_set,
                trusted_registered_stage_edit,
            ):
                self.assertTrue(trusted_registered_stage_edit)
                self.assertEqual(
                    artifact_set.default_report_path,
                    output_path,
                )
                self.assertEqual(analysis, declared_artifact_set.analysis_path)
                self.assertTrue(artifact_set.analysis_path.is_file())
                self.assertTrue(artifact_set.render_plan_path.is_file())
                self.assertTrue(artifact_set.selection_path.is_file())
                self.assertTrue(artifact_set.video_path.is_file())
                report = self._post_render_report(
                    "PASS",
                    input_artifact_fingerprints=(
                        fingerprint_post_render_artifacts(artifact_set)
                    ),
                )
                report_path = output_path
                report_path.write_text(
                    json.dumps(report.to_dict()), encoding="utf-8"
                )
                return report, report_path

            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa",
                side_effect=evaluate,
            ) as evaluator:
                report_path = _enforce_automatic_stage_edit_post_render_qa(
                    registered_stage_edit_bundle=SimpleNamespace(),
                    manifest=manifest,
                    manifest_path=analysis,
                    reporter=_MemoryReporter(),
                )

            evaluator.assert_called_once()
            self.assertIsNotNone(report_path)
            assert report_path is not None
            self.assertTrue(report_path.is_file())
            self.assertTrue(video.is_file())
            persisted = json.loads(analysis.read_text(encoding="utf-8"))
            self.assertEqual("delivery", persisted["artifact_role"])
            self.assertTrue(
                persisted["app_result_exposure"]["video_output_exposed"]
            )
            self.assertTrue(
                persisted["app_result_exposure"][
                    "delivery_completion_message_allowed"
                ]
            )
            provenance = persisted["post_render_qa"]
            self.assertEqual("PASS", provenance["verdict"])
            self.assertTrue(provenance["passed"])
            self.assertEqual(
                AUTOMATIC_STAGE_EDIT_POST_RENDER_QA_CONTRACT,
                provenance["contract"],
            )
            self.assertEqual(provenance, persisted["render"]["post_render_qa"])
            self.assertEqual(
                provenance,
                persisted["stage_edit_generation"]["post_render_qa"],
            )

    def test_registered_automatic_post_render_conditional_preserves_and_blocks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            analysis, video, manifest = self._post_render_artifact_shell(root)

            def evaluate(
                artifact_set,
                *,
                output_path,
                declared_artifact_set,
                trusted_registered_stage_edit,
            ):
                self.assertTrue(trusted_registered_stage_edit)
                self.assertEqual(
                    artifact_set.default_report_path,
                    output_path,
                )
                self.assertEqual(analysis, declared_artifact_set.analysis_path)
                report = self._post_render_report(
                    "CONDITIONAL",
                    input_artifact_fingerprints=(
                        fingerprint_post_render_artifacts(artifact_set)
                    ),
                )
                report_path = output_path
                report_path.write_text(
                    json.dumps(report.to_dict()), encoding="utf-8"
                )
                return report, report_path

            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa",
                side_effect=evaluate,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "verdict CONDITIONAL.*retained only for diagnostics.*"
                    "QA report preserved",
                ):
                    _enforce_automatic_stage_edit_post_render_qa(
                        registered_stage_edit_bundle=SimpleNamespace(),
                        manifest=manifest,
                        manifest_path=analysis,
                        reporter=_MemoryReporter(),
                    )

            report_path = root / (
                "IVE-BANG-BANG-20260824-120000-post-render-qa.json"
            )
            self.assertTrue(video.is_file())
            self.assertTrue(report_path.is_file())
            persisted = json.loads(analysis.read_text(encoding="utf-8"))
            self.assertEqual("rejected", persisted["artifact_role"])
            self.assertFalse(
                persisted["app_result_exposure"]["video_output_exposed"]
            )
            self.assertFalse(
                persisted["app_result_exposure"][
                    "delivery_completion_message_allowed"
                ]
            )
            self.assertEqual("CONDITIONAL", persisted["post_render_qa"]["verdict"])
            self.assertFalse(persisted["post_render_qa"]["passed"])

    def test_non_registered_path_never_invokes_mandatory_post_render_qa(self) -> None:
        with patch(
            "crosscut.pipeline.evaluate_and_write_post_render_qa"
        ) as evaluator:
            result = _enforce_automatic_stage_edit_post_render_qa(
                registered_stage_edit_bundle=None,
                manifest={},
                manifest_path=Path("/does/not/exist-analysis.json"),
                reporter=_MemoryReporter(),
            )

        self.assertIsNone(result)
        evaluator.assert_not_called()

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
                feature_cache_directory=Path("/legacy/source-cache"),
                feature_derived_cache_directory=Path(
                    "/direct/visual-derived-cache"
                ),
                require_persistent_feature_cache=True,
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
            self.assertEqual(
                call.kwargs["feature_cache_directory"],
                Path("/legacy/source-cache"),
            )
            self.assertEqual(
                call.kwargs["feature_derived_cache_directory"],
                Path("/direct/visual-derived-cache"),
            )
            self.assertIs(
                call.kwargs["require_persistent_feature_cache"],
                True,
            )
        payload = batch.to_dict()
        self.assertEqual("stage_edit_generation_only", payload["candidate_pool"])
        self.assertFalse(payload["ordinary_visual_baseline_candidate_pool_eligible"])
        self.assertFalse(payload["silent_baseline_fallback_allowed"])

    def test_required_direct_cache_failure_propagates_before_classification(
        self,
    ) -> None:
        bridge = _bridge_result()
        with patch(
            "crosscut.stage_edit_visual_candidate_adapter."
            "build_prescribed_visual_timeline_candidate_set",
            side_effect=PersistentVisualFeatureCacheInitializationError(
                "required persistent visual feature cache initialization failed"
            ),
        ) as prescribed:
            with self.assertRaises(
                PersistentVisualFeatureCacheInitializationError
            ):
                materialize_stage_edit_visual_candidate(
                    bridge.retained_candidates[0],
                    candidate_index=0,
                    review_rank=0,
                    sources=bridge.retained_candidates[0].edit.sources,
                    audio_offsets=(0.0, 0.0),
                    sync_scores=(0.9, 0.8),
                    scene_cut_times=((), ()),
                    source_timewarps=(None, None),
                    feature_derived_cache_directory=Path(
                        "/direct/visual-derived-cache"
                    ),
                    require_persistent_feature_cache=True,
                )

        prescribed.assert_called_once()

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

    def test_v3_render_contract_uses_truthful_generic_audit_without_v2_alias(self) -> None:
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
        v3_shots = []
        for index, shot in enumerate(candidate.retained_candidate.edit.shots):
            track = EditorialTransformTrack(
                (
                    EditorialTransformKeyframe(0.0, 1.04, 0.50, 0.50, 0.0),
                    EditorialTransformKeyframe(2.0, 1.10, 0.52, 0.49, 0.0),
                ),
                reason="stage_edit_dynamic_crop_head_v3",
            )
            decode = DynamicCropTrackDecodeV3(
                source_id=("source-a", "source-b")[index],
                source_index=shot.segment.source_index,
                eligible=True,
                loss_masked=False,
                track=track,
                predicted_keyframe_count=2,
                applied_keyframe_count=2,
                minimum_safe_zoom_projection_count=0,
                rejection_reasons=(),
                motion_logit=20.0,
                shot_duration_seconds=shot.segment.duration,
            )
            v3_shots.append(
                replace(
                    shot,
                    segment=replace(shot.segment, editorial_transform=track),
                    dynamic_crop_decode_v3=decode,
                )
            )
        v3_edit = replace(
            candidate.retained_candidate.edit,
            shots=tuple(v3_shots),
            candidate_history=(
                *candidate.retained_candidate.edit.candidate_history[:-1],
                {
                    **candidate.retained_candidate.edit.candidate_history[-1],
                    "dynamic_crop_version": 3,
                },
            ),
        )
        v3_retained = replace(candidate.retained_candidate, edit=v3_edit)
        v3_candidate = replace(
            candidate,
            retained_candidate=v3_retained,
            candidate_record=replace(
                candidate.candidate_record,
                retained_candidate=v3_retained,
            ),
        )
        plan = RenderPlan(
            sources=v3_edit.sources,
            segments=tuple(
                replace(shot.segment, editorial_transform=None)
                for shot in v3_edit.shots
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("/tmp/stage-edit-v3-audit.mp4"),
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
        bundle = _GroupRenderCandidateBundle(
            candidate_index=0,
            review_rank=0,
            cadence_profile="stage_edit_v3",
            pre_render_score=10.0,
            visual_timeline=v3_candidate.visual_plan,
            boundary_repairs=None,
            timeline_review=None,
            face_match_plan=None,
            group_transition_plan=GroupTransitionPlan(plan.transitions),
            boundary_affine_plan=None,
            editorial_motion_plan=None,
            render_plan=plan,
        )
        motion_timeline = SimpleNamespace(
            authority=SimpleNamespace(
                authority_content_sha256="d" * 64,
            ),
            timeline_evidence_content_sha256="e" * 64,
            framing_track_applied_count=0,
            motion_event_applied_count=0,
            motion_event_applied_fraction=0.0,
            motion_budget_within_range=True,
            static_or_cut_time_reframe_excluded_count=2,
        )
        with (
            patch(
                "crosscut.pipeline._stage_edit_exact_shot_clock_from_edit",
                return_value={"contract": "test-exact-clock"},
            ),
            patch(
                "crosscut.pipeline."
                "corroborate_dynamic_crop_motion_timeline_v3",
                return_value=motion_timeline,
            ),
        ):
            promoted = _apply_stage_edit_candidate_render_contract(
                bundle,
                v3_candidate,
                motion_authority_v3=DynamicCropMotionAuthorityV3.create(
                    release_freeze_content_sha256="a" * 64,
                    release_freeze_file_sha256="b" * 64,
                    source_commit_oid="c" * 40,
                    candidate_id="production-balanced-test",
                ),
            )
        event = promoted.candidate_lineage[-1]
        self.assertEqual(event["stage"], "dynamic_crop_v3_render_plan_overlay")
        self.assertEqual(event["dynamic_crop_version"], 3)
        self.assertEqual(event["dynamic_crop_suppressed_segment_indices"], [0, 1])
        self.assertNotIn("dynamic_crop_v2_suppressed_segment_indices", event)
        self.assertIn(
            "stage_edit_dynamic_crop_v3_render_contract_applied",
            promoted.diagnostics,
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
            boundary_repairs=SimpleNamespace(),
            timeline_review=None,
            face_match_plan=None,
            group_transition_plan=transitions,
            boundary_affine_plan=None,
            editorial_motion_plan=None,
            render_plan=original_plan,
            candidate_origin="explicit_stage_edit_generation",
            candidate_lineage=({"stage": "initial"},),
            splice_graph_authorization={"proxy_eligible": True},
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
            timeline_review=SimpleNamespace(passed=True),
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
                "crosscut.pipeline._budgeted_repair_visual_timeline_boundaries",
                return_value=targeted,
            ),
            patch("crosscut.pipeline._record_boundary_repair_assessments"),
            patch(
                "crosscut.pipeline._authorize_partial_reedit_splice_lineage",
                return_value=(
                    targeted,
                    {
                        "proxy_eligible": True,
                        "geometry_proof_boundary_indices": [],
                    },
                ),
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
                boundary_descriptor_cache=SimpleNamespace(),
                splice_feasibility_graph=SimpleNamespace(),
                boundary_search_budget=_BoundarySearchBudget(
                    structural_edge_bound=1
                ),
                learned_issue_boundary_indices=(0,),
                candidate_bundle_postprocessor=postprocessor,
            )

        postprocessor.assert_called_once()
        postprocessed_bundle, postprocessed_candidate = (
            postprocessor.call_args.args
        )
        self.assertIs(targeted, postprocessed_bundle.boundary_repairs)
        self.assertEqual(
            {
                "proxy_eligible": True,
                "geometry_proof_boundary_indices": [],
            },
            postprocessed_bundle.splice_graph_authorization,
        )
        self.assertEqual(2, postprocessed_candidate.candidate_index)
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
            montage_policy, montage_profile_path = _current_montage_policy_fixture(root)
            (
                _motion_profile,
                motion_profile_path,
                motion_profile_file_sha256,
            ) = _current_motion_cut_profile_fixture(root, montage_policy)
            request = AnalysisRequest.from_dict(
                {
                    "group": "IVE",
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "review_first_enabled": False,
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
                        stage_edit_exact_shot_clock={
                            "contract": EXACT_STAGE_EDIT_SHOT_CLOCK_CONTRACT,
                            "source_index_domain": "eligible_source_index",
                            "shots": [
                                {
                                    "shot_index": 0,
                                    "source_index": index,
                                    "section_role": "chorus_like",
                                    "section_progress": 0.0,
                                    "phrase_progress": 0.0,
                                }
                            ],
                        },
                    )
                )
            batch = _typed_visual_batch(quality_gated_single_delivery=True)
            hybrid_candidate_set, hybrid_materialized = _visual_fixture()
            materialize_hybrid = Mock(return_value=(hybrid_materialized,))

            def _write_generation(path: Path):
                path.write_text("{}", encoding="utf-8")
                return path

            generation = SimpleNamespace(
                retained_candidates=tuple(object() for _ in range(8)),
                learned_section_director=SimpleNamespace(
                    evidence=(object(),),
                    section_plan=section_plan,
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
                def __init__(
                    self,
                    reviews: tuple[RenderedCandidateReview, ...],
                ) -> None:
                    self.reviews = reviews
                    self.selected = next(
                        (
                            item
                            for item in reviews
                            if item.candidate.candidate_index == 6
                        ),
                        reviews[0],
                    )
                    self.selected_candidate_index = (
                        self.selected.candidate.candidate_index
                    )
                    self.successful_review_count = len(reviews)
                    self.passing_candidate_count = sum(
                        item.passed for item in reviews
                    )
                    self.minimum_required_candidate_count = 1
                    self.minimum_candidate_count_satisfied = True
                    self.learned_reviewer_rerank = {
                        "selected_repair_issue_boundary_indices": [0],
                        "selected_repair_issues": [
                            {
                                "issue_id": "learned-boundary-0",
                                "start_seconds": 3.8,
                                "end_seconds": 4.2,
                                "causes": [
                                    "learned_editorial_discontinuity"
                                ],
                                "severity": "error",
                            }
                        ],
                        "selection_mode": (
                            "learned_quality_within_deterministic_hard_safe_pool"
                        ),
                    }

                def to_dict(self):
                    return {
                        "selected_candidate_index": (
                            self.selected_candidate_index
                        ),
                        "reviewed_candidate_count": len(self.reviews),
                        "passed": True,
                    }

            def _review_single_candidate(
                candidates,
                **_kwargs,
            ):
                bundle = tuple(candidates)[0]
                candidate = RenderedCandidate(
                    candidate_index=bundle.candidate_index,
                    render_plan=bundle.render_plan,
                    pre_render_score=bundle.pre_render_score,
                    payload=bundle,
                )
                review = RenderedCandidateReview(
                    candidate=candidate,
                    review_payload={
                        "passed": True,
                        "weighted_score": 90.0,
                        "ranking_weighted_score": 90.0,
                        "partial_reedit_boundary_issue_indices": [],
                        "visible_boundary_score": {
                            "issue_fraction": 0.0,
                            "p10": 90.0,
                        },
                    },
                )
                return SimpleNamespace(reviews=(review,))

            def _select_reviewed_candidates(reviews, **_kwargs):
                return _Selection(tuple(reviews))

            def _select_stage_edit_cohort(bundles, reviews, **_kwargs):
                bundle_items = tuple(bundles)
                selection = _Selection(tuple(reviews))
                selected_index = selection.selected_candidate_index
                selected_bundle = next(
                    item
                    for item in bundle_items
                    if item.candidate_index == selected_index
                )
                return (
                    ((selected_bundle, selection.selected),),
                    {
                        "contract": (
                            "stage_edit_best_available_decoded_delivery_v1"
                        ),
                        "comparison_candidate_count": len(bundle_items),
                        "required_delivery_count": 1,
                        "selected_candidate_index": selected_index,
                        "comparison": selection.to_dict(),
                    },
                )

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

            def _montage_review(*, bundle, **_kwargs):
                self.assertIsNotNone(bundle.stage_edit_exact_shot_clock)
                return (
                    {
                        "passed": True,
                        "failures": [],
                        "highlight_aggregate": {
                            "soft_targets": {
                                "aggregate_alignment_score": 0.80,
                            }
                        },
                    },
                    {
                        "stage_edit_exact_shot_clock": (
                            bundle.stage_edit_exact_shot_clock
                        ),
                        "dynamic_crop_version": (
                            bundle.stage_edit_dynamic_crop_version
                        ),
                        "dynamic_crop_v3_decodes": None,
                        "dynamic_crop_motion_v3": None,
                    },
                )

            montage_review = Mock(side_effect=_montage_review)

            def _pass_post_render_qa(artifact_set, **_kwargs):
                report = self._post_render_report(
                    "PASS",
                    input_artifact_fingerprints=(
                        fingerprint_post_render_artifacts(artifact_set)
                    ),
                )
                report_path = artifact_set.default_report_path
                report_path.write_text(
                    json.dumps(report.to_dict()), encoding="utf-8"
                )
                return report, report_path

            def _write_final_plan(_plan, output_path: Path) -> None:
                output_path.write_text("{}", encoding="utf-8")

            def _render_final(plan: RenderPlan, **_kwargs) -> str:
                plan.output.write_bytes(b"fixture final mp4")
                return "h264_videotoolbox"

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
                    "crosscut.pipeline.detect_scene_changes_in_video_cached",
                    return_value=(
                        (),
                        {
                            "analysis_reused": False,
                            "detection_policy": "synthetic_test",
                        },
                    ),
                ),
                patch(
                    "crosscut.pipeline.estimate_shared_performance_range",
                    side_effect=ValueError("synthetic range unavailable"),
                ),
                patch("crosscut.pipeline.build_section_plan", return_value=section_plan),
                patch.multiple(
                    "crosscut.pipeline",
                    DEFAULT_IVE_PROFESSIONAL_MONTAGE_PROFILE_PATH=montage_profile_path,
                    DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_PATH=(
                        motion_profile_path
                    ),
                    DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_FILE_SHA256=(
                        motion_profile_file_sha256
                    ),
                    build_group_draft=lambda *_args, **_kwargs: group_draft,
                    analyze_source_derivative_diversity=(
                        lambda *_args, **_kwargs: derivative_metadata
                    ),
                    materialize_visual_timeline_candidates=(
                        materialize_hybrid
                    ),
                    _review_stage_edit_bundle_montage=montage_review,
                    evaluate_and_write_post_render_qa=_pass_post_render_qa,
                    select_rendered_candidate_reviews=(
                        _select_reviewed_candidates
                    ),
                    select_stage_edit_delivery_cohort=(
                        _select_stage_edit_cohort
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
                    "crosscut.pipeline.build_visual_timeline_candidate_set",
                    return_value=hybrid_candidate_set,
                ) as build_baseline,
                patch(
                    "crosscut.pipeline._derive_complete_group_candidate",
                    side_effect=lambda item, **_kwargs: ordinary_bundles[
                        item.candidate_index
                    ],
                ),
                patch(
                    "crosscut.pipeline._apply_stage_edit_candidate_render_contract",
                    side_effect=lambda _bundle, item, **_kwargs: promoted_bundles[
                        item.candidate_index
                    ],
                ) as apply_contract,
                patch(
                    "crosscut.pipeline._review_complete_group_candidates",
                    side_effect=_review_single_candidate,
                ) as review_candidates,
                patch(
                    "crosscut.pipeline._attempt_stage_edit_windowed_beam_reedit",
                    side_effect=_stage_beam_repair,
                ) as stage_beam_repair,
                patch(
                    "crosscut.pipeline.write_plan",
                    side_effect=_write_final_plan,
                ),
                patch(
                    "crosscut.pipeline.render",
                    side_effect=_render_final,
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
            build_baseline.assert_called_once()
            self.assertEqual(
                8,
                build_baseline.call_args.kwargs["maximum_results"],
            )
            materialize_hybrid.assert_called_once_with(
                hybrid_candidate_set,
                maximum_candidates=1,
                passed_only=True,
            )
            self.assertEqual(8, apply_contract.call_count)
            self.assertEqual(8, review_candidates.call_count)
            stage_beam_repair.assert_called_once()
            montage_review.assert_called_once()
            reviewed = tuple(
                call.args[0][0]
                for call in review_candidates.call_args_list
            )
            self.assertEqual(8, len(reviewed))
            self.assertTrue(
                all(
                    call.kwargs["learned_reviewer_session"]
                    is reviewer_session
                    for call in review_candidates.call_args_list
                )
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
