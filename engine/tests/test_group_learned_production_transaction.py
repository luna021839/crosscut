"""Exercise learned proposals through analyze's real promotion and QA handoff.

Media/model measurements are scripted. Bundle construction, camera overlay,
ownership, persistence and the final QA activation consumer are production code.
These tests do not claim inference, decoded-media or editorial quality evidence.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import json
from time import monotonic
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.audio import AlignmentResult, SourceAlignment
from crosscut.cut_local_sync import analyze_cut_local_sync, resolve_cut_local_sync_activation
from crosscut.draft import DraftSegment, GroupDraftTimeline
from crosscut.pipeline import analyze, _rendered_montage_config_from_policy
from crosscut.optional_native_worker import optional_native_scope, _encode_message
from crosscut.render import EditorialTransformKeyframe, EditorialTransformTrack, RenderSegment
from crosscut.request import AnalysisRequest
from crosscut.review_first_selection_contract import review_first_release_binding_valid
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.stage_edit import StageShot
from crosscut.stage_edit_visual_candidate_adapter import StageEditMaterializedVisualCandidate
from crosscut.visual_boundary_repair import VisualBoundaryRepairPlan
from crosscut.visual_score import VisualTimelinePlan
from engine.tests.test_pipeline import (
    _MemoryReporter, _analysis_media, _current_montage_policy_fixture,
    _source_derivative_diversity_fixture, _visual_choice,
)


class _ReachedFinalQA(BaseException):
    pass


class GroupLearnedProductionTransactionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.sources = tuple(self.root / f"source-{index}.mp4" for index in range(9))
        for source in self.sources:
            source.touch()
        self.request = AnalysisRequest.from_dict(dict(
            song="I AM", edit_mode="group", aspect_ratio="16:9", source_mode="automatic",
            review_first_enabled=True, sources=[str(path) for path in self.sources],
            output_directory=str(self.root / "out"),
        ))
        self.section = SectionPlan((SectionDirective(
            0, SectionRole.VERSE_LIKE, 0., 8., .5, .9, 1, 4, 13.5, .34, 0., False, False,
        ),))
        self.baseline = VisualTimelinePlan(tuple(
            _visual_choice(index, 0, index * 4., (index + 1) * 4.) for index in range(2)
        ), 2.)
        self.learned = replace(self.baseline, choices=tuple(
            replace(choice, source_index=index + 1, shot_id=f"learned-{index}")
            for index, choice in enumerate(self.baseline.choices)
        ), geometry_basis="prescribed_render_geometry_v1",
            source_clock_basis="prescribed_render_segment_clock_v1")
        self.track = EditorialTransformTrack((
            EditorialTransformKeyframe(0., 1.10, .49, .5, 0.),
            EditorialTransformKeyframe(3., 1.15, .51, .5, 0.),
        ), reason="stage_edit_dynamic_crop_head_v2")
        shots = tuple(StageShot(
            shot_index=index, section_index=0,
            segment=RenderSegment(index + 1, index * 4., 4., index * 4.,
                                  editorial_transform=self.track),
            decision_id=f"learned-{index}", section_role="verse_like",
            section_progress=index / 2., phrase_progress=index / 2.,
        ) for index in range(2))
        retained = SimpleNamespace(rank=0, score=1., edit=SimpleNamespace(
            sources=self.sources, shots=shots, candidate_history=({"dynamic_crop_version": 2},),
        ))
        review = SimpleNamespace(passed=True, to_dict=lambda: {"passed": True})
        record = SimpleNamespace(candidate_index=0, review_rank=0, cadence_profile="learned_fixture",
                                 review=review, to_dict=lambda: {"candidate_index": 0})
        self.materialized = StageEditMaterializedVisualCandidate(
            candidate_record=record, base_materialized=SimpleNamespace(segments=()),
            visual_plan=self.learned, retained_candidate=retained,
            quality_gated_single_delivery=True,
        )
        policy, _ = _current_montage_policy_fixture(self.root)
        self.registry = SimpleNamespace(to_dict=lambda: {
            "activation": "automatic_registered_production_bundle", "fixture": True,
        })
        self.generation = SimpleNamespace(retained_candidates=(retained,),
            learned_section_director=SimpleNamespace(section_plan=self.section),
            to_dict=lambda: {"retained_candidate_count": 1})
        self.context = context = dict(bundle=self.registry, motion=None, policy=policy, motion_profile=None,
            profile_payload=None, montage_config=_rendered_montage_config_from_policy(
                policy, require_complete_role_duration_policy=True),
            snapshot=None, causal_audit=None, generation=self.generation,
            audit_path=self.root / "generation-audit.json")
        alignment = AlignmentResult(reference=str(self.sources[0]), sample_rate=4000,
            alignments=tuple(SourceAlignment(source=str(source), is_reference=index == 0,
                offset_samples=0, offset_seconds=0., score=1., overlap_samples=48000,
                overlap_seconds=12., score_margin=1., overlap_ratio=1.)
                for index, source in enumerate(self.sources)))
        draft = GroupDraftTimeline(source_count=9, common_master_start=0., common_master_end=8.,
            beat_period_seconds=.5, audio_offsets=(0.,) * 9, sync_scores=(1.,) * 9,
            segments=tuple(DraftSegment(segment_index=index, source_index=0,
                canonical_master_start=index * 4., canonical_master_end=(index + 1) * 4.,
                output_start_seconds=index * 4., source_local_start=index * 4.,
                source_local_end=(index + 1) * 4., audio_offset_seconds=0., sync_score=1.,
                estimated_beat_count=8., start_boundary_reasons=("range_edge",),
                end_boundary_reasons=("phrase",)) for index in range(2)))
        rhythm = SimpleNamespace(bpm=120., beats=(object(),), phrase_candidates=(),
                                 beat_times_seconds=(0., 4., 8.), to_dict=lambda: {})
        self.stack.enter_context(patch.dict("os.environ", {}, clear=True))
        for name, value in {
            "_load_stage_edit_montage_policy": policy,
            "_align_sequentially": (alignment, None), "analyze_rhythm": rhythm,
            "build_section_plan": self.section, "build_group_draft": draft,
            "analyze_source_derivative_diversity": _source_derivative_diversity_fixture(9),
            "detect_scene_changes_in_video_cached": ((), {"detection_policy": "synthetic_test"}),
            "build_visual_timeline": self.baseline,
            "repair_visual_timeline_boundaries": VisualBoundaryRepairPlan(
                self.baseline, (self.baseline,), (), 1, ()),
            "analyze_source_look_match": None,
            "_review_first_registered_group_generation": context,
            "materialize_stage_edit_visual_candidate": self.materialized,
            "build_boundary_affine_plan": None, "build_boundary_subject_alignment_plan": None,
            "build_editorial_motion_plan": None,
        }.items():
            self.stack.enter_context(patch(f"crosscut.pipeline.{name}", return_value=value))
        self.stack.enter_context(patch("crosscut.pipeline.probe_media",
                                      side_effect=lambda path: _analysis_media(path, duration=12.)))
        self.stack.enter_context(patch("crosscut.pipeline.build_face_match_plan",
                                      side_effect=ValueError("fixture has no face measurements")))
        self.stack.enter_context(patch("crosscut.pipeline.build_group_transition_plan", return_value=None))
        def write_fixture_artifacts(plan, output=None, **kwargs):
            # This is deliberately not video. Only the artifact identity path
            # is real; decoded quality/model measurements are explicitly mocked.
            output = Path(output or plan.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"SYNTHETIC-TEST-ARTIFACT-NOT-ENCODED-MEDIA")
            output.with_suffix(".plan.json").write_text(json.dumps(plan.to_json(), sort_keys=True))
            return "fixture_encoder"
        def proxy_fixture(plan, output, **kwargs):
            write_fixture_artifacts(plan, output)
            return plan, SimpleNamespace(payload={
                "passed": False, "weighted_score": 45., "boundary_issue_indices": [0], "proxy": {},
            })
        self.proxy = self.stack.enter_context(patch("crosscut.pipeline.render_and_review_proxy",
            side_effect=proxy_fixture))
        # Keep the montage serialization/identity producer real. Only expensive
        # pixel/montage measurements are scripted, as an editorial quality miss.
        import crosscut.pipeline as pipeline
        def montage(**kwargs):
            analysis = pipeline._stage_edit_montage_analysis_payload(**{
                key: kwargs[key] for key in ("bundle", "proxy_review_payload", "section_plan",
                                            "source_derivative_diversity", "source_index_mapping")})
            return {"passed": False, "weighted_score": 45.}, analysis
        self.stack.enter_context(patch("crosscut.pipeline._review_stage_edit_bundle_montage", side_effect=montage))
        self.stack.enter_context(patch("crosscut.pipeline._apply_bounded_human_target_proxy_rank"))
        self.render = self.stack.enter_context(patch("crosscut.pipeline.render", side_effect=write_fixture_artifacts))
        self.sync_payload = dict(passed=True, retained_cross_source_boundary_count=1,
            measured_boundary_count=1, unknown_boundary_count=0, residual_p95_frames=0.)
        self.sync = self.stack.enter_context(patch("crosscut.pipeline.analyze_cut_local_sync",
                                                 return_value=self.sync_payload))
        self.manifest = None
        from crosscut.first_result_controller import FirstResultController
        self.promoted_bundles = []
        promote = FirstResultController.promote
        def capture_promoted(controller, value, **kwargs):
            result = promote(controller, value, **kwargs)
            self.promoted_bundles.append(controller.incumbent)
            return result
        self.stack.enter_context(patch.object(FirstResultController, 'promote', new=capture_promoted))
        def final_qa(**kwargs):
            self.manifest = kwargs["manifest"]
            self.registry_at_qa = kwargs["registered_stage_edit_bundle"]
            raise _ReachedFinalQA()
        self.stack.enter_context(patch("crosscut.pipeline._enforce_automatic_stage_edit_post_render_qa",
                                      side_effect=final_qa))

    def reach_final_qa(self):
        with self.assertRaises(_ReachedFinalQA):
            analyze(self.request, _MemoryReporter(), review_rendered_proxies=True)
        self.render.assert_called_once()
        self.assertIsNotNone(self.manifest)
        return self.render.call_args.args[0]

    def test_accepted_learned_camera_reaches_renderer_and_qa_with_consistent_identity(self):
        plan = self.reach_final_qa()
        self.assertEqual(tuple(segment.source_index for segment in plan.segments), (1, 2))
        self.assertTrue(all(segment.editorial_transform is not None for segment in plan.segments))
        self.assertEqual(plan.to_json(), self.proxy.call_args.args[0].to_json())
        self.assertEqual(plan.output_frame_count, round(8. * plan.fps))
        self.sync.assert_called_once()
        selection = self.manifest["rendered_candidate_selection"]
        self.assertTrue(review_first_release_binding_valid(selection, self.manifest["render"]))
        activation = resolve_cut_local_sync_activation(self.manifest, selection,
            trusted_registered_stage_edit=True)
        self.assertTrue(activation["valid"], activation["errors"])
        self.assertTrue(self.manifest["review_first_visual_selection"]["dynamic_crop_applied_to_render"])
        # A decoded editorial miss stays a miss, even after valid model/sync
        # provenance is accepted by the independent contract consumer.
        self.assertFalse(selection["passed"])
        self.assertFalse(selection["stage_edit_render_promotion"]["decoded_selected_proxy_passed"])
        owned = self.promoted_bundles[-1]
        self.assertEqual(owned.render_plan.to_json(),plan.to_json())
        self.assertEqual(owned.editorial_motion_plan.to_dict(),self.manifest['render']['editorial_motion'])
        self.assertEqual(tuple(item.track for item in owned.editorial_motion_plan.segments),
            tuple(segment.editorial_transform for segment in plan.segments))
        self.assertEqual(owned.to_dict()['dynamic_crop_application'],
            self.manifest['review_first_visual_selection']['dynamic_crop_application'])
        self.assertFalse(owned.editorial_motion_plan.to_dict()['policy']['quality_approved'])

    def test_missing_complete_motion_after_learned_overlay_retains_baseline_before_sync(self):
        import crosscut.pipeline as pipeline
        original = pipeline._apply_stage_edit_candidate_render_contract
        def incomplete(*args,**kwargs):
            return replace(original(*args,**kwargs),editorial_motion_plan=None)
        with patch('crosscut.pipeline._apply_stage_edit_candidate_render_contract',side_effect=incomplete) as apply:
            plan = self.reach_final_qa()
        apply.assert_called_once()
        self.sync.assert_not_called()
        self.assertEqual(tuple(segment.source_index for segment in plan.segments),(0,0))
        rejected = self.manifest['review_first_visual_selection']['stage_edit_rejections']
        self.assertTrue(any('complete motion source/camera' in item.get('error','') for item in rejected))

    def test_stale_learned_camera_inventory_is_not_silently_refreshed_by_validator(self):
        import crosscut.pipeline as pipeline
        original = pipeline._apply_stage_edit_candidate_render_contract
        def stale(*args,**kwargs):
            bundle = original(*args,**kwargs)
            motion = bundle.editorial_motion_plan
            return replace(bundle,editorial_motion_plan=replace(motion,segments=(
                replace(motion.segments[0],track=None),*motion.segments[1:])))
        with patch('crosscut.pipeline._apply_stage_edit_candidate_render_contract',side_effect=stale) as apply:
            plan = self.reach_final_qa()
        apply.assert_called_once()
        self.sync.assert_not_called()
        self.assertEqual(tuple(segment.source_index for segment in plan.segments),(0,0))
        self.assertIsNone(self.registry_at_qa)

    def test_overlay_does_not_drop_prior_boundary_receipts_to_build_numeric_inventory(self):
        import crosscut.pipeline as pipeline
        original = pipeline._review_first_group_bundle
        def with_prior_receipt(**kwargs):
            bundle = original(**kwargs)
            return replace(bundle,editorial_motion_plan=replace(bundle.editorial_motion_plan,
                required_boundary_failures=((0,('old-measured-failure',),None),)))
        with patch('crosscut.pipeline._review_first_group_bundle',side_effect=with_prior_receipt):
            plan = self.reach_final_qa()
        self.sync.assert_not_called()
        self.assertEqual(tuple(segment.source_index for segment in plan.segments),(0,0))
        rejected = self.manifest['review_first_visual_selection']['stage_edit_rejections']
        self.assertTrue(any('replay of existing motion receipts' in item.get('error','') for item in rejected))

    def test_learned_singleton_identity_cannot_be_forged_by_copying_the_pool_token(self):
        self.reach_final_qa()
        for field, value in (
            ("activation", "explicit_stage_edit_generation_request_only"),
            ("selected_candidate_index", 1),
            ("selected_candidate_index", False),
            ("selected_candidate_origin", "ordinary_visual_baseline"),
            ("selected_candidate_lineage", []),
            ("required_complete_candidate_count", 8),
            ("required_complete_candidate_count", True),
            ("decoded_selected_proxy_passed", True),
            ("delivery_completion_message_allowed", True),
        ):
            with self.subTest(field=field):
                selected = deepcopy(self.manifest["rendered_candidate_selection"])
                selected["stage_edit_render_promotion"][field] = value
                activation = resolve_cut_local_sync_activation(self.manifest, selected,
                    trusted_registered_stage_edit=True)
                self.assertFalse(activation["valid"], activation)

    def test_v3_camera_motion_and_evidence_survive_the_full_successful_path(self):
        from crosscut.stage_edit_dynamic_crop_motion_authority_v3 import DynamicCropMotionAuthorityV3
        from crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import DynamicCropTrackDecodeV3
        shots = []
        for index in range(10):
            source_index = index % len(self.sources)
            points = (EditorialTransformKeyframe(0., 1.02),)
            if index == 0:
                points += (EditorialTransformKeyframe(.6, 1.05),)
            track = EditorialTransformTrack(points, reason="stage_edit_dynamic_crop_head_v3")
            segment = RenderSegment(source_index, index * .8, .8, index * .8,
                                    editorial_transform=track)
            decode = DynamicCropTrackDecodeV3(
                source_id=f"camera-{source_index}", source_index=source_index, eligible=True,
                loss_masked=False, track=track, predicted_keyframe_count=len(points),
                applied_keyframe_count=len(points), minimum_safe_zoom_projection_count=0,
                rejection_reasons=(), motion_logit=10., shot_duration_seconds=.8,
            )
            shots.append(StageShot(index, 0, segment, decision_id=f"v3-{index}",
                section_role="verse_like", section_progress=index / 10.,
                phrase_progress=index / 10., dynamic_crop_decode_v3=decode))
        retained = SimpleNamespace(rank=0, score=1., edit=SimpleNamespace(
            sources=self.sources, shots=tuple(shots), candidate_history=({"dynamic_crop_version": 3},),
        ))
        visual = replace(self.learned, choices=tuple(
            _visual_choice(index, shot.segment.source_index, index * .8, (index + 1) * .8)
            for index, shot in enumerate(shots)))
        materialized = replace(self.materialized, retained_candidate=retained, visual_plan=visual)
        self.context["motion"] = DynamicCropMotionAuthorityV3.create(
            release_freeze_content_sha256="a" * 64, release_freeze_file_sha256="b" * 64,
            source_commit_oid="c" * 40, candidate_id="synthetic-production-path-test")
        self.generation.retained_candidates = (retained,)
        self.sync.return_value = {**self.sync_payload, "retained_cross_source_boundary_count": 9,
                                  "measured_boundary_count": 9}
        with patch("crosscut.pipeline.materialize_stage_edit_visual_candidate", return_value=materialized):
            plan = self.reach_final_qa()
        self.assertEqual(len(plan.segments), 10)
        self.assertGreater(len(plan.segments[0].editorial_transform.keyframes), 1)
        selection = self.manifest["rendered_candidate_selection"]
        complete = selection["complete_candidates"][0]
        self.assertEqual(complete["stage_edit_dynamic_crop_version"], 3)
        self.assertEqual(complete["dynamic_crop_motion_v3"]["motion_event_applied_count"], 1)
        activation = resolve_cut_local_sync_activation(self.manifest, selection,
            trusted_registered_stage_edit=True)
        self.assertTrue(activation["valid"], activation["errors"])

    def test_failed_learned_sync_rejects_proposal_and_renders_retained_baseline(self):
        self.sync.return_value = {**self.sync_payload, "passed": False,
                                  "unknown_boundary_count": 1}
        plan = self.reach_final_qa()
        self.assertEqual(tuple(segment.source_index for segment in plan.segments), (0, 0))
        self.assertIsNone(self.manifest["stage_edit_generation"])
        self.assertIsNone(self.registry_at_qa)
        self.assertTrue(self.manifest["review_first_visual_selection"]["first_result_controller"]["baseline_retained"])
        self.assertTrue(any(row["status"] == "rejected" for row in
                            self.manifest["review_first_visual_selection"]["first_result_controller"]["attempts"]))

    def test_failed_sync_native_call_preserves_baseline_and_does_not_retry_after_promotion(self):
        self.sync.side_effect = ValueError("scripted sync decoder failure")
        plan = self.reach_final_qa()
        self.assertEqual(tuple(segment.source_index for segment in plan.segments), (0, 0))
        self.sync.assert_called_once()
        self.assertIsNone(self.registry_at_qa)

    def test_learned_sync_cutoff_retains_baseline_and_records_the_cutoff(self):
        from crosscut.review_first_runtime import ReviewFirstSearchCutoff
        self.sync.side_effect = ReviewFirstSearchCutoff("scripted native sync deadline")
        plan = self.reach_final_qa()
        self.assertEqual(tuple(segment.source_index for segment in plan.segments), (0, 0))
        self.sync.assert_called_once()
        self.assertIsNone(self.registry_at_qa)
        attempts = self.manifest["review_first_visual_selection"]["first_result_controller"]["attempts"]
        self.assertTrue(any(row["stage"].startswith("stage_edit_candidate_rank_")
                            and row["status"] == "search_cutoff" for row in attempts), attempts)

    def test_user_cancellation_propagates_without_exposing_a_fallback(self):
        self.sync.side_effect = KeyboardInterrupt("scripted user cancel")
        with self.assertRaises(KeyboardInterrupt):
            analyze(self.request, _MemoryReporter(), review_rendered_proxies=True)
        self.render.assert_not_called()
        self.assertIsNone(self.manifest)

    def test_native_sync_measurement_enters_killable_optional_process_boundary(self):
        plan = self.reach_final_qa()
        deadline = monotonic() + 20.
        with optional_native_scope(deadline), patch(
            "crosscut.optional_native_worker.run_isolated_python_call", return_value={"fixture": True}
        ) as worker:
            result = analyze_cut_local_sync(plan, source_timewarps=(None,) * 9, timewarp_payloads=())
        worker.assert_called_once()
        self.assertEqual(worker.call_args.args[:2], ("crosscut.cut_local_sync", "analyze_cut_local_sync"))
        self.assertEqual(worker.call_args.kwargs["deadline"], deadline)
        self.assertTrue(_encode_message(worker.call_args.args[2]))
        self.assertEqual(result, {"fixture": True})


if __name__ == "__main__":
    unittest.main()
