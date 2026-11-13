from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crosscut import stage_edit_dynamic_crop_motion_authority_v3 as motion
from crosscut.render import (
    AspectRatio,
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    TransitionKind,
)
from crosscut.stage_edit_dynamic_crop_motion_authority_v3 import (
    DynamicCropMotionAuthorityV3,
    DynamicCropMotionAuthorityV3Error,
    DynamicCropMotionSegmentEvidenceV3,
    DynamicCropMotionTimelineEvidenceV3,
    _canonical_sha256,
    apply_dynamic_crop_motion_authority_to_render_plan_v3,
    corroborate_dynamic_crop_motion_timeline_v3,
    load_dynamic_crop_motion_authority_from_registered_identity_v3,
    load_dynamic_crop_motion_authority_v3,
)
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
    DynamicCropHeadPredictionV3,
    DynamicCropTrackDecodeV3,
)
from crosscut.stage_edit_production_registry import (
    RegisteredStageEditMotionAuthorityIdentityV3,
)


class DynamicCropMotionAuthorityV3Tests(unittest.TestCase):
    @staticmethod
    def _authority() -> DynamicCropMotionAuthorityV3:
        return DynamicCropMotionAuthorityV3.create(
            release_freeze_content_sha256="a" * 64,
            release_freeze_file_sha256="b" * 64,
            source_commit_oid="c" * 40,
            candidate_id="production-balanced-test",
        )

    @staticmethod
    def _track(
        *keyframes: tuple[float, float, float, float, float]
    ) -> EditorialTransformTrack:
        return EditorialTransformTrack(
            tuple(EditorialTransformKeyframe(*item) for item in keyframes),
            reason="stage_edit_dynamic_crop_head_v3",
        )

    @staticmethod
    def _decode(
        track: EditorialTransformTrack,
        *,
        source_index: int = 0,
        source_id: str = "source-0",
        motion_logit: float = 20.0,
    ) -> DynamicCropTrackDecodeV3:
        return DynamicCropTrackDecodeV3(
            source_id=source_id,
            source_index=source_index,
            eligible=True,
            loss_masked=False,
            track=track,
            predicted_keyframe_count=len(track.keyframes),
            applied_keyframe_count=len(track.keyframes),
            minimum_safe_zoom_projection_count=0,
            rejection_reasons=(),
            motion_logit=motion_logit,
            shot_duration_seconds=1.0,
        )

    @staticmethod
    def _plan(
        tracks: tuple[EditorialTransformTrack, ...],
        *,
        source_indices: tuple[int, ...] | None = None,
        fps: float = 30.0,
    ) -> RenderPlan:
        indices = source_indices or (0,) * len(tracks)
        starts = tuple(float(index) for index in range(len(tracks)))
        rendered_tracks = tuple(
            EditorialTransformTrack(
                track.keyframes,
                reason=f"{track.reason}_stage_edit_preserved",
            )
            for track in tracks
        )
        return RenderPlan(
            sources=(Path("source-0.mp4"), Path("source-1.mp4")),
            segments=tuple(
                RenderSegment(
                    source_index=source_index,
                    source_start=0.0,
                    duration=1.0,
                    master_start=master_start,
                    editorial_transform=track,
                )
                for source_index, master_start, track in zip(
                    indices, starts, rendered_tracks, strict=True
                )
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=fps,
        )

    def _evidence(
        self,
        *,
        render_plan: RenderPlan,
        decodes: tuple[DynamicCropTrackDecodeV3, ...],
        authority: DynamicCropMotionAuthorityV3 | None = None,
        minimum_motion_applied_fraction: float = 0.0,
        maximum_motion_applied_fraction: float = 1.0,
    ) -> DynamicCropMotionTimelineEvidenceV3:
        selected_authority = authority or self._authority()
        final_plan = apply_dynamic_crop_motion_authority_to_render_plan_v3(
            render_plan=render_plan,
            decodes=decodes,
            authority=selected_authority,
        )
        return corroborate_dynamic_crop_motion_timeline_v3(
            render_plan=final_plan,
            decodes=decodes,
            authority=selected_authority,
            minimum_motion_applied_fraction=minimum_motion_applied_fraction,
            maximum_motion_applied_fraction=maximum_motion_applied_fraction,
        )

    def test_one_keyframe_is_framing_not_motion_even_with_high_logit(self) -> None:
        track = self._track((0.0, 1.02, 0.5, 0.5, 0.0))
        evidence = self._evidence(
            render_plan=self._plan((track,)),
            decodes=(self._decode(track),),
            authority=self._authority(),
        )

        segment = evidence.segment_evidence[0]
        self.assertTrue(segment.framing_track_applied)
        self.assertTrue(segment.motion_logit_passed)
        self.assertFalse(segment.physical_motion_corroborated)
        self.assertFalse(segment.motion_event_applied)
        self.assertEqual(
            "static_or_cut_time_reframe_excluded",
            segment.motion_budget_classification,
        )
        self.assertEqual(evidence.motion_event_applied_count, 0)
        self.assertEqual(evidence.static_or_cut_time_reframe_excluded_count, 1)

    def test_zero_init_logit_and_prior_framing_cannot_qualify_as_motion(self) -> None:
        prediction = DynamicCropHeadPredictionV3(
            source_id="source-0",
            source_index=0,
            values=((1.02, 0.5, 0.5, 0.0),) * 8,
            count_logits=(0.0,) * 8,
            time_interval_logits=(0.0,) * 35,
            motion_logit=0.0,
        )
        decoded = prediction.decode(1.0)
        assert decoded.track is not None
        evidence = self._evidence(
            render_plan=self._plan((decoded.track,)),
            decodes=(decoded,),
            authority=self._authority(),
        )

        segment = evidence.segment_evidence[0]
        self.assertTrue(segment.framing_track_applied)
        self.assertTrue(segment.motion_logit_passed)
        self.assertFalse(segment.physical_motion_corroborated)
        self.assertFalse(segment.motion_event_applied)

    def test_within_segment_renderer_range_and_logit_apply_motion(self) -> None:
        track = self._track(
            (0.0, 1.02, 0.5, 0.5, 0.0),
            (0.8, 1.05, 0.5, 0.5, 0.0),
        )
        evidence = self._evidence(
            render_plan=self._plan((track,)),
            decodes=(self._decode(track),),
            authority=self._authority(),
            minimum_motion_applied_fraction=0.0,
            maximum_motion_applied_fraction=1.0,
        )

        segment = evidence.segment_evidence[0]
        self.assertGreaterEqual(segment.zoom_range, 0.008)
        self.assertTrue(segment.zoom_range_corroborated)
        self.assertTrue(segment.physical_motion_corroborated)
        self.assertTrue(segment.motion_event_applied)
        self.assertEqual("within_segment_motion_event", segment.motion_budget_classification)
        self.assertEqual(
            segment.overlay_disposition,
            "preserved_same_duration_renamed",
        )
        self.assertEqual(
            tuple(
                (clock.component, clock.frame_index)
                for clock in segment.renderer_visible_first_motion_clocks
            ),
            (("zoom", 1),),
        )
        self.assertEqual(evidence.motion_event_applied_count, 1)

    def test_targeted_repair_duration_scale_is_exactly_bound(self) -> None:
        original = self._track(
            (0.0, 1.02, 0.5, 0.5, 0.0),
            (0.8, 1.05, 0.5, 0.5, 0.0),
        )
        final = EditorialTransformTrack(
            tuple(
                EditorialTransformKeyframe(
                    item.time_seconds * 1.25,
                    item.zoom,
                    item.center_x,
                    item.center_y,
                    item.roll_degrees,
                )
                for item in original.keyframes
            ),
            reason=(
                f"{original.reason}_stage_edit_targeted_repair_reparameterized"
            ),
        )
        plan = RenderPlan(
            sources=(Path("source-0.mp4"),),
            segments=(RenderSegment(0, 0.0, 1.25, 0.0, editorial_transform=final),),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
        )
        evidence = self._evidence(
            render_plan=plan,
            decodes=(self._decode(original),),
            authority=self._authority(),
            minimum_motion_applied_fraction=0.0,
            maximum_motion_applied_fraction=1.0,
        )

        segment = evidence.segment_evidence[0]
        self.assertEqual(
            segment.overlay_disposition,
            "targeted_repair_duration_reparameterized",
        )
        self.assertEqual(segment.time_reparameterization_scale, 1.25)
        self.assertEqual(segment.decoded_shot_duration_seconds, 1.0)
        self.assertEqual(segment.final_segment_duration_seconds, 1.25)
        self.assertNotEqual(
            segment.original_decode_track_sha256,
            segment.rendered_track_sha256,
        )

    def test_verified_face_dissolve_suppression_is_explicit_not_track_loss(self) -> None:
        left = self._track((0.0, 1.02, 0.5, 0.5, 0.0))
        right = self._track((0.0, 1.03, 0.5, 0.5, 0.0))
        plan = RenderPlan(
            sources=(Path("source-0.mp4"), Path("source-1.mp4")),
            segments=(
                RenderSegment(0, 0.0, 1.0, 0.0),
                RenderSegment(1, 0.0, 1.0, 1.0),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
            transitions=(
                RenderTransition(
                    kind=TransitionKind.FACE_DISSOLVE,
                    frame_count=3,
                    reason="verified_face_alignment",
                    face_mismatch=0.05,
                    tier="A",
                ),
            ),
        )
        evidence = self._evidence(
            render_plan=plan,
            decodes=(
                self._decode(left),
                self._decode(right, source_index=1, source_id="source-1"),
            ),
            authority=self._authority(),
        )

        self.assertTrue(
            all(item.face_dissolve_suppressed for item in evidence.segment_evidence)
        )
        self.assertTrue(
            all(
                item.overlay_disposition
                == "verified_face_dissolve_suppressed"
                for item in evidence.segment_evidence
            )
        )
        self.assertEqual(evidence.framing_track_applied_count, 0)
        self.assertEqual(evidence.motion_event_applied_count, 0)

    def test_verified_geometry_override_preserves_tracking_without_model_attribution(
        self,
    ) -> None:
        model_track = self._track(
            (0.0, 1.02, 0.5, 0.5, 0.0),
            (0.8, 1.06, 0.5, 0.5, 0.0),
        )
        tracking_track = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.08, 0.49, 0.5, 0.0),
                EditorialTransformKeyframe(0.8, 1.00, 0.5, 0.5, 0.0),
            ),
            reason="boundary_local_exact_subject_alignment_b0",
        )
        plan = RenderPlan(
            sources=(Path("source-0.mp4"),),
            segments=(
                RenderSegment(
                    0,
                    0.0,
                    1.0,
                    0.0,
                    editorial_transform=tracking_track,
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
        )
        decode = self._decode(model_track)
        authority = self._authority()

        final = apply_dynamic_crop_motion_authority_to_render_plan_v3(
            render_plan=plan,
            decodes=(decode,),
            authority=authority,
            deterministic_geometry_override_segment_indices=(0,),
        )
        self.assertIs(final.segments[0].editorial_transform, tracking_track)
        evidence = corroborate_dynamic_crop_motion_timeline_v3(
            render_plan=final,
            decodes=(decode,),
            authority=authority,
            deterministic_geometry_override_segment_indices=(0,),
            minimum_motion_applied_fraction=0.0,
            maximum_motion_applied_fraction=1.0,
        )

        segment = evidence.segment_evidence[0]
        self.assertEqual(
            segment.overlay_disposition,
            "deterministic_geometry_override_suppressed_model_track",
        )
        self.assertFalse(segment.framing_track_applied)
        self.assertFalse(segment.motion_event_applied)
        self.assertIn(
            "deterministic_geometry_override_suppressed_model_track",
            segment.failure_reasons,
        )
        self.assertEqual(evidence.motion_event_applied_count, 0)

    def test_geometry_override_cannot_relabel_a_model_overlay(self) -> None:
        track = self._track(
            (0.0, 1.02, 0.5, 0.5, 0.0),
            (0.8, 1.05, 0.5, 0.5, 0.0),
        )
        with self.assertRaisesRegex(
            DynamicCropMotionAuthorityV3Error,
            "lacks verified non-model geometry",
        ):
            apply_dynamic_crop_motion_authority_to_render_plan_v3(
                render_plan=self._plan((track,)),
                decodes=(self._decode(track),),
                authority=self._authority(),
                deterministic_geometry_override_segment_indices=(0,),
            )

    def test_physical_motion_below_frozen_logit_threshold_is_not_applied(self) -> None:
        track = self._track(
            (0.0, 1.02, 0.5, 0.5, 0.0),
            (0.8, 1.05, 0.5, 0.5, 0.0),
        )
        decode = self._decode(track, motion_logit=-0.001)
        provisional_plan = self._plan((track,))
        authority = self._authority()
        final_plan = apply_dynamic_crop_motion_authority_to_render_plan_v3(
            render_plan=provisional_plan,
            decodes=(decode,),
            authority=authority,
        )
        final_track = final_plan.segments[0].editorial_transform
        assert final_track is not None
        provisional_track = provisional_plan.segments[0].editorial_transform
        assert provisional_track is not None
        self.assertEqual(len(final_track.keyframes), 1)
        self.assertEqual(final_track.keyframes[0], provisional_track.evaluate(0.0))
        self.assertIn("suppressed_to_static_framing", final_track.reason)
        evidence = corroborate_dynamic_crop_motion_timeline_v3(
            render_plan=final_plan,
            decodes=(decode,),
            authority=authority,
            minimum_motion_applied_fraction=0.0,
            maximum_motion_applied_fraction=1.0,
        )

        segment = evidence.segment_evidence[0]
        self.assertTrue(segment.provisional_physical_motion_corroborated)
        self.assertFalse(segment.physical_motion_corroborated)
        self.assertFalse(segment.motion_logit_passed)
        self.assertFalse(segment.motion_event_applied)
        self.assertTrue(segment.motion_suppressed_to_static_framing)
        self.assertEqual(
            segment.overlay_disposition,
            "loso_or_physical_motion_suppressed_to_static_framing",
        )
        self.assertEqual(
            segment.motion_suppression_reasons,
            ("loso_motion_logit_below_threshold",),
        )
        self.assertFalse(segment.renderer_motion_without_loso_authority)
        self.assertEqual(
            evidence.renderer_motion_without_loso_authority_count,
            0,
        )
        self.assertTrue(evidence.motion_authority_consistent)
        self.assertIn(
            "motion_logit_below_loso_frozen_threshold",
            segment.failure_reasons,
        )

    def test_high_logit_subthreshold_curve_remains_static_framing(self) -> None:
        track = self._track(
            (0.0, 1.020, 0.5, 0.5, 0.0),
            (0.8, 1.027, 0.5, 0.5, 0.0),
        )
        decode = self._decode(track)
        provisional_plan = self._plan((track,))
        authority = self._authority()
        final_plan = apply_dynamic_crop_motion_authority_to_render_plan_v3(
            render_plan=provisional_plan,
            decodes=(decode,),
            authority=authority,
        )
        final_track = final_plan.segments[0].editorial_transform
        assert final_track is not None
        self.assertEqual(len(final_track.keyframes), 1)
        self.assertEqual(final_track.keyframes[0].zoom, 1.020)
        evidence = corroborate_dynamic_crop_motion_timeline_v3(
            render_plan=final_plan,
            decodes=(decode,),
            authority=authority,
            minimum_motion_applied_fraction=0.0,
            maximum_motion_applied_fraction=1.0,
        )

        segment = evidence.segment_evidence[0]
        self.assertTrue(segment.motion_logit_passed)
        self.assertFalse(segment.provisional_physical_motion_corroborated)
        self.assertFalse(segment.physical_motion_corroborated)
        self.assertFalse(segment.motion_event_applied)
        self.assertTrue(segment.motion_suppressed_to_static_framing)
        self.assertEqual(
            segment.motion_suppression_reasons,
            ("renderer_physical_range_below_threshold",),
        )

    def test_unsanitized_unqualified_motion_cannot_reach_evidence(self) -> None:
        dynamic = self._track(
            (0.0, 1.02, 0.5, 0.5, 0.0),
            (0.8, 1.05, 0.5, 0.5, 0.0),
        )
        subphysical = self._track(
            (0.0, 1.020, 0.5, 0.5, 0.0),
            (0.8, 1.027, 0.5, 0.5, 0.0),
        )
        cases = (
            (dynamic, self._decode(dynamic, motion_logit=-0.001)),
            (subphysical, self._decode(subphysical)),
        )
        for track, decode in cases:
            with self.subTest(track=track), self.assertRaisesRegex(
                DynamicCropMotionAuthorityV3Error,
                "unqualified provisional motion reached",
            ):
                corroborate_dynamic_crop_motion_timeline_v3(
                    render_plan=self._plan((track,)),
                    decodes=(decode,),
                    authority=self._authority(),
                    minimum_motion_applied_fraction=0.0,
                    maximum_motion_applied_fraction=1.0,
                )

    def test_authorized_motion_cannot_be_pre_suppressed_to_static(self) -> None:
        track = self._track(
            (0.0, 1.02, 0.5, 0.5, 0.0),
            (0.8, 1.05, 0.5, 0.5, 0.0),
        )
        decode = self._decode(track)
        provisional = self._plan((track,))
        provisional_track = provisional.segments[0].editorial_transform
        assert provisional_track is not None
        tampered = RenderPlan(
            sources=provisional.sources,
            segments=(
                RenderSegment(
                    source_index=0,
                    source_start=0.0,
                    duration=1.0,
                    master_start=0.0,
                    editorial_transform=(
                        motion._static_framing_track_from_provisional(
                            provisional_track
                        )
                    ),
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
        )
        with self.assertRaisesRegex(
            DynamicCropMotionAuthorityV3Error,
            "authorized motion was suppressed",
        ):
            apply_dynamic_crop_motion_authority_to_render_plan_v3(
                render_plan=tampered,
                decodes=(decode,),
                authority=self._authority(),
            )

    def test_production_motion_budget_fails_closed_below_and_above_range(self) -> None:
        static = self._track((0.0, 1.02, 0.5, 0.5, 0.0))
        dynamic = self._track(
            (0.0, 1.02, 0.5, 0.5, 0.0),
            (0.8, 1.05, 0.5, 0.5, 0.0),
        )
        for track in (static, dynamic):
            decode = self._decode(track)
            authority = self._authority()
            final = apply_dynamic_crop_motion_authority_to_render_plan_v3(
                render_plan=self._plan((track,)),
                decodes=(decode,),
                authority=authority,
            )
            with self.subTest(track=track), self.assertRaisesRegex(
                DynamicCropMotionAuthorityV3Error,
                "outside the frozen production budget",
            ):
                corroborate_dynamic_crop_motion_timeline_v3(
                    render_plan=final,
                    decodes=(decode,),
                    authority=authority,
                )

    def test_static_reframe_across_hard_cut_never_consumes_motion_budget(self) -> None:
        left = self._track((0.0, 1.02, 0.5, 0.5, 0.0))
        right = self._track((0.0, 1.10, 0.5, 0.5, 0.0))
        evidence = self._evidence(
            render_plan=self._plan(
                (left, right), source_indices=(0, 1)
            ),
            decodes=(
                self._decode(left),
                self._decode(
                    right, source_index=1, source_id="source-1"
                ),
            ),
            authority=self._authority(),
        )

        self.assertEqual(evidence.framing_track_applied_count, 2)
        self.assertEqual(evidence.motion_event_applied_count, 0)
        self.assertEqual(evidence.static_or_cut_time_reframe_excluded_count, 2)
        self.assertTrue(
            all(
                item.motion_budget_classification
                == "static_or_cut_time_reframe_excluded"
                for item in evidence.segment_evidence
            )
        )

    def test_keyframe_change_after_last_output_sample_is_not_corroborated(self) -> None:
        last_sample = 29.0 / 30.0
        track = self._track(
            (0.0, 1.02, 0.5, 0.5, 0.0),
            (last_sample, 1.02, 0.5, 0.5, 0.0),
            (1.0, 1.08, 0.5, 0.5, 0.0),
        )
        evidence = self._evidence(
            render_plan=self._plan((track,)),
            decodes=(self._decode(track),),
            authority=self._authority(),
        )

        segment = evidence.segment_evidence[0]
        self.assertEqual(segment.renderer_frame_count, 30)
        self.assertAlmostEqual(segment.zoom_range, 0.0)
        self.assertFalse(segment.physical_motion_corroborated)
        self.assertFalse(segment.motion_event_applied)

    def test_authority_and_evidence_tampering_fail_closed(self) -> None:
        authority = self._authority()
        tampered_authority = authority.to_dict()
        tampered_authority["motion_logit_threshold"] = -1.0
        with self.assertRaisesRegex(
            DynamicCropMotionAuthorityV3Error, "content hash"
        ):
            DynamicCropMotionAuthorityV3.from_dict(tampered_authority)

        track = self._track((0.0, 1.02, 0.5, 0.5, 0.0))
        timeline = self._evidence(
            render_plan=self._plan((track,)),
            decodes=(self._decode(track),),
            authority=authority,
        )
        segment_payload = timeline.segment_evidence[0].to_dict()
        segment_payload["motion_event_applied"] = True
        segment_payload["evidence_content_sha256"] = None
        segment_payload["evidence_content_sha256"] = _canonical_sha256(
            segment_payload
        )
        with self.assertRaisesRegex(
            DynamicCropMotionAuthorityV3Error, "three authorities"
        ):
            DynamicCropMotionSegmentEvidenceV3.from_dict(segment_payload)

        timeline_payload = deepcopy(timeline.to_dict())
        timeline_payload["motion_event_applied_count"] = 1
        timeline_payload["timeline_evidence_content_sha256"] = None
        timeline_payload["timeline_evidence_content_sha256"] = _canonical_sha256(
            timeline_payload
        )
        with self.assertRaisesRegex(
            DynamicCropMotionAuthorityV3Error, "summary differs"
        ):
            DynamicCropMotionTimelineEvidenceV3.from_dict(timeline_payload)

    def test_loader_binds_strict_loso_freeze_content_file_and_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            freeze_path = (Path(temporary) / "freeze.json").resolve()
            freeze_path.write_text("{}", encoding="utf-8")
            freeze = {
                "freeze_content_sha256": "d" * 64,
                "source_provenance": {"head_commit_oid": "e" * 40},
                "candidate_id": "production-balanced-test",
            }
            with patch(
                "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher."
                "load_dynamic_crop_v3_release_freeze",
                return_value=freeze,
            ) as strict_loader:
                authority = load_dynamic_crop_motion_authority_v3(freeze_path)

        strict_loader.assert_called_once_with(freeze_path.resolve())
        self.assertEqual(authority.release_freeze_content_sha256, "d" * 64)
        self.assertEqual(authority.source_commit_oid, "e" * 40)
        self.assertEqual(
            authority.release_freeze_file_sha256,
            motion._canonical_sha256({}),
        )
        self.assertEqual(authority.motion_logit_threshold, 0.0)
        self.assertIn(
            "predeclared_before_loso_heldout",
            authority.threshold_freeze_contract,
        )

    def test_direct_loader_rejects_symlink_and_mid_replay_file_change(self) -> None:
        freeze = {
            "freeze_content_sha256": "d" * 64,
            "source_provenance": {"head_commit_oid": "e" * 40},
            "candidate_id": "production-balanced-test",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = (root / "freeze.json").resolve()
            target.write_text("{}", encoding="utf-8")
            symlink = root / "freeze-link.json"
            symlink.symlink_to(target)
            with patch(
                "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher."
                "load_dynamic_crop_v3_release_freeze",
                return_value=freeze,
            ) as strict_loader:
                with self.assertRaisesRegex(
                    DynamicCropMotionAuthorityV3Error,
                    "normalized non-symlink",
                ):
                    load_dynamic_crop_motion_authority_v3(symlink)
            strict_loader.assert_not_called()

            def mutate_during_replay(path: Path) -> dict[str, object]:
                path.write_text('{"changed":true}', encoding="utf-8")
                return freeze

            with patch(
                "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher."
                "load_dynamic_crop_v3_release_freeze",
                side_effect=mutate_during_replay,
            ):
                with self.assertRaisesRegex(
                    DynamicCropMotionAuthorityV3Error,
                    "changed during strict replay",
                ):
                    load_dynamic_crop_motion_authority_v3(target)

    def test_registered_identity_is_strictly_replayed_before_runtime_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            freeze_path = (Path(temporary) / "freeze.json").resolve()
            freeze_path.write_text("{}", encoding="utf-8")
            identity = RegisteredStageEditMotionAuthorityIdentityV3(
                release_freeze_path=freeze_path,
                release_freeze_file_sha256=motion._file_sha256(freeze_path),
                release_freeze_content_sha256="d" * 64,
                source_commit_oid="e" * 40,
                candidate_id="production-balanced-test",
                motion_logit_threshold=0.0,
                threshold_freeze_contract=(
                    motion.DYNAMIC_CROP_MOTION_THRESHOLD_FREEZE_CONTRACT
                ),
            )
            freeze = {
                "freeze_content_sha256": "d" * 64,
                "source_provenance": {"head_commit_oid": "e" * 40},
                "candidate_id": "production-balanced-test",
            }
            with patch(
                "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher."
                "load_dynamic_crop_v3_release_freeze",
                return_value=freeze,
            ) as strict_loader:
                authority = (
                    load_dynamic_crop_motion_authority_from_registered_identity_v3(
                        identity
                    )
                )

            strict_loader.assert_called_once_with(freeze_path)
            self.assertEqual(
                identity.release_freeze_content_sha256,
                authority.release_freeze_content_sha256,
            )
            bad_identity = RegisteredStageEditMotionAuthorityIdentityV3(
                release_freeze_path=freeze_path,
                release_freeze_file_sha256="f" * 64,
                release_freeze_content_sha256="d" * 64,
                source_commit_oid="e" * 40,
                candidate_id="production-balanced-test",
                motion_logit_threshold=0.0,
                threshold_freeze_contract=(
                    motion.DYNAMIC_CROP_MOTION_THRESHOLD_FREEZE_CONTRACT
                ),
            )
            with self.assertRaisesRegex(
                DynamicCropMotionAuthorityV3Error,
                "file identity changed",
            ):
                load_dynamic_crop_motion_authority_from_registered_identity_v3(
                    bad_identity
                )

    def test_renderer_track_mismatch_and_missing_logit_fail_closed(self) -> None:
        rendered = self._track((0.0, 1.02, 0.5, 0.5, 0.0))
        selected = self._track((0.0, 1.03, 0.5, 0.5, 0.0))
        with self.assertRaisesRegex(
            DynamicCropMotionAuthorityV3Error, "exact StageEdit overlay"
        ):
            self._evidence(
                render_plan=self._plan((rendered,)),
                decodes=(self._decode(selected),),
                authority=self._authority(),
            )

        decode = self._decode(rendered)
        missing_logit = DynamicCropTrackDecodeV3(
            source_id=decode.source_id,
            source_index=decode.source_index,
            eligible=True,
            loss_masked=False,
            track=decode.track,
            predicted_keyframe_count=decode.predicted_keyframe_count,
            applied_keyframe_count=decode.applied_keyframe_count,
            minimum_safe_zoom_projection_count=0,
            rejection_reasons=(),
        )
        with self.assertRaisesRegex(
            DynamicCropMotionAuthorityV3Error, "lost its motion logit"
        ):
            self._evidence(
                render_plan=self._plan((rendered,)),
                decodes=(missing_logit,),
                authority=self._authority(),
            )

    def test_prediction_decode_retains_exact_motion_logit(self) -> None:
        prediction = DynamicCropHeadPredictionV3(
            source_id="source-0",
            source_index=0,
            values=((1.02, 0.5, 0.5, 0.0),) * 8,
            count_logits=(20.0,) + (-20.0,) * 7,
            time_interval_logits=(0.0,) * 35,
            motion_logit=-0.375,
        )
        decoded = prediction.decode(1.0)
        self.assertTrue(decoded.eligible)
        self.assertEqual(decoded.motion_logit, -0.375)

    def test_decode_strict_round_trip_retains_track_logit_duration_and_source(self) -> None:
        track = self._track(
            (0.0, 1.02, 0.5, 0.5, 0.0),
            (0.8, 1.05, 0.505, 0.5, 0.0),
        )
        decoded = self._decode(track, motion_logit=0.375)
        payload = decoded.to_dict()

        self.assertEqual(decoded, DynamicCropTrackDecodeV3.from_dict(payload))
        self.assertEqual(track.reason, payload["track"]["reason"])
        self.assertEqual(0.375, payload["motion_logit"])
        self.assertEqual(1.0, payload["shot_duration_seconds"])
        self.assertEqual("source-0", payload["source_id"])
        for mutate, message in (
            (lambda value: value.pop("track"), "keys are not exact"),
            (
                lambda value: value["track"]["keyframes"][0].update(
                    {"legacy_zoom": 1.02}
                ),
                "keyframe 0 keys are not exact",
            ),
            (
                lambda value: value.update({"eligible": 1}),
                "eligible must be boolean",
            ),
            (
                lambda value: value.update({"motion_logit": None}),
                "needs motion logit and shot duration",
            ),
            (
                lambda value: value.update({"source_id": None}),
                "needs an exact source identity",
            ),
        ):
            malformed = deepcopy(payload)
            mutate(malformed)
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                DynamicCropTrackDecodeV3.from_dict(malformed)


if __name__ == "__main__":
    unittest.main()
