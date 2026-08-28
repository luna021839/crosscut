from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

from crosscut.ive import AspectRatio
from crosscut.pipeline import (
    _GroupRenderCandidateBundle,
    _apply_stage_edit_candidate_render_contract,
    _refresh_stage_edit_dynamic_crop_motion_evidence_v3,
)
from crosscut.render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderPlan,
    RenderSegment,
)
from crosscut.stage_edit import StageShot
from crosscut.stage_edit_dynamic_crop_motion_authority_v3 import (
    DynamicCropMotionAuthorityV3,
)
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
    DynamicCropTrackDecodeV3,
)
from crosscut.stage_edit_visual_candidate_adapter import (
    StageEditMaterializedVisualCandidate,
    StageEditVisualCandidateError,
)


class _PassingReview:
    passed = True

    def to_dict(self) -> dict[str, object]:
        return {"passed": True}


class _CandidateRecord:
    candidate_index = 0
    review_rank = 0
    cadence_profile = "neutral-focused-test"
    review = _PassingReview()

    def to_dict(self) -> dict[str, object]:
        return {"candidate_index": 0}


class _PayloadObject:
    def to_dict(self, *args: object) -> dict[str, object]:
        return {}


class PipelineDynamicCropV3MotionContractTests(unittest.TestCase):
    @staticmethod
    def _authority() -> DynamicCropMotionAuthorityV3:
        return DynamicCropMotionAuthorityV3.create(
            release_freeze_content_sha256="a" * 64,
            release_freeze_file_sha256="b" * 64,
            source_commit_oid="c" * 40,
            candidate_id="neutral-release-candidate",
        )

    @staticmethod
    def _track(*, moving: bool) -> EditorialTransformTrack:
        keyframes = (
            EditorialTransformKeyframe(0.0, 1.02, 0.5, 0.5, 0.0),
            EditorialTransformKeyframe(0.8, 1.05, 0.5, 0.5, 0.0),
        ) if moving else (
            EditorialTransformKeyframe(0.0, 1.02, 0.5, 0.5, 0.0),
        )
        return EditorialTransformTrack(
            keyframes,
            reason="stage_edit_dynamic_crop_head_v3",
        )

    def _fixture(
        self,
        *,
        model_motion_index: int = 0,
        deterministic_geometry_override_index: int | None = None,
    ) -> tuple[_GroupRenderCandidateBundle, StageEditMaterializedVisualCandidate]:
        shots: list[StageShot] = []
        render_segments: list[RenderSegment] = []
        for index in range(10):
            source_index = index % 2
            track = self._track(moving=index == model_motion_index)
            segment = RenderSegment(
                source_index=source_index,
                source_start=0.0,
                duration=1.0,
                master_start=float(index),
                editorial_transform=track,
            )
            decode = DynamicCropTrackDecodeV3(
                source_id=f"camera-{source_index}",
                source_index=source_index,
                eligible=True,
                loss_masked=False,
                track=track,
                predicted_keyframe_count=len(track.keyframes),
                applied_keyframe_count=len(track.keyframes),
                minimum_safe_zoom_projection_count=0,
                rejection_reasons=(),
                motion_logit=10.0,
                shot_duration_seconds=1.0,
            )
            shots.append(
                StageShot(
                    shot_index=index,
                    section_index=0,
                    segment=segment,
                    decision_id=f"decision-{index}",
                    section_role="chorus_like",
                    section_progress=index / 10.0,
                    phrase_progress=(index % 4) / 4.0,
                    dynamic_crop_decode_v3=decode,
                )
            )
            deterministic_track = (
                EditorialTransformTrack(
                    (
                        EditorialTransformKeyframe(
                            0.0, 1.08, 0.49, 0.5, 0.0
                        ),
                        EditorialTransformKeyframe(
                            0.8, 1.00, 0.5, 0.5, 0.0
                        ),
                    ),
                    reason="boundary_local_exact_subject_alignment_b0",
                )
                if index == deterministic_geometry_override_index
                else None
            )
            render_segments.append(
                RenderSegment(
                    source_index=source_index,
                    source_start=0.0,
                    duration=1.0,
                    master_start=float(index),
                    editorial_transform=deterministic_track,
                )
            )
        plan = RenderPlan(
            sources=(Path("camera-0.mp4"), Path("camera-1.mp4")),
            segments=tuple(render_segments),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("candidate.mp4"),
            fps=30.0,
        )
        edit = SimpleNamespace(
            shots=tuple(shots),
            candidate_history=({"dynamic_crop_version": 3},),
        )
        retained = SimpleNamespace(rank=0, score=1.0, edit=edit)
        materialized = StageEditMaterializedVisualCandidate(
            candidate_record=_CandidateRecord(),
            base_materialized=SimpleNamespace(segments=tuple(render_segments)),
            visual_plan=_PayloadObject(),
            retained_candidate=retained,
        )
        bundle = _GroupRenderCandidateBundle(
            candidate_index=0,
            review_rank=0,
            cadence_profile="neutral-focused-test",
            pre_render_score=1.0,
            visual_timeline=_PayloadObject(),
            boundary_repairs=None,
            timeline_review=None,
            face_match_plan=None,
            group_transition_plan=_PayloadObject(),
            boundary_affine_plan=None,
            editorial_motion_plan=None,
            render_plan=plan,
        )
        return bundle, materialized

    def test_v3_pipeline_applies_only_corroborated_motion(self) -> None:
        bundle, materialized = self._fixture()
        result = _apply_stage_edit_candidate_render_contract(
            bundle,
            materialized,
            motion_authority_v3=self._authority(),
        )

        evidence = result.dynamic_crop_motion_timeline_v3
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(result.stage_edit_dynamic_crop_version, 3)
        self.assertEqual(len(result.stage_edit_dynamic_crop_v3_decodes or ()), 10)
        self.assertEqual(evidence.motion_event_applied_count, 1)
        self.assertEqual(evidence.motion_event_applied_fraction, 0.1)
        self.assertEqual(
            sum(
                item.motion_suppressed_to_static_framing
                for item in evidence.segment_evidence
            ),
            9,
        )

    def test_v3_pipeline_fails_without_release_authority(self) -> None:
        bundle, materialized = self._fixture()

        with self.assertRaisesRegex(
            StageEditVisualCandidateError,
            "strictly replayed production motion authority",
        ):
            _apply_stage_edit_candidate_render_contract(bundle, materialized)

    def test_v3_pipeline_preserves_verified_geometry_before_model_crop(self) -> None:
        bundle, materialized = self._fixture(
            model_motion_index=1,
            deterministic_geometry_override_index=0,
        )
        original_track = bundle.render_plan.segments[0].editorial_transform
        assert original_track is not None

        result = _apply_stage_edit_candidate_render_contract(
            bundle,
            materialized,
            motion_authority_v3=self._authority(),
        )

        self.assertIs(
            result.render_plan.segments[0].editorial_transform,
            original_track,
        )
        evidence = result.dynamic_crop_motion_timeline_v3
        assert evidence is not None
        self.assertEqual(
            evidence.segment_evidence[0].overlay_disposition,
            "deterministic_geometry_override_suppressed_model_track",
        )
        self.assertEqual(evidence.motion_event_applied_count, 1)
        event = result.candidate_lineage[-1]
        self.assertFalse(event["ordinary_editorial_motion_discarded"])
        self.assertEqual(
            event["deterministic_geometry_precedence_segment_indices"],
            [0],
        )

    def test_final_output_replacement_refreshes_full_plan_evidence(self) -> None:
        bundle, materialized = self._fixture()
        result = _apply_stage_edit_candidate_render_contract(
            bundle,
            materialized,
            motion_authority_v3=self._authority(),
        )
        assert result.dynamic_crop_motion_timeline_v3 is not None
        before = result.dynamic_crop_motion_timeline_v3
        rebound = _refresh_stage_edit_dynamic_crop_motion_evidence_v3(
            replace(
                result,
                render_plan=replace(
                    result.render_plan,
                    output=Path("delivery.mp4"),
                ),
            )
        )

        assert rebound.dynamic_crop_motion_timeline_v3 is not None
        after = rebound.dynamic_crop_motion_timeline_v3
        self.assertNotEqual(
            before.render_plan_content_sha256,
            after.render_plan_content_sha256,
        )
        self.assertEqual(after.motion_event_applied_count, 1)
        self.assertEqual(after.to_dict(), rebound.to_dict()["dynamic_crop_motion_v3"])


if __name__ == "__main__":
    unittest.main()
