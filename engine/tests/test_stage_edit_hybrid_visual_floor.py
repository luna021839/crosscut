from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.cut_local_sync import resolve_cut_local_sync_activation
from crosscut.ive import AspectRatio
from crosscut.render import RenderSegment
from crosscut.section_director import (
    SectionDirective,
    SectionPlan,
    SectionRole,
)
from crosscut.stage_edit import StageEdit, StageShot
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
    DynamicCropHeadPredictionV3,
)
from crosscut.stage_edit_pipeline_bridge import (
    RetainedStageEditCandidate,
    StageEditPipelineBridgeResult,
)
from crosscut.stage_edit_semimarkov_beam import StageEditHeadFrame
from crosscut.stage_edit_visual_candidate_adapter import (
    HYBRID_VISUAL_FLOOR_CANDIDATE_ORIGIN,
    HYBRID_VISUAL_FLOOR_CONTRACT,
    STAGE_EDIT_HYBRID_VISUAL_FLOOR_CANDIDATE_POOL,
    StageEditMaterializedVisualCandidate,
    StageEditVisualCandidateRecord,
    StageEditVisualCandidateError,
    _build_hybrid_visual_floor_retained_candidate_from_parts,
    materialize_stage_edit_visual_candidates,
)
from crosscut.timeline import ShotType
from crosscut.visual_candidate_materialization import (
    MaterializedVisualTimelineCandidate,
)
from crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan


def _section_plan() -> SectionPlan:
    return SectionPlan(
        (
            SectionDirective(
                section_index=0,
                role=SectionRole.CHORUS_LIKE,
                start_seconds=0.0,
                end_seconds=2.0,
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


def _prediction(source_index: int) -> DynamicCropHeadPredictionV3:
    return DynamicCropHeadPredictionV3(
        source_id=f"source-{source_index}",
        source_index=source_index,
        values=(
            (1.02, 0.50, 0.50, 0.0),
            (1.06, 0.52, 0.49, 0.0),
            *((1.06, 0.52, 0.49, 0.0),) * 6,
        ),
        count_logits=(0.0, 8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0),
        time_interval_logits=(0.0,) * 35,
        motion_logit=1.2,
    )


def _frame(
    time_seconds: float,
    *,
    effective_mask: tuple[bool, bool] = (True, True),
) -> StageEditHeadFrame:
    return StageEditHeadFrame(
        master_time_seconds=time_seconds,
        cut_logit=0.0,
        source_logits=(0.0, 0.0),
        duration_logits=(0.0,),
        transition_logits=(0.0,),
        crop=(1.0, 0.5, 0.5, 0.0),
        available_source_mask=effective_mask,
        hard_source_gate_mask=effective_mask,
        section_progress=min(1.0, time_seconds / 2.0),
        phrase_progress=min(1.0, time_seconds / 2.0),
        section_role=SectionRole.CHORUS_LIKE.value,
        dynamic_crop_v3_by_source=tuple(
            _prediction(source_index) if allowed else None
            for source_index, allowed in enumerate(effective_mask)
        ),
    )


def _visual_plan() -> VisualTimelinePlan:
    return VisualTimelinePlan(
        choices=(
            VisualTimelineChoice(
                segment_index=0,
                source_index=0,
                canonical_master_start=0.0,
                canonical_master_end=1.0,
                source_local_start=0.0,
                source_local_end=1.0,
                quality_score=0.9,
                composition_score=0.8,
                boundary_score=0.7,
                shot_scale=0.6,
                shot_type=ShotType.CLOSE_UP,
                shot_id="source-0-shot",
            ),
            VisualTimelineChoice(
                segment_index=1,
                source_index=1,
                canonical_master_start=1.0,
                canonical_master_end=2.0,
                source_local_start=1.0,
                source_local_end=2.0,
                quality_score=0.9,
                composition_score=0.8,
                boundary_score=0.7,
                shot_scale=0.6,
                shot_type=ShotType.CLOSE_UP,
                shot_id="source-1-shot",
            ),
        ),
        optimizer_score=4.5,
    )


def _template() -> StageEdit:
    sources = (Path("/tmp/hybrid-a.mp4"), Path("/tmp/hybrid-b.mp4"))
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
                shot_index=0,
                section_index=0,
                segment=RenderSegment(0, 0.0, 2.0, 0.0),
                decision_id="template",
            ),
        ),
        boundaries=(),
        output=Path("/tmp/stage-edit.mp4"),
    )


class StageEditHybridVisualFloorTests(unittest.TestCase):
    def test_reviewed_visual_path_receives_source_exact_v3_decodes(self) -> None:
        retained = _build_hybrid_visual_floor_retained_candidate_from_parts(
            visual_plan=_visual_plan(),
            segments=(
                RenderSegment(0, 0.0, 1.0, 0.0),
                RenderSegment(1, 1.0, 1.0, 1.0),
            ),
            common_master_start=0.0,
            head_frames=(_frame(0.0), _frame(0.5), _frame(1.0)),
            canonical_source_ids=("source-0", "source-1"),
            template_edit=_template(),
            candidate_rank=9,
            optimizer_score=4.5,
            cadence_profile="reviewed-floor",
            path_signature=("source-0-shot", "source-1-shot"),
            continuity_authority="test-continuity",
        )

        self.assertEqual(
            retained.candidate_origin,
            HYBRID_VISUAL_FLOOR_CANDIDATE_ORIGIN,
        )
        self.assertEqual(
            tuple(shot.segment.source_index for shot in retained.edit.shots),
            (0, 1),
        )
        self.assertEqual(
            tuple(
                shot.dynamic_crop_decode_v3.source_index
                for shot in retained.edit.shots
                if shot.dynamic_crop_decode_v3 is not None
            ),
            (0, 1),
        )
        self.assertTrue(
            all(
                shot.segment.editorial_transform is not None
                for shot in retained.edit.shots
            )
        )
        self.assertEqual(
            retained.edit.candidate_history[0]["generator"],
            HYBRID_VISUAL_FLOOR_CONTRACT,
        )
        self.assertEqual(
            retained.to_dict()["candidate_origin"],
            HYBRID_VISUAL_FLOOR_CANDIDATE_ORIGIN,
        )

    def test_source_gate_failure_inside_visual_shot_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            StageEditVisualCandidateError,
            "fails a Transformer hard gate inside shot 0",
        ):
            _build_hybrid_visual_floor_retained_candidate_from_parts(
                visual_plan=_visual_plan(),
                segments=(
                    RenderSegment(0, 0.0, 1.0, 0.0),
                    RenderSegment(1, 1.0, 1.0, 1.0),
                ),
                common_master_start=0.0,
                head_frames=(
                    _frame(0.0),
                    _frame(0.5, effective_mask=(False, True)),
                    _frame(1.0),
                ),
                canonical_source_ids=("source-0", "source-1"),
                template_edit=_template(),
                candidate_rank=9,
                optimizer_score=4.5,
                cadence_profile="reviewed-floor",
                path_signature=("source-0-shot", "source-1-shot"),
                continuity_authority="test-continuity",
            )

    def test_retained_candidate_origin_is_closed_enum(self) -> None:
        base = _build_hybrid_visual_floor_retained_candidate_from_parts(
            visual_plan=_visual_plan(),
            segments=(
                RenderSegment(0, 0.0, 1.0, 0.0),
                RenderSegment(1, 1.0, 1.0, 1.0),
            ),
            common_master_start=0.0,
            head_frames=(_frame(0.0), _frame(0.5), _frame(1.0)),
            canonical_source_ids=("source-0", "source-1"),
            template_edit=_template(),
            candidate_rank=9,
            optimizer_score=4.5,
            cadence_profile="reviewed-floor",
            path_signature=("source-0-shot", "source-1-shot"),
            continuity_authority="test-continuity",
        )
        with self.assertRaisesRegex(ValueError, "origin is unsupported"):
            replace(base, candidate_origin="raw_unreviewed_baseline")

    def test_batch_keeps_seven_passing_learned_paths_plus_typed_floor(self) -> None:
        hybrid_retained = _build_hybrid_visual_floor_retained_candidate_from_parts(
            visual_plan=_visual_plan(),
            segments=(
                RenderSegment(0, 0.0, 1.0, 0.0),
                RenderSegment(1, 1.0, 1.0, 1.0),
            ),
            common_master_start=0.0,
            head_frames=(_frame(0.0), _frame(0.5), _frame(1.0)),
            canonical_source_ids=("source-0", "source-1"),
            template_edit=_template(),
            candidate_rank=9,
            optimizer_score=4.5,
            cadence_profile="reviewed-floor",
            path_signature=("source-0-shot", "source-1-shot"),
            continuity_authority="test-continuity",
        )
        learned = tuple(
            replace(
                hybrid_retained,
                rank=rank,
                candidate_origin="stage_edit_semimarkov",
            )
            for rank in range(1, 9)
        )
        bridge = object.__new__(StageEditPipelineBridgeResult)
        object.__setattr__(
            bridge,
            "preflight",
            SimpleNamespace(
                dynamic_crop_version=3,
                source_ids=("source-0", "source-1"),
            ),
        )
        object.__setattr__(
            bridge,
            "head_materialization",
            SimpleNamespace(dynamic_crop_version=3),
        )
        object.__setattr__(bridge, "learned_section_director", None)
        object.__setattr__(bridge, "retained_candidates", learned)

        review = SimpleNamespace(passed=True, to_dict=lambda: {"passed": True})

        def wrapped(
            retained: RetainedStageEditCandidate,
            candidate_index: int,
        ) -> StageEditMaterializedVisualCandidate:
            base_record = SimpleNamespace(
                review=review,
                optimizer_score=retained.score,
                path_signature=(f"candidate-{candidate_index}",),
                to_dict=lambda: {"passed": True},
            )
            record = StageEditVisualCandidateRecord(
                candidate_index=candidate_index,
                review_rank=candidate_index,
                cadence_profile="test",
                base_record=base_record,
                retained_candidate=retained,
            )
            return StageEditMaterializedVisualCandidate(
                candidate_record=record,
                base_materialized=None,
                visual_plan=replace(
                    _visual_plan(),
                    selected_candidate_index=candidate_index,
                ),
                retained_candidate=retained,
            )

        raw_floor = object.__new__(MaterializedVisualTimelineCandidate)

        def materialize_one(candidate, *, candidate_index, **_kwargs):
            if candidate.rank == 3:
                raise StageEditVisualCandidateError("unsafe learned path")
            return wrapped(candidate, candidate_index)

        with (
            patch(
                "crosscut.stage_edit_visual_candidate_adapter."
                "materialize_stage_edit_visual_candidate",
                side_effect=materialize_one,
            ),
            patch(
                "crosscut.stage_edit_visual_candidate_adapter."
                "_build_hybrid_visual_floor_retained_candidate",
                return_value=hybrid_retained,
            ),
            patch(
                "crosscut.stage_edit_visual_candidate_adapter."
                "_wrap_hybrid_visual_floor_candidate",
                side_effect=lambda _raw, retained, *, candidate_index: wrapped(
                    retained,
                    candidate_index,
                ),
            ),
        ):
            batch = materialize_stage_edit_visual_candidates(
                bridge,
                sources=_template().sources,
                audio_offsets=(0.0, 0.0),
                sync_scores=(1.0, 1.0),
                scene_cut_times=((), ()),
                source_timewarps=(None, None),
                hybrid_visual_floor_candidate=raw_floor,
            )

        self.assertEqual(len(batch.candidates), 8)
        self.assertEqual(
            tuple(item.candidate_index for item in batch.candidates),
            tuple(range(8)),
        )
        self.assertEqual(batch.hybrid_visual_floor_candidate_count, 1)
        self.assertEqual(
            batch.candidates[-1].candidate_origin,
            HYBRID_VISUAL_FLOOR_CANDIDATE_ORIGIN,
        )
        self.assertEqual(len(batch.learned_candidate_rejections), 1)
        self.assertEqual(
            batch.candidate_pool,
            STAGE_EDIT_HYBRID_VISUAL_FLOOR_CANDIDATE_POOL,
        )

    def test_offline_activation_accepts_only_the_typed_hybrid_pool_marker(self) -> None:
        analysis = {
            "stage_edit_generation": {
                "activation": {"activation": "explicit_request_bundle"}
            },
            "render": {
                "candidate_origin": "explicit_stage_edit_generation",
                "stage_edit_promotion_contract": "test-contract",
            },
        }
        selection = {
            "stage_edit_render_promotion": {
                "activation": "explicit_stage_edit_generation_request_only",
                "candidate_pool": (
                    STAGE_EDIT_HYBRID_VISUAL_FLOOR_CANDIDATE_POOL
                ),
            }
        }
        resolved = resolve_cut_local_sync_activation(analysis, selection)
        self.assertTrue(resolved["valid"])
        self.assertEqual(resolved["errors"], [])


if __name__ == "__main__":
    unittest.main()
