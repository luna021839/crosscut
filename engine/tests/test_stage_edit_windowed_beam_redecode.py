from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from crosscut.group_transition import GroupTransitionPlan
from crosscut.pipeline import (
    _GroupRenderCandidateBundle,
    _attempt_stage_edit_windowed_beam_reedit,
)
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter import (
    DynamicCropHeadPredictionV2,
)
from crosscut.stage_edit_pipeline_bridge import (
    RetainedStageEditCandidate,
    StageEditPipelineBridgeResult,
)
from crosscut.stage_edit_repair_loop import (
    StageEditRepairExclusions,
    StageEditRepairRequest,
    StageEditRepairReview,
    StageEditRepairWindow,
)
from crosscut.stage_edit_semimarkov_beam import (
    StageEditSemiMarkovConfig,
    decode_stage_edit_semimarkov_beam,
)
from crosscut.stage_edit_windowed_beam_redecode import (
    STAGE_EDIT_WINDOWED_BEAM_CONTRACT,
    StageEditWindowedBeamRedecoder,
)
from crosscut.stage_edit_visual_candidate_adapter import (
    StageEditMaterializedVisualCandidate,
)
from crosscut.visual_score import VisualTimelinePlan
from engine.tests.test_pipeline import _visual_choice
from engine.tests.test_stage_edit_semimarkov_beam import (
    _frames,
    _metadata,
    _plan,
    _ready_report,
    _sources,
)


def _dynamic_prediction() -> DynamicCropHeadPredictionV2:
    return DynamicCropHeadPredictionV2(
        normalized_times=(0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        values=(
            (1.04, 0.500, 0.500, 0.0),
            (1.08, 0.505, 0.500, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        ),
        presence_logits=(
            20.0,
            20.0,
            -20.0,
            -20.0,
            -20.0,
            -20.0,
            -20.0,
            -20.0,
        ),
        motion_logit=20.0,
    )


class StageEditWindowedBeamRedecodeTests(unittest.TestCase):
    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_reviewer_window_invokes_real_beam_and_returns_spliceable_proposals(
        self,
        _readiness,
    ) -> None:
        duration = 8
        source_count = 4
        sources = _sources(source_count, float(duration))
        frames = tuple(
            replace(
                frame,
                crop=(99.0, 99.0, -99.0, 99.0),
                dynamic_crop_v2=_dynamic_prediction(),
            )
            for frame in _frames(duration, source_count)
        )
        config = StageEditSemiMarkovConfig(
            beam_width=16,
            timeline_count=8,
            minimum_unique_sources=3,
            maximum_source_occupancy_fraction=0.75,
        )
        decoded = decode_stage_edit_semimarkov_beam(
            checkpoint_directory=Path("/tmp/stageedit-production"),
            sources=sources,
            source_metadata=_metadata(source_count, float(duration)),
            head_frames=frames,
            section_plan=_plan(duration),
            output=Path("/tmp/stageedit-window-base.mp4"),
            config=config,
            fps=30.0,
        )
        retained = tuple(
            RetainedStageEditCandidate(
                rank=timeline.rank,
                score=timeline.score,
                edit=timeline.edit,
                exact_clock_reprojected=True,
                timewarped_segment_count=0,
            )
            for timeline in decoded.timelines
        )
        bridge = StageEditPipelineBridgeResult(
            preflight=SimpleNamespace(
                source_ids=tuple(f"source-{index}" for index in range(source_count)),
                checkpoint_directory=Path("/tmp/stageedit-production"),
            ),
            head_materialization=SimpleNamespace(frames=frames),
            semimarkov=decoded,
            retained_candidates=retained,
            selection_policy={
                "visual_baseline_replaced": False,
                "selected_for_render": False,
            },
        )
        base = retained[0].edit
        self.assertGreaterEqual(len(base.shots), 3)
        start_shot = 1
        end_shot = len(base.shots) - 1
        first = base.shots[start_shot].segment
        last = base.shots[end_shot - 1].segment
        start = base.master_audio_start + first.master_start
        end = base.master_audio_start + last.master_start + last.duration
        window = StageEditRepairWindow(
            issue_ids=("learned-window",),
            causes=("learned_editorial_discontinuity",),
            requested_start_seconds=start,
            requested_end_seconds=end,
            context_start_seconds=start,
            context_end_seconds=end,
            canonical_start_seconds=start,
            canonical_end_seconds=end,
            actual_start_seconds=start,
            actual_end_seconds=end,
            start_shot=start_shot,
            end_shot_exclusive=end_shot,
        )
        request = StageEditRepairRequest(
            iteration=1,
            base_edit=base,
            current_review=StageEditRepairReview(50.0, 50.0, -1.0),
            window=window,
            canonical_boundaries_seconds=tuple(
                frame.master_time_seconds for frame in frames
            )
            + (float(duration),),
            exclusions=StageEditRepairExclusions(),
            forbidden_edit_sha256s=(),
        )
        redecoder = StageEditWindowedBeamRedecoder(
            bridge_result=bridge,
            sources=sources,
            source_durations_seconds=(float(duration),) * source_count,
            source_start_offsets_seconds=(0.0,) * source_count,
            internal_cut_times_seconds=((),) * source_count,
            source_timewarps=(None,) * source_count,
            source_quality_log_priors=(0.0,) * source_count,
            output=Path("/tmp/stageedit-window-redecode.mp4"),
        )

        with patch(
            "crosscut.stage_edit_windowed_beam_redecode."
            "decode_stage_edit_semimarkov_beam",
            wraps=decode_stage_edit_semimarkov_beam,
        ) as actual_decoder:
            proposals = redecoder(request)

        actual_decoder.assert_called_once()
        self.assertGreater(len(proposals), 0, redecoder.audit_payload())
        proposal = proposals[0]
        self.assertTrue(proposal.audit["actual_public_decoder_invoked"])
        self.assertTrue(proposal.audit["exact_source_clock_reprojected"])
        self.assertTrue(
            all(
                shot.segment.editorial_transform is not None
                for shot in proposal.replacement_shots
            )
        )
        spliced = base.replace_window(
            start_shot=start_shot,
            end_shot_exclusive=end_shot,
            replacement_shots=proposal.replacement_shots,
            replacement_boundaries=proposal.replacement_boundaries,
            reason="test actual local beam",
            score_before=None,
            score_after=None,
        )
        self.assertEqual(base.duration_seconds, spliced.duration_seconds)
        self.assertEqual(base.shots[:start_shot], spliced.shots[:start_shot])
        self.assertEqual(
            base.shots[end_shot:],
            spliced.shots[-len(base.shots[end_shot:]):],
        )
        audit = redecoder.audit_payload()
        self.assertEqual(STAGE_EDIT_WINDOWED_BEAM_CONTRACT, audit["contract"])
        self.assertEqual(1, audit["call_count"])
        self.assertTrue(audit["calls"][0]["actual_public_decoder_invoked"])

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_pipeline_commits_actual_beam_proposal_only_after_proxy_revalidation(
        self,
        _readiness,
    ) -> None:
        duration = 8
        source_count = 4
        sources = _sources(source_count, float(duration))
        frames = tuple(
            replace(
                frame,
                crop=(99.0, 99.0, -99.0, 99.0),
                dynamic_crop_v2=_dynamic_prediction(),
            )
            for frame in _frames(duration, source_count)
        )
        config = StageEditSemiMarkovConfig(
            beam_width=16,
            timeline_count=8,
            minimum_unique_sources=3,
            maximum_source_occupancy_fraction=0.75,
        )
        decoded = decode_stage_edit_semimarkov_beam(
            checkpoint_directory=Path("/tmp/stageedit-production"),
            sources=sources,
            source_metadata=_metadata(source_count, float(duration)),
            head_frames=frames,
            section_plan=_plan(duration),
            output=Path("/tmp/stageedit-window-selection-base.mp4"),
            config=config,
            fps=30.0,
        )
        retained = tuple(
            RetainedStageEditCandidate(
                rank=timeline.rank,
                score=timeline.score,
                edit=timeline.edit,
                exact_clock_reprojected=True,
                timewarped_segment_count=0,
            )
            for timeline in decoded.timelines
        )
        bridge = StageEditPipelineBridgeResult(
            preflight=SimpleNamespace(
                source_ids=tuple(f"source-{index}" for index in range(source_count)),
                checkpoint_directory=Path("/tmp/stageedit-production"),
            ),
            head_materialization=SimpleNamespace(frames=frames),
            semimarkov=decoded,
            retained_candidates=retained,
            selection_policy={
                "visual_baseline_replaced": False,
                "selected_for_render": False,
            },
        )
        base_retained = retained[0]
        base = base_retained.edit
        boundary_index = max(0, len(base.boundaries) // 2)
        outgoing = base.shots[boundary_index].segment
        issue_at = (
            base.master_audio_start + outgoing.master_start + outgoing.duration
        )
        learned_issue = {
            "issue_id": "learned-local-beam-selection",
            "start_seconds": issue_at - 0.05,
            "end_seconds": issue_at + 0.05,
            "causes": ["learned_editorial_discontinuity"],
            "severity": "error",
        }

        candidate_record = SimpleNamespace(
            candidate_index=0,
            review_rank=0,
            cadence_profile="stage_edit_test",
        )

        def materialized_for(
            candidate: RetainedStageEditCandidate,
        ) -> StageEditMaterializedVisualCandidate:
            visual_plan = VisualTimelinePlan(
                choices=tuple(
                    _visual_choice(
                        index,
                        shot.segment.source_index,
                        base.master_audio_start + shot.segment.master_start,
                        base.master_audio_start
                        + shot.segment.master_start
                        + shot.segment.duration,
                    )
                    for index, shot in enumerate(candidate.edit.shots)
                ),
                optimizer_score=candidate.score,
            )
            return StageEditMaterializedVisualCandidate(
                candidate_record=candidate_record,
                base_materialized=SimpleNamespace(
                    visual_plan=visual_plan,
                    segments=tuple(
                        shot.segment for shot in candidate.edit.shots
                    ),
                ),
                visual_plan=visual_plan,
                retained_candidate=candidate,
            )

        def bundle_for(
            materialized: StageEditMaterializedVisualCandidate,
            **_kwargs,
        ) -> _GroupRenderCandidateBundle:
            edit = materialized.retained_candidate.edit
            return _GroupRenderCandidateBundle(
                candidate_index=0,
                review_rank=0,
                cadence_profile="stage_edit_test",
                pre_render_score=materialized.retained_candidate.score,
                visual_timeline=materialized.visual_plan,
                boundary_repairs=None,
                timeline_review=SimpleNamespace(passed=True),
                face_match_plan=None,
                group_transition_plan=GroupTransitionPlan(
                    tuple(boundary.transition for boundary in edit.boundaries)
                ),
                boundary_affine_plan=None,
                editorial_motion_plan=None,
                render_plan=edit.to_render_plan(),
                candidate_origin="explicit_stage_edit_generation",
                candidate_lineage=({"stage": "initial_exact8_proxy_selection"},),
            )

        selected_materialized = materialized_for(base_retained)
        selected_bundle = bundle_for(selected_materialized)
        original_review = {
            "passed": False,
            "weighted_score": 10.0,
            "partial_reedit_boundary_issue_indices": [boundary_index],
        }

        def repair_objectives(_edit, payload):
            value = 100.0 if payload and payload.get("proposal_proxy") else 10.0
            return StageEditRepairReview(value, value, value)

        def proxy_selection(*_args, **_kwargs):
            return SimpleNamespace(
                selected=SimpleNamespace(
                    review_payload={
                        "passed": True,
                        "weighted_score": 100.0,
                        "proposal_proxy": True,
                        "partial_reedit_boundary_issue_indices": [],
                    }
                )
            )

        def apply_repaired_render_contract(bundle, _materialized):
            return replace(
                bundle,
                candidate_lineage=bundle.candidate_lineage
                + (
                    {
                        "stage": "proposal_dynamic_crop_overlay",
                        "dynamic_crop_v2_suppressed_segment_indices": [1],
                    },
                ),
            )

        request = SimpleNamespace(
            sources=sources,
            source_cache_directory=None,
        )
        learned_reviewer = SimpleNamespace(
            review_candidate=Mock(
                return_value=SimpleNamespace(
                    candidate_index=0,
                    repair_issues=(),
                    materialization=SimpleNamespace(
                        feature_snapshot_sha256="a" * 64
                    ),
                )
            ),
            provenance_dict=lambda: {"checkpoint_id": "reviewer-test"},
        )
        with tempfile.TemporaryDirectory() as raw_scratch:
            with (
                patch(
                    "crosscut.stage_edit_windowed_beam_redecode."
                    "decode_stage_edit_semimarkov_beam",
                    wraps=decode_stage_edit_semimarkov_beam,
                ) as actual_decoder,
                patch(
                    "crosscut.pipeline.materialize_stage_edit_visual_candidate",
                    side_effect=lambda candidate, **_kwargs: materialized_for(
                        candidate
                    ),
                ) as materialize_candidate,
                patch(
                    "crosscut.pipeline._derive_complete_group_candidate",
                    side_effect=bundle_for,
                ) as derive_candidate,
                patch(
                    "crosscut.pipeline._apply_stage_edit_candidate_render_contract",
                    side_effect=apply_repaired_render_contract,
                ) as apply_render_contract,
                patch(
                    "crosscut.pipeline.review_rendered_candidates",
                    side_effect=proxy_selection,
                ) as review_proxy,
                patch(
                    "crosscut.pipeline._stage_edit_proxy_repair_review",
                    side_effect=repair_objectives,
                ),
            ):
                result = _attempt_stage_edit_windowed_beam_reedit(
                    selected_bundle,
                    selected_materialized,
                    original_review,
                    bridge_result=bridge,
                    learned_repair_issues=(learned_issue,),
                    request=request,
                    eligible_indices=tuple(range(source_count)),
                    audio_offsets=(0.0,) * source_count,
                    sync_scores=(0.95,) * source_count,
                    scene_cut_times=((),) * source_count,
                    source_timewarps=(None,) * source_count,
                    source_aspect_ratios=(16.0 / 9.0,) * source_count,
                    source_durations_seconds=(float(duration),) * source_count,
                    beat_period_seconds=0.5,
                    phrase_times=(),
                    section_plan=_plan(duration),
                    common_master_start=base.master_audio_start,
                    width=base.width,
                    height=base.height,
                    scratch_directory=Path(raw_scratch),
                    learned_reviewer_session=learned_reviewer,
                )

        actual_decoder.assert_called_once()
        self.assertGreater(review_proxy.call_count, 0)
        self.assertTrue(
            all(
                call.args[0][0].rank_metadata is not None
                and call.args[0][0].rank_metadata.source_switches_per_minute
                >= 0.0
                for call in review_proxy.call_args_list
            )
        )
        self.assertEqual(review_proxy.call_count, materialize_candidate.call_count)
        self.assertEqual(review_proxy.call_count, derive_candidate.call_count)
        self.assertEqual(review_proxy.call_count, apply_render_contract.call_count)
        self.assertEqual(
            review_proxy.call_count,
            learned_reviewer.review_candidate.call_count,
        )
        self.assertEqual(
            "windowed_semimarkov_redecode",
            result.audit_payload["selected_version"],
        )
        self.assertTrue(result.audit_payload["changed"])
        self.assertGreater(
            result.audit_payload["repair_loop"]["committed_iteration_count"],
            0,
        )
        self.assertIn(
            "learned_reviewer_issue_suggestions",
            result.audit_payload["repair_loop"]["final_review"]["audit"],
        )
        self.assertNotEqual(
            result.audit_payload["repair_loop"]["initial_edit_sha256"],
            result.audit_payload["repair_loop"]["final_edit_sha256"],
        )
        self.assertEqual(
            1,
            result.audit_payload["windowed_beam_redecode"]["call_count"],
        )
        self.assertTrue(result.selected_review_payload["proposal_proxy"])
        self.assertTrue(result.selected_review_payload["passed"])
        self.assertIsNot(result.selected_bundle, selected_bundle)
        selected_evidence = result.audit_payload["selected_redecode_evidence"]
        self.assertTrue(selected_evidence["actual_semimarkov_redecode"])
        self.assertTrue(selected_evidence["new_proxy_passed"])
        self.assertTrue(
            selected_evidence["learned_reviewer_reinvoked_after_splice"]
        )
        self.assertEqual(
            [1],
            selected_evidence[
                "dynamic_crop_v2_suppressed_segment_indices"
            ],
        )
        self.assertTrue(
            selected_evidence["proposal_materialization_lineage_preserved"]
        )
        lineage = result.selected_bundle.candidate_lineage[-1]
        self.assertTrue(lineage["actual_public_decoder_invoked"])
        self.assertTrue(lineage["actual_semimarkov_redecode"])
        self.assertTrue(lineage["new_proxy_passed"])
        self.assertTrue(lineage["learned_reviewer_reinvoked_after_splice"])
        self.assertEqual(
            [1], lineage["dynamic_crop_v2_suppressed_segment_indices"]
        )
        self.assertTrue(lineage["decoded_proxy_revalidated_after_splice"])
        self.assertFalse(lineage["heuristic_timing_repair_used_as_reviewer_repair"])
        self.assertTrue(
            any(
                item.get("dynamic_crop_v2_suppressed_segment_indices") == [1]
                for item in result.selected_bundle.candidate_lineage
            )
        )


if __name__ == "__main__":
    unittest.main()
