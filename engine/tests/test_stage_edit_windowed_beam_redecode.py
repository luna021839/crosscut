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
    _stage_edit_montage_analysis_payload,
    _stage_edit_perceptual_repair_targets,
    _stage_edit_exact_shot_clock_from_edit,
    _stage_edit_proxy_repair_review,
    _stage_edit_targeted_source_substitution_evidence,
)
from crosscut.render import NormalizedCrop, RenderPlan
from crosscut.rendered_proxy_review import (
    PERCEPTUAL_SOURCE_ADMISSION_ANALYSIS_BASIS,
)
from crosscut.rendered_montage_grammar_review import (
    _exact_stage_edit_anchor_metrics,
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


def _perceptual_review_fields(
    *,
    passed: bool,
    repair_target_boundary_indices: list[int],
    weighted_score: float,
) -> dict[str, object]:
    penalty = 0.0 if passed else 12.0
    return {
        "perceptual_source_admission": {
            "analysis_basis": PERCEPTUAL_SOURCE_ADMISSION_ANALYSIS_BASIS,
            "index_domains": {
                "source_index": "render_plan_sources_array_index",
                "boundary_index": "render_plan_transition_index",
            },
            "raw_frames_persisted": False,
            "original_pixels_serialized": False,
            "admission": {
                "passed": passed,
                "reason_codes": (
                    []
                    if passed
                    else ["perceptual_rapid_quality_alternation"]
                ),
                "repair_target_boundary_indices": (
                    repair_target_boundary_indices
                ),
                "selection_penalty_points": penalty,
            },
        },
        "perceptual_source_admission_penalty_points": penalty,
        "weighted_score_before_perceptual_source_admission": (
            weighted_score + penalty
        ),
        "weighted_score": weighted_score,
    }


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
            # This fixture isolates source-substitution repair.  The separate
            # semi-Markov tests exercise the full-song montage gate and score.
            montage_grammar_score_weight=0.0,
            enforce_montage_grammar=False,
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
        end_shot = start_shot + 1
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
        base_window_sources = tuple(
            shot.segment.source_index
            for shot in base.shots[start_shot:end_shot]
        )
        self.assertTrue(
            any(
                tuple(
                    shot.segment.source_index
                    for shot in proposal.replacement_shots
                )
                != base_window_sources
                for proposal in proposals
            ),
            redecoder.audit_payload(),
        )
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
    def test_perceptual_admission_is_a_fail_closed_decoded_gate(
        self,
        _readiness,
    ) -> None:
        duration = 8
        source_count = 4
        decoded = decode_stage_edit_semimarkov_beam(
            checkpoint_directory=Path("/tmp/stageedit-production"),
            sources=_sources(source_count, float(duration)),
            source_metadata=_metadata(source_count, float(duration)),
            head_frames=_frames(duration, source_count),
            section_plan=_plan(duration),
            output=Path("/tmp/stageedit-perceptual-gate.mp4"),
            config=StageEditSemiMarkovConfig(
                beam_width=16,
                timeline_count=8,
                minimum_unique_sources=3,
                maximum_source_occupancy_fraction=0.75,
                enforce_montage_grammar=False,
            ),
            fps=30.0,
        )
        edit = decoded.timelines[0].edit
        target = next(
            index
            for index in range(len(edit.boundaries))
            if edit.shots[index].segment.source_index
            != edit.shots[index + 1].segment.source_index
        )
        eligible_indices = (1, 3, 5, 6)
        request_sources = [
            Path(f"/tmp/stageedit-ineligible-{index}.mp4")
            for index in range(7)
        ]
        for eligible_index, source in zip(
            eligible_indices,
            edit.sources,
            strict=True,
        ):
            request_sources[eligible_index] = source
        global_review_plan = RenderPlan(
            sources=tuple(request_sources),
            segments=tuple(
                replace(
                    shot.segment,
                    source_index=eligible_indices[
                        shot.segment.source_index
                    ],
                )
                for shot in edit.shots
            ),
            reference_source=eligible_indices[edit.reference_source],
            aspect_ratio=edit.aspect_ratio,
            output=edit.output,
            master_audio_start=edit.master_audio_start,
            width=edit.width,
            height=edit.height,
            fps=edit.fps,
            transitions=tuple(
                boundary.transition for boundary in edit.boundaries
            ),
        )
        failing_payload = {
            "passed": False,
            "partial_reedit_boundary_issue_indices": [],
            **_perceptual_review_fields(
                passed=False,
                repair_target_boundary_indices=[target],
                weighted_score=88.0,
            ),
        }
        failing_review = _stage_edit_proxy_repair_review(
            edit,
            failing_payload,
            require_perceptual_source_admission=True,
        )
        self.assertEqual(88.0, failing_review.decoded_objective)
        self.assertEqual((), failing_review.technical_errors)
        self.assertEqual(
            101.0,
            failing_review.audit["perceptual_source_admission"][
                "decoded_pass_bonus_points"
            ],
        )
        self.assertEqual(
            [target],
            failing_review.audit["perceptual_source_admission"][
                "used_boundary_indices"
            ],
        )
        _, mapped_audit = _stage_edit_perceptual_repair_targets(
            edit,
            failing_payload,
            required=True,
            review_render_plan=global_review_plan,
        )
        mapped_pair = mapped_audit["boundary_source_pairs"][0]
        self.assertEqual(
            eligible_indices[
                edit.shots[target].segment.source_index
            ],
            mapped_pair["outgoing_render_plan_source_index"],
        )
        self.assertEqual(
            edit.shots[target].segment.source_index,
            mapped_pair["outgoing_stage_edit_source_index"],
        )
        self.assertEqual(
            "render_plan_sources_array_index",
            mapped_pair["render_plan_source_index_domain"],
        )
        self.assertEqual(
            "eligible_source_index",
            mapped_pair["stage_edit_source_index_domain"],
        )
        self.assertNotEqual(
            mapped_pair["outgoing_render_plan_source_index"],
            mapped_pair["outgoing_stage_edit_source_index"],
        )
        crop_only_shots = list(edit.shots)
        crop_only_shots[target] = replace(
            crop_only_shots[target],
            segment=replace(
                crop_only_shots[target].segment,
                crop=NormalizedCrop(0.0, 0.0, 1.0, 1.0),
            ),
        )
        crop_only_edit = replace(edit, shots=tuple(crop_only_shots))
        timing_or_crop_only = (
            _stage_edit_targeted_source_substitution_evidence(
                edit,
                crop_only_edit,
                (target,),
            )
        )
        self.assertTrue(timing_or_crop_only["required"])
        self.assertFalse(timing_or_crop_only["satisfied"])
        self.assertFalse(
            timing_or_crop_only["global_source_sequence_changed"]
        )
        self.assertFalse(
            timing_or_crop_only[
                "timing_or_crop_only_change_counts_as_substitution"
            ]
        )
        mismatched_segments = list(global_review_plan.segments)
        mismatched_segments[target] = replace(
            mismatched_segments[target],
            source_index=0,
        )
        mismatched_plan = replace(
            global_review_plan,
            segments=tuple(mismatched_segments),
        )
        with self.assertRaisesRegex(ValueError, "media identity"):
            _stage_edit_perceptual_repair_targets(
                edit,
                failing_payload,
                required=True,
                review_render_plan=mismatched_plan,
            )
        self.assertEqual(
            [
                "decoded-proxy-perceptual-source-admission-"
                f"boundary-{target}"
            ],
            [issue.issue_id for issue in failing_review.issue_windows],
        )

        passing_payload = {
            "passed": True,
            "partial_reedit_boundary_issue_indices": [],
            **_perceptual_review_fields(
                passed=True,
                repair_target_boundary_indices=[],
                weighted_score=1.0,
            ),
        }
        passing_review = _stage_edit_proxy_repair_review(
            edit,
            passing_payload,
            require_perceptual_source_admission=True,
        )
        self.assertEqual(102.0, passing_review.decoded_objective)
        self.assertGreater(
            passing_review.decoded_objective,
            failing_review.decoded_objective,
        )

        lower_failing_review = _stage_edit_proxy_repair_review(
            edit,
            {
                "passed": False,
                "partial_reedit_boundary_issue_indices": [],
                **_perceptual_review_fields(
                    passed=False,
                    repair_target_boundary_indices=[target],
                    weighted_score=10.0,
                ),
            },
            require_perceptual_source_admission=True,
        )
        higher_failing_review = _stage_edit_proxy_repair_review(
            edit,
            {
                "passed": False,
                "partial_reedit_boundary_issue_indices": [],
                **_perceptual_review_fields(
                    passed=False,
                    repair_target_boundary_indices=[target],
                    weighted_score=20.0,
                ),
            },
            require_perceptual_source_admission=True,
        )
        self.assertEqual(10.0, lower_failing_review.decoded_objective)
        self.assertEqual(20.0, higher_failing_review.decoded_objective)
        self.assertGreater(
            higher_failing_review.decoded_objective,
            lower_failing_review.decoded_objective,
        )

        malformed = {
            **failing_payload,
            "perceptual_source_admission": {
                **failing_payload["perceptual_source_admission"],
                "admission": {
                    **failing_payload["perceptual_source_admission"][
                        "admission"
                    ],
                    "selection_penalty_points": float("nan"),
                },
            },
            "perceptual_source_admission_penalty_points": float("nan"),
        }
        with self.assertRaisesRegex(ValueError, "finite number"):
            _stage_edit_perceptual_repair_targets(edit, malformed)
        malformed_indices = {
            **failing_payload,
            "perceptual_source_admission": {
                **failing_payload["perceptual_source_admission"],
                "admission": {
                    **failing_payload["perceptual_source_admission"][
                        "admission"
                    ],
                    "repair_target_boundary_indices": [True],
                },
            },
        }
        with self.assertRaisesRegex(ValueError, "sorted unique in-range"):
            _stage_edit_perceptual_repair_targets(
                edit,
                malformed_indices,
            )
        malformed_empty_targets = {
            **failing_payload,
            "perceptual_source_admission": {
                **failing_payload["perceptual_source_admission"],
                "admission": {
                    **failing_payload["perceptual_source_admission"][
                        "admission"
                    ],
                    "repair_target_boundary_indices": [],
                },
            },
        }
        with self.assertRaisesRegex(ValueError, "at least one repair target"):
            _stage_edit_perceptual_repair_targets(
                edit,
                malformed_empty_targets,
            )
        with self.assertRaisesRegex(ValueError, "missing required scalar payload"):
            _stage_edit_proxy_repair_review(
                edit,
                {
                    "passed": False,
                    "weighted_score": 99.0,
                    "partial_reedit_boundary_issue_indices": [],
                },
                require_perceptual_source_admission=True,
            )
        legacy_timing_only = _stage_edit_proxy_repair_review(
            edit,
            {
                "passed": False,
                "weighted_score": 55.0,
                "partial_reedit_boundary_issue_indices": [target],
            },
        )
        self.assertFalse(
            legacy_timing_only.audit["perceptual_source_admission"][
                "available"
            ]
        )
        legacy_substitution = (
            _stage_edit_targeted_source_substitution_evidence(
                edit,
                edit,
                (),
            )
        )
        self.assertFalse(legacy_substitution["required"])
        self.assertTrue(legacy_substitution["satisfied"])

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
        eligible_indices = (1, 3, 5, 6)
        request_sources = [
            Path(f"/tmp/stageedit-ineligible-{index}.mp4")
            for index in range(7)
        ]
        for eligible_index, source in zip(
            eligible_indices,
            sources,
            strict=True,
        ):
            request_sources[eligible_index] = source
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
            enforce_montage_grammar=False,
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
        perceptual_boundary_index = next(
            (
                index
                for index in range(len(base.boundaries))
                if index != boundary_index
                and base.shots[index].segment.source_index
                != base.shots[index + 1].segment.source_index
            ),
            boundary_index,
        )
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
        candidate_edits_by_render_plan_id = {}

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
            global_render_plan = RenderPlan(
                sources=tuple(request_sources),
                segments=tuple(
                    replace(
                        shot.segment,
                        source_index=eligible_indices[
                            shot.segment.source_index
                        ],
                    )
                    for shot in edit.shots
                ),
                reference_source=eligible_indices[edit.reference_source],
                aspect_ratio=edit.aspect_ratio,
                output=edit.output,
                master_audio_start=edit.master_audio_start,
                width=edit.width,
                height=edit.height,
                fps=edit.fps,
                transitions=tuple(
                    boundary.transition for boundary in edit.boundaries
                ),
            )
            candidate_edits_by_render_plan_id[id(global_render_plan)] = edit
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
                render_plan=global_render_plan,
                candidate_origin="explicit_stage_edit_generation",
                candidate_lineage=(
                    {
                        "stage": "initial_exact8_proxy_selection",
                        "trial_lineage_count": len(edit.lineage),
                        "exact_music_structure_shot_clock": {
                            "contract": (
                                "serialized_inference_token_section_and_"
                                "phrase_progress_per_selected_shot_v1"
                            ),
                            "source_index_domain": "eligible_source_index",
                            "shots": [
                                {
                                    "shot_index": shot.shot_index,
                                    "source_index": shot.segment.source_index,
                                    "section_role": "chorus_like",
                                    "section_progress": (
                                        shot.shot_index
                                        / max(1, len(edit.shots) - 1)
                                    ),
                                    "phrase_progress": (
                                        shot.shot_index % 2
                                    ) / 2.0,
                                }
                                for shot in edit.shots
                            ],
                        },
                    },
                ),
                stage_edit_exact_shot_clock=(
                    _stage_edit_exact_shot_clock_from_edit(edit)
                ),
            )

        selected_materialized = materialized_for(base_retained)
        selected_bundle = bundle_for(selected_materialized)
        original_review = {
            "passed": False,
            "partial_reedit_boundary_issue_indices": [boundary_index],
            **_perceptual_review_fields(
                passed=False,
                repair_target_boundary_indices=[
                    perceptual_boundary_index
                ],
                weighted_score=10.0,
            ),
        }

        def repair_objectives(edit, payload, **kwargs):
            value = (
                float(payload["weighted_score"])
                if payload is not None
                else -1_000_000.0
            )
            actual = _stage_edit_proxy_repair_review(
                edit,
                payload,
                **kwargs,
            )
            return replace(actual, cadence_objective=value)

        def proxy_selection(*args, **_kwargs):
            rendered_candidate = args[0][0]
            candidate_bundle = rendered_candidate.payload
            lineage_count = int(
                candidate_bundle.candidate_lineage[0][
                    "trial_lineage_count"
                ]
            )
            proposal_plan = rendered_candidate.render_plan
            proposal_edit = candidate_edits_by_render_plan_id[
                id(proposal_plan)
            ]
            targeted_substitution = (
                _stage_edit_targeted_source_substitution_evidence(
                    base,
                    proposal_edit,
                    (perceptual_boundary_index,),
                )["satisfied"]
            )
            remaining_target = next(
                index
                for index in range(len(proposal_plan.transitions))
                if proposal_plan.segments[index].source_index
                != proposal_plan.segments[index + 1].source_index
            )
            admission_passed = bool(
                targeted_substitution and lineage_count >= 2
            )
            weighted_score = (
                30.0
                if admission_passed
                else 20.0
                if targeted_substitution
                else 10.0
            )
            return SimpleNamespace(
                selected=SimpleNamespace(
                    review_payload={
                        "passed": admission_passed,
                        "proposal_proxy": True,
                        "partial_reedit_boundary_issue_indices": [],
                        **_perceptual_review_fields(
                            passed=admission_passed,
                            repair_target_boundary_indices=(
                                []
                                if admission_passed
                                else [remaining_target]
                            ),
                            weighted_score=weighted_score,
                        ),
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
            sources=tuple(request_sources),
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
        actual_window_call = StageEditWindowedBeamRedecoder.__call__

        def one_fresh_targeted_beam_proposal_per_iteration(
            redecoder,
            repair_request,
        ):
            proposals = actual_window_call(redecoder, repair_request)
            current_source_path = tuple(
                shot.segment.source_index
                for shot in repair_request.base_edit.shots
            )
            for proposal in proposals:
                trial = repair_request.base_edit.replace_window(
                    start_shot=repair_request.window.start_shot,
                    end_shot_exclusive=(
                        repair_request.window.end_shot_exclusive
                    ),
                    replacement_shots=proposal.replacement_shots,
                    replacement_boundaries=proposal.replacement_boundaries,
                    reason="test staged actual beam proposal",
                    score_before=None,
                    score_after=None,
                )
                trial_source_path = tuple(
                    shot.segment.source_index for shot in trial.shots
                )
                if (
                    trial_source_path != current_source_path
                    and _stage_edit_targeted_source_substitution_evidence(
                        base,
                        trial,
                        (perceptual_boundary_index,),
                    )["satisfied"]
                ):
                    return (proposal,)
            return ()

        with tempfile.TemporaryDirectory() as raw_scratch:
            with (
                patch.object(
                    StageEditWindowedBeamRedecoder,
                    "__call__",
                    new=one_fresh_targeted_beam_proposal_per_iteration,
                ),
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
                    eligible_indices=eligible_indices,
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

        self.assertEqual(2, actual_decoder.call_count)
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
        self.assertEqual(
            [perceptual_boundary_index],
            result.audit_payload["perceptual_repair_boundary_indices"],
        )
        self.assertEqual(
            (
                "perceptual_source_admission.admission."
                "repair_target_boundary_indices"
            ),
            result.audit_payload["perceptual_source_admission"][
                "index_source_field"
            ],
        )
        self.assertEqual(
            [boundary_index],
            result.audit_payload[
                "deterministic_visible_boundary_issue_indices"
            ],
        )
        reviewer_issue_ids = {
            item["issue_id"]
            for item in result.audit_payload["reviewer_issue_windows"]
        }
        self.assertIn("learned-local-beam-selection", reviewer_issue_ids)
        self.assertIn(
            f"decoded-proxy-initial-boundary-{boundary_index}",
            reviewer_issue_ids,
        )
        self.assertIn(
            (
                "decoded-proxy-perceptual-source-admission-initial-"
                f"boundary-{perceptual_boundary_index}"
            ),
            reviewer_issue_ids,
        )
        self.assertTrue(result.audit_payload["changed"])
        self.assertGreater(
            result.audit_payload["repair_loop"]["committed_iteration_count"],
            1,
            result.audit_payload["repair_loop"],
        )
        repair_iterations = result.audit_payload["repair_loop"]["iterations"]
        self.assertEqual(2, len(repair_iterations))
        first_committed = next(
            item
            for item in repair_iterations[0]["proposal_audits"]
            if item["committed"]
        )
        second_committed = next(
            item
            for item in repair_iterations[1]["proposal_audits"]
            if item["committed"]
        )
        self.assertFalse(
            first_committed["review"]["audit"][
                "perceptual_source_admission"
            ]["admission_passed"]
        )
        self.assertEqual(
            20.0,
            first_committed["review"]["decoded_objective"],
        )
        self.assertTrue(
            second_committed["review"]["audit"][
                "perceptual_source_admission"
            ]["admission_passed"]
        )
        self.assertEqual(
            131.0,
            second_committed["review"]["decoded_objective"],
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
            2,
            result.audit_payload["windowed_beam_redecode"]["call_count"],
        )
        self.assertTrue(result.selected_review_payload["proposal_proxy"])
        self.assertTrue(result.selected_review_payload["passed"])
        self.assertIsNot(result.selected_bundle, selected_bundle)
        selected_evidence = result.audit_payload["selected_redecode_evidence"]
        self.assertTrue(selected_evidence["actual_semimarkov_redecode"])
        self.assertTrue(selected_evidence["new_proxy_passed"])
        self.assertTrue(
            selected_evidence["source_sequence_changed_by_redecode"]
        )
        self.assertTrue(
            selected_evidence[
                "perceptual_targeted_source_substitution_required"
            ]
        )
        self.assertTrue(
            selected_evidence[
                "perceptual_targeted_source_substitution_satisfied"
            ]
        )
        self.assertIn(
            perceptual_boundary_index,
            selected_evidence[
                "perceptual_substituted_target_boundary_indices"
            ],
        )
        self.assertTrue(
            selected_evidence[
                "new_proxy_perceptual_source_admission_passed"
            ]
        )
        self.assertEqual(
            30.0,
            selected_evidence["new_proxy_penalized_weighted_score"],
        )
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
        self.assertEqual(
            1,
            sum(
                "exact_music_structure_shot_clock" in item
                for item in result.selected_bundle.candidate_lineage
            ),
        )
        self.assertTrue(
            any(
                "superseded_exact_music_structure_shot_clock" in item
                for item in result.selected_bundle.candidate_lineage
            )
        )
        final_clock = result.selected_bundle.stage_edit_exact_shot_clock
        self.assertIsNotNone(final_clock)
        self.assertEqual(
            [
                eligible_indices.index(segment.source_index)
                for segment in result.selected_bundle.render_plan.segments
            ],
            [item["source_index"] for item in final_clock["shots"]],
        )
        self.assertNotEqual(
            selected_bundle.stage_edit_exact_shot_clock,
            final_clock,
        )
        montage_analysis = _stage_edit_montage_analysis_payload(
            bundle=result.selected_bundle,
            proxy_review_payload={"passed": True},
            section_plan=_plan(duration),
            source_derivative_diversity=SimpleNamespace(
                to_dict=lambda: {
                    "source_count": source_count,
                    "family_ids": list(range(source_count)),
                }
            ),
            source_index_mapping={"bound": True},
        )
        exact_metrics = _exact_stage_edit_anchor_metrics(
            montage_analysis,
            choices=montage_analysis["visual_timeline"]["choices"],
            sections=montage_analysis["section_plan"]["sections"],
        )
        self.assertEqual(
            len(result.selected_bundle.visual_timeline.choices),
            exact_metrics["shot_clock_count"],
        )


if __name__ == "__main__":
    unittest.main()
