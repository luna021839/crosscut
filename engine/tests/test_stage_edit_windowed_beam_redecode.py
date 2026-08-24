from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from crosscut.group_transition import GroupTransitionPlan
from crosscut.pipeline import (
    _GroupRenderCandidateBundle,
    _attempt_stage_edit_windowed_beam_reedit,
    _committed_repair_dynamic_crop_version,
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
from crosscut.stage_edit_feature_materializer import StageEditFeatureSnapshot
from crosscut.stage_edit_head_materializer import (
    StageEditHeadMaterializationResult,
)
from crosscut.stage_edit_pipeline_bridge import (
    STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT,
    RetainedStageEditCandidate,
    StageEditPipelineBridgePreflight,
    StageEditPipelineBridgeResult,
)
from crosscut.stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
)
from crosscut.stage_edit_repair_loop import (
    StageEditRepairExclusions,
    StageEditRepairRequest,
    StageEditRepairReview,
    StageEditRepairWindow,
)
from crosscut.stage_edit_semimarkov_beam import (
    HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
    HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
    StageEditSemiMarkovConfig,
    StageEditSemiMarkovResult,
    StageEditSourceContinuity,
    decode_stage_edit_semimarkov_beam,
)
from crosscut.stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
    StageEditSourcePairScorer,
    StageEditSourcePairTrainingResult,
    new_stage_edit_source_pair_checkpoint,
    save_stage_edit_source_pair_checkpoint,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditReadinessReport,
    StageEditSequence,
    StageEditToken,
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
    _single_feature_scorer,
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


def _splice_continuity_observation(high: bool) -> StageEditSourceContinuity:
    scale = 0.80 if high else 0.20
    center = 0.90 if high else 0.10
    value = 1.0 if high else 0.0
    return StageEditSourceContinuity(
        face_scale=scale,
        face_center_x=center,
        face_center_y=center,
        body_center_x=center,
        body_center_y=center,
        camera_motion_x=value,
        camera_motion_y=value,
        pose_motion=value,
        camera_zoom_velocity=value,
        camera_roll_normalized=value,
        exposure=value,
        color_temperature_normalized=value,
        sharpness=value,
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


def _typed_snapshot(
    source_ids: tuple[str, ...],
    master_times_seconds: tuple[float, ...] = (0.0,),
) -> StageEditFeatureSnapshot:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    candidates = tuple(
        SourceCandidateToken(
            source_id=source_id,
            features=(0.1 + index * 0.01,) * schema.raw_source_dimension,
            missing_mask=(False,) * schema.raw_source_dimension,
            available=True,
            hard_gate_passed=True,
        )
        for index, source_id in enumerate(source_ids)
    )
    sequence = StageEditSequence(
        "ive:windowed-redecode",
        tuple(
            StageEditToken(
                master_time_seconds=master_time_seconds,
                music_features=(0.1,) * len(schema.music_feature_names),
                section_features=(0.2,) * len(schema.section_feature_names),
                source_candidates=candidates,
                recent_edit_features=(0.0,)
                * len(schema.recent_edit_feature_names),
                style_features=(0.0,) * len(schema.style_feature_names),
                missing_mask=(False,) * schema.raw_context_dimension,
            )
            for master_time_seconds in master_times_seconds
        ),
    )
    return StageEditFeatureSnapshot.build(sequence)


def _typed_readiness(
    checkpoint: Path,
    dynamic_crop_version: int,
) -> StageEditReadinessReport:
    return StageEditReadinessReport(
        architecture_available=True,
        learned_inference_ready=True,
        status="production_checkpoint_ready",
        reason="typed windowed-redecode fixture",
        mlx_version="test",
        checkpoint_path=str(checkpoint.resolve()),
        training_steps=100,
        checkpoint_format=f"dynamic_crop_v{dynamic_crop_version}",
        dynamic_crop_version=dynamic_crop_version,
        checkpoint_metadata_filename=(
            f"dynamic-crop-v{dynamic_crop_version}-metadata.json"
        ),
    )


def _saved_source_pair_evidence(
    directory: Path,
    scorer: StageEditSourcePairScorer,
) -> dict[str, object]:
    corpus_sha256 = "c" * 64
    training = StageEditSourcePairTrainingResult(
        scorer=scorer,
        training_steps=1,
        reference_count=1,
        group_count=1,
        initial_loss=0.0,
        final_loss=0.0,
        margin=0.0,
        corpus_sha256=corpus_sha256,
    )
    checkpoint = new_stage_edit_source_pair_checkpoint(
        "windowed-redecode-source-pair",
        training,
        created_at="2026-08-24T12:00:00+09:00",
    )
    path = save_stage_edit_source_pair_checkpoint(directory, checkpoint)
    return {
        "source_pair_scorer": scorer,
        "source_pair_scorer_path": path,
        "source_pair_scorer_file_sha256": hashlib.sha256(
            path.read_bytes()
        ).hexdigest(),
        "source_pair_scorer_fingerprint": scorer.fingerprint,
        "source_pair_checkpoint_sha256": checkpoint.checkpoint_sha256,
        "source_pair_corpus_sha256": corpus_sha256,
    }


def _typed_bridge_result(
    *,
    source_ids: tuple[str, ...],
    checkpoint_directory: Path,
    frames: tuple[object, ...],
    semimarkov: StageEditSemiMarkovResult,
    retained_candidates: tuple[RetainedStageEditCandidate, ...],
    dynamic_crop_version: int = 2,
    continuity_authority: str = HANDCRAFTED_CONTINUITY_AUTHORITY,
    source_pair_evidence: dict[str, object] | None = None,
) -> StageEditPipelineBridgeResult:
    readiness = _typed_readiness(
        checkpoint_directory,
        dynamic_crop_version,
    )
    pair_kwargs = source_pair_evidence or {}
    with patch(
        "crosscut.stage_edit_pipeline_bridge.stage_edit_readiness_report",
        return_value=readiness,
    ):
        preflight = StageEditPipelineBridgePreflight(
            snapshot_path=Path("/tmp/stageedit-windowed-snapshot.json"),
            checkpoint_directory=checkpoint_directory.resolve(),
            source_ids=source_ids,
            snapshot=_typed_snapshot(
                source_ids,
                tuple(frame.master_time_seconds for frame in frames),
            ),
            readiness=readiness,
            dynamic_crop_version=dynamic_crop_version,
            continuity_authority=continuity_authority,
            **pair_kwargs,
        )
        retained = tuple(
            replace(
                item,
                edit=replace(
                    item.edit,
                    candidate_history=(
                        *item.edit.candidate_history,
                        {
                            "bridge": STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT,
                            "dynamic_crop_version": dynamic_crop_version,
                            "continuity_authority": continuity_authority,
                        },
                    ),
                ),
            )
            for item in retained_candidates
        )
        scorer = preflight.source_pair_scorer
        digest = "a" * 64
        head_materialization = StageEditHeadMaterializationResult(
            frames=frames,
            sequence_id="ive:windowed-redecode",
            canonical_source_ids=source_ids,
            checkpoint_directory=str(checkpoint_directory.resolve()),
            checkpoint_id="typed-windowed-redecode-checkpoint",
            training_steps=readiness.training_steps,
            metadata_sha256=digest,
            weights_sha256=digest,
            supplied_sequence_sha256=digest,
            input_sequence_sha256=digest,
            output_frames_sha256=digest,
            token_schema_fingerprint=DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
            dynamic_crop_config_fingerprint=digest,
            readiness=readiness,
            decision_history_input_contract={"contract_sha256": digest},
            checkpoint_format=f"dynamic_crop_v{dynamic_crop_version}",
            dynamic_crop_version=dynamic_crop_version,
            checkpoint_metadata_filename=(
                f"dynamic-crop-v{dynamic_crop_version}-metadata.json"
            ),
        )
        return StageEditPipelineBridgeResult(
            preflight=preflight,
            head_materialization=head_materialization,
            semimarkov=semimarkov,
            retained_candidates=retained,
            selection_policy={
                "visual_baseline_replaced": False,
                "selected_for_render": False,
                "dynamic_crop_version": dynamic_crop_version,
                "continuity_authority": continuity_authority,
                "continuity_scorer_contract": (
                    scorer.contract
                    if scorer is not None
                    else HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
                ),
                "continuity_scorer_fingerprint": (
                    preflight.source_pair_scorer_fingerprint
                    if preflight.source_pair_scorer_fingerprint is not None
                    else HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
                ),
            },
        )


class StageEditWindowedBeamRedecodeTests(unittest.TestCase):
    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_repaired_lineage_is_cross_bound_to_bridge_dynamic_crop_version(
        self,
        _readiness,
    ) -> None:
        duration = 8
        source_count = 4
        sources = _sources(source_count, float(duration))
        frames = _frames(duration, source_count)
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
            output=Path("/tmp/stageedit-version-lineage.mp4"),
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
        source_ids = tuple(
            f"source-{index}" for index in range(source_count)
        )
        v3_bridge = _typed_bridge_result(
            source_ids=source_ids,
            checkpoint_directory=Path("/tmp/stageedit-production"),
            frames=frames,
            semimarkov=decoded,
            retained_candidates=retained,
            dynamic_crop_version=3,
        )
        valid_v3_lineage = tuple(
            v3_bridge.retained_candidates[0].edit.candidate_history
        )
        self.assertEqual(
            3,
            _committed_repair_dynamic_crop_version(
                valid_v3_lineage,
                v3_bridge,
            ),
        )
        for lineage, reason in (
            (({"stage": "version-was-lost"},), "omitted its model version"),
            (({"dynamic_crop_version": 2},), "differs from the bridge"),
        ):
            with self.subTest(lineage=lineage):
                with self.assertRaisesRegex(RuntimeError, reason):
                    _committed_repair_dynamic_crop_version(
                        lineage,
                        v3_bridge,
                    )

        v2_bridge = _typed_bridge_result(
            source_ids=source_ids,
            checkpoint_directory=Path("/tmp/stageedit-production"),
            frames=frames,
            semimarkov=decoded,
            retained_candidates=retained,
            dynamic_crop_version=2,
        )
        self.assertEqual(
            2,
            _committed_repair_dynamic_crop_version(
                ({"stage": "legacy-v2-lineage"},),
                v2_bridge,
            ),
        )

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
        raw_retained = tuple(
            RetainedStageEditCandidate(
                rank=timeline.rank,
                score=timeline.score,
                edit=timeline.edit,
                exact_clock_reprojected=True,
                timewarped_segment_count=0,
            )
            for timeline in decoded.timelines
        )
        bridge = _typed_bridge_result(
            source_ids=tuple(
                f"source-{index}" for index in range(source_count)
            ),
            checkpoint_directory=Path("/tmp/stageedit-production"),
            frames=frames,
            semimarkov=decoded,
            retained_candidates=raw_retained,
        )
        retained = bridge.retained_candidates
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
        self.assertEqual(
            HANDCRAFTED_CONTINUITY_AUTHORITY,
            proposal.audit["continuity_authority"],
        )
        self.assertEqual(
            HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
            proposal.audit["continuity_scorer_contract"],
        )
        self.assertEqual(
            HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
            proposal.audit["continuity_scorer_fingerprint"],
        )
        exterior = proposal.audit["exterior_splice_continuity"]
        self.assertEqual(2, exterior["expected_edge_count"])
        self.assertEqual(2, exterior["evaluated_edge_count"])
        self.assertTrue(exterior["full_edit_validation_passed"])
        self.assertEqual(
            ["left", "right"],
            [edge["side"] for edge in exterior["edges"]],
        )
        self.assertTrue(
            all(
                edge["authority_rule_evaluated"]
                and -1.0 <= edge["raw_score"] <= 0.0
                and edge["continuity_scorer_fingerprint"]
                == HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
                for edge in exterior["edges"]
            )
        )
        self.assertAlmostEqual(
            proposal.audit["candidate_selection_score"],
            proposal.audit["decoder_timeline_score"]
            + exterior["weighted_score_total"],
        )
        self.assertTrue(
            proposal.audit["exterior_score_applied_to_candidate_selection"]
        )
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
        self.assertFalse(audit["calls"][0]["learned_source_pair_scorer_bound"])
        self.assertIsNone(
            actual_decoder.call_args.kwargs["source_pair_scorer"]
        )

        scorer = StageEditSourcePairScorer.uniform()
        learned_config = replace(
            config,
            continuity_authority=PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
            continuity_scorer_contract=STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
            continuity_scorer_fingerprint=scorer.fingerprint,
        )
        pair_directory = tempfile.TemporaryDirectory()
        self.addCleanup(pair_directory.cleanup)
        pair_evidence = _saved_source_pair_evidence(
            Path(pair_directory.name),
            scorer,
        )
        learned_semimarkov = decode_stage_edit_semimarkov_beam(
            checkpoint_directory=Path("/tmp/stageedit-production"),
            sources=sources,
            source_metadata=_metadata(source_count, float(duration)),
            head_frames=frames,
            section_plan=_plan(duration),
            output=Path("/tmp/stageedit-window-base-learned.mp4"),
            config=learned_config,
            fps=30.0,
            source_pair_scorer=scorer,
        )
        learned_retained = tuple(
            RetainedStageEditCandidate(
                rank=timeline.rank,
                score=timeline.score,
                edit=timeline.edit,
                exact_clock_reprojected=True,
                timewarped_segment_count=0,
            )
            for timeline in learned_semimarkov.timelines
        )
        learned_bridge = _typed_bridge_result(
            source_ids=tuple(
                f"source-{index}" for index in range(source_count)
            ),
            checkpoint_directory=Path("/tmp/stageedit-production"),
            frames=frames,
            semimarkov=learned_semimarkov,
            retained_candidates=learned_retained,
            dynamic_crop_version=3,
            continuity_authority=PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
            source_pair_evidence=pair_evidence,
        )
        learned_redecoder = StageEditWindowedBeamRedecoder(
            bridge_result=learned_bridge,
            sources=sources,
            source_durations_seconds=(float(duration),) * source_count,
            source_start_offsets_seconds=(0.0,) * source_count,
            internal_cut_times_seconds=((),) * source_count,
            source_timewarps=(None,) * source_count,
            source_quality_log_priors=(0.0,) * source_count,
            output=Path("/tmp/stageedit-window-redecode-learned.mp4"),
        )
        with patch(
            "crosscut.stage_edit_windowed_beam_redecode."
            "decode_stage_edit_semimarkov_beam",
            wraps=decode_stage_edit_semimarkov_beam,
        ) as learned_decoder:
            learned_proposals = learned_redecoder(request)
        self.assertIs(
            learned_decoder.call_args.kwargs["source_pair_scorer"],
            learned_bridge.preflight.source_pair_scorer,
        )
        learned_audit = learned_redecoder.audit_payload()["calls"][0]
        self.assertTrue(learned_audit["learned_source_pair_scorer_bound"])
        self.assertEqual(
            learned_audit["continuity_scorer_fingerprint"], scorer.fingerprint
        )
        self.assertGreater(len(learned_proposals), 0, learned_audit)
        learned_proposal_audit = learned_proposals[0].audit
        self.assertEqual(
            PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
            learned_proposal_audit["continuity_authority"],
        )
        self.assertEqual(
            STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
            learned_proposal_audit["continuity_scorer_contract"],
        )
        self.assertEqual(
            scorer.fingerprint,
            learned_proposal_audit["continuity_scorer_fingerprint"],
        )
        learned_exterior = learned_proposal_audit[
            "exterior_splice_continuity"
        ]
        self.assertTrue(learned_exterior["full_edit_validation_passed"])
        self.assertEqual(2, learned_exterior["evaluated_edge_count"])
        self.assertTrue(
            all(
                edge["scorer_evaluated"]
                and edge["continuity_scorer_fingerprint"]
                == scorer.fingerprint
                for edge in learned_exterior["edges"]
            )
        )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_exterior_continuity_reranks_both_authorities_and_right_edge_fails_closed(
        self,
        _readiness,
    ) -> None:
        duration = 8
        source_count = 4
        sources = _sources(source_count, float(duration))
        low = _splice_continuity_observation(False)
        high = _splice_continuity_observation(True)
        learned_scorer = _single_feature_scorer("camera_motion_distance")
        cases = (
            (
                HANDCRAFTED_CONTINUITY_AUTHORITY,
                None,
                HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
                HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
            ),
            (
                PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
                learned_scorer,
                STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
                learned_scorer.fingerprint,
            ),
        )
        for authority, scorer, contract, fingerprint in cases:
            with self.subTest(authority=authority):
                config = StageEditSemiMarkovConfig(
                    beam_width=32,
                    timeline_count=8,
                    minimum_shot_seconds=1.0,
                    maximum_shot_seconds=1.0,
                    minimum_unique_sources=3,
                    maximum_source_occupancy_fraction=0.75,
                    enforce_aba_exclusion=False,
                    enforce_montage_grammar=False,
                    maximum_global_switches_per_minute=120.0,
                    section_cut_density_multiplier=10.0,
                    cut_score_weight=0.0,
                    source_score_weight=1.0,
                    duration_score_weight=0.0,
                    transition_score_weight=0.0,
                    crop_correction_weight=0.0,
                    source_quality_weight=0.0,
                    boundary_continuity_weight=1.0,
                    cadence_score_weight=0.0,
                    diversity_score_weight=0.0,
                    occupancy_balance_weight=0.0,
                    montage_grammar_score_weight=0.0,
                    continuity_authority=authority,
                    continuity_scorer_contract=contract,
                    continuity_scorer_fingerprint=fingerprint,
                )
                base_frames = _frames(duration, source_count)
                decoded = decode_stage_edit_semimarkov_beam(
                    checkpoint_directory=Path("/tmp/stageedit-production"),
                    sources=sources,
                    source_metadata=_metadata(source_count, float(duration)),
                    head_frames=base_frames,
                    section_plan=_plan(duration),
                    output=Path(
                        f"/tmp/stageedit-exterior-rerank-{authority}.mp4"
                    ),
                    config=config,
                    fps=30.0,
                    source_pair_scorer=scorer,
                )
                base = decoded.timelines[0].edit
                self.assertEqual(
                    (0, 1, 2, 3, 0, 1, 2, 3),
                    tuple(
                        shot.segment.source_index for shot in base.shots
                    ),
                )
                frames = list(base_frames)
                frames[1] = replace(
                    frames[1],
                    source_logits=(-20.0, 2.9, -20.0, 3.0),
                    source_continuity=(low, low, low, high),
                )
                frames[2] = replace(
                    frames[2],
                    source_continuity=(low, low, low, high),
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
                source_pair_evidence = None
                if scorer is not None:
                    pair_directory = tempfile.TemporaryDirectory()
                    self.addCleanup(pair_directory.cleanup)
                    source_pair_evidence = _saved_source_pair_evidence(
                        Path(pair_directory.name),
                        scorer,
                    )
                bridge = _typed_bridge_result(
                    source_ids=tuple(
                        f"source-{index}" for index in range(source_count)
                    ),
                    checkpoint_directory=Path("/tmp/stageedit-production"),
                    frames=tuple(frames),
                    semimarkov=decoded,
                    retained_candidates=retained,
                    dynamic_crop_version=(3 if scorer is not None else 2),
                    continuity_authority=authority,
                    source_pair_evidence=source_pair_evidence,
                )
                window = StageEditRepairWindow(
                    issue_ids=("exterior-rerank",),
                    causes=("learned_editorial_discontinuity",),
                    requested_start_seconds=1.0,
                    requested_end_seconds=2.0,
                    context_start_seconds=1.0,
                    context_end_seconds=2.0,
                    canonical_start_seconds=1.0,
                    canonical_end_seconds=2.0,
                    actual_start_seconds=1.0,
                    actual_end_seconds=2.0,
                    start_shot=1,
                    end_shot_exclusive=2,
                )
                request = StageEditRepairRequest(
                    iteration=1,
                    base_edit=base,
                    current_review=StageEditRepairReview(
                        50.0,
                        50.0,
                        -1.0,
                    ),
                    window=window,
                    canonical_boundaries_seconds=tuple(
                        float(index) for index in range(duration + 1)
                    ),
                    exclusions=StageEditRepairExclusions(),
                    forbidden_edit_sha256s=(),
                )

                def redecoder_for(
                    value: StageEditPipelineBridgeResult,
                ) -> StageEditWindowedBeamRedecoder:
                    return StageEditWindowedBeamRedecoder(
                        bridge_result=value,
                        sources=sources,
                        source_durations_seconds=(float(duration),)
                        * source_count,
                        source_start_offsets_seconds=(0.0,) * source_count,
                        internal_cut_times_seconds=((),) * source_count,
                        source_timewarps=(None,) * source_count,
                        source_quality_log_priors=(0.0,) * source_count,
                        output=Path(
                            f"/tmp/stageedit-exterior-window-{authority}.mp4"
                        ),
                    )

                redecoder = redecoder_for(bridge)
                proposals = redecoder(request)
                self.assertGreaterEqual(len(proposals), 2)
                self.assertEqual(
                    [1, 3],
                    [
                        proposal.replacement_shots[0].segment.source_index
                        for proposal in proposals[:2]
                    ],
                )
                self.assertEqual([1, 2], [value.rank for value in proposals[:2]])
                self.assertEqual(
                    [2, 1],
                    [
                        value.audit["decoder_timeline_rank"]
                        for value in proposals[:2]
                    ],
                )
                expected_pairs = (
                    ((0, 1), (1, 2)),
                    ((0, 3), (3, 2)),
                )
                for proposal, pairs in zip(
                    proposals[:2],
                    expected_pairs,
                    strict=True,
                ):
                    exterior = proposal.audit[
                        "exterior_splice_continuity"
                    ]
                    self.assertEqual(2, exterior["expected_edge_count"])
                    self.assertEqual(2, exterior["evaluated_edge_count"])
                    self.assertTrue(exterior["full_edit_validation_passed"])
                    self.assertEqual(
                        [("left", 1.0), ("right", 2.0)],
                        [
                            (
                                edge["side"],
                                edge["boundary_time_seconds"],
                            )
                            for edge in exterior["edges"]
                        ],
                    )
                    self.assertEqual(
                        list(pairs),
                        [
                            (
                                edge["outgoing_source_index"],
                                edge["incoming_source_index"],
                            )
                            for edge in exterior["edges"]
                        ],
                    )
                    self.assertTrue(
                        all(
                            edge["scorer_evaluated"]
                            and edge["continuity_scorer_fingerprint"]
                            == fingerprint
                            for edge in exterior["edges"]
                        )
                    )
                    self.assertAlmostEqual(
                        proposal.audit["candidate_selection_score"],
                        proposal.audit["decoder_timeline_score"]
                        + exterior["weighted_score_total"],
                    )
                self.assertAlmostEqual(
                    0.0,
                    proposals[0].audit["exterior_splice_continuity"][
                        "raw_score_total"
                    ],
                )
                self.assertAlmostEqual(
                    -2.0,
                    proposals[1].audit["exterior_splice_continuity"][
                        "raw_score_total"
                    ],
                )

                if scorer is not None:
                    broken_frames = list(frames)
                    broken_frames[2] = replace(
                        broken_frames[2],
                        source_continuity=(),
                    )
                    broken_bridge = replace(
                        bridge,
                        head_materialization=replace(
                            bridge.head_materialization,
                            frames=tuple(broken_frames),
                        ),
                    )
                    broken_redecoder = redecoder_for(broken_bridge)
                    self.assertEqual((), broken_redecoder(request))
                    self.assertTrue(
                        any(
                            "learned exterior continuity is missing exact observations"
                            in reason
                            for rejection in broken_redecoder.audit_payload()[
                                "calls"
                            ][0]["rejections"]
                            for reason in rejection["reasons"]
                        )
                    )

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
        bridge = _typed_bridge_result(
            source_ids=tuple(
                f"source-{index}" for index in range(source_count)
            ),
            checkpoint_directory=Path("/tmp/stageedit-production"),
            frames=frames,
            semimarkov=decoded,
            retained_candidates=retained,
        )
        retained = bridge.retained_candidates
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
        for committed in (first_committed, second_committed):
            decoder_audit = committed["decoder_audit"]
            self.assertEqual(
                HANDCRAFTED_CONTINUITY_AUTHORITY,
                decoder_audit["continuity_authority"],
            )
            self.assertEqual(
                HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
                decoder_audit["continuity_scorer_contract"],
            )
            self.assertEqual(
                HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
                decoder_audit["continuity_scorer_fingerprint"],
            )
            exterior = decoder_audit["exterior_splice_continuity"]
            self.assertTrue(exterior["full_edit_validation_passed"])
            self.assertEqual(
                exterior["expected_edge_count"],
                exterior["evaluated_edge_count"],
            )
            self.assertTrue(
                decoder_audit[
                    "exterior_score_applied_to_candidate_selection"
                ]
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
        self.assertEqual(
            [1], selected_evidence["dynamic_crop_suppressed_segment_indices"]
        )
        self.assertEqual(selected_evidence["dynamic_crop_version"], 2)
        self.assertTrue(
            selected_evidence["proposal_materialization_lineage_preserved"]
        )
        self.assertEqual(
            HANDCRAFTED_CONTINUITY_AUTHORITY,
            selected_evidence["continuity_authority"],
        )
        self.assertEqual(
            HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
            selected_evidence["continuity_scorer_contract"],
        )
        self.assertEqual(
            HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
            selected_evidence["continuity_scorer_fingerprint"],
        )
        self.assertEqual(
            "committed_window_semimarkov_redecode",
            selected_evidence["continuity_provenance_owner"],
        )
        self.assertFalse(
            selected_evidence[
                "initial_decode_continuity_provenance_authoritative_for_repaired_edges"
            ]
        )
        committed_exterior = selected_evidence[
            "committed_window_exterior_continuity_evaluations"
        ]
        self.assertEqual(2, len(committed_exterior))
        self.assertEqual(
            [1, 2],
            [item["iteration"] for item in committed_exterior],
        )
        self.assertTrue(
            all(
                item["exterior_splice_continuity"][
                    "full_edit_validation_passed"
                ]
                for item in committed_exterior
            )
        )
        committed_records = [first_committed, second_committed]
        for iteration_number, evaluation, committed in zip(
            (1, 2),
            committed_exterior,
            committed_records,
            strict=True,
        ):
            decoder_audit = committed["decoder_audit"]
            self.assertEqual(iteration_number, evaluation["iteration"])
            self.assertEqual(
                committed["proposal_id"],
                evaluation["proposal_id"],
            )
            self.assertEqual(
                decoder_audit["decoder_timeline_rank"],
                evaluation["decoder_timeline_rank"],
            )
            self.assertEqual(
                decoder_audit["candidate_selection_rank"],
                evaluation["candidate_selection_rank"],
            )
            self.assertEqual(
                decoder_audit["candidate_selection_score"],
                evaluation["candidate_selection_score"],
            )
            self.assertEqual(
                decoder_audit["exterior_splice_continuity"],
                evaluation["exterior_splice_continuity"],
            )
            self.assertTrue(
                all(
                    -1.0 <= edge["raw_score"] <= 0.0
                    and edge["weighted_score"]
                    == edge["raw_score"] * edge["weight"]
                    for edge in evaluation[
                        "exterior_splice_continuity"
                    ]["edges"]
                )
            )
        lineage = result.selected_bundle.candidate_lineage[-1]
        self.assertTrue(lineage["actual_public_decoder_invoked"])
        self.assertTrue(lineage["actual_semimarkov_redecode"])
        self.assertTrue(lineage["new_proxy_passed"])
        self.assertTrue(lineage["learned_reviewer_reinvoked_after_splice"])
        self.assertEqual(
            [1], lineage["dynamic_crop_v2_suppressed_segment_indices"]
        )
        self.assertEqual(
            [1], lineage["dynamic_crop_suppressed_segment_indices"]
        )
        self.assertTrue(lineage["decoded_proxy_revalidated_after_splice"])
        self.assertFalse(lineage["heuristic_timing_repair_used_as_reviewer_repair"])
        self.assertEqual(
            HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
            lineage["continuity_scorer_fingerprint"],
        )
        self.assertEqual(
            committed_exterior,
            lineage["committed_window_exterior_continuity_evaluations"],
        )
        self.assertFalse(
            lineage[
                "initial_decode_continuity_provenance_authoritative_for_repaired_edges"
            ]
        )
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
