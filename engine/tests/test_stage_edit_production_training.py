from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from crosscut.ive import AspectRatio
from crosscut.render import EditorialTransformKeyframe
from crosscut.section_director import (
    SectionDirective,
    SectionPlan,
    SectionRole,
)
from crosscut.stage_edit_dynamic_crop_head_v2 import (
    DynamicCropTrajectoryTargetV2,
    StageEditDynamicCropTargetSequenceV2,
)
from crosscut.stage_edit_dynamic_crop_training import (
    StageEditDynamicCropTrainingCorpusV2,
)
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter import (
    DYNAMIC_CROP_V2_SEMIMARKOV_ADAPTER_CONTRACT,
)
from crosscut.stage_edit_head_materializer import (
    materialize_stage_edit_head_frames_v2,
)
from crosscut.stage_edit_production_training import (
    ProductionQualityThresholds,
    STAGE_EDIT_SELECTION_FREEZE_FILENAME,
    _structured_head_coverage_gate,
    evaluate_stage_edit_production_quality,
    prepare_stage_edit_production_input_sequences,
    production_song_split,
    train_production_stage_edit,
)
from crosscut.stage_edit_reference_adapter import (
    ReferenceTrainingCorpus,
    ReferenceTrainingExample,
)
from crosscut.stage_edit_semimarkov_beam import (
    StageEditSemiMarkovConfig,
    StageEditSourceMetadata,
    decode_stage_edit_semimarkov_beam,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditSequence,
    StageEditTarget,
    StageEditTargetSequence,
    StageEditToken,
    StageEditTransformerConfig,
    mlx_runtime_version,
    stage_edit_recent_history_mask_contract,
    stage_edit_readiness_report,
)
from crosscut.stage_edit_transition_authority import (
    RENDERER_OWNED_LEARNED_EVALUATION_SKIP_REASON,
    RENDERER_OWNED_TRANSITION_AUTHORITY,
)

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None


def _passing_metrics() -> dict:
    return {
        "cut": {
            "label_count": 30,
            "positive_count": 10,
            "auroc": 0.82,
            "balanced_accuracy_at_0_5": 0.73,
            "balanced_accuracy_at_frozen_validation_threshold": 0.73,
        },
        "source": {
            "listwise_group_count": 10,
            "top1_accuracy": 0.50,
            "mean_reciprocal_rank": 0.70,
        },
        "duration": {
            "label_count": 10,
            "accuracy": 0.50,
            "observed_class_count": 3,
            "label_entropy_bits": 1.20,
            "label_histogram": {"1": 3, "2": 4, "3": 3},
        },
        "transition": {
            "label_count": 10,
            "accuracy": 0.80,
            "observed_class_count": 2,
            "label_entropy_bits": 0.97,
            "label_histogram": {"0": 6, "1": 4},
        },
        "crop": {
            "coordinate_label_count": 40,
            "coordinates": {
                "zoom": {"label_count": 10, "mean_absolute_error": 0.03},
                "center_x": {"label_count": 10, "mean_absolute_error": 0.04},
                "center_y": {"label_count": 10, "mean_absolute_error": 0.05},
                "roll_degrees": {
                    "label_count": 10,
                    "mean_absolute_error": 0.40,
                },
            },
        },
    }


def _passing_dynamic_metrics() -> dict:
    return {
        "trajectory": {
            "track_label_count": 10,
            "keyframe_label_count": 20,
            "coordinates": {
                "zoom": {"label_count": 20, "mean_absolute_error": 0.03},
                "center_x": {"label_count": 20, "mean_absolute_error": 0.03},
                "center_y": {"label_count": 20, "mean_absolute_error": 0.03},
                "roll_degrees": {
                    "label_count": 20,
                    "mean_absolute_error": 0.30,
                },
            },
            "normalized_time_mae": 0.03,
            "presence": {
                "label_count": 80,
                "f1": 0.80,
                "keyframe_count_exact_accuracy": 0.60,
            },
            "motion": {"label_count": 10, "accuracy": 0.70},
            "renderer_decode": {
                "attempt_count": 10,
                "eligible_count": 9,
                "eligible_rate": 0.90,
            },
        }
    }


class StageEditProductionQualityTests(unittest.TestCase):
    def test_renderer_owned_structured_coverage_keeps_duration_and_skips_logits(
        self,
    ) -> None:
        examples = (
            SimpleNamespace(
                targets=SimpleNamespace(
                    targets=(
                        SimpleNamespace(
                            duration_bucket=0, transition_label="hard_cut"
                        ),
                        SimpleNamespace(
                            duration_bucket=1, transition_label="hard_cut"
                        ),
                    )
                )
            ),
        )
        heldout = {
            "duration": {
                "label_count": 6,
                "observed_class_count": 2,
                "label_entropy_bits": 1.0,
                "label_histogram": {"0": 3, "1": 3},
            },
            "transition": {
                "label_count": 6,
                "observed_class_count": 1,
                "label_entropy_bits": 0.0,
                "label_histogram": {"0": 6},
                "accuracy": 0.0,
            },
        }
        gate = _structured_head_coverage_gate(
            examples,
            heldout,
            thresholds=ProductionQualityThresholds(),
            transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(
            "skipped_non_authoritative", gate["heldout"]["transition"]["status"]
        )
        self.assertEqual(
            {"hard_cut": 6}, gate["heldout"]["transition"]["label_histogram"]
        )

    def test_production_preprocessing_scrubs_human_history_snapshot_slice(self) -> None:
        schema = DEFAULT_STAGE_EDIT_SCHEMA
        example = _training_example("BANG", "BANG-human-history")
        prepared = prepare_stage_edit_production_input_sequences((example,))[0]
        recent_start = len(schema.music_feature_names) + len(
            schema.section_feature_names
        )
        recent_stop = recent_start + len(schema.recent_edit_feature_names)
        self.assertTrue(
            any(value != 0.0 for value in example.sequence.tokens[1].recent_edit_features)
        )
        for token in prepared.tokens:
            self.assertEqual(
                (0.0,) * len(schema.recent_edit_feature_names),
                token.recent_edit_features,
            )
            self.assertEqual(
                (True,) * len(schema.recent_edit_feature_names),
                token.missing_mask[recent_start:recent_stop],
            )

    def test_every_renderer_active_head_must_pass_fixed_unseen_song_gate(
        self,
    ) -> None:
        passed = evaluate_stage_edit_production_quality(
            _passing_metrics(),
            dynamic_crop_v2_metrics=_passing_dynamic_metrics(),
        )
        self.assertTrue(passed.passed)
        self.assertEqual((), passed.reasons)

        failing = _passing_metrics()
        failing["source"] = dict(failing["source"], top1_accuracy=0.1)
        failing["crop"] = dict(failing["crop"])
        failing["crop"]["coordinates"] = dict(
            failing["crop"]["coordinates"]
        )
        failing["crop"]["coordinates"]["roll_degrees"] = {
            "label_count": 0,
            "mean_absolute_error": None,
        }
        rejected = evaluate_stage_edit_production_quality(
            failing,
            dynamic_crop_v2_metrics=_passing_dynamic_metrics(),
        )
        self.assertFalse(rejected.passed)
        self.assertIn("heldout_source_top1_below_threshold", rejected.reasons)
        self.assertNotIn(
            "insufficient_heldout_crop_roll_degrees_labels", rejected.reasons
        )

        failing_dynamic = _passing_dynamic_metrics()
        failing_dynamic["trajectory"] = dict(failing_dynamic["trajectory"])
        failing_dynamic["trajectory"]["coordinates"] = dict(
            failing_dynamic["trajectory"]["coordinates"]
        )
        failing_dynamic["trajectory"]["coordinates"]["roll_degrees"] = {
            "label_count": 0,
            "mean_absolute_error": None,
        }
        dynamic_rejected = evaluate_stage_edit_production_quality(
            _passing_metrics(),
            dynamic_crop_v2_metrics=failing_dynamic,
        )
        self.assertFalse(dynamic_rejected.passed)
        self.assertIn(
            "insufficient_heldout_dynamic_crop_roll_degrees_labels",
            dynamic_rejected.reasons,
        )

        degenerate = _passing_metrics()
        degenerate["transition"] = {
            "label_count": 20,
            "accuracy": 1.0,
            "observed_class_count": 1,
            "label_entropy_bits": 0.0,
            "label_histogram": {"0": 20},
        }
        degenerate_rejected = evaluate_stage_edit_production_quality(
            degenerate,
            dynamic_crop_v2_metrics=_passing_dynamic_metrics(),
        )
        self.assertFalse(degenerate_rejected.passed)
        self.assertIn(
            "insufficient_heldout_transition_class_coverage",
            degenerate_rejected.reasons,
        )
        self.assertIn(
            "heldout_transition_label_entropy_below_threshold",
            degenerate_rejected.reasons,
        )

        renderer_owned = evaluate_stage_edit_production_quality(
            degenerate,
            dynamic_crop_v2_metrics=_passing_dynamic_metrics(),
            transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
        )
        self.assertTrue(renderer_owned.passed)
        transition_audit = renderer_owned.metrics["transition"]
        self.assertEqual("skipped_non_authoritative", transition_audit["status"])
        self.assertEqual(
            RENDERER_OWNED_LEARNED_EVALUATION_SKIP_REASON,
            transition_audit["reason"],
        )
        self.assertEqual(20, transition_audit["label_count"])
        self.assertIsNone(transition_audit["learned_accuracy"])
        self.assertFalse(transition_audit["learned_logits_consumed_for_quality"])

    def test_complete_song_split_never_leaks_heldout_references(self) -> None:
        examples = tuple(
            ReferenceTrainingExample(
                reference_id=f"{song}-ref-{index}",
                song_id=song,
                style_id="human-stage-mix",
                sequence=None,  # split contract does not inspect tensors
                targets=None,
                statistics={},
            )
            for song in ("BANG", "BLACKHOLE", "XOXZ")
            for index in range(2)
        )
        corpus = ReferenceTrainingCorpus(examples, "data", (), {})
        split = production_song_split(corpus, held_out_song_ids=("XOXZ",))
        self.assertEqual(("BANG", "BLACKHOLE"), split.training_song_ids)
        self.assertEqual(("XOXZ",), split.held_out_song_ids)
        self.assertTrue(all(item.song_id != "XOXZ" for item in split.train))
        self.assertTrue(all(item.song_id != "XOXZ" for item in split.validation))
        self.assertTrue(all(item.song_id == "XOXZ" for item in split.held_out))
        self.assertEqual(2, len(split.validation))
        self.assertEqual(
            {"BANG", "BLACKHOLE"},
            {str(item.song_id) for item in split.validation},
        )
        partition_ids = [
            item.reference_id
            for partition in (split.train, split.validation, split.held_out)
            for item in partition
        ]
        self.assertEqual(len(partition_ids), len(set(partition_ids)))
        self.assertFalse(split.to_dict()["held_out_used_for_model_selection"])
        required = production_song_split(
            corpus,
            held_out_song_ids=("XOXZ",),
            required_training_song_ids=("BANG",),
        )
        self.assertEqual(("BANG",), required.required_training_song_ids)
        with self.assertRaisesRegex(ValueError, "required training songs"):
            production_song_split(
                corpus,
                held_out_song_ids=("XOXZ",),
                required_training_song_ids=("REBEL-HEART",),
            )


def _candidate(source_id: str, value: float) -> SourceCandidateToken:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    return SourceCandidateToken(
        source_id,
        tuple([value] * schema.raw_source_dimension),
        tuple([False] * schema.raw_source_dimension),
    )


def _training_example(song: str, reference: str) -> ReferenceTrainingExample:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    tokens = tuple(
        StageEditToken(
            master_time_seconds=index * 0.5,
            music_features=tuple([0.1 + 0.1 * index] * len(schema.music_feature_names)),
            section_features=tuple([0.2] * len(schema.section_feature_names)),
            source_candidates=(_candidate("A", 0.2), _candidate("B", 0.8)),
            recent_edit_features=tuple(
                [0.1 * index] * len(schema.recent_edit_feature_names)
            ),
            style_features=tuple([0.4] * len(schema.style_feature_names)),
            missing_mask=tuple([False] * schema.raw_context_dimension),
        )
        for index in range(4)
    )
    targets = (
        StageEditTarget(cut=0.0),
        StageEditTarget(
            cut=1.0,
            selected_source_id="B",
            rejected_source_ids=("A",),
            duration_bucket=2,
            transition_label="hard_cut",
            crop=(1.05, 0.50, 0.48, 0.2),
        ),
        StageEditTarget(cut=0.0),
        StageEditTarget(
            cut=1.0,
            selected_source_id="A",
            rejected_source_ids=("B",),
            duration_bucket=1,
            transition_label="zoom_match",
            crop=(1.08, 0.48, 0.50, -0.2),
        ),
    )
    return ReferenceTrainingExample(
        reference_id=reference,
        song_id=song,
        style_id="human-stage-mix",
        sequence=StageEditSequence(reference, tokens),
        targets=StageEditTargetSequence(reference, targets),
        statistics={
            "dense_editorial_source_switch_positive_count": 2,
            "editorial_source_switch_positive_count": 2,
            "cut_positive_count": 2,
            "native_source_camera_cut_count": 0,
            "native_source_camera_cut_editorial_positive_count": 0,
            "materialized_feature_snapshot_used": True,
            "shared_feature_materializer_used": True,
        },
        feature_snapshot_sha256=hashlib.sha256(reference.encode()).hexdigest(),
    )


def _dynamic_crop_corpus(
    examples: tuple[ReferenceTrainingExample, ...],
) -> StageEditDynamicCropTrainingCorpusV2:
    sequences = {}
    for example in examples:
        targets = [None] * len(example.sequence.tokens)
        for token_index, source_id in ((1, "B"), (3, "A")):
            targets[token_index] = DynamicCropTrajectoryTargetV2(
                shot_index=token_index,
                source_id=source_id,
                shot_duration_seconds=1.0,
                keyframes=(
                    EditorialTransformKeyframe(0.0, 1.04, 0.50, 0.50, 0.0),
                    EditorialTransformKeyframe(0.5, 1.06, 0.51, 0.50, 0.0),
                ),
                motion_label="dynamic",
                motion_label_loss_masked=False,
                artifact_content_sha256="a" * 64,
                build_fingerprint_sha256="b" * 64,
            )
        sequences[example.sequence.sequence_id] = (
            StageEditDynamicCropTargetSequenceV2(
                example.sequence.sequence_id, tuple(targets)
            )
        )
    return StageEditDynamicCropTrainingCorpusV2(
        sequences=sequences,
        data_id="dynamic-test",
        artifact_paths=(),
        statistics={
            "track_label_count": 2 * len(examples),
            "keyframe_label_count": 4 * len(examples),
            "motion_label_count": 2 * len(examples),
        },
    )


@unittest.skipUnless(mlx_runtime_version() is not None, "MLX is optional")
class StageEditProductionTrainingSmokeTests(unittest.TestCase):
    def test_exact_saved_checkpoint_needs_song_gate_and_explicit_authorization(self) -> None:
        schema = DEFAULT_STAGE_EDIT_SCHEMA
        examples = tuple(
            _training_example(song, f"{song}-ref-{index}")
            for song in ("BANG", "BLACKHOLE", "XOXZ")
            for index in range(2)
        )
        corpus = ReferenceTrainingCorpus(
            examples=examples,
            data_id="a" * 64,
            source_paths=(),
            statistics={
                "crop_coordinate_supervision_count": 24,
                "transition_supervision_count": 6,
            },
        )
        config = StageEditTransformerConfig.from_schema(
            schema,
            model_dimension=16,
            num_heads=4,
            num_layers=1,
            feed_forward_dimension=32,
            max_sequence_length=8,
            max_sources=2,
            dropout=0.0,
        )
        thresholds = ProductionQualityThresholds(
            minimum_cut_labels=2,
            minimum_cut_positives=1,
            minimum_cut_negatives=1,
            minimum_cut_auroc=0.0,
            minimum_cut_balanced_accuracy=0.0,
            minimum_source_groups=1,
            minimum_source_top1_accuracy=0.0,
            minimum_source_mrr=0.0,
            minimum_duration_labels=1,
            minimum_duration_accuracy=0.0,
            minimum_transition_labels=1,
            minimum_transition_accuracy=0.0,
            minimum_crop_labels_per_coordinate=1,
            maximum_zoom_mae=100.0,
            maximum_center_mae=100.0,
            maximum_roll_mae_degrees=100.0,
            minimum_dynamic_crop_tracks=1,
            minimum_dynamic_crop_keyframes=1,
            minimum_dynamic_crop_motion_labels=1,
            maximum_dynamic_crop_zoom_mae=100.0,
            maximum_dynamic_crop_center_mae=100.0,
            maximum_dynamic_crop_roll_mae_degrees=100.0,
            maximum_dynamic_crop_normalized_time_mae=100.0,
            minimum_dynamic_crop_presence_f1=0.0,
            minimum_dynamic_crop_keyframe_count_accuracy=0.0,
            minimum_dynamic_crop_motion_accuracy=0.0,
            minimum_dynamic_crop_renderer_decode_rate=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            result = train_production_stage_edit(
                corpus,
                checkpoint_directory=checkpoint,
                held_out_song_ids=("XOXZ",),
                model_config=config,
                epochs=1,
                early_stopping_patience=1,
                batch_size=1,
                learning_rate=1.0e-3,
                cut_pretrain_steps=0,
                seed=19,
                device=mx.cpu,
                dynamic_crop_corpus=_dynamic_crop_corpus(examples),
                quality_thresholds=thresholds,
                authorize_production_activation=True,
            )
            self.assertTrue(result.production_ready)
            self.assertTrue(result.quality_gate.passed)
            self.assertTrue(result.report_path.is_file())
            selection_freeze_path = (
                checkpoint / STAGE_EDIT_SELECTION_FREEZE_FILENAME
            )
            self.assertTrue(selection_freeze_path.is_file())
            selection_freeze = json.loads(
                selection_freeze_path.read_text(encoding="utf-8")
            )
            self.assertFalse(selection_freeze["heldout_metrics_present"])
            self.assertFalse(
                selection_freeze[
                    "heldout_used_for_model_or_threshold_selection"
                ]
            )
            self.assertFalse(
                selection_freeze[
                    "style_or_attitude_configuration_mutable_after_freeze"
                ]
            )
            readiness = stage_edit_readiness_report(checkpoint)
            self.assertTrue(readiness.learned_inference_ready)
            self.assertEqual("production_checkpoint_ready", readiness.status)

            # Prove the exact saved v2 weights perform a real forward, preserve
            # K=8 trajectories, and enter the variable-duration beam.  A toy
            # corpus is architecture evidence only; it is not a quality claim.
            inference_sequence = examples[-1].sequence
            materialized = materialize_stage_edit_head_frames_v2(
                checkpoint_directory=checkpoint,
                sequence=inference_sequence,
                canonical_source_ids=("A", "B"),
                schema=schema,
            )
            self.assertEqual(4, len(materialized.frames))
            self.assertNotEqual(
                materialized.supplied_sequence_sha256,
                materialized.input_sequence_sha256,
            )
            self.assertEqual(
                stage_edit_recent_history_mask_contract(schema),
                materialized.decision_history_input_contract,
            )
            self.assertTrue(
                all(frame.dynamic_crop_v2 is not None for frame in materialized.frames)
            )
            plan = SectionPlan(
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
                        target_source_switches_per_minute=22.0,
                        camera_motion_budget=0.7,
                        target_roll_events_per_minute=0.2,
                        establish_with_wide_shot=False,
                        protect_last_beat_from_extra_cut=False,
                    ),
                )
            )
            decoded = decode_stage_edit_semimarkov_beam(
                checkpoint_directory=checkpoint,
                sources=(Path("/tmp/A.mp4"), Path("/tmp/B.mp4")),
                source_metadata=(
                    StageEditSourceMetadata(0, "A", 2.0),
                    StageEditSourceMetadata(1, "B", 2.0),
                ),
                head_frames=materialized.frames,
                section_plan=plan,
                output=Path("/tmp/stageedit-toy-production.mp4"),
                schema=schema,
                config=StageEditSemiMarkovConfig(
                    beam_width=8,
                    timeline_count=8,
                    minimum_unique_sources=2,
                ),
                aspect_ratio=AspectRatio.LANDSCAPE,
                width=1920,
                height=1080,
                fps=30.0,
            )
            self.assertTrue(decoded.timelines)
            self.assertTrue(
                all(
                    decision.crop_contract
                    == DYNAMIC_CROP_V2_SEMIMARKOV_ADAPTER_CONTRACT
                    for timeline in decoded.timelines
                    for decision in timeline.decisions
                )
            )

            report = result.report
            self.assertEqual(
                stage_edit_recent_history_mask_contract(schema),
                report["decision_history_input_contract"],
            )
            self.assertFalse(
                report["source_scoring_contract"][
                    "absolute_source_index_parameters"
                ]
            )
            self.assertEqual(
                "training_partition_only_excluding_reference_validation",
                report["cut_imbalance_contract"]["scope"],
            )
            self.assertFalse(
                report["cut_imbalance_contract"]["heldout_labels_consumed"]
            )
            self.assertTrue(report["structured_head_coverage_gate"]["passed"])
            self.assertEqual(
                ["BANG", "BLACKHOLE"],
                report["held_out_evaluation"]["training_song_ids"],
            )
            self.assertEqual(
                ["XOXZ"], report["held_out_evaluation"]["held_out_song_ids"]
            )
            self.assertFalse(
                report["split"]["held_out_used_for_model_selection"]
            )
            self.assertEqual(2, len(report["split"]["validation_reference_ids"]))
            self.assertFalse(
                report["optimization"]["unseen_song_used_for_model_selection"]
            )
            self.assertTrue(
                report["checkpoint"]["production_activation_authorized"]
            )
            missing_snapshot_examples = list(examples)
            missing_snapshot_examples[0] = replace(
                missing_snapshot_examples[0],
                feature_snapshot_sha256=None,
                statistics={
                    **missing_snapshot_examples[0].statistics,
                    "materialized_feature_snapshot_used": False,
                },
            )
            missing_snapshot_corpus = replace(
                corpus, examples=tuple(missing_snapshot_examples)
            )
            with self.assertRaisesRegex(
                ValueError, "verified causal feature snapshot"
            ):
                train_production_stage_edit(
                    missing_snapshot_corpus,
                    checkpoint_directory=Path(directory) / "missing-features",
                    held_out_song_ids=("XOXZ",),
                    model_config=config,
                    epochs=1,
                    early_stopping_patience=1,
                    batch_size=1,
                    device=mx.cpu,
                    quality_thresholds=thresholds,
                )
            with self.assertRaisesRegex(ValueError, "other model format"):
                train_production_stage_edit(
                    corpus,
                    checkpoint_directory=checkpoint,
                    held_out_song_ids=("XOXZ",),
                    model_config=config,
                    epochs=1,
                    early_stopping_patience=1,
                    batch_size=1,
                    device=mx.cpu,
                    quality_thresholds=thresholds,
                )


if __name__ == "__main__":
    unittest.main()
