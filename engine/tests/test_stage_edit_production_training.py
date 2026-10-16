from __future__ import annotations

import copy
from dataclasses import replace
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
    DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
    DynamicCropTrajectoryTargetV2,
    StageEditDynamicCropTargetSequenceV2,
)
from crosscut.stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
    DYNAMIC_CROP_V3_MAX_KEYFRAMES,
    DYNAMIC_CROP_V3_MODEL_CONTRACT,
    DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
    DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION,
    DYNAMIC_CROP_V3_TARGET_CONTRACT,
    StageEditTransformerDynamicCropV3,
    load_stage_edit_dynamic_crop_checkpoint_v3,
)
from crosscut.stage_edit_dynamic_crop_training import (
    StageEditDynamicCropTrainingCorpusV2,
)
from crosscut.stage_edit_dynamic_crop_training_v3 import (
    StageEditDynamicCropTrainingCorpusV3,
    build_stage_edit_dynamic_crop_training_corpus_v3,
)
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter import (
    DYNAMIC_CROP_V2_SEMIMARKOV_ADAPTER_CONTRACT,
)
from crosscut.stage_edit_head_materializer import (
    materialize_stage_edit_head_frames_v2,
)
from crosscut.stage_edit_production_training import (
    DYNAMIC_CROP_V3_PRIOR_PROJECTION_RANK,
    DYNAMIC_CROP_V3_VALIDATION_SELECTION_CONTRACT,
    ProductionQualityThresholds,
    STAGE_EDIT_SELECTION_FREEZE_FILENAME,
    _HierarchicalSourceWindowSampler,
    _HierarchicalDynamicCropV3PriorSampler,
    _DynamicCropV3PriorProjectionNormalizer,
    _DynamicCropV3StructuredPriorAdapterModule,
    _balanced_cut_windows,
    _cut_only_pretraining_loss_contract,
    _dynamic_crop_v3_validation_selection_evidence,
    _dynamic_crop_v3_prior_inventory,
    _dynamic_crop_v3_prior_history_samples,
    _dynamic_crop_v3_prior_structured_selection_evidence,
    _dynamic_crop_v3_structured_prior_head_snapshot,
    _dynamic_crop_v3_structured_prior_projections,
    _require_unambiguous_checkpoint_namespace,
    _source_only_pretraining_loss_contract,
    _source_pretraining_inventory,
    _structured_head_coverage_gate,
    evaluate_stage_edit_production_quality,
    prepare_stage_edit_production_input_sequences,
    production_song_split,
    train_production_stage_edit,
)
from crosscut.stage_edit_reference_adapter import (
    ReferenceTrainingCorpus,
    ReferenceTrainingExample,
    verify_reference_training_feature_snapshot,
)
from crosscut.stage_edit_semimarkov_beam import (
    StageEditSemiMarkovConfig,
    StageEditSourceMetadata,
    decode_stage_edit_semimarkov_beam,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_CONTRACT,
    SourceCandidateToken,
    StageEditSequence,
    StageEditLossContract,
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
from tests.stage_edit_causal_sensing_test_support import (
    production_gate_statistics,
    synthetic_production_snapshot,
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


def _passing_dynamic_v3_metrics() -> dict:
    return {
        "schema_version": 3,
        "target_contract": DYNAMIC_CROP_V3_TARGET_CONTRACT,
        "model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
        "output_contract": DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
        "trajectory": {
            "track_label_count": 10,
            "keyframe_label_count": 20,
            "target_source_slot_mismatch_count": 0,
            "coordinates": {
                "zoom": {"label_count": 20, "mean_absolute_error": 0.03},
                "center_x": {"label_count": 20, "mean_absolute_error": 0.03},
                "center_y": {"label_count": 20, "mean_absolute_error": 0.03},
                "roll_degrees": {
                    "label_count": 20,
                    "mean_absolute_error": 0.30,
                },
            },
            "aggregate_normalized_improvement_over_static_prior": 0.04,
            "decoded_observed_time": {
                "contract": "actual_decoded_source_count_time_renderer_on_original_observations_v1",
                "complete": True, "expected_tracks": 10, "evaluated_tracks": 10,
                "source_identity_mismatch_count": 0, "predicted_count_and_time_consumed": True,
                "synthetic_target_points_used": False, "improvement_over_static_prior": 0.04,
            },
            "teacher_forced_count_normalized_time_mae": 0.03,
            "count": {
                "label_count": 10,
                "exact_accuracy": 0.60,
                "target_histogram_1_to_8": [3, 3, 4, 0, 0, 0, 0, 0],
                "predicted_histogram_1_to_8": [2, 4, 4, 0, 0, 0, 0, 0],
            },
            "motion": {
                "label_count": 10,
                "accuracy": 0.70,
                "balanced_accuracy": 0.70,
                "recall": {"static": 0.60, "dynamic": 0.80},
            },
            "curve_delta_and_range": {
                "zoom": {
                    "target_mean_range": 0.04,
                    "predicted_mean_range": 0.03,
                },
                "center_x": {
                    "target_mean_range": 0.01,
                    "predicted_mean_range": 0.008,
                },
                "center_y": {
                    "target_mean_range": 0.01,
                    "predicted_mean_range": 0.007,
                },
                "roll_degrees": {
                    "target_mean_range": 0.20,
                    "predicted_mean_range": 0.16,
                },
            },
            "renderer_decode": {
                "attempt_count": 10,
                "eligible_count": 10,
                "eligible_rate": 1.0,
                "minimum_safe_zoom_projection_count": 0,
                "rejection_reasons": {},
            },
        },
        "legacy_v2_checkpoint_or_output_used": False,
        "legacy_static_crop_head_used_for_trajectory": False,
    }


class StageEditProductionQualityTests(unittest.TestCase):
    def test_prior_history_counts_every_batch_member_and_rejects_foreign_labels(self):
        from copy import deepcopy

        samples = [
            {"song_id": "A", "reference_id": "a", "token_index": 1, "source_id": "s1"},
            {"song_id": "B", "reference_id": "b", "token_index": 2, "source_id": "s2"},
        ]
        record = {**samples[0], "batch_size": 2, "batch_tracks": samples}
        arguments = dict(expected_batch_size=2, reference_song_ids={"a": "A", "b": "B"})
        self.assertEqual(tuple(samples), _dynamic_crop_v3_prior_history_samples(record, **arguments))
        mutations = (
            lambda value: value.update(batch_size=True),
            lambda value: value["batch_tracks"].pop(),
            lambda value: value["batch_tracks"][1].update(reference_id="validation"),
            lambda value: value["batch_tracks"][1].update(song_id="A"),
            lambda value: value["batch_tracks"][1].update(token_index=True),
            lambda value: value.update(source_id="other"),
        )
        for mutate in mutations:
            bad = deepcopy(record)
            mutate(bad)
            with self.assertRaises(ValueError):
                _dynamic_crop_v3_prior_history_samples(bad, **arguments)
        self.assertEqual(
            ({"song_id": "A", "reference_id": "a"},),
            _dynamic_crop_v3_prior_history_samples(
                {"song_id": "A", "reference_id": "a"},
                expected_batch_size=None,
                reference_song_ids={"a": "A"},
            ),
        )

    def test_checkpoint_namespace_never_reinterprets_v2_as_v3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME).write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "other model format"):
                _require_unambiguous_checkpoint_namespace(
                    checkpoint,
                    dynamic_crop_version=3,
                )

    def test_dynamic_crop_v3_checkpoint_selection_gate_is_exact(self) -> None:
        passing = _dynamic_crop_v3_validation_selection_evidence(
            _passing_dynamic_v3_metrics()
        )
        self.assertTrue(passing["selectable"])
        self.assertEqual(
            DYNAMIC_CROP_V3_VALIDATION_SELECTION_CONTRACT,
            passing["contract"],
        )
        self.assertGreater(passing["score_component"], 0.0)
        self.assertFalse(passing["static_prior_can_be_selected"])

        mutations = (
            (
                ("trajectory", "renderer_decode", "eligible_rate"),
                0.99,
                "dynamic_crop_v3_validation_renderer_decode_not_exact",
            ),
            (
                (
                    "trajectory",
                    "renderer_decode",
                    "minimum_safe_zoom_projection_count",
                ),
                1,
                "dynamic_crop_v3_validation_projection_not_zero",
            ),
            (
                ("trajectory", "target_source_slot_mismatch_count"),
                1,
                "dynamic_crop_v3_validation_source_slot_mismatch",
            ),
            (
                (
                    "trajectory",
                    "aggregate_normalized_improvement_over_static_prior",
                ),
                0.0,
                "dynamic_crop_v3_validation_did_not_beat_static_prior",
            ),
            (
                ("trajectory", "count", "predicted_histogram_1_to_8"),
                [10, 0, 0, 0, 0, 0, 0, 0],
                "dynamic_crop_v3_validation_count_prediction_collapsed",
            ),
            (
                ("trajectory", "motion", "recall", "static"),
                0.0,
                "dynamic_crop_v3_validation_motion_prediction_collapsed",
            ),
            (
                (
                    "trajectory",
                    "curve_delta_and_range",
                    "zoom",
                    "predicted_mean_range",
                ),
                0.0,
                "dynamic_crop_v3_validation_curve_prediction_collapsed",
            ),
        )
        for path, value, expected_reason in mutations:
            with self.subTest(reason=expected_reason):
                metrics = copy.deepcopy(_passing_dynamic_v3_metrics())
                destination = metrics
                for key in path[:-1]:
                    destination = destination[key]
                destination[path[-1]] = value
                rejected = _dynamic_crop_v3_validation_selection_evidence(
                    metrics
                )
                if expected_reason == (
                    "dynamic_crop_v3_validation_curve_prediction_collapsed"
                ):
                    for coordinate in ("center_x", "center_y", "roll_degrees"):
                        metrics["trajectory"]["curve_delta_and_range"][coordinate][
                            "predicted_mean_range"
                        ] = 0.0
                    rejected = _dynamic_crop_v3_validation_selection_evidence(
                        metrics
                    )
                self.assertFalse(rejected["selectable"])
                self.assertIn(expected_reason, rejected["reasons"])

    def test_dynamic_crop_v3_prior_rejects_single_class_count_collapse(
        self,
    ) -> None:
        def metrics(
            predicted_histogram: list[int],
            confusion: list[list[int]],
            *,
            count_accuracy: float,
            time_mae: float,
            motion_balanced_accuracy: float,
        ) -> dict:
            padded_confusion = [
                row + [0] * (8 - len(row)) for row in confusion
            ] + [[0] * 8 for _ in range(8 - len(confusion))]
            return {
                "trajectory": {
                    "count": {
                        "exact_accuracy": count_accuracy,
                        "target_histogram_1_to_8": [2, 2, 2, 0, 0, 0, 0, 0],
                        "predicted_histogram_1_to_8": (
                            predicted_histogram + [0] * (8 - len(predicted_histogram))
                        ),
                        "confusion_target_rows_predicted_columns": (
                            padded_confusion
                        ),
                    },
                    "teacher_forced_count_normalized_time_mae": time_mae,
                    "motion": {
                        "balanced_accuracy": motion_balanced_accuracy
                    },
                }
            }

        initial = metrics(
            [6, 0, 0],
            [[2, 0, 0], [2, 0, 0], [2, 0, 0]],
            count_accuracy=1.0 / 3.0,
            time_mae=0.18,
            motion_balanced_accuracy=0.5,
        )
        improved = metrics(
            [2, 2, 2],
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            count_accuracy=1.0,
            time_mae=0.15,
            motion_balanced_accuracy=0.55,
        )
        accepted = _dynamic_crop_v3_prior_structured_selection_evidence(
            improved,
            initial,
        )
        self.assertTrue(accepted["selectable"])
        self.assertGreater(accepted["score_component"], 0.0)

        collapsed = _dynamic_crop_v3_prior_structured_selection_evidence(
            metrics(
                [0, 6, 0],
                [[0, 2, 0], [0, 2, 0], [0, 2, 0]],
                count_accuracy=1.0 / 3.0,
                time_mae=0.14,
                motion_balanced_accuracy=0.55,
            ),
            initial,
        )
        self.assertFalse(collapsed["selectable"])
        self.assertIn(
            "dynamic_crop_v3_prior_count_prediction_collapsed",
            collapsed["reasons"],
        )

    def test_dynamic_crop_v3_quality_uses_count_and_exact_renderer_schema(
        self,
    ) -> None:
        passed = evaluate_stage_edit_production_quality(
            _passing_metrics(),
            dynamic_crop_v3_metrics=_passing_dynamic_v3_metrics(),
        )
        self.assertTrue(passed.passed)
        with self.assertRaisesRegex(ValueError, "exactly one dynamic crop version"):
            evaluate_stage_edit_production_quality(
                _passing_metrics(),
                dynamic_crop_v2_metrics=_passing_dynamic_metrics(),
                dynamic_crop_v3_metrics=_passing_dynamic_v3_metrics(),
            )

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
        valid_gate = dict(
            example.statistics["production_causal_sensing_gate"]
        )
        partial_gate = dict(valid_gate)
        partial_gate.pop("ive_performer_state_content_sha256")
        wrong_mode_gate = dict(valid_gate)
        wrong_mode_gate["edit_mode"] = "member"
        old_schema_gate = dict(valid_gate)
        old_schema_gate["feature_snapshot_schema_version"] = 3
        for label, invalid_gate in (
            ("empty", None),
            ("partial", partial_gate),
            ("wrong-mode", wrong_mode_gate),
            ("old", old_schema_gate),
        ):
            with self.subTest(invalid_gate=label), self.assertRaisesRegex(
                ValueError,
                "verified causal feature snapshot",
            ):
                prepare_stage_edit_production_input_sequences(
                    (
                        replace(
                            example,
                            statistics={
                                **example.statistics,
                                "production_causal_sensing_gate": invalid_gate,
                            },
                        ),
                    )
                )
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

    def test_production_preprocessing_rejects_forged_statistics_and_reused_evidence(
        self,
    ) -> None:
        example = _training_example("BANG", "BANG-forged-snapshot")
        forged = replace(
            example,
            feature_snapshot_sha256="f" * 64,
            verified_feature_snapshot_evidence=None,
            statistics={
                **example.statistics,
                "materialized_feature_snapshot_used": True,
                "shared_feature_materializer_used": True,
            },
        )
        with self.assertRaisesRegex(
            ValueError,
            "verified causal feature snapshot",
        ):
            prepare_stage_edit_production_input_sequences((forged,))

        first = example.sequence.tokens[0]
        tampered = replace(
            example,
            sequence=replace(
                example.sequence,
                tokens=(
                    replace(
                        first,
                        music_features=(
                            first.music_features[0] + 0.01,
                            *first.music_features[1:],
                        ),
                    ),
                    *example.sequence.tokens[1:],
                ),
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            "verified causal feature snapshot",
        ):
            prepare_stage_edit_production_input_sequences((tampered,))

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
    snapshot = synthetic_production_snapshot(
        StageEditSequence(reference, tokens)
    )
    sequence = snapshot.sequence
    verified_evidence = verify_reference_training_feature_snapshot(
        reference_id=reference,
        sequence=sequence,
        feature_snapshot=snapshot,
        production_edit_mode="group",
    )
    return ReferenceTrainingExample(
        reference_id=reference,
        song_id=song,
        style_id="human-stage-mix",
        sequence=sequence,
        targets=StageEditTargetSequence(reference, targets),
        statistics={
            "dense_editorial_source_switch_positive_count": 2,
            "editorial_source_switch_positive_count": 2,
            "cut_positive_count": 2,
            "native_source_camera_cut_count": 0,
            "native_source_camera_cut_editorial_positive_count": 0,
            "materialized_feature_snapshot_used": True,
            "shared_feature_materializer_used": True,
            "production_causal_sensing_gate": production_gate_statistics(snapshot),
        },
        feature_snapshot_sha256=snapshot.snapshot_sha256,
        verified_feature_snapshot_evidence=verified_evidence,
    )


def _dynamic_crop_corpus(
    examples: tuple[ReferenceTrainingExample, ...],
) -> StageEditDynamicCropTrainingCorpusV2:
    sequences = {}
    for example in examples:
        targets = [None] * len(example.sequence.tokens)
        for token_index, source_id in ((1, "B"), (3, "A")):
            if token_index == 1:
                keyframes = (
                    EditorialTransformKeyframe(
                        0.0, 1.04, 0.50, 0.50, 0.0
                    ),
                )
                motion_label = "static"
            else:
                keyframes = (
                    EditorialTransformKeyframe(
                        0.0, 1.03, 0.49, 0.50, -0.2
                    ),
                    EditorialTransformKeyframe(
                        0.4, 1.08, 0.51, 0.49, 0.3
                    ),
                    EditorialTransformKeyframe(
                        0.8, 1.05, 0.50, 0.51, 0.0
                    ),
                )
                motion_label = "dynamic"
            targets[token_index] = DynamicCropTrajectoryTargetV2(
                shot_index=token_index,
                source_id=source_id,
                shot_duration_seconds=1.0,
                keyframes=keyframes,
                motion_label=motion_label,
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


def _dynamic_crop_corpus_v3(
    examples: tuple[ReferenceTrainingExample, ...],
) -> StageEditDynamicCropTrainingCorpusV3:
    return build_stage_edit_dynamic_crop_training_corpus_v3(
        _dynamic_crop_corpus(examples)
    )


def _source_balance_example(
    song: str,
    reference: str,
    event_count: int,
) -> ReferenceTrainingExample:
    base = _training_example(song, reference)
    template = base.sequence.tokens[0]
    tokens = tuple(
        replace(template, master_time_seconds=index * 0.5)
        for index in range(event_count + 1)
    )
    targets = [
        StageEditTarget(
            cut=None,
            selected_source_id="A",
            rejected_source_ids=("B",),
        )
    ]
    targets.extend(
        StageEditTarget(
            cut=1.0,
            selected_source_id="A" if index % 2 else "B",
            rejected_source_ids=("B",) if index % 2 else ("A",),
        )
        for index in range(1, event_count + 1)
    )
    sequence = StageEditSequence(reference, tokens)
    snapshot = synthetic_production_snapshot(sequence)
    verified_evidence = verify_reference_training_feature_snapshot(
        reference_id=reference,
        sequence=snapshot.sequence,
        feature_snapshot=snapshot,
        production_edit_mode="group",
    )
    return replace(
        base,
        sequence=snapshot.sequence,
        targets=StageEditTargetSequence(reference, tuple(targets)),
        statistics={
            **base.statistics,
            "production_causal_sensing_gate": production_gate_statistics(snapshot),
        },
        feature_snapshot_sha256=snapshot.snapshot_sha256,
        verified_feature_snapshot_evidence=verified_evidence,
    )


class StageEditSourcePretrainingTests(unittest.TestCase):
    def test_dynamic_crop_v3_prior_sampler_balances_song_then_reference(self) -> None:
        examples = tuple(
            _training_example(song, reference)
            for song, reference in (
                ("SONG-A", "a-dominant"),
                ("SONG-A", "a-sparse"),
                ("SONG-B", "b-one"),
                ("SONG-B", "b-two"),
            )
        )
        inventory = _dynamic_crop_v3_prior_inventory(
            examples, _dynamic_crop_corpus_v3(examples)
        )
        first = _HierarchicalDynamicCropV3PriorSampler(
            inventory.tracks, seed=87
        )
        second = _HierarchicalDynamicCropV3PriorSampler(
            inventory.tracks, seed=87
        )
        first_trace = [first.sample() for _ in range(400)]
        second_trace = [second.sample() for _ in range(400)]
        self.assertEqual(
            [(item.reference_id, item.token_index) for item in first_trace],
            [(item.reference_id, item.token_index) for item in second_trace],
        )
        self.assertEqual(
            {"SONG-A": 200, "SONG-B": 200},
            first.per_song_sampled_counts,
        )
        self.assertEqual(
            {
                "a-dominant": 100,
                "a-sparse": 100,
                "b-one": 100,
                "b-two": 100,
            },
            first.per_reference_sampled_counts,
        )
        batch=first.sample_batch(12)
        self.assertEqual([(t.reference_id,t.token_index) for t in batch],
            [(t.reference_id,t.token_index) for t in [second.sample() for _ in range(12)]])
        self.assertEqual({'SONG-A':206,'SONG-B':206},first.per_song_sampled_counts)
        for invalid in (0,65,True,1.5):
            with self.assertRaises(ValueError):first.sample_batch(invalid)
        self.assertEqual({'SONG-A':206,'SONG-B':206},first.per_song_sampled_counts)

    def test_sampler_balances_song_then_reference_then_group(self) -> None:
        examples = (
            _source_balance_example("SONG-A", "dominant-70", 70),
            _source_balance_example("SONG-A", "sparse-1", 1),
            _source_balance_example("SONG-B", "medium-3", 3),
            _source_balance_example("SONG-B", "medium-2", 2),
        )
        inventory = _source_pretraining_inventory(
            examples,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
            window_length=32,
        )
        self.assertEqual(4, inventory.opening_source_excluded_count)
        self.assertEqual(70, inventory.per_reference_confirmed_group_counts["dominant-70"])
        self.assertEqual(1, inventory.per_reference_confirmed_group_counts["sparse-1"])

        first = _HierarchicalSourceWindowSampler(inventory.windows, seed=91)
        second = _HierarchicalSourceWindowSampler(inventory.windows, seed=91)
        first_trace = [first.sample().group_id for _ in range(400)]
        second_trace = [second.sample().group_id for _ in range(400)]
        self.assertEqual(first_trace, second_trace)
        self.assertEqual({"SONG-A": 200, "SONG-B": 200}, first.per_song_sampled_counts)
        self.assertEqual(
            {
                "dominant-70": 100,
                "medium-2": 100,
                "medium-3": 100,
                "sparse-1": 100,
            },
            first.per_reference_sampled_counts,
        )
        dominant_group_counts = {
            group_id: first_trace.count(group_id)
            for group_id in set(first_trace)
            if group_id.startswith("dominant-70:")
        }
        self.assertLessEqual(
            max(dominant_group_counts.values())
            - min(dominant_group_counts.values()),
            1,
        )

    def test_cut_only_cannot_update_source_and_source_only_is_exact(self) -> None:
        contract = StageEditLossContract()
        cut_only = _cut_only_pretraining_loss_contract(contract)
        self.assertEqual(0.0, cut_only.source_listwise_weight)
        self.assertEqual(0.0, cut_only.source_pairwise_weight)
        self.assertEqual(1.0, cut_only.cut_weight)

        examples = (_training_example("SONG", "reference"),)
        positive, negative = _balanced_cut_windows(
            examples,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
            window_length=32,
        )
        self.assertTrue(
            all(
                target.selected_source_id is None
                for window in (*positive, *negative)
                for target in window.targets.targets
            )
        )

        source_only = _source_only_pretraining_loss_contract(contract)
        self.assertEqual("eligible_all_hybrid", source_only.source_objective)
        self.assertGreater(source_only.source_listwise_weight, 0.0)
        self.assertGreater(source_only.source_pairwise_weight, 0.0)
        self.assertEqual(0.0, source_only.cut_weight)
        self.assertEqual(0.0, source_only.duration_weight)
        self.assertEqual(0.0, source_only.transition_weight)
        self.assertEqual(0.0, source_only.crop_weight)


@unittest.skipUnless(mlx_runtime_version() is not None, "MLX is optional")
class StageEditProductionTrainingSmokeTests(unittest.TestCase):
    def test_trajectory_prior_adapter_can_modify_all_renderer_rows(self) -> None:
        dimension = 16
        values_stop = DYNAMIC_CROP_V3_MAX_KEYFRAMES * 4
        initial = {
            "context_center": mx.zeros((dimension,)),
            "source_center": mx.zeros((dimension,)),
            "context_weight": mx.arange(
                DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION * dimension,
                dtype=mx.float32,
            ).reshape((DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION, dimension)),
            "context_bias": mx.arange(
                DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION,
                dtype=mx.float32,
            ),
            "source_weight": mx.arange(
                DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION * dimension,
                dtype=mx.float32,
            ).reshape((DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION, dimension)),
        }
        context_projection, source_projection = (
            _dynamic_crop_v3_structured_prior_projections(
                dimension,
                seed=29,
                requested_rank=DYNAMIC_CROP_V3_PRIOR_PROJECTION_RANK,
            )
        )
        adapter = _DynamicCropV3StructuredPriorAdapterModule(
            DYNAMIC_CROP_V3_PRIOR_PROJECTION_RANK
        )
        adapter.context_adapter.weight = mx.ones_like(
            adapter.context_adapter.weight
        )
        adapter.context_adapter.bias = mx.ones_like(
            adapter.context_adapter.bias
        )
        adapter.source_adapter.weight = mx.ones_like(
            adapter.source_adapter.weight
        )
        normalizer = _DynamicCropV3PriorProjectionNormalizer(
            context_mean=mx.zeros((DYNAMIC_CROP_V3_PRIOR_PROJECTION_RANK,)),
            context_scale=mx.ones((DYNAMIC_CROP_V3_PRIOR_PROJECTION_RANK,)),
            source_mean=mx.zeros((DYNAMIC_CROP_V3_PRIOR_PROJECTION_RANK,)),
            source_scale=mx.ones((DYNAMIC_CROP_V3_PRIOR_PROJECTION_RANK,)),
            context_center=mx.zeros((dimension,)),
            source_center=mx.zeros((dimension,)),
            evidence={},
        )
        snapshot = _dynamic_crop_v3_structured_prior_head_snapshot(
            adapter,
            initial,
            context_projection,
            source_projection,
            normalizer,
        )

        for name in ("context_weight", "source_weight"):
            self.assertNotEqual(
                initial[name][:values_stop].tolist(),
                snapshot[name][:values_stop].tolist(),
            )
            self.assertNotEqual(
                initial[name][values_stop:].tolist(),
                snapshot[name][values_stop:].tolist(),
            )
        self.assertNotEqual(
            initial["context_bias"][:values_stop].tolist(),
            snapshot["context_bias"][:values_stop].tolist(),
        )
        self.assertNotEqual(
            initial["context_bias"][values_stop:].tolist(),
            snapshot["context_bias"][values_stop:].tolist(),
        )

    def test_v3_trains_saves_reloads_and_reports_distinct_contracts(self) -> None:
        schema = DEFAULT_STAGE_EDIT_SCHEMA
        examples = tuple(
            _training_example(song, f"{song}-v3-ref-{index}")
            for song in ("BANG", "BLACKHOLE", "XOXZ")
            for index in range(2)
        )
        corpus = ReferenceTrainingCorpus(
            examples=examples,
            data_id="3" * 64,
            source_paths=(),
            statistics={
                "crop_coordinate_supervision_count": 24,
                "transition_supervision_count": 6,
            },
        )
        dynamic_corpus = _dynamic_crop_corpus_v3(examples)
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
            checkpoint = Path(directory) / "v3-checkpoint"
            result = train_production_stage_edit(
                corpus,
                checkpoint_directory=checkpoint,
                held_out_song_ids=("XOXZ",),
                model_config=config,
                epochs=8,
                early_stopping_patience=16,
                batch_size=2,
                learning_rate=3.0e-3,
                cut_pretrain_steps=0,
                source_pretrain_steps=0,
                dynamic_crop_v3_prior_pretrain_steps=500,
                dynamic_crop_v3_prior_pretrain_learning_rate=3.0e-4,
                dynamic_crop_v3_prior_pretrain_warmup_steps=50,
                finetune_warmup_steps=0,
                minimum_finetune_optimizer_steps=1,
                minimum_checkpoint_selection_finetune_steps=1,
                seed=29,
                device=mx.cpu,
                dynamic_crop_corpus=dynamic_corpus,
                quality_thresholds=thresholds,
                authorize_production_activation=False,
            )

            self.assertTrue(
                (checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME).is_file()
            )
            self.assertTrue(
                (checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME).is_file()
            )
            self.assertFalse((checkpoint / "dynamic-crop-v2-metadata.json").exists())
            reloaded_model, reloaded_metadata = (
                load_stage_edit_dynamic_crop_checkpoint_v3(
                    checkpoint,
                    expected_schema=schema,
                )
            )
            self.assertIsInstance(
                reloaded_model, StageEditTransformerDynamicCropV3
            )
            self.assertEqual(
                DYNAMIC_CROP_V3_MODEL_CONTRACT,
                reloaded_metadata.model_contract,
            )
            self.assertGreater(reloaded_metadata.training_steps, 0)

            report = result.report
            self.assertEqual(3, report["checkpoint"]["dynamic_crop_version"])
            self.assertTrue(report["checkpoint"]["dynamic_crop_v3_checkpoint"])
            self.assertFalse(report["checkpoint"]["dynamic_crop_v2_checkpoint"])
            self.assertEqual(3, report["model"]["dynamic_crop_version"])
            self.assertTrue(report["model"]["dynamic_crop_v3_enabled"])
            self.assertFalse(report["model"]["dynamic_crop_v2_enabled"])
            activation_checkpointing = report["model"][
                "activation_checkpointing"
            ]
            self.assertEqual(
                DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_CONTRACT,
                activation_checkpointing["contract"],
            )
            self.assertTrue(activation_checkpointing["enabled"])
            self.assertTrue(activation_checkpointing["training_only"])
            self.assertFalse(
                activation_checkpointing["serialized_parameter_tree_changed"]
            )
            self.assertFalse(reloaded_model.base.encoder.checkpoint)
            self.assertIsNone(report["data"]["dynamic_crop_v2"])
            self.assertEqual(
                dynamic_corpus.data_id,
                report["data"]["dynamic_crop_v3"]["data_id"],
            )
            self.assertEqual(3, report["data"]["dynamic_crop_version"])
            selected = report["optimization"][
                "best_validation_selection_evidence"
            ]
            self.assertTrue(selected["selectable"])
            self.assertGreater(selected["dynamic_crop_v3_score_component"], 0.0)
            self.assertTrue(
                selected["dynamic_crop_v3_selection_gate"]["selectable"]
            )
            self.assertEqual(
                DYNAMIC_CROP_V3_VALIDATION_SELECTION_CONTRACT,
                report["optimization"]["dynamic_crop_v3_model_selection"],
            )
            self.assertEqual(
                report["optimization"][
                    "effective_dynamic_crop_v3_loss_contract"
                ],
                report["optimization"][
                    "effective_dynamic_crop_loss_contract"
                ],
            )
            self.assertTrue(
                report["optimization"][
                    "reloaded_checkpoint_validation_selection_evidence"
                ]["dynamic_crop_v3_selection_gate"]["selectable"]
            )
            self.assertIsNotNone(
                report["history"][0][
                    "training_song_reference_validation_dynamic_crop_v3"
                ]
            )
            self.assertIsNone(
                report["history"][0][
                    "training_song_reference_validation_dynamic_crop_v2"
                ]
            )
            self.assertIsNotNone(
                report["held_out_evaluation"]["dynamic_crop_v3_metrics"]
            )
            self.assertIsNone(
                report["held_out_evaluation"]["dynamic_crop_v2_metrics"]
            )
            self.assertFalse(
                report["held_out_evaluation"][
                    "heldout_dynamic_crop_metrics_used_for_checkpoint_selection"
                ]
            )
            prior = report["optimization"][
                "dynamic_crop_v3_prior_pretraining"
            ]
            self.assertEqual(
                500,
                report["optimization"][
                    "dynamic_crop_v3_prior_pretraining_optimizer_steps"
                ],
            )
            self.assertEqual(
                500,
                len(report["dynamic_crop_v3_prior_pretraining_history"]),
            )
            self.assertTrue(
                prior["parameter_invariance"][
                    "all_declared_invariant_groups_byte_identical"
                ]
            )
            candidate = prior["candidate_selection"]
            trained_prior_selected = (
                candidate["selected_candidate_origin"]
                == "trained_prior_candidate"
            )
            self.assertEqual(
                trained_prior_selected,
                prior["parameter_invariance"]["context_value_bias_changed"]
            )
            self.assertEqual(
                trained_prior_selected,
                prior["parameter_invariance"][
                    "dynamic_head_parameter_tree_changed"
                ],
            )
            self.assertEqual(
                trained_prior_selected,
                prior["parameter_invariance"][
                    "context_non_value_bias_changed"
                ],
            )
            self.assertEqual(
                trained_prior_selected,
                prior["parameter_invariance"]["context_weight_changed"],
            )
            self.assertEqual(
                trained_prior_selected,
                prior["parameter_invariance"]["source_head_changed"],
            )
            self.assertEqual(
                trained_prior_selected,
                prior["parameter_invariance"][
                    "context_value_weight_changed"
                ]
            )
            self.assertEqual(
                trained_prior_selected,
                prior["parameter_invariance"][
                    "context_structured_weight_changed"
                ],
            )
            self.assertEqual(
                trained_prior_selected,
                prior["parameter_invariance"]["source_value_weight_changed"]
            )
            self.assertEqual(
                trained_prior_selected,
                prior["parameter_invariance"][
                    "source_structured_weight_changed"
                ],
            )
            self.assertTrue(
                prior["parameter_invariance"]["value_row_gradients_computed"]
            )
            self.assertTrue(
                prior["parameter_invariance"][
                    "structured_low_rank_adapter_gradients_computed"
                ]
            )
            self.assertTrue(
                prior["parameter_invariance"][
                    "count_time_motion_bias_gradients_computed"
                ]
            )
            self.assertEqual(
                [50, 100, 200, 300, 400, 500],
                candidate["predeclared_candidate_steps"],
            )
            self.assertEqual(6, len(candidate["candidate_evaluations"]))
            selectable_candidates = [
                item
                for item in candidate["candidate_evaluations"]
                if item["selection"]["selectable"]
            ]
            if selectable_candidates:
                expected_selected = max(
                    selectable_candidates,
                    key=lambda item: (
                        item["selection"]["score"],
                        -item["candidate_step"],
                    ),
                )
                self.assertEqual(
                    expected_selected["candidate_step"],
                    candidate["selected_candidate_step"],
                )
                self.assertEqual(
                    expected_selected["value_bias_sha256"],
                    candidate["selected_candidate_value_bias_sha256"],
                )
                self.assertEqual(
                    expected_selected["head_parameter_sha256"],
                    candidate["selected_candidate_head_parameter_sha256"],
                )
            else:
                self.assertEqual(0, candidate["selected_candidate_step"])
                self.assertIn(
                    candidate["selected_candidate_origin"],
                    {
                        "validated_initial_dynamic_head",
                        "unvalidated_initial_dynamic_head_fallback",
                    },
                )
            candidate_gate = candidate["selection"][
                "dynamic_crop_v3_selection_gate"
            ]
            self.assertFalse(candidate["validation_gradients_computed"])
            self.assertFalse(candidate["heldout_forward_performed"])
            retention = prior["retention"]
            reset_steps = [
                step
                for epoch in report["history"]
                for step in epoch["finetune_optimizer_reset_before_steps"]
            ]
            phase_steps = [
                step
                for epoch in report["history"]
                for step in epoch["finetune_phase_local_optimizer_steps"]
            ]
            if candidate["retention_authorized"]:
                self.assertTrue(candidate_gate["selectable"])
                self.assertGreater(candidate_gate["score_component"], 0.0)
                self.assertTrue(
                    retention["all_dynamic_head_parameters_byte_identical"]
                )
                self.assertTrue(
                    retention["all_feature_extractor_parameters_byte_identical"]
                )
                self.assertEqual(
                    retention["warmup_feature_sha256"],
                    retention["retained_feature_sha256"],
                )
                self.assertTrue(retention["feature_dropout_disabled_during_retention"])
                self.assertTrue(retention["feature_extractor_released_after_retention"])
                self.assertTrue(retention["feature_extractor_parameters_changed_after_release"])
                self.assertTrue(
                    retention[
                        "optimizer_state_reset_before_dynamic_head_release"
                    ]
                )
                self.assertEqual(
                    0,
                    retention["released_joint_optimizer_initial_step"],
                )
                self.assertEqual([2], reset_steps)
                self.assertEqual([1, 1, 2, 3, 4, 5, 6, 7], phase_steps)
            else:
                self.assertFalse(candidate_gate["selectable"])
                self.assertIsNone(retention)
                initializer = candidate["joint_training_initializer"]
                self.assertFalse(initializer["checkpoint_qualification_or_retention_authorized"])
                self.assertFalse(initializer["validation_or_heldout_inputs_used"])
                self.assertFalse(initializer["gradients_computed"])
                self.assertEqual(0, initializer["new_encoder_forwards"])
                self.assertLessEqual(initializer["selected_training_loss"], initializer["initial_training_loss"])
                self.assertGreater(initializer["training_trajectory_count"], 0)
                self.assertEqual([], reset_steps)
                self.assertEqual(list(range(1, len(phase_steps) + 1)), phase_steps)
            self.assertIn(
                report["optimization"]["best_checkpoint_origin"],
                {
                    "training_only_dynamic_head_prior_retained_to_burn_in",
                    "validated_initial_dynamic_head_retained_to_burn_in",
                    "joint_finetuning",
                },
            )

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
                source_pretrain_steps=500,
                minimum_finetune_optimizer_steps=1,
                minimum_checkpoint_selection_finetune_steps=1,
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
            self.assertTrue(
                selection_freeze["selected_after_checkpoint_burn_in"]
            )
            self.assertGreaterEqual(
                selection_freeze["selected_finetune_optimizer_step"], 1
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
            optimization = report["optimization"]
            self.assertEqual(2, optimization["multi_head_finetuning_optimizer_steps"])
            self.assertEqual(500, optimization["source_pretraining_optimizer_steps"])
            self.assertEqual(502, optimization["source_head_update_steps"])
            self.assertEqual(
                0, optimization["source_pretraining_optimizer_initial_step"]
            )
            self.assertEqual(
                500,
                optimization["source_pretraining_optimizer_state_final_step"],
            )
            self.assertTrue(
                optimization[
                    "source_pretraining_optimizer_state_reset_after_cut_pretraining"
                ]
            )
            self.assertTrue(
                optimization[
                    "finetune_optimizer_state_reset_after_cut_and_source_pretraining"
                ]
            )
            self.assertEqual(0, optimization["checkpoint_writes_before_joint_finetuning"])
            source_sampling = optimization["source_pretraining"]["sampling"]
            self.assertEqual(4, source_sampling["confirmed_source_event_count"])
            self.assertEqual(
                500, sum(source_sampling["per_reference_sampled_counts"].values())
            )
            self.assertEqual(500, len(report["source_pretraining_history"]))
            self.assertAlmostEqual(
                3.0e-5,
                report["source_pretraining_history"][0][
                    "actual_learning_rate"
                ],
                places=10,
            )
            self.assertEqual(
                1,
                report["source_pretraining_history"][0]["source_pretrain_step"],
            )
            self.assertEqual(
                500,
                report["source_pretraining_history"][-1]["source_pretrain_step"],
            )
            self.assertEqual(
                500,
                report["source_pretraining_history"][-1]["optimizer_step"],
            )
            self.assertEqual(502, report["history"][0]["optimizer_step"])
            self.assertTrue(
                all(
                    not record["checkpoint_selection_eligible"]
                    for record in report["source_pretraining_history"]
                )
            )
            self.assertEqual(0, optimization["finetune_optimizer_initial_step"])
            self.assertEqual(2, optimization["finetune_optimizer_state_final_step"])
            self.assertTrue(
                optimization[
                    "finetune_optimizer_state_reset_after_cut_pretraining"
                ]
            )
            self.assertFalse(optimization["early_stopping_triggered"])
            self.assertEqual(
                "planned_finetune_steps_completed",
                optimization["training_stop_reason"],
            )
            self.assertEqual(
                "finetune_optimizer_steps",
                optimization["early_stopping_patience_unit"],
            )
            self.assertEqual([1, 2], report["history"][0]["finetune_optimizer_steps"])
            self.assertEqual(
                2,
                len(report["history"][0]["finetune_actual_learning_rates"]),
            )
            history_dynamic = report["history"][0][
                "training_song_reference_validation_dynamic_crop_v2"
            ]
            self.assertEqual(2, history_dynamic["schema_version"])
            self.assertEqual(
                history_dynamic,
                optimization["selected_checkpoint_validation_metrics"][
                    "dynamic_crop_v2"
                ],
            )
            self.assertTrue(
                report["checkpoint"]["production_activation_authorized"]
            )
            with self.assertRaisesRegex(
                ValueError, "planned multi-head finetuning.*floor"
            ):
                train_production_stage_edit(
                    corpus,
                    checkpoint_directory=Path(directory) / "undertrained",
                    held_out_song_ids=("XOXZ",),
                    model_config=config,
                    epochs=1,
                    early_stopping_patience=1,
                    batch_size=1,
                    cut_pretrain_steps=0,
                    source_pretrain_steps=0,
                    device=mx.cpu,
                    quality_thresholds=thresholds,
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
                    source_pretrain_steps=0,
                    minimum_finetune_optimizer_steps=1,
                    minimum_checkpoint_selection_finetune_steps=1,
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
                    source_pretrain_steps=0,
                    minimum_finetune_optimizer_steps=1,
                    minimum_checkpoint_selection_finetune_steps=1,
                    device=mx.cpu,
                    quality_thresholds=thresholds,
                )


if __name__ == "__main__":
    unittest.main()
