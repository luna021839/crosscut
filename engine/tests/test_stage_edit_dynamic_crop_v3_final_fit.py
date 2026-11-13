from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from crosscut import stage_edit_dynamic_crop_v3_final_fit as final_fit
from crosscut.stage_edit_dynamic_crop_capacity_probe import SELECTION_SONG_IDS
from crosscut.stage_edit_dynamic_crop_v3_capacity_smoke import (
    V3_PROBE_TRAINING_SCHEDULE as PROBE_TRAINING_SCHEDULE,
)
from crosscut.stage_edit_dynamic_crop_head_v3 import (
    DynamicCropV3LossContract,
    StageEditTransformerDynamicCropV3,
    load_stage_edit_dynamic_crop_checkpoint_v3,
)
from crosscut.stage_edit_reference_adapter import ReferenceTrainingCorpus
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_CONTRACT,
    StageEditLossContract,
    StageEditTransformerConfig,
    mlx_runtime_version,
)
from tests.test_stage_edit_production_training import (
    _dynamic_crop_corpus_v3,
    _training_example,
)

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None


class DynamicCropV3FinalFitContractTests(unittest.TestCase):
    def _corpus(self, *, references_per_song: int = 5):
        examples = tuple(
            SimpleNamespace(
                song_id=song,
                reference_id=f"{song}-ref-{index}",
                sequence=SimpleNamespace(sequence_id=f"seq-{song}-{index}"),
            )
            for song in SELECTION_SONG_IDS
            for index in range(references_per_song)
        )
        return SimpleNamespace(examples=examples)

    def _report(self, freeze_path: Path) -> dict:
        cut_steps = PROBE_TRAINING_SCHEDULE["cut_pretraining_steps"]
        source_steps = PROBE_TRAINING_SCHEDULE["source_pretraining_steps"]
        prior_steps = PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_pretraining_steps"
        ]
        finetune_steps = PROBE_TRAINING_SCHEDULE[
            "finetune_steps_planned"
        ]
        retention_steps = PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_retention_finetune_steps"
        ]
        total_steps = cut_steps + source_steps + prior_steps + finetune_steps
        candidate_steps = tuple(
            final_fit.production.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
        )
        selected_candidate_step = 200
        candidate_digests = {
            step: hashlib.sha256(f"candidate-{step}".encode()).hexdigest()
            for step in candidate_steps
        }
        invariant_digests = {
            key: hashlib.sha256(key.encode()).hexdigest()
            for key in (
                "base_parameter_tree_sha256",
                "context_weight_sha256",
                "source_head_parameter_tree_sha256",
                "context_non_value_bias_sha256",
            )
        }
        prior_before_digests = {
            **invariant_digests,
            "context_value_bias_sha256": hashlib.sha256(
                b"initial-value-bias"
            ).hexdigest(),
        }
        prior_after_digests = {
            **invariant_digests,
            "context_value_bias_sha256": candidate_digests[
                selected_candidate_step
            ],
        }
        retained_head_keys = (
            "context_weight_sha256",
            "source_head_parameter_tree_sha256",
            "context_non_value_bias_sha256",
            "context_value_bias_sha256",
        )
        peak = float(PROBE_TRAINING_SCHEDULE["learning_rate"])
        warmup = int(PROBE_TRAINING_SCHEDULE["finetune_warmup_steps"])
        minimum = peak * float(
            PROBE_TRAINING_SCHEDULE["minimum_learning_rate_ratio"]
        )
        decay_steps = finetune_steps - warmup
        learning_rates = []
        for schedule_index in range(finetune_steps):
            if schedule_index < warmup:
                learning_rates.append(
                    peak * 0.1
                    + (peak - peak * 0.1) * schedule_index / warmup
                )
            else:
                local = min(schedule_index - warmup, decay_steps)
                learning_rates.append(
                    minimum
                    + (peak - minimum)
                    * 0.5
                    * (1.0 + math.cos(math.pi * local / decay_steps))
                )
        prior_peak = float(
            PROBE_TRAINING_SCHEDULE[
                "dynamic_crop_v3_prior_pretraining_peak_learning_rate"
            ]
        )
        prior_warmup = int(
            PROBE_TRAINING_SCHEDULE[
                "dynamic_crop_v3_prior_pretraining_warmup_steps"
            ]
        )
        prior_minimum = prior_peak * float(
            PROBE_TRAINING_SCHEDULE["minimum_learning_rate_ratio"]
        )
        prior_decay = prior_steps - prior_warmup
        prior_learning_rates = []
        for schedule_index in range(prior_steps):
            if schedule_index < prior_warmup:
                prior_learning_rates.append(
                    prior_peak * 0.1
                    + (prior_peak - prior_peak * 0.1)
                    * schedule_index
                    / prior_warmup
                )
            else:
                local = min(schedule_index - prior_warmup, prior_decay)
                prior_learning_rates.append(
                    prior_minimum
                    + (prior_peak - prior_minimum)
                    * 0.5
                    * (1.0 + math.cos(math.pi * local / prior_decay))
                )
        references_by_song = {
            song: [f"{song}-ref-{index}" for index in range(5)]
            for song in SELECTION_SONG_IDS
        }
        reference_sample_counts = {
            f"{song}-ref-{index}": (
                34
                if song == SELECTION_SONG_IDS[0] and index < 2
                else 33
                if song == SELECTION_SONG_IDS[0]
                else 34
                if song == SELECTION_SONG_IDS[1] and index < 2
                else 33
                if song == SELECTION_SONG_IDS[1]
                else 34
                if index == 0
                else 33
            )
            for song in SELECTION_SONG_IDS
            for index in range(5)
        }
        sampled_tracks = [
            (song, reference_id)
            for song in SELECTION_SONG_IDS
            for reference_id in references_by_song[song]
            for _ in range(reference_sample_counts[reference_id])
        ]
        return {
            "schema_version": final_fit.FINAL_FIT_REPORT_SCHEMA_VERSION,
            "run_kind": final_fit.FINAL_FIT_RUN_KIND,
            "contract": final_fit.FINAL_FIT_CONTRACT,
            "created_at": "2026-08-24T00:00:00+00:00",
            "source_commit": "b" * 40,
            "release_freeze": {
                "path": str(freeze_path.resolve()),
                "file_sha256": "a" * 64,
            },
            "dynamic_crop_version": 3,
            "model_contract": final_fit.DYNAMIC_CROP_V3_MODEL_CONTRACT,
            "target_contract": final_fit.DYNAMIC_CROP_V3_TARGET_CONTRACT,
            "output_contract": final_fit.DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
            "loss_contract": final_fit.DYNAMIC_CROP_V3_LOSS_CONTRACT,
            "all_reference_fit": {
                "song_ids": list(SELECTION_SONG_IDS),
                "reference_count": 15,
                "references_by_song": references_by_song,
                "all_references_consumed_by_optimizer": True,
                "reference_validation_partition_created": False,
                "held_out_partition_created": False,
                "final_fit_data_used_for_candidate_selection": False,
                "final_fit_data_used_for_schedule_selection": False,
                "final_fit_data_used_for_threshold_selection": False,
            },
            "frozen_decisions": {
                "candidate_config": {},
                "base_loss_contract": {},
                "dynamic_crop_loss_contract": {},
                "training_schedule": dict(PROBE_TRAINING_SCHEDULE),
                "dynamic_crop_v3_prior_candidate_step": (
                    selected_candidate_step
                ),
                "cut_probability_threshold": 0.5,
                "quality_thresholds": {"cut_auroc_minimum": 0.5},
                "decisions_frozen_before_final_fit": True,
            },
            "training": {
                "optimizer_steps": total_steps,
                "cut_pretraining_optimizer_steps": cut_steps,
                "source_pretraining_optimizer_steps": source_steps,
                "dynamic_crop_v3_prior_pretraining_optimizer_steps": (
                    prior_steps
                ),
                "finetune_optimizer_steps": finetune_steps,
                "dynamic_crop_v3_prior_pretraining": {
                    "optimizer_weight_decay": 0.0,
                    "optimizer_initial_step": 0,
                    "optimizer_state_final_step": prior_steps,
                    "checkpoint_selection_performed": False,
                    "validation_metrics_computed": False,
                    "heldout_metrics_computed": False,
                    "predeclared_candidate_steps": list(candidate_steps),
                    "candidate_selection_contract": (
                        final_fit.production.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_SELECTION_CONTRACT
                    ),
                    "candidate_snapshots": [
                        {
                            "candidate_step": step,
                            "value_bias_sha256": candidate_digests[step],
                            "training_only_updates": True,
                        }
                        for step in candidate_steps
                    ],
                    "frozen_selected_candidate_step": selected_candidate_step,
                    "frozen_selected_candidate_value_bias_sha256": (
                        candidate_digests[selected_candidate_step]
                    ),
                    "final_fit_data_used_for_candidate_selection": False,
                    "sampling": {
                        "per_song_sampled_counts": {
                            SELECTION_SONG_IDS[0]: 167,
                            SELECTION_SONG_IDS[1]: 167,
                            SELECTION_SONG_IDS[2]: 166,
                        },
                        "per_reference_sampled_counts": reference_sample_counts,
                        "reference_song_ids": {
                            reference_id: song
                            for song in SELECTION_SONG_IDS
                            for reference_id in references_by_song[song]
                        },
                    },
                    "parameter_invariance": {
                        "all_declared_invariant_groups_byte_identical": True,
                        "context_value_bias_changed": True,
                        "before": prior_before_digests,
                        "after": prior_after_digests,
                        "invariant_parameter_groups": [
                            "base_parameter_tree_sha256",
                            "context_weight_sha256",
                            "source_head_parameter_tree_sha256",
                            "context_non_value_bias_sha256",
                        ],
                    },
                    "retention": {
                        "retention_finetune_step_count": retention_steps,
                        "retention_optimizer_state_final_step": (
                            retention_steps
                        ),
                        "released_joint_optimizer_initial_step": 0,
                        "released_joint_optimizer_state_final_step": (
                            finetune_steps - retention_steps
                        ),
                        "optimizer_state_reset_before_dynamic_head_release": (
                            True
                        ),
                        "released_joint_optimizer_schedule_offset_steps": (
                            retention_steps
                        ),
                        "all_dynamic_head_parameters_byte_identical_at_release": (
                            True
                        ),
                        "warmup_head_digests": {
                            key: prior_after_digests[key]
                            for key in retained_head_keys
                        },
                        "retention_boundary_head_digests": {
                            key: prior_after_digests[key]
                            for key in retained_head_keys
                        },
                        "post_joint_head_digests": {
                            key: hashlib.sha256(
                                f"post-joint-{key}".encode()
                            ).hexdigest()
                            for key in retained_head_keys
                        },
                    },
                },
                "dynamic_crop_v3_prior_pretraining_history": [
                    {
                        "phase_step": step,
                        "optimizer_step": cut_steps + source_steps + step,
                        "song_id": song,
                        "reference_id": reference_id,
                        "actual_learning_rate": prior_learning_rates[step - 1],
                        "checkpoint_selection_performed": False,
                        "validation_labels_consumed": False,
                        "heldout_labels_consumed": False,
                        "predeclared_candidate_snapshot": (
                            step in candidate_steps
                        ),
                        "candidate_value_bias_sha256": (
                            candidate_digests[step]
                            if step in candidate_steps
                            else None
                        ),
                    }
                    for step, (song, reference_id) in enumerate(
                        sampled_tracks, start=1
                    )
                ],
                "finetune_history": [
                    {
                        "phase_local_optimizer_steps": [
                            *range(1, retention_steps + 1),
                            *range(
                                1,
                                finetune_steps - retention_steps + 1,
                            ),
                        ],
                        "optimizer_reset_before_finetune_steps": [
                            retention_steps + 1
                        ],
                        "dynamic_crop_v3_prior_retention_finetune_steps": (
                            list(range(1, retention_steps + 1))
                        ),
                        "actual_learning_rates": learning_rates,
                        "checkpoint_selection_performed": False,
                        "validation_labels_consumed": False,
                    }
                ],
                "checkpoint_selection_performed": False,
                "validation_metrics_computed": False,
                "heldout_metrics_computed": False,
                "activation_checkpointing": {
                    "contract": DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_CONTRACT,
                    "enabled": True,
                    "scope": "base_transformer_encoder_all_layers",
                    "implementation": "mlx.nn.TransformerEncoder.checkpoint",
                    "training_only": True,
                    "serialized_parameter_tree_changed": False,
                    "inference_runtime_requires_checkpointing": False,
                    "previous_value": False,
                },
            },
            "checkpoint": {
                "metadata_filename": "dynamic-crop-v3-metadata.json",
                "weight_filename": "dynamic-crop-v3-model.safetensors",
                "training_steps": total_steps,
                "inference_activation_checkpointing_enabled": False,
            },
            "authority": {
                "production_ready": False,
                "production_registry_activation_authorized": False,
                "production_registry_activation_requested": False,
                "learned_source_pair_authority_present": False,
                "blind_holdout_quality_claim_authorized": False,
                "xoxz_dynamic_crop_v3_blind_evidence_present": False,
                "attitude_dynamic_crop_v3_blind_evidence_present": False,
                "human_parity_claim_authorized": False,
                "experimental_unseen_ive_inference_only": True,
            },
        }

    def test_all_fifteen_references_are_required(self) -> None:
        examples, by_song = final_fit._validate_complete_reference_corpus(
            self._corpus()
        )
        self.assertEqual(len(examples), 15)
        self.assertEqual(set(by_song), set(SELECTION_SONG_IDS))
        with self.assertRaisesRegex(
            final_fit.DynamicCropV3FinalFitError,
            "exactly fifteen",
        ):
            final_fit._validate_complete_reference_corpus(
                self._corpus(references_per_song=4)
            )

    def test_schedule_is_exact_and_not_tunable_at_final_fit(self) -> None:
        self.assertEqual(
            final_fit._validate_schedule(PROBE_TRAINING_SCHEDULE),
            dict(PROBE_TRAINING_SCHEDULE),
        )
        changed = dict(PROBE_TRAINING_SCHEDULE)
        changed["finetune_steps_planned"] = 121
        with self.assertRaisesRegex(
            final_fit.DynamicCropV3FinalFitError,
            "exact LOSO training schedule",
        ):
            final_fit._validate_schedule(changed)
        type_forged = dict(PROBE_TRAINING_SCHEDULE)
        type_forged["gradient_clip_norm"] = True
        with self.assertRaisesRegex(
            final_fit.DynamicCropV3FinalFitError,
            "exact LOSO training schedule",
        ):
            final_fit._validate_schedule(type_forged)

    def test_report_loader_requires_inactive_no_blind_claims(self) -> None:
        with self.subTest("passing inactive report"):
            root = Path(self._testMethodName).resolve()
            root.mkdir(exist_ok=True)
            try:
                report_path = root / final_fit.FINAL_FIT_REPORT_FILENAME
                payload = self._report(root / "freeze.json")
                report_path.write_text(json.dumps(payload), encoding="utf-8")
                restored = final_fit.load_dynamic_crop_v3_final_fit_report(
                    report_path
                )
                self.assertFalse(restored["authority"]["production_ready"])
            finally:
                report_path.unlink(missing_ok=True)
                root.rmdir()

    def test_report_loader_rejects_blind_or_registry_authority(self) -> None:
        root = Path(f"{self._testMethodName}-tmp").resolve()
        root.mkdir(exist_ok=True)
        try:
            for key in (
                "production_registry_activation_authorized",
                "blind_holdout_quality_claim_authorized",
                "xoxz_dynamic_crop_v3_blind_evidence_present",
                "attitude_dynamic_crop_v3_blind_evidence_present",
                "human_parity_claim_authorized",
            ):
                payload = self._report(root / "freeze.json")
                payload["authority"][key] = True
                report_path = root / final_fit.FINAL_FIT_REPORT_FILENAME
                report_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(
                    final_fit.DynamicCropV3FinalFitError,
                    "authority is not inactive",
                ):
                    final_fit.load_dynamic_crop_v3_final_fit_report(report_path)
        finally:
            (root / final_fit.FINAL_FIT_REPORT_FILENAME).unlink(missing_ok=True)
            root.rmdir()

    def test_report_loader_rejects_prior_evidence_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            report_path = root / final_fit.FINAL_FIT_REPORT_FILENAME
            mutations = {
                "invariant digest": lambda payload: payload["training"][
                    "dynamic_crop_v3_prior_pretraining"
                ]["parameter_invariance"]["before"].__setitem__(
                    "base_parameter_tree_sha256", "f" * 64
                ),
                "reference song lineage": lambda payload: payload["training"][
                    "dynamic_crop_v3_prior_pretraining_history"
                ][0].__setitem__("song_id", SELECTION_SONG_IDS[1]),
                "prior learning rate": lambda payload: payload["training"][
                    "dynamic_crop_v3_prior_pretraining_history"
                ][0].__setitem__("actual_learning_rate", 9.0e-4),
            }
            for label, mutate in mutations.items():
                with self.subTest(label):
                    payload = self._report(root / "freeze.json")
                    mutate(payload)
                    report_path.write_text(
                        json.dumps(payload), encoding="utf-8"
                    )
                    with self.assertRaises(
                        final_fit.DynamicCropV3FinalFitError
                    ):
                        final_fit.load_dynamic_crop_v3_final_fit_report(
                            report_path
                        )

    def test_git_sha1_and_sha256_object_ids_are_accepted(self) -> None:
        self.assertTrue(final_fit._is_commit_oid("a" * 40))
        self.assertTrue(final_fit._is_commit_oid("b" * 64))
        self.assertFalse(final_fit._is_commit_oid("c" * 39))

    @unittest.skipUnless(mlx_runtime_version() is not None, "MLX is optional")
    def test_tiny_final_fit_uses_standard_v3_checkpoint_loader(self) -> None:
        examples = tuple(
            _training_example(song, f"{song}-final-ref-{index}")
            for song in SELECTION_SONG_IDS
            for index in range(5)
        )
        corpus = ReferenceTrainingCorpus(
            examples=examples,
            data_id="9" * 64,
            source_paths=(),
            statistics={
                "crop_coordinate_supervision_count": 60,
                "transition_supervision_count": 30,
            },
        )
        dynamic_corpus = _dynamic_crop_corpus_v3(examples)
        config = StageEditTransformerConfig.from_schema(
            DEFAULT_STAGE_EDIT_SCHEMA,
            model_dimension=16,
            num_heads=4,
            num_layers=1,
            feed_forward_dimension=32,
            max_sequence_length=8,
            max_sources=2,
            dropout=0.0,
        )
        base_loss = StageEditLossContract()
        dynamic_loss = DynamicCropV3LossContract(base=base_loss)
        tiny_schedule = dict(PROBE_TRAINING_SCHEDULE)
        tiny_schedule.update(
            {
                "requested_epochs": 1,
                "early_stopping_patience": 1,
                "batch_size": 2,
                "cut_pretraining_steps": 1,
                "cut_pretraining_window_length": 2,
                "cut_pretraining_warmup_steps": 0,
                "source_pretraining_steps": 1,
                "source_pretraining_window_length": 2,
                "source_pretraining_warmup_steps": 0,
                "dynamic_crop_v3_prior_pretraining_steps": 500,
                "dynamic_crop_v3_prior_pretraining_warmup_steps": 0,
                "dynamic_crop_v3_prior_retention_finetune_steps": 4,
                "finetune_steps_planned": 8,
                "finetune_warmup_steps": 0,
                "minimum_finetune_optimizer_steps": 1,
                "minimum_checkpoint_selection_finetune_steps": 1,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            freeze = root / "release-freeze.json"
            freeze.write_text("{}", encoding="utf-8")
            freeze_sha = hashlib.sha256(freeze.read_bytes()).hexdigest()
            checkpoint = root / "checkpoint"
            with (
                patch.object(final_fit, "PRODUCTION_VOLUME_ROOT", root),
                patch.object(final_fit, "PROBE_TRAINING_SCHEDULE", tiny_schedule),
            ):
                result = final_fit.train_dynamic_crop_v3_final_fit(
                    corpus,
                    dynamic_crop_corpus=dynamic_corpus,
                    checkpoint_directory=checkpoint,
                    model_config=config,
                    base_loss_contract=base_loss,
                    dynamic_crop_loss_contract=dynamic_loss,
                    frozen_training_schedule=tiny_schedule,
                    frozen_dynamic_crop_v3_prior_candidate_step=50,
                    frozen_cut_probability_threshold=0.5,
                    frozen_quality_thresholds={"cut_auroc_minimum": 0.5},
                    release_freeze_path=freeze,
                    release_freeze_sha256=freeze_sha,
                    source_commit="a" * 40,
                    device=mx.cpu,
                )
                restored = final_fit.load_dynamic_crop_v3_final_fit_report(
                    result.report_path
                )
            model, metadata = load_stage_edit_dynamic_crop_checkpoint_v3(
                checkpoint,
                expected_schema=DEFAULT_STAGE_EDIT_SCHEMA,
            )
            self.assertIsInstance(model, StageEditTransformerDynamicCropV3)
            self.assertEqual(metadata.checkpoint_id, result.checkpoint_id)
            self.assertEqual(
                metadata.weight_sha256, result.checkpoint_weight_sha256
            )
            self.assertEqual(510, result.optimizer_steps)
            self.assertFalse(restored["authority"]["production_ready"])
            self.assertFalse((checkpoint / "training-report.json").exists())


if __name__ == "__main__":
    unittest.main()
