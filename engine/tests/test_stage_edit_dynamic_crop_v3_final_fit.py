from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from crosscut import stage_edit_dynamic_crop_v3_final_fit as final_fit
from crosscut.stage_edit_dynamic_crop_capacity_probe import (
    PROBE_TRAINING_SCHEDULE,
    SELECTION_SONG_IDS,
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
        references_by_song = {
            song: [f"{song}-ref-{index}" for index in range(5)]
            for song in SELECTION_SONG_IDS
        }
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
                "cut_probability_threshold": 0.5,
                "quality_thresholds": {"cut_auroc_minimum": 0.5},
                "decisions_frozen_before_final_fit": True,
            },
            "training": {
                "optimizer_steps": 1120,
                "cut_pretraining_optimizer_steps": 500,
                "source_pretraining_optimizer_steps": 500,
                "finetune_optimizer_steps": 120,
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
                "training_steps": 1120,
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
            self.assertEqual(10, result.optimizer_steps)
            self.assertFalse(restored["authority"]["production_ready"])
            self.assertFalse((checkpoint / "training-report.json").exists())


if __name__ == "__main__":
    unittest.main()
