from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from crosscut import stage_edit_dynamic_crop_v3_final_release_launcher as release
from crosscut import stage_edit_dynamic_crop_v3_final_fit as final_fit
from crosscut.stage_edit_dynamic_crop_capacity_probe import SELECTION_SONG_IDS
from crosscut.stage_edit_dynamic_crop_v3_capacity_smoke import (
    V3_PROBE_TRAINING_SCHEDULE as PROBE_TRAINING_SCHEDULE,
)
from crosscut.stage_edit_dynamic_crop_head_v3 import (
    DynamicCropHeadV3Config,
    DynamicCropV3LossContract,
    StageEditTransformerDynamicCropV3,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v3,
    save_stage_edit_dynamic_crop_checkpoint_v3,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_CONTRACT,
    StageEditLossContract,
    StageEditTransformerConfig,
    mlx_runtime_version,
)

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None


class DynamicCropV3FinalReleaseLauncherTests(unittest.TestCase):
    def _candidate(self):
        return next(
            candidate
            for candidate in release._candidate_defaults_by_id().values()
            if candidate.candidate_name == release.SELECTED_CANDIDATE_NAME
        )

    def test_final_fit_corpus_loader_is_explicitly_group_scoped(self) -> None:
        candidate = self._candidate()
        reference_ids_by_song = {
            song: (f"{song}-ref",) for song in SELECTION_SONG_IDS
        }
        supervision_paths_by_song = {
            song: {f"{song}-ref": Path(f"/tmp/{song}-reference.json")}
            for song in SELECTION_SONG_IDS
        }
        snapshot_paths_by_song = {
            song: {f"{song}-ref": Path(f"/tmp/{song}-snapshot.json")}
            for song in SELECTION_SONG_IDS
        }
        preflight = SimpleNamespace(
            candidate_id=candidate.candidate_id,
            reference_ids_by_song=reference_ids_by_song,
            supervision_paths_by_song=supervision_paths_by_song,
            snapshot_paths_by_song=snapshot_paths_by_song,
            dynamic_manifest_paths_by_song={
                song: Path(f"/tmp/{song}-dynamic.json")
                for song in SELECTION_SONG_IDS
            },
            checkpoint_directory=Path("/tmp/final-fit-checkpoint"),
            frozen_dynamic_crop_v3_prior_candidate_step=200,
            frozen_cut_probability_threshold=0.5,
            quality_thresholds={},
            release_freeze_path=Path("/tmp/release-freeze.json"),
            release_freeze_file_sha256="f" * 64,
            source_commit="test-commit",
        )
        corpus = SimpleNamespace(examples=())
        expected = object()
        with (
            patch.object(release, "_snapshot_map", return_value={}),
            patch.object(
                release,
                "load_reference_training_corpus",
                return_value=corpus,
            ) as load_corpus,
            patch.object(
                release,
                "_dynamic_artifact_map",
                return_value={},
            ),
            patch.object(
                release,
                "build_stage_edit_dynamic_crop_training_corpus_v2",
                return_value=object(),
            ),
            patch.object(
                release,
                "build_stage_edit_dynamic_crop_training_corpus_v3",
                return_value=object(),
            ),
            patch.object(
                release,
                "train_dynamic_crop_v3_final_fit",
                return_value=expected,
            ),
        ):
            result = release._execute_final_fit(preflight)
        self.assertIs(expected, result)
        self.assertEqual(
            "group",
            load_corpus.call_args.kwargs["production_edit_mode"],
        )
        self.assertIsNone(
            load_corpus.call_args.kwargs["production_target_member"]
        )

    def _loso_fixture(self, root: Path):
        candidate = self._candidate()
        plan_path = root / "capacity-plan.json"
        plan_path.write_text("{}", encoding="utf-8")
        thresholds = (0.30, 0.55, 0.42)
        prior_candidate_steps = (50, 200, 200)
        receipt_payloads = {}
        receipt_paths = []
        reports = {}
        selections = {}
        for song, threshold, prior_candidate_step in zip(
            SELECTION_SONG_IDS,
            thresholds,
            prior_candidate_steps,
            strict=True,
        ):
            slug = song.lower()
            receipt_path = root / f"{slug}-receipt.json"
            report_path = root / f"{slug}-report.json"
            selection_path = root / f"{slug}-selection.json"
            metadata_path = root / f"{slug}-metadata.json"
            weights_path = root / f"{slug}-weights.safetensors"
            for path in (
                receipt_path,
                report_path,
                selection_path,
                metadata_path,
                weights_path,
            ):
                path.write_text("{}", encoding="utf-8")
            threshold_freeze = {
                "selected_probability_threshold": threshold,
                "selected_before_heldout_forward": True,
            }
            prior_digest = hashlib.sha256(
                f"{song}-{prior_candidate_step}".encode()
            ).hexdigest()
            reports[report_path.resolve()] = {
                "held_out_evaluation": {
                    "passed": True,
                    "quality_gate": {
                        "passed": True,
                        "thresholds": {"cut_auroc_minimum": 0.5},
                    },
                    "heldout_dynamic_crop_metrics_used_for_checkpoint_selection": False,
                },
                "split": {
                    "training_song_ids": sorted(
                        set(SELECTION_SONG_IDS) - {song}
                    ),
                    "held_out_song_ids": [song],
                    "held_out_used_for_model_selection": False,
                },
                "optimization": {
                    "unseen_song_used_for_model_selection": False,
                    "dynamic_crop_v3_prior_pretraining": {
                        "candidate_selection": {
                            "contract": (
                                release.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_SELECTION_CONTRACT
                            ),
                            "predeclared_candidate_steps": list(
                                release.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
                            ),
                            "retention_authorized": True,
                            "selected_candidate_step": prior_candidate_step,
                            "selected_candidate_value_bias_sha256": prior_digest,
                        }
                    },
                },
                "cut_threshold_freeze": threshold_freeze,
            }
            selections[selection_path.resolve()] = {
                "heldout_metrics_present": False,
                "heldout_used_for_model_or_threshold_selection": False,
                "cut_threshold_freeze": threshold_freeze,
                "selected_finetune_optimizer_step": 80,
                "dynamic_crop_v3_prior_selected_candidate_step": (
                    prior_candidate_step
                ),
                "dynamic_crop_v3_prior_selected_candidate_value_bias_sha256": (
                    prior_digest
                ),
            }
            receipt_payloads[receipt_path.resolve()] = {
                "candidate_id": candidate.candidate_id,
                "candidate_name": candidate.candidate_name,
                "validation_song_id": song,
                "training_song_ids": sorted(
                    set(SELECTION_SONG_IDS) - {song}
                ),
                "plan": release._pointer(plan_path, "capacity plan"),
                "production_training_report": release._pointer(
                    report_path, "fold report"
                ),
                "selection_freeze": release._pointer(
                    selection_path, "selection freeze"
                ),
                "checkpoint_metadata": release._pointer(
                    metadata_path, "metadata"
                ),
                "checkpoint_weights": release._pointer(
                    weights_path, "weights"
                ),
                "checkpoint_id": f"checkpoint-{slug}",
                "checkpoint_weight_sha256": "b" * 64,
            }
            receipt_paths.append(receipt_path.resolve())
        return (
            receipt_paths,
            receipt_payloads,
            reports,
            selections,
            plan_path,
        )

    def _load_loso(self, root: Path):
        (
            receipt_paths,
            receipt_payloads,
            reports,
            selections,
            plan_path,
        ) = self._loso_fixture(root)

        def load_receipt(path: Path):
            return receipt_payloads[Path(path).resolve()]

        def read_json(path: Path, _label: str):
            resolved = Path(path).resolve()
            if resolved in reports:
                return reports[resolved]
            return selections[resolved]

        mocks = (
            patch.object(
                release,
                "load_dynamic_crop_v3_capacity_fold_evidence",
                side_effect=load_receipt,
            ),
            patch.object(
                release,
                "_strict_capacity_plan",
                return_value={
                    "source_provenance": {"head_commit_oid": "a" * 40},
                    "candidates": [self._candidate().to_dict()],
                },
            ),
            patch.object(release, "_read_json", side_effect=read_json),
        )
        return receipt_paths, receipt_payloads, reports, selections, plan_path, mocks

    def _write_receipt_payload(
        self,
        evidence: Path,
        unsigned: dict,
    ) -> Path:
        material = dict(unsigned)
        material["artifact_content_sha256"] = None
        digest = release._canonical_sha256(material)
        payload = {**material, "artifact_content_sha256": digest}
        path = evidence / f"{release.V3_FINAL_FIT_RECEIPT_PREFIX}{digest}.json"
        path.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def _receipt_replay_fixture(self, root: Path):
        cut_steps = int(PROBE_TRAINING_SCHEDULE["cut_pretraining_steps"])
        source_steps = int(PROBE_TRAINING_SCHEDULE["source_pretraining_steps"])
        prior_steps = int(
            PROBE_TRAINING_SCHEDULE[
                "dynamic_crop_v3_prior_pretraining_steps"
            ]
        )
        finetune_steps = int(
            PROBE_TRAINING_SCHEDULE["finetune_steps_planned"]
        )
        retention_steps = int(
            PROBE_TRAINING_SCHEDULE[
                "dynamic_crop_v3_prior_retention_finetune_steps"
            ]
        )
        total_steps = cut_steps + source_steps + prior_steps + finetune_steps
        prior_candidate_steps = tuple(
            release.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
        )
        selected_prior_candidate_step = 200
        prior_candidate_digests = {
            step: hashlib.sha256(f"candidate-{step}".encode()).hexdigest()
            for step in prior_candidate_steps
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
            "context_value_bias_sha256": prior_candidate_digests[
                selected_prior_candidate_step
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
        checkpoint = root / "checkpoint"
        evidence = root / "evidence"
        evidence.mkdir()
        plan_path = root / "plan.json"
        freeze_path = root / "freeze.json"
        plan_path.write_text("{}", encoding="utf-8")
        freeze_path.write_text("{}", encoding="utf-8")
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
        model = StageEditTransformerDynamicCropV3(config)
        mx.eval(model.parameters())
        metadata = new_stage_edit_dynamic_crop_checkpoint_metadata_v3(
            "synthetic-final-fit",
            DEFAULT_STAGE_EDIT_SCHEMA,
            config,
            training_steps=total_steps,
            weights_trained=True,
            training_data_id="synthetic-final-fit-data",
        )
        metadata = save_stage_edit_dynamic_crop_checkpoint_v3(
            checkpoint, model, metadata
        )
        source_commit = "a" * 40
        quality_thresholds = {"cut_auroc_minimum": 0.5}
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
        activation = {
            "contract": DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_CONTRACT,
            "enabled": True,
            "scope": "base_transformer_encoder_all_layers",
            "implementation": "mlx.nn.TransformerEncoder.checkpoint",
            "training_only": True,
            "serialized_parameter_tree_changed": False,
            "inference_runtime_requires_checkpointing": False,
            "previous_value": False,
        }
        report = {
            "schema_version": final_fit.FINAL_FIT_REPORT_SCHEMA_VERSION,
            "run_kind": final_fit.FINAL_FIT_RUN_KIND,
            "contract": final_fit.FINAL_FIT_CONTRACT,
            "created_at": "2026-08-24T00:00:00+00:00",
            "source_commit": source_commit,
            "release_freeze": release._pointer(freeze_path, "release freeze"),
            "dynamic_crop_version": 3,
            "model_contract": release.DYNAMIC_CROP_V3_MODEL_CONTRACT,
            "target_contract": release.DYNAMIC_CROP_V3_TARGET_CONTRACT,
            "output_contract": release.DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
            "loss_contract": release.DYNAMIC_CROP_V3_LOSS_CONTRACT,
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
                "candidate_config": config.to_dict(),
                "base_loss_contract": asdict(base_loss),
                "dynamic_crop_loss_contract": asdict(dynamic_loss),
                "training_schedule": dict(PROBE_TRAINING_SCHEDULE),
                "dynamic_crop_v3_prior_candidate_step": (
                    selected_prior_candidate_step
                ),
                "cut_probability_threshold": 0.5,
                "quality_thresholds": quality_thresholds,
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
                    "predeclared_candidate_steps": list(prior_candidate_steps),
                    "candidate_selection_contract": (
                        release.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_SELECTION_CONTRACT
                    ),
                    "candidate_snapshots": [
                        {
                            "candidate_step": step,
                            "value_bias_sha256": prior_candidate_digests[step],
                            "training_only_updates": True,
                        }
                        for step in prior_candidate_steps
                    ],
                    "frozen_selected_candidate_step": (
                        selected_prior_candidate_step
                    ),
                    "frozen_selected_candidate_value_bias_sha256": (
                        prior_candidate_digests[selected_prior_candidate_step]
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
                            step in prior_candidate_steps
                        ),
                        "candidate_value_bias_sha256": (
                            prior_candidate_digests[step]
                            if step in prior_candidate_steps
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
                "activation_checkpointing": activation,
            },
            "checkpoint": {
                "directory": str(checkpoint),
                "metadata_filename": release.DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
                "weight_filename": release.DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
                "checkpoint_id": metadata.checkpoint_id,
                "training_steps": metadata.training_steps,
                "weight_sha256": metadata.weight_sha256,
                "model_config": config.to_dict(),
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
        report_path = checkpoint / final_fit.FINAL_FIT_REPORT_FILENAME
        report_path.write_text(json.dumps(report), encoding="utf-8")
        candidate_id = "c" * 64
        candidate = SimpleNamespace(
            candidate_id=candidate_id,
            candidate_name=release.SELECTED_CANDIDATE_NAME,
            model_config=config,
            base_loss_contract=base_loss,
            dynamic_crop_config=DynamicCropHeadV3Config(),
            dynamic_crop_loss_contract=dynamic_loss,
        )
        plan = {
            "source_provenance": {"head_commit_oid": source_commit},
            "checkpoint_directory": str(checkpoint),
            "evidence_directory": str(evidence),
            "release_freeze": release._pointer(freeze_path, "release freeze"),
            "candidate_id": candidate_id,
            "songs": [
                {
                    "song_id": song,
                    "reference_ids": list(references_by_song[song]),
                }
                for song in SELECTION_SONG_IDS
            ],
            "training_schedule": dict(PROBE_TRAINING_SCHEDULE),
            "frozen_dynamic_crop_v3_prior_candidate_step": (
                selected_prior_candidate_step
            ),
            "frozen_cut_probability_threshold": 0.5,
            "quality_thresholds": quality_thresholds,
        }
        unsigned = {
            "schema_version": release.V3_FINAL_FIT_RECEIPT_SCHEMA_VERSION,
            "analysis_basis": release.V3_FINAL_FIT_RECEIPT_ANALYSIS_BASIS,
            "contract": release.V3_FINAL_FIT_RECEIPT_CONTRACT,
            "candidate_name": release.SELECTED_CANDIDATE_NAME,
            "candidate_id": candidate_id,
            "source_commit": source_commit,
            "dynamic_crop_version": 3,
            "model_contract": release.DYNAMIC_CROP_V3_MODEL_CONTRACT,
            "target_contract": release.DYNAMIC_CROP_V3_TARGET_CONTRACT,
            "output_contract": release.DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
            "loss_contract": release.DYNAMIC_CROP_V3_LOSS_CONTRACT,
            "checkpoint_id": metadata.checkpoint_id,
            "checkpoint_weight_sha256": metadata.weight_sha256,
            "training_steps": metadata.training_steps,
            "reference_count": 15,
            "plan": release._pointer(plan_path, "final-fit plan"),
            "release_freeze": release._pointer(freeze_path, "release freeze"),
            "final_fit_report": release._pointer(report_path, "final-fit report"),
            "checkpoint_metadata": release._pointer(
                checkpoint / release.DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
                "checkpoint metadata",
            ),
            "checkpoint_weights": release._pointer(
                checkpoint / release.DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
                "checkpoint weights",
            ),
            "production_activation_authorized": False,
            "production_registry_activation_requested": False,
            "learned_source_pair_authority_present": False,
            "blind_holdout_quality_claim_authorized": False,
            "xoxz_dynamic_crop_v3_blind_evidence_present": False,
            "attitude_dynamic_crop_v3_blind_evidence_present": False,
            "human_parity_claim_authorized": False,
            "experimental_unseen_ive_inference_only": True,
            "artifact_content_sha256": None,
        }
        receipt_path = self._write_receipt_payload(evidence, unsigned)
        return SimpleNamespace(
            checkpoint=checkpoint,
            evidence=evidence,
            plan_path=plan_path,
            freeze_path=freeze_path,
            report_path=report_path,
            plan=plan,
            report=report,
            candidate=candidate,
            unsigned=unsigned,
            receipt_path=receipt_path,
        )

    def test_three_same_candidate_receipts_freeze_median_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_paths, _payloads, _reports, _selections, plan, mocks = (
                self._load_loso(root)
            )
            with mocks[0], mocks[1], mocks[2]:
                result = release._load_complete_same_candidate_loso(
                    receipt_paths
                )
            self.assertEqual(
                result["candidate"].candidate_name,
                release.SELECTED_CANDIDATE_NAME,
            )
            self.assertEqual(result["capacity_plan_path"], plan.resolve())
            self.assertEqual(result["frozen_cut_probability_threshold"], 0.42)
            self.assertEqual(
                result["fold_dynamic_crop_v3_prior_selected_candidate_steps"],
                [50, 200, 200],
            )
            self.assertEqual(
                result["frozen_dynamic_crop_v3_prior_candidate_step"], 200
            )
            self.assertEqual(
                {fold["validation_song_id"] for fold in result["folds"]},
                set(SELECTION_SONG_IDS),
            )
            self.assertTrue(all(fold["heldout_gate_passed"] for fold in result["folds"]))

    def test_loso_admission_requires_exactly_three_distinct_heldouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_paths, payloads, _reports, _selections, _plan, mocks = (
                self._load_loso(root)
            )
            payloads[receipt_paths[-1]]["validation_song_id"] = SELECTION_SONG_IDS[0]
            with mocks[0], mocks[1], mocks[2], self.assertRaisesRegex(
                release.DynamicCropV3FinalReleaseError,
                "cover every fixed held-out song",
            ):
                release._load_complete_same_candidate_loso(receipt_paths)

    def test_loso_admission_rejects_any_heldout_selection_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_paths, _payloads, _reports, selections, _plan, mocks = (
                self._load_loso(root)
            )
            first_selection = next(iter(selections.values()))
            first_selection[
                "heldout_used_for_model_or_threshold_selection"
            ] = True
            with mocks[0], mocks[1], mocks[2], self.assertRaisesRegex(
                release.DynamicCropV3FinalReleaseError,
                "influenced model or threshold selection",
            ):
                release._load_complete_same_candidate_loso(receipt_paths)

    def test_loso_admission_rejects_candidate_mixing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_paths, payloads, _reports, _selections, _plan, mocks = (
                self._load_loso(root)
            )
            payloads[receipt_paths[-1]]["candidate_id"] = "c" * 64
            payloads[receipt_paths[-1]]["candidate_name"] = "compact-balanced"
            with mocks[0], mocks[1], mocks[2], self.assertRaisesRegex(
                release.DynamicCropV3FinalReleaseError,
                "same production-balanced candidate",
            ):
                release._load_complete_same_candidate_loso(receipt_paths)

    def test_capacity_plan_must_predeclare_only_selected_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_paths, _payloads, _reports, _selections, _plan, mocks = (
                self._load_loso(root)
            )
            candidates = list(release._candidate_defaults_by_id().values())
            selected = self._candidate()
            other = next(
                candidate
                for candidate in candidates
                if candidate.candidate_id != selected.candidate_id
            )
            with (
                mocks[0],
                mocks[2],
                patch.object(
                    release,
                    "_strict_capacity_plan",
                    return_value={
                        "source_provenance": {"head_commit_oid": "a" * 40},
                        "candidates": [selected.to_dict(), other.to_dict()],
                    },
                ),
                self.assertRaisesRegex(
                    release.DynamicCropV3FinalReleaseError,
                    "predeclare only production-balanced",
                ),
            ):
                release._load_complete_same_candidate_loso(receipt_paths)

    def test_final_fit_plan_binds_freeze_manifests_and_no_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._candidate()
            freeze_path = root / "freeze.json"
            capacity_plan_path = root / "capacity-plan.json"
            freeze_path.write_text("{}", encoding="utf-8")
            capacity_plan_path.write_text("{}", encoding="utf-8")
            source = {
                "repository_root": str(root / "repo"),
                "head_commit_oid": "a" * 40,
            }
            authority = {
                "production_activation_authorized": False,
                "production_registry_activation_requested": False,
                "learned_source_pair_authority_present": False,
                "blind_holdout_quality_claim_authorized": False,
                "xoxz_dynamic_crop_v3_blind_evidence_present": False,
                "attitude_dynamic_crop_v3_blind_evidence_present": False,
                "human_parity_claim_authorized": False,
                "experimental_unseen_ive_inference_only": True,
            }
            freeze = {
                "source_provenance": source,
                "freeze_content_sha256": "d" * 64,
                "candidate_name": candidate.candidate_name,
                "candidate_id": candidate.candidate_id,
                "candidate_contract": candidate.contract_dict(),
                "training_schedule": dict(PROBE_TRAINING_SCHEDULE),
                "frozen_dynamic_crop_v3_prior_candidate_step": 200,
                "frozen_cut_probability_threshold": 0.42,
                "quality_thresholds": {"cut_auroc_minimum": 0.5},
                "professional_montage_profile": {
                    "path": str(root / "profile.json"),
                    "file_sha256": "e" * 64,
                },
                "professional_montage_profile_identity": {
                    "schema_version": 2
                },
                "capacity_plan": release._pointer(
                    capacity_plan_path, "capacity plan"
                ),
                "authority": authority,
            }
            songs = [{"song_id": song} for song in SELECTION_SONG_IDS]
            capacity_plan = {
                "songs": songs,
                "selection_dataset_manifest": {
                    "path": str(root / "selection.json"),
                    "file_sha256": "f" * 64,
                },
                "selection_dataset_content_sha256": "1" * 64,
            }
            output = root / "plans"
            with (
                patch.object(release, "_strict_release_freeze", return_value=freeze),
                patch.object(
                    release,
                    "_under_production_volume",
                    side_effect=lambda path, _label: Path(path).resolve(),
                ),
                patch.object(
                    release,
                    "_outside_clean_repository",
                    side_effect=lambda path, _label: Path(path).resolve(),
                ),
                patch.object(
                    release,
                    "_strict_capacity_plan",
                    return_value=capacity_plan,
                ),
                patch.object(release, "_require_same_source_provenance"),
                patch.object(
                    release.capacity.v2_launcher,
                    "_validate_artifact_path_isolation",
                ),
            ):
                plan_path = release.create_dynamic_crop_v3_final_fit_plan(
                    release_freeze_path=freeze_path,
                    checkpoint_directory=root / "checkpoint",
                    evidence_directory=root / "evidence",
                    output_directory=output,
                )
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["candidate_id"], candidate.candidate_id)
            self.assertEqual(payload["songs"], songs)
            self.assertEqual(payload["training_schedule"], dict(PROBE_TRAINING_SCHEDULE))
            self.assertEqual(payload["authority"], authority)
            self.assertTrue(payload["all_reference_fit"]["all_references_consumed_by_optimizer"])
            self.assertNotIn("source_pair_scorer", json.dumps(payload))

    def test_content_addressed_plan_tamper_fails_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / f"{release.V3_FINAL_FIT_PLAN_PREFIX}{'a' * 64}.json"
            path.write_text(json.dumps({"candidate_name": "tampered"}), encoding="utf-8")
            with self.assertRaises(
                release.DynamicCropV3FinalReleaseError
            ):
                release._strict_final_fit_plan(path)

    def test_cli_exposes_no_activation_or_source_pair_flags(self) -> None:
        help_text = release.build_parser().format_help()
        self.assertNotIn("authorize-production-activation", help_text)
        self.assertNotIn("production-registry", help_text)
        self.assertNotIn("source-pair", help_text)

    @unittest.skipUnless(mlx_runtime_version() is not None, "MLX is optional")
    def test_receipt_loader_replays_checkpoint_through_standard_v3_loader(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._receipt_replay_fixture(Path(directory).resolve())
            with (
                patch.object(
                    release,
                    "_candidate_defaults_by_id",
                    return_value={
                        fixture.candidate.candidate_id: fixture.candidate
                    },
                ),
                patch.object(
                    release,
                    "_strict_final_fit_plan",
                    return_value=fixture.plan,
                ),
            ):
                restored = release.load_dynamic_crop_v3_final_fit_receipt(
                    fixture.receipt_path
                )
            self.assertEqual(
                restored["checkpoint_id"], "synthetic-final-fit"
            )

    @unittest.skipUnless(mlx_runtime_version() is not None, "MLX is optional")
    def test_receipt_rejects_rehashed_forged_report_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._receipt_replay_fixture(Path(directory).resolve())
            forged_report = json.loads(json.dumps(fixture.report))
            forged_report["frozen_decisions"]["candidate_config"][
                "model_dimension"
            ] = 32
            fixture.report_path.write_text(
                json.dumps(forged_report), encoding="utf-8"
            )
            forged_unsigned = dict(fixture.unsigned)
            forged_unsigned["final_fit_report"] = release._pointer(
                fixture.report_path, "forged final-fit report"
            )
            forged_receipt = self._write_receipt_payload(
                fixture.evidence, forged_unsigned
            )
            with (
                patch.object(
                    release,
                    "_candidate_defaults_by_id",
                    return_value={
                        fixture.candidate.candidate_id: fixture.candidate
                    },
                ),
                patch.object(
                    release,
                    "_strict_final_fit_plan",
                    return_value=fixture.plan,
                ),
                self.assertRaisesRegex(
                    release.DynamicCropV3FinalReleaseError,
                    "checkpoint identity differs",
                ),
            ):
                release.load_dynamic_crop_v3_final_fit_receipt(forged_receipt)

    @unittest.skipUnless(mlx_runtime_version() is not None, "MLX is optional")
    def test_receipt_rejects_rehashed_forged_report_reference_scope(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._receipt_replay_fixture(Path(directory).resolve())
            forged_report = json.loads(json.dumps(fixture.report))
            forged_report["all_reference_fit"]["references_by_song"][
                SELECTION_SONG_IDS[0]
            ][0] = "forged-reference-id"
            fixture.report_path.write_text(
                json.dumps(forged_report), encoding="utf-8"
            )
            forged_unsigned = dict(fixture.unsigned)
            forged_unsigned["final_fit_report"] = release._pointer(
                fixture.report_path, "forged final-fit report"
            )
            forged_receipt = self._write_receipt_payload(
                fixture.evidence, forged_unsigned
            )
            with (
                patch.object(
                    release,
                    "_candidate_defaults_by_id",
                    return_value={
                        fixture.candidate.candidate_id: fixture.candidate
                    },
                ),
                patch.object(
                    release,
                    "_strict_final_fit_plan",
                    return_value=fixture.plan,
                ),
                self.assertRaisesRegex(
                    release.DynamicCropV3FinalReleaseError,
                    "checkpoint identity differs",
                ),
            ):
                release.load_dynamic_crop_v3_final_fit_receipt(forged_receipt)

    @unittest.skipUnless(mlx_runtime_version() is not None, "MLX is optional")
    def test_receipt_rejects_replayed_nonstandard_weight_filename(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._receipt_replay_fixture(Path(directory).resolve())
            standard_weight = (
                fixture.checkpoint
                / release.DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
            )
            alternate_weight = fixture.checkpoint / "replayed-weights.safetensors"
            alternate_weight.write_bytes(standard_weight.read_bytes())
            metadata_path = (
                fixture.checkpoint
                / release.DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
            )
            replayed_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            replayed_metadata["weight_filename"] = alternate_weight.name
            metadata_path.write_text(
                json.dumps(replayed_metadata), encoding="utf-8"
            )
            forged_unsigned = dict(fixture.unsigned)
            forged_unsigned["checkpoint_metadata"] = release._pointer(
                metadata_path, "replayed checkpoint metadata"
            )
            forged_receipt = self._write_receipt_payload(
                fixture.evidence, forged_unsigned
            )
            with (
                patch.object(
                    release,
                    "_candidate_defaults_by_id",
                    return_value={
                        fixture.candidate.candidate_id: fixture.candidate
                    },
                ),
                patch.object(
                    release,
                    "_strict_final_fit_plan",
                    return_value=fixture.plan,
                ),
                self.assertRaisesRegex(
                    release.DynamicCropV3FinalReleaseError,
                    "checkpoint identity differs",
                ),
            ):
                release.load_dynamic_crop_v3_final_fit_receipt(forged_receipt)

    @unittest.skipUnless(mlx_runtime_version() is not None, "MLX is optional")
    def test_receipt_pointers_reject_report_freeze_metadata_and_weight_tampering(
        self,
    ) -> None:
        mutations = (
            ("report_path", "text"),
            ("freeze_path", "text"),
            ("metadata", "text"),
            ("weights", "bytes"),
        )
        for attribute, mode in mutations:
            with self.subTest(attribute), tempfile.TemporaryDirectory() as directory:
                fixture = self._receipt_replay_fixture(Path(directory).resolve())
                path = (
                    fixture.checkpoint
                    / release.DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
                    if attribute == "weights"
                    else fixture.checkpoint
                    / release.DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
                    if attribute == "metadata"
                    else Path(getattr(fixture, attribute))
                )
                if mode == "bytes":
                    path.write_bytes(path.read_bytes() + b"tamper")
                else:
                    path.write_text(
                        path.read_text(encoding="utf-8") + "\n",
                        encoding="utf-8",
                    )
                with (
                    patch.object(
                        release,
                        "_candidate_defaults_by_id",
                        return_value={
                            fixture.candidate.candidate_id: fixture.candidate
                        },
                    ),
                    patch.object(
                        release,
                        "_strict_final_fit_plan",
                        return_value=fixture.plan,
                    ),
                    self.assertRaisesRegex(
                        release.DynamicCropV3FinalReleaseError,
                        "SHA-256 mismatch",
                    ),
                ):
                    release.load_dynamic_crop_v3_final_fit_receipt(
                        fixture.receipt_path
                    )

    @unittest.skipUnless(mlx_runtime_version() is not None, "MLX is optional")
    def test_rehashed_receipt_cannot_add_activation_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._receipt_replay_fixture(Path(directory).resolve())
            forged = dict(fixture.unsigned)
            forged["production_activation_authorized"] = True
            forged_path = self._write_receipt_payload(fixture.evidence, forged)
            with self.assertRaisesRegex(
                release.DynamicCropV3FinalReleaseError,
                "contract or authority differs",
            ):
                release.load_dynamic_crop_v3_final_fit_receipt(forged_path)


if __name__ == "__main__":
    unittest.main()
