from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, patch

from crosscut.stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
)
from crosscut.stage_edit_dynamic_crop_head_v3 import DynamicCropV3LossContract
from crosscut.stage_edit_feature_materializer import (
    FEATURE_MATERIALIZER_LEGACY_CONTRACT_V3,
    FEATURE_SNAPSHOT_LEGACY_SCHEMA_VERSION,
    SourceFeatureObservation,
    StageEditFeatureFrame,
    _payload_sha256,
    materialize_stage_edit_feature_snapshot,
)
from crosscut.stage_edit_production_train_cli import (
    _dynamic_artifact_map,
    _snapshot_map,
    _training_result_exit_code,
    build_parser,
    main,
)


class StageEditProductionTrainCliTests(unittest.TestCase):
    def test_legacy_schema3_snapshot_fails_production_snapshot_gate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "legacy-schema3-feature-snapshot.json"
            snapshot = materialize_stage_edit_feature_snapshot(
                "ive:legacy-schema3-gate-probe",
                (
                    StageEditFeatureFrame(
                        0.5,
                        (SourceFeatureObservation("source-a"),),
                    ),
                ),
                performance_end_seconds=1.0,
                inspection_only_without_causal_sensing=True,
            )
            legacy = deepcopy(snapshot.to_dict())
            legacy["schema_version"] = FEATURE_SNAPSHOT_LEGACY_SCHEMA_VERSION
            legacy["provenance"]["producer"] = (
                FEATURE_MATERIALIZER_LEGACY_CONTRACT_V3
            )
            legacy.pop("causal_sensing_binding")
            legacy["input_policy"] = {
                "training_inference_materializer_shared": True,
                "reference_attribution_match_inputs_present": False,
                "selected_labels_present": False,
                "missing_values_are_explicitly_masked": True,
                "source_availability_and_hard_gate_are_distinct": True,
                "temporal_pose_sidecar_hash_bound": True,
                "temporal_pose_sidecar_enters_transformer_tensors": False,
                "legacy_snapshots_cannot_claim_pose_direction": True,
            }
            unsigned = deepcopy(legacy)
            unsigned.pop("snapshot_sha256")
            legacy["snapshot_sha256"] = _payload_sha256(unsigned)
            path.write_text(json.dumps(legacy), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "inspection-only"):
                _snapshot_map((f"reference-a={path}",))

    def test_evaluation_only_exit_code_tracks_the_heldout_gate(self) -> None:
        self.assertEqual(
            0,
            _training_result_exit_code(
                activation_authorized=False,
                production_ready=False,
                quality_gate_passed=True,
            ),
        )
        self.assertEqual(
            2,
            _training_result_exit_code(
                activation_authorized=False,
                production_ready=False,
                quality_gate_passed=False,
            ),
        )

    def test_authorized_exit_code_requires_production_ready(self) -> None:
        self.assertEqual(
            2,
            _training_result_exit_code(
                activation_authorized=True,
                production_ready=False,
                quality_gate_passed=True,
            ),
        )
        self.assertEqual(
            0,
            _training_result_exit_code(
                activation_authorized=True,
                production_ready=True,
                quality_gate_passed=True,
            ),
        )

    def test_complete_dynamic_crop_manifest_expands_to_reference_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "shot-0.json"
            second = root / "shot-1.json"
            first.write_text("{}", encoding="utf-8")
            second.write_text("{}", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "reference_id": "reference-a",
                                "artifact": {
                                    "path": str(first),
                                    "file_sha256": "a" * 64,
                                    "content_sha256": "b" * 64,
                                },
                            },
                            {
                                "reference_id": "reference-b",
                                "artifact": {
                                    "path": str(second),
                                    "file_sha256": "a" * 64,
                                    "content_sha256": "b" * 64,
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "validate_reference_dynamic_crop_manifest"
                ) as validate,
                patch(
                    "crosscut.stage_edit_production_train_cli._file_sha256",
                    return_value="a" * 64,
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "load_reference_dynamic_crop_supervision",
                    return_value=SimpleNamespace(
                        payload={
                            "provenance": {
                                "artifact_content_sha256": "b" * 64
                            }
                        }
                    ),
                ),
            ):
                result = _dynamic_artifact_map((), (manifest,))

            validate.assert_called_once_with(
                ANY,
                verify_artifact_files=False,
            )
            self.assertEqual((first.resolve(),), result["reference-a"])
            self.assertEqual((second.resolve(),), result["reference-b"])

    def test_manifest_and_explicit_duplicate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifact = root / "shot.json"
            artifact.write_text("{}", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "reference_id": "reference-a",
                                "artifact": {"path": str(artifact)},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "validate_reference_dynamic_crop_manifest"
                ),
                self.assertRaisesRegex(ValueError, "duplicate dynamic crop"),
            ):
                _dynamic_artifact_map(
                    (f"reference-a={artifact}",),
                    (manifest,),
                )

    def test_parser_accepts_manifest_without_per_shot_arguments(self) -> None:
        parsed = build_parser().parse_args(
            [
                "--input",
                "/tmp/input.json",
                "--song-id",
                "reference-a=IVE-XOXZ",
                "--held-out-song",
                "IVE-XOXZ",
                "--feature-snapshot",
                "reference-a=/tmp/features.json",
                "--dynamic-crop-manifest",
                "/tmp/crop-manifest.json",
                "--checkpoint-dir",
                "/tmp/checkpoint",
            ]
        )

        self.assertEqual([], parsed.dynamic_crop_artifact)
        self.assertEqual(
            [Path("/tmp/crop-manifest.json")],
            parsed.dynamic_crop_manifest,
        )
        self.assertIsNone(parsed.professional_montage_grammar_profile)
        self.assertEqual(2, parsed.dynamic_crop_version)
        self.assertIsNone(parsed.continuity_authority)
        self.assertIsNone(parsed.source_pair_scorer)
        self.assertIsNone(parsed.source_pair_production_report)

    def test_v3_is_explicitly_promoted_and_routed_to_the_v3_loss_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            checkpoint = root / "v3-checkpoint"
            base_corpus = SimpleNamespace(examples=())
            crop_v2_corpus = SimpleNamespace(name="crop-v2")
            crop_v3_corpus = SimpleNamespace(name="crop-v3")
            quality_gate = SimpleNamespace(passed=True, to_dict=lambda: {})
            training_result = SimpleNamespace(
                checkpoint_directory=checkpoint,
                report_path=checkpoint / "training-report.json",
                best_epoch=1,
                best_training_step=120,
                production_ready=False,
                quality_gate=quality_gate,
            )
            with (
                patch(
                    "crosscut.stage_edit_production_train_cli._snapshot_map",
                    return_value={},
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "load_reference_training_corpus",
                    return_value=base_corpus,
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "_dynamic_artifact_map",
                    return_value={},
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "build_stage_edit_dynamic_crop_training_corpus_v2",
                    return_value=crop_v2_corpus,
                ) as build_v2,
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "build_stage_edit_dynamic_crop_training_corpus_v3",
                    return_value=crop_v3_corpus,
                ) as build_v3,
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "train_production_stage_edit",
                    return_value=training_result,
                ) as train,
                patch("builtins.print"),
            ):
                exit_code = main(
                    [
                        "--input",
                        str(root / "input.json"),
                        "--song-id",
                        "reference-a=IVE-TEST",
                        "--held-out-song",
                        "IVE-TEST",
                        "--feature-snapshot",
                        f"reference-a={root / 'features.json'}",
                        "--dynamic-crop-manifest",
                        str(root / "crop-manifest.json"),
                        "--dynamic-crop-version",
                        "3",
                        "--dynamic-crop-v3-prior-pretrain-steps",
                        "500",
                        "--dynamic-crop-v3-prior-pretrain-learning-rate",
                        "0.0003",
                        "--dynamic-crop-v3-prior-pretrain-warmup-steps",
                        "50",
                        "--checkpoint-dir",
                        str(checkpoint),
                    ]
                )

            self.assertEqual(0, exit_code)
            build_v2.assert_called_once_with(
                base_corpus.examples,
                artifact_paths_by_reference={},
            )
            build_v3.assert_called_once_with(crop_v2_corpus)
            kwargs = train.call_args.kwargs
            self.assertIs(crop_v3_corpus, kwargs["dynamic_crop_corpus"])
            self.assertNotIn("dynamic_crop_loss_contract", kwargs)
            self.assertIsInstance(
                kwargs["dynamic_crop_v3_loss_contract"],
                DynamicCropV3LossContract,
            )
            self.assertEqual(500, kwargs["source_pretrain_steps"])
            self.assertEqual(32, kwargs["source_pretrain_window_length"])
            self.assertEqual(
                500, kwargs["dynamic_crop_v3_prior_pretrain_steps"]
            )
            self.assertEqual(
                3.0e-4,
                kwargs["dynamic_crop_v3_prior_pretrain_learning_rate"],
            )
            self.assertEqual(
                50, kwargs["dynamic_crop_v3_prior_pretrain_warmup_steps"]
            )
            self.assertEqual(120, kwargs["minimum_finetune_optimizer_steps"])
            self.assertEqual(
                60,
                kwargs["minimum_checkpoint_selection_finetune_steps"],
            )

    def test_authorized_activation_requires_explicit_montage_profile(self) -> None:
        base = [
            "--input",
            "/tmp/input.json",
            "--song-id",
            "reference-a=IVE-XOXZ",
            "--held-out-song",
            "IVE-XOXZ",
            "--feature-snapshot",
            "reference-a=/tmp/features.json",
            "--dynamic-crop-manifest",
            "/tmp/crop-manifest.json",
            "--checkpoint-dir",
            "/tmp/new-checkpoint",
        ]
        with (
            patch(
                "crosscut.stage_edit_production_train_cli."
                "load_reference_training_corpus"
            ) as load,
            self.assertRaisesRegex(
                ValueError,
                "requires --professional-montage-grammar-profile",
            ),
        ):
            main([*base, "--authorize-production-activation"])
        load.assert_not_called()

        with (
            patch(
                "crosscut.stage_edit_production_train_cli."
                "load_reference_training_corpus"
            ) as load,
            self.assertRaisesRegex(
                ValueError,
                "requires --authorize-production-activation",
            ),
        ):
            main(
                [
                    *base,
                    "--professional-montage-grammar-profile",
                    "/tmp/montage-profile.json",
                ]
            )
        load.assert_not_called()

        with (
            patch(
                "crosscut.stage_edit_production_train_cli."
                "load_reference_training_corpus"
            ) as load,
            self.assertRaisesRegex(ValueError, "explicit absolute path"),
        ):
            main(
                [
                    *base,
                    "--authorize-production-activation",
                    "--professional-montage-grammar-profile",
                    "relative-profile.json",
                ]
            )
        load.assert_not_called()

        parsed = build_parser().parse_args(
            [
                *base,
                "--authorize-production-activation",
                "--professional-montage-grammar-profile",
                "/tmp/montage-profile.json",
            ]
        )
        self.assertEqual(
            Path("/tmp/montage-profile.json"),
            parsed.professional_montage_grammar_profile,
        )

    def test_authorized_registration_receives_the_exact_profile_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            checkpoint = root / "checkpoint"
            profile = root / "montage-profile.json"
            registry = root / "active-production-bundle.json"
            quality_gate = SimpleNamespace(passed=True, to_dict=lambda: {})
            training_result = SimpleNamespace(
                checkpoint_directory=checkpoint,
                report_path=checkpoint / "training-report.json",
                best_epoch=3,
                best_training_step=9,
                production_ready=True,
                quality_gate=quality_gate,
            )
            with (
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "load_professional_montage_policy",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli._snapshot_map",
                    return_value={},
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "load_reference_training_corpus",
                    return_value=SimpleNamespace(examples=()),
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "_dynamic_artifact_map",
                    return_value={},
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "build_stage_edit_dynamic_crop_training_corpus_v2",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "train_production_stage_edit",
                    return_value=training_result,
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "register_stage_edit_production_bundle",
                    return_value=registry,
                ) as register,
                patch("builtins.print"),
            ):
                exit_code = main(
                    [
                        "--input",
                        str(root / "input.json"),
                        "--song-id",
                        "reference-a=IVE-XOXZ",
                        "--held-out-song",
                        "IVE-XOXZ",
                        "--feature-snapshot",
                        f"reference-a={root / 'features.json'}",
                        "--dynamic-crop-manifest",
                        str(root / "crop-manifest.json"),
                        "--checkpoint-dir",
                        str(checkpoint),
                        "--authorize-production-activation",
                        "--professional-montage-grammar-profile",
                        str(profile),
                        "--production-registry",
                        str(registry),
                    ]
                )

            self.assertEqual(0, exit_code)
            register.assert_called_once_with(
                checkpoint,
                professional_montage_grammar_profile_path=profile,
                registry_path=registry,
                dynamic_crop_version=2,
                continuity_authority=None,
                source_pair_scorer_path=None,
                source_pair_production_report_path=None,
            )

    def test_v3_activation_requires_one_explicit_continuity_authority(self) -> None:
        base = [
            "--input",
            "/tmp/input.json",
            "--song-id",
            "reference-a=IVE-XOXZ",
            "--held-out-song",
            "IVE-XOXZ",
            "--feature-snapshot",
            "reference-a=/tmp/features.json",
            "--dynamic-crop-manifest",
            "/tmp/crop-manifest.json",
            "--dynamic-crop-version",
            "3",
            "--checkpoint-dir",
            "/tmp/new-v3-checkpoint",
            "--authorize-production-activation",
            "--professional-montage-grammar-profile",
            "/tmp/montage-profile.json",
        ]
        with (
            patch(
                "crosscut.stage_edit_production_train_cli."
                "load_reference_training_corpus"
            ) as load,
            self.assertRaisesRegex(ValueError, "requires --continuity-authority"),
        ):
            main(base)
        load.assert_not_called()

        with (
            patch(
                "crosscut.stage_edit_production_train_cli."
                "load_reference_training_corpus"
            ) as load,
            self.assertRaisesRegex(ValueError, "requires both --source-pair"),
        ):
            main(
                [
                    *base,
                    "--continuity-authority",
                    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
                ]
            )
        load.assert_not_called()

        with (
            patch(
                "crosscut.stage_edit_production_train_cli."
                "load_reference_training_corpus"
            ) as load,
            self.assertRaisesRegex(ValueError, "rejects source-pair"),
        ):
            main(
                [
                    *base,
                    "--continuity-authority",
                    HANDCRAFTED_CONTINUITY_AUTHORITY,
                    "--source-pair-scorer",
                    "/tmp/source-pair.json",
                ]
            )
        load.assert_not_called()

    def test_v3_learned_activation_routes_exact_pair_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            checkpoint = root / "checkpoint-v3"
            profile = root / "montage-profile.json"
            registry = root / "active-production-bundle.json"
            scorer = root / "stage-edit-source-pair-checkpoint.json"
            pair_report = root / "stage-edit-source-pair-production-report.json"
            scorer.write_text("{}", encoding="utf-8")
            pair_report.write_text("{}", encoding="utf-8")
            quality_gate = SimpleNamespace(passed=True, to_dict=lambda: {})
            training_result = SimpleNamespace(
                checkpoint_directory=checkpoint,
                report_path=checkpoint / "training-report.json",
                best_epoch=3,
                best_training_step=120,
                production_ready=True,
                quality_gate=quality_gate,
            )
            with (
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "load_professional_montage_policy",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli._snapshot_map",
                    return_value={},
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "load_reference_training_corpus",
                    return_value=SimpleNamespace(examples=()),
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "_dynamic_artifact_map",
                    return_value={},
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "build_stage_edit_dynamic_crop_training_corpus_v2",
                    return_value=SimpleNamespace(name="v2"),
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "build_stage_edit_dynamic_crop_training_corpus_v3",
                    return_value=SimpleNamespace(name="v3"),
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "train_production_stage_edit",
                    return_value=training_result,
                ),
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "register_stage_edit_production_bundle",
                    return_value=registry,
                ) as register,
                patch("builtins.print"),
            ):
                exit_code = main(
                    [
                        "--input",
                        str(root / "input.json"),
                        "--song-id",
                        "reference-a=IVE-XOXZ",
                        "--held-out-song",
                        "IVE-XOXZ",
                        "--feature-snapshot",
                        f"reference-a={root / 'features.json'}",
                        "--dynamic-crop-manifest",
                        str(root / "crop-manifest.json"),
                        "--dynamic-crop-version",
                        "3",
                        "--checkpoint-dir",
                        str(checkpoint),
                        "--authorize-production-activation",
                        "--professional-montage-grammar-profile",
                        str(profile),
                        "--production-registry",
                        str(registry),
                        "--continuity-authority",
                        PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
                        "--source-pair-scorer",
                        str(scorer),
                        "--source-pair-production-report",
                        str(pair_report),
                    ]
                )

            self.assertEqual(0, exit_code)
            register.assert_called_once_with(
                checkpoint,
                professional_montage_grammar_profile_path=profile,
                registry_path=registry,
                dynamic_crop_version=3,
                continuity_authority=PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
                source_pair_scorer_path=scorer,
                source_pair_production_report_path=pair_report,
            )

    def test_existing_checkpoint_directory_fails_before_loading_data(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            checkpoint = Path(raw) / "existing-checkpoint"
            checkpoint.mkdir()
            with (
                patch(
                    "crosscut.stage_edit_production_train_cli."
                    "load_reference_training_corpus"
                ) as load,
                self.assertRaisesRegex(FileExistsError, "refusing to overwrite"),
            ):
                main(
                    [
                        "--input",
                        "/tmp/input.json",
                        "--song-id",
                        "reference-a=IVE-XOXZ",
                        "--held-out-song",
                        "IVE-XOXZ",
                        "--feature-snapshot",
                        "reference-a=/tmp/features.json",
                        "--dynamic-crop-manifest",
                        "/tmp/crop-manifest.json",
                        "--checkpoint-dir",
                        str(checkpoint),
                    ]
                )
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
