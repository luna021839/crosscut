from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crosscut import stage_edit_release_fold_launcher as launcher
from crosscut.stage_edit_transformer import (
    PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
    PRODUCTION_TRAINING_RUN_KIND,
)
from tests.test_professional_montage_policy import _profile, _write


def _write_dummy_manifests(
    root: Path, song_ids: tuple[str, ...]
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    groups: list[dict[str, Path]] = [{}, {}, {}]
    for song_id in song_ids:
        for index, family in enumerate(("materialization", "feature", "dynamic")):
            path = root / f"{song_id}-{family}.json"
            path.write_text(json.dumps({"song": song_id, "family": family}))
            groups[index][song_id] = path
    return groups[0], groups[1], groups[2]


def _write_montage_profile(root: Path) -> Path:
    path = root / "professional-montage-grammar-profile.json"
    _write(path, _profile())
    return path


def _create_development_plan(root: Path, checkpoint: Path) -> Path:
    manifests = _write_dummy_manifests(root, launcher.DEVELOPMENT_SONG_IDS)
    return launcher.create_stage_edit_release_development_plan(
        materialization_manifests=manifests[0],
        feature_manifests=manifests[1],
        dynamic_crop_manifests=manifests[2],
        checkpoint_directory=checkpoint,
        output_directory=root,
    )


def _write_passing_development_checkpoint(
    checkpoint: Path, development_plan: Path
) -> None:
    checkpoint.mkdir(parents=True)
    config = launcher.production_stage_edit_config().to_dict()
    fingerprint = launcher._canonical_sha256(config)
    training_songs = sorted(launcher.DEVELOPMENT_TRAINING_SONG_IDS)
    weight_path = checkpoint / "stage-edit-dynamic-crop-v2.safetensors"
    weight_path.write_bytes(b"synthetic-test-weights")
    weight_sha = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    checkpoint_id = "synthetic-development-checkpoint"
    selection = {
        "schema_version": 1,
        "contract": "stage_edit_model_config_checkpoint_and_threshold_freeze_v1",
        "model_config": config,
        "model_config_fingerprint_sha256": fingerprint,
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_sha256": weight_sha,
        "training_song_ids": training_songs,
        "held_out_song_ids_reserved_for_one_forward": [launcher.XOXZ],
        "heldout_metrics_present": False,
        "heldout_used_for_model_or_threshold_selection": False,
        "style_or_attitude_configuration_mutable_after_freeze": False,
    }
    selection_path = checkpoint / launcher.STAGE_EDIT_SELECTION_FREEZE_FILENAME
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    training = dict(launcher.DEFAULT_RELEASE_TRAINING_CONFIG)
    report = {
        "schema_version": PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
        "run_kind": PRODUCTION_TRAINING_RUN_KIND,
        "production_ready": False,
        "production_readiness_reason": "activation_not_explicitly_authorized",
        "checkpoint": {
            "checkpoint_id": checkpoint_id,
            "weight_filename": weight_path.name,
            "weight_sha256": weight_sha,
            "production_activation_authorized": False,
        },
        "held_out_evaluation": {
            "passed": True,
            "quality_gate": {"passed": True},
            "training_song_ids": training_songs,
            "held_out_song_ids": [launcher.XOXZ],
            "checkpoint_id": checkpoint_id,
            "weight_sha256": weight_sha,
            "cut_threshold_frozen_before_heldout_forward": True,
        },
        "structured_head_coverage_gate": {"passed": True},
        "split": {
            "training_song_ids": training_songs,
            "held_out_song_ids": [launcher.XOXZ],
            "required_training_song_ids": training_songs,
            "held_out_used_for_model_selection": False,
        },
        "optimization": {
            "unseen_song_used_for_model_selection": False,
            "requested_epochs": training["epochs"],
            "early_stopping_patience": training["patience"],
            "batch_size": training["batch_size"],
            "learning_rate": training["learning_rate"],
            "weight_decay": training["weight_decay"],
            "gradient_clip_norm": training["gradient_clip_norm"],
            "seed": training["seed"],
            "learning_rate_schedule": {
                "cut_pretraining_steps": training["cut_pretrain_steps"],
                "cut_pretraining_peak_learning_rate": training[
                    "cut_pretrain_learning_rate"
                ],
                "cut_pretraining_window_length": training[
                    "cut_pretrain_window_length"
                ],
                "cut_pretraining_warmup_steps": training[
                    "cut_pretrain_warmup_steps"
                ],
                "finetune_warmup_steps": training["finetune_warmup_steps"],
                "minimum_learning_rate_ratio": training[
                    "minimum_learning_rate_ratio"
                ],
            },
        },
        "cut_imbalance_contract": {
            "maximum_positive_weight": training[
                "maximum_cut_positive_weight"
            ]
        },
        "model": {"config": config},
        "selection_freeze_evidence": {
            "path": str(selection_path),
            "file_sha256": selection_sha,
            "written_before_heldout_forward": True,
        },
    }
    (checkpoint / launcher.PRODUCTION_TRAINING_REPORT_FILENAME).write_text(
        json.dumps(report), encoding="utf-8"
    )


def _bindings(root: Path, songs: tuple[str, ...]):
    materializations = {}
    features = {}
    for song_index, song_id in enumerate(songs):
        reference_ids = tuple(
            f"synthetic-ref-{song_index}-{index}" for index in range(5)
        )
        materializations[song_id] = launcher.legacy._MaterializationBinding(
            reference_ids=reference_ids,
            supervision_paths={
                ref: root / f"{ref}-supervision.json" for ref in reference_ids
            },
            supervision_sha256={ref: "a" * 64 for ref in reference_ids},
            attribution_paths={
                ref: root / f"{ref}-attribution.json" for ref in reference_ids
            },
            attribution_sha256={ref: "b" * 64 for ref in reference_ids},
            known_shot_counts={ref: 1 for ref in reference_ids},
        )
        features[song_id] = launcher.legacy._FeatureBinding(
            snapshot_paths={
                ref: root / f"{ref}-snapshot.json" for ref in reference_ids
            }
        )
    return materializations, features


class StageEditReleaseFoldLauncherTests(unittest.TestCase):
    def test_development_expands_four_songs_without_activation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            checkpoint = root / "development-checkpoint"
            with patch.object(launcher.legacy, "PRODUCTION_VOLUME_ROOT", root):
                plan = _create_development_plan(root, checkpoint)
                materializations, features = _bindings(
                    root, launcher.DEVELOPMENT_SONG_IDS
                )

                def materialization(_pointer, *, song_id):
                    return (
                        launcher.legacy.ManifestPointer(root / "m", "a" * 64),
                        materializations[song_id],
                    )

                def feature(_pointer, *, song_id, materialization):
                    del materialization
                    return (
                        launcher.legacy.ManifestPointer(root / "f", "b" * 64),
                        features[song_id],
                    )

                def dynamic(_pointer, *, song_id, materialization):
                    del materialization
                    return launcher.legacy.ManifestPointer(
                        root / f"{song_id}-dynamic.json", "c" * 64
                    )

                with (
                    patch.object(
                        launcher.legacy,
                        "_validate_storage",
                        return_value=20 * 1024**3,
                    ),
                    patch.object(
                        launcher.legacy,
                        "_validate_materialization_manifest",
                        side_effect=materialization,
                    ),
                    patch.object(
                        launcher.legacy,
                        "_validate_feature_manifest",
                        side_effect=feature,
                    ),
                    patch.object(
                        launcher.legacy,
                        "_validate_dynamic_crop_manifest",
                        side_effect=dynamic,
                    ),
                ):
                    result = launcher.preflight_stage_edit_release_development(
                        plan
                    )

            self.assertEqual(20, result.train_cli_args.count("--input"))
            self.assertEqual(
                4, result.train_cli_args.count("--dynamic-crop-manifest")
            )
            self.assertEqual(
                3, result.train_cli_args.count("--required-training-song")
            )
            self.assertNotIn(
                "--authorize-production-activation", result.train_cli_args
            )
            self.assertNotIn("--production-registry", result.train_cli_args)
            self.assertEqual(launcher.XOXZ, result.held_out_song_id)

    def test_final_plan_is_impossible_before_development_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            with patch.object(launcher.legacy, "PRODUCTION_VOLUME_ROOT", root):
                development = _create_development_plan(
                    root, root / "development-checkpoint"
                )
                with patch.object(launcher, "_song_records") as song_records:
                    with self.assertRaisesRegex(
                        launcher.ReleaseFoldPreflightError,
                        "regular non-symlink",
                    ):
                        launcher.create_stage_edit_release_final_plan(
                            development_plan_path=development,
                            development_freeze_path=root / "missing-freeze.json",
                            materialization_manifests={},
                            feature_manifests={},
                            dynamic_crop_manifests={},
                            checkpoint_directory=root / "final",
                            output_directory=root,
                            professional_montage_grammar_profile=(
                                root / "missing-profile.json"
                            ),
                        )
                    song_records.assert_not_called()

    def test_freeze_rebinds_report_selection_config_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            checkpoint = root / "development-checkpoint"
            with patch.object(launcher.legacy, "PRODUCTION_VOLUME_ROOT", root):
                development = _create_development_plan(root, checkpoint)
                _write_passing_development_checkpoint(checkpoint, development)
                freeze = launcher.freeze_stage_edit_release_development(
                    development, output_directory=root
                )
                payload = launcher._strict_development_freeze(
                    freeze, expected_plan_path=development
                )
                self.assertTrue(payload["development_quality_gate_passed"])
                self.assertFalse(payload["production_activation_authorized"])
                selection = checkpoint / launcher.STAGE_EDIT_SELECTION_FREEZE_FILENAME
                selection.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(
                    launcher.ReleaseFoldPreflightError, "SHA-256 mismatch"
                ):
                    launcher._strict_development_freeze(
                        freeze, expected_plan_path=development
                    )

    def test_final_expands_five_songs_and_authorizes_only_blind_pass(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            development_checkpoint = root / "development-checkpoint"
            final_checkpoint = root / "final-checkpoint"
            with patch.object(launcher.legacy, "PRODUCTION_VOLUME_ROOT", root):
                development = _create_development_plan(
                    root, development_checkpoint
                )
                _write_passing_development_checkpoint(
                    development_checkpoint, development
                )
                freeze = launcher.freeze_stage_edit_release_development(
                    development, output_directory=root
                )
                development_payload = launcher._strict_development_plan(development)
                development_records = launcher._indexed_song_records(
                    development_payload, launcher.DEVELOPMENT_SONG_IDS
                )
                manifests = [dict(), dict(), dict()]
                for song_id in launcher.DEVELOPMENT_SONG_IDS:
                    record = development_records[song_id]
                    for index, key in enumerate(
                        (
                            "materialization_manifest",
                            "feature_manifest",
                            "dynamic_crop_manifest",
                        )
                    ):
                        manifests[index][song_id] = Path(record[key]["path"])
                attitude = _write_dummy_manifests(root, (launcher.ATTITUDE,))
                for index in range(3):
                    manifests[index].update(attitude[index])
                montage_profile = _write_montage_profile(root)
                final = launcher.create_stage_edit_release_final_plan(
                    development_plan_path=development,
                    development_freeze_path=freeze,
                    materialization_manifests=manifests[0],
                    feature_manifests=manifests[1],
                    dynamic_crop_manifests=manifests[2],
                    checkpoint_directory=final_checkpoint,
                    output_directory=root,
                    professional_montage_grammar_profile=montage_profile,
                    production_registry=root / "active.json",
                )
                second_final = launcher.create_stage_edit_release_final_plan(
                    development_plan_path=development,
                    development_freeze_path=freeze,
                    materialization_manifests=manifests[0],
                    feature_manifests=manifests[1],
                    dynamic_crop_manifests=manifests[2],
                    checkpoint_directory=root / "different-final-checkpoint",
                    output_directory=root,
                    professional_montage_grammar_profile=montage_profile,
                    production_registry=root / "active.json",
                )
                first_payload = launcher._strict_final_plan(final)
                second_payload = launcher._strict_final_plan(second_final)
                self.assertEqual(
                    launcher.FINAL_RELEASE_PLAN_SCHEMA_VERSION,
                    first_payload["schema_version"],
                )
                self.assertEqual(
                    first_payload["blind_consumption_marker"],
                    second_payload["blind_consumption_marker"],
                )
                self.assertEqual(
                    str(montage_profile),
                    first_payload["professional_montage_grammar_profile"][
                        "path"
                    ],
                )
                materializations, features = _bindings(
                    root, launcher.FINAL_SONG_IDS
                )

                def materialization(_pointer, *, song_id):
                    return (
                        launcher.legacy.ManifestPointer(root / "m", "a" * 64),
                        materializations[song_id],
                    )

                def feature(_pointer, *, song_id, materialization):
                    del materialization
                    return (
                        launcher.legacy.ManifestPointer(root / "f", "b" * 64),
                        features[song_id],
                    )

                def dynamic(_pointer, *, song_id, materialization):
                    del materialization
                    return launcher.legacy.ManifestPointer(
                        root / f"{song_id}-dynamic.json", "c" * 64
                    )

                with (
                    patch.object(
                        launcher.legacy,
                        "_validate_storage",
                        return_value=20 * 1024**3,
                    ),
                    patch.object(
                        launcher.legacy,
                        "_validate_materialization_manifest",
                        side_effect=materialization,
                    ),
                    patch.object(
                        launcher.legacy,
                        "_validate_feature_manifest",
                        side_effect=feature,
                    ),
                    patch.object(
                        launcher.legacy,
                        "_validate_dynamic_crop_manifest",
                        side_effect=dynamic,
                    ),
                ):
                    result = launcher.preflight_stage_edit_release_final(final)

            self.assertEqual(25, result.train_cli_args.count("--input"))
            self.assertEqual(
                5, result.train_cli_args.count("--dynamic-crop-manifest")
            )
            self.assertEqual(
                4, result.train_cli_args.count("--required-training-song")
            )
            self.assertEqual(launcher.ATTITUDE, result.held_out_song_id)
            self.assertIn(
                "--authorize-production-activation", result.train_cli_args
            )
            profile_option = result.train_cli_args.index(
                "--professional-montage-grammar-profile"
            )
            self.assertEqual(
                str(montage_profile),
                result.train_cli_args[profile_option + 1],
            )
            self.assertIn("--production-registry", result.train_cli_args)

    def test_final_plan_revalidates_immutable_montage_profile(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            development_checkpoint = root / "development-checkpoint"
            with patch.object(launcher.legacy, "PRODUCTION_VOLUME_ROOT", root):
                development = _create_development_plan(
                    root, development_checkpoint
                )
                _write_passing_development_checkpoint(
                    development_checkpoint, development
                )
                freeze = launcher.freeze_stage_edit_release_development(
                    development, output_directory=root
                )
                development_payload = launcher._strict_development_plan(
                    development
                )
                records = launcher._indexed_song_records(
                    development_payload, launcher.DEVELOPMENT_SONG_IDS
                )
                manifests = [dict(), dict(), dict()]
                for song_id in launcher.DEVELOPMENT_SONG_IDS:
                    for index, key in enumerate(
                        (
                            "materialization_manifest",
                            "feature_manifest",
                            "dynamic_crop_manifest",
                        )
                    ):
                        manifests[index][song_id] = Path(records[song_id][key]["path"])
                attitude = _write_dummy_manifests(root, (launcher.ATTITUDE,))
                for index in range(3):
                    manifests[index].update(attitude[index])
                montage_profile = _write_montage_profile(root)
                final = launcher.create_stage_edit_release_final_plan(
                    development_plan_path=development,
                    development_freeze_path=freeze,
                    materialization_manifests=manifests[0],
                    feature_manifests=manifests[1],
                    dynamic_crop_manifests=manifests[2],
                    checkpoint_directory=root / "final-checkpoint",
                    output_directory=root,
                    professional_montage_grammar_profile=montage_profile,
                )
                montage_profile.write_text("{}", encoding="utf-8")
                with patch.object(
                    launcher.legacy, "_validate_storage"
                ) as validate_storage:
                    with self.assertRaisesRegex(
                        launcher.ReleaseFoldPreflightError,
                        "profile is missing, changed, or invalid",
                    ):
                        launcher.preflight_stage_edit_release_final(final)
                    validate_storage.assert_not_called()

    def test_create_final_parser_requires_explicit_montage_profile(self) -> None:
        parser = launcher.build_parser()
        minimal = [
            "create-final",
            "--development-plan",
            "/tmp/development.json",
            "--development-freeze",
            "/tmp/freeze.json",
            "--checkpoint-dir",
            "/tmp/checkpoint",
            "--output-directory",
            "/tmp/output",
            "--materialization-manifest",
            "song=/tmp/materialization.json",
            "--feature-manifest",
            "song=/tmp/features.json",
            "--dynamic-crop-manifest",
            "song=/tmp/dynamic.json",
        ]
        with self.assertRaises(SystemExit):
            parser.parse_args(minimal)
        parsed = parser.parse_args(
            [
                *minimal,
                "--professional-montage-grammar-profile",
                "/tmp/montage.json",
            ]
        )
        self.assertEqual(
            Path("/tmp/montage.json"),
            parsed.professional_montage_grammar_profile,
        )

    def test_create_final_cli_forwards_explicit_montage_profile(self) -> None:
        profile = Path("/tmp/professional-montage.json")
        argv = [
            "create-final",
            "--development-plan",
            "/tmp/development.json",
            "--development-freeze",
            "/tmp/freeze.json",
            "--checkpoint-dir",
            "/tmp/checkpoint",
            "--output-directory",
            "/tmp/output",
            "--materialization-manifest",
            "song=/tmp/materialization.json",
            "--feature-manifest",
            "song=/tmp/features.json",
            "--dynamic-crop-manifest",
            "song=/tmp/dynamic.json",
            "--professional-montage-grammar-profile",
            str(profile),
        ]
        with (
            patch.object(launcher, "_keyed_paths", return_value={}),
            patch.object(
                launcher,
                "create_stage_edit_release_final_plan",
                return_value=Path("/tmp/final-plan.json"),
            ) as create_final,
            patch("builtins.print"),
        ):
            self.assertEqual(0, launcher.main(argv))
        self.assertEqual(
            profile,
            create_final.call_args.kwargs[
                "professional_montage_grammar_profile"
            ],
        )

    def test_final_launch_consumes_marker_even_when_quality_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            marker = root / "blind-consumed.json"
            preflight = launcher.ReleaseFoldPreflight(
                stage="final_blind",
                plan_path=root / "plan.json",
                plan_sha256="a" * 64,
                held_out_song_id=launcher.ATTITUDE,
                training_song_ids=launcher.FINAL_TRAINING_SONG_IDS,
                checkpoint_directory=root / "checkpoint",
                blind_consumption_marker=marker,
                train_cli_args=("--authorize-production-activation",),
                reference_ids_by_song={},
            )
            with (
                patch.object(
                    launcher,
                    "preflight_stage_edit_release_final",
                    return_value=preflight,
                ),
                patch.object(
                    launcher, "production_train_main", return_value=2
                ) as train,
            ):
                self.assertEqual(
                    2, launcher.launch_stage_edit_release_final(root / "plan.json")
                )
                self.assertTrue(marker.is_file())
                self.assertFalse(
                    json.loads(marker.read_text())["retry_allowed"]
                )
                with self.assertRaisesRegex(
                    launcher.ReleaseFoldPreflightError, "already consumed"
                ):
                    launcher.launch_stage_edit_release_final(root / "plan.json")
                train.assert_called_once_with(preflight.train_cli_args)


if __name__ == "__main__":
    unittest.main()
