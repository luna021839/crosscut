from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from crosscut import stage_edit_release_fold_launcher as launcher
from crosscut.stage_edit_dynamic_crop_capacity_probe import (
    default_dynamic_crop_v2_probe_candidates,
    write_dynamic_crop_capacity_selection,
)
from crosscut.stage_edit_transformer import (
    PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
    PRODUCTION_TRAINING_RUN_KIND,
)
from tests.test_professional_montage_policy import _profile, _write
from tests.test_stage_edit_dynamic_crop_capacity_probe import BUDGET, _inputs


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


def _write_capacity_selection(root: Path) -> Path:
    capacity_root = root / "capacity-selection-evidence"
    capacity_root.mkdir(parents=True, exist_ok=True)
    candidates = default_dynamic_crop_v2_probe_candidates()
    dataset, folds, smokes = _inputs(
        capacity_root,
        tuple((candidate, (1.0, 1.0, 1.0)) for candidate in candidates),
    )
    return write_dynamic_crop_capacity_selection(
        capacity_root,
        candidates=candidates,
        fold_run_evidence_paths=folds,
        selection_dataset_manifest_path=dataset,
        mlx_smoke_evidence_paths_by_candidate=smokes,
        device_memory_budget=BUDGET,
        baseline_candidate_id=candidates[0].candidate_id,
    )


def _create_development_plan(
    root: Path, checkpoint: Path, *, capacity_selection: Path | None = None
) -> Path:
    manifests = _write_dummy_manifests(root, launcher.DEVELOPMENT_SONG_IDS)
    return launcher.create_stage_edit_release_development_plan(
        materialization_manifests=manifests[0],
        feature_manifests=manifests[1],
        dynamic_crop_manifests=manifests[2],
        checkpoint_directory=checkpoint,
        output_directory=root,
        capacity_selection_path=(
            capacity_selection
            if capacity_selection is not None
            else _write_capacity_selection(root)
        ),
    )


def _write_passing_development_checkpoint(
    checkpoint: Path, development_plan: Path
) -> None:
    checkpoint.mkdir(parents=True)
    model_training = json.loads(
        development_plan.read_text(encoding="utf-8")
    )["model_training_contract"]
    config = model_training["model_config"]
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
    effective_loss = dict(model_training["base_loss_contract"])
    effective_loss["cut_positive_weight"] = 2.0
    effective_dynamic_loss = json.loads(
        json.dumps(model_training["dynamic_crop_loss_contract"])
    )
    effective_dynamic_loss["base"]["cut_positive_weight"] = 2.0
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
        "transition_authority_contract": model_training[
            "transition_authority_contract"
        ],
        "transition_authority_contract_sha256": model_training[
            "transition_authority_contract_sha256"
        ],
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
            "effective_loss_contract": effective_loss,
            "effective_dynamic_crop_loss_contract": effective_dynamic_loss,
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
            ],
            "applied_positive_weight": 2.0,
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
    def setUp(self) -> None:
        # These release-launcher fixtures deliberately use synthetic smoke
        # repositories. The capacity-probe suite separately exercises the
        # live clean-checkout attestation; only that external boundary is
        # replaced here, while the exact four-candidate release contract and
        # every content/provenance binding remain active.
        for target in (
            "crosscut.stage_edit_dynamic_crop_capacity_probe."
            "_enforce_actual_smoke_source_paths",
            "crosscut.stage_edit_dynamic_crop_capacity_probe."
            "_enforce_live_release_smoke_attestation",
        ):
            attestation = patch(target)
            attestation.start()
            self.addCleanup(attestation.stop)

    def test_pre_capacity_binding_development_plan_schema_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            with patch.object(launcher.legacy, "PRODUCTION_VOLUME_ROOT", root):
                current = _create_development_plan(root, root / "checkpoint")
                payload = json.loads(current.read_text(encoding="utf-8"))
                payload.pop("plan_sha256")
                payload["schema_version"] = 3
                payload["analysis_basis"] = (
                    "ive_stageedit_release_development_plan_v3"
                )
                stale = launcher._write_content_addressed(
                    payload,
                    output_directory=root / "stale",
                    prefix="stageedit-release-development-plan",
                    hash_field="plan_sha256",
                )
                with self.assertRaisesRegex(
                    launcher.ReleaseFoldPreflightError,
                    "schema mismatch",
                ):
                    launcher._strict_development_plan(stale)

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
                    patch.object(
                        launcher,
                        "_preflight_structured_head_label_coverage",
                        return_value={"passed": True},
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
            self.assertIn("--model-config-json", result.train_cli_args)
            self.assertIn("--base-loss-contract-json", result.train_cli_args)
            self.assertIn(
                "--dynamic-crop-loss-contract-json", result.train_cli_args
            )
            self.assertEqual(launcher.XOXZ, result.held_out_song_id)
            self.assertEqual(
                json.loads(plan.read_text(encoding="utf-8"))[
                    "capacity_selection"
                ],
                result.capacity_selection,
            )

    def test_exact_hard_cut_metadata_passes_renderer_owned_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            paths_by_song = {}
            for song_index, song_id in enumerate(launcher.DEVELOPMENT_SONG_IDS):
                paths = []
                for reference_index in range(2):
                    path = root / f"{song_index}-{reference_index}.json"
                    path.write_text(
                        json.dumps(
                            {
                                "reference_id": (
                                    f"ref-{song_index}-{reference_index}"
                                ),
                                "shot_labels": [
                                    {
                                        "edit_labels": {
                                            "source_transition": {
                                                "supervision_eligible": True,
                                                "transition": "hard_cut",
                                            }
                                        }
                                    }
                                ],
                                "listwise_records": [
                                    {
                                        "edit_labels": {
                                            "source_transition": {
                                                "supervision_eligible": True,
                                                "transition": "hard_cut",
                                            }
                                        }
                                    }
                                ],
                                "pairwise_records": [
                                    {
                                        "edit_labels": {
                                            "source_transition": {
                                                "supervision_eligible": True,
                                                "transition": "hard_cut",
                                            }
                                        }
                                    }
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    paths.append(path)
                paths_by_song[song_id] = tuple(paths)

            def example(song_id: str, duration_bucket: int) -> SimpleNamespace:
                return SimpleNamespace(
                    song_id=song_id,
                    targets=SimpleNamespace(
                        targets=(
                            SimpleNamespace(
                                duration_bucket=duration_bucket,
                                transition_label="hard_cut",
                            ),
                        )
                    )
                )

            split = SimpleNamespace(
                train=tuple(
                    example(song, reference_index)
                    for song in launcher.DEVELOPMENT_TRAINING_SONG_IDS
                    for reference_index in range(2)
                ),
                held_out=(
                    example(launcher.XOXZ, 0),
                    example(launcher.XOXZ, 1),
                ),
            )
            corpus = SimpleNamespace(examples=split.train + split.held_out)
            with (
                patch.object(
                    launcher,
                    "load_reference_training_corpus",
                    return_value=corpus,
                ),
                patch.object(
                    launcher, "production_song_split", return_value=split
                ),
            ):
                evidence = launcher._preflight_structured_head_label_coverage(
                    paths_by_song,
                    contract=launcher._structured_head_preflight_contract(),
                )
            self.assertTrue(evidence["passed"])
            self.assertEqual(
                "skipped_non_authoritative",
                evidence["training"]["transition"]["status"],
            )
            self.assertTrue(
                all(
                    audit["label_count"] == 2
                    for audit in evidence["raw_transition_label_audit"].values()
                )
            )

    def test_fabricated_nonhard_transition_label_fails_preflight(self) -> None:
        for fabricated in ("zoom_match", "whip_pan", "cross_dissolve"):
            with self.subTest(fabricated=fabricated), tempfile.TemporaryDirectory() as raw:
                root = Path(raw).resolve()
                paths_by_song = {}
                for song_index, song_id in enumerate(launcher.DEVELOPMENT_SONG_IDS):
                    path = root / f"{song_index}.json"
                    label = (
                        fabricated
                        if song_id == launcher.REBEL_HEART
                        else "hard_cut"
                    )
                    path.write_text(
                        json.dumps(
                            {
                                "reference_id": f"ref-{song_index}",
                                "shot_labels": [
                                    {
                                        "edit_labels": {
                                            "source_transition": {
                                                "supervision_eligible": True,
                                                "transition": label,
                                            }
                                        }
                                    }
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    paths_by_song[song_id] = (path,)
                with self.assertRaisesRegex(
                    launcher.ReleaseFoldPreflightError,
                    "unrecognized or non-hard transition label",
                ):
                    launcher._preflight_structured_head_label_coverage(
                        paths_by_song,
                        contract=launcher._structured_head_preflight_contract(),
                    )

    def test_model_capacity_and_source_objective_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            with patch.object(launcher.legacy, "PRODUCTION_VOLUME_ROOT", root):
                plan = _create_development_plan(root, root / "checkpoint")
                payload = json.loads(plan.read_text(encoding="utf-8"))
                payload.pop("plan_sha256")

                changed_capacity = json.loads(json.dumps(payload))
                changed_capacity["model_training_contract"]["model_config"][
                    "model_dimension"
                ] = 384
                capacity_plan = launcher._write_content_addressed(
                    changed_capacity,
                    output_directory=root / "capacity-tamper",
                    prefix="stageedit-release-development-plan",
                    hash_field="plan_sha256",
                )
                with self.assertRaisesRegex(
                    launcher.ReleaseFoldPreflightError,
                    "model config hash mismatch",
                ):
                    launcher._strict_development_plan(capacity_plan)

                changed_objective = json.loads(json.dumps(payload))
                changed_objective["model_training_contract"][
                    "source_objective"
                ] = "listwise"
                objective_plan = launcher._write_content_addressed(
                    changed_objective,
                    output_directory=root / "objective-tamper",
                    prefix="stageedit-release-development-plan",
                    hash_field="plan_sha256",
                )
                with self.assertRaisesRegex(
                    launcher.ReleaseFoldPreflightError,
                    "source objective binding mismatch",
                ):
                    launcher._strict_development_plan(objective_plan)

                changed_authority = json.loads(json.dumps(payload))
                model_contract = changed_authority["model_training_contract"]
                model_contract["model_config"]["transition_authority"] = (
                    "learned_multiclass_transition_head_v1"
                )
                model_contract["model_config_sha256"] = launcher._canonical_sha256(
                    model_contract["model_config"]
                )
                authority_plan = launcher._write_content_addressed(
                    changed_authority,
                    output_directory=root / "authority-tamper",
                    prefix="stageedit-release-development-plan",
                    hash_field="plan_sha256",
                )
                with self.assertRaisesRegex(
                    launcher.ReleaseFoldPreflightError,
                    "renderer-owned transition authority differs",
                ):
                    launcher._strict_development_plan(authority_plan)

    def test_verified_capacity_selection_contract_and_origin_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            capacity_selection = _write_capacity_selection(root)
            manifests = _write_dummy_manifests(
                root, launcher.DEVELOPMENT_SONG_IDS
            )
            with patch.object(launcher.legacy, "PRODUCTION_VOLUME_ROOT", root):
                plan = launcher.create_stage_edit_release_development_plan(
                    materialization_manifests=manifests[0],
                    feature_manifests=manifests[1],
                    dynamic_crop_manifests=manifests[2],
                    checkpoint_directory=root / "checkpoint",
                    output_directory=root,
                    capacity_selection_path=capacity_selection,
                )
                payload = launcher._strict_development_plan(plan)
            selection = launcher.load_dynamic_crop_capacity_selection(
                capacity_selection
            )
            self.assertEqual(
                selection["release_model_training_contract"],
                payload["model_training_contract"],
            )
            self.assertEqual(
                selection["selected_candidate_id"],
                payload["capacity_selection"]["selected_candidate_id"],
            )
            self.assertEqual(
                selection["release_model_training_contract_origin"],
                payload["capacity_selection"][
                    "release_model_training_contract_origin"
                ],
            )
            self.assertEqual(
                selection["artifact_content_sha256"],
                payload["capacity_selection"]["artifact_content_sha256"],
            )

    def test_capacity_selection_symlink_and_binding_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            capacity_selection = _write_capacity_selection(root)
            manifests = _write_dummy_manifests(
                root, launcher.DEVELOPMENT_SONG_IDS
            )
            symlink = root / "capacity-selection-symlink.json"
            symlink.symlink_to(capacity_selection)
            with patch.object(launcher.legacy, "PRODUCTION_VOLUME_ROOT", root):
                with self.assertRaisesRegex(
                    launcher.ReleaseFoldPreflightError, "must not be a symlink"
                ):
                    launcher.create_stage_edit_release_development_plan(
                        materialization_manifests=manifests[0],
                        feature_manifests=manifests[1],
                        dynamic_crop_manifests=manifests[2],
                        checkpoint_directory=root / "symlink-checkpoint",
                        output_directory=root / "symlink-plan",
                        capacity_selection_path=symlink,
                    )

                plan = launcher.create_stage_edit_release_development_plan(
                    materialization_manifests=manifests[0],
                    feature_manifests=manifests[1],
                    dynamic_crop_manifests=manifests[2],
                    checkpoint_directory=root / "checkpoint",
                    output_directory=root,
                    capacity_selection_path=capacity_selection,
                )
                payload = json.loads(plan.read_text(encoding="utf-8"))
                payload.pop("plan_sha256")
                payload["capacity_selection"]["selection_identity_sha256"] = (
                    "f" * 64
                )
                tampered = launcher._write_content_addressed(
                    payload,
                    output_directory=root / "binding-tamper",
                    prefix="stageedit-release-development-plan",
                    hash_field="plan_sha256",
                )
                with self.assertRaisesRegex(
                    launcher.ReleaseFoldPreflightError,
                    "binding changed or does not match",
                ):
                    launcher._strict_development_plan(tampered)

                capacity_selection.write_text("{}\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    launcher.ReleaseFoldPreflightError,
                    "capacity selection is missing, unsafe, or invalid",
                ):
                    launcher._strict_development_plan(plan)

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
                development_payload = launcher._strict_development_plan(
                    development
                )
                self.assertTrue(payload["development_quality_gate_passed"])
                self.assertFalse(payload["production_activation_authorized"])
                self.assertEqual(
                    development_payload["capacity_selection"],
                    payload["capacity_selection"],
                )
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
                real_file_sha256 = launcher._file_sha256

                def reject_premarker_attitude_hash(path, name):
                    if launcher.ATTITUDE in str(path):
                        raise AssertionError(
                            "ATTITUDE was hashed before blind consumption"
                        )
                    return real_file_sha256(path, name)

                with patch.object(
                    launcher,
                    "_file_sha256",
                    side_effect=reject_premarker_attitude_hash,
                ):
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
                    development_payload["capacity_selection"],
                    first_payload["capacity_selection"],
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
                self.assertEqual(4, len(first_payload["songs"]))
                self.assertNotIn(
                    launcher.ATTITUDE,
                    {item["song_id"] for item in first_payload["songs"]},
                )
                deferred_attitude = first_payload[
                    "deferred_blind_held_out_song"
                ]
                self.assertEqual(
                    launcher.DEFERRED_BLIND_MANIFEST_CONTRACT,
                    deferred_attitude["inspection_contract"],
                )
                self.assertFalse(
                    any("sha256" in key for key in deferred_attitude)
                )
                blind_marker = Path(first_payload["blind_consumption_marker"])
                materializations, features = _bindings(
                    root, launcher.FINAL_SONG_IDS
                )
                attitude_validations = []

                def guarded_release_hash(path, name):
                    if launcher.ATTITUDE in str(path):
                        self.assertTrue(blind_marker.is_file())
                    return real_file_sha256(path, name)

                def materialization(_pointer, *, song_id):
                    if song_id == launcher.ATTITUDE:
                        self.assertTrue(blind_marker.is_file())
                        attitude_validations.append(song_id)
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
                        launcher,
                        "_file_sha256",
                        side_effect=guarded_release_hash,
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
                    patch.object(
                        launcher, "production_train_main", return_value=0
                    ) as train,
                ):
                    result = launcher.preflight_stage_edit_release_final(final)
                    self.assertEqual((), result.train_cli_args)
                    self.assertTrue(result.held_out_manifest_inspection_deferred)
                    self.assertNotIn(
                        launcher.ATTITUDE, result.reference_ids_by_song
                    )
                    self.assertEqual([], attitude_validations)
                    self.assertEqual(
                        0, launcher.launch_stage_edit_release_final(final)
                    )
                    consumed_args = train.call_args.args[0]

            self.assertEqual([launcher.ATTITUDE], attitude_validations)
            self.assertTrue(blind_marker.is_file())
            self.assertEqual(25, consumed_args.count("--input"))
            self.assertEqual(
                5, consumed_args.count("--dynamic-crop-manifest")
            )
            self.assertEqual(
                4, consumed_args.count("--required-training-song")
            )
            self.assertEqual(launcher.ATTITUDE, result.held_out_song_id)
            self.assertIn(
                "--authorize-production-activation", consumed_args
            )
            profile_option = consumed_args.index(
                "--professional-montage-grammar-profile"
            )
            self.assertEqual(
                str(montage_profile),
                consumed_args[profile_option + 1],
            )
            self.assertIn("--production-registry", consumed_args)
            self.assertEqual(
                first_payload["capacity_selection"],
                result.capacity_selection,
            )

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

    def test_create_development_parser_requires_capacity_selection(self) -> None:
        parser = launcher.build_parser()
        minimal = [
            "create-development",
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
                "--capacity-selection",
                "/tmp/dynamic-crop-v2-capacity-selection.json",
            ]
        )
        self.assertEqual(
            Path("/tmp/dynamic-crop-v2-capacity-selection.json"),
            parsed.capacity_selection,
        )

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

    def test_attitude_cli_path_is_normalized_without_resolve_or_stat(self) -> None:
        with patch.object(
            Path,
            "resolve",
            side_effect=AssertionError("ATTITUDE path must remain deferred"),
        ) as resolve:
            result = launcher._keyed_paths(
                [f"{launcher.ATTITUDE}=/tmp/blind/../attitude.json"],
                expected_song_ids=(launcher.ATTITUDE,),
                option="--materialization-manifest",
                content_inspection_deferred_song_ids=(launcher.ATTITUDE,),
            )
        resolve.assert_not_called()
        self.assertEqual(
            Path("/tmp/attitude.json"), result[launcher.ATTITUDE]
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
                capacity_selection={"selected_candidate_id": "a" * 64},
                blind_consumption_marker=marker,
                held_out_manifest_inspection_deferred=True,
                train_cli_args=(),
                reference_ids_by_song={},
            )
            consumed = launcher.ReleaseFoldPreflight(
                stage="final_blind",
                plan_path=preflight.plan_path,
                plan_sha256=preflight.plan_sha256,
                held_out_song_id=launcher.ATTITUDE,
                training_song_ids=launcher.FINAL_TRAINING_SONG_IDS,
                checkpoint_directory=preflight.checkpoint_directory,
                capacity_selection=preflight.capacity_selection,
                blind_consumption_marker=marker,
                held_out_manifest_inspection_deferred=False,
                train_cli_args=("--authorize-production-activation",),
                reference_ids_by_song={},
            )
            ordering = []

            def fsync_parent(path):
                self.assertEqual(marker, path)
                self.assertTrue(marker.is_file())
                ordering.append("directory_fsynced")

            def consumed_preflight(_plan, *, consumption_marker):
                self.assertEqual(marker, consumption_marker)
                self.assertTrue(marker.is_file())
                self.assertEqual(["directory_fsynced"], ordering)
                ordering.append("attitude_inspection")
                return consumed

            with (
                patch.object(
                    launcher,
                    "preflight_stage_edit_release_final",
                    return_value=preflight,
                ),
                patch.object(
                    launcher,
                    "_preflight_consumed_stage_edit_release_final",
                    side_effect=consumed_preflight,
                ),
                patch.object(
                    launcher,
                    "_fsync_parent_directory",
                    side_effect=fsync_parent,
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
                train.assert_called_once_with(consumed.train_cli_args)
                self.assertEqual(
                    ["directory_fsynced", "attitude_inspection"], ordering
                )

    def test_attitude_manifest_failure_after_marker_cannot_retry(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            marker = root / "blind-consumed.json"
            preflight = launcher.ReleaseFoldPreflight(
                stage="final_blind",
                plan_path=root / "plan.json",
                plan_sha256="b" * 64,
                held_out_song_id=launcher.ATTITUDE,
                training_song_ids=launcher.FINAL_TRAINING_SONG_IDS,
                checkpoint_directory=root / "checkpoint",
                capacity_selection={"selected_candidate_id": "b" * 64},
                blind_consumption_marker=marker,
                held_out_manifest_inspection_deferred=True,
                train_cli_args=(),
                reference_ids_by_song={},
            )

            def reject_attitude(_plan, *, consumption_marker):
                self.assertEqual(marker, consumption_marker)
                self.assertTrue(marker.is_file())
                raise launcher.ReleaseFoldPreflightError(
                    "ATTITUDE manifest invalid after consumption"
                )

            with (
                patch.object(
                    launcher,
                    "preflight_stage_edit_release_final",
                    return_value=preflight,
                ),
                patch.object(
                    launcher,
                    "_preflight_consumed_stage_edit_release_final",
                    side_effect=reject_attitude,
                ) as consumed_preflight,
                patch.object(launcher, "production_train_main") as train,
            ):
                with self.assertRaisesRegex(
                    launcher.ReleaseFoldPreflightError,
                    "manifest invalid after consumption",
                ):
                    launcher.launch_stage_edit_release_final(preflight.plan_path)
                self.assertTrue(marker.is_file())
                with self.assertRaisesRegex(
                    launcher.ReleaseFoldPreflightError, "already consumed"
                ):
                    launcher.launch_stage_edit_release_final(preflight.plan_path)
                consumed_preflight.assert_called_once()
                train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
