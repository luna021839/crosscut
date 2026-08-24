from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import tempfile
import unittest
from unittest.mock import patch

from crosscut import stage_edit_dynamic_crop_v3_capacity_fold_launcher as launcher
from crosscut.stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
    DYNAMIC_CROP_V3_LOSS_CONTRACT,
    DYNAMIC_CROP_V3_MODEL_CONTRACT,
    DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
    DYNAMIC_CROP_V3_SCHEMA_VERSION,
    DYNAMIC_CROP_V3_TARGET_CONTRACT,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v3,
)
from crosscut.stage_edit_production_training import (
    STAGE_EDIT_SELECTION_FREEZE_FILENAME,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    PRODUCTION_CHECKPOINT_SCOPE,
    PRODUCTION_TRAINING_REPORT_FILENAME,
    PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
    PRODUCTION_TRAINING_RUN_KIND,
)


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path.resolve()


def _fake_source(commit: str = "a" * 40) -> dict[str, object]:
    return {
        "contract": "clean_head_and_exact_interpreter_source_blobs_v1",
        "repository_root": "/private/tmp/fake-v3-capacity-repository",
        "head_commit_oid": commit,
        "repository_status_clean": True,
        "repository_status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
        "source_files": {
            path: {
                "git_blob_oid": "b" * 40,
                "file_sha256": "c" * 64,
            }
            for path in launcher.V3_SOURCE_RELATIVE_PATHS
        },
    }


def _fake_manifest_set(root: Path) -> launcher._ValidatedManifestSet:
    selection = _write_json(root / "selection.json", {"fixture": True})
    records: list[dict[str, object]] = []
    references: dict[str, tuple[str, ...]] = {}
    supervision: dict[str, dict[str, Path]] = {}
    snapshots: dict[str, dict[str, Path]] = {}
    dynamics: dict[str, Path] = {}
    for song_index, song_id in enumerate(launcher.SELECTION_SONG_IDS):
        materialization = _write_json(
            root / f"materialization-{song_index}.json", {"fixture": song_index}
        )
        feature = _write_json(
            root / f"feature-{song_index}.json", {"fixture": song_index}
        )
        dynamic = _write_json(
            root / f"dynamic-{song_index}.json", {"fixture": song_index}
        )
        refs = tuple(f"song-{song_index}-reference-{index}" for index in range(5))
        references[song_id] = refs
        supervision[song_id] = {
            ref: root / f"{ref}-supervision.json" for ref in refs
        }
        snapshots[song_id] = {
            ref: root / f"{ref}-snapshot.json" for ref in refs
        }
        dynamics[song_id] = dynamic
        records.append(
            {
                "song_id": song_id,
                "reference_ids": list(refs),
                "materialization_manifest": launcher.v2_launcher._pointer(
                    materialization, f"{song_id} materialization"
                ),
                "materialization_outputs": [],
                "feature_manifest": launcher.v2_launcher._pointer(
                    feature, f"{song_id} feature"
                ),
                "feature_snapshots": [],
                "dynamic_crop_manifest": launcher.v2_launcher._pointer(
                    dynamic, f"{song_id} dynamic"
                ),
            }
        )
    return launcher._ValidatedManifestSet(
        selection_dataset_pointer=launcher.v2_launcher._pointer(
            selection, "selection dataset"
        ),
        selection_dataset_content_sha256="d" * 64,
        song_records=tuple(records),
        reference_ids_by_song=references,
        supervision_paths_by_song=MappingProxyType(supervision),
        snapshot_paths_by_song=MappingProxyType(snapshots),
        dynamic_manifest_paths_by_song=MappingProxyType(dynamics),
    )


def _manifest_maps(
    bundle: launcher._ValidatedManifestSet,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    materialization: dict[str, Path] = {}
    feature: dict[str, Path] = {}
    dynamic: dict[str, Path] = {}
    for record in bundle.song_records:
        song = str(record["song_id"])
        materialization[song] = Path(record["materialization_manifest"]["path"])
        feature[song] = Path(record["feature_manifest"]["path"])
        dynamic[song] = Path(record["dynamic_crop_manifest"]["path"])
    return materialization, feature, dynamic


def _smoke_path(root: Path, suffix: str = "e") -> Path:
    return _write_json(
        root
        / f"{launcher.V3_CAPACITY_SMOKE_ARTIFACT_PREFIX}{suffix * 64}.json",
        {"fixture": True},
    )


def _smoke_payload(
    candidate: object,
    *,
    runtime: bool = True,
    capacity: bool = True,
    clean: bool = True,
    commit: str = "a" * 40,
) -> dict[str, object]:
    empty_status_sha256 = hashlib.sha256(b"").hexdigest()
    source_files = {
        role: {
            "path": str(
                Path("/private/tmp/fake-v3-capacity-repository") / relative
            ),
            "file_sha256": "c" * 64,
            "repo_relative_path": relative,
            "observed_commit_blob_sha256": "c" * 64,
            "matches_observed_commit": clean,
        }
        for role, relative in (
            launcher.V3_CAPACITY_SMOKE_SOURCE_RELATIVE_PATHS.items()
        )
    }
    return {
        "artifact_content_sha256": "e" * 64,
        "artifact_family": "dynamic_crop_v3_capacity_smoke",
        "dynamic_crop_version": 3,
        "activation_checkpointing": dict(
            launcher.V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
        ),
        "candidate": candidate.to_dict(),
        "semantic_metrics_authorized": False,
        "model_selection_authorized": False,
        "execution": {
            "runtime_execution_passed": runtime,
            "within_release_memory_budget": capacity,
            "capacity_smoke_passed": runtime and capacity,
            "activation_checkpointing": dict(
                launcher.V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
            ),
            "semantic_metrics_authorized": False,
            "model_selection_authorized": False,
        },
        "code_provenance": {
            "runner_contract": launcher.V3_CAPACITY_SMOKE_RUNNER_CONTRACT,
            "release_trust_authorized": clean,
            "repository_status_clean": clean,
            "repository_status_sha256": (
                empty_status_sha256 if clean else "1" * 64
            ),
            "source_files_match_observed_commit": clean,
            "tracked_diff_policy": (
                "require_clean" if clean else "record_development"
            ),
            "expected_git_commit": commit,
            "observed_git_commit": commit,
            "source_files": source_files,
        },
    }


def _create_plan(
    root: Path,
    bundle: launcher._ValidatedManifestSet,
    *,
    smoke_payload: dict[str, object] | None = None,
    selectors: tuple[str, ...] | None = ("compact-balanced",),
) -> tuple[Path, Path]:
    candidate = launcher.default_dynamic_crop_v3_capacity_candidates()[0]
    smoke = _smoke_path(root)
    payload = smoke_payload or _smoke_payload(candidate)
    materialization, feature, dynamic = _manifest_maps(bundle)
    token = root.name.replace(".", "-")
    with (
        patch.object(launcher, "load_dynamic_crop_v3_capacity_smoke", return_value=payload),
        patch.object(launcher, "_capture_source_provenance", return_value=_fake_source()),
        patch.object(launcher, "_validate_manifest_set", return_value=bundle),
    ):
        plan = launcher.create_dynamic_crop_v3_capacity_fold_plan(
            materialization_manifests=materialization,
            feature_manifests=feature,
            dynamic_crop_manifests=dynamic,
            selection_dataset_manifest_path=Path(
                bundle.selection_dataset_pointer["path"]
            ),
            capacity_smoke_paths=(smoke,),
            checkpoint_root=Path(f"/Volumes/CrossCutMedia/v3-fold-fixture-{token}"),
            evidence_directory=Path(
                f"/Volumes/CrossCutMedia/v3-fold-fixture-{token}-evidence"
            ),
            output_directory=root / "plans",
            candidate_selectors=selectors,
        )
    return plan, smoke


class DynamicCropV3CapacityFoldLauncherTests(unittest.TestCase):
    def test_admitted_candidate_emits_three_exact_v3_inactive_commands(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            bundle = _fake_manifest_set(root)
            candidate = launcher.default_dynamic_crop_v3_capacity_candidates()[0]
            payload = _smoke_payload(candidate)
            plan_path, _smoke = _create_plan(root, bundle, smoke_payload=payload)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(1, len(plan["candidates"]))
            self.assertEqual(3, len(plan["runs"]))
            self.assertFalse(plan["capacity_selection_authorized"])
            self.assertEqual(
                dict(launcher.V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE),
                plan["capacity_fold_contract"]["activation_checkpointing"],
            )
            for run in plan["runs"]:
                args = run["train_cli_args"]
                self.assertEqual("3", args[args.index("--dynamic-crop-version") + 1])
                self.assertEqual(
                    candidate.contract_dict()["dynamic_crop_loss_contract"],
                    json.loads(
                        args[args.index("--dynamic-crop-loss-contract-json") + 1]
                    ),
                )
                for flag, key in (
                    ("--source-pretrain-steps", "source_pretraining_steps"),
                    (
                        "--source-pretrain-warmup-steps",
                        "source_pretraining_warmup_steps",
                    ),
                    (
                        "--minimum-finetune-optimizer-steps",
                        "minimum_finetune_optimizer_steps",
                    ),
                    (
                        "--minimum-checkpoint-selection-finetune-steps",
                        "minimum_checkpoint_selection_finetune_steps",
                    ),
                ):
                    self.assertEqual(
                        str(launcher.PROBE_TRAINING_SCHEDULE[key]),
                        args[args.index(flag) + 1],
                    )
                self.assertNotIn("--authorize-production-activation", args)
                self.assertNotIn("--production-registry", args)
                self.assertEqual(
                    dict(
                        launcher.V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
                    ),
                    run["activation_checkpointing"],
                )
                self.assertFalse(run["fold_selection_authorized"])
                self.assertFalse(run["capacity_selection_authorized"])

    def test_failed_or_dirty_smoke_cannot_admit_candidate(self) -> None:
        candidate = launcher.default_dynamic_crop_v3_capacity_candidates()[0]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            smoke = _smoke_path(root)
            for payload in (
                _smoke_payload(candidate, runtime=False),
                _smoke_payload(candidate, capacity=False),
                _smoke_payload(candidate, clean=False),
            ):
                with self.subTest(payload=payload):
                    with patch.object(
                        launcher,
                        "load_dynamic_crop_v3_capacity_smoke",
                        return_value=payload,
                    ), patch.object(
                        launcher,
                        "_capture_source_provenance",
                        return_value=_fake_source(),
                    ):
                        with self.assertRaisesRegex(
                            launcher.DynamicCropV3CapacityFoldLauncherError,
                            "did not pass clean",
                        ):
                            launcher._admit_capacity_smokes(
                                (smoke,),
                                expected_source_provenance=_fake_source(),
                            )

    def test_non_rematerialized_smoke_cannot_admit_candidate(self) -> None:
        candidate = launcher.default_dynamic_crop_v3_capacity_candidates()[0]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            smoke = _smoke_path(root)
            for location in ("root", "execution"):
                for enabled, previous in ((False, False), (1, 0)):
                    payload = _smoke_payload(candidate)
                    target = (
                        payload if location == "root" else payload["execution"]
                    )
                    target["activation_checkpointing"] = {
                        **dict(
                            launcher.V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
                        ),
                        "enabled": enabled,
                        "previous_value": previous,
                    }
                    with self.subTest(
                        location=location,
                        enabled=enabled,
                    ), patch.object(
                        launcher,
                        "load_dynamic_crop_v3_capacity_smoke",
                        return_value=payload,
                    ), patch.object(
                        launcher,
                        "_capture_source_provenance",
                        return_value=_fake_source(),
                    ):
                        with self.assertRaisesRegex(
                            launcher.DynamicCropV3CapacityFoldLauncherError,
                            "did not pass clean",
                        ):
                            launcher._admit_capacity_smokes(
                                (smoke,),
                                expected_source_provenance=_fake_source(),
                            )

    def test_rehashed_development_smoke_cannot_forge_clean_admission(self) -> None:
        candidate = launcher.default_dynamic_crop_v3_capacity_candidates()[0]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            smoke = _smoke_path(root)
            payload = _smoke_payload(candidate)
            code = payload["code_provenance"]
            assert isinstance(code, dict)
            code["tracked_diff_policy"] = "record_development"
            code["repository_status_sha256"] = "1" * 64
            source_files = code["source_files"]
            assert isinstance(source_files, dict)
            first = next(iter(source_files.values()))
            assert isinstance(first, dict)
            first["matches_observed_commit"] = False
            with patch.object(
                launcher,
                "load_dynamic_crop_v3_capacity_smoke",
                return_value=payload,
            ), patch.object(
                launcher,
                "_capture_source_provenance",
                return_value=_fake_source(),
            ):
                with self.assertRaisesRegex(
                    launcher.DynamicCropV3CapacityFoldLauncherError,
                    "did not pass clean",
                ):
                    launcher._admit_capacity_smokes(
                        (smoke,),
                        expected_source_provenance=_fake_source(),
                    )

    def test_candidate_without_its_own_smoke_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            bundle = _fake_manifest_set(root)
            second = launcher.default_dynamic_crop_v3_capacity_candidates()[1]
            with self.assertRaisesRegex(
                launcher.DynamicCropV3CapacityFoldLauncherError,
                "lacks an admitted",
            ):
                _create_plan(
                    root,
                    bundle,
                    selectors=(second.candidate_name,),
                )

    def test_preflight_replays_smoke_and_v3_command(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            bundle = _fake_manifest_set(root)
            candidate = launcher.default_dynamic_crop_v3_capacity_candidates()[0]
            payload = _smoke_payload(candidate)
            plan, _smoke = _create_plan(root, bundle, smoke_payload=payload)
            with (
                patch.object(
                    launcher,
                    "load_dynamic_crop_v3_capacity_smoke",
                    return_value=payload,
                ),
                patch.object(launcher, "_capture_source_provenance", return_value=_fake_source()),
                patch.object(launcher, "_validate_manifest_set", return_value=bundle),
                patch.object(
                    launcher.v2_launcher.legacy,
                    "_validate_storage",
                    return_value=20 * 1024**3,
                ),
            ):
                preflight = launcher.preflight_dynamic_crop_v3_capacity_fold(
                    plan,
                    candidate_selector=candidate.candidate_name,
                    held_out_song_id=launcher.SELECTION_SONG_IDS[-1],
                )
            self.assertEqual(3, int(preflight.train_cli_args[
                preflight.train_cli_args.index("--dynamic-crop-version") + 1
            ]))
            self.assertFalse(preflight.to_dict()["capacity_selection_authorized"])
            self.assertEqual(2, len(preflight.training_song_ids))

    def test_rehashed_plan_cannot_add_activation_authority(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            bundle = _fake_manifest_set(root)
            candidate = launcher.default_dynamic_crop_v3_capacity_candidates()[0]
            payload = _smoke_payload(candidate)
            plan, _smoke = _create_plan(root, bundle, smoke_payload=payload)
            value = json.loads(plan.read_text(encoding="utf-8"))
            value["capacity_selection_authorized"] = True
            value["plan_content_sha256"] = None
            digest = launcher._canonical_sha256(value)
            value["plan_content_sha256"] = digest
            forged = plan.parent / f"{launcher.V3_CAPACITY_FOLD_PLAN_PREFIX}{digest}.json"
            forged.write_text(json.dumps(value), encoding="utf-8")
            with (
                patch.object(
                    launcher,
                    "load_dynamic_crop_v3_capacity_smoke",
                    return_value=payload,
                ),
                patch.object(launcher, "_capture_source_provenance", return_value=_fake_source()),
            ):
                with self.assertRaisesRegex(
                    launcher.DynamicCropV3CapacityFoldLauncherError,
                    "contract differs",
                ):
                    launcher._strict_plan(forged)

    def test_completed_fold_writes_nonselecting_v3_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            candidate = launcher.default_dynamic_crop_v3_capacity_candidates()[0]
            heldout = launcher.SELECTION_SONG_IDS[2]
            checkpoint = root / launcher._run_id(candidate, heldout)
            evidence = root / "evidence"
            checkpoint.mkdir()
            evidence.mkdir()
            plan = _write_json(root / "plan.json", {"fixture": True})
            smoke = _write_json(root / "smoke.json", {"fixture": True})
            selection = _write_json(root / "selection.json", {"fixture": True})
            weights = checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
            weights.write_bytes(b"real-v3-weights-fixture")
            weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
            metadata = new_stage_edit_dynamic_crop_checkpoint_metadata_v3(
                "checkpoint-v3-fixture",
                DEFAULT_STAGE_EDIT_SCHEMA,
                candidate.model_config,
                training_steps=1120,
                weights_trained=True,
                training_data_id="v3-fixture-data",
            )
            metadata = replace(metadata, weight_sha256=weight_hash)
            _write_json(
                checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
                metadata.to_dict(),
            )
            base = asdict(candidate.base_loss_contract)
            dynamic_loss = asdict(candidate.dynamic_crop_loss_contract)
            report = {
                "schema_version": PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
                "run_kind": PRODUCTION_TRAINING_RUN_KIND,
                "checkpoint_scope": PRODUCTION_CHECKPOINT_SCOPE,
                "production_ready": False,
                "production_readiness_reason": "activation_not_explicitly_authorized",
                "model": {
                    "config": candidate.model_config.to_dict(),
                    "dynamic_crop_version": 3,
                    "dynamic_crop_v3_enabled": True,
                    "dynamic_crop_v2_enabled": False,
                    "dynamic_crop_model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
                    "dynamic_crop_target_contract": DYNAMIC_CROP_V3_TARGET_CONTRACT,
                    "dynamic_crop_output_contract": DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
                    "activation_checkpointing": dict(
                        launcher.V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
                    ),
                },
                "split": {
                    "training_song_ids": list(launcher.SELECTION_SONG_IDS[:2]),
                    "held_out_song_ids": [launcher.SELECTION_SONG_IDS[2]],
                    "held_out_used_for_model_selection": False,
                    "training_reference_ids": [],
                    "validation_reference_ids": [],
                    "held_out_reference_ids": [],
                },
                "optimization": {
                    "source_pretraining_optimizer_steps": 500,
                    "learning_rate_schedule": {
                        "source_pretraining_steps": 500,
                        "source_pretraining_peak_learning_rate": 3.0e-4,
                        "source_pretraining_window_length": 32,
                        "source_pretraining_warmup_steps": 50,
                    },
                    "minimum_finetune_optimizer_steps": 120,
                    "minimum_checkpoint_selection_finetune_steps": 60,
                    "unseen_song_used_for_model_selection": False,
                    "effective_loss_contract": base,
                    "effective_dynamic_crop_v3_loss_contract": dynamic_loss,
                    "effective_dynamic_crop_loss_contract_version": 3,
                },
                "held_out_evaluation": {
                    "passed": True,
                    "quality_gate": {"passed": True},
                    "training_song_ids": list(launcher.SELECTION_SONG_IDS[:2]),
                    "held_out_song_ids": [launcher.SELECTION_SONG_IDS[2]],
                    "held_out_reference_count": 0,
                    "checkpoint_id": metadata.checkpoint_id,
                    "weight_sha256": weight_hash,
                    "cut_threshold_frozen_before_heldout_forward": True,
                    "heldout_dynamic_crop_metrics_used_for_checkpoint_selection": False,
                    "dynamic_crop_version": 3,
                    "dynamic_crop_v2_metrics": None,
                    "dynamic_crop_v3_metrics": {
                        "schema_version": DYNAMIC_CROP_V3_SCHEMA_VERSION,
                        "model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
                        "target_contract": DYNAMIC_CROP_V3_TARGET_CONTRACT,
                        "output_contract": DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
                        "loss_contract": DYNAMIC_CROP_V3_LOSS_CONTRACT,
                        "legacy_v2_checkpoint_or_output_used": False,
                        "production_activation_authorized": False,
                    },
                },
                "checkpoint": {
                    "metadata_filename": DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
                    "weight_filename": DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
                    "checkpoint_id": metadata.checkpoint_id,
                    "training_steps": metadata.training_steps,
                    "weight_sha256": weight_hash,
                    "production_activation_authorized": False,
                },
            }
            _write_json(checkpoint / PRODUCTION_TRAINING_REPORT_FILENAME, report)
            _write_json(
                checkpoint / STAGE_EDIT_SELECTION_FREEZE_FILENAME,
                {
                    "heldout_metrics_present": False,
                    "heldout_used_for_model_or_threshold_selection": False,
                    "held_out_song_ids_reserved_for_one_forward": [
                        launcher.SELECTION_SONG_IDS[2]
                    ],
                    "checkpoint_id": metadata.checkpoint_id,
                    "checkpoint_weight_sha256": weight_hash,
                    "minimum_checkpoint_selection_finetune_steps": 60,
                    "selected_after_checkpoint_burn_in": True,
                    "training_reference_ids": [],
                    "validation_reference_ids": [],
                },
            )
            preflight = launcher.DynamicCropV3CapacityFoldPreflight(
                plan_path=plan,
                plan_content_sha256="a" * 64,
                candidate_name=candidate.candidate_name,
                candidate_id=candidate.candidate_id,
                capacity_smoke_path=smoke,
                held_out_song_id=heldout,
                training_song_ids=tuple(launcher.SELECTION_SONG_IDS[:2]),
                checkpoint_directory=checkpoint,
                evidence_directory=evidence,
                selection_dataset_manifest_path=selection,
                reference_ids_by_song={},
                train_cli_args=(),
                command_argv=(),
                free_bytes=1,
            )
            with (
                patch.object(launcher, "PRODUCTION_VOLUME_ROOT", root),
                patch.object(
                    launcher,
                    "_revalidate_preflight_receipt_binding",
                ),
            ):
                receipt = launcher.record_dynamic_crop_v3_capacity_fold_evidence(
                    preflight
                )
            selection_pointer = launcher.v2_launcher._pointer(
                selection, "selection dataset"
            )
            bundle = launcher._ValidatedManifestSet(
                selection_dataset_pointer=selection_pointer,
                selection_dataset_content_sha256="d" * 64,
                song_records=(),
                reference_ids_by_song={},
                supervision_paths_by_song=MappingProxyType({}),
                snapshot_paths_by_song=MappingProxyType({}),
                dynamic_manifest_paths_by_song=MappingProxyType({}),
            )
            admitted_smoke = launcher._AdmittedCapacitySmoke(
                candidate=candidate,
                path=smoke,
                pointer=launcher.v2_launcher._pointer(smoke, "capacity smoke"),
                payload={"artifact_content_sha256": "e" * 64},
                source_commit="a" * 40,
            )
            train_args: tuple[str, ...] = ()
            plan_run = {
                "run_id": launcher._run_id(candidate, heldout),
                "candidate_name": candidate.candidate_name,
                "candidate_id": candidate.candidate_id,
                "capacity_smoke_content_sha256": "e" * 64,
                "held_out_song_id": heldout,
                "training_song_ids": list(launcher.SELECTION_SONG_IDS[:2]),
                "checkpoint_directory": str(checkpoint),
                "evidence_directory": str(evidence),
                "train_cli_args": [],
                "command_argv": launcher._command_argv(
                    train_args,
                    repository_root=Path(
                        str(_fake_source()["repository_root"])
                    ),
                ),
                "dynamic_crop_version": 3,
                "activation_checkpointing": dict(
                    launcher.V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
                ),
                "fold_selection_authorized": False,
                "capacity_selection_authorized": False,
                "production_activation_authorized": False,
                "production_registry_activation_requested": False,
            }
            plan_payload = {
                "plan_content_sha256": "a" * 64,
                "source_provenance": _fake_source(),
                "capacity_smokes": [],
                "candidates": [candidate.to_dict()],
                "selection_dataset_manifest": dict(selection_pointer),
                "selection_dataset_content_sha256": "d" * 64,
                "songs": [],
                "checkpoint_root": str(root),
                "evidence_directory": str(evidence),
                "runs": [plan_run],
            }
            with (
                patch.object(launcher, "PRODUCTION_VOLUME_ROOT", root),
                patch.object(
                    launcher,
                    "_admitted_smokes_from_plan",
                    return_value={candidate.candidate_id: admitted_smoke},
                ),
                patch.object(
                    launcher, "_strict_plan", return_value=plan_payload
                ),
                patch.object(
                    launcher,
                    "_candidates_from_plan",
                    return_value=(candidate,),
                ),
                patch.object(
                    launcher, "_manifest_maps_from_plan", return_value=({}, {}, {})
                ),
                patch.object(
                    launcher, "_validate_manifest_set", return_value=bundle
                ),
                patch.object(
                    launcher, "_build_train_cli_args", return_value=train_args
                ),
                patch.object(
                    launcher,
                    "_validate_completed_v3_fold",
                    return_value=(report, {}, metadata),
                ),
            ):
                loaded = launcher.load_dynamic_crop_v3_capacity_fold_evidence(
                    receipt
                )
                cloned = json.loads(json.dumps(loaded))
                cloned["validation_song_id"] = launcher.SELECTION_SONG_IDS[0]
                cloned["training_song_ids"] = sorted(
                    set(launcher.SELECTION_SONG_IDS)
                    - {launcher.SELECTION_SONG_IDS[0]}
                )
                cloned["artifact_content_sha256"] = None
                cloned_digest = launcher._canonical_sha256(cloned)
                cloned["artifact_content_sha256"] = cloned_digest
                cloned_path = _write_json(
                    evidence
                    / f"{launcher.V3_CAPACITY_FOLD_EVIDENCE_PREFIX}"
                    f"{cloned_digest}.json",
                    cloned,
                )
                with self.assertRaisesRegex(
                    launcher.DynamicCropV3CapacityFoldLauncherError,
                    "checkpoint paths differ|selected plan run",
                ):
                    launcher.load_dynamic_crop_v3_capacity_fold_evidence(
                        cloned_path
                    )
            self.assertFalse(loaded["fold_selection_authorized"])
            self.assertFalse(loaded["capacity_selection_authorized"])
            self.assertFalse(loaded["production_activation_authorized"])
            self.assertEqual(
                dict(launcher.V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE),
                loaded["activation_checkpointing"],
            )
            self.assertTrue(
                loaded["all_three_fold_receipts_required_before_future_selection"]
            )
            self.assertEqual(
                DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
                Path(loaded["checkpoint_metadata"]["path"]).name,
            )
            forged = json.loads(json.dumps(loaded))
            forged["activation_checkpointing"]["enabled"] = False
            forged["artifact_content_sha256"] = None
            forged_digest = launcher._canonical_sha256(forged)
            forged["artifact_content_sha256"] = forged_digest
            forged_path = _write_json(
                evidence
                / f"{launcher.V3_CAPACITY_FOLD_EVIDENCE_PREFIX}"
                f"{forged_digest}.json",
                forged,
            )
            with patch.object(launcher, "PRODUCTION_VOLUME_ROOT", root):
                with self.assertRaisesRegex(
                    launcher.DynamicCropV3CapacityFoldLauncherError,
                    "contract or authority differs",
                ):
                    launcher.load_dynamic_crop_v3_capacity_fold_evidence(
                        forged_path
                    )


if __name__ == "__main__":
    unittest.main()
