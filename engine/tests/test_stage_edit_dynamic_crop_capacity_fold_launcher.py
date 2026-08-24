from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut import stage_edit_dynamic_crop_capacity_fold_launcher as launcher


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path.resolve()


def _fake_source_provenance() -> dict[str, object]:
    return {
        "contract": "clean_head_and_exact_interpreter_source_blobs_v1",
        "repository_root": "/private/tmp/fake-capacity-repository",
        "head_commit_oid": "a" * 40,
        "repository_status_clean": True,
        "repository_status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
        "source_files": {
            "engine/crosscut/fake.py": {
                "git_blob_oid": "b" * 40,
                "file_sha256": "c" * 64,
            }
        },
    }


def _fake_manifest_set(root: Path) -> launcher._ValidatedManifestSet:
    selection = _write_json(root / "selection-dataset.json", {"fixture": True})
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
                "materialization_manifest": launcher._pointer(
                    materialization, f"{song_id} materialization"
                ),
                "materialization_outputs": [],
                "feature_manifest": launcher._pointer(
                    feature, f"{song_id} feature"
                ),
                "feature_snapshots": [],
                "dynamic_crop_manifest": launcher._pointer(
                    dynamic, f"{song_id} dynamic"
                ),
            }
        )
    return launcher._ValidatedManifestSet(
        selection_dataset_pointer=launcher._pointer(selection, "selection dataset"),
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
        song_id = str(record["song_id"])
        materialization[song_id] = Path(
            str(record["materialization_manifest"]["path"])
        )
        feature[song_id] = Path(str(record["feature_manifest"]["path"]))
        dynamic[song_id] = Path(str(record["dynamic_crop_manifest"]["path"]))
    return materialization, feature, dynamic


def _create_plan(
    root: Path,
    bundle: launcher._ValidatedManifestSet,
    *,
    candidates: tuple[str, ...] | None = ("compact-balanced",),
) -> Path:
    materialization, feature, dynamic = _manifest_maps(bundle)
    token = root.name.replace(".", "-")
    with (
        patch.object(
            launcher,
            "_capture_source_provenance",
            return_value=_fake_source_provenance(),
        ),
        patch.object(launcher, "_validate_manifest_set", return_value=bundle),
    ):
        return launcher.create_capacity_fold_plan(
            materialization_manifests=materialization,
            feature_manifests=feature,
            dynamic_crop_manifests=dynamic,
            selection_dataset_manifest_path=Path(
                bundle.selection_dataset_pointer["path"]
            ),
            checkpoint_root=Path(f"/Volumes/CrossCutMedia/capacity-fixture-{token}"),
            evidence_directory=Path(
                f"/Volumes/CrossCutMedia/capacity-fixture-{token}-evidence"
            ),
            output_directory=root / "plans",
            candidate_selectors=candidates,
        )


class DynamicCropCapacityFoldLauncherTests(unittest.TestCase):
    def test_one_candidate_emits_three_exact_inactive_fold_commands(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            bundle = _fake_manifest_set(root)
            plan_path = _create_plan(root, bundle)
            with (
                patch.object(
                    launcher,
                    "_capture_source_provenance",
                    return_value=_fake_source_provenance(),
                ),
                patch.object(
                    launcher, "_validate_manifest_set", return_value=bundle
                ),
                patch.object(
                    launcher.legacy,
                    "_validate_storage",
                    return_value=20 * 1024**3,
                ),
            ):
                result = launcher.preflight_capacity_fold(
                    plan_path,
                    candidate_selector="compact-balanced",
                    held_out_song_id=launcher.REBEL_HEART,
                )

            args = result.train_cli_args
            self.assertEqual(15, args.count("--input"))
            self.assertEqual(15, args.count("--song-id"))
            self.assertEqual(15, args.count("--feature-snapshot"))
            self.assertEqual(3, args.count("--dynamic-crop-manifest"))
            self.assertEqual(2, args.count("--required-training-song"))
            self.assertEqual(1, args.count("--held-out-song"))
            self.assertEqual(
                tuple(
                    sorted(
                        set(launcher.SELECTION_SONG_IDS)
                        - {launcher.REBEL_HEART}
                    )
                ),
                result.training_song_ids,
            )
            for flag in (
                "--model-config-json",
                "--base-loss-contract-json",
                "--dynamic-crop-loss-contract-json",
                "--epochs",
                "--patience",
                "--batch-size",
                "--cut-pretrain-steps",
                "--source-pretrain-steps",
                "--source-pretrain-learning-rate",
                "--source-pretrain-window-length",
                "--source-pretrain-warmup-steps",
                "--minimum-finetune-optimizer-steps",
                "--minimum-checkpoint-selection-finetune-steps",
                "--device",
            ):
                self.assertIn(flag, args)
            self.assertEqual(
                str(launcher.PROBE_TRAINING_SCHEDULE["requested_epochs"]),
                args[args.index("--epochs") + 1],
            )
            self.assertEqual("gpu", args[args.index("--device") + 1])
            self.assertEqual(
                str(
                    launcher.PROBE_TRAINING_SCHEDULE[
                        "source_pretraining_steps"
                    ]
                ),
                args[args.index("--source-pretrain-steps") + 1],
            )
            self.assertEqual(
                str(
                    launcher.PROBE_TRAINING_SCHEDULE[
                        "minimum_checkpoint_selection_finetune_steps"
                    ]
                ),
                args[
                    args.index(
                        "--minimum-checkpoint-selection-finetune-steps"
                    )
                    + 1
                ],
            )
            self.assertNotIn("--authorize-production-activation", args)
            self.assertNotIn("--production-registry", args)
            self.assertNotIn("--professional-montage-grammar-profile", args)
            self.assertEqual(
                (
                    "/usr/bin/env",
                    "PYTHONPATH=/private/tmp/fake-capacity-repository/engine",
                    "python3",
                    "-m",
                    launcher.TRAIN_MODULE,
                ),
                result.command_argv[:5],
            )

    def test_omitted_candidate_emits_all_defaults_times_three_folds(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            bundle = _fake_manifest_set(root)
            plan_path = _create_plan(root, bundle, candidates=None)
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            expected_candidates = launcher.default_dynamic_crop_v2_probe_candidates()
            self.assertEqual(len(expected_candidates), len(payload["candidates"]))
            self.assertEqual(
                len(expected_candidates) * len(launcher.SELECTION_SONG_IDS),
                len(payload["runs"]),
            )
            self.assertEqual(
                {
                    (candidate.candidate_id, song_id)
                    for candidate in expected_candidates
                    for song_id in launcher.SELECTION_SONG_IDS
                },
                {
                    (run["candidate_id"], run["held_out_song_id"])
                    for run in payload["runs"]
                },
            )

    def test_plan_is_content_addressed_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            bundle = _fake_manifest_set(root)
            plan_path = _create_plan(root, bundle)
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(
                f"{launcher.CAPACITY_FOLD_PLAN_PREFIX}"
                f"{payload['plan_content_sha256']}.json",
                plan_path.name,
            )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                _create_plan(root, bundle)

    def test_plan_tampering_is_rejected_before_manifest_replay(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            bundle = _fake_manifest_set(root)
            plan_path = _create_plan(root, bundle)
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            payload["runs"][0]["production_activation_authorized"] = True
            plan_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                launcher.CapacityFoldLauncherError, "content hash mismatch"
            ):
                launcher._strict_plan(plan_path)

    def test_rehashed_plan_cannot_change_an_unselected_fold_command(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            bundle = _fake_manifest_set(root)
            plan_path = _create_plan(root, bundle)
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            payload["runs"][0]["train_cli_args"].append(
                "--authorize-production-activation"
            )
            payload["plan_content_sha256"] = None
            digest = launcher._canonical_sha256(payload)
            payload["plan_content_sha256"] = digest
            forged = plan_path.parent / (
                f"{launcher.CAPACITY_FOLD_PLAN_PREFIX}{digest}.json"
            )
            forged.write_text(json.dumps(payload), encoding="utf-8")
            with (
                patch.object(
                    launcher,
                    "_capture_source_provenance",
                    return_value=_fake_source_provenance(),
                ),
                patch.object(
                    launcher, "_validate_manifest_set", return_value=bundle
                ),
                patch.object(
                    launcher.legacy,
                    "_validate_storage",
                    return_value=20 * 1024**3,
                ),
            ):
                with self.assertRaisesRegex(
                    launcher.CapacityFoldLauncherError,
                    "run commands differ from replay",
                ):
                    launcher.preflight_capacity_fold(
                        forged,
                        candidate_selector="compact-balanced",
                        held_out_song_id=launcher.REBEL_HEART,
                    )

    def test_current_head_must_match_the_plan(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            bundle = _fake_manifest_set(root)
            plan_path = _create_plan(root, bundle)
            changed = _fake_source_provenance()
            changed["head_commit_oid"] = "e" * 40
            with patch.object(
                launcher, "_capture_source_provenance", return_value=changed
            ):
                with self.assertRaisesRegex(
                    launcher.CapacityFoldLauncherError,
                    "clean HEAD/source blobs differ",
                ):
                    launcher._strict_plan(plan_path)

    def test_quarantined_path_is_rejected_before_resolution_or_open(self) -> None:
        title = "X" + "OXZ"
        raw_value = f"{launcher.BANG_BANG}=/tmp/{title}/manifest.json"
        with patch.object(launcher.Path, "resolve") as resolve:
            with self.assertRaisesRegex(
                launcher.CapacityFoldLauncherError, "quarantined song"
            ):
                launcher._keyed_paths(
                    (raw_value,), "--materialization-manifest"
                )
        resolve.assert_not_called()

    def test_quarantined_parent_symlink_is_rejected_after_canonical_resolution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            title = "X" + "OXZ"
            target = root / title
            target.mkdir()
            alias = root / "safe-looking-parent"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(
                launcher.CapacityFoldLauncherError, "quarantined song"
            ):
                launcher._scan_payload_for_quarantined_text(
                    {"path": str(alias / "manifest.json")}, "fixture"
                )

    def test_safe_parent_symlink_is_not_an_admissible_path_alias(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            target = root / "safe-target"
            target.mkdir()
            alias = root / "safe-alias"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(
                launcher.CapacityFoldLauncherError, "symlink component"
            ):
                launcher._absolute_safe_path(
                    alias / "manifest.json", "fixture manifest"
                )

    def test_plan_output_cannot_create_a_planned_checkpoint_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            volume = root / "ssd"
            checkpoint = volume / "checkpoints"
            evidence = volume / "evidence"
            bundle = _fake_manifest_set(root)
            materialization, feature, dynamic = _manifest_maps(bundle)
            candidate = launcher.default_dynamic_crop_v2_probe_candidates()[0]
            blocked_run = checkpoint / launcher._run_id(
                candidate, launcher.BANG_BANG
            )
            with (
                patch.object(launcher, "PRODUCTION_VOLUME_ROOT", volume),
                patch.object(
                    launcher,
                    "_capture_source_provenance",
                    return_value=_fake_source_provenance(),
                ),
                patch.object(
                    launcher, "_validate_manifest_set", return_value=bundle
                ),
            ):
                with self.assertRaisesRegex(
                    launcher.CapacityFoldLauncherError,
                    "plan output directory must not overlap",
                ):
                    launcher.create_capacity_fold_plan(
                        materialization_manifests=materialization,
                        feature_manifests=feature,
                        dynamic_crop_manifests=dynamic,
                        selection_dataset_manifest_path=Path(
                            bundle.selection_dataset_pointer["path"]
                        ),
                        checkpoint_root=checkpoint,
                        evidence_directory=evidence,
                        output_directory=blocked_run,
                        candidate_selectors=(candidate.candidate_name,),
                    )
            self.assertFalse(blocked_run.exists())

    def test_evidence_and_plan_output_directories_must_be_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            with self.assertRaisesRegex(
                launcher.CapacityFoldLauncherError,
                "plan output directory and evidence directory must not overlap",
            ):
                launcher._validate_artifact_path_isolation(
                    plan_path=root / "artifacts",
                    plan_label="plan output directory",
                    evidence_directory=root / "artifacts" / "evidence",
                    planned_checkpoint_directories=(
                        root / "checkpoints" / "run-one",
                    ),
                )

    def test_evidence_cannot_be_nested_in_a_planned_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            planned_checkpoint = root / "checkpoints" / "run-one"
            with self.assertRaisesRegex(
                launcher.CapacityFoldLauncherError,
                "evidence directory must not overlap",
            ):
                launcher._validate_artifact_path_isolation(
                    plan_path=root / "plans",
                    plan_label="plan output directory",
                    evidence_directory=planned_checkpoint / "evidence",
                    planned_checkpoint_directories=(planned_checkpoint,),
                )

    def test_materialization_outputs_rehash_every_dataset_and_supervision(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            reference = "synthetic-reference"
            dataset = _write_json(root / "dataset.json", {"reference_id": reference})
            supervision = _write_json(
                root / "supervision.json", {"reference_id": reference}
            )
            manifest = _write_json(
                root / "materialization.json",
                {
                    "outputs": [
                        {
                            "reference_id": reference,
                            "dataset_path": str(dataset),
                            "dataset_file_sha256": launcher._stable_file_sha256(
                                dataset, "dataset"
                            ),
                            "supervision_path": str(supervision),
                            "supervision_file_sha256": launcher._stable_file_sha256(
                                supervision, "supervision"
                            ),
                        }
                    ]
                },
            )
            binding = launcher.legacy._MaterializationBinding(
                reference_ids=(reference,),
                supervision_paths={reference: supervision},
                supervision_sha256={
                    reference: launcher._stable_file_sha256(
                        supervision, "supervision"
                    )
                },
                attribution_paths={reference: root / "attribution.json"},
                attribution_sha256={reference: "a" * 64},
                known_shot_counts={reference: 1},
            )
            records = launcher._materialization_output_records(
                manifest,
                song_id=launcher.BANG_BANG,
                materialization=binding,
            )
            self.assertEqual(reference, records[0]["reference_id"])
            supervision.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                launcher.CapacityFoldLauncherError, "binding mismatch"
            ):
                launcher._materialization_output_records(
                    manifest,
                    song_id=launcher.BANG_BANG,
                    materialization=binding,
                )

    def test_feature_snapshot_binds_file_hash_reference_and_song(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            reference = "synthetic-reference"
            snapshot_sha = "a" * 64
            snapshot = _write_json(
                root
                / f"{reference}-stageedit-feature-snapshot-{snapshot_sha[:16]}.json",
                {"fixture": True},
            )
            manifest = _write_json(
                root / "feature.json",
                {
                    "song_id": launcher.BLACKHOLE,
                    "snapshots": [
                        {
                            "reference_id": reference,
                            "path": str(snapshot),
                            "snapshot_sha256": snapshot_sha,
                        }
                    ],
                },
            )
            materialization = launcher.legacy._MaterializationBinding(
                reference_ids=(reference,),
                supervision_paths={reference: root / "supervision.json"},
                supervision_sha256={reference: "b" * 64},
                attribution_paths={reference: root / "attribution.json"},
                attribution_sha256={reference: "c" * 64},
                known_shot_counts={reference: 1},
            )
            feature = launcher.legacy._FeatureBinding(
                snapshot_paths={reference: snapshot}
            )
            restored = SimpleNamespace(
                snapshot_sha256=snapshot_sha,
                sequence=SimpleNamespace(
                    sequence_id=(
                        f"stageedit-features:{launcher.BLACKHOLE}:{reference}"
                    )
                ),
            )
            with patch.object(
                launcher.legacy.StageEditFeatureSnapshot,
                "from_dict",
                return_value=restored,
            ):
                records = launcher._feature_snapshot_records(
                    manifest,
                    song_id=launcher.BLACKHOLE,
                    materialization=materialization,
                    feature=feature,
                )
            self.assertEqual(snapshot_sha, records[0]["snapshot_sha256"])
            self.assertEqual(
                launcher._stable_file_sha256(snapshot, "snapshot"),
                records[0]["file_sha256"],
            )

    def test_real_git_provenance_requires_clean_tracked_blob(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            tracked = root / "tracked.py"
            tracked.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(("git", "init", str(root)), check=True, capture_output=True)
            subprocess.run(
                ("git", "-C", str(root), "config", "user.email", "fixture@example.invalid"),
                check=True,
            )
            subprocess.run(
                ("git", "-C", str(root), "config", "user.name", "Fixture"),
                check=True,
            )
            subprocess.run(
                ("git", "-C", str(root), "add", "tracked.py"), check=True
            )
            subprocess.run(
                ("git", "-C", str(root), "commit", "-m", "fixture"),
                check=True,
                capture_output=True,
            )
            provenance = launcher._capture_source_provenance(
                root, source_relative_paths=("tracked.py",)
            )
            self.assertTrue(provenance["repository_status_clean"])
            self.assertEqual(
                launcher._stable_file_sha256(tracked, "tracked source"),
                provenance["source_files"]["tracked.py"]["file_sha256"],
            )
            tracked.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(
                launcher.CapacityFoldLauncherError, "completely clean"
            ):
                launcher._capture_source_provenance(
                    root, source_relative_paths=("tracked.py",)
                )

    def test_successful_launch_records_fold_evidence_and_failure_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            checkpoint = root / "checkpoint"
            dataset = _write_json(root / "dataset.json", {})
            candidate = launcher.default_dynamic_crop_v2_probe_candidates()[0]
            preflight = launcher.CapacityFoldPreflight(
                plan_path=root / "plan.json",
                plan_content_sha256="a" * 64,
                candidate_name=candidate.candidate_name,
                candidate_id=candidate.candidate_id,
                held_out_song_id=launcher.REBEL_HEART,
                training_song_ids=tuple(
                    sorted(
                        set(launcher.SELECTION_SONG_IDS)
                        - {launcher.REBEL_HEART}
                    )
                ),
                checkpoint_directory=checkpoint,
                evidence_directory=root / "evidence",
                selection_dataset_manifest_path=dataset,
                reference_ids_by_song={},
                train_cli_args=("--held-out-song", launcher.REBEL_HEART),
                command_argv=("python3", "-m", launcher.TRAIN_MODULE),
                free_bytes=20 * 1024**3,
            )

            def successful_train(_args: object) -> int:
                checkpoint.mkdir()
                for filename in (
                    launcher.PRODUCTION_TRAINING_REPORT_FILENAME,
                    launcher.STAGE_EDIT_SELECTION_FREEZE_FILENAME,
                    launcher.DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
                    launcher.DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME,
                ):
                    (checkpoint / filename).write_text("fixture", encoding="utf-8")
                return 0

            evidence = root / "evidence.json"
            with (
                patch.object(
                    launcher, "preflight_capacity_fold", return_value=preflight
                ),
                patch.object(
                    launcher, "production_train_main", side_effect=successful_train
                ),
                patch.object(
                    launcher,
                    "_strict_plan",
                    return_value={"source_provenance": _fake_source_provenance()},
                ),
                patch.object(launcher, "_validate_source_provenance"),
                patch.object(launcher, "_require_same_source_provenance"),
                patch.object(
                    launcher,
                    "write_fold_run_evidence",
                    return_value=evidence,
                ) as write_evidence,
            ):
                code, observed, observed_evidence = launcher.launch_capacity_fold(
                    root / "plan.json",
                    candidate_selector=candidate.candidate_name,
                    held_out_song_id=launcher.REBEL_HEART,
                )
            self.assertEqual(0, code)
            self.assertIs(preflight, observed)
            self.assertEqual(evidence, observed_evidence)
            write_evidence.assert_called_once()
            call = write_evidence.call_args.kwargs
            self.assertEqual(candidate.candidate_id, call["candidate_id"])
            self.assertEqual(launcher.REBEL_HEART, call["validation_song_id"])

            with (
                patch.object(
                    launcher, "preflight_capacity_fold", return_value=preflight
                ),
                patch.object(
                    launcher,
                    "_strict_plan",
                    return_value={"source_provenance": _fake_source_provenance()},
                ),
                patch.object(launcher, "_require_same_source_provenance"),
                patch.object(launcher, "production_train_main", return_value=2),
                patch.object(launcher, "record_capacity_fold_evidence") as record,
            ):
                code, _, failed_evidence = launcher.launch_capacity_fold(
                    root / "plan.json",
                    candidate_selector=candidate.candidate_name,
                    held_out_song_id=launcher.REBEL_HEART,
                )
            self.assertEqual(2, code)
            self.assertIsNone(failed_evidence)
            record.assert_not_called()


if __name__ == "__main__":
    unittest.main()
