from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crosscut import stage_edit_production_fold_launcher as launcher
from crosscut.stage_edit_feature_snapshot_job import (
    FEATURE_SNAPSHOT_JOB_PROVENANCE_CONTRACTS,
    FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION,
)


def _production_contract() -> dict[str, object]:
    return {
        "artist": "IVE",
        "song_ids": list(launcher.PRODUCTION_SONG_IDS),
        "expected_reference_count_per_song": 5,
        "expected_source_count_per_song": 10,
        "sampling": dict(launcher.PRODUCTION_SAMPLING_CONTRACT),
        "recent_history_mask_contract": dict(
            launcher.PRODUCTION_RECENT_HISTORY_MASK_CONTRACT
        ),
        "held_out_used_for_model_selection": False,
        "synthetic_or_fake_training_data_allowed": False,
        "quality_threshold_override_allowed": False,
    }


def _write_plan(root: Path, checkpoint_root: Path) -> Path:
    dummy_hash = "a" * 64
    unsigned = {
        "schema_version": launcher.PLAN_SCHEMA_VERSION,
        "analysis_basis": launcher.PLAN_ANALYSIS_BASIS,
        "production_contract": _production_contract(),
        "required_volume": str(launcher.PRODUCTION_VOLUME_ROOT),
        "minimum_free_bytes": launcher.MINIMUM_FREE_BYTES,
        "songs": [
            {
                "song_id": song_id,
                "materialization_manifest": {
                    "path": str(root / f"{song_id}-materialization.json"),
                    "file_sha256": dummy_hash,
                },
                "feature_manifest": {
                    "path": str(root / f"{song_id}-feature.json"),
                    "file_sha256": dummy_hash,
                },
                "dynamic_crop_manifest": {
                    "path": str(root / f"{song_id}-dynamic.json"),
                    "file_sha256": dummy_hash,
                },
            }
            for song_id in launcher.PRODUCTION_SONG_IDS
        ],
        "folds": [
            {
                "fold_id": fold_id,
                "held_out_song_id": held_out,
                "checkpoint_directory": str(checkpoint_root / fold_id),
            }
            for fold_id, held_out in launcher.PRODUCTION_FOLDS.items()
        ],
    }
    plan_hash = launcher._canonical_sha256(unsigned)
    path = root / f"stageedit-production-fold-plan-{plan_hash[:16]}.json"
    path.write_text(
        json.dumps({**unsigned, "plan_sha256": plan_hash}),
        encoding="utf-8",
    )
    return path


class StageEditProductionFoldLauncherTests(unittest.TestCase):
    def test_production_feature_contract_is_exact_current_v3(self) -> None:
        self.assertEqual(FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION, 3)
        self.assertEqual(
            launcher.REQUIRED_FEATURE_PROVENANCE_CONTRACTS,
            frozenset(FEATURE_SNAPSHOT_JOB_PROVENANCE_CONTRACTS),
        )
        self.assertNotIn(
            "synchronized_source_causal_stage_edit_feature_snapshot_v2",
            launcher.REQUIRED_FEATURE_PROVENANCE_CONTRACTS,
        )

    def test_preflight_expands_fifteen_references_and_three_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            plan_path = _write_plan(root, root / "checkpoints")

            materializations: dict[str, launcher._MaterializationBinding] = {}
            features: dict[str, launcher._FeatureBinding] = {}
            for song_index, song_id in enumerate(launcher.PRODUCTION_SONG_IDS):
                references = tuple(
                    f"ref-{song_index}-{index}" for index in range(5)
                )
                materializations[song_id] = launcher._MaterializationBinding(
                    reference_ids=references,
                    supervision_paths={
                        value: root / f"{value}-supervision.json"
                        for value in references
                    },
                    supervision_sha256={value: "b" * 64 for value in references},
                    attribution_paths={
                        value: root / f"{value}-attribution.json"
                        for value in references
                    },
                    attribution_sha256={value: "c" * 64 for value in references},
                    known_shot_counts={value: 1 for value in references},
                )
                features[song_id] = launcher._FeatureBinding(
                    snapshot_paths={
                        value: root / f"{value}-snapshot.json"
                        for value in references
                    }
                )

            def materialization_side_effect(
                _pointer: object, *, song_id: str
            ) -> tuple[launcher.ManifestPointer, launcher._MaterializationBinding]:
                return (
                    launcher.ManifestPointer(root / "m.json", "d" * 64),
                    materializations[song_id],
                )

            def feature_side_effect(
                _pointer: object,
                *,
                song_id: str,
                materialization: object,
            ) -> tuple[launcher.ManifestPointer, launcher._FeatureBinding]:
                del materialization
                return (
                    launcher.ManifestPointer(root / "f.json", "e" * 64),
                    features[song_id],
                )

            def dynamic_side_effect(
                _pointer: object, *, song_id: str, materialization: object
            ) -> launcher.ManifestPointer:
                del materialization
                return launcher.ManifestPointer(
                    root / f"{song_id}-dynamic.json", "f" * 64
                )

            with (
                patch.object(launcher, "_validate_storage", return_value=20 * 1024**3),
                patch.object(
                    launcher,
                    "_validate_materialization_manifest",
                    side_effect=materialization_side_effect,
                ),
                patch.object(
                    launcher,
                    "_validate_feature_manifest",
                    side_effect=feature_side_effect,
                ),
                patch.object(
                    launcher,
                    "_validate_dynamic_crop_manifest",
                    side_effect=dynamic_side_effect,
                ),
            ):
                result = launcher.preflight_stage_edit_production_fold(
                    plan_path,
                    fold_id="heldout-xoxz",
                )

            args = result.train_cli_args
            self.assertEqual(15, args.count("--input"))
            self.assertEqual(15, args.count("--song-id"))
            self.assertEqual(15, args.count("--feature-snapshot"))
            self.assertEqual(3, args.count("--dynamic-crop-manifest"))
            self.assertEqual("IVE-XOXZ", result.held_out_song_id)
            self.assertEqual(
                ("IVE-BANG-BANG", "IVE-BLACKHOLE"),
                result.training_song_ids,
            )
            self.assertNotIn("--authorize-production-activation", args)
            self.assertNotIn("--production-registry", args)

    def test_content_addressed_plan_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            plan_path = _write_plan(root, root / "checkpoints")
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            payload["minimum_free_bytes"] = 1
            plan_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                launcher.ProductionFoldPreflightError,
                "content hash mismatch",
            ):
                launcher._strict_plan_payload(plan_path)

    def test_storage_rejects_existing_checkpoint_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            volume = Path(raw).resolve()
            checkpoint = volume / "existing"
            checkpoint.mkdir()
            with (
                patch.object(launcher, "PRODUCTION_VOLUME_ROOT", volume),
                patch.object(launcher.os.path, "ismount", return_value=True),
            ):
                with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                    launcher._validate_storage(checkpoint)

    def test_plan_binds_exact_schema_hashed_recent_history_contract(self) -> None:
        contract = dict(launcher.PRODUCTION_RECENT_HISTORY_MASK_CONTRACT)
        self.assertEqual(
            "transformer_recent_edit_slice_fully_unknown_train_and_serve_v1",
            contract["contract"],
        )
        self.assertEqual(
            launcher.DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
            contract["token_schema_fingerprint"],
        )
        digest = contract.pop("contract_sha256")
        self.assertEqual(
            digest,
            hashlib.sha256(launcher._canonical_bytes(contract)).hexdigest(),
        )

    def test_launch_delegates_exact_preflight_arguments(self) -> None:
        fake = launcher.ProductionFoldPreflight(
            plan_path=Path("/tmp/plan.json"),
            plan_sha256="a" * 64,
            fold_id="heldout-xoxz",
            held_out_song_id="IVE-XOXZ",
            training_song_ids=("IVE-BANG-BANG", "IVE-BLACKHOLE"),
            checkpoint_directory=Path("/Volumes/CrossCutMedia/checkpoint"),
            free_bytes=20 * 1024**3,
            recent_history_mask_contract=dict(
                launcher.PRODUCTION_RECENT_HISTORY_MASK_CONTRACT
            ),
            reference_ids_by_song={},
            train_cli_args=("--held-out-song", "IVE-XOXZ"),
        )
        with (
            patch.object(
                launcher,
                "preflight_stage_edit_production_fold",
                return_value=fake,
            ),
            patch.object(launcher, "production_train_main", return_value=7) as train,
        ):
            result = launcher.main(
                [
                    "launch",
                    "--plan",
                    "/tmp/plan.json",
                    "--fold",
                    "heldout-xoxz",
                ]
            )
        self.assertEqual(7, result)
        train.assert_called_once_with(fake.train_cli_args)


if __name__ == "__main__":
    unittest.main()
