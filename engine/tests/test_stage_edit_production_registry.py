from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crosscut.stage_edit_dynamic_crop_head_v2 import (
    DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v2,
)
from crosscut.stage_edit_production_registry import (
    STAGE_EDIT_PRODUCTION_REGISTRY_ENV,
    StageEditProductionRegistryError,
    StageEditProductionRegistryNotReadyError,
    discover_registered_stage_edit_production_bundle,
    load_registered_stage_edit_production_bundle,
    register_stage_edit_production_bundle,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    StageEditReadinessReport,
    smoke_stage_edit_config,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ready(checkpoint: Path, *, steps: int = 7) -> StageEditReadinessReport:
    return StageEditReadinessReport(
        architecture_available=True,
        learned_inference_ready=True,
        status="production_checkpoint_ready",
        reason="test production evidence",
        mlx_version="test",
        checkpoint_path=str(checkpoint.resolve()),
        training_steps=steps,
    )


def _checkpoint(root: Path, name: str = "checkpoint") -> Path:
    checkpoint = root / name
    checkpoint.mkdir()
    weights = checkpoint / DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME
    weights.write_bytes(b"verified-dynamic-crop-v2-weights")
    metadata = new_stage_edit_dynamic_crop_checkpoint_metadata_v2(
        "ive-production-v2",
        DEFAULT_STAGE_EDIT_SCHEMA,
        smoke_stage_edit_config(),
        training_steps=7,
        weights_trained=True,
        training_data_id="three-song-heldout",
    )
    metadata = replace(metadata, weight_sha256=_sha256(weights))
    (checkpoint / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME).write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (checkpoint / "training-report.json").write_text(
        json.dumps({"production": "passed"}),
        encoding="utf-8",
    )
    return checkpoint


class StageEditProductionRegistryTests(unittest.TestCase):
    def test_registration_is_immutable_idempotent_and_revalidated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = _checkpoint(root)
            registry = root / "registry" / "active-production-bundle.json"
            with patch(
                "crosscut.stage_edit_production_registry.stage_edit_readiness_report",
                return_value=_ready(checkpoint),
            ):
                first = register_stage_edit_production_bundle(
                    checkpoint,
                    registry_path=registry,
                )
                before = first.read_bytes()
                second = register_stage_edit_production_bundle(
                    checkpoint,
                    registry_path=registry,
                )
                self.assertEqual(first, second)
                self.assertEqual(before, second.read_bytes())
                loaded = load_registered_stage_edit_production_bundle(registry)

            self.assertEqual(loaded.checkpoint_directory, checkpoint.resolve())
            self.assertEqual(loaded.checkpoint_id, "ive-production-v2")
            self.assertEqual(loaded.maximum_sources, 8)
            self.assertEqual(loaded.registry_path, registry.resolve())

            (checkpoint / "training-report.json").write_text(
                json.dumps({"production": "tampered"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                StageEditProductionRegistryNotReadyError,
                "evidence changed",
            ):
                load_registered_stage_edit_production_bundle(registry)

    def test_different_active_bundle_cannot_overwrite_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_checkpoint = _checkpoint(root, "first")
            second_checkpoint = _checkpoint(root, "second")
            registry = root / "active-production-bundle.json"
            with patch(
                "crosscut.stage_edit_production_registry.stage_edit_readiness_report",
                side_effect=lambda path: _ready(Path(path)),
            ):
                register_stage_edit_production_bundle(
                    first_checkpoint,
                    registry_path=registry,
                )
                with self.assertRaisesRegex(
                    StageEditProductionRegistryError,
                    "different active",
                ):
                    register_stage_edit_production_bundle(
                        second_checkpoint,
                        registry_path=registry,
                    )

    def test_configured_registry_is_mandatory_and_malformed_never_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.json"
            with patch.dict(
                os.environ,
                {STAGE_EDIT_PRODUCTION_REGISTRY_ENV: str(missing)},
                clear=False,
            ):
                with self.assertRaises(StageEditProductionRegistryError):
                    discover_registered_stage_edit_production_bundle()

            malformed = root / "malformed.json"
            malformed.write_text("{}", encoding="utf-8")
            with patch.dict(
                os.environ,
                {STAGE_EDIT_PRODUCTION_REGISTRY_ENV: str(malformed)},
                clear=False,
            ):
                with self.assertRaisesRegex(
                    StageEditProductionRegistryError,
                    "schema",
                ):
                    discover_registered_stage_edit_production_bundle()


if __name__ == "__main__":
    unittest.main()
