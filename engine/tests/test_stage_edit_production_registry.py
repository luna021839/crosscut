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
from crosscut.stage_edit_transition_authority import (
    RENDERER_OWNED_TRANSITION_AUTHORITY,
)
from tests.test_professional_montage_policy import _profile, _write


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


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
        replace(
            smoke_stage_edit_config(),
            transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
        ),
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


def _montage_profile(root: Path, name: str = "montage-profile.json") -> Path:
    path = root / name
    _write(path, _profile())
    return path


class StageEditProductionRegistryTests(unittest.TestCase):
    def test_registry_rejects_transition_authority_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = _checkpoint(root)
            montage_profile = _montage_profile(root)
            registry = root / "active-production-bundle.json"
            with patch(
                "crosscut.stage_edit_production_registry.stage_edit_readiness_report",
                return_value=_ready(checkpoint),
            ):
                register_stage_edit_production_bundle(
                    checkpoint,
                    professional_montage_grammar_profile_path=montage_profile,
                    registry_path=registry,
                )
            payload = json.loads(registry.read_text(encoding="utf-8"))
            payload["transition_authority"]["mode"] = (
                "learned_multiclass_transition_head_v1"
            )
            unsigned = dict(payload)
            unsigned.pop("registry_sha256")
            payload["registry_sha256"] = _canonical_sha256(unsigned)
            registry.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                StageEditProductionRegistryError,
                "transition authority is unsupported",
            ):
                load_registered_stage_edit_production_bundle(registry)

    def test_registration_is_immutable_idempotent_and_revalidated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = _checkpoint(root)
            montage_profile = _montage_profile(root)
            registry = root / "registry" / "active-production-bundle.json"
            with patch(
                "crosscut.stage_edit_production_registry.stage_edit_readiness_report",
                return_value=_ready(checkpoint),
            ):
                first = register_stage_edit_production_bundle(
                    checkpoint,
                    professional_montage_grammar_profile_path=montage_profile,
                    registry_path=registry,
                )
                before = first.read_bytes()
                second = register_stage_edit_production_bundle(
                    checkpoint,
                    professional_montage_grammar_profile_path=montage_profile,
                    registry_path=registry,
                )
                self.assertEqual(first, second)
                self.assertEqual(before, second.read_bytes())
                loaded = load_registered_stage_edit_production_bundle(registry)

            self.assertEqual(loaded.checkpoint_directory, checkpoint.resolve())
            self.assertEqual(loaded.checkpoint_id, "ive-production-v2")
            self.assertEqual(loaded.maximum_sources, 8)
            self.assertEqual(loaded.registry_path, registry.resolve())
            self.assertEqual(
                loaded.professional_montage_grammar_profile_path,
                montage_profile,
            )
            self.assertEqual(
                loaded.professional_montage_grammar_profile_identity,
                json.loads(registry.read_text())[  # type: ignore[index]
                    "professional_montage_grammar_profile"
                ],
            )
            registry_payload = json.loads(registry.read_text(encoding="utf-8"))
            self.assertEqual(3, registry_payload["schema_version"])
            self.assertEqual(
                {
                    "path",
                    "analysis_basis",
                    "schema_version",
                    "file_sha256",
                    "content_sha256",
                },
                set(registry_payload["professional_montage_grammar_profile"]),
            )

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
            montage_profile = _montage_profile(root)
            registry = root / "active-production-bundle.json"
            with patch(
                "crosscut.stage_edit_production_registry.stage_edit_readiness_report",
                side_effect=lambda path: _ready(Path(path)),
            ):
                register_stage_edit_production_bundle(
                    first_checkpoint,
                    professional_montage_grammar_profile_path=montage_profile,
                    registry_path=registry,
                )
                with self.assertRaisesRegex(
                    StageEditProductionRegistryError,
                    "different active",
                ):
                    register_stage_edit_production_bundle(
                        second_checkpoint,
                        professional_montage_grammar_profile_path=montage_profile,
                        registry_path=registry,
                    )

    def test_profile_file_is_rehashed_and_missing_or_tampered_fails_closed(
        self,
    ) -> None:
        for failure in ("tampered", "missing"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                checkpoint = _checkpoint(root)
                montage_profile = _montage_profile(root)
                registry = root / "active-production-bundle.json"
                with patch(
                    "crosscut.stage_edit_production_registry."
                    "stage_edit_readiness_report",
                    return_value=_ready(checkpoint),
                ):
                    register_stage_edit_production_bundle(
                        checkpoint,
                        professional_montage_grammar_profile_path=(
                            montage_profile
                        ),
                        registry_path=registry,
                    )
                if failure == "tampered":
                    montage_profile.write_text("{}\n", encoding="utf-8")
                else:
                    montage_profile.unlink()
                with self.assertRaisesRegex(
                    StageEditProductionRegistryNotReadyError,
                    "montage grammar profile is missing, changed, or invalid",
                ):
                    load_registered_stage_edit_production_bundle(registry)

    def test_registry_profile_identity_mismatch_and_malformed_evidence_fail(
        self,
    ) -> None:
        for failure in ("identity", "malformed"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                checkpoint = _checkpoint(root)
                montage_profile = _montage_profile(root)
                registry = root / "active-production-bundle.json"
                with patch(
                    "crosscut.stage_edit_production_registry."
                    "stage_edit_readiness_report",
                    return_value=_ready(checkpoint),
                ):
                    register_stage_edit_production_bundle(
                        checkpoint,
                        professional_montage_grammar_profile_path=(
                            montage_profile
                        ),
                        registry_path=registry,
                    )
                payload = json.loads(registry.read_text(encoding="utf-8"))
                profile_evidence = payload[
                    "professional_montage_grammar_profile"
                ]
                if failure == "identity":
                    profile_evidence["content_sha256"] = "0" * 64
                else:
                    del profile_evidence["analysis_basis"]
                unsigned = dict(payload)
                unsigned.pop("registry_sha256")
                payload["registry_sha256"] = _canonical_sha256(unsigned)
                registry.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                expected_error = (
                    StageEditProductionRegistryNotReadyError
                    if failure == "identity"
                    else StageEditProductionRegistryError
                )
                with self.assertRaises(expected_error):
                    load_registered_stage_edit_production_bundle(registry)

    def test_registration_requires_absolute_non_symlink_valid_profile(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoint = _checkpoint(root)
            profile = _montage_profile(root)
            link = root / "profile-link.json"
            link.symlink_to(profile)
            malformed = root / "malformed-profile.json"
            malformed.write_text("{}\n", encoding="utf-8")
            for candidate, pattern in (
                (Path("relative-profile.json"), "explicit and absolute"),
                (link, "non-symlink"),
                (malformed, "profile"),
            ):
                with self.subTest(candidate=candidate), self.assertRaisesRegex(
                    StageEditProductionRegistryNotReadyError,
                    pattern,
                ):
                    register_stage_edit_production_bundle(
                        checkpoint,
                        professional_montage_grammar_profile_path=candidate,
                        registry_path=root / "registry.json",
                    )

    def test_idempotence_includes_the_profile_path_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoint = _checkpoint(root)
            first_profile = _montage_profile(root, "profile-a.json")
            second_profile = root / "profile-b.json"
            second_profile.write_bytes(first_profile.read_bytes())
            registry = root / "active-production-bundle.json"
            with patch(
                "crosscut.stage_edit_production_registry."
                "stage_edit_readiness_report",
                return_value=_ready(checkpoint),
            ):
                register_stage_edit_production_bundle(
                    checkpoint,
                    professional_montage_grammar_profile_path=first_profile,
                    registry_path=registry,
                )
                with self.assertRaisesRegex(
                    StageEditProductionRegistryError,
                    "different active",
                ):
                    register_stage_edit_production_bundle(
                        checkpoint,
                        professional_montage_grammar_profile_path=second_profile,
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
