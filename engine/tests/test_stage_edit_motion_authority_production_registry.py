from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crosscut.stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
)
from crosscut.stage_edit_production_registry import (
    STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_AUTHORITY_V3_KIND,
    STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_AUTHORITY_V3_SCHEMA_VERSION,
    STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_THRESHOLD_FREEZE_CONTRACT_V3,
    STAGE_EDIT_PRODUCTION_REGISTRY_V3_SCHEMA_VERSION,
    StageEditProductionRegistryError,
    StageEditProductionRegistryNotReadyError,
    load_registered_stage_edit_production_bundle,
    register_stage_edit_production_bundle,
)
from tests.test_stage_edit_production_registry import (
    _canonical_sha256,
    _checkpoint_v3,
    _montage_profile,
    _ready_v3,
)


_PROMOTION_RUN_KIND = (
    "dynamic_crop_v3_final_fit_exact_weight_production_promotion"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_promotion_report(
    checkpoint: Path,
    *,
    release_freeze_path: Path,
    release_freeze_content_sha256: str,
    source_commit_oid: str,
) -> None:
    report = {
        "run_kind": _PROMOTION_RUN_KIND,
        "source_commit": source_commit_oid,
        "lineage": {
            "release_freeze": {
                "path": str(release_freeze_path),
                "file_sha256": _file_sha256(release_freeze_path),
            },
            "release_freeze_content_sha256": (
                release_freeze_content_sha256
            ),
        },
    }
    (checkpoint / "training-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _freeze_payload(
    *,
    release_freeze_content_sha256: str,
    source_commit_oid: str,
    candidate_id: str = "production-balanced-v3-candidate",
) -> dict[str, object]:
    return {
        "freeze_content_sha256": release_freeze_content_sha256,
        "source_provenance": {"head_commit_oid": source_commit_oid},
        "candidate_id": candidate_id,
    }


def _reseal_registry(path: Path, mutate) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    unsigned = dict(payload)
    unsigned.pop("registry_sha256")
    payload["registry_sha256"] = _canonical_sha256(unsigned)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


class StageEditMotionAuthorityProductionRegistryTests(unittest.TestCase):
    def _register_promoted(
        self,
        root: Path,
        *,
        release_freeze_path: Path | None = None,
        report_source_commit_oid: str = "a" * 40,
        freeze_source_commit_oid: str = "a" * 40,
        release_freeze_content_sha256: str = "b" * 64,
        candidate_id: str = "production-balanced-v3-candidate",
    ) -> tuple[Path, Path, dict[str, object]]:
        checkpoint, _pair_scorer, _pair_report = _checkpoint_v3(root)
        montage_profile = _montage_profile(root)
        if release_freeze_path is None:
            release_freeze_path = root / "release-freeze.json"
            release_freeze_path.write_bytes(b"strict-release-freeze-v3")
            release_freeze_path = release_freeze_path.resolve()
        elif not release_freeze_path.is_symlink():
            release_freeze_path = release_freeze_path.resolve()
        _write_promotion_report(
            checkpoint,
            release_freeze_path=release_freeze_path,
            release_freeze_content_sha256=release_freeze_content_sha256,
            source_commit_oid=report_source_commit_oid,
        )
        registry = root / "registry" / "active-production-bundle.json"
        freeze = _freeze_payload(
            release_freeze_content_sha256=release_freeze_content_sha256,
            source_commit_oid=freeze_source_commit_oid,
            candidate_id=candidate_id,
        )
        with (
            patch(
                "crosscut.stage_edit_production_registry."
                "stage_edit_readiness_report",
                return_value=_ready_v3(checkpoint),
            ),
            patch(
                "crosscut.stage_edit_production_registry."
                "_validate_v3_training_report_declarations",
            ),
            patch(
                "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher."
                "load_dynamic_crop_v3_release_freeze",
                return_value=freeze,
            ),
        ):
            register_stage_edit_production_bundle(
                checkpoint,
                professional_montage_grammar_profile_path=montage_profile,
                registry_path=registry,
                dynamic_crop_version=3,
                continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
            )
        return checkpoint, registry, freeze

    def test_promoted_v3_round_trips_exact_typed_motion_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, registry, freeze = self._register_promoted(root)
            with (
                patch(
                    "crosscut.stage_edit_production_registry."
                    "stage_edit_readiness_report",
                    return_value=_ready_v3(checkpoint),
                ),
                patch(
                    "crosscut.stage_edit_production_registry."
                    "_validate_v3_training_report_declarations",
                ),
                patch(
                    "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher."
                    "load_dynamic_crop_v3_release_freeze",
                    return_value=freeze,
                ),
            ):
                loaded = load_registered_stage_edit_production_bundle(registry)

            manifest = json.loads(registry.read_text(encoding="utf-8"))
            self.assertEqual(
                STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_AUTHORITY_V3_SCHEMA_VERSION,
                manifest["schema_version"],
            )
            self.assertEqual(
                STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_AUTHORITY_V3_KIND,
                manifest["registry_kind"],
            )
            self.assertTrue(loaded.dynamic_crop_v3_motion_authority_qualified)
            identity = loaded.motion_authority_identity_v3
            self.assertIsNotNone(identity)
            assert identity is not None
            self.assertEqual(
                (root / "release-freeze.json").resolve(),
                identity.release_freeze_path,
            )
            self.assertEqual(
                _file_sha256(identity.release_freeze_path),
                identity.release_freeze_file_sha256,
            )
            self.assertEqual(
                "b" * 64, identity.release_freeze_content_sha256
            )
            self.assertEqual("a" * 40, identity.source_commit_oid)
            self.assertEqual(
                "production-balanced-v3-candidate", identity.candidate_id
            )
            self.assertEqual(0.0, identity.motion_logit_threshold)
            self.assertEqual(
                STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_THRESHOLD_FREEZE_CONTRACT_V3,
                identity.threshold_freeze_contract,
            )
            self.assertEqual(
                identity.to_dict(), manifest["dynamic_crop_v3_motion_authority"]
            )
            public = loaded.to_dict()["dynamic_crop_v3_motion_authority"]
            self.assertEqual(
                {
                    "qualified": True,
                    "legacy_without_motion_authority": False,
                    "identity": identity.to_dict(),
                },
                public,
            )

    def test_legacy_v3_remains_loadable_but_never_qualifies_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, _pair_scorer, _pair_report = _checkpoint_v3(root)
            registry = root / "legacy-v3.json"
            with patch(
                "crosscut.stage_edit_production_registry."
                "stage_edit_readiness_report",
                return_value=_ready_v3(checkpoint),
            ):
                register_stage_edit_production_bundle(
                    checkpoint,
                    professional_montage_grammar_profile_path=_montage_profile(
                        root
                    ),
                    registry_path=registry,
                    dynamic_crop_version=3,
                    continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
                )
                loaded = load_registered_stage_edit_production_bundle(registry)

            manifest = json.loads(registry.read_text(encoding="utf-8"))
            self.assertEqual(
                STAGE_EDIT_PRODUCTION_REGISTRY_V3_SCHEMA_VERSION,
                manifest["schema_version"],
            )
            self.assertNotIn("dynamic_crop_v3_motion_authority", manifest)
            self.assertIsNone(loaded.motion_authority_identity_v3)
            self.assertFalse(loaded.dynamic_crop_v3_motion_authority_qualified)
            self.assertEqual(
                {
                    "qualified": False,
                    "legacy_without_motion_authority": True,
                    "identity": None,
                },
                loaded.to_dict()["dynamic_crop_v3_motion_authority"],
            )

    def test_resealed_registry_motion_identity_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, registry, freeze = self._register_promoted(root)
            _reseal_registry(
                registry,
                lambda payload: payload["dynamic_crop_v3_motion_authority"].update(
                    {"release_freeze_content_sha256": "c" * 64}
                ),
            )
            with (
                patch(
                    "crosscut.stage_edit_production_registry."
                    "stage_edit_readiness_report",
                    return_value=_ready_v3(checkpoint),
                ),
                patch(
                    "crosscut.stage_edit_production_registry."
                    "_validate_v3_training_report_declarations",
                ),
                patch(
                    "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher."
                    "load_dynamic_crop_v3_release_freeze",
                    return_value=freeze,
                ),
            ):
                with self.assertRaisesRegex(
                    StageEditProductionRegistryNotReadyError,
                    "no longer matches",
                ):
                    load_registered_stage_edit_production_bundle(registry)

    def test_resealed_registry_cannot_reselect_frozen_motion_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, registry, freeze = self._register_promoted(root)
            _reseal_registry(
                registry,
                lambda payload: payload["dynamic_crop_v3_motion_authority"].update(
                    {"motion_logit_threshold": 0.01}
                ),
            )
            with (
                patch(
                    "crosscut.stage_edit_production_registry."
                    "stage_edit_readiness_report",
                    return_value=_ready_v3(checkpoint),
                ),
                patch(
                    "crosscut.stage_edit_production_registry."
                    "_validate_v3_training_report_declarations",
                ),
                patch(
                    "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher."
                    "load_dynamic_crop_v3_release_freeze",
                    return_value=freeze,
                ),
            ):
                with self.assertRaisesRegex(
                    StageEditProductionRegistryError,
                    "motion authority identity is invalid",
                ):
                    load_registered_stage_edit_production_bundle(registry)

    def test_release_freeze_staleness_is_rechecked_on_every_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, registry, freeze = self._register_promoted(root)
            (root / "release-freeze.json").write_bytes(b"changed-after-register")
            with (
                patch(
                    "crosscut.stage_edit_production_registry."
                    "stage_edit_readiness_report",
                    return_value=_ready_v3(checkpoint),
                ),
                patch(
                    "crosscut.stage_edit_production_registry."
                    "_validate_v3_training_report_declarations",
                ),
                patch(
                    "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher."
                    "load_dynamic_crop_v3_release_freeze",
                    return_value=freeze,
                ),
            ):
                with self.assertRaisesRegex(
                    StageEditProductionRegistryNotReadyError,
                    "file identity changed",
                ):
                    load_registered_stage_edit_production_bundle(registry)

    def test_release_freeze_symlink_cannot_register_motion_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "real-release-freeze.json"
            target.write_bytes(b"strict-release-freeze-v3")
            symlink = root / "release-freeze.json"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(
                StageEditProductionRegistryNotReadyError,
                "absolute and non-symlink",
            ):
                self._register_promoted(root, release_freeze_path=symlink)

    def test_wrong_release_freeze_source_lineage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                StageEditProductionRegistryNotReadyError,
                "lineage identity differs",
            ):
                self._register_promoted(
                    root,
                    report_source_commit_oid="a" * 40,
                    freeze_source_commit_oid="d" * 40,
                )


if __name__ == "__main__":
    unittest.main()
