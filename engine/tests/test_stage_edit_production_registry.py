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
from crosscut.stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v3,
)
from crosscut.stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
)
from crosscut.stage_edit_production_registry import (
    STAGE_EDIT_PRODUCTION_REGISTRY_ENV,
    STAGE_EDIT_PRODUCTION_REGISTRY_KIND,
    STAGE_EDIT_PRODUCTION_REGISTRY_V3_KIND,
    STAGE_EDIT_PRODUCTION_REGISTRY_V3_SCHEMA_VERSION,
    StageEditProductionRegistryError,
    StageEditProductionRegistryNotReadyError,
    discover_registered_stage_edit_production_bundle,
    load_registered_stage_edit_production_bundle,
    register_stage_edit_production_bundle,
)
from crosscut.stage_edit_source_pair_production import (
    produce_stage_edit_source_pair_checkpoint,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_CONTRACT,
    DYNAMIC_CROP_V3_TRAINING_CORPUS_CONTRACT,
    DYNAMIC_CROP_V3_VALIDATION_SELECTION_GATE_CONTRACT,
    StageEditReadinessReport,
    smoke_stage_edit_config,
)
from crosscut.stage_edit_transition_authority import (
    RENDERER_OWNED_TRANSITION_AUTHORITY,
)
from tests.test_professional_montage_policy import _profile, _write
from tests.test_stage_edit_source_pair_production import _corpus


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


def _ready_v3(checkpoint: Path, *, steps: int = 11) -> StageEditReadinessReport:
    return StageEditReadinessReport(
        architecture_available=True,
        learned_inference_ready=True,
        status="production_checkpoint_ready",
        reason="test v3 production evidence",
        mlx_version="test",
        checkpoint_path=str(checkpoint.resolve()),
        training_steps=steps,
        checkpoint_format="dynamic_crop_v3",
        dynamic_crop_version=3,
        checkpoint_metadata_filename=(
            DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
        ),
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


def _dynamic_crop_v3_selection_gate() -> dict[str, object]:
    return {
        "contract": DYNAMIC_CROP_V3_VALIDATION_SELECTION_GATE_CONTRACT,
        "selectable": True,
        "reasons": [],
        "score_component": 0.125,
        "renderer_decode_attempt_count": 8,
        "renderer_decode_eligible_count": 8,
        "renderer_decode_eligible_rate": 1.0,
        "minimum_safe_zoom_projection_count": 0,
        "target_source_slot_mismatch_count": 0,
        "aggregate_normalized_improvement_over_static_prior": 0.125,
        "static_prior_can_be_selected": False,
        "heldout_metrics_consumed": False,
    }


def _checkpoint_v3(
    root: Path,
    name: str = "checkpoint-v3",
) -> tuple[Path, Path, Path]:
    checkpoint = root / name
    checkpoint.mkdir()
    weights = checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
    weights.write_bytes(b"verified-dynamic-crop-v3-weights")
    metadata = new_stage_edit_dynamic_crop_checkpoint_metadata_v3(
        "ive-production-v3",
        DEFAULT_STAGE_EDIT_SCHEMA,
        replace(
            smoke_stage_edit_config(),
            transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
        ),
        training_steps=11,
        weights_trained=True,
        training_data_id="base-corpus+dynamic-crop-v3-corpus",
    )
    metadata = replace(metadata, weight_sha256=_sha256(weights))
    (checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME).write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selected_gate = _dynamic_crop_v3_selection_gate()
    reloaded_gate = _dynamic_crop_v3_selection_gate()
    training_report = {
        "checkpoint": {
            "checkpoint_id": metadata.checkpoint_id,
            "training_steps": metadata.training_steps,
            "weight_sha256": metadata.weight_sha256,
            "production_activation_authorized": True,
            "production_supervision_gate_passed": True,
            "checkpoint_format": "dynamic_crop_v3",
            "dynamic_crop_version": 3,
            "metadata_filename": DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
            "dynamic_crop_v2_checkpoint": False,
            "dynamic_crop_v3_checkpoint": True,
            "dynamic_crop_v3_data_id": "dynamic-crop-v3-corpus",
            "dynamic_crop_model_contract": metadata.model_contract,
            "dynamic_crop_target_contract": metadata.target_contract,
            "dynamic_crop_output_contract": metadata.output_contract,
        },
        "model": {
            "dynamic_crop_version": 3,
            "dynamic_crop_v2_enabled": False,
            "dynamic_crop_v3_enabled": True,
            "activation_checkpointing": {
                "contract": (
                    DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_CONTRACT
                ),
                "enabled": True,
                "scope": "base_transformer_encoder_all_layers",
                "implementation": "mlx.nn.TransformerEncoder.checkpoint",
                "training_only": True,
                "serialized_parameter_tree_changed": False,
                "inference_runtime_requires_checkpointing": False,
                "previous_value": False,
            },
        },
        "data": {
            "dynamic_crop_version": 3,
            "dynamic_crop_v2": None,
            "dynamic_crop_v3": {
                "data_id": "dynamic-crop-v3-corpus",
                "contract": DYNAMIC_CROP_V3_TRAINING_CORPUS_CONTRACT,
                "target_contract": metadata.target_contract,
            },
        },
        "optimization": {
            "dynamic_crop_v3_model_selection": (
                DYNAMIC_CROP_V3_VALIDATION_SELECTION_GATE_CONTRACT
            ),
            "best_validation_selection_evidence": {
                "dynamic_crop_v3_selection_gate": selected_gate,
            },
            "reloaded_checkpoint_validation_selection_evidence": {
                "dynamic_crop_v3_selection_gate": reloaded_gate,
            },
        },
        "held_out_evaluation": {
            "passed": True,
            "dynamic_crop_version": 3,
            "dynamic_crop_v2_metrics": None,
            "dynamic_crop_v3_metrics": {"trajectory": "held-out-only"},
            "heldout_dynamic_crop_metrics_used_for_checkpoint_selection": False,
        },
    }
    (checkpoint / "training-report.json").write_text(
        json.dumps(training_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pair_directory = root / f"{name}-source-pair"
    pair_directory.mkdir()
    pair_result = produce_stage_edit_source_pair_checkpoint(
        _corpus(),
        pair_directory,
        checkpoint_id=f"{name}-source-pair",
        created_at="2026-08-24T12:00:00+09:00",
        steps=60,
        learning_rate=0.30,
    )
    return checkpoint, pair_result.checkpoint_path, pair_result.report_path


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
            self.assertEqual(loaded.dynamic_crop_version, 2)
            self.assertIsNone(loaded.continuity_authority)
            self.assertIsNone(loaded.source_pair_scorer_path)
            self.assertIsNone(loaded.source_pair_checkpoint_sha256)
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
            self.assertNotIn("dynamic_crop_version", registry_payload)
            self.assertNotIn("continuity_authority", registry_payload)
            self.assertNotIn("source_pair_scorer", registry_payload)
            v2_public = loaded.to_dict()
            self.assertNotIn("dynamic_crop_version", v2_public)
            self.assertNotIn("continuity_authority", v2_public)
            self.assertNotIn("source_pair_scorer", v2_public)
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

    def test_new_registry_is_rolled_back_when_post_write_replay_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = _checkpoint(root)
            montage_profile = _montage_profile(root)
            registry = root / "registry" / "active-production-bundle.json"
            with (
                patch(
                    "crosscut.stage_edit_production_registry."
                    "stage_edit_readiness_report",
                    return_value=_ready(checkpoint),
                ),
                patch(
                    "crosscut.stage_edit_production_registry."
                    "load_registered_stage_edit_production_bundle",
                    side_effect=StageEditProductionRegistryNotReadyError(
                        "forced post-write replay failure"
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    StageEditProductionRegistryNotReadyError,
                    "forced post-write replay failure",
                ):
                    register_stage_edit_production_bundle(
                        checkpoint,
                        professional_montage_grammar_profile_path=montage_profile,
                        registry_path=registry,
                    )
            self.assertFalse(registry.exists())
            self.assertFalse(
                registry.with_name(f".{registry.name}.tmp").exists()
            )

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

    def test_v3_registry_binds_dynamic_crop_and_source_pair_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, pair_scorer, pair_report = _checkpoint_v3(root)
            montage_profile = _montage_profile(root)
            registry = root / "registry" / "active-production-bundle.json"
            with patch(
                "crosscut.stage_edit_production_registry."
                "stage_edit_readiness_report",
                return_value=_ready_v3(checkpoint),
            ):
                first = register_stage_edit_production_bundle(
                    checkpoint,
                    professional_montage_grammar_profile_path=montage_profile,
                    registry_path=registry,
                    dynamic_crop_version=3,
                    continuity_authority=(
                        PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                    ),
                    source_pair_scorer_path=pair_scorer,
                    source_pair_production_report_path=pair_report,
                )
                encoded = first.read_bytes()
                second = register_stage_edit_production_bundle(
                    checkpoint,
                    professional_montage_grammar_profile_path=montage_profile,
                    registry_path=registry,
                    dynamic_crop_version=3,
                    continuity_authority=(
                        PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                    ),
                    source_pair_scorer_path=pair_scorer,
                    source_pair_production_report_path=pair_report,
                )
                loaded = load_registered_stage_edit_production_bundle(registry)

            self.assertEqual(first, second)
            self.assertEqual(encoded, second.read_bytes())
            self.assertEqual(loaded.dynamic_crop_version, 3)
            self.assertEqual(
                loaded.continuity_authority,
                PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
            )
            self.assertEqual(loaded.source_pair_scorer_path, pair_scorer)
            self.assertEqual(
                loaded.source_pair_production_report_path,
                pair_report,
            )
            self.assertEqual(64, len(loaded.source_pair_scorer_fingerprint or ""))
            self.assertEqual(64, len(loaded.source_pair_checkpoint_sha256 or ""))
            self.assertEqual(64, len(loaded.source_pair_corpus_sha256 or ""))

            manifest = json.loads(registry.read_text(encoding="utf-8"))
            self.assertEqual(
                STAGE_EDIT_PRODUCTION_REGISTRY_V3_SCHEMA_VERSION,
                manifest["schema_version"],
            )
            self.assertEqual(
                STAGE_EDIT_PRODUCTION_REGISTRY_V3_KIND,
                manifest["registry_kind"],
            )
            self.assertEqual(
                "ive_stage_edit_dynamic_crop_v3_active_production_bundle",
                manifest["registry_kind"],
            )
            self.assertEqual(3, manifest["dynamic_crop_version"])
            self.assertEqual(
                PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
                manifest["continuity_authority"],
            )
            self.assertEqual(
                str(pair_scorer), manifest["source_pair_scorer"]["path"]
            )
            self.assertEqual(
                loaded.source_pair_scorer_fingerprint,
                manifest["source_pair_scorer"]["scorer_fingerprint"],
            )
            self.assertEqual(
                loaded.source_pair_checkpoint_sha256,
                manifest["source_pair_scorer"]["checkpoint_sha256"],
            )
            self.assertEqual(
                loaded.source_pair_corpus_sha256,
                manifest["source_pair_scorer"]["corpus_sha256"],
            )
            self.assertEqual(
                loaded.source_pair_production_report_sha256,
                manifest["source_pair_production_report"]["report_sha256"],
            )
            public = loaded.to_dict()
            self.assertEqual(3, public["dynamic_crop_version"])
            self.assertEqual(
                PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
                public["continuity_authority"],
            )
            self.assertEqual(
                str(pair_scorer),
                public["source_pair_scorer"]["checkpoint_path"],
            )
            self.assertEqual(
                loaded.source_pair_checkpoint_sha256,
                public["source_pair_scorer"]["checkpoint_sha256"],
            )

            manifest["registry_kind"] = STAGE_EDIT_PRODUCTION_REGISTRY_KIND
            unsigned = dict(manifest)
            unsigned.pop("registry_sha256")
            manifest["registry_sha256"] = _canonical_sha256(unsigned)
            registry.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                StageEditProductionRegistryError,
                "activation contract is unsupported",
            ):
                load_registered_stage_edit_production_bundle(registry)

    def test_v3_registration_requires_both_exact_source_pair_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, pair_scorer, pair_report = _checkpoint_v3(root)
            montage_profile = _montage_profile(root)
            with patch(
                "crosscut.stage_edit_production_registry."
                "stage_edit_readiness_report",
                return_value=_ready_v3(checkpoint),
            ):
                for scorer, report in (
                    (None, None),
                    (pair_scorer, None),
                    (None, pair_report),
                ):
                    with self.subTest(scorer=scorer, report=report):
                        with self.assertRaisesRegex(
                            StageEditProductionRegistryNotReadyError,
                            "requires both",
                        ):
                            register_stage_edit_production_bundle(
                                checkpoint,
                                professional_montage_grammar_profile_path=(
                                    montage_profile
                                ),
                                registry_path=root / "missing-pair.json",
                                dynamic_crop_version=3,
                                continuity_authority=(
                                    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                                ),
                                source_pair_scorer_path=scorer,
                                source_pair_production_report_path=report,
                            )

                with self.assertRaisesRegex(
                    StageEditProductionRegistryNotReadyError,
                    "missing or unsafe",
                ):
                    register_stage_edit_production_bundle(
                        checkpoint,
                        professional_montage_grammar_profile_path=(
                            montage_profile
                        ),
                        registry_path=root / "missing-file.json",
                        dynamic_crop_version=3,
                        continuity_authority=(
                            PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                        ),
                        source_pair_scorer_path=pair_scorer,
                        source_pair_production_report_path=(
                            root / "source-pair-production-report.json"
                        ),
                    )

            checkpoint_v2 = _checkpoint(root, "checkpoint-v2")
            with self.assertRaisesRegex(
                StageEditProductionRegistryNotReadyError,
                "v2 cannot register source-pair",
            ):
                register_stage_edit_production_bundle(
                    checkpoint_v2,
                    professional_montage_grammar_profile_path=montage_profile,
                    registry_path=root / "v2-with-pair.json",
                    source_pair_scorer_path=pair_scorer,
                    source_pair_production_report_path=pair_report,
                )

    def test_v3_handcrafted_authority_round_trips_without_pair_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, _unused_scorer, _unused_report = _checkpoint_v3(root)
            montage_profile = _montage_profile(root)
            registry = root / "active-production-bundle.json"
            with patch(
                "crosscut.stage_edit_production_registry."
                "stage_edit_readiness_report",
                return_value=_ready_v3(checkpoint),
            ):
                first = register_stage_edit_production_bundle(
                    checkpoint,
                    professional_montage_grammar_profile_path=montage_profile,
                    registry_path=registry,
                    dynamic_crop_version=3,
                    continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
                )
                encoded = first.read_bytes()
                second = register_stage_edit_production_bundle(
                    checkpoint,
                    professional_montage_grammar_profile_path=montage_profile,
                    registry_path=registry,
                    dynamic_crop_version=3,
                    continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
                )
                loaded = load_registered_stage_edit_production_bundle(registry)

            self.assertEqual(first, second)
            self.assertEqual(encoded, second.read_bytes())
            self.assertEqual(loaded.dynamic_crop_version, 3)
            self.assertEqual(
                HANDCRAFTED_CONTINUITY_AUTHORITY,
                loaded.continuity_authority,
            )
            self.assertIsNone(loaded.source_pair_scorer_path)
            self.assertIsNone(loaded.source_pair_checkpoint_sha256)
            manifest = json.loads(registry.read_text(encoding="utf-8"))
            self.assertEqual(
                HANDCRAFTED_CONTINUITY_AUTHORITY,
                manifest["continuity_authority"],
            )
            self.assertNotIn("source_pair_scorer", manifest)
            self.assertNotIn("source_pair_production_report", manifest)
            self.assertNotIn(
                "source_pair_absolute_paths_required",
                manifest["storage_policy"],
            )
            public = loaded.to_dict()
            self.assertEqual(
                HANDCRAFTED_CONTINUITY_AUTHORITY,
                public["continuity_authority"],
            )
            self.assertNotIn("source_pair_scorer", public)

            for authority, pattern in (
                (None, "continuity authority is unsupported"),
                ("unknown_continuity_authority", "authority is unsupported"),
                (PRODUCTION_LEARNED_CONTINUITY_AUTHORITY, "schema"),
            ):
                with self.subTest(tampered_authority=authority):
                    tampered = json.loads(json.dumps(manifest))
                    if authority is None:
                        del tampered["continuity_authority"]
                    else:
                        tampered["continuity_authority"] = authority
                    unsigned = dict(tampered)
                    unsigned.pop("registry_sha256")
                    tampered["registry_sha256"] = _canonical_sha256(unsigned)
                    registry.write_text(
                        json.dumps(tampered, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        StageEditProductionRegistryError,
                        pattern,
                    ):
                        load_registered_stage_edit_production_bundle(registry)

    def test_v3_authority_is_explicit_and_pair_arguments_match_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, pair_scorer, pair_report = _checkpoint_v3(root)
            montage_profile = _montage_profile(root)
            for authority in (None, "unknown", " handcrafted_boundary_continuity_v1"):
                with self.subTest(authority=authority):
                    with self.assertRaisesRegex(
                        StageEditProductionRegistryNotReadyError,
                        "explicit supported continuity authority",
                    ):
                        register_stage_edit_production_bundle(
                            checkpoint,
                            professional_montage_grammar_profile_path=(
                                montage_profile
                            ),
                            registry_path=root / "invalid-authority.json",
                            dynamic_crop_version=3,
                            continuity_authority=authority,
                        )
            with self.assertRaisesRegex(
                StageEditProductionRegistryNotReadyError,
                "handcrafted.*cannot register source-pair",
            ):
                register_stage_edit_production_bundle(
                    checkpoint,
                    professional_montage_grammar_profile_path=montage_profile,
                    registry_path=root / "handcrafted-with-pair.json",
                    dynamic_crop_version=3,
                    continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
                    source_pair_scorer_path=pair_scorer,
                    source_pair_production_report_path=pair_report,
                )

            checkpoint_v2 = _checkpoint(root, "checkpoint-v2-authority")
            with self.assertRaisesRegex(
                StageEditProductionRegistryNotReadyError,
                "v2 cannot register a continuity authority",
            ):
                register_stage_edit_production_bundle(
                    checkpoint_v2,
                    professional_montage_grammar_profile_path=montage_profile,
                    registry_path=root / "v2-with-authority.json",
                    continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
                )

    def test_v3_source_pair_file_identity_tampering_fails_closed(self) -> None:
        for artifact in ("scorer", "report"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                checkpoint, pair_scorer, pair_report = _checkpoint_v3(root)
                montage_profile = _montage_profile(root)
                registry = root / "active-production-bundle.json"
                with patch(
                    "crosscut.stage_edit_production_registry."
                    "stage_edit_readiness_report",
                    return_value=_ready_v3(checkpoint),
                ):
                    register_stage_edit_production_bundle(
                        checkpoint,
                        professional_montage_grammar_profile_path=(
                            montage_profile
                        ),
                        registry_path=registry,
                        dynamic_crop_version=3,
                        continuity_authority=(
                            PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                        ),
                        source_pair_scorer_path=pair_scorer,
                        source_pair_production_report_path=pair_report,
                    )
                target = pair_scorer if artifact == "scorer" else pair_report
                target.write_bytes(target.read_bytes() + b"\n")
                with self.assertRaisesRegex(
                    StageEditProductionRegistryNotReadyError,
                    "source-pair artifact identity changed",
                ):
                    load_registered_stage_edit_production_bundle(registry)

    def test_v3_source_pair_report_round_trip_gates_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, pair_scorer, pair_report = _checkpoint_v3(root)
            montage_profile = _montage_profile(root)
            original = json.loads(pair_report.read_text(encoding="utf-8"))
            cases = (
                ("self_hash", "self-hash"),
                ("checkpoint", "checkpoint does not round-trip"),
                ("corpus", "corpus or scorer does not round-trip"),
                ("authorization", "production authorization"),
                ("fold", "all-fold or aggregate"),
                ("aggregate", "all-fold or aggregate"),
            )
            for failure, pattern in cases:
                with self.subTest(failure=failure):
                    report = json.loads(json.dumps(original))
                    if failure == "self_hash":
                        report["report_sha256"] = "0" * 64
                    elif failure == "checkpoint":
                        report["checkpoint"]["checkpoint_sha256"] = "0" * 64
                    elif failure == "corpus":
                        report["final_full_corpus_training"][
                            "corpus_sha256"
                        ] = "0" * 64
                    elif failure == "authorization":
                        report["production_authorization"]["authorized"] = False
                    elif failure == "fold":
                        report["cross_validation"]["folds"][0][
                            "activation_gate"
                        ]["authorized"] = False
                    else:
                        report["cross_validation"][
                            "aggregate_activation_gate"
                        ]["authorized"] = False

                    if failure in ("fold", "aggregate"):
                        unsigned_validation = dict(report["cross_validation"])
                        unsigned_validation.pop("validation_sha256")
                        report["cross_validation"][
                            "validation_sha256"
                        ] = _canonical_sha256(unsigned_validation)
                    if failure != "self_hash":
                        unsigned_report = dict(report)
                        unsigned_report.pop("report_sha256")
                        report["report_sha256"] = _canonical_sha256(
                            unsigned_report
                        )
                    pair_report.write_text(
                        json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    with patch(
                        "crosscut.stage_edit_production_registry."
                        "stage_edit_readiness_report",
                        return_value=_ready_v3(checkpoint),
                    ):
                        with self.assertRaisesRegex(
                            StageEditProductionRegistryNotReadyError,
                            pattern,
                        ):
                            register_stage_edit_production_bundle(
                                checkpoint,
                                professional_montage_grammar_profile_path=(
                                    montage_profile
                                ),
                                registry_path=root / f"{failure}.json",
                                dynamic_crop_version=3,
                                continuity_authority=(
                                    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                                ),
                                source_pair_scorer_path=pair_scorer,
                                source_pair_production_report_path=pair_report,
                            )

    def test_v3_training_declarations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, pair_scorer, pair_report = _checkpoint_v3(root)
            montage_profile = _montage_profile(root)
            training_report_path = checkpoint / "training-report.json"
            original = json.loads(training_report_path.read_text(encoding="utf-8"))
            for failure, pattern in (
                ("checkpoint", "exact v3 checkpoint"),
                ("activation_checkpointing", "activation checkpointing"),
                ("activation_checkpointing_int", "activation checkpointing"),
                ("activation_checkpointing_missing", "activation checkpointing"),
                ("gate", "validation gate"),
                ("heldout", "held-out evidence"),
            ):
                with self.subTest(failure=failure):
                    report = json.loads(json.dumps(original))
                    if failure == "checkpoint":
                        report["checkpoint"]["dynamic_crop_version"] = 2
                    elif failure == "activation_checkpointing":
                        report["model"]["activation_checkpointing"][
                            "enabled"
                        ] = False
                    elif failure == "activation_checkpointing_int":
                        report["model"]["activation_checkpointing"][
                            "enabled"
                        ] = 1
                        report["model"]["activation_checkpointing"][
                            "previous_value"
                        ] = 0
                    elif failure == "activation_checkpointing_missing":
                        report["model"]["activation_checkpointing"].pop(
                            "training_only"
                        )
                    elif failure == "gate":
                        report["optimization"][
                            "best_validation_selection_evidence"
                        ]["dynamic_crop_v3_selection_gate"]["selectable"] = False
                    else:
                        report["held_out_evaluation"][
                            "heldout_dynamic_crop_metrics_used_for_checkpoint_selection"
                        ] = True
                    training_report_path.write_text(
                        json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    with patch(
                        "crosscut.stage_edit_production_registry."
                        "stage_edit_readiness_report",
                        return_value=_ready_v3(checkpoint),
                    ):
                        with self.assertRaisesRegex(
                            StageEditProductionRegistryNotReadyError,
                            pattern,
                        ):
                            register_stage_edit_production_bundle(
                                checkpoint,
                                professional_montage_grammar_profile_path=(
                                    montage_profile
                                ),
                                registry_path=root / f"training-{failure}.json",
                                dynamic_crop_version=3,
                                continuity_authority=(
                                    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                                ),
                                source_pair_scorer_path=pair_scorer,
                                source_pair_production_report_path=pair_report,
                            )

    def test_v2_v3_checkpoint_namespaces_never_mix_or_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, pair_scorer, pair_report = _checkpoint_v3(root)
            montage_profile = _montage_profile(root)
            with self.assertRaisesRegex(
                StageEditProductionRegistryNotReadyError,
                "cannot be interpreted as v2",
            ):
                register_stage_edit_production_bundle(
                    checkpoint,
                    professional_montage_grammar_profile_path=montage_profile,
                    registry_path=root / "v3-as-v2.json",
                )

            (checkpoint / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME).write_text(
                "{}\n", encoding="utf-8"
            )
            (checkpoint / DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME).write_bytes(
                b"mixed-v2-weights"
            )
            with self.assertRaisesRegex(
                StageEditProductionRegistryNotReadyError,
                "mixed",
            ):
                register_stage_edit_production_bundle(
                    checkpoint,
                    professional_montage_grammar_profile_path=montage_profile,
                    registry_path=root / "mixed.json",
                    dynamic_crop_version=3,
                    continuity_authority=(
                        PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                    ),
                    source_pair_scorer_path=pair_scorer,
                    source_pair_production_report_path=pair_report,
                )


if __name__ == "__main__":
    unittest.main()
