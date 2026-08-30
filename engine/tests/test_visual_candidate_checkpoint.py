from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from crosscut.timeline import ShotType
from crosscut import crop_aware_candidate_rerank, visual_score
from crosscut.visual_candidate_checkpoint import (
    CheckpointCodeFingerprintMigration,
    VISUAL_CANDIDATE_CHECKPOINT_ANALYSIS_BASIS,
    VISUAL_CANDIDATE_CHECKPOINT_CODE_FINGERPRINT_MIGRATIONS,
    VISUAL_CANDIDATE_CHECKPOINT_CONTRACT,
    canonical_json_sha256,
    find_latest_matching_visual_candidate_checkpoint,
    load_visual_candidate_checkpoint,
    materialized_candidate_checkpoint_payload,
)
from crosscut.visual_feature_cache import python_symbol_dependency_sha256
from crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan


def _candidate(index: int) -> SimpleNamespace:
    summaries = tuple(
        {"candidate_index": value, "hard_failure_codes": ["focus"]}
        for value in range(8)
    )
    plan = VisualTimelinePlan(
        choices=(
            VisualTimelineChoice(
                segment_index=0,
                source_index=0,
                canonical_master_start=0.0,
                canonical_master_end=1.0,
                source_local_start=0.0,
                source_local_end=1.0,
                quality_score=0.8,
                composition_score=0.7,
                boundary_score=0.6,
                shot_scale=0.3,
                shot_type=ShotType.MEDIUM,
                shot_id="0:0",
            ),
            VisualTimelineChoice(
                segment_index=1,
                source_index=1,
                canonical_master_start=1.0,
                canonical_master_end=2.0,
                source_local_start=1.0,
                source_local_end=2.0,
                quality_score=0.8,
                composition_score=0.7,
                boundary_score=0.6,
                shot_scale=0.3,
                shot_type=ShotType.MEDIUM,
                shot_id="1:1",
            ),
        ),
        optimizer_score=90.0 - index,
        selected_candidate_index=index,
        candidate_review_summaries=summaries,
    )
    return SimpleNamespace(
        candidate_index=index,
        review_rank=index,
        cadence_profile="balanced",
        visual_plan=plan,
        candidate_record=SimpleNamespace(
            review=SimpleNamespace(
                hard_failures=(SimpleNamespace(code="focus_mismatch_exceeded"),)
            )
        ),
    )


class VisualCandidateCheckpointTests(unittest.TestCase):
    @staticmethod
    def _write_checkpoint(path: Path, identity: dict[str, object]) -> None:
        path.write_text(
            json.dumps(
                {
                    "checkpoint_contract": (
                        VISUAL_CANDIDATE_CHECKPOINT_CONTRACT
                    ),
                    "analysis_basis": (
                        VISUAL_CANDIDATE_CHECKPOINT_ANALYSIS_BASIS
                    ),
                    "input_identity": identity,
                    "input_identity_sha256": canonical_json_sha256(identity),
                    "materialized_candidate_plans": [
                        materialized_candidate_checkpoint_payload(
                            _candidate(index),
                            selected_for_initial_stream=True,
                        )
                        for index in range(8)
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_round_trip_reconstructs_full_plans_and_exact_initial_eight(self) -> None:
        identity = {"sources": ["a", "b"], "policy": "v1"}
        digest = canonical_json_sha256(identity)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "IVE-XOXZ-pre-derivation-candidate-selection.json"
            payload = {
                "checkpoint_contract": VISUAL_CANDIDATE_CHECKPOINT_CONTRACT,
                "analysis_basis": VISUAL_CANDIDATE_CHECKPOINT_ANALYSIS_BASIS,
                "input_identity": identity,
                "input_identity_sha256": digest,
                "materialized_candidate_plans": [
                    materialized_candidate_checkpoint_payload(
                        _candidate(index),
                        selected_for_initial_stream=True,
                    )
                    for index in range(8)
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_visual_candidate_checkpoint(
                path,
                expected_input_identity_sha256=digest,
                expected_source_count=2,
            )
            latest = find_latest_matching_visual_candidate_checkpoint(
                root,
                expected_input_identity_sha256=digest,
                expected_source_count=2,
            )

        self.assertEqual(len(loaded.candidates), 8)
        self.assertEqual(len(loaded.initial_candidates), 8)
        self.assertEqual(loaded.candidates[3].candidate_index, 3)
        self.assertEqual(
            loaded.candidates[3].pre_review_hard_failure_codes,
            ("focus_mismatch_exceeded",),
        )
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.path, loaded.path)

    def test_input_identity_mismatch_fails_closed(self) -> None:
        identity = {"sources": ["a", "b"]}
        digest = canonical_json_sha256(identity)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_text(
                json.dumps(
                    {
                        "checkpoint_contract": (
                            VISUAL_CANDIDATE_CHECKPOINT_CONTRACT
                        ),
                        "analysis_basis": (
                            VISUAL_CANDIDATE_CHECKPOINT_ANALYSIS_BASIS
                        ),
                        "input_identity": identity,
                        "input_identity_sha256": digest,
                        "materialized_candidate_plans": [
                            materialized_candidate_checkpoint_payload(
                                _candidate(index),
                                selected_for_initial_stream=True,
                            )
                            for index in range(8)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "input identity mismatch"):
                load_visual_candidate_checkpoint(
                    path,
                    expected_input_identity_sha256="f" * 64,
                    expected_source_count=2,
                )

    def test_exact_directional_code_fingerprint_migration_reuses_plan(self) -> None:
        legacy_selector = "1" * 64
        current_selector = "2" * 64
        legacy_crop = "3" * 64
        current_crop = "4" * 64
        legacy = {
            "sources": ["a", "b"],
            "canonical_spans": [[0.0, 1.0]],
            "selector_code_sha256": legacy_selector,
            "crop_rerank_code_sha256": legacy_crop,
        }
        current = {
            **legacy,
            "selector_code_sha256": current_selector,
            "crop_rerank_code_sha256": current_crop,
            "code_fingerprint_contract": "python_module_symbol_dependency_ast_v2",
        }
        migration = CheckpointCodeFingerprintMigration(
            contract="test_exact_code_hash_v1_to_v2",
            legacy_selector_code_sha256=legacy_selector,
            current_selector_code_sha256=current_selector,
            legacy_crop_rerank_code_sha256=legacy_crop,
            current_crop_rerank_code_sha256=current_crop,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = (
                root
                / "IVE-XOXZ-pre-derivation-candidate-selection.json"
            )
            self._write_checkpoint(path, legacy)

            loaded = load_visual_candidate_checkpoint(
                path,
                expected_input_identity_sha256=canonical_json_sha256(current),
                expected_source_count=2,
                expected_input_identity=current,
                code_fingerprint_migrations=(migration,),
            )
            latest = find_latest_matching_visual_candidate_checkpoint(
                root,
                expected_input_identity_sha256=canonical_json_sha256(current),
                expected_source_count=2,
                expected_input_identity=current,
                code_fingerprint_migrations=(migration,),
            )

        self.assertEqual(
            loaded.input_identity_sha256,
            canonical_json_sha256(current),
        )
        self.assertEqual(
            loaded.stored_input_identity_sha256,
            canonical_json_sha256(legacy),
        )
        self.assertEqual(
            loaded.code_fingerprint_migration,
            "test_exact_code_hash_v1_to_v2",
        )
        self.assertIsNotNone(latest)

    def test_production_migration_is_bound_to_the_audited_checkpoint(self) -> None:
        migration = next(
            item
            for item in VISUAL_CANDIDATE_CHECKPOINT_CODE_FINGERPRINT_MIGRATIONS
            if item.legacy_fingerprint_contract is None
        )
        legacy = {
            "sources": ["a", "b"],
            "canonical_spans": [[0.0, 1.0]],
            "selector_code_sha256": (
                "363acfa826d846bfc62396f483ce1adc6d0b1ff903ffd32c39630dc2db70e857"
            ),
            "crop_rerank_code_sha256": (
                "23aa118ec3a822544ca36d90ba72c821a6e9157175efe863b7c181b6e73a97ba"
            ),
        }
        current = {
            **legacy,
            "selector_code_sha256": (
                "fa8ed04566a3d6c0d95ddb6eeb58d10e1331c99beb8585d2759d2b2b6841ab0f"
            ),
            "crop_rerank_code_sha256": (
                "0b522e5cbd12142d9b6d4b18e36fb982132892a4b5addaa8f84423d530b476b9"
            ),
            "code_fingerprint_contract": (
                "python_module_symbol_dependency_ast_v2"
            ),
        }
        earlier_selector = {
            **legacy,
            "selector_code_sha256": (
                "f94d6868cc1cbdc9cfb3a8733057873a5664b769462fc2bc996a91d7d5e66f7a"
            ),
        }

        self.assertTrue(migration.matches(legacy, current))
        self.assertFalse(migration.matches(earlier_selector, current))

    def test_production_migration_accepts_only_the_audited_intermediate_hash(
        self,
    ) -> None:
        migration = next(
            item
            for item in VISUAL_CANDIDATE_CHECKPOINT_CODE_FINGERPRINT_MIGRATIONS
            if item.legacy_selector_code_sha256.startswith("9ad134")
        )
        legacy = {
            "sources": ["a", "b"],
            "canonical_spans": [[0.0, 1.0]],
            "selector_code_sha256": (
                "9ad134a8bac4a4946bb56d5ca4d161ee226320fe2504b826fda6ad1c5abc5982"
            ),
            "crop_rerank_code_sha256": (
                "0b522e5cbd12142d9b6d4b18e36fb982132892a4b5addaa8f84423d530b476b9"
            ),
            "code_fingerprint_contract": (
                "python_module_symbol_dependency_ast_v2"
            ),
        }
        current = {
            **legacy,
            "selector_code_sha256": (
                "fa8ed04566a3d6c0d95ddb6eeb58d10e1331c99beb8585d2759d2b2b6841ab0f"
            ),
        }
        unknown = {
            **legacy,
            "selector_code_sha256": "f" * 64,
        }

        self.assertTrue(migration.matches(legacy, current))
        self.assertFalse(migration.matches(unknown, current))
        self.assertFalse(migration.matches(current, legacy))

    def test_production_migration_accepts_strict_cache_predecessor(self) -> None:
        migration = next(
            item
            for item in VISUAL_CANDIDATE_CHECKPOINT_CODE_FINGERPRINT_MIGRATIONS
            if item.legacy_selector_code_sha256.startswith("59aa2115")
        )
        legacy = {
            "sources": ["a", "b"],
            "canonical_spans": [[0.0, 1.0]],
            "selector_code_sha256": (
                "59aa2115e68c773db670641ecf44581006ab6a0bb4dc7d70f8f6e65d87e731e8"
            ),
            "crop_rerank_code_sha256": (
                "0b522e5cbd12142d9b6d4b18e36fb982132892a4b5addaa8f84423d530b476b9"
            ),
            "code_fingerprint_contract": (
                "python_module_symbol_dependency_ast_v2"
            ),
        }
        current = {
            **legacy,
            "selector_code_sha256": (
                "fa8ed04566a3d6c0d95ddb6eeb58d10e1331c99beb8585d2759d2b2b6841ab0f"
            ),
        }

        self.assertTrue(migration.matches(legacy, current))
        self.assertFalse(migration.matches(current, legacy))

    def test_production_migration_accepts_joint_stream_predecessor(self) -> None:
        migration = next(
            item
            for item in VISUAL_CANDIDATE_CHECKPOINT_CODE_FINGERPRINT_MIGRATIONS
            if item.legacy_selector_code_sha256.startswith("d1529dae")
        )
        legacy = {
            "sources": ["a", "b"],
            "canonical_spans": [[0.0, 1.0]],
            "selector_code_sha256": (
                "d1529daef216659bf3fa9b482a1fb5ab1bf53ec8c1ac1b3f486b2d65ae349230"
            ),
            "crop_rerank_code_sha256": (
                "0b522e5cbd12142d9b6d4b18e36fb982132892a4b5addaa8f84423d530b476b9"
            ),
            "code_fingerprint_contract": (
                "python_module_symbol_dependency_ast_v2"
            ),
        }
        current = {
            **legacy,
            "selector_code_sha256": (
                "fa8ed04566a3d6c0d95ddb6eeb58d10e1331c99beb8585d2759d2b2b6841ab0f"
            ),
        }

        self.assertTrue(migration.matches(legacy, current))
        self.assertFalse(migration.matches(current, legacy))

    def test_production_migration_accepts_workspace_seed_predecessor(self) -> None:
        migration = next(
            item
            for item in VISUAL_CANDIDATE_CHECKPOINT_CODE_FINGERPRINT_MIGRATIONS
            if item.legacy_selector_code_sha256.startswith("8004e6ff")
        )
        legacy = {
            "sources": ["a", "b"],
            "canonical_spans": [[0.0, 1.0]],
            "selector_code_sha256": (
                "8004e6ff1e9813da730b07e3c031b940666cf27707cd1b231fc6af36d0d40aa2"
            ),
            "crop_rerank_code_sha256": (
                "0b522e5cbd12142d9b6d4b18e36fb982132892a4b5addaa8f84423d530b476b9"
            ),
            "code_fingerprint_contract": (
                "python_module_symbol_dependency_ast_v2"
            ),
        }
        current = {
            **legacy,
            "selector_code_sha256": (
                "fa8ed04566a3d6c0d95ddb6eeb58d10e1331c99beb8585d2759d2b2b6841ab0f"
            ),
        }

        self.assertTrue(migration.matches(legacy, current))
        self.assertFalse(migration.matches(current, legacy))

    def test_production_migration_accepts_face_deferral_predecessor(self) -> None:
        migration = next(
            item
            for item in VISUAL_CANDIDATE_CHECKPOINT_CODE_FINGERPRINT_MIGRATIONS
            if item.legacy_selector_code_sha256.startswith("91442b64")
        )
        legacy = {
            "sources": ["a", "b"],
            "canonical_spans": [[0.0, 1.0]],
            "selector_code_sha256": (
                "91442b648f3a9c17ec359c01edc67e85a2422fcc4f4f273b6990d1e0d6e0bd35"
            ),
            "crop_rerank_code_sha256": (
                "0b522e5cbd12142d9b6d4b18e36fb982132892a4b5addaa8f84423d530b476b9"
            ),
            "code_fingerprint_contract": (
                "python_module_symbol_dependency_ast_v2"
            ),
        }
        current = {
            **legacy,
            "selector_code_sha256": (
                "fa8ed04566a3d6c0d95ddb6eeb58d10e1331c99beb8585d2759d2b2b6841ab0f"
            ),
        }

        self.assertTrue(migration.matches(legacy, current))
        self.assertFalse(migration.matches(current, legacy))

    def test_production_migration_current_hashes_match_runtime_code(self) -> None:
        selector_digest = python_symbol_dependency_sha256(
            Path(visual_score.__file__),
            root_symbols=("build_visual_timeline_candidate_set",),
        )
        crop_digest = python_symbol_dependency_sha256(
            Path(crop_aware_candidate_rerank.__file__),
            root_symbols=(
                "rerank_materialized_visual_candidates_for_safe_crops",
            ),
        )
        for migration in (
            VISUAL_CANDIDATE_CHECKPOINT_CODE_FINGERPRINT_MIGRATIONS
        ):
            self.assertEqual(
                migration.current_selector_code_sha256,
                selector_digest,
            )
            self.assertEqual(
                migration.current_crop_rerank_code_sha256,
                crop_digest,
            )

    def test_production_migration_accepts_descriptor_basis_cache_predecessor(
        self,
    ) -> None:
        migration = next(
            item
            for item in VISUAL_CANDIDATE_CHECKPOINT_CODE_FINGERPRINT_MIGRATIONS
            if item.legacy_selector_code_sha256.startswith("0d10dbfa")
        )
        legacy = {
            "sources": ["a", "b"],
            "canonical_spans": [[0.0, 1.0]],
            "selector_code_sha256": (
                "0d10dbfa3fbf72f51a85692702ec0677750e64111996028c398ec971aff428fb"
            ),
            "crop_rerank_code_sha256": (
                "0b522e5cbd12142d9b6d4b18e36fb982132892a4b5addaa8f84423d530b476b9"
            ),
            "code_fingerprint_contract": (
                "python_module_symbol_dependency_ast_v2"
            ),
        }
        current = {
            **legacy,
            "selector_code_sha256": (
                "fa8ed04566a3d6c0d95ddb6eeb58d10e1331c99beb8585d2759d2b2b6841ab0f"
            ),
        }

        self.assertTrue(migration.matches(legacy, current))
        self.assertFalse(migration.matches(current, legacy))

    def test_code_migration_rejects_non_code_reverse_unknown_and_tamper(self) -> None:
        legacy_selector = "1" * 64
        current_selector = "2" * 64
        legacy_crop = "3" * 64
        current_crop = "4" * 64
        legacy = {
            "sources": ["a", "b"],
            "canonical_spans": [[0.0, 1.0]],
            "selector_code_sha256": legacy_selector,
            "crop_rerank_code_sha256": legacy_crop,
        }
        current = {
            **legacy,
            "selector_code_sha256": current_selector,
            "crop_rerank_code_sha256": current_crop,
            "code_fingerprint_contract": "python_module_symbol_dependency_ast_v2",
        }
        migration = CheckpointCodeFingerprintMigration(
            contract="test_exact_code_hash_v1_to_v2",
            legacy_selector_code_sha256=legacy_selector,
            current_selector_code_sha256=current_selector,
            legacy_crop_rerank_code_sha256=legacy_crop,
            current_crop_rerank_code_sha256=current_crop,
        )
        mutations = (
            {**current, "sources": ["a", "foreign"]},
            {
                **legacy,
                "selector_code_sha256": "9" * 64,
                "crop_rerank_code_sha256": current_crop,
                "code_fingerprint_contract": (
                    "python_module_symbol_dependency_ast_v2"
                ),
            },
            {
                **legacy,
                "selector_code_sha256": legacy_selector,
                "crop_rerank_code_sha256": legacy_crop,
                "code_fingerprint_contract": (
                    "python_module_symbol_dependency_ast_v2"
                ),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            self._write_checkpoint(path, legacy)
            for mutated in mutations:
                with self.subTest(mutated=mutated):
                    with self.assertRaisesRegex(
                        ValueError,
                        "input identity mismatch",
                    ):
                        load_visual_candidate_checkpoint(
                            path,
                            expected_input_identity_sha256=(
                                canonical_json_sha256(mutated)
                            ),
                            expected_source_count=2,
                            expected_input_identity=mutated,
                            code_fingerprint_migrations=(migration,),
                        )

            self._write_checkpoint(path, current)
            with self.assertRaisesRegex(ValueError, "input identity mismatch"):
                load_visual_candidate_checkpoint(
                    path,
                    expected_input_identity_sha256=(
                        canonical_json_sha256(legacy)
                    ),
                    expected_source_count=2,
                    expected_input_identity=legacy,
                    code_fingerprint_migrations=(migration,),
                )

            self._write_checkpoint(path, legacy)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["input_identity"]["sources"] = ["tampered"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_visual_candidate_checkpoint(
                    path,
                    expected_input_identity_sha256=canonical_json_sha256(current),
                    expected_source_count=2,
                    expected_input_identity=current,
                    code_fingerprint_migrations=(migration,),
                )

    def test_code_migration_non_code_comparison_is_canonical_json_exact(self) -> None:
        migration = CheckpointCodeFingerprintMigration(
            contract="test_canonical_json_exact",
            legacy_selector_code_sha256="1" * 64,
            current_selector_code_sha256="2" * 64,
            legacy_crop_rerank_code_sha256="3" * 64,
            current_crop_rerank_code_sha256="4" * 64,
        )
        legacy = {
            "maximum_results": 32,
            "enabled": False,
            "selector_code_sha256": "1" * 64,
            "crop_rerank_code_sha256": "3" * 64,
        }
        current = {
            "maximum_results": 32,
            "enabled": False,
            "selector_code_sha256": "2" * 64,
            "crop_rerank_code_sha256": "4" * 64,
            "code_fingerprint_contract": (
                "python_module_symbol_dependency_ast_v2"
            ),
        }

        self.assertTrue(migration.matches(legacy, current))
        self.assertFalse(
            migration.matches(
                legacy,
                {**current, "maximum_results": 32.0},
            )
        )
        self.assertFalse(
            migration.matches(legacy, {**current, "enabled": 0})
        )
        self.assertFalse(
            migration.matches(
                {**legacy, "code_fingerprint_contract": None},
                current,
            )
        )


if __name__ == "__main__":
    unittest.main()
