from __future__ import annotations

from types import SimpleNamespace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crosscut.pipeline import _registered_stage_edit_bridge_kwargs
from crosscut.stage_edit_continuity_authority import (
    CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2,
    HANDCRAFTED_CONTINUITY_AUTHORITY,
    PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
)
from crosscut.stage_edit_contextual_beam_qa_adapter_v2 import (
    derive_stage_edit_contextual_beam_qa_case_v2,
)
from crosscut.stage_edit_source_pair_contextual_v2_production import (
    cross_validate_stage_edit_contextual_source_pair_v2,
    load_stage_edit_contextual_source_pair_evaluation_bundle_v2,
    new_stage_edit_contextual_source_pair_checkpoint_v2,
    new_stage_edit_contextual_source_pair_production_report_v2,
    save_stage_edit_contextual_source_pair_checkpoint_v2,
    save_stage_edit_contextual_source_pair_report_v2,
    save_stage_edit_contextual_training_corpus_v2,
    train_stage_edit_contextual_source_pair_v2,
)
from crosscut.stage_edit_production_registry import (
    STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_BEAM_QA_MANIFEST_CONTRACT,
    STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_V2_KIND,
    STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_V2_SCHEMA_VERSION,
    STAGE_EDIT_PRODUCTION_REGISTRY_ENV,
    STAGE_EDIT_PRODUCTION_REGISTRY_V3_KIND,
    STAGE_EDIT_PRODUCTION_REGISTRY_V3_SCHEMA_VERSION,
    StageEditProductionRegistryNotReadyError,
    _contextual_beam_qa_manifest_from_verified_cases,
    _registered_contextual_source_pair_identity,
    discover_registered_stage_edit_production_bundle,
    register_stage_edit_production_bundle,
)
from tests.test_stage_edit_contextual_beam_qa_adapter_v2 import (
    _Fixture,
    _ProductionFixture,
)
from tests.test_stage_edit_production_registry import (
    _checkpoint_v3,
    _montage_profile,
    _ready_v3,
)
from tests.test_stage_edit_source_pair_contextual_v2 import _corpus


def _verified_envelope(root: Path):
    root.mkdir()
    fixture = _Fixture(root)
    with patch(
        "crosscut.stage_edit_contextual_beam_qa_adapter_v2."
        "evaluate_post_render_qa",
        return_value=fixture.report,
    ):
        return derive_stage_edit_contextual_beam_qa_case_v2(
            semimarkov=fixture.semimarkov,
            selected_timeline_rank=1,
            post_render_report=fixture.report,
            decode_report_path=fixture.decode_path,
            post_render_report_path=fixture.report_path,
        )


def _identity(directory: Path, envelope) -> dict[str, object]:
    manifest = {
        "contract": (
            STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_BEAM_QA_MANIFEST_CONTRACT
        ),
        "cases": [
            {
                "case": envelope.case.to_dict(),
                "derivation": envelope.derivation.to_dict(),
            }
        ],
    }
    return {
        "bundle_directory": str(directory),
        "checkpoint_sha256": "1" * 64,
        "checkpoint_file_sha256": "2" * 64,
        "report_sha256": "3" * 64,
        "report_file_sha256": "4" * 64,
        "beam_qa_report_sha256": "5" * 64,
        "corpus_sha256": "6" * 64,
        "corpus_file_sha256": "7" * 64,
        "token_schema_fingerprint": "8" * 64,
        "scorer_fingerprint": "9" * 64,
        "verified_beam_qa_cases": (envelope,),
        "beam_qa_manifest": manifest,
    }


class StageEditContextualProductionRegistryTests(unittest.TestCase):
    def test_report_free_bootstrap_reaches_registered_production_discovery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary).resolve()
            contextual_directory = root / "contextual-v2"
            contextual_directory.mkdir()
            corpus = _corpus()
            validation = cross_validate_stage_edit_contextual_source_pair_v2(
                corpus
            )
            training = train_stage_edit_contextual_source_pair_v2(corpus)
            checkpoint = new_stage_edit_contextual_source_pair_checkpoint_v2(
                "report-free-bootstrap",
                training,
                created_at="2026-08-25T12:00:00+09:00",
            )
            corpus_path = save_stage_edit_contextual_training_corpus_v2(
                contextual_directory, corpus
            )
            checkpoint_path = save_stage_edit_contextual_source_pair_checkpoint_v2(
                contextual_directory, checkpoint
            )
            self.assertFalse(
                (
                    contextual_directory
                    / "source-pair-contextual-production-report-v2.json"
                ).exists()
            )
            evaluation_bundle = (
                load_stage_edit_contextual_source_pair_evaluation_bundle_v2(
                    contextual_directory,
                    expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
                    expected_checkpoint_file_sha256=hashlib.sha256(
                        checkpoint_path.read_bytes()
                    ).hexdigest(),
                    expected_corpus_sha256=corpus.content_sha256,
                    expected_corpus_file_sha256=hashlib.sha256(
                        corpus_path.read_bytes()
                    ).hexdigest(),
                    expected_token_schema_fingerprint=(
                        corpus.token_schema_fingerprint
                    ),
                    expected_scorer_fingerprint=checkpoint.scorer_fingerprint,
                    expected_runtime_equivalence_sha256=(
                        checkpoint.runtime_equivalence_sha256
                    ),
                )
            )
            fixtures = (
                _ProductionFixture(
                    root / "beam-qa-bang",
                    song="BANG",
                    evaluation_bundle=evaluation_bundle,
                ),
                _ProductionFixture(
                    root / "beam-qa-blackhole",
                    song="BLACKHOLE",
                    evaluation_bundle=evaluation_bundle,
                ),
            )
            reports_by_video = {
                fixture.video_path: fixture.report for fixture in fixtures
            }

            def evaluate_bound_report(artifact_set, **_kwargs):
                return reports_by_video[artifact_set.video_path]

            with patch(
                "crosscut.stage_edit_contextual_beam_qa_adapter_v2."
                "evaluate_post_render_qa",
                side_effect=evaluate_bound_report,
            ):
                envelopes = tuple(
                    derive_stage_edit_contextual_beam_qa_case_v2(
                        semimarkov=fixture.semimarkov,
                        selected_timeline_rank=1,
                        post_render_report=fixture.report,
                        decode_report_path=fixture.decode_path,
                        post_render_report_path=fixture.report_path,
                        contextual_evaluation_bundle_v2=evaluation_bundle,
                    )
                    for fixture in fixtures
                )
                self.assertTrue(all(value.case.authorized for value in envelopes))
                report = new_stage_edit_contextual_source_pair_production_report_v2(
                    report_id="report-free-bootstrap",
                    corpus=corpus,
                    cross_validation=validation,
                    final_training=training,
                    checkpoint=checkpoint,
                    beam_qa_evaluation_corpus_sha256="e" * 64,
                    verified_beam_qa_cases=envelopes,
                    created_at="2026-08-25T12:01:00+09:00",
                )
                save_stage_edit_contextual_source_pair_report_v2(
                    contextual_directory, report
                )

                dynamic_crop_checkpoint, _scorer, _legacy_report = _checkpoint_v3(
                    root
                )
                montage_profile = _montage_profile(root)
                registry = root / "active-production-bundle.json"
                with patch(
                    "crosscut.stage_edit_production_registry."
                    "stage_edit_readiness_report",
                    return_value=_ready_v3(dynamic_crop_checkpoint),
                ):
                    register_stage_edit_production_bundle(
                        dynamic_crop_checkpoint,
                        professional_montage_grammar_profile_path=montage_profile,
                        registry_path=registry,
                        dynamic_crop_version=3,
                        continuity_authority=(
                            PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
                        ),
                        contextual_source_pair_bundle_directory=(
                            contextual_directory
                        ),
                        contextual_source_pair_verified_beam_qa_cases=envelopes,
                    )
                    with patch.dict(
                        os.environ,
                        {STAGE_EDIT_PRODUCTION_REGISTRY_ENV: str(registry)},
                        clear=False,
                    ):
                        discovered = (
                            discover_registered_stage_edit_production_bundle()
                        )
            self.assertIsNotNone(discovered)
            assert discovered is not None
            self.assertEqual(
                report.report_sha256,
                discovered.contextual_source_pair_report_sha256,
            )
            self.assertEqual(
                checkpoint.scorer_fingerprint,
                discovered.contextual_source_pair_scorer_fingerprint,
            )
            self.assertEqual(
                tuple(value.case for value in envelopes),
                tuple(
                    value.case
                    for value in (
                        discovered.contextual_source_pair_verified_beam_qa_cases
                    )
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                _registered_stage_edit_bridge_kwargs(
                    SimpleNamespace(
                        dynamic_crop_version=3,
                        continuity_authority=(
                            CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2
                        ),
                    )
                )
            checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"\n")
            with patch.dict(
                os.environ,
                {STAGE_EDIT_PRODUCTION_REGISTRY_ENV: str(registry)},
                clear=False,
            ):
                with self.assertRaisesRegex(
                    StageEditProductionRegistryNotReadyError,
                    "Beam-QA replay failed",
                ):
                    discover_registered_stage_edit_production_bundle()

    def test_contextual_registration_and_automatic_discovery_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary).resolve()
            checkpoint, _unused_scorer, _unused_report = _checkpoint_v3(root)
            montage_profile = _montage_profile(root)
            contextual_directory = root / "contextual-v2"
            contextual_directory.mkdir()
            envelope = _verified_envelope(root / "beam-qa")
            identity = _identity(contextual_directory, envelope)
            registry = root / "active-production-bundle.json"

            with (
                patch(
                    "crosscut.stage_edit_production_registry."
                    "stage_edit_readiness_report",
                    return_value=_ready_v3(checkpoint),
                ),
                patch(
                    "crosscut.stage_edit_production_registry."
                    "_validated_contextual_source_pair_identity",
                    return_value=identity,
                ),
                patch(
                    "crosscut.stage_edit_production_registry."
                    "load_registered_stage_edit_production_bundle",
                    return_value=object(),
                ),
            ):
                register_stage_edit_production_bundle(
                    checkpoint,
                    professional_montage_grammar_profile_path=montage_profile,
                    registry_path=registry,
                    dynamic_crop_version=3,
                    continuity_authority=(
                        PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
                    ),
                    contextual_source_pair_bundle_directory=(
                        contextual_directory
                    ),
                    contextual_source_pair_verified_beam_qa_cases=(envelope,),
                )

            payload = json.loads(registry.read_text(encoding="utf-8"))
            self.assertEqual(
                STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_V2_SCHEMA_VERSION,
                payload["schema_version"],
            )
            self.assertEqual(
                STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_V2_KIND,
                payload["registry_kind"],
            )
            self.assertEqual(
                identity["beam_qa_manifest"], payload["contextual_beam_qa_cases"]
            )
            self.assertEqual(
                identity["checkpoint_file_sha256"],
                payload["contextual_source_pair_v2"]["checkpoint"][
                    "file_sha256"
                ],
            )

            with (
                patch.dict(
                    os.environ,
                    {STAGE_EDIT_PRODUCTION_REGISTRY_ENV: str(registry)},
                    clear=False,
                ),
                patch(
                    "crosscut.stage_edit_production_registry."
                    "stage_edit_readiness_report",
                    return_value=_ready_v3(checkpoint),
                ),
                patch(
                    "crosscut.stage_edit_production_registry."
                    "_registered_contextual_source_pair_identity",
                    return_value=identity,
                ) as contextual_loader,
            ):
                loaded = discover_registered_stage_edit_production_bundle()
            self.assertIsNotNone(loaded)
            assert loaded is not None
            contextual_loader.assert_called_once()
            self.assertEqual(
                PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
                loaded.continuity_authority,
            )
            self.assertEqual(
                contextual_directory,
                loaded.contextual_source_pair_bundle_directory,
            )
            self.assertEqual(
                (envelope,), loaded.contextual_source_pair_verified_beam_qa_cases
            )

    def test_contextual_registry_tamper_fails_even_after_outer_reseal(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary).resolve()
            contextual_directory = root / "contextual-v2"
            contextual_directory.mkdir()
            envelope = _verified_envelope(root / "beam-qa")
            identity = _identity(contextual_directory, envelope)
            raw_manifest = identity["beam_qa_manifest"]
            payload = {
                "contextual_source_pair_v2": {
                    "bundle_directory": str(contextual_directory),
                    "checkpoint": {
                        "filename": "source-pair-contextual-scorer-v2.json",
                        "content_sha256": "1" * 64,
                        "file_sha256": "0" * 64,
                    },
                    "production_report": {
                        "filename": (
                            "source-pair-contextual-production-report-v2.json"
                        ),
                        "content_sha256": "3" * 64,
                        "file_sha256": "4" * 64,
                        "beam_qa_report_sha256": "5" * 64,
                    },
                    "training_corpus": {
                        "filename": (
                            "source-pair-contextual-training-corpus-v2.json"
                        ),
                        "content_sha256": "6" * 64,
                        "file_sha256": "7" * 64,
                    },
                    "token_schema_fingerprint": "8" * 64,
                    "scorer_fingerprint": "9" * 64,
                },
                "contextual_beam_qa_cases": raw_manifest,
            }
            with (
                patch(
                    "crosscut.stage_edit_contextual_beam_qa_adapter_v2."
                    "rehydrate_stage_edit_contextual_beam_qa_case_v2",
                    return_value=envelope,
                    create=True,
                ) as rehydrate,
                patch(
                    "crosscut.stage_edit_production_registry."
                    "_validated_contextual_source_pair_identity",
                    return_value=identity,
                ),
            ):
                with self.assertRaisesRegex(
                    StageEditProductionRegistryNotReadyError,
                    "artifact identity changed",
                ):
                    _registered_contextual_source_pair_identity(payload)
            rehydrate.assert_called_once()

    def test_registered_bridge_kwargs_are_complete_and_reject_raw_cases(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary).resolve()
            envelope = _verified_envelope(root / "beam-qa")
            contextual_directory = root / "contextual-v2"
            contextual_directory.mkdir()
            bundle = SimpleNamespace(
                dynamic_crop_version=3,
                continuity_authority=(
                    PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
                ),
                source_pair_scorer_path=None,
                source_pair_scorer_file_sha256=None,
                source_pair_scorer_fingerprint=None,
                source_pair_checkpoint_sha256=None,
                source_pair_corpus_sha256=None,
                contextual_source_pair_bundle_directory=contextual_directory,
                contextual_source_pair_checkpoint_sha256="1" * 64,
                contextual_source_pair_checkpoint_file_sha256="2" * 64,
                contextual_source_pair_report_sha256="3" * 64,
                contextual_source_pair_report_file_sha256="4" * 64,
                contextual_source_pair_beam_qa_report_sha256="5" * 64,
                contextual_source_pair_corpus_sha256="6" * 64,
                contextual_source_pair_corpus_file_sha256="7" * 64,
                contextual_source_pair_token_schema_fingerprint="8" * 64,
                contextual_source_pair_verified_beam_qa_cases=(envelope,),
            )
            kwargs = _registered_stage_edit_bridge_kwargs(bundle)
            self.assertEqual(
                PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
                kwargs["continuity_authority"],
            )
            self.assertEqual(12, len(kwargs))
            self.assertEqual(
                (envelope,),
                kwargs["contextual_source_pair_verified_beam_qa_cases"],
            )

            bundle.contextual_source_pair_verified_beam_qa_cases = (
                envelope.case,
            )
            with self.assertRaisesRegex(RuntimeError, "adapter-issued"):
                _registered_stage_edit_bridge_kwargs(bundle)

            with self.assertRaisesRegex(
                StageEditProductionRegistryNotReadyError,
                "raw or stale Beam-QA DTOs",
            ):
                _contextual_beam_qa_manifest_from_verified_cases(
                    (envelope.case,)
                )

    def test_legacy_registry_versions_and_bridge_kwargs_remain_unchanged(self) -> None:
        self.assertEqual(4, STAGE_EDIT_PRODUCTION_REGISTRY_V3_SCHEMA_VERSION)
        self.assertEqual(
            "ive_stage_edit_dynamic_crop_v3_active_production_bundle",
            STAGE_EDIT_PRODUCTION_REGISTRY_V3_KIND,
        )
        self.assertEqual(
            {
                "dynamic_crop_version": 2,
                "continuity_authority": HANDCRAFTED_CONTINUITY_AUTHORITY,
            },
            _registered_stage_edit_bridge_kwargs(
                SimpleNamespace(dynamic_crop_version=2, continuity_authority=None)
            ),
        )
        learned = SimpleNamespace(
            dynamic_crop_version=3,
            continuity_authority=PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
            source_pair_scorer_path=Path("/absolute/source-pair-scorer.json"),
            source_pair_scorer_file_sha256="1" * 64,
            source_pair_scorer_fingerprint="2" * 64,
            source_pair_checkpoint_sha256="3" * 64,
            source_pair_corpus_sha256="4" * 64,
        )
        learned_kwargs = _registered_stage_edit_bridge_kwargs(learned)
        self.assertEqual(
            PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
            learned_kwargs["continuity_authority"],
        )
        self.assertNotIn(
            "contextual_source_pair_bundle_directory", learned_kwargs
        )


if __name__ == "__main__":
    unittest.main()
