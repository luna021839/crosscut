from __future__ import annotations

from dataclasses import asdict, replace
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from crosscut import stage_edit_dynamic_crop_v3_production_promotion as promotion
from crosscut.stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
    StageEditTransformerDynamicCropV3,
    load_stage_edit_dynamic_crop_checkpoint_v3,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v3,
    save_stage_edit_dynamic_crop_checkpoint_v3,
)
from crosscut.stage_edit_production_registry import (
    StageEditProductionRegistryNotReadyError,
    _validate_v3_training_report_declarations,
)
from crosscut.professional_montage_policy import (
    load_professional_montage_policy,
)
from crosscut.stage_edit_source_pair_production import (
    produce_stage_edit_source_pair_checkpoint,
)
from crosscut.stage_edit_production_training import (
    ProductionQualityThresholds,
    evaluate_stage_edit_production_quality,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT,
    smoke_stage_edit_config,
    StageEditReadinessReport,
    stage_edit_readiness_report,
)
from crosscut.stage_edit_transition_authority import (
    RENDERER_OWNED_TRANSITION_AUTHORITY,
)
from tests.test_stage_edit_production_training import (
    _passing_dynamic_v3_metrics,
    _passing_metrics,
)
from tests.test_stage_edit_production_registry import _montage_profile
from tests.test_stage_edit_source_pair_production import (
    _corpus as _source_pair_production_corpus,
)

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DynamicCropV3ProductionPromotionTests(unittest.TestCase):
    def test_evaluation_corpus_loader_is_explicitly_group_scoped(self) -> None:
        evidence = {
            song: SimpleNamespace(
                reference_ids=(f"{song}-ref",),
                supervision_paths={
                    f"{song}-ref": Path(f"/tmp/{song}-reference.json")
                },
                snapshot_paths={
                    f"{song}-ref": Path(f"/tmp/{song}-snapshot.json")
                },
                dynamic_crop_manifest_path=Path(f"/tmp/{song}-dynamic.json"),
            )
            for song in promotion.EVALUATION_SONG_IDS
        }
        preflight = SimpleNamespace(song_evidence=evidence)
        corpus = SimpleNamespace(examples=())
        expected_dynamic = object()
        with (
            patch.object(promotion, "_snapshot_map", return_value={}),
            patch.object(
                promotion,
                "load_reference_training_corpus",
                return_value=corpus,
            ) as load_corpus,
            patch.object(
                promotion,
                "_dynamic_artifact_map",
                return_value={},
            ),
            patch.object(
                promotion,
                "build_stage_edit_dynamic_crop_training_corpus_v2",
                return_value=object(),
            ),
            patch.object(
                promotion,
                "build_stage_edit_dynamic_crop_training_corpus_v3",
                return_value=expected_dynamic,
            ),
        ):
            result_corpus, result_dynamic = promotion._build_evaluation_corpus(
                preflight
            )
        self.assertIs(corpus, result_corpus)
        self.assertIs(expected_dynamic, result_dynamic)
        self.assertEqual(
            "group",
            load_corpus.call_args.kwargs["production_edit_mode"],
        )
        self.assertIsNone(
            load_corpus.call_args.kwargs["production_target_member"]
        )

    def test_final_fit_training_corpus_loader_is_explicitly_group_scoped(
        self,
    ) -> None:
        reference_ids_by_song = {
            song: (f"{song}-ref",)
            for song in promotion.final_release.SELECTION_SONG_IDS
        }
        validated = SimpleNamespace(
            reference_ids_by_song=reference_ids_by_song,
            supervision_paths_by_song={
                song: {
                    f"{song}-ref": Path(f"/tmp/{song}-reference.json")
                }
                for song in promotion.final_release.SELECTION_SONG_IDS
            },
            snapshot_paths_by_song={
                song: {f"{song}-ref": Path(f"/tmp/{song}-snapshot.json")}
                for song in promotion.final_release.SELECTION_SONG_IDS
            },
        )
        corpus = object()
        with (
            patch.object(
                promotion,
                "_verify_pointer",
                return_value=Path("/tmp/pointer.json"),
            ),
            patch.object(
                promotion.final_release,
                "_strict_final_fit_plan",
                return_value={"songs": {}, "selection_dataset_manifest": {}},
            ),
            patch.object(
                promotion.final_release,
                "_manifest_maps_from_plan",
                return_value=({}, {}, {}),
            ),
            patch.object(
                promotion.final_release,
                "_validate_manifest_set",
                return_value=validated,
            ),
            patch.object(promotion, "_snapshot_map", return_value={}),
            patch.object(
                promotion,
                "load_reference_training_corpus",
                return_value=corpus,
            ) as load_corpus,
        ):
            result = promotion._build_final_fit_training_corpus(
                {"lineage": {"final_fit_plan": {}}}
            )
        self.assertIs(corpus, result)
        self.assertEqual(
            "group",
            load_corpus.call_args.kwargs["production_edit_mode"],
        )
        self.assertIsNone(
            load_corpus.call_args.kwargs["production_target_member"]
        )

    def _pointer(self, path: Path) -> dict[str, str]:
        return {"path": str(path.resolve()), "file_sha256": _sha(path)}

    def _source_pair_identity(
        self, root: Path
    ) -> tuple[Path, Path, dict[str, str]]:
        root = Path(root).resolve()
        scorer = root / "source-pair-scorer.json"
        scorer.write_text('{"scorer": "test"}', encoding="utf-8")
        report = root / "source-pair-production-report.json"
        report.write_text('{"report": "test"}', encoding="utf-8")
        return scorer, report, {
            "scorer_path": str(scorer),
            "production_report_path": str(report),
            "scorer_file_sha256": _sha(scorer),
            "production_report_file_sha256": _sha(report),
            "production_report_sha256": "1" * 64,
            "cross_validation_sha256": "2" * 64,
            "scorer_fingerprint": "3" * 64,
            "checkpoint_sha256": "4" * 64,
            "corpus_sha256": "5" * 64,
        }

    def _real_source_pair_identity(
        self, root: Path
    ) -> tuple[Path, Path, dict[str, object]]:
        directory = Path(root).resolve() / "real-source-pair"
        result = produce_stage_edit_source_pair_checkpoint(
            _source_pair_production_corpus(),
            directory,
            checkpoint_id="promotion-integration-source-pair",
            created_at="2026-08-25T12:00:00+09:00",
            steps=60,
            learning_rate=0.30,
        )
        identity = promotion._source_pair_production_identity(
            result.checkpoint_path,
            result.report_path,
        )
        return result.checkpoint_path, result.report_path, identity

    def _learned_plan_and_preflight(
        self,
        *,
        plan: dict,
        preflight: promotion.DynamicCropV3ProductionPromotionPreflight,
        identity: dict[str, str],
    ) -> tuple[dict, promotion.DynamicCropV3ProductionPromotionPreflight]:
        learned_plan = copy.deepcopy(plan)
        learned_plan["source_pair_production"] = copy.deepcopy(identity)
        learned_plan["authority"] = promotion._promotion_authority_contract(
            learned_source_pair_authority_present=True
        )
        return learned_plan, replace(
            preflight,
            continuity_authority=(
                promotion.PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
            ),
            source_pair_scorer_path=Path(identity["scorer_path"]),
            source_pair_production_report_path=Path(
                identity["production_report_path"]
            ),
            source_pair_production_identity=copy.deepcopy(identity),
        )

    def _checkpoint_and_report(self, root: Path):
        checkpoint = root / "promoted"
        checkpoint.mkdir()
        weights = checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
        weights.write_bytes(b"byte-identical-final-fit-weight")
        weight_sha = _sha(weights)
        metadata = new_stage_edit_dynamic_crop_checkpoint_metadata_v3(
            "v3-final-fit-test",
            DEFAULT_STAGE_EDIT_SCHEMA,
            replace(
                smoke_stage_edit_config(),
                transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
            ),
            training_steps=111,
            weights_trained=True,
            training_data_id="all-fifteen+dynamic-v3",
        )
        metadata = replace(metadata, weight_sha256=weight_sha)
        metadata_path = checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
        metadata_path.write_text(
            json.dumps(metadata.to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )

        thresholds = ProductionQualityThresholds()
        threshold_payload = asdict(thresholds)
        source_checkpoint = root / "inactive-final-fit"
        source_checkpoint.mkdir()
        source_weights = (
            source_checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
        )
        source_weights.write_bytes(weights.read_bytes())
        source_metadata = (
            source_checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
        )
        source_metadata.write_bytes(metadata_path.read_bytes())
        source_report = source_checkpoint / promotion.final_release.FINAL_FIT_REPORT_FILENAME
        source_report.write_text(
            json.dumps(
                {
                    "source_commit": "a" * 40,
                    "checkpoint": {
                        "directory": str(source_checkpoint.resolve()),
                        "checkpoint_id": metadata.checkpoint_id,
                        "weight_sha256": weight_sha,
                    },
                    "frozen_decisions": {
                        "cut_probability_threshold": 0.5,
                        "quality_thresholds": threshold_payload,
                    },
                }
            ),
            encoding="utf-8",
        )

        pointer_files: list[Path] = []
        final_fit_receipt = root / "lineage-0.json"
        final_fit_receipt.write_text(
            json.dumps({"artifact_content_sha256": "1" * 64}),
            encoding="utf-8",
        )
        pointer_files.append(final_fit_receipt)
        final_fit_plan = root / "lineage-1.json"
        final_fit_plan.write_text(json.dumps({"kind": "plan"}), encoding="utf-8")
        pointer_files.append(final_fit_plan)
        release_freeze = root / "lineage-2.json"
        release_freeze.write_text(
            json.dumps(
                {
                    "freeze_content_sha256": "2" * 64,
                    "source_provenance": {"head_commit_oid": "a" * 40},
                    "frozen_cut_probability_threshold": 0.5,
                    "quality_thresholds": threshold_payload,
                }
            ),
            encoding="utf-8",
        )
        pointer_files.append(release_freeze)
        folds = []
        for index, song in enumerate(promotion.final_release.SELECTION_SONG_IDS):
            fold_receipt = root / f"lineage-{3 + index}.json"
            fold_receipt.write_text(
                json.dumps(
                    {
                        "validation_song_id": song,
                        "checkpoint_id": f"fold-{index}",
                        "checkpoint_weight_sha256": f"{index + 1:064x}",
                        "artifact_content_sha256": f"{index + 4:064x}",
                    }
                ),
                encoding="utf-8",
            )
            pointer_files.append(fold_receipt)
            folds.append(
                {
                    "validation_song_id": song,
                    "checkpoint_id": f"fold-{index}",
                    "checkpoint_weight_sha256": f"{index + 1:064x}",
                    "receipt": self._pointer(fold_receipt),
                    "receipt_content_sha256": f"{index + 4:064x}",
                }
            )
        for index in range(6, 9):
            path = root / f"lineage-{index}.json"
            path.write_text(json.dumps({"index": index}), encoding="utf-8")
            pointer_files.append(path)
        lineage = {
            "final_fit_receipt": self._pointer(pointer_files[0]),
            "final_fit_receipt_content_sha256": "1" * 64,
            "final_fit_plan": self._pointer(pointer_files[1]),
            "release_freeze": self._pointer(pointer_files[2]),
            "release_freeze_content_sha256": "2" * 64,
            "loso_fold_receipts": folds,
            "source_checkpoint": {
                "directory": str(source_checkpoint.resolve()),
                "metadata": self._pointer(source_metadata),
                "weights": self._pointer(source_weights),
                "final_fit_report": self._pointer(source_report),
                "checkpoint_id": metadata.checkpoint_id,
                "training_steps": metadata.training_steps,
                "weight_sha256": weight_sha,
                "production_activation_authorized_in_metadata": False,
            },
        }
        protocol = promotion._evaluation_protocol_contract(
            checkpoint_id=metadata.checkpoint_id,
            checkpoint_weight_sha256=weight_sha,
            cut_probability_threshold=0.5,
            quality_thresholds=threshold_payload,
        )
        base_metrics = _passing_metrics()
        base_metrics["transition"]["label_histogram"] = {"0": 10}
        base_metrics["cut"]["frozen_validation_threshold"] = 0.5
        base_metrics["cut"][
            "threshold_contract"
        ] = STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT
        dynamic_metrics = _passing_dynamic_v3_metrics()
        quality = evaluate_stage_edit_production_quality(
            base_metrics,
            dynamic_crop_v3_metrics=dynamic_metrics,
            thresholds=thresholds,
            transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
        ).to_dict()
        self.assertTrue(quality["passed"])
        structured = {
            "contract": "duration_learned_transition_or_renderer_metadata_coverage_gate_v2",
            "passed": True,
            "reasons": [],
            "thresholds": {
                "minimum_duration_observed_classes": (
                    thresholds.minimum_duration_observed_classes
                ),
                "minimum_duration_label_entropy_bits": (
                    thresholds.minimum_duration_label_entropy_bits
                ),
                "minimum_transition_observed_classes": (
                    thresholds.minimum_transition_observed_classes
                ),
                "minimum_transition_label_entropy_bits": (
                    thresholds.minimum_transition_label_entropy_bits
                ),
            },
            "transition_authority": promotion.transition_authority_contract(
                RENDERER_OWNED_TRANSITION_AUTHORITY
            ),
            "training_partition": {"duration": {}, "transition": {}},
            "heldout": {"duration": {}, "transition": {}},
            "reference_validation_labels_consumed": False,
            "heldout_used_for_model_selection": False,
        }
        self._recomputed_structured_gate = copy.deepcopy(structured)
        evaluations = []
        for song in promotion.EVALUATION_SONG_IDS:
            evaluations.append(
                {
                    "song_id": song,
                    "evidence_status": promotion.EVIDENCE_STATUS[song],
                    "blind_holdout_claim_authorized": False,
                    "human_parity_claim_authorized": False,
                    "reference_ids": [f"{song}-ref-{index}" for index in range(5)],
                    "reference_count": 5,
                    "evidence_manifests": {
                        "materialization_manifest": self._pointer(
                            pointer_files[6]
                        ),
                        "feature_manifest": self._pointer(pointer_files[7]),
                        "dynamic_crop_manifest": self._pointer(
                            pointer_files[8]
                        ),
                    },
                    "checkpoint_id": metadata.checkpoint_id,
                    "checkpoint_weight_sha256": weight_sha,
                    "base_metrics": copy.deepcopy(base_metrics),
                    "dynamic_crop_v3_metrics": copy.deepcopy(dynamic_metrics),
                    "quality_gate": copy.deepcopy(quality),
                    "structured_head_coverage_gate": copy.deepcopy(structured),
                    "evaluation_used_for_optimizer_or_gradient": False,
                    "evaluation_used_for_model_threshold_or_policy_reselection": False,
                }
            )
        plan_path = root / "promotion-plan.json"
        plan_path.write_text("{}", encoding="utf-8")
        plan_content_sha = "9" * 64
        evaluation_evidence = [
            {
                "song_id": evaluation["song_id"],
                "evidence_status": evaluation["evidence_status"],
                "blind_holdout_claim_authorized": False,
                "human_parity_claim_authorized": False,
                **copy.deepcopy(evaluation["evidence_manifests"]),
            }
            for evaluation in evaluations
        ]
        self._promotion_plan = {
            "plan_content_sha256": plan_content_sha,
            "source_provenance": {"head_commit_oid": "a" * 40},
            "lineage": copy.deepcopy(lineage),
            "evaluation_evidence": evaluation_evidence,
            "evaluation_protocol": copy.deepcopy(protocol),
            "destination_checkpoint_directory": str(checkpoint.resolve()),
            "evidence_directory": str((root / "promotion-evidence").resolve()),
            "production_registry_path": str((root / "registry.json").resolve()),
            "professional_montage_profile": {
                "path": str((root / "profile.json").resolve())
            },
            "authority": promotion._promotion_authority_contract(),
        }
        self._planned_reference_ids = {
            evaluation["song_id"]: tuple(evaluation["reference_ids"])
            for evaluation in evaluations
        }
        metadata_sha = _sha(metadata_path)
        unsigned = {
            "schema_version": promotion.PRODUCTION_PROMOTION_REPORT_SCHEMA_VERSION,
            "run_kind": promotion.PRODUCTION_PROMOTION_RUN_KIND,
            "contract": promotion.PRODUCTION_PROMOTION_REPORT_CONTRACT,
            "checkpoint_scope": promotion.PRODUCTION_PROMOTION_CHECKPOINT_SCOPE,
            "created_at": "2026-08-25T00:00:00+00:00",
            "source_commit": "a" * 40,
            "production_ready": True,
            "promotion_plan": self._pointer(plan_path),
            "promotion_plan_content_sha256": plan_content_sha,
            "lineage": lineage,
            "checkpoint": {
                "directory": str(checkpoint.resolve()),
                "checkpoint_format": "dynamic_crop_v3",
                "dynamic_crop_version": 3,
                "metadata_filename": DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
                "weight_filename": DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
                "checkpoint_id": metadata.checkpoint_id,
                "training_steps": metadata.training_steps,
                "weight_sha256": weight_sha,
                "model_contract": metadata.model_contract,
                "target_contract": metadata.target_contract,
                "output_contract": metadata.output_contract,
                "production_activation_authorized": True,
                "copy_integrity": {
                    "source_checkpoint_directory": str(source_checkpoint.resolve()),
                    "source_metadata_sha256": metadata_sha,
                    "source_weight_sha256": weight_sha,
                    "destination_metadata_sha256": metadata_sha,
                    "destination_weight_sha256": weight_sha,
                    "metadata_byte_identical": True,
                    "weights_byte_identical": True,
                    "atomically_installed": True,
                },
            },
            "evaluation_protocol": protocol,
            "evaluations": evaluations,
            "aggregate_gate": {
                "passed": True,
                "evaluation_song_ids": list(promotion.EVALUATION_SONG_IDS),
                "quality_gate_pass_count": 2,
                "structured_head_gate_pass_count": 2,
                "renderer_exact_gate_pass_count": 2,
                "all_evaluations_used_exact_copied_weight": True,
                "registry_activation_authorized": True,
            },
            "authority": promotion._promotion_authority_contract(),
        }
        report = {
            **unsigned,
            "report_sha256": promotion._canonical_sha256(unsigned),
        }
        report_path = checkpoint / "training-report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return checkpoint, metadata, weight_sha, report, report_path

    @contextmanager
    def _patched_plan_replay(self):
        def validate_evidence(record, *, expected_song_id):
            return promotion.PromotionSongEvidence(
                song_id=expected_song_id,
                evidence_status=promotion.EVIDENCE_STATUS[expected_song_id],
                materialization_manifest_path=Path(
                    record["materialization_manifest"]["path"]
                ),
                feature_manifest_path=Path(record["feature_manifest"]["path"]),
                dynamic_crop_manifest_path=Path(
                    record["dynamic_crop_manifest"]["path"]
                ),
                reference_ids=self._planned_reference_ids[expected_song_id],
                supervision_paths={},
                snapshot_paths={},
            )

        with (
            patch.object(
                promotion, "_strict_plan", return_value=self._promotion_plan
            ),
            patch.object(
                promotion,
                "_validate_song_evidence_record",
                side_effect=validate_evidence,
            ),
            patch.object(
                promotion,
                "_build_final_fit_training_corpus",
                return_value=type(
                    "FinalFitCorpus", (), {"examples": ()}
                )(),
            ),
            patch.object(
                promotion,
                "_structured_head_coverage_gate",
                return_value=copy.deepcopy(self._recomputed_structured_gate),
            ),
        ):
            yield

    def _resign(self, report: dict) -> None:
        unsigned = dict(report)
        unsigned.pop("report_sha256", None)
        report["report_sha256"] = promotion._canonical_sha256(unsigned)

    def test_promotion_report_is_accepted_by_shared_readiness_and_registry_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, metadata, weight_sha, report, _ = self._checkpoint_and_report(
                Path(directory)
            )
            with self._patched_plan_replay():
                self.assertEqual(
                    (),
                    promotion.production_promotion_report_errors(
                        report,
                        metadata=metadata,
                        actual_weight_sha256=weight_sha,
                        checkpoint_directory=checkpoint,
                    ),
                )
                readiness = stage_edit_readiness_report(checkpoint)
                self.assertTrue(readiness.learned_inference_ready, readiness.reason)
                self.assertIn("non-blind external evaluation", readiness.reason)
                self.assertIn("exact promoted final-fit weight", readiness.reason)
                self.assertNotIn("held-out", readiness.reason)
                _validate_v3_training_report_declarations(
                    report,
                    metadata=metadata,
                    weight_sha256=weight_sha,
                    checkpoint_directory=checkpoint,
                )

    def test_promotion_report_rejects_training_claims_copy_tamper_and_renderer_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, metadata, weight_sha, original, _ = self._checkpoint_and_report(
                Path(directory)
            )
            mutations = (
                lambda report: report["evaluation_protocol"].__setitem__(
                    "optimizer_steps", 1
                ),
                lambda report: report.__setitem__(
                    "created_at", "2026-08-25T00:00:00"
                ),
                lambda report: report["checkpoint"]["copy_integrity"].__setitem__(
                    "weights_byte_identical", False
                ),
                lambda report: report["checkpoint"]["copy_integrity"].__setitem__(
                    "source_checkpoint_directory", "/tmp/forged-final-fit"
                ),
                lambda report: report["checkpoint"]["copy_integrity"].__setitem__(
                    "source_metadata_sha256", "f" * 64
                ),
                lambda report: report["authority"].__setitem__(
                    "blind_holdout_quality_claim_authorized", True
                ),
                lambda report: report["evaluations"][0][
                    "dynamic_crop_v3_metrics"
                ]["trajectory"]["renderer_decode"].__setitem__(
                    "minimum_safe_zoom_projection_count", 1
                ),
                lambda report: report["evaluations"][0][
                    "dynamic_crop_v3_metrics"
                ]["trajectory"]["renderer_decode"].__setitem__(
                    "eligible_count",
                    report["evaluations"][0]["dynamic_crop_v3_metrics"][
                        "trajectory"
                    ]["renderer_decode"]["attempt_count"]
                    - 1,
                ),
                lambda report: report["evaluations"][0][
                    "dynamic_crop_v3_metrics"
                ]["trajectory"]["renderer_decode"].__setitem__(
                    "eligible_count",
                    float(
                        report["evaluations"][0]["dynamic_crop_v3_metrics"][
                            "trajectory"
                        ]["renderer_decode"]["attempt_count"]
                    ),
                ),
                lambda report: report["evaluations"][0][
                    "structured_head_coverage_gate"
                ].__setitem__("training_partition", {"forged": {}}),
            )
            for mutate in mutations:
                with self.subTest(mutation=mutate):
                    report = copy.deepcopy(original)
                    mutate(report)
                    self._resign(report)
                    with self._patched_plan_replay():
                        errors = promotion.production_promotion_report_errors(
                            report,
                            metadata=metadata,
                            actual_weight_sha256=weight_sha,
                            checkpoint_directory=checkpoint,
                        )
                    self.assertTrue(errors)

    def test_resigned_renderer_and_structured_gate_tamper_fail_all_dispatches(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, metadata, weight_sha, original, report_path = (
                self._checkpoint_and_report(Path(directory))
            )
            mutations = {
                "renderer_exact": lambda report: report["evaluations"][0][
                    "dynamic_crop_v3_metrics"
                ]["trajectory"]["renderer_decode"].__setitem__(
                    "eligible_count",
                    report["evaluations"][0]["dynamic_crop_v3_metrics"][
                        "trajectory"
                    ]["renderer_decode"]["attempt_count"]
                    - 1,
                ),
                "structured_replay": lambda report: report["evaluations"][0][
                    "structured_head_coverage_gate"
                ].__setitem__("training_partition", {"forged": {}}),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    report = copy.deepcopy(original)
                    mutate(report)
                    self._resign(report)
                    report_path.write_text(json.dumps(report), encoding="utf-8")
                    with self._patched_plan_replay():
                        errors = promotion.production_promotion_report_errors(
                            report,
                            metadata=metadata,
                            actual_weight_sha256=weight_sha,
                            checkpoint_directory=checkpoint,
                        )
                        readiness = stage_edit_readiness_report(checkpoint)
                        with self.assertRaises(
                            StageEditProductionRegistryNotReadyError
                        ):
                            _validate_v3_training_report_declarations(
                                report,
                                metadata=metadata,
                                weight_sha256=weight_sha,
                                checkpoint_directory=checkpoint,
                            )
                    self.assertTrue(errors)
                    self.assertFalse(readiness.learned_inference_ready)
                    self.assertEqual(
                        "production_training_report_rejected",
                        readiness.status,
                    )

    def test_non_object_report_roots_fail_closed_in_readiness(self):
        malformed_roots = ([], 7, "primitive", None)
        for root_value in malformed_roots:
            with self.subTest(root_value=root_value), tempfile.TemporaryDirectory() as directory:
                checkpoint, _, _, _, report_path = self._checkpoint_and_report(
                    Path(directory)
                )
                report_path.write_text(
                    json.dumps(root_value), encoding="utf-8"
                )
                readiness = stage_edit_readiness_report(checkpoint)
                self.assertFalse(readiness.learned_inference_ready)
                self.assertEqual(
                    "production_training_report_rejected", readiness.status
                )
                self.assertIn("root must be an object", readiness.reason)

    def test_missing_predeclared_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, metadata, weight_sha, report, _ = self._checkpoint_and_report(
                Path(directory)
            )
            evidence_path = Path(
                report["evaluations"][0]["evidence_manifests"][
                    "feature_manifest"
                ]["path"]
            )
            evidence_path.unlink()
            with self._patched_plan_replay():
                errors = promotion.production_promotion_report_errors(
                    report,
                    metadata=metadata,
                    actual_weight_sha256=weight_sha,
                    checkpoint_directory=checkpoint,
                )
            self.assertTrue(errors)
            self.assertTrue(
                any("evidence" in error or "manifest" in error for error in errors),
                errors,
            )

    def test_posthoc_manifest_and_reference_substitution_cannot_be_resigned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, metadata, weight_sha, original, _ = (
                self._checkpoint_and_report(root)
            )
            substitute = root / "posthoc-evidence.json"
            substitute.write_text("{}", encoding="utf-8")
            report = copy.deepcopy(original)
            report["evaluations"][1]["evidence_manifests"][
                "feature_manifest"
            ] = self._pointer(substitute)
            report["evaluations"][1]["reference_ids"] = [
                f"posthoc-{index}" for index in range(5)
            ]
            self._resign(report)
            with self._patched_plan_replay():
                errors = promotion.production_promotion_report_errors(
                    report,
                    metadata=metadata,
                    actual_weight_sha256=weight_sha,
                    checkpoint_directory=checkpoint,
                )
            self.assertTrue(
                any("differ" in error and "plan" in error for error in errors),
                errors,
            )

    def test_parser_and_evidence_statuses_are_explicitly_non_blind(self):
        parsed = promotion.build_parser().parse_args(
            ["preflight", "--plan", "/tmp/plan.json"]
        )
        self.assertEqual("preflight", parsed.command)
        authority = promotion._promotion_authority_contract()
        self.assertFalse(authority["blind_holdout_quality_claim_authorized"])
        self.assertFalse(authority["human_parity_claim_authorized"])
        self.assertEqual(
            "previously_observed_development_evidence_non_blind",
            authority["xoxz_evidence_status"],
        )
        self.assertEqual(
            "predeclared_release_evidence_non_blind",
            authority["attitude_evidence_status"],
        )

    def test_create_cli_pairs_learned_artifacts_and_missing_half_fails_closed(self):
        parsed = promotion.build_parser().parse_args(
            [
                "create",
                "--final-fit-receipt",
                "/tmp/final-fit.json",
                "--materialization-manifest",
                "IVE-XOXZ=/tmp/materialization.json",
                "--feature-manifest",
                "IVE-XOXZ=/tmp/features.json",
                "--dynamic-crop-manifest",
                "IVE-XOXZ=/tmp/dynamic.json",
                "--checkpoint-directory",
                "/tmp/checkpoint",
                "--evidence-directory",
                "/tmp/evidence",
                "--output-directory",
                "/tmp/output",
                "--source-pair-scorer",
                "/tmp/source-pair-scorer.json",
                "--source-pair-production-report",
                "/tmp/source-pair-production-report.json",
            ]
        )
        self.assertEqual(
            Path("/tmp/source-pair-scorer.json"), parsed.source_pair_scorer
        )
        self.assertEqual(
            Path("/tmp/source-pair-production-report.json"),
            parsed.source_pair_production_report,
        )
        placeholders = {
            song_id: Path(f"/tmp/{song_id}.json")
            for song_id in promotion.EVALUATION_SONG_IDS
        }
        with self.assertRaisesRegex(
            promotion.DynamicCropV3ProductionPromotionError,
            "requires both scorer and production report",
        ):
            promotion.create_dynamic_crop_v3_production_promotion_plan(
                final_fit_receipt_path=Path("/tmp/final-fit.json"),
                materialization_manifests=placeholders,
                feature_manifests=placeholders,
                dynamic_crop_manifests=placeholders,
                checkpoint_directory=Path("/tmp/checkpoint"),
                evidence_directory=Path("/tmp/evidence"),
                output_directory=Path("/tmp/output"),
                source_pair_scorer_path=Path(
                    "/tmp/source-pair-scorer.json"
                ),
            )

    def test_handcrafted_preflight_to_dict_retains_exact_v1_shape(self):
        preflight = promotion.DynamicCropV3ProductionPromotionPreflight(
            plan_path=Path("/absolute/promotion-plan.json"),
            plan_content_sha256="1" * 64,
            source_checkpoint_directory=Path("/absolute/source"),
            destination_checkpoint_directory=Path("/absolute/destination"),
            evidence_directory=Path("/absolute/evidence"),
            production_registry_path=Path("/absolute/registry.json"),
            professional_montage_profile_path=Path("/absolute/profile.json"),
            checkpoint_id="handcrafted-v1",
            checkpoint_weight_sha256="2" * 64,
            frozen_cut_probability_threshold=0.5,
            quality_thresholds={"minimum": 0.75},
            song_evidence={},
            free_bytes=123456,
        )

        self.assertEqual(
            {
                "plan_path": "/absolute/promotion-plan.json",
                "plan_content_sha256": "1" * 64,
                "source_checkpoint_directory": "/absolute/source",
                "destination_checkpoint_directory": "/absolute/destination",
                "evidence_directory": "/absolute/evidence",
                "production_registry_path": "/absolute/registry.json",
                "professional_montage_profile_path": "/absolute/profile.json",
                "checkpoint_id": "handcrafted-v1",
                "checkpoint_weight_sha256": "2" * 64,
                "frozen_cut_probability_threshold": 0.5,
                "quality_thresholds": {"minimum": 0.75},
                "evaluation_songs": {},
                "free_bytes": 123456,
                "optimizer_or_gradient_step_authorized": False,
                "model_threshold_or_policy_reselection_authorized": False,
                "blind_holdout_claim_authorized": False,
                "human_parity_claim_authorized": False,
                "continuity_authority": (
                    promotion.HANDCRAFTED_CONTINUITY_AUTHORITY
                ),
            },
            preflight.to_dict(),
        )

    def test_source_pair_artifact_rejects_symlink_parent_before_registry_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            actual = root / "actual-source-pair"
            actual.mkdir()
            scorer = actual / "stage-edit-source-pair-scorer-v1.json"
            report = actual / "stage-edit-source-pair-production-report-v1.json"
            scorer.write_text("{}", encoding="utf-8")
            report.write_text("{}", encoding="utf-8")
            alias = root / "mutable-alias"
            alias.symlink_to(actual, target_is_directory=True)

            for scorer_path, report_path in (
                (alias / scorer.name, report),
                (scorer, alias / report.name),
            ):
                with self.subTest(
                    scorer_path=scorer_path, report_path=report_path
                ):
                    with (
                        patch.object(
                            promotion.production_registry,
                            "_validated_source_pair_identity",
                        ) as registry_gate,
                        self.assertRaisesRegex(
                            promotion.DynamicCropV3ProductionPromotionError,
                            "must not traverse a symlink",
                        ),
                    ):
                        promotion._source_pair_production_identity(
                            scorer_path, report_path
                        )
                    registry_gate.assert_not_called()

    def test_source_pair_artifact_symlink_loop_uses_public_error_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            loop = root / "source-pair-loop"
            loop.symlink_to(loop)
            report = root / "source-pair-production-report.json"
            report.write_text("{}", encoding="utf-8")

            with (
                patch.object(
                    promotion.production_registry,
                    "_validated_source_pair_identity",
                ) as registry_gate,
                self.assertRaisesRegex(
                    promotion.DynamicCropV3ProductionPromotionError,
                    "missing or unsafe",
                ),
            ):
                promotion._source_pair_production_identity(loop, report)
            registry_gate.assert_not_called()

    def test_source_pair_identity_reuses_registry_gate_and_detects_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer, report, identity = self._source_pair_identity(
                Path(directory)
            )
            with patch.object(
                promotion.production_registry,
                "_validated_source_pair_identity",
                return_value=copy.deepcopy(identity),
            ) as validator:
                self.assertEqual(
                    identity,
                    promotion._source_pair_production_identity(scorer, report),
                )
                self.assertEqual(
                    identity,
                    promotion._validate_source_pair_production_identity(
                        identity
                    ),
                )
            self.assertEqual(2, validator.call_count)
            changed = copy.deepcopy(identity)
            changed["scorer_file_sha256"] = "f" * 64
            with patch.object(
                promotion.production_registry,
                "_validated_source_pair_identity",
                return_value=changed,
            ):
                with self.assertRaisesRegex(
                    promotion.DynamicCropV3ProductionPromotionError,
                    "identity changed",
                ):
                    promotion._validate_source_pair_production_identity(
                        identity
                    )

    def test_receipt_bundle_replay_checks_every_registered_pair_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, identity = self._source_pair_identity(Path(directory))
            bundle_values = {
                "continuity_authority": (
                    promotion.PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                ),
                "source_pair_scorer_path": Path(identity["scorer_path"]),
                "source_pair_production_report_path": Path(
                    identity["production_report_path"]
                ),
                "source_pair_scorer_file_sha256": identity[
                    "scorer_file_sha256"
                ],
                "source_pair_production_report_file_sha256": identity[
                    "production_report_file_sha256"
                ],
                "source_pair_production_report_sha256": identity[
                    "production_report_sha256"
                ],
                "source_pair_scorer_fingerprint": identity[
                    "scorer_fingerprint"
                ],
                "source_pair_checkpoint_sha256": identity[
                    "checkpoint_sha256"
                ],
                "source_pair_corpus_sha256": identity["corpus_sha256"],
            }

            def bundle(values):
                return type("RegisteredBundle", (), values)()

            with patch.object(
                promotion,
                "_source_pair_production_identity",
                return_value=copy.deepcopy(identity),
            ):
                self.assertEqual(
                    identity,
                    promotion._registered_bundle_source_pair_identity(
                        bundle(bundle_values)
                    ),
                )
                tamper_fields = tuple(
                    name
                    for name in bundle_values
                    if name != "continuity_authority"
                )
                for field in tamper_fields:
                    with self.subTest(field=field):
                        tampered = dict(bundle_values)
                        tampered[field] = (
                            Path("/tmp/swapped.json")
                            if field.endswith("_path")
                            else "f" * 64
                        )
                        with self.assertRaisesRegex(
                            promotion.DynamicCropV3ProductionPromotionError,
                            "identity differs",
                        ):
                            promotion._registered_bundle_source_pair_identity(
                                bundle(tampered)
                            )

    def test_learned_report_binds_plan_identity_and_rejects_resigned_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, metadata, weight_sha, report, _ = (
                self._checkpoint_and_report(root)
            )
            _, _, identity = self._source_pair_identity(root)
            original_plan = self._promotion_plan
            learned_plan = copy.deepcopy(original_plan)
            learned_plan["source_pair_production"] = copy.deepcopy(identity)
            learned_plan["authority"] = promotion._promotion_authority_contract(
                learned_source_pair_authority_present=True
            )
            learned_report = copy.deepcopy(report)
            learned_report["source_pair_production"] = copy.deepcopy(identity)
            learned_report["authority"] = copy.deepcopy(
                learned_plan["authority"]
            )
            self._resign(learned_report)
            self._promotion_plan = learned_plan
            try:
                with (
                    patch.object(
                        promotion.production_registry,
                        "_validated_source_pair_identity",
                        return_value=copy.deepcopy(identity),
                    ),
                    self._patched_plan_replay(),
                ):
                    self.assertEqual(
                        (),
                        promotion.production_promotion_report_errors(
                            learned_report,
                            metadata=metadata,
                            actual_weight_sha256=weight_sha,
                            checkpoint_directory=checkpoint,
                        ),
                    )
                    tampered = copy.deepcopy(learned_report)
                    tampered["source_pair_production"][
                        "cross_validation_sha256"
                    ] = "e" * 64
                    self._resign(tampered)
                    errors = promotion.production_promotion_report_errors(
                        tampered,
                        metadata=metadata,
                        actual_weight_sha256=weight_sha,
                        checkpoint_directory=checkpoint,
                    )
                    self.assertTrue(
                        any("source-pair" in error for error in errors), errors
                    )

                    mismatched = copy.deepcopy(learned_report)
                    mismatched.pop("source_pair_production")
                    self._resign(mismatched)
                    mismatch_errors = (
                        promotion.production_promotion_report_errors(
                            mismatched,
                            metadata=metadata,
                            actual_weight_sha256=weight_sha,
                            checkpoint_directory=checkpoint,
                        )
                    )
                    self.assertTrue(mismatch_errors)
            finally:
                self._promotion_plan = original_plan

    def test_launch_copies_exact_bytes_and_never_runs_optimizer_or_reselection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, metadata, weight_sha, source_report, _ = (
                self._checkpoint_and_report(root)
            )
            source = Path(
                source_report["lineage"]["source_checkpoint"]["directory"]
            )
            destination = (root / "installed").resolve()
            evidence = root / "evidence"
            evidence.mkdir()
            registry = root / "registry.json"
            receipt = evidence / "receipt.json"
            receipt.write_text("{}", encoding="utf-8")
            plan_path = root / "plan.json"
            plan_path.write_text("{}", encoding="utf-8")
            preflight = promotion.DynamicCropV3ProductionPromotionPreflight(
                plan_path=plan_path,
                plan_content_sha256="1" * 64,
                source_checkpoint_directory=source,
                destination_checkpoint_directory=destination,
                evidence_directory=evidence,
                production_registry_path=registry,
                professional_montage_profile_path=root / "profile.json",
                checkpoint_id=metadata.checkpoint_id,
                checkpoint_weight_sha256=weight_sha,
                frozen_cut_probability_threshold=0.5,
                quality_thresholds=asdict(ProductionQualityThresholds()),
                song_evidence={},
                free_bytes=10 * 1024**3,
            )
            plan = {
                "plan_content_sha256": "1" * 64,
                "lineage": copy.deepcopy(source_report["lineage"]),
                "evaluation_protocol": copy.deepcopy(
                    source_report["evaluation_protocol"]
                ),
                "source_provenance": {"head_commit_oid": "a" * 40},
                "destination_checkpoint_directory": str(destination),
                "evidence_directory": str(evidence.resolve()),
                "production_registry_path": str(registry.resolve()),
                "professional_montage_profile": {
                    "path": str((root / "profile.json").resolve())
                },
                "authority": promotion._promotion_authority_contract(),
            }
            ready = StageEditReadinessReport(
                architecture_available=True,
                learned_inference_ready=True,
                status="production_checkpoint_ready",
                reason="test promotion",
                mlx_version="test",
                checkpoint_path=str(destination.resolve()),
                training_steps=metadata.training_steps,
                checkpoint_format="dynamic_crop_v3",
                dynamic_crop_version=3,
                checkpoint_metadata_filename=(
                    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
                ),
            )
            with (
                patch.object(
                    promotion,
                    "preflight_dynamic_crop_v3_production_promotion",
                    return_value=preflight,
                ),
                patch.object(promotion, "_strict_plan", return_value=plan),
                patch.object(
                    promotion,
                    "_evaluate_exact_copied_checkpoint",
                    return_value=copy.deepcopy(source_report["evaluations"]),
                ) as evaluate_mock,
                patch.object(
                    promotion,
                    "production_promotion_report_errors",
                    return_value=(),
                ),
                patch.object(
                    promotion, "stage_edit_readiness_report", return_value=ready
                ),
                patch.object(
                    promotion,
                    "register_stage_edit_production_bundle",
                    return_value=registry,
                ) as register_mock,
                patch.object(
                    promotion,
                    "_record_promotion_receipt",
                    return_value=(receipt, False),
                ),
            ):
                result = promotion.launch_dynamic_crop_v3_production_promotion(
                    plan_path
                )
            self.assertEqual(destination, result.checkpoint_directory)
            self.assertEqual(
                (source / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME).read_bytes(),
                (
                    destination / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
                ).read_bytes(),
            )
            self.assertEqual(
                (source / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME).read_bytes(),
                (
                    destination / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
                ).read_bytes(),
            )
            self.assertEqual(0, result.report["evaluation_protocol"]["optimizer_steps"])
            self.assertFalse(
                result.report["evaluation_protocol"][
                    "checkpoint_or_model_reselection_performed"
                ]
            )
            evaluate_mock.assert_called_once()
            register_mock.assert_called_once()
            register_kwargs = register_mock.call_args.kwargs
            self.assertEqual(3, register_kwargs["dynamic_crop_version"])
            self.assertEqual(
                promotion.HANDCRAFTED_CONTINUITY_AUTHORITY,
                register_kwargs["continuity_authority"],
            )
            self.assertIsNone(register_kwargs["source_pair_scorer_path"])
            self.assertIsNone(
                register_kwargs["source_pair_production_report_path"]
            )

    def test_shared_registry_call_passes_exact_learned_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, metadata, weight_sha, _, _ = (
                self._checkpoint_and_report(root)
            )
            scorer, pair_report, identity = self._source_pair_identity(root)
            registry = root / "learned-registry.json"
            preflight = promotion.DynamicCropV3ProductionPromotionPreflight(
                plan_path=root / "learned-plan.json",
                plan_content_sha256="1" * 64,
                source_checkpoint_directory=checkpoint,
                destination_checkpoint_directory=checkpoint,
                evidence_directory=root,
                production_registry_path=registry,
                professional_montage_profile_path=root / "profile.json",
                checkpoint_id=metadata.checkpoint_id,
                checkpoint_weight_sha256=weight_sha,
                frozen_cut_probability_threshold=0.5,
                quality_thresholds=asdict(ProductionQualityThresholds()),
                song_evidence={},
                free_bytes=10 * 1024**3,
                continuity_authority=(
                    promotion.PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                ),
                source_pair_scorer_path=scorer,
                source_pair_production_report_path=pair_report,
                source_pair_production_identity=identity,
            )
            with patch.object(
                promotion,
                "register_stage_edit_production_bundle",
                return_value=registry,
            ) as register_mock:
                self.assertEqual(
                    registry, promotion._register_promoted_bundle(preflight)
                )
            register_kwargs = register_mock.call_args.kwargs
            self.assertEqual(
                promotion.PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
                register_kwargs["continuity_authority"],
            )
            self.assertEqual(scorer, register_kwargs["source_pair_scorer_path"])
            self.assertEqual(
                pair_report,
                register_kwargs["source_pair_production_report_path"],
            )

    def test_failed_quality_gate_removes_temporary_copy_and_never_registers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, metadata, weight_sha, source_report, _ = (
                self._checkpoint_and_report(root)
            )
            source = Path(
                source_report["lineage"]["source_checkpoint"]["directory"]
            )
            destination = (root / "failed-install").resolve()
            evidence = root / "evidence"
            evidence.mkdir()
            registry = root / "registry.json"
            plan_path = root / "plan.json"
            plan_path.write_text("{}", encoding="utf-8")
            preflight = promotion.DynamicCropV3ProductionPromotionPreflight(
                plan_path=plan_path,
                plan_content_sha256="1" * 64,
                source_checkpoint_directory=source,
                destination_checkpoint_directory=destination,
                evidence_directory=evidence,
                production_registry_path=registry,
                professional_montage_profile_path=root / "profile.json",
                checkpoint_id=metadata.checkpoint_id,
                checkpoint_weight_sha256=weight_sha,
                frozen_cut_probability_threshold=0.5,
                quality_thresholds=asdict(ProductionQualityThresholds()),
                song_evidence={},
                free_bytes=10 * 1024**3,
            )
            plan = {
                "plan_content_sha256": "1" * 64,
                "lineage": copy.deepcopy(source_report["lineage"]),
                "evaluation_protocol": copy.deepcopy(
                    source_report["evaluation_protocol"]
                ),
                "source_provenance": {"head_commit_oid": "a" * 40},
                "destination_checkpoint_directory": str(destination),
                "evidence_directory": str(evidence.resolve()),
                "production_registry_path": str(registry.resolve()),
                "professional_montage_profile": {
                    "path": str((root / "profile.json").resolve())
                },
                "authority": promotion._promotion_authority_contract(),
            }
            failed_evaluations = copy.deepcopy(source_report["evaluations"])
            failed_evaluations[0]["quality_gate"] = {
                "passed": False,
                "reasons": ["forced quality failure"],
            }
            with (
                patch.object(
                    promotion,
                    "preflight_dynamic_crop_v3_production_promotion",
                    return_value=preflight,
                ),
                patch.object(promotion, "_strict_plan", return_value=plan),
                patch.object(
                    promotion,
                    "_evaluate_exact_copied_checkpoint",
                    return_value=failed_evaluations,
                ),
                patch.object(
                    promotion, "register_stage_edit_production_bundle"
                ) as register_mock,
            ):
                with self.assertRaises(
                    promotion.DynamicCropV3ProductionPromotionError
                ):
                    promotion.launch_dynamic_crop_v3_production_promotion(
                        plan_path
                    )
            register_mock.assert_not_called()
            self.assertFalse(destination.exists())
            self.assertEqual(
                [], list(root.glob(".failed-install.promotion-tmp-*"))
            )

    def test_post_install_failures_roll_back_owned_checkpoint_and_receipt(self):
        for failure_stage in (
            "readiness",
            "registry",
            "destination_swap",
            "receipt_swap",
        ):
            with self.subTest(failure_stage=failure_stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, metadata, weight_sha, source_report, _ = (
                    self._checkpoint_and_report(root)
                )
                source = Path(
                    source_report["lineage"]["source_checkpoint"]["directory"]
                )
                destination = (root / f"{failure_stage}-install").resolve()
                evidence = root / "evidence"
                evidence.mkdir()
                registry = root / "registry.json"
                receipt = evidence / "owned-receipt.json"
                plan_path = root / "plan.json"
                plan_path.write_text("{}", encoding="utf-8")
                preflight = promotion.DynamicCropV3ProductionPromotionPreflight(
                    plan_path=plan_path,
                    plan_content_sha256="1" * 64,
                    source_checkpoint_directory=source,
                    destination_checkpoint_directory=destination,
                    evidence_directory=evidence,
                    production_registry_path=registry,
                    professional_montage_profile_path=root / "profile.json",
                    checkpoint_id=metadata.checkpoint_id,
                    checkpoint_weight_sha256=weight_sha,
                    frozen_cut_probability_threshold=0.5,
                    quality_thresholds=asdict(ProductionQualityThresholds()),
                    song_evidence={},
                    free_bytes=10 * 1024**3,
                )
                plan = {
                    "plan_content_sha256": "1" * 64,
                    "lineage": copy.deepcopy(source_report["lineage"]),
                    "evaluation_protocol": copy.deepcopy(
                        source_report["evaluation_protocol"]
                    ),
                    "source_provenance": {"head_commit_oid": "a" * 40},
                    "destination_checkpoint_directory": str(destination),
                    "evidence_directory": str(evidence.resolve()),
                    "production_registry_path": str(registry.resolve()),
                    "professional_montage_profile": {
                        "path": str((root / "profile.json").resolve())
                    },
                    "authority": promotion._promotion_authority_contract(),
                }
                ready = StageEditReadinessReport(
                    architecture_available=True,
                    learned_inference_ready=(failure_stage != "readiness"),
                    status=(
                        "production_checkpoint_ready"
                        if failure_stage != "readiness"
                        else "production_training_report_rejected"
                    ),
                    reason="forced post-install failure",
                    mlx_version="test",
                    checkpoint_path=str(destination),
                    training_steps=metadata.training_steps,
                    checkpoint_format="dynamic_crop_v3",
                    dynamic_crop_version=3,
                    checkpoint_metadata_filename=(
                        DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
                    ),
                )

                def stage_receipt(**_kwargs):
                    receipt.write_text("{}", encoding="utf-8")
                    return receipt, True

                def fail_registry(*_args, **_kwargs):
                    if failure_stage == "destination_swap":
                        destination.rename(root / "preserved-owned-checkpoint")
                        destination.mkdir()
                        (destination / "foreign.txt").write_text(
                            "foreign replacement", encoding="utf-8"
                        )
                    elif failure_stage == "receipt_swap":
                        receipt.unlink()
                        receipt.write_text(
                            "foreign replacement", encoding="utf-8"
                        )
                    raise RuntimeError("forced registry failure")

                with (
                    patch.object(
                        promotion,
                        "preflight_dynamic_crop_v3_production_promotion",
                        return_value=preflight,
                    ),
                    patch.object(promotion, "_strict_plan", return_value=plan),
                    patch.object(
                        promotion,
                        "_evaluate_exact_copied_checkpoint",
                        return_value=copy.deepcopy(source_report["evaluations"]),
                    ),
                    patch.object(
                        promotion,
                        "production_promotion_report_errors",
                        return_value=(),
                    ),
                    patch.object(
                        promotion, "stage_edit_readiness_report", return_value=ready
                    ),
                    patch.object(
                        promotion,
                        "_record_promotion_receipt",
                        side_effect=stage_receipt,
                    ) as receipt_mock,
                    patch.object(
                        promotion,
                        "register_stage_edit_production_bundle",
                        side_effect=fail_registry,
                    ) as register_mock,
                ):
                    with self.assertRaises(
                        (promotion.DynamicCropV3ProductionPromotionError, RuntimeError)
                    ) as raised:
                        promotion.launch_dynamic_crop_v3_production_promotion(
                            plan_path
                        )
                self.assertFalse(registry.exists())
                if failure_stage == "readiness":
                    self.assertFalse(destination.exists())
                    self.assertFalse(receipt.exists())
                    receipt_mock.assert_not_called()
                    register_mock.assert_not_called()
                elif failure_stage == "registry":
                    self.assertFalse(destination.exists())
                    self.assertFalse(receipt.exists())
                    receipt_mock.assert_called_once()
                    register_mock.assert_called_once()
                else:
                    self.assertIsInstance(
                        raised.exception,
                        promotion.DynamicCropV3ProductionPromotionError,
                    )
                    self.assertIn("unsafe", str(raised.exception))
                    self.assertTrue(destination.exists())
                    self.assertTrue(receipt.exists())
                    receipt_mock.assert_called_once()
                    register_mock.assert_called_once()
                    if failure_stage == "destination_swap":
                        self.assertEqual(
                            "foreign replacement",
                            (destination / "foreign.txt").read_text(
                                encoding="utf-8"
                            ),
                        )
                        self.assertTrue(
                            (root / "preserved-owned-checkpoint").exists()
                        )
                    else:
                        self.assertEqual(
                            "foreign replacement",
                            receipt.read_text(encoding="utf-8"),
                        )

    def test_exact_completed_same_plan_is_idempotently_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, metadata, weight_sha, report, report_path = (
                self._checkpoint_and_report(root)
            )
            registry = root / "registry.json"
            registry.write_text("{}", encoding="utf-8")
            receipt = root / "promotion-evidence" / "receipt.json"
            receipt.parent.mkdir()
            receipt.write_text("{}", encoding="utf-8")
            preflight = promotion.DynamicCropV3ProductionPromotionPreflight(
                plan_path=Path(report["promotion_plan"]["path"]),
                plan_content_sha256=report["promotion_plan_content_sha256"],
                source_checkpoint_directory=Path(
                    report["lineage"]["source_checkpoint"]["directory"]
                ),
                destination_checkpoint_directory=checkpoint,
                evidence_directory=receipt.parent,
                production_registry_path=registry,
                professional_montage_profile_path=root / "profile.json",
                checkpoint_id=metadata.checkpoint_id,
                checkpoint_weight_sha256=weight_sha,
                frozen_cut_probability_threshold=0.5,
                quality_thresholds=asdict(ProductionQualityThresholds()),
                song_evidence={},
                free_bytes=10 * 1024**3,
            )
            ready = StageEditReadinessReport(
                architecture_available=True,
                learned_inference_ready=True,
                status="production_checkpoint_ready",
                reason="exact completed promotion",
                mlx_version="test",
                checkpoint_path=str(checkpoint),
                training_steps=metadata.training_steps,
                checkpoint_format="dynamic_crop_v3",
                dynamic_crop_version=3,
                checkpoint_metadata_filename=(
                    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
                ),
            )
            with (
                patch.object(
                    promotion, "_strict_plan", return_value=self._promotion_plan
                ),
                patch.object(
                    promotion,
                    "_preflight_from_existing_plan",
                    return_value=preflight,
                ),
                patch.object(
                    promotion,
                    "load_dynamic_crop_v3_production_promotion_report",
                    return_value=report,
                ),
                patch.object(
                    promotion, "stage_edit_readiness_report", return_value=ready
                ),
                patch.object(
                    promotion, "_expected_receipt_path", return_value=receipt
                ),
                patch.object(
                    promotion,
                    "_load_dynamic_crop_v3_production_promotion_receipt",
                    return_value={},
                ) as receipt_loader,
                patch.object(
                    promotion, "register_stage_edit_production_bundle"
                ) as register_mock,
            ):
                first = promotion.launch_dynamic_crop_v3_production_promotion(
                    preflight.plan_path
                )
                second = promotion.launch_dynamic_crop_v3_production_promotion(
                    preflight.plan_path
                )
            self.assertEqual(checkpoint, first.checkpoint_directory)
            self.assertEqual(first, second)
            self.assertEqual(2, receipt_loader.call_count)
            register_mock.assert_not_called()

    def test_learned_existing_plan_resume_replays_receipt_and_registers_same_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, metadata, weight_sha, report, report_path = (
                self._checkpoint_and_report(root)
            )
            scorer, pair_report, identity = self._source_pair_identity(root)
            registry = root / "learned-resume-registry.json"
            receipt = root / "promotion-evidence" / "learned-receipt.json"
            receipt.parent.mkdir()
            receipt.write_text("{}", encoding="utf-8")
            base_preflight = promotion.DynamicCropV3ProductionPromotionPreflight(
                plan_path=Path(report["promotion_plan"]["path"]),
                plan_content_sha256=report[
                    "promotion_plan_content_sha256"
                ],
                source_checkpoint_directory=Path(
                    report["lineage"]["source_checkpoint"]["directory"]
                ),
                destination_checkpoint_directory=checkpoint,
                evidence_directory=receipt.parent,
                production_registry_path=registry,
                professional_montage_profile_path=root / "profile.json",
                checkpoint_id=metadata.checkpoint_id,
                checkpoint_weight_sha256=weight_sha,
                frozen_cut_probability_threshold=0.5,
                quality_thresholds=asdict(ProductionQualityThresholds()),
                song_evidence={},
                free_bytes=10 * 1024**3,
            )
            base_plan = copy.deepcopy(self._promotion_plan)
            base_plan["production_registry_path"] = str(registry.resolve())
            base_plan["destination_checkpoint_directory"] = str(checkpoint)
            base_plan["evidence_directory"] = str(receipt.parent.resolve())
            plan, preflight = self._learned_plan_and_preflight(
                plan=base_plan,
                preflight=base_preflight,
                identity=identity,
            )
            ready = StageEditReadinessReport(
                architecture_available=True,
                learned_inference_ready=True,
                status="production_checkpoint_ready",
                reason="learned resume",
                mlx_version="test",
                checkpoint_path=str(checkpoint),
                training_steps=metadata.training_steps,
                checkpoint_format="dynamic_crop_v3",
                dynamic_crop_version=3,
                checkpoint_metadata_filename=(
                    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
                ),
            )
            with (
                patch.object(promotion, "_strict_plan", return_value=plan),
                patch.object(
                    promotion,
                    "_preflight_from_existing_plan",
                    return_value=preflight,
                ),
                patch.object(
                    promotion.production_registry,
                    "_validated_source_pair_identity",
                    return_value=copy.deepcopy(identity),
                ),
                patch.object(
                    promotion,
                    "load_dynamic_crop_v3_production_promotion_report",
                    return_value=report,
                ),
                patch.object(
                    promotion, "stage_edit_readiness_report", return_value=ready
                ),
                patch.object(
                    promotion, "_expected_receipt_path", return_value=receipt
                ),
                patch.object(
                    promotion,
                    "_load_dynamic_crop_v3_production_promotion_receipt",
                    return_value={},
                ) as receipt_loader,
                patch.object(
                    promotion,
                    "register_stage_edit_production_bundle",
                    return_value=registry,
                ) as register_mock,
            ):
                result = promotion._resume_or_return_existing_promotion(
                    preflight.plan_path
                )
            self.assertIsNotNone(result)
            receipt_loader.assert_called_once_with(
                receipt, require_registry=False
            )
            register_kwargs = register_mock.call_args.kwargs
            self.assertEqual(
                promotion.PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
                register_kwargs["continuity_authority"],
            )
            self.assertEqual(scorer, register_kwargs["source_pair_scorer_path"])
            self.assertEqual(
                pair_report,
                register_kwargs["source_pair_production_report_path"],
            )

    def test_real_source_pair_artifacts_survive_receipt_registry_and_resume_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            checkpoint, metadata, weight_sha, report, report_path = (
                self._checkpoint_and_report(root)
            )
            scorer, pair_report, identity = self._real_source_pair_identity(root)
            profile = _montage_profile(root)
            profile_identity = load_professional_montage_policy(
                profile
            ).identity.to_dict()
            registry = root / "learned-production-registry.json"
            evidence = root / "promotion-evidence"
            evidence.mkdir(exist_ok=True)
            plan_path = Path(report["promotion_plan"]["path"])

            base_plan = copy.deepcopy(self._promotion_plan)
            base_plan["destination_checkpoint_directory"] = str(checkpoint)
            base_plan["evidence_directory"] = str(evidence)
            base_plan["production_registry_path"] = str(registry)
            base_plan["professional_montage_profile"] = profile_identity
            base_preflight = promotion.DynamicCropV3ProductionPromotionPreflight(
                plan_path=plan_path,
                plan_content_sha256=report[
                    "promotion_plan_content_sha256"
                ],
                source_checkpoint_directory=Path(
                    report["lineage"]["source_checkpoint"]["directory"]
                ),
                destination_checkpoint_directory=checkpoint,
                evidence_directory=evidence,
                production_registry_path=registry,
                professional_montage_profile_path=profile,
                checkpoint_id=metadata.checkpoint_id,
                checkpoint_weight_sha256=weight_sha,
                frozen_cut_probability_threshold=0.5,
                quality_thresholds=asdict(ProductionQualityThresholds()),
                song_evidence={},
                free_bytes=10 * 1024**3,
            )
            plan, preflight = self._learned_plan_and_preflight(
                plan=base_plan,
                preflight=base_preflight,
                identity=identity,
            )
            plan_path.write_text(
                json.dumps(plan, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            report["promotion_plan"] = self._pointer(plan_path)
            report["source_pair_production"] = copy.deepcopy(identity)
            report["authority"] = copy.deepcopy(plan["authority"])
            self._resign(report)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            ready = StageEditReadinessReport(
                architecture_available=True,
                learned_inference_ready=True,
                status="production_checkpoint_ready",
                reason="real learned source-pair promotion integration",
                mlx_version="test",
                checkpoint_path=str(checkpoint),
                training_steps=metadata.training_steps,
                checkpoint_format="dynamic_crop_v3",
                dynamic_crop_version=3,
                checkpoint_metadata_filename=(
                    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
                ),
            )

            original_plan = self._promotion_plan
            self._promotion_plan = plan
            try:
                with (
                    self._patched_plan_replay(),
                    patch(
                        "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher."
                        "load_dynamic_crop_v3_release_freeze",
                        return_value={
                            "freeze_content_sha256": "2" * 64,
                            "source_provenance": {"head_commit_oid": "a" * 40},
                            "candidate_id": "production-balanced-test",
                        },
                    ),
                    patch.object(
                        promotion,
                        "stage_edit_readiness_report",
                        return_value=ready,
                    ),
                    patch.object(
                        promotion.production_registry,
                        "stage_edit_readiness_report",
                        return_value=ready,
                    ),
                ):
                    self.assertEqual(
                        identity,
                        promotion._validate_source_pair_production_identity(
                            identity
                        ),
                    )
                    self.assertEqual(
                        report,
                        promotion.load_dynamic_crop_v3_production_promotion_report(
                            report_path
                        ),
                    )
                    receipt, created = promotion._record_promotion_receipt(
                        preflight=preflight,
                        report_path=report_path,
                        report=report,
                    )
                    self.assertTrue(created)
                    staged_receipt = (
                        promotion._load_dynamic_crop_v3_production_promotion_receipt(
                            receipt, require_registry=False
                        )
                    )
                    self.assertEqual(identity, staged_receipt["source_pair_production"])

                    result = promotion._resume_or_return_existing_promotion(
                        plan_path
                    )
                    self.assertIsNotNone(result)
                    assert result is not None
                    self.assertEqual(receipt, result.receipt_path)
                    self.assertEqual(registry, result.registry_path)
                    bundle = promotion.load_registered_stage_edit_production_bundle(
                        registry
                    )
                    self.assertEqual(
                        identity,
                        promotion._registered_bundle_source_pair_identity(bundle),
                    )
                    activated_receipt = (
                        promotion.load_dynamic_crop_v3_production_promotion_receipt(
                            receipt
                        )
                    )
                    self.assertEqual(
                        identity,
                        activated_receipt["source_pair_production"],
                    )

                    resumed = promotion._resume_or_return_existing_promotion(
                        plan_path
                    )
                    self.assertEqual(result, resumed)
            finally:
                self._promotion_plan = original_plan

    @unittest.skipIf(mx is None, "MLX is not installed")
    def test_mlx_checkpoint_survives_byte_identical_promotion_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            copied = root / "copied"
            copied.mkdir()
            config = smoke_stage_edit_config()
            model = StageEditTransformerDynamicCropV3(config)
            mx.eval(model.parameters())
            metadata = new_stage_edit_dynamic_crop_checkpoint_metadata_v3(
                "mlx-byte-copy",
                DEFAULT_STAGE_EDIT_SCHEMA,
                config,
                training_steps=1,
                weights_trained=True,
                training_data_id="unit-test",
            )
            saved = save_stage_edit_dynamic_crop_checkpoint_v3(
                source, model, metadata
            )
            metadata_sha = promotion._copy_file_byte_identical(
                source / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
                copied / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
                "test metadata",
            )
            weight_sha = promotion._copy_file_byte_identical(
                source / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
                copied / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
                "test weights",
            )
            restored_model, restored = load_stage_edit_dynamic_crop_checkpoint_v3(
                copied,
                expected_schema=DEFAULT_STAGE_EDIT_SCHEMA,
            )
            mx.eval(restored_model.parameters())
            self.assertEqual(saved, restored)
            self.assertEqual(saved.weight_sha256, weight_sha)
            self.assertEqual(
                metadata_sha,
                _sha(copied / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME),
            )


if __name__ == "__main__":
    unittest.main()
