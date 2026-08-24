from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from crosscut.stage_edit_dynamic_crop_capacity_probe import (
    BANG_BANG,
    BLACKHOLE,
    REBEL_HEART,
    FOLD_EVIDENCE_PREFIX,
    SMOKE_EVIDENCE_ANALYSIS_BASIS,
    SMOKE_EVIDENCE_PREFIX,
    DeviceMemoryBudget,
    DynamicCropCapacityCandidate,
    DynamicCropCapacityProbeError,
    FoldEvaluation,
    SelectionPolicy,
    build_dynamic_crop_capacity_selection,
    default_dynamic_crop_v2_probe_candidates,
    load_dynamic_crop_capacity_selection,
    write_dynamic_crop_capacity_selection,
    write_fold_run_evidence,
    write_selection_dataset_manifest,
)
from crosscut.stage_edit_dynamic_crop_head_v2 import (
    DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME,
    DYNAMIC_CROP_V2_LOSS_CONTRACT,
    DYNAMIC_CROP_V2_MODEL_CONTRACT,
    DYNAMIC_CROP_V2_SCHEMA_VERSION,
    DYNAMIC_CROP_V2_TARGET_CONTRACT,
    DynamicCropHeadV2Config,
    DynamicCropV2LossContract,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v2,
)
from crosscut.stage_edit_production_training import (
    STAGE_EDIT_SELECTION_FREEZE_FILENAME,
    ProductionQualityThresholds,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    PRODUCTION_CHECKPOINT_SCOPE,
    PRODUCTION_TRAINING_REPORT_FILENAME,
    PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
    PRODUCTION_TRAINING_RUN_KIND,
    StageEditLossContract,
    StageEditTransformerConfig,
)
from crosscut.stage_edit_transition_authority import (
    RENDERER_OWNED_TRANSITION_AUTHORITY,
    StageEditTransitionAuthority,
)
from crosscut.stage_edit_dynamic_crop_capacity_probe import PROBE_TRAINING_SCHEDULE


SONGS = (BANG_BANG, BLACKHOLE, REBEL_HEART)
REFERENCES = {
    song: tuple(f"{song.lower()}-ref-{index}" for index in range(5))
    for song in SONGS
}
BUDGET = DeviceMemoryBudget(
    device_name="Apple M4 Pro",
    physical_memory_bytes=25_769_803_776,
)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_addressed(root: Path, prefix: str, unsigned: dict[str, object]) -> Path:
    payload = {**unsigned, "artifact_content_sha256": None}
    digest = _canonical_hash(payload)
    payload["artifact_content_sha256"] = digest
    path = root / f"{prefix}{digest}.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def _candidate(name: str, dimension: int) -> DynamicCropCapacityCandidate:
    base = StageEditLossContract(
        transition_weight=0.0,
        transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
    )
    return DynamicCropCapacityCandidate(
        candidate_name=name,
        model_config=StageEditTransformerConfig.from_schema(
            DEFAULT_STAGE_EDIT_SCHEMA,
            model_dimension=dimension,
            num_heads=4,
            num_layers=2,
            feed_forward_dimension=dimension * 3,
            max_sequence_length=32,
            max_sources=8,
            dropout=0.0,
            transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
        ),
        dynamic_crop_config=DynamicCropHeadV2Config(),
        base_loss_contract=base,
        dynamic_crop_loss_contract=DynamicCropV2LossContract(base=base),
    )


def _dataset(root: Path, references=REFERENCES) -> Path:
    manifests = {}
    for song in SONGS:
        path = root / f"{song}-source-manifest.json"
        path.write_text(
            json.dumps(
                {
                    "song_id": song,
                    "input_contract": {
                        "style_id": {
                            BANG_BANG: "bang-bang-human-reference-10source-v2",
                            BLACKHOLE: "blackhole-human-reference-10source-v2",
                            REBEL_HEART: "rebel-heart-human-reference-10source-v3-current-code",
                        }[song],
                        "expected_reference_count": 5,
                        "inputs": [
                            {"reference_id": reference}
                            for reference in references[song]
                        ],
                    },
                    "statistics": {"reference_count": 5},
                    "outputs": [
                        {"reference_id": reference}
                        for reference in references[song]
                    ],
                }
            ),
            encoding="utf-8",
        )
        manifests[song] = path.resolve()
    return write_selection_dataset_manifest(
        root,
        manifest_paths_by_song=manifests,
        reference_ids_by_song=references,
    )


def _smoke(
    root: Path,
    candidate: DynamicCropCapacityCandidate,
    *,
    passed: bool = True,
) -> Path:
    return _write_addressed(
        root,
        SMOKE_EVIDENCE_PREFIX,
        {
            "schema_version": 1,
            "analysis_basis": SMOKE_EVIDENCE_ANALYSIS_BASIS,
            "candidate_id": candidate.candidate_id,
            "model_config_sha256": _canonical_hash(candidate.model_config.to_dict()),
            "device_name": BUDGET.device_name,
            "physical_memory_bytes": BUDGET.physical_memory_bytes,
            "max_recommended_working_set_size": (
                BUDGET.max_recommended_working_set_size
            ),
            "max_buffer_length": BUDGET.max_buffer_length,
            "architecture": BUDGET.architecture,
            "mlx_version": BUDGET.mlx_version,
            "os_reserve_bytes": BUDGET.os_reserve_bytes,
            "batch_size": 2,
            "dtype_bytes": 4,
            "sequence_length": candidate.model_config.max_sequence_length,
            "max_sources": candidate.model_config.max_sources,
            "model_allocated": passed,
            "forward_completed": passed,
            "backward_completed": passed,
            "optimizer_step_completed": passed,
            "optimizer_state_materialized": passed,
            "measured_peak_bytes": 1024**3 if passed else None,
            "passed": passed,
            "failure_kind": None if passed else "out_of_memory",
            "process_exit_code": 0 if passed else 1,
            "termination_signal": None,
            "runner_log_sha256": "d" * 64,
        },
    )


def _fold(
    root: Path,
    candidate: DynamicCropCapacityCandidate,
    heldout_song: str,
    dataset_path: Path,
    metric_value: float,
    *,
    report_schema_version: int = PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
) -> Path:
    fold_root = root / candidate.candidate_name / heldout_song
    fold_root.mkdir(parents=True)
    weight_path = fold_root / DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME
    weight_path.write_bytes(f"weights:{candidate.candidate_id}:{heldout_song}".encode())
    weight_sha = _hash_file(weight_path)
    checkpoint_id = f"checkpoint-{candidate.candidate_name}-{heldout_song}"
    metadata = new_stage_edit_dynamic_crop_checkpoint_metadata_v2(
        checkpoint_id,
        DEFAULT_STAGE_EDIT_SCHEMA,
        candidate.model_config,
        training_steps=10,
        weights_trained=True,
        dynamic_crop_config=DynamicCropHeadV2Config(),
    )
    metadata = replace(
        metadata,
        weight_filename=weight_path.name,
        weight_sha256=weight_sha,
    )
    metadata_path = fold_root / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata.to_dict()), encoding="utf-8")

    training_songs = tuple(sorted(set(SONGS) - {heldout_song}))
    internal_validation_refs = tuple(REFERENCES[song][0] for song in training_songs)
    training_refs = tuple(
        sorted(
            ref
            for song in training_songs
            for ref in REFERENCES[song]
            if ref not in internal_validation_refs
        )
    )
    base_effective = asdict(candidate.base_loss_contract)
    base_effective["cut_positive_weight"] = 2.0
    dynamic_effective = asdict(candidate.dynamic_crop_loss_contract)
    dynamic_effective["base"]["cut_positive_weight"] = 2.0
    freeze = {
        "schema_version": 1,
        "contract": "stage_edit_model_config_checkpoint_and_threshold_freeze_v1",
        "model_config": candidate.model_config.to_dict(),
        "model_config_fingerprint_sha256": _canonical_hash(
            candidate.model_config.to_dict()
        ),
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_sha256": weight_sha,
        "training_song_ids": list(training_songs),
        "training_reference_ids": list(training_refs),
        "validation_reference_ids": list(internal_validation_refs),
        "held_out_song_ids_reserved_for_one_forward": [heldout_song],
        "heldout_metrics_present": False,
        "heldout_used_for_model_or_threshold_selection": False,
        "style_or_attitude_configuration_mutable_after_freeze": False,
    }
    freeze_path = fold_root / STAGE_EDIT_SELECTION_FREEZE_FILENAME
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    schedule = {
        "contract": PROBE_TRAINING_SCHEDULE["schedule_contract"],
        "cut_pretraining_steps": PROBE_TRAINING_SCHEDULE["cut_pretraining_steps"],
        "cut_pretraining_peak_learning_rate": PROBE_TRAINING_SCHEDULE[
            "cut_pretraining_peak_learning_rate"
        ],
        "cut_pretraining_window_length": PROBE_TRAINING_SCHEDULE[
            "cut_pretraining_window_length"
        ],
        "cut_pretraining_warmup_steps": PROBE_TRAINING_SCHEDULE[
            "cut_pretraining_warmup_steps"
        ],
        "finetune_warmup_steps": PROBE_TRAINING_SCHEDULE[
            "finetune_warmup_steps"
        ],
        "minimum_learning_rate_ratio": PROBE_TRAINING_SCHEDULE[
            "minimum_learning_rate_ratio"
        ],
    }
    report = {
        "schema_version": report_schema_version,
        "run_kind": PRODUCTION_TRAINING_RUN_KIND,
        "checkpoint_scope": PRODUCTION_CHECKPOINT_SCOPE,
        "production_ready": False,
        "production_readiness_reason": "activation_not_explicitly_authorized",
        "transition_authority_contract": StageEditTransitionAuthority(
            mode=RENDERER_OWNED_TRANSITION_AUTHORITY
        ).to_dict(),
        "transition_authority_contract_sha256": StageEditTransitionAuthority(
            mode=RENDERER_OWNED_TRANSITION_AUTHORITY
        ).fingerprint,
        "model": {
            "config": candidate.model_config.to_dict(),
            "dynamic_crop_v2_enabled": True,
        },
        "split": {
            "training_song_ids": list(training_songs),
            "held_out_song_ids": [heldout_song],
            "training_reference_ids": list(training_refs),
            "validation_reference_ids": list(internal_validation_refs),
            "held_out_reference_ids": list(REFERENCES[heldout_song]),
            "held_out_used_for_model_selection": False,
        },
        "checkpoint": {
            "checkpoint_id": checkpoint_id,
            "weight_sha256": weight_sha,
            "production_activation_authorized": False,
        },
        "selection_freeze_evidence": {
            "path": str(freeze_path.resolve()),
            "file_sha256": _hash_file(freeze_path),
            "written_before_heldout_forward": True,
        },
        "structured_head_coverage_gate": {"passed": True},
        "optimization": {
            "seed": PROBE_TRAINING_SCHEDULE["seed"],
            "requested_epochs": PROBE_TRAINING_SCHEDULE["requested_epochs"],
            "early_stopping_patience": PROBE_TRAINING_SCHEDULE[
                "early_stopping_patience"
            ],
            "batch_size": PROBE_TRAINING_SCHEDULE["batch_size"],
            "learning_rate": PROBE_TRAINING_SCHEDULE["learning_rate"],
            "weight_decay": PROBE_TRAINING_SCHEDULE["weight_decay"],
            "gradient_clip_norm": PROBE_TRAINING_SCHEDULE["gradient_clip_norm"],
            "learning_rate_schedule": schedule,
            "effective_loss_contract": base_effective,
            "effective_dynamic_crop_loss_contract": dynamic_effective,
            "unseen_song_used_for_model_selection": False,
        },
        "held_out_evaluation": {
            "passed": True,
            "checkpoint_id": checkpoint_id,
            "weight_sha256": weight_sha,
            "training_song_ids": list(training_songs),
            "held_out_song_ids": [heldout_song],
            "held_out_reference_count": 5,
            "cut_threshold_frozen_before_heldout_forward": True,
            "quality_gate": {
                "passed": True,
                "thresholds": ProductionQualityThresholds().to_dict(),
            },
            "dynamic_crop_v2_metrics": {
                "schema_version": DYNAMIC_CROP_V2_SCHEMA_VERSION,
                "model_contract": DYNAMIC_CROP_V2_MODEL_CONTRACT,
                "target_contract": DYNAMIC_CROP_V2_TARGET_CONTRACT,
                "loss_contract": DYNAMIC_CROP_V2_LOSS_CONTRACT,
                "loss_components": {
                    "base_total": metric_value,
                    "dynamic_crop_values": metric_value,
                    "dynamic_crop_times": metric_value,
                    "dynamic_crop_presence": metric_value,
                    "dynamic_crop_motion": metric_value,
                },
            },
        },
    }
    report_path = fold_root / PRODUCTION_TRAINING_REPORT_FILENAME
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return write_fold_run_evidence(
        root,
        candidate_id=candidate.candidate_id,
        validation_song_id=heldout_song,
        dataset_manifest_path=dataset_path,
        production_training_report_path=report_path.resolve(),
        selection_freeze_path=freeze_path.resolve(),
        checkpoint_metadata_path=metadata_path.resolve(),
        checkpoint_weights_path=weight_path.resolve(),
    )


def _inputs(
    root: Path,
    candidate_values: tuple[tuple[DynamicCropCapacityCandidate, tuple[float, ...]], ...],
):
    dataset_path = _dataset(root)
    folds = []
    smokes = {}
    for candidate, values in candidate_values:
        smokes[candidate.candidate_id] = _smoke(root, candidate)
        folds.extend(
            _fold(root, candidate, song, dataset_path, value)
            for song, value in zip(SONGS, values, strict=True)
        )
    return dataset_path, tuple(folds), smokes


class DynamicCropCapacityProbeTests(unittest.TestCase):
    def test_fold_evaluation_has_no_public_scalar_constructor(self) -> None:
        with self.assertRaises(TypeError):
            FoldEvaluation(candidate_id="a" * 64)  # type: ignore[call-arg]

    def test_incomplete_reference_dataset_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            incomplete = dict(REFERENCES)
            incomplete[BANG_BANG] = REFERENCES[BANG_BANG][:-1]
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "exactly five"
            ):
                _dataset(root, incomplete)

    def test_missing_real_mlx_smoke_fails_before_selection(self) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path = _dataset(root)
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "real MLX smoke"
            ):
                build_dynamic_crop_capacity_selection(
                    candidates=(candidate,),
                    fold_run_evidence_paths=(),
                    selection_dataset_manifest_path=dataset_path,
                    mlx_smoke_evidence_paths_by_candidate={},
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=candidate.candidate_id,
                )

    def test_fake_report_schema_and_zero_metrics_cannot_authorize(self) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path = _dataset(root)
            bad_fold = _fold(
                root,
                candidate,
                BANG_BANG,
                dataset_path,
                1.0,
                report_schema_version=0,
            )
            other_folds = tuple(
                _fold(root, candidate, song, dataset_path, 1.0)
                for song in (BLACKHOLE, REBEL_HEART)
            )
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "production report contract"
            ):
                build_dynamic_crop_capacity_selection(
                    candidates=(candidate,),
                    fold_run_evidence_paths=(bad_fold, *other_folds),
                    selection_dataset_manifest_path=dataset_path,
                    mlx_smoke_evidence_paths_by_candidate={
                        candidate.candidate_id: _smoke(root, candidate)
                    },
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=candidate.candidate_id,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path, folds, smokes = _inputs(
                root, ((candidate, (0.0, 0.0, 0.0)),)
            )
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "no candidate passes"
            ):
                build_dynamic_crop_capacity_selection(
                    candidates=(candidate,),
                    fold_run_evidence_paths=folds,
                    selection_dataset_manifest_path=dataset_path,
                    mlx_smoke_evidence_paths_by_candidate=smokes,
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=candidate.candidate_id,
                )

    def test_catastrophic_fold_is_rejected(self) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path, folds, smokes = _inputs(
                root, ((candidate, (1.0, 1.0, 1000.0)),)
            )
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "no candidate passes"
            ):
                build_dynamic_crop_capacity_selection(
                    candidates=(candidate,),
                    fold_run_evidence_paths=folds,
                    selection_dataset_manifest_path=dataset_path,
                    mlx_smoke_evidence_paths_by_candidate=smokes,
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=candidate.candidate_id,
                )

    def test_tiny_loss_difference_prefers_smaller_model(self) -> None:
        small = _candidate("small", 32)
        large = _candidate("large", 64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path, folds, smokes = _inputs(
                root,
                (
                    (large, (1.0, 1.0, 1.0)),
                    (small, (1.0 + 1.0e-8,) * 3),
                ),
            )
            payload = build_dynamic_crop_capacity_selection(
                candidates=(large, small),
                fold_run_evidence_paths=tuple(reversed(folds)),
                selection_dataset_manifest_path=dataset_path,
                mlx_smoke_evidence_paths_by_candidate=smokes,
                device_memory_budget=BUDGET,
                baseline_candidate_id=large.candidate_id,
            )
            self.assertEqual(small.candidate_id, payload["selected_candidate_id"])

    def test_failed_large_smoke_rejects_only_large_candidate(self) -> None:
        small = _candidate("small", 32)
        large = _candidate("large", 64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path, folds, smokes = _inputs(
                root,
                (
                    (small, (1.0, 1.0, 1.0)),
                    (large, (0.9, 0.9, 0.9)),
                ),
            )
            smokes[large.candidate_id] = _smoke(root, large, passed=False)
            payload = build_dynamic_crop_capacity_selection(
                candidates=(small, large),
                fold_run_evidence_paths=folds,
                selection_dataset_manifest_path=dataset_path,
                mlx_smoke_evidence_paths_by_candidate=smokes,
                device_memory_budget=BUDGET,
                baseline_candidate_id=small.candidate_id,
            )
        self.assertEqual(small.candidate_id, payload["selected_candidate_id"])
        large_row = next(
            row
            for row in payload["ranking"]
            if row["candidate_id"] == large.candidate_id
        )
        self.assertFalse(large_row["eligible"])
        self.assertIn("real_mlx_smoke_failed", large_row["rejection_reasons"])

    def test_symlink_evidence_and_exclusive_overwrite_are_rejected(self) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path, folds, smokes = _inputs(
                root, ((candidate, (1.0, 1.0, 1.0)),)
            )
            symlink = root / "dataset-link.json"
            symlink.symlink_to(dataset_path)
            with self.assertRaisesRegex(DynamicCropCapacityProbeError, "symlink"):
                build_dynamic_crop_capacity_selection(
                    candidates=(candidate,),
                    fold_run_evidence_paths=folds,
                    selection_dataset_manifest_path=symlink,
                    mlx_smoke_evidence_paths_by_candidate=smokes,
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=candidate.candidate_id,
                )
            path = write_dynamic_crop_capacity_selection(
                root,
                candidates=(candidate,),
                fold_run_evidence_paths=folds,
                selection_dataset_manifest_path=dataset_path,
                mlx_smoke_evidence_paths_by_candidate=smokes,
                device_memory_budget=BUDGET,
                baseline_candidate_id=candidate.candidate_id,
            )
            self.assertEqual(
                path.read_bytes(),
                path.read_bytes(),
            )
            with self.assertRaises(FileExistsError):
                write_dynamic_crop_capacity_selection(
                    root,
                    candidates=(candidate,),
                    fold_run_evidence_paths=folds,
                    selection_dataset_manifest_path=dataset_path,
                    mlx_smoke_evidence_paths_by_candidate=smokes,
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=candidate.candidate_id,
                )

    def test_round_trip_emits_release_projection_and_origin(self) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path, folds, smokes = _inputs(
                root, ((candidate, (1.0, 1.0, 1.0)),)
            )
            path = write_dynamic_crop_capacity_selection(
                root,
                candidates=(candidate,),
                fold_run_evidence_paths=folds,
                selection_dataset_manifest_path=dataset_path,
                mlx_smoke_evidence_paths_by_candidate=smokes,
                device_memory_budget=BUDGET,
                baseline_candidate_id=candidate.candidate_id,
            )
            payload = load_dynamic_crop_capacity_selection(path)
        self.assertEqual(
            {
                "contract_version",
                "model_config",
                "model_config_sha256",
                "base_loss_contract",
                "base_loss_contract_sha256",
                "dynamic_crop_loss_contract",
                "dynamic_crop_loss_contract_sha256",
                "source_objective",
                "transition_authority_contract",
                "transition_authority_contract_sha256",
            },
            set(payload["release_model_training_contract"]),
        )
        self.assertEqual(
            payload["selection_identity_sha256"],
            payload["release_model_training_contract_origin"][
                "selection_identity_sha256"
            ],
        )

    def test_default_candidates_include_intermediate_and_forbid_dynamic_config(self) -> None:
        candidates = default_dynamic_crop_v2_probe_candidates()
        self.assertEqual(4, len(candidates))
        self.assertEqual({512, 640, 768}, {c.model_config.model_dimension for c in candidates})
        with self.assertRaisesRegex(
            DynamicCropCapacityProbeError, "non-default"
        ):
            replace(candidates[0], dynamic_crop_config=DynamicCropHeadV2Config(presence_threshold=0.6))
        from crosscut.stage_edit_release_fold_launcher import (
            _validate_model_training_contract,
        )

        for candidate in candidates:
            _validate_model_training_contract(
                candidate.release_model_training_contract()
            )

    def test_memory_budget_cannot_claim_float16_batch1_or_unsafe_fraction(self) -> None:
        for kwargs in (
            {"batch_size": 1},
            {"dtype_bytes": 2},
            {"maximum_training_fraction": 0.8},
        ):
            with self.assertRaises(DynamicCropCapacityProbeError):
                DeviceMemoryBudget(
                    device_name="Apple M4 Pro",
                    physical_memory_bytes=25_769_803_776,
                    **kwargs,
                )


if __name__ == "__main__":
    unittest.main()
