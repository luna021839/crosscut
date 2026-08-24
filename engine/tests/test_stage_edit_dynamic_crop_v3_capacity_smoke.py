from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crosscut import stage_edit_dynamic_crop_v3_capacity_smoke as capacity_smoke
from crosscut.render import EditorialTransformKeyframe
from crosscut.stage_edit_dynamic_crop_capacity_probe import (
    DeviceMemoryBudget,
    default_dynamic_crop_v2_probe_candidates,
)
from crosscut.stage_edit_dynamic_crop_head_v2 import (
    DynamicCropTrajectoryTargetV2,
    StageEditDynamicCropTargetSequenceV2,
)
from crosscut.stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_LOSS_CONTRACT,
    DYNAMIC_CROP_V3_MODEL_CONTRACT,
    StageEditTransformerDynamicCropV3,
)
from crosscut.stage_edit_dynamic_crop_v3_capacity_smoke import (
    DynamicCropV3CapacitySmokeError,
    V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE,
    V3_CAPACITY_SMOKE_ANALYSIS_BASIS,
    V3_CAPACITY_SMOKE_SCHEMA_VERSION,
    _enable_v3_capacity_activation_checkpointing,
    _normalize_execution,
    _pack_fixed_lane_training_arrays_v3,
    _validate_candidate_payload,
    _validate_smoke_log_execution_binding,
    default_dynamic_crop_v3_capacity_candidates,
    load_dynamic_crop_v3_capacity_smoke,
)
from crosscut.stage_edit_reference_adapter import ReferenceTrainingExample
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditSequence,
    StageEditTarget,
    StageEditTargetSequence,
    StageEditToken,
    StageEditTransformerConfig,
)


BUDGET = DeviceMemoryBudget(
    device_name="Apple M4 Pro",
    physical_memory_bytes=25_769_803_776,
)


def _token(value: float) -> StageEditToken:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    sources = tuple(
        SourceCandidateToken(
            source_id=f"source-{index}",
            features=(value + index,) * schema.raw_source_dimension,
            missing_mask=(False,) * schema.raw_source_dimension,
        )
        for index in range(2)
    )
    return StageEditToken(
        master_time_seconds=value,
        music_features=(value,) * len(schema.music_feature_names),
        section_features=(value,) * len(schema.section_feature_names),
        source_candidates=sources,
        recent_edit_features=(0.0,) * len(schema.recent_edit_feature_names),
        style_features=(value,) * len(schema.style_feature_names),
        missing_mask=(False,) * schema.raw_context_dimension,
    )


def _example(reference: str, values: tuple[float, ...]) -> ReferenceTrainingExample:
    sequence_id = f"sequence:{reference}"
    tokens = tuple(_token(value) for value in values)
    targets = tuple(
        StageEditTarget(
            cut=float(index % 2),
            selected_source_id="source-0",
            rejected_source_ids=("source-1",),
            duration_bucket=index,
            transition_label="hard_cut",
            crop=(1.02, 0.5, 0.5, 0.0),
        )
        for index, _value in enumerate(values)
    )
    return ReferenceTrainingExample(
        reference_id=reference,
        song_id="IVE-BANG-BANG",
        style_id="unit",
        sequence=StageEditSequence(sequence_id, tokens),
        targets=StageEditTargetSequence(sequence_id, targets),
        statistics={},
        feature_snapshot_sha256="a" * 64,
    )


def _dynamic(
    example: ReferenceTrainingExample, *, source_id: str = "source-1"
) -> StageEditDynamicCropTargetSequenceV2:
    targets = tuple(
        DynamicCropTrajectoryTargetV2(
            shot_index=index,
            source_id=source_id,
            shot_duration_seconds=1.0,
            keyframes=(
                EditorialTransformKeyframe(0.0, 1.02, 0.5, 0.5, 0.0),
                EditorialTransformKeyframe(1.0, 1.08, 0.5, 0.5, 0.0),
            ),
            motion_label="dynamic",
            motion_label_loss_masked=False,
            artifact_content_sha256=f"{index + 1:064x}",
            build_fingerprint_sha256=f"{index + 11:064x}",
        )
        for index, _token_value in enumerate(example.sequence.tokens)
    )
    return StageEditDynamicCropTargetSequenceV2(
        sequence_id=example.sequence.sequence_id,
        targets=targets,
    )


def _memory_stage(name: str, peak: int) -> dict[str, object]:
    return {
        "stage": name,
        "synchronized": True,
        "active_memory_bytes": peak // 2,
        "cache_memory_bytes": 0,
        "peak_memory_bytes": peak,
        "elapsed_seconds": 1.0,
    }


def _tree(element_count: int) -> dict[str, object]:
    return {
        "element_count": element_count,
        "leaf_count": 2,
        "nbytes": element_count * 4,
        "nonfinite_count": 0,
        "key_digest_sha256": "9" * 64,
    }


def _clean_code_provenance(commit: str = "a" * 40) -> dict[str, object]:
    return {
        "runner_contract": capacity_smoke.V3_CAPACITY_SMOKE_RUNNER_CONTRACT,
        "expected_git_commit": commit,
        "observed_git_commit": commit,
        "tracked_diff_policy": "require_clean",
        "repository_status_clean": True,
        "repository_status_sha256": hashlib.sha256(b"").hexdigest(),
        "source_files_match_observed_commit": True,
        "source_files": {
            role: {
                "path": str(Path("/Users/developer/review-root") / relative),
                "file_sha256": "c" * 64,
                "repo_relative_path": relative,
                "observed_commit_blob_sha256": "c" * 64,
                "matches_observed_commit": True,
            }
            for role, relative in (
                capacity_smoke.V3_CAPACITY_SMOKE_SOURCE_RELATIVE_PATHS.items()
            )
        },
        "release_trust_authorized": True,
    }


class DynamicCropV3CapacitySmokeTests(unittest.TestCase):
    def test_pinned_candidates_are_v3_only_and_parameter_distinct(self) -> None:
        v2 = default_dynamic_crop_v2_probe_candidates()
        v3 = default_dynamic_crop_v3_capacity_candidates()
        self.assertEqual(4, len(v3))
        self.assertEqual(
            tuple(item.candidate_name for item in v2),
            tuple(item.candidate_name for item in v3),
        )
        for legacy, candidate in zip(v2, v3, strict=True):
            contract = candidate.contract_dict()
            self.assertEqual(3, contract["dynamic_crop_version"])
            self.assertEqual(DYNAMIC_CROP_V3_MODEL_CONTRACT, contract["model_contract"])
            self.assertEqual(DYNAMIC_CROP_V3_LOSS_CONTRACT, contract["loss_contract"])
            self.assertEqual("StageEditTransformerDynamicCropV3", contract["model_class"])
            self.assertEqual("DynamicCropV3LossContract", contract["loss_class"])
            self.assertEqual(
                dict(V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE),
                contract["activation_checkpointing"],
            )
            self.assertFalse(contract["legacy_v2_checkpoint_compatible"])
            self.assertNotEqual(legacy.candidate_id, candidate.candidate_id)
            self.assertGreater(candidate.parameter_count, 0)

    def test_real_features_and_source_exact_v3_targets_tile_together(self) -> None:
        candidate = default_dynamic_crop_v3_capacity_candidates()[0]
        first = _example("reference-a", (0.00, 0.02))
        second = _example("reference-b", (0.04, 0.06))
        inputs, base, dynamic, provenance = _pack_fixed_lane_training_arrays_v3(
            examples_by_reference={
                first.reference_id: first,
                second.reference_id: second,
            },
            dynamic_by_reference={
                first.reference_id: _dynamic(first),
                second.reference_id: _dynamic(second),
            },
            lane_assignments={"A": (first.reference_id,), "B": (second.reference_id,)},
            candidate=candidate,
            sequence_length=4,
            max_sources=2,
        )
        self.assertEqual((2, 4, 2, 720), inputs["source_features"].shape)
        np.testing.assert_array_equal(base["source_targets"], 0)
        self.assertEqual((2, 4, 2, 8, 4), dynamic["values"].shape)
        self.assertTrue(np.all(dynamic["track_mask"][:, :, 1]))
        self.assertFalse(np.any(dynamic["track_mask"][:, :, 0]))
        np.testing.assert_array_equal(dynamic["target_source_indices"], 1)
        np.testing.assert_array_equal(dynamic["count_targets"][:, :, 1], 2)
        self.assertEqual(3, provenance["dynamic_crop_version"])
        self.assertEqual(DYNAMIC_CROP_V3_MODEL_CONTRACT, provenance["model_contract"])
        self.assertEqual(DYNAMIC_CROP_V3_LOSS_CONTRACT, provenance["loss_contract"])
        self.assertEqual(0, provenance["source_slot_mismatch_count"])
        self.assertFalse(provenance["legacy_v2_model_or_loss_used"])
        self.assertFalse(provenance["semantic_metrics_authorized"])

    def test_missing_v3_target_source_fails_closed(self) -> None:
        candidate = default_dynamic_crop_v3_capacity_candidates()[0]
        first = _example("reference-a", (0.00,))
        second = _example("reference-b", (0.04,))
        with self.assertRaisesRegex(
            DynamicCropV3CapacitySmokeError, "no unique candidate slot"
        ):
            _pack_fixed_lane_training_arrays_v3(
                examples_by_reference={
                    first.reference_id: first,
                    second.reference_id: second,
                },
                dynamic_by_reference={
                    first.reference_id: _dynamic(first, source_id="missing"),
                    second.reference_id: _dynamic(second),
                },
                lane_assignments={
                    "A": (first.reference_id,),
                    "B": (second.reference_id,),
                },
                candidate=candidate,
                sequence_length=2,
                max_sources=2,
            )

    def test_mixed_v2_candidate_contract_is_rejected(self) -> None:
        candidate = default_dynamic_crop_v3_capacity_candidates()[0]
        payload = candidate.to_dict()
        mixed_contract = dict(payload["contract"])
        mixed_contract["dynamic_crop_version"] = 2
        mixed = {**payload, "contract": mixed_contract}
        with self.assertRaisesRegex(
            DynamicCropV3CapacitySmokeError, "mixed, stale"
        ):
            _validate_candidate_payload(mixed)

    def test_schema_v1_smoke_without_rematerialization_is_rejected(self) -> None:
        with patch(
            "crosscut.stage_edit_dynamic_crop_v3_capacity_smoke._addressed_payload",
            return_value={
                "schema_version": 1,
                "analysis_basis": (
                    "real_mlx_dynamic_crop_v3_release_shape_capacity_smoke_v1"
                ),
                "artifact_content_sha256": "a" * 64,
            },
        ):
            with self.assertRaisesRegex(
                DynamicCropV3CapacitySmokeError, "keys are not exact"
            ):
                load_dynamic_crop_v3_capacity_smoke("/tmp/old-smoke.json")

    def test_schema_v2_with_non_rematerialized_evidence_is_rejected(self) -> None:
        candidate = default_dynamic_crop_v3_capacity_candidates()[0]
        non_rematerialized = {
            **dict(V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE),
            "enabled": False,
        }
        payload = {
            "schema_version": V3_CAPACITY_SMOKE_SCHEMA_VERSION,
            "analysis_basis": V3_CAPACITY_SMOKE_ANALYSIS_BASIS,
            "artifact_family": "dynamic_crop_v3_capacity_smoke",
            "dynamic_crop_version": 3,
            "activation_checkpointing": non_rematerialized,
            "candidate": candidate.to_dict(),
            "dataset_provenance": {},
            "code_provenance": {},
            "runtime_provenance": {},
            "hardware_provenance": {},
            "device_memory_budget": {},
            "execution": {},
            "runner_log": {},
            "supervisor_attestation": {},
            "capacity_runtime_evidence_only": True,
            "semantic_metrics_authorized": False,
            "model_selection_authorized": False,
            "artifact_content_sha256": "a" * 64,
        }
        with patch(
            "crosscut.stage_edit_dynamic_crop_v3_capacity_smoke._addressed_payload",
            return_value=payload,
        ):
            with self.assertRaisesRegex(
                DynamicCropV3CapacitySmokeError,
                "schema or authority differs",
            ):
                load_dynamic_crop_v3_capacity_smoke("/tmp/non-remat.json")

        payload["activation_checkpointing"] = {
            **dict(V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE),
            "enabled": 1,
            "previous_value": 0,
        }
        with patch(
            "crosscut.stage_edit_dynamic_crop_v3_capacity_smoke._addressed_payload",
            return_value=payload,
        ):
            with self.assertRaisesRegex(
                DynamicCropV3CapacitySmokeError,
                "schema or authority differs",
            ):
                load_dynamic_crop_v3_capacity_smoke("/tmp/non-remat-types.json")

    def test_execution_requires_fresh_exact_v3_full_training_step(self) -> None:
        candidate = default_dynamic_crop_v3_capacity_candidates()[0]
        peak = BUDGET.allowed_training_bytes - 1024
        hashes = {name: name[0] * 64 for name in ("dataset", "code", "runtime", "hardware")}
        suite = "e" * 64
        nonce = "f" * 64
        event = {
            "event": "worker_result",
            "dynamic_crop_version": 3,
            "model_class": "StageEditTransformerDynamicCropV3",
            "loss_class": "DynamicCropV3LossContract",
            "candidate_id": candidate.candidate_id,
            "suite_id": suite,
            "run_nonce_sha256": nonce,
            "worker_pid": 200,
            "provenance_hashes": hashes,
            "parameter_count": candidate.parameter_count,
            "activation_checkpointing": dict(
                V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
            ),
            "semantic_metrics_authorized": False,
            "model_selection_authorized": False,
            "tensor_provenance": {
                "source_tensor_shape": [2, 2048, 32, 720],
                "dynamic_crop_version": 3,
                "model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
                "loss_contract": DYNAMIC_CROP_V3_LOSS_CONTRACT,
                "legacy_v2_model_or_loss_used": False,
                "real_input_materialized": True,
            },
            "memory_stages": [
                _memory_stage(name, peak)
                for name in (
                    "input_materialized",
                    "model_allocated",
                    "forward_backward_completed",
                    "gradient_clip_completed",
                    "optimizer_step_completed",
                )
            ],
            "measured_peak_bytes": peak,
            "runtime_execution_passed": True,
            "gradients_and_state_finite": True,
            "optimizer_state_materialized": True,
            "within_release_memory_budget": True,
            "parameter_tree": _tree(candidate.parameter_count),
            "parameter_tree_before_checkpointing": _tree(
                candidate.parameter_count
            ),
            "gradient_tree": _tree(candidate.parameter_count),
            "post_update_parameter_tree": _tree(candidate.parameter_count),
            "optimizer_state_tree": _tree(candidate.parameter_count * 2),
            "loss_value_for_execution_only": 1.0,
            "gradient_global_norm_before_clip": 2.0,
        }
        result = _normalize_execution(
            event=event,
            returncode=0,
            candidate=candidate,
            budget=BUDGET,
            supervisor_pid=100,
            worker_pid=200,
            suite_id=suite,
            run_nonce_sha256=nonce,
            expected_hashes=hashes,
        )
        self.assertTrue(result["runtime_execution_passed"])
        self.assertTrue(result["capacity_smoke_passed"])
        self.assertFalse(result["semantic_metrics_authorized"])
        self.assertFalse(result["model_selection_authorized"])
        self.assertEqual(
            dict(V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE),
            result["activation_checkpointing"],
        )

        mixed_event = {**event, "model_class": "StageEditTransformerDynamicCropV2"}
        rejected = _normalize_execution(
            event=mixed_event,
            returncode=0,
            candidate=candidate,
            budget=BUDGET,
            supervisor_pid=100,
            worker_pid=200,
            suite_id=suite,
            run_nonce_sha256=nonce,
            expected_hashes=hashes,
        )
        self.assertFalse(rejected["runtime_execution_passed"])
        self.assertFalse(rejected["capacity_smoke_passed"])

        non_rematerialized = {
            **event,
            "activation_checkpointing": {
                **dict(V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE),
                "enabled": False,
            },
        }
        rejected = _normalize_execution(
            event=non_rematerialized,
            returncode=0,
            candidate=candidate,
            budget=BUDGET,
            supervisor_pid=100,
            worker_pid=200,
            suite_id=suite,
            run_nonce_sha256=nonce,
            expected_hashes=hashes,
        )
        self.assertFalse(rejected["runtime_execution_passed"])
        self.assertIsNone(rejected["activation_checkpointing"])

    def test_memory_overage_preserves_execution_but_fails_capacity_gate(self) -> None:
        candidate = default_dynamic_crop_v3_capacity_candidates()[0]
        peak = BUDGET.allowed_training_bytes + 1
        hashes = {name: name[0] * 64 for name in ("dataset", "code", "runtime", "hardware")}
        stages = [
            _memory_stage(name, peak)
            for name in (
                "input_materialized",
                "model_allocated",
                "forward_backward_completed",
                "gradient_clip_completed",
                "optimizer_step_completed",
            )
        ]
        event = {
            "event": "worker_result",
            "dynamic_crop_version": 3,
            "model_class": "StageEditTransformerDynamicCropV3",
            "loss_class": "DynamicCropV3LossContract",
            "candidate_id": candidate.candidate_id,
            "suite_id": "e" * 64,
            "run_nonce_sha256": "f" * 64,
            "worker_pid": 2,
            "provenance_hashes": hashes,
            "parameter_count": candidate.parameter_count,
            "activation_checkpointing": dict(
                V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
            ),
            "semantic_metrics_authorized": False,
            "model_selection_authorized": False,
            "tensor_provenance": {
                "source_tensor_shape": [2, 2048, 32, 720],
                "dynamic_crop_version": 3,
                "model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
                "loss_contract": DYNAMIC_CROP_V3_LOSS_CONTRACT,
                "legacy_v2_model_or_loss_used": False,
                "real_input_materialized": True,
            },
            "memory_stages": stages,
            "measured_peak_bytes": peak,
            "runtime_execution_passed": True,
            "gradients_and_state_finite": True,
            "optimizer_state_materialized": True,
            "within_release_memory_budget": False,
            "parameter_tree": _tree(candidate.parameter_count),
            "parameter_tree_before_checkpointing": _tree(
                candidate.parameter_count
            ),
            "gradient_tree": _tree(candidate.parameter_count),
            "post_update_parameter_tree": _tree(candidate.parameter_count),
            "optimizer_state_tree": _tree(candidate.parameter_count * 2),
            "loss_value_for_execution_only": 1.0,
            "gradient_global_norm_before_clip": 2.0,
        }
        result = _normalize_execution(
            event=event,
            returncode=0,
            candidate=candidate,
            budget=BUDGET,
            supervisor_pid=1,
            worker_pid=2,
            suite_id="e" * 64,
            run_nonce_sha256="f" * 64,
            expected_hashes=hashes,
        )
        self.assertTrue(result["runtime_execution_passed"])
        self.assertFalse(result["within_release_memory_budget"])
        self.assertFalse(result["capacity_smoke_passed"])

    def test_loader_replays_exact_worker_log_instead_of_trusting_receipt(self) -> None:
        candidate = default_dynamic_crop_v3_capacity_candidates()[0]
        peak = BUDGET.allowed_training_bytes - 1024
        suite = "e" * 64
        nonce = "f" * 64
        code = _clean_code_provenance()
        provenance = {
            "dataset": {"dataset": "fixture"},
            "code": code,
            "runtime": {"runtime": "fixture"},
            "hardware": {"hardware": "fixture"},
        }
        hashes = {
            name: capacity_smoke._sha256(value)
            for name, value in provenance.items()
        }
        event = {
            "event": "worker_result",
            "dynamic_crop_version": 3,
            "model_class": "StageEditTransformerDynamicCropV3",
            "loss_class": "DynamicCropV3LossContract",
            "candidate_id": candidate.candidate_id,
            "suite_id": suite,
            "run_nonce_sha256": nonce,
            "worker_pid": 200,
            "provenance_hashes": hashes,
            "parameter_count": candidate.parameter_count,
            "activation_checkpointing": dict(
                V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
            ),
            "semantic_metrics_authorized": False,
            "model_selection_authorized": False,
            "tensor_provenance": {
                "source_tensor_shape": [2, 2048, 32, 720],
                "dynamic_crop_version": 3,
                "model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
                "loss_contract": DYNAMIC_CROP_V3_LOSS_CONTRACT,
                "legacy_v2_model_or_loss_used": False,
                "real_input_materialized": True,
            },
            "memory_stages": [
                _memory_stage(name, peak)
                for name in (
                    "input_materialized",
                    "model_allocated",
                    "forward_backward_completed",
                    "gradient_clip_completed",
                    "optimizer_step_completed",
                )
            ],
            "measured_peak_bytes": peak,
            "runtime_execution_passed": True,
            "gradients_and_state_finite": True,
            "optimizer_state_materialized": True,
            "within_release_memory_budget": True,
            "parameter_tree": _tree(candidate.parameter_count),
            "parameter_tree_before_checkpointing": _tree(
                candidate.parameter_count
            ),
            "gradient_tree": _tree(candidate.parameter_count),
            "post_update_parameter_tree": _tree(candidate.parameter_count),
            "optimizer_state_tree": _tree(candidate.parameter_count * 2),
            "loss_value_for_execution_only": 1.0,
            "gradient_global_norm_before_clip": 2.0,
        }
        normalized = _normalize_execution(
            event=event,
            returncode=0,
            candidate=candidate,
            budget=BUDGET,
            supervisor_pid=100,
            worker_pid=200,
            suite_id=suite,
            run_nonce_sha256=nonce,
            expected_hashes=hashes,
        )
        command = [
            "/usr/bin/python3",
            "-P",
            "-S",
            "-m",
            "crosscut.stage_edit_dynamic_crop_v3_capacity_smoke",
            "--worker",
            "--candidate-name",
            candidate.candidate_name,
            "--expected-candidate-id",
            candidate.candidate_id,
            "--suite-id",
            suite,
            "--expected-dataset-sha256",
            hashes["dataset"],
            "--expected-code-sha256",
            hashes["code"],
            "--expected-runtime-sha256",
            hashes["runtime"],
            "--expected-hardware-sha256",
            hashes["hardware"],
            "--expected-git-commit",
            "a" * 40,
            "--tracked-diff-policy",
            "require_clean",
        ]
        for index in range(3):
            command.extend(("--feature-manifest", f"song-{index}=feature"))
            command.extend(
                ("--materialization-manifest", f"song-{index}=materialization")
            )
            command.extend(("--dynamic-crop-manifest", f"song-{index}=crop"))
        stdout = (
            capacity_smoke.V3_CAPACITY_SMOKE_EVENT_PREFIX
            + json.dumps(event, separators=(",", ":"))
            + "\n"
        )
        with tempfile.TemporaryDirectory(dir="/Users/developer") as raw:
            directory = Path(raw)
            log = capacity_smoke._content_addressed_payload(
                {
                    "schema_version": 1,
                    "analysis_basis": (
                        "dynamic_crop_v3_fresh_worker_process_log_v1"
                    ),
                    "suite_id": suite,
                    "supervisor_pid": 100,
                    "worker_pid": 200,
                    "fresh_process": True,
                    "run_nonce_sha256": nonce,
                    "started_at_unix_ns": 1,
                    "ended_at_unix_ns": 2,
                    "returncode": 0,
                    "command": command,
                    "stdout": stdout,
                    "stderr": "",
                }
            )
            log_path = directory / (
                f"{capacity_smoke.V3_CAPACITY_SMOKE_LOG_PREFIX}"
                f"{log['artifact_content_sha256']}.json"
            )
            log_path.write_text(json.dumps(log), encoding="utf-8")
            root = {
                "dataset_provenance": provenance["dataset"],
                "code_provenance": code,
                "runtime_provenance": provenance["runtime"],
                "hardware_provenance": provenance["hardware"],
                "device_memory_budget": BUDGET.to_dict(),
                "execution": normalized,
                "runner_log": {
                    "path": str(log_path),
                    "file_sha256": capacity_smoke._file_sha256(log_path),
                },
                "supervisor_attestation": {
                    "process_contract": (
                        capacity_smoke.V3_CAPACITY_SMOKE_PROCESS_CONTRACT
                    ),
                    "suite_id": suite,
                    "supervisor_pid": 100,
                    "worker_pid": 200,
                    "fresh_process": True,
                    "run_nonce_sha256": nonce,
                    "worker_event_count": 1,
                    "worker_event_sha256": capacity_smoke._sha256((event,)),
                },
            }
            _validate_smoke_log_execution_binding(
                root, candidate=candidate, code=code
            )
            forged = json.loads(json.dumps(root))
            forged["execution"]["worker_receipt"]["candidate_id"] = "0" * 64
            with self.assertRaisesRegex(
                DynamicCropV3CapacitySmokeError,
                "differs from the bound worker log event",
            ):
                _validate_smoke_log_execution_binding(
                    forged, candidate=candidate, code=code
                )

    def test_activation_checkpointing_enables_exact_encoder_switch(self) -> None:
        config = StageEditTransformerConfig.from_schema(
            DEFAULT_STAGE_EDIT_SCHEMA,
            model_dimension=64,
            num_heads=4,
            num_layers=1,
            feed_forward_dimension=128,
            max_sequence_length=8,
            max_sources=2,
            dropout=0.0,
        )
        model = StageEditTransformerDynamicCropV3(config)
        self.assertFalse(model.base.encoder.checkpoint)
        evidence = _enable_v3_capacity_activation_checkpointing(model)
        self.assertTrue(model.base.encoder.checkpoint)
        self.assertEqual(
            dict(V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE), evidence
        )
        with self.assertRaisesRegex(RuntimeError, "previous_value=false"):
            _enable_v3_capacity_activation_checkpointing(model)


if __name__ == "__main__":
    unittest.main()
