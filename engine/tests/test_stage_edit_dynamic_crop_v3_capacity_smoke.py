from __future__ import annotations

import unittest

import numpy as np

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
)
from crosscut.stage_edit_dynamic_crop_v3_capacity_smoke import (
    DynamicCropV3CapacitySmokeError,
    _normalize_execution,
    _pack_fixed_lane_training_arrays_v3,
    _validate_candidate_payload,
    default_dynamic_crop_v3_capacity_candidates,
)
from crosscut.stage_edit_reference_adapter import ReferenceTrainingExample
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditSequence,
    StageEditTarget,
    StageEditTargetSequence,
    StageEditToken,
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


if __name__ == "__main__":
    unittest.main()
