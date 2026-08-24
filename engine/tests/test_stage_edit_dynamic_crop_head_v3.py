from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

import numpy as np

from crosscut.render import EditorialTransformKeyframe
from crosscut.stage_edit_dynamic_crop_head_v2 import (
    DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
    DynamicCropTrajectoryTargetV2,
    StageEditDynamicCropTargetSequenceV2,
)
from crosscut.stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
    DYNAMIC_CROP_V3_MAX_KEYFRAMES,
    DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
    DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION,
    DYNAMIC_CROP_V3_TIME_BLOCK_SLICES,
    DYNAMIC_CROP_V3_TIME_LOGIT_COUNT,
    StageEditDynamicCropModelOutputV3,
    StageEditTransformerDynamicCropV3,
    dynamic_crop_v3_times_by_count,
    evaluate_stage_edit_dynamic_crop_v3,
    load_stage_edit_dynamic_crop_checkpoint_v3,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v3,
    renderer_range_normalized_values_v3,
    renderer_safe_values_from_raw_v3,
    save_stage_edit_dynamic_crop_checkpoint_v3,
    stage_edit_dynamic_crop_v3_parameter_count,
    tensorize_stage_edit_dynamic_crop_supervision_v3,
)
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
    DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT,
    DynamicCropHeadPredictionV3,
    decode_dynamic_crop_model_source_v3,
    dynamic_crop_prediction_from_model_source_v3,
    dynamic_crop_predictions_from_model_token_v3,
)
import crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 as adapter_v3
from crosscut.stage_edit_dynamic_crop_target_v3 import (
    simplify_dynamic_crop_trajectory_v3,
)
from crosscut.stage_edit_transformer import (
    StageEditSequence,
    mlx_runtime_version,
    smoke_stage_edit_config,
    tensorize_stage_edit_batch,
    tensorize_stage_edit_supervision,
)
from tests.test_stage_edit_transformer import (
    SCHEMA,
    _candidate,
    _sequence,
    _targets,
    _token,
)

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten
except ImportError:  # pragma: no cover - optional outside Apple Silicon.
    mx = None
    tree_flatten = None


def _trajectory(
    *,
    source_id: str = "C",
    motion_label: str = "dynamic",
    keyframes: tuple[EditorialTransformKeyframe, ...] | None = None,
) -> DynamicCropTrajectoryTargetV2:
    return DynamicCropTrajectoryTargetV2(
        shot_index=0,
        source_id=source_id,
        shot_duration_seconds=1.0,
        keyframes=keyframes
        or (
            EditorialTransformKeyframe(0.0, 1.04, 0.500, 0.500, 0.0),
            EditorialTransformKeyframe(0.5, 1.11, 0.508, 0.496, 0.2),
            EditorialTransformKeyframe(1.0, 1.05, 0.510, 0.500, 0.0),
        ),
        motion_label=motion_label,
        motion_label_loss_masked=False,
        artifact_content_sha256="0" * 64,
        build_fingerprint_sha256="1" * 64,
    )


def _permuted_sequence(*, reverse: bool) -> StageEditSequence:
    available = (
        (_candidate("C", 0.6), _candidate("A", 0.2))
        if reverse
        else (_candidate("A", 0.2), _candidate("C", 0.6))
    )
    candidates = available + (_candidate("B", 0.4, gate=False),)
    return StageEditSequence(
        "ive:permutation",
        tuple(
            _token(timestamp, candidates=candidates)
            for timestamp in (0.0, 0.5, 1.0)
        ),
    )


@unittest.skipUnless(mlx_runtime_version() is not None, "MLX is optional")
class StageEditDynamicCropHeadV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = smoke_stage_edit_config(SCHEMA)

    def _batches(
        self,
        trajectory: DynamicCropTrajectoryTargetV2,
        *,
        sequence: StageEditSequence | None = None,
    ):
        sequence = sequence or _sequence()
        with mx.stream(mx.cpu):
            batch = tensorize_stage_edit_batch(
                (sequence,), SCHEMA, self.config, device=mx.cpu
            )
            base = tensorize_stage_edit_supervision(
                (_targets(sequence.sequence_id),), batch, SCHEMA, device=mx.cpu
            )
            dynamic = tensorize_stage_edit_dynamic_crop_supervision_v3(
                (
                    StageEditDynamicCropTargetSequenceV2(
                        sequence.sequence_id, (trajectory, None, None)
                    ),
                ),
                batch,
                device=mx.cpu,
            )
        return batch, base, dynamic

    def test_v2_sequence_is_simplified_then_bound_to_exact_source_slot(self) -> None:
        trajectory = _trajectory(source_id="C")
        simplified = simplify_dynamic_crop_trajectory_v3(trajectory)
        batch, _, dynamic = self._batches(trajectory)

        self.assertEqual(dynamic.values.shape, (1, 3, 8, 8, 4))
        self.assertEqual(dynamic.normalized_times.shape, (1, 3, 8, 8))
        self.assertEqual(dynamic.count_targets.shape, (1, 3, 8))
        self.assertEqual(int(dynamic.target_source_indices[0, 0].item()), 2)
        self.assertTrue(bool(dynamic.track_mask[0, 0, 2].item()))
        self.assertFalse(bool(dynamic.track_mask[0, 0, 0].item()))
        self.assertEqual(
            int(dynamic.count_targets[0, 0, 2].item()),
            len(simplified.keyframes),
        )
        self.assertEqual(batch.source_ids[0][0][2], "C")

    def test_missing_or_hard_gated_target_source_fails_closed(self) -> None:
        for source_id, message in (("missing", "unique candidate"), ("B", "hard gate")):
            with self.subTest(source_id=source_id):
                with self.assertRaisesRegex(ValueError, message):
                    self._batches(_trajectory(source_id=source_id))

    def test_zero_initialized_model_has_source_axis_and_safe_static_prior(self) -> None:
        batch, _, _ = self._batches(_trajectory())
        with mx.stream(mx.cpu):
            model = StageEditTransformerDynamicCropV3(self.config)
            model.eval()
            output = model(batch)
            mx.eval(
                output.dynamic_crop_values,
                output.dynamic_crop_count_logits,
                output.dynamic_crop_time_interval_logits,
                output.dynamic_crop_motion_logits,
            )

        self.assertEqual(output.output_contract, DYNAMIC_CROP_V3_OUTPUT_CONTRACT)
        self.assertEqual(output.dynamic_crop_values.shape, (1, 3, 8, 8, 4))
        self.assertEqual(output.dynamic_crop_count_logits.shape, (1, 3, 8, 8))
        self.assertEqual(
            output.dynamic_crop_time_interval_logits.shape, (1, 3, 8, 35)
        )
        self.assertEqual(output.dynamic_crop_motion_logits.shape, (1, 3, 8))
        expected = np.asarray((1.02, 0.5, 0.5, 0.0))
        actual_values = np.asarray(output.dynamic_crop_values)
        np.testing.assert_allclose(
            actual_values,
            np.broadcast_to(expected, actual_values.shape),
            rtol=0.0,
            atol=2e-7,
        )
        np.testing.assert_array_equal(
            np.argmax(np.asarray(output.dynamic_crop_count_logits), axis=-1) + 1,
            np.ones((1, 3, 8), dtype=np.int64),
        )

    def test_renderer_manifold_accepts_extreme_raw_values(self) -> None:
        raw = np.asarray(
            [
                (-1.0e4, -1.0e4, -1.0e4, -1.0e4),
                (1.0e4, 1.0e4, -1.0e4, 1.0e4),
                (-1.0e4, 1.0e4, 0.0, 1.0e4),
                (1.0e4, 0.0, 1.0e4, -1.0e4),
            ],
            dtype=np.float32,
        )
        with mx.stream(mx.cpu):
            mapped = renderer_safe_values_from_raw_v3(mx.array(raw))
            mx.eval(mapped)
        mapped_values = np.asarray(mapped, dtype=np.float64)

        for value in mapped_values:
            EditorialTransformKeyframe(0.0, *value.tolist())
            self.assertLessEqual(np.hypot(value[1] - 0.5, value[2] - 0.5), 0.04)
            self.assertLessEqual(abs(value[3]), 1.60)
            self.assertGreaterEqual(value[0], 1.0)
            self.assertLessEqual(value[0], 1.18)

    def test_extreme_random_two_to_eight_keyframe_tracks_decode_without_projection(
        self,
    ) -> None:
        rng = np.random.default_rng(20260824)
        per_count = 700
        sample_count = per_count * 7
        raw = rng.normal(0.0, 30.0, size=(sample_count, 8, 4)).astype(
            np.float32
        )
        time_logits = rng.normal(0.0, 30.0, size=(sample_count, 35))
        with mx.stream(mx.cpu):
            mapped = renderer_safe_values_from_raw_v3(mx.array(raw))
            mx.eval(mapped)
        mapped_values = np.asarray(mapped)

        failures: list[str] = []
        projection_count = 0
        for sample_index in range(sample_count):
            count = 2 + sample_index % 7
            count_logits = np.full(8, -20.0)
            count_logits[count - 1] = 20.0
            prediction = DynamicCropHeadPredictionV3(
                source_id="random-extreme",
                source_index=0,
                values=tuple(
                    tuple(float(component) for component in item)
                    for item in mapped_values[sample_index]
                ),
                count_logits=tuple(count_logits),
                time_interval_logits=tuple(time_logits[sample_index]),
                motion_logit=0.0,
            )
            decoded = prediction.decode(2.0)
            if not decoded.eligible:
                failures.extend(decoded.rejection_reasons)
            projection_count += decoded.minimum_safe_zoom_projection_count

        self.assertEqual(failures, [])
        self.assertEqual(projection_count, 0)

    def test_renderer_range_normalization_equalizes_ten_percent_residuals(self) -> None:
        physical = mx.array((1.018, 0.504, 0.496, 0.16), dtype=mx.float32)
        normalized = np.asarray(renderer_range_normalized_values_v3(physical))
        np.testing.assert_allclose(
            normalized, np.asarray((0.1, 0.1, -0.1, 0.1)), atol=2e-7
        )

    def test_zero_center_renderer_manifold_gradient_is_finite(self) -> None:
        raw = mx.zeros((8, 4), dtype=mx.float32)
        gradient = mx.grad(
            lambda value: renderer_safe_values_from_raw_v3(value).sum()
        )(raw)
        mx.eval(gradient)
        result = np.asarray(gradient)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(np.abs(result[:, 1:3]) > 0.0))

    def test_count_conditional_time_blocks_ignore_other_counts_and_stay_interior(self) -> None:
        base = np.zeros((1, DYNAMIC_CROP_V3_TIME_LOGIT_COUNT), dtype=np.float32)
        changed = base.copy()
        count = 5
        start, stop = DYNAMIC_CROP_V3_TIME_BLOCK_SLICES[count]
        changed[:, :start] = 1.0e4
        changed[:, stop:] = -1.0e4
        with mx.stream(mx.cpu):
            base_times = dynamic_crop_v3_times_by_count(mx.array(base))
            changed_times = dynamic_crop_v3_times_by_count(mx.array(changed))
            extreme = base.copy()
            extreme[0, start:stop] = (1.0e4, -1.0e4, -1.0e4, -1.0e4, -1.0e4)
            extreme_times = dynamic_crop_v3_times_by_count(mx.array(extreme))
            mx.eval(base_times, changed_times, extreme_times)

        np.testing.assert_array_equal(
            np.asarray(base_times)[0, count - 1],
            np.asarray(changed_times)[0, count - 1],
        )
        selected = np.asarray(extreme_times)[0, count - 1, :count]
        self.assertEqual(selected[0], 0.0)
        self.assertTrue(np.all(np.diff(selected) > 0.0))
        self.assertLess(selected[-1], 1.0)

    def test_source_conditioning_is_permutation_equivariant_and_nonconstant(self) -> None:
        sequences = (_permuted_sequence(reverse=False), _permuted_sequence(reverse=True))
        with mx.stream(mx.cpu):
            batches = tuple(
                tensorize_stage_edit_batch(
                    (sequence,), SCHEMA, self.config, device=mx.cpu
                )
                for sequence in sequences
            )
            mx.random.seed(419)
            model = StageEditTransformerDynamicCropV3(self.config)
            model.dynamic_crop_source_head.weight = (
                0.01
                * mx.random.normal(model.dynamic_crop_source_head.weight.shape)
            )
            model.eval()
            outputs = tuple(model(batch) for batch in batches)
            mx.eval(
                *(output.dynamic_crop_count_logits for output in outputs),
                *(output.dynamic_crop_values for output in outputs),
            )

        def slot(batch, source_id: str) -> int:
            return batch.source_ids[0][0].index(source_id)

        for source_id in ("A", "C"):
            left_slot = slot(batches[0], source_id)
            right_slot = slot(batches[1], source_id)
            np.testing.assert_allclose(
                np.asarray(outputs[0].dynamic_crop_count_logits)[0, :, left_slot],
                np.asarray(outputs[1].dynamic_crop_count_logits)[0, :, right_slot],
                rtol=0.0,
                atol=2e-6,
            )
            np.testing.assert_allclose(
                np.asarray(outputs[0].dynamic_crop_values)[0, :, left_slot],
                np.asarray(outputs[1].dynamic_crop_values)[0, :, right_slot],
                rtol=0.0,
                atol=2e-6,
            )
        left = np.asarray(outputs[0].dynamic_crop_count_logits)
        self.assertFalse(
            np.allclose(
                left[0, 0, slot(batches[0], "A")],
                left[0, 0, slot(batches[0], "C")],
            )
        )

    def test_source_specific_adapter_decodes_count_five_and_masks_rejected_slot(self) -> None:
        batch, _, _ = self._batches(_trajectory())
        shape = tuple(int(value) for value in batch.source_candidate_mask.shape)
        values = np.broadcast_to(
            np.asarray((1.02, 0.5, 0.5, 0.0), dtype=np.float32),
            shape + (8, 4),
        ).copy()
        count_logits = np.zeros(shape + (8,), dtype=np.float32)
        count_logits[0, 0, 2, 4] = 20.0
        output = StageEditDynamicCropModelOutputV3(
            cut_timing_logits=None,
            source_selection_logits=None,
            duration_bucket_logits=None,
            transition_logits=None,
            crop=None,
            effective_source_mask=np.logical_and(
                np.asarray(batch.source_candidate_mask),
                np.asarray(batch.hard_source_gate_mask),
            ),
            dynamic_crop_values=values,
            dynamic_crop_count_logits=count_logits,
            dynamic_crop_time_interval_logits=np.zeros(shape + (35,), dtype=np.float32),
            dynamic_crop_motion_logits=np.zeros(shape, dtype=np.float32),
        )

        prediction = dynamic_crop_prediction_from_model_source_v3(
            output, batch, batch_index=0, token_index=0, source_index=2
        )
        self.assertEqual(prediction.source_id, "C")
        self.assertEqual(prediction.predicted_keyframe_count, 5)
        decoded = prediction.decode(2.0)
        self.assertTrue(decoded.eligible, decoded.rejection_reasons)
        self.assertEqual(decoded.adapter_contract, DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT)
        self.assertEqual(decoded.applied_keyframe_count, 5)
        self.assertEqual(decoded.minimum_safe_zoom_projection_count, 0)
        assert decoded.track is not None
        self.assertEqual(decoded.track.reason, "stage_edit_dynamic_crop_head_v3")
        self.assertTrue(
            all(
                right.time_seconds > left.time_seconds
                for left, right in zip(
                    decoded.track.keyframes, decoded.track.keyframes[1:]
                )
            )
        )

        with mock.patch.object(
            adapter_v3,
            "_validate_source_output_contract_v3",
            wraps=adapter_v3._validate_source_output_contract_v3,
        ) as validate_contract:
            predictions = dynamic_crop_predictions_from_model_token_v3(
                output, batch, batch_index=0, token_index=0
            )
        self.assertEqual(validate_contract.call_count, 1)
        self.assertIsNone(predictions[1])
        rejected = decode_dynamic_crop_model_source_v3(
            output,
            batch,
            batch_index=0,
            token_index=0,
            source_index=1,
            shot_duration_seconds=2.0,
        )
        self.assertFalse(rejected.eligible)
        self.assertTrue(rejected.loss_masked)
        self.assertIsNone(rejected.track)
        self.assertIn("hard-gated", rejected.rejection_reasons[0])

    def test_declared_parameter_count_matches_mlx_tree_and_budget_delta(self) -> None:
        with mx.stream(mx.cpu):
            model = StageEditTransformerDynamicCropV3(self.config)
            parameters = tree_flatten(model.parameters())
        actual = sum(
            int(np.prod(tuple(int(size) for size in value.shape)))
            for _, value in parameters
        )
        declared = stage_edit_dynamic_crop_v3_parameter_count(self.config)
        self.assertEqual(actual, declared)
        dynamic_parameters = (2 * self.config.model_dimension + 1) * (
            DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION
        )
        self.assertEqual(
            dynamic_parameters,
            sum(
                int(np.prod(tuple(int(size) for size in value.shape)))
                for name, value in parameters
                if name.startswith("dynamic_crop_")
            ),
        )

    def test_evaluator_reports_source_count_motion_curve_prior_and_renderer(self) -> None:
        batch, base, dynamic = self._batches(_trajectory())
        with mx.stream(mx.cpu):
            model = StageEditTransformerDynamicCropV3(self.config)
            report = evaluate_stage_edit_dynamic_crop_v3(
                model, batch, base, dynamic
            )

        trajectory = report["trajectory"]
        self.assertEqual(trajectory["track_label_count"], 1)
        self.assertEqual(trajectory["target_source_slot_mismatch_count"], 0)
        self.assertEqual(trajectory["count"]["label_count"], 1)
        self.assertIn("balanced_accuracy", trajectory["motion"])
        self.assertIn("zoom", trajectory["curve_delta_and_range"])
        self.assertIn(
            "aggregate_normalized_improvement_over_static_prior", trajectory
        )
        self.assertEqual(trajectory["renderer_decode"]["attempt_count"], 1)
        self.assertEqual(trajectory["renderer_decode"]["eligible_count"], 1)
        self.assertFalse(report["production_activation_authorized"])

    def test_v3_checkpoint_roundtrip_is_distinct_and_v2_directory_fails_closed(self) -> None:
        batch, _, _ = self._batches(_trajectory())
        with mx.stream(mx.cpu):
            model = StageEditTransformerDynamicCropV3(self.config)
            model.eval()
            expected = model(batch).dynamic_crop_values
            mx.eval(expected)
            metadata = new_stage_edit_dynamic_crop_checkpoint_metadata_v3(
                "crop-v3-smoke",
                SCHEMA,
                self.config,
                training_steps=1,
                weights_trained=True,
                training_data_id="simplified-source-slot-v3",
            )
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                saved = save_stage_edit_dynamic_crop_checkpoint_v3(
                    root, model, metadata
                )
                loaded, restored = load_stage_edit_dynamic_crop_checkpoint_v3(
                    root, expected_schema=SCHEMA
                )
                loaded.eval()
                actual = loaded(batch).dynamic_crop_values
                mx.eval(actual)

                self.assertTrue(
                    (root / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME).is_file()
                )
                self.assertTrue(
                    (root / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME).is_file()
                )
                self.assertFalse(
                    (root / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME).exists()
                )
                self.assertEqual(
                    saved.contract_fingerprint_sha256,
                    restored.contract_fingerprint_sha256,
                )
                np.testing.assert_allclose(
                    np.asarray(actual), np.asarray(expected), rtol=0.0, atol=0.0
                )

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME).write_text(
                    json.dumps({"format_version": 1}), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "v2 checkpoint is incompatible"):
                    load_stage_edit_dynamic_crop_checkpoint_v3(root)


if __name__ == "__main__":
    unittest.main()
