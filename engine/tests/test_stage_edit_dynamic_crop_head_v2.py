from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from crosscut.reference_dynamic_crop_supervision import (
    build_reference_dynamic_crop_supervision,
)
from crosscut.render import EditorialTransformKeyframe
from crosscut.stage_edit_dynamic_crop_adapter import (
    build_stage_edit_dynamic_crop_target,
    load_stage_edit_dynamic_crop_target,
)
from crosscut.stage_edit_dynamic_crop_head_v2 import (
    DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME,
    DYNAMIC_CROP_V2_MAX_KEYFRAMES,
    DYNAMIC_CROP_V2_MODEL_CONTRACT,
    DYNAMIC_CROP_V2_TARGET_CONTRACT,
    DYNAMIC_CROP_V2_TIME_PARAMETERIZATION,
    DynamicCropTrajectoryTargetV2,
    StageEditDynamicCropModelOutputV2,
    StageEditDynamicCropTargetSequenceV2,
    StageEditTransformerDynamicCropV2,
    evaluate_stage_edit_dynamic_crop_v2,
    load_stage_edit_dynamic_crop_checkpoint_v2,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v2,
    save_stage_edit_dynamic_crop_checkpoint_v2,
    stage_edit_dynamic_crop_v2_parameter_count,
    stage_edit_dynamic_crop_v2_loss,
    tensorize_stage_edit_dynamic_crop_supervision_v2,
)
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter import (
    decode_dynamic_crop_model_token_v2,
    decode_dynamic_crop_prediction_v2,
    decode_dynamic_crop_target_v2,
)
from crosscut.stage_edit_transformer import (
    StageEditTransformer,
    mlx_runtime_version,
    smoke_stage_edit_config,
    tensorize_stage_edit_batch,
    tensorize_stage_edit_supervision,
)
from tests.test_reference_crop_supervision import _renderer_affine
from tests.test_reference_dynamic_crop_supervision import (
    _config as crop_v2_config,
    _request as crop_v2_request,
    _write_canonical_inputs,
    _write_dynamic_pair,
)
from tests.test_stage_edit_transformer import SCHEMA, _sequence, _targets

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
except ImportError:  # pragma: no cover - optional outside Apple Silicon.
    mx = None
    nn = None
    tree_flatten = None


ACTUAL_BANG_CROP_V2 = Path(
    "/Volumes/CrossCutMedia/CrossCut-Scratch/reference-edits/bang-bang/"
    "evidence-crop-v2-real-shot-start-v3/"
    "0O-bHFSeYxQ-reference-dynamic-crop-supervision-crop-v2-"
    "shot-0002-EzlauTqFrRo.json"
)


def _trajectory(
    *,
    keyframes: tuple[EditorialTransformKeyframe, ...] | None = None,
    motion_label: str = "dynamic",
    motion_masked: bool = False,
) -> DynamicCropTrajectoryTargetV2:
    return DynamicCropTrajectoryTargetV2(
        shot_index=0,
        source_id="A",
        shot_duration_seconds=1.0,
        keyframes=keyframes
        or (
            EditorialTransformKeyframe(0.0, 1.04, 0.500, 0.500, 0.0),
            EditorialTransformKeyframe(0.5, 1.08, 0.505, 0.500, 0.0),
            EditorialTransformKeyframe(1.0, 1.10, 0.510, 0.500, 0.0),
        ),
        motion_label=motion_label,
        motion_label_loss_masked=motion_masked,
        artifact_content_sha256="0" * 64,
        build_fingerprint_sha256="1" * 64,
    )


@unittest.skipUnless(mlx_runtime_version() is not None, "MLX is optional")
class StageEditDynamicCropHeadV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = smoke_stage_edit_config(SCHEMA)

    def _batches(self, trajectory: DynamicCropTrajectoryTargetV2):
        with mx.stream(mx.cpu):
            batch = tensorize_stage_edit_batch(
                (_sequence(),), SCHEMA, self.config, device=mx.cpu
            )
            base = tensorize_stage_edit_supervision(
                (_targets(),), batch, SCHEMA, device=mx.cpu
            )
            dynamic = tensorize_stage_edit_dynamic_crop_supervision_v2(
                (
                    StageEditDynamicCropTargetSequenceV2(
                        "ive:test", (trajectory, None, None)
                    ),
                ),
                batch,
                device=mx.cpu,
            )
        return batch, base, dynamic

    def _assert_all_dynamic_heads_receive_gradient(
        self, trajectory: DynamicCropTrajectoryTargetV2
    ) -> float:
        with mx.stream(mx.cpu):
            mx.random.seed(319)
            batch, base, dynamic = self._batches(trajectory)
            model = StageEditTransformerDynamicCropV2(self.config)
            model.train()
            loss_and_grad = nn.value_and_grad(
                model,
                lambda trained: stage_edit_dynamic_crop_v2_loss(
                    trained, batch, base, dynamic
                )[0],
            )
            loss, gradients = loss_and_grad(model)
            mx.eval(loss, gradients)
            flattened = dict(tree_flatten(gradients))
            for prefix in (
                "dynamic_crop_values_head",
                "dynamic_crop_time_head",
                "dynamic_crop_presence_head",
                "dynamic_crop_motion_head",
            ):
                values = [
                    value
                    for name, value in flattened.items()
                    if name.startswith(prefix + ".")
                ]
                self.assertTrue(values, f"missing gradient namespace {prefix}")
                magnitude = sum(
                    float(mx.sum(mx.abs(value)).item()) for value in values
                )
                self.assertGreater(magnitude, 0.0, prefix)
            return float(loss.item())

    def test_k8_tensorization_preserves_variable_track_and_explicit_masks(self) -> None:
        trajectory = _trajectory()
        _, _, dynamic = self._batches(trajectory)

        self.assertEqual(dynamic.values.shape, (1, 3, 8, 4))
        self.assertEqual(dynamic.normalized_times.shape, (1, 3, 8))
        self.assertEqual(dynamic.keyframe_mask.shape, (1, 3, 8))
        self.assertEqual(dynamic.presence_targets.shape, (1, 3, 8))
        self.assertEqual(dynamic.motion_targets.shape, (1, 3))
        np.testing.assert_array_equal(
            np.asarray(dynamic.keyframe_mask[0, 0]),
            np.array([True, True, True, False, False, False, False, False]),
        )
        np.testing.assert_array_equal(
            np.asarray(dynamic.presence_mask[0, 0]), np.ones(8, dtype=bool)
        )
        np.testing.assert_array_equal(
            np.asarray(dynamic.presence_targets[0, 0]),
            np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            np.asarray(dynamic.normalized_times[0, 0, :3]),
            np.array([0.0, 0.5, 1.0]),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(int(dynamic.keyframe_count_targets[0, 0].item()), 3)
        self.assertTrue(bool(dynamic.motion_mask[0, 0].item()))
        self.assertEqual(float(dynamic.motion_targets[0, 0].item()), 1.0)
        self.assertEqual(DYNAMIC_CROP_V2_MAX_KEYFRAMES, 8)
        self.assertEqual(trajectory.target_contract, DYNAMIC_CROP_V2_TARGET_CONTRACT)

    def test_single_point_motion_label_is_independently_loss_masked(self) -> None:
        trajectory = _trajectory(
            keyframes=(EditorialTransformKeyframe(0.0, 1.05),),
            motion_label="static",
            motion_masked=True,
        )
        _, _, dynamic = self._batches(trajectory)

        self.assertTrue(bool(dynamic.track_mask[0, 0].item()))
        self.assertTrue(bool(dynamic.keyframe_mask[0, 0, 0].item()))
        self.assertFalse(bool(dynamic.motion_mask[0, 0].item()))

    def test_model_has_distinct_checkpoint_contract_and_monotonic_k8_output(
        self,
    ) -> None:
        batch, base, dynamic = self._batches(_trajectory())
        with mx.stream(mx.cpu):
            mx.random.seed(313)
            legacy = StageEditTransformer(self.config)
            model = StageEditTransformerDynamicCropV2(self.config)
            model.eval()
            output = model(batch)
            mx.eval(
                output.dynamic_crop_values,
                output.dynamic_crop_normalized_times,
                output.dynamic_crop_presence_logits,
                output.dynamic_crop_motion_logits,
            )

        self.assertEqual(output.dynamic_crop_values.shape, (1, 3, 8, 4))
        self.assertEqual(output.dynamic_crop_normalized_times.shape, (1, 3, 8))
        self.assertEqual(output.dynamic_crop_presence_logits.shape, (1, 3, 8))
        self.assertEqual(output.dynamic_crop_motion_logits.shape, (1, 3))
        times = np.asarray(output.dynamic_crop_normalized_times)
        self.assertTrue(np.all(np.diff(times, axis=-1) > 0.0))
        self.assertTrue(np.all((times >= 0.0) & (times <= 1.0)))
        np.testing.assert_array_equal(times[..., 0], np.zeros(times.shape[:-1]))
        self.assertEqual(
            model.checkpoint_contract["model_contract"],
            DYNAMIC_CROP_V2_MODEL_CONTRACT,
        )
        self.assertFalse(
            model.checkpoint_contract[
                "legacy_checkpoint_parameter_layout_compatible"
            ]
        )
        self.assertFalse(
            model.checkpoint_contract["production_activation_authorized"]
        )
        self.assertEqual(
            DYNAMIC_CROP_V2_TIME_PARAMETERIZATION,
            model.checkpoint_contract["time_parameterization"],
        )
        legacy_names = {name for name, _ in tree_flatten(legacy.parameters())}
        v2_names = {name for name, _ in tree_flatten(model.parameters())}
        self.assertTrue(
            all(
                name.startswith("base.")
                for name in v2_names
                if "dynamic_crop" not in name
            )
        )
        self.assertFalse(any(name.startswith("base.") for name in legacy_names))

        report = evaluate_stage_edit_dynamic_crop_v2(
            model, batch, base, dynamic
        )
        self.assertEqual(report["trajectory"]["track_label_count"], 1)
        self.assertEqual(report["trajectory"]["keyframe_label_count"], 3)
        self.assertFalse(report["legacy_static_crop_head_used_for_trajectory"])
        self.assertFalse(report["production_activation_authorized"])

    def test_declared_parameter_count_matches_actual_mlx_parameter_tree(self) -> None:
        with mx.stream(mx.cpu):
            model = StageEditTransformerDynamicCropV2(self.config)
            parameters = tree_flatten(model.parameters())

        actual = sum(
            int(np.prod(tuple(int(size) for size in value.shape)))
            for _, value in parameters
        )
        self.assertEqual(
            actual,
            stage_edit_dynamic_crop_v2_parameter_count(self.config),
        )

    def test_v2_checkpoint_round_trip_uses_separate_files_and_fingerprint(self) -> None:
        batch, _, _ = self._batches(_trajectory())
        with mx.stream(mx.cpu):
            mx.random.seed(317)
            model = StageEditTransformerDynamicCropV2(self.config)
            model.eval()
            expected = model(batch).dynamic_crop_values
            mx.eval(expected)
            metadata = new_stage_edit_dynamic_crop_checkpoint_metadata_v2(
                "crop-v2-smoke",
                SCHEMA,
                self.config,
                training_steps=1,
                weights_trained=True,
                training_data_id="synthetic-and-bang-v2",
            )
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                saved = save_stage_edit_dynamic_crop_checkpoint_v2(
                    root, model, metadata
                )
                loaded, restored = load_stage_edit_dynamic_crop_checkpoint_v2(
                    root, expected_schema=SCHEMA
                )
                loaded.eval()
                actual = loaded(batch).dynamic_crop_values
                mx.eval(actual)

                self.assertTrue(
                    (root / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME).is_file()
                )
                self.assertTrue(
                    (root / DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME).is_file()
                )
                self.assertFalse((root / "metadata.json").exists())
                self.assertFalse((root / "model.safetensors").exists())
                self.assertEqual(
                    saved.contract_fingerprint_sha256,
                    restored.contract_fingerprint_sha256,
                )
                self.assertFalse(restored.production_activation_authorized)
                self.assertEqual(
                    DYNAMIC_CROP_V2_TIME_PARAMETERIZATION,
                    restored.time_parameterization,
                )
                np.testing.assert_allclose(
                    np.asarray(actual), np.asarray(expected), rtol=0.0, atol=0.0
                )

    def test_synthetic_crop_v2_artifact_reaches_every_dynamic_head_gradient(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transforms = [
                _renderer_affine(
                    480,
                    270,
                    zoom=1.045 + 0.005 * index,
                    center_x=0.500 + 0.0008 * index,
                    center_y=0.500,
                    roll_degrees=0.0,
                )
                for index in range(18)
            ]
            reference, source = _write_dynamic_pair(root, transforms)
            attribution, evidence, shot = _write_canonical_inputs(
                root, reference, source
            )
            artifact = build_reference_dynamic_crop_supervision(
                crop_v2_request(attribution, evidence, shot, reference, source),
                config=crop_v2_config(),
            )
            adapter_target = build_stage_edit_dynamic_crop_target(artifact.payload)

        self.assertIsNotNone(adapter_target)
        assert adapter_target is not None
        trajectory = DynamicCropTrajectoryTargetV2.from_adapter_target(
            adapter_target
        )
        self.assertEqual(len(trajectory.keyframes), 4)
        self.assertEqual(trajectory.motion_label, "dynamic")
        self.assertGreater(
            self._assert_all_dynamic_heads_receive_gradient(trajectory), 0.0
        )

    @unittest.skipUnless(
        ACTUAL_BANG_CROP_V2.exists(), "actual BANG crop-v2 artifact unavailable"
    )
    def test_actual_bang_crop_v2_artifact_trains_and_decodes_renderer_track(
        self,
    ) -> None:
        adapter_target = load_stage_edit_dynamic_crop_target(
            ACTUAL_BANG_CROP_V2, verify_input_files=True
        )
        self.assertIsNotNone(adapter_target)
        assert adapter_target is not None
        trajectory = DynamicCropTrajectoryTargetV2.from_adapter_target(
            adapter_target
        )

        self.assertEqual(trajectory.source_id, "EzlauTqFrRo")
        self.assertEqual(len(trajectory.keyframes), 2)
        self.assertEqual(trajectory.keyframes[0].time_seconds, 0.0)
        self.assertGreater(
            self._assert_all_dynamic_heads_receive_gradient(trajectory), 0.0
        )
        decoded = decode_dynamic_crop_target_v2(trajectory)
        self.assertTrue(decoded.eligible, decoded.rejection_reasons)
        self.assertFalse(decoded.loss_masked)
        self.assertIsNotNone(decoded.track)
        self.assertEqual(decoded.applied_keyframe_count, 2)


class DynamicCropSemiMarkovAdapterV2Tests(unittest.TestCase):
    @staticmethod
    def _prediction(
        *,
        times: tuple[float, ...] = (0.0, 0.5),
        values: tuple[tuple[float, float, float, float], ...] = (
            (1.04, 0.5, 0.5, 0.0),
            (1.08, 0.505, 0.5, 0.0),
        ),
    ):
        count = len(times)
        return decode_dynamic_crop_prediction_v2(
            normalized_times=times + (0.0,) * (8 - count),
            values=values + ((0.0, 0.0, 0.0, 0.0),) * (8 - count),
            presence_logits=(20.0,) * count + (-20.0,) * (8 - count),
            shot_duration_seconds=2.0,
        )

    def test_minimum_safe_zoom_projection_is_the_only_transform_mutation(self) -> None:
        decoded = self._prediction(
            times=(0.0, 0.5),
            values=((0.99, 0.5, 0.5, 0.0), (1.04, 0.505, 0.5, 0.0)),
        )

        self.assertTrue(decoded.eligible, decoded.rejection_reasons)
        assert decoded.track is not None
        self.assertEqual(decoded.minimum_safe_zoom_projection_count, 1)
        self.assertEqual(decoded.track.keyframes[0].zoom, 1.0)
        self.assertEqual(decoded.track.keyframes[1].center_x, 0.505)
        self.assertEqual(decoded.track.keyframes[1].roll_degrees, 0.0)

    def test_model_output_token_connection_produces_renderer_track(self) -> None:
        values = np.zeros((1, 1, 8, 4), dtype=np.float32)
        values[0, 0, 0] = (1.04, 0.500, 0.500, 0.0)
        values[0, 0, 1] = (1.08, 0.505, 0.500, 0.0)
        times = np.zeros((1, 1, 8), dtype=np.float32)
        times[0, 0, :2] = (0.0, 0.5)
        presence = np.full((1, 1, 8), -20.0, dtype=np.float32)
        presence[0, 0, :2] = 20.0
        output = StageEditDynamicCropModelOutputV2(
            cut_timing_logits=None,
            source_selection_logits=None,
            duration_bucket_logits=None,
            transition_logits=None,
            crop=None,
            effective_source_mask=None,
            dynamic_crop_values=values,
            dynamic_crop_normalized_times=times,
            dynamic_crop_presence_logits=presence,
            dynamic_crop_motion_logits=np.zeros((1, 1), dtype=np.float32),
        )

        decoded = decode_dynamic_crop_model_token_v2(
            output,
            batch_index=0,
            token_index=0,
            shot_duration_seconds=2.0,
        )

        self.assertTrue(decoded.eligible, decoded.rejection_reasons)
        assert decoded.track is not None
        self.assertEqual(len(decoded.track.keyframes), 2)
        self.assertEqual(decoded.track.keyframes[1].time_seconds, 1.0)

    def test_nonzero_start_is_rejected_without_synthesized_keyframe(self) -> None:
        decoded = self._prediction(times=(0.1, 0.5))

        self.assertFalse(decoded.eligible)
        self.assertTrue(decoded.loss_masked)
        self.assertIsNone(decoded.track)
        self.assertEqual(decoded.applied_keyframe_count, 0)
        self.assertIn("does not start", decoded.rejection_reasons[0])

    def test_noncontiguous_presence_is_rejected_without_compacting_keyframes(self) -> None:
        values = np.zeros((1, 1, 8, 4), dtype=np.float32)
        values[0, 0, 0] = (1.04, 0.500, 0.500, 0.0)
        values[0, 0, 2] = (1.08, 0.505, 0.500, 0.0)
        times = np.linspace(0.0, 1.0, 8, dtype=np.float32).reshape(1, 1, 8)
        presence = np.full((1, 1, 8), -20.0, dtype=np.float32)
        presence[0, 0, (0, 2)] = 20.0
        output = StageEditDynamicCropModelOutputV2(
            cut_timing_logits=None,
            source_selection_logits=None,
            duration_bucket_logits=None,
            transition_logits=None,
            crop=None,
            effective_source_mask=None,
            dynamic_crop_values=values,
            dynamic_crop_normalized_times=times,
            dynamic_crop_presence_logits=presence,
            dynamic_crop_motion_logits=np.zeros((1, 1), dtype=np.float32),
        )

        decoded = decode_dynamic_crop_model_token_v2(
            output,
            batch_index=0,
            token_index=0,
            shot_duration_seconds=2.0,
        )

        self.assertFalse(decoded.eligible)
        self.assertTrue(decoded.loss_masked)
        self.assertIsNone(decoded.track)
        self.assertIn("contiguous prefix", decoded.rejection_reasons[0])

    def test_out_of_domain_pan_and_excessive_zoom_fail_closed(self) -> None:
        pan = self._prediction(values=((1.10, 0.55, 0.5, 0.0), (1.10, 0.5, 0.5, 0.0)))
        zoom = self._prediction(values=((1.19, 0.5, 0.5, 0.0), (1.10, 0.5, 0.5, 0.0)))

        for decoded in (pan, zoom):
            self.assertFalse(decoded.eligible)
            self.assertTrue(decoded.loss_masked)
            self.assertIsNone(decoded.track)
            self.assertEqual(decoded.minimum_safe_zoom_projection_count, 0)

    def test_renderer_interpolation_failure_rejects_entire_track(self) -> None:
        decoded = self._prediction(
            times=(0.0, 1.0),
            values=(
                (
                    1.0325846172744377,
                    0.49955211673394156,
                    0.5002950952466916,
                    0.5404180985226001,
                ),
                (
                    1.0872770599014896,
                    0.5099961525870863,
                    0.49354861234563674,
                    1.568702272916933,
                ),
            ),
        )

        self.assertFalse(decoded.eligible)
        self.assertTrue(decoded.loss_masked)
        self.assertIsNone(decoded.track)
        self.assertIn("interpolation", decoded.rejection_reasons[0])


if __name__ == "__main__":
    unittest.main()
