from dataclasses import replace
import unittest

from crosscut.dynamic_crop_source_frame_binding import (
    CropSourceFrameObservation, DynamicCropSourceFrameBinding,
    source_frame_binding_from_validated_crop_payload,
)
from crosscut.stage_edit_dynamic_crop_adapter import (
    DynamicCropKeyframeTarget, StageEditDynamicCropTarget,
)
from crosscut.stage_edit_dynamic_crop_head_v2 import DynamicCropTrajectoryTargetV2
from crosscut.stage_edit_dynamic_crop_target_v3 import simplify_dynamic_crop_trajectory_v3


def binding():
    return DynamicCropSourceFrameBinding("A", "a" * 64, 30., (
        CropSourceFrameObservation(0., 10., 20., 601),
        CropSourceFrameObservation(.5, 10.5, 20.5, 614),
        CropSourceFrameObservation(1., 11., 21., 630),
    ))


def adapter_target(source_binding=None):
    return StageEditDynamicCropTarget(
        shot_index=0, source_id="A", shot_duration_seconds=1.,
        motion_label="static", motion_label_loss_masked=False,
        keyframes=tuple(DynamicCropKeyframeTarget(t, 1.05, .5, .5, 0.) for t in (0., .5, 1.)),
        artifact_content_sha256="0" * 64, build_fingerprint_sha256="1" * 64,
        source_frame_binding=source_binding,
    )


class DynamicCropSourceFrameBindingTests(unittest.TestCase):
    def test_target_provenance_does_not_leak_into_model_inputs_or_change_geometry_loss(self):
        import numpy as np
        from crosscut.stage_edit_transformer import mlx_runtime_version
        if mlx_runtime_version() is None:
            self.skipTest('MLX is optional')
        import mlx.core as mx
        from crosscut.stage_edit_transformer import smoke_stage_edit_config, tensorize_stage_edit_batch
        from crosscut.stage_edit_dynamic_crop_head_v2 import (
            StageEditDynamicCropTargetSequenceV2, tensorize_stage_edit_dynamic_crop_supervision_v2,
        )
        from crosscut.stage_edit_dynamic_crop_head_v3 import tensorize_stage_edit_dynamic_crop_supervision_v3
        from tests.test_stage_edit_transformer import SCHEMA, _sequence
        sequence = _sequence()
        batch = tensorize_stage_edit_batch((sequence,), SCHEMA, smoke_stage_edit_config(SCHEMA), device=mx.cpu)
        before = np.asarray(batch.source_features).copy()
        plain = DynamicCropTrajectoryTargetV2.from_adapter_target(adapter_target())
        bound = DynamicCropTrajectoryTargetV2.from_adapter_target(adapter_target(binding()))
        for tensorize in (tensorize_stage_edit_dynamic_crop_supervision_v2, tensorize_stage_edit_dynamic_crop_supervision_v3):
            outputs = [tensorize((StageEditDynamicCropTargetSequenceV2(sequence.sequence_id, (target, None, None)),), batch, device=mx.cpu) for target in (plain, bound)]
            for name in ('values', 'normalized_times', 'keyframe_mask', 'motion_targets', 'track_mask'):
                np.testing.assert_array_equal(np.asarray(getattr(outputs[0], name)), np.asarray(getattr(outputs[1], name)))
        np.testing.assert_array_equal(before, np.asarray(batch.source_features))
        self.assertFalse(hasattr(batch, 'source_frame_binding'))

    def test_signed_phase_is_preserved_separately_from_geometry(self):
        target = adapter_target(binding())
        self.assertEqual(target.source_frame_binding.canonical_frame_indices, (600, 615, 630))
        self.assertEqual(target.source_frame_binding.frame_deltas, (1, -1, 0))
        serialized = target.to_target_dict()
        self.assertFalse(serialized['source_frame_binding']['model_input'])
        self.assertTrue(serialized['source_frame_binding']['source_frame_reconciliation_required'])
        self.assertEqual(set(serialized['keyframes'][0]), {'time_within_shot_seconds', 'zoom', 'center_x', 'center_y', 'roll_degrees'})

    def test_v3_static_simplification_keeps_all_source_observations(self):
        observed = binding()
        trajectory = DynamicCropTrajectoryTargetV2.from_adapter_target(adapter_target(observed))
        simplified = simplify_dynamic_crop_trajectory_v3(trajectory)
        self.assertEqual(len(simplified.keyframes), 1)
        self.assertIs(simplified.source_frame_binding, observed)
        self.assertEqual(len(simplified.source_frame_binding.observations), 3)
        self.assertEqual(simplified.to_dict()['source_frame_binding']['frame_deltas'], [1, -1, 0])

    def test_legacy_geometry_is_not_silently_retimed_or_masked(self):
        plain = DynamicCropTrajectoryTargetV2.from_adapter_target(adapter_target())
        bound = DynamicCropTrajectoryTargetV2.from_adapter_target(adapter_target(binding()))
        self.assertEqual(plain.keyframes, bound.keyframes)
        self.assertEqual(simplify_dynamic_crop_trajectory_v3(plain).keyframes, simplify_dynamic_crop_trajectory_v3(bound).keyframes)
        self.assertIsNone(plain.source_frame_binding)

    def test_source_and_keyframe_bindings_cannot_be_swapped(self):
        target = DynamicCropTrajectoryTargetV2.from_adapter_target(adapter_target(binding()))
        with self.assertRaisesRegex(ValueError, 'bind every'):
            replace(target, source_id="B")
        with self.assertRaisesRegex(ValueError, 'bind every'):
            replace(target, source_frame_binding=replace(binding(), observations=binding().observations[:2]))
        simplified = simplify_dynamic_crop_trajectory_v3(target)
        with self.assertRaisesRegex(ValueError, 'all original'):
            replace(simplified, source_frame_binding=replace(binding(), observations=binding().observations[:1]))

    def test_clock_quantization_matches_feature_extractor_ties_to_even(self):
        observed = replace(binding(), observations=(CropSourceFrameObservation(0., 10., 20.5 / 30., 20),))
        self.assertEqual(observed.canonical_frame_indices, (20,))
        self.assertFalse(observed.requires_source_frame_reconciliation)

    def test_bad_indices_times_and_hashes_are_rejected(self):
        with self.assertRaises(ValueError):
            replace(binding(), source_frame_rate=float('nan'))
        with self.assertRaises(ValueError):
            replace(binding(), source_media_sha256="unknown")
        with self.assertRaises(ValueError):
            CropSourceFrameObservation(0., 0., 0., True)
        with self.assertRaises(ValueError):
            replace(binding(), observations=(CropSourceFrameObservation(.1, 10., 20., 600),))
        with self.assertRaises(ValueError):
            replace(binding(), observations=(binding().observations[0], CropSourceFrameObservation(.5, 12., 20., 600)))

    def test_reversed_selected_frames_are_evidence_not_auto_repaired(self):
        observed = replace(binding(), observations=(
            CropSourceFrameObservation(0., 10., 20., 601),
            CropSourceFrameObservation(.01, 10.01, 20.01, 600),
        ))
        self.assertEqual(tuple(o.source_frame_index for o in observed.observations), (601, 600))

    def test_piecewise_clock_uses_absolute_points_without_double_offset(self):
        payload = {'shot': {'source_id':'A'}, 'provenance': {'source_media': {
            'media_sha256':'a'*64, 'frame_rate':30., 'clock': {
                'global_offset_seconds':5., 'control_points': [
                    {'master_seconds':0., 'source_seconds':5.},
                    {'master_seconds':10., 'source_seconds':15.},
                ]},
        }}, 'supervision': {'renderer_keyframes':[{
            'time_within_shot_seconds':0., 'master_seconds':2.,
            'source_frame_index':211, 'source_frame_seconds':211/30.,
        }]}}
        observed = source_frame_binding_from_validated_crop_payload(payload)
        self.assertEqual(observed.canonical_frame_indices, (210,))
        self.assertEqual(observed.frame_deltas, (1,))
        payload['supervision']['renderer_keyframes'][0]['source_frame_seconds'] = 7.
        with self.assertRaisesRegex(ValueError, 'timestamp differ'):
            source_frame_binding_from_validated_crop_payload(payload)


if __name__ == '__main__':
    unittest.main()
