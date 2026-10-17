from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from crosscut.render import EditorialTransformKeyframe
from crosscut.stage_edit_prior_fit_metrics import evaluate_materialized_prior_fit
from crosscut.stage_edit_production_training import _DynamicCropV3PriorTrack, _DynamicCropV3PriorHeadModule
from crosscut.stage_edit_dynamic_crop_head_v3 import _PRIOR_ZOOM_LOGIT, renderer_safe_values_from_raw_v3
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import DynamicCropHeadPredictionV3
from crosscut.stage_edit_decoded_curve_metrics import decoded_observation_error

try:
    import mlx.core as mx
except ImportError:
    mx = None


@unittest.skipIf(mx is None, 'Apple MLX is required')
class MaterializedPriorFitTests(unittest.TestCase):
    def setUp(self):
        points = (EditorialTransformKeyframe(0., 1.03, .5, .5, 0.),
                  EditorialTransformKeyframe(.3, 1.06, .5, .5, 0.),
                  EditorialTransformKeyframe(.7, 1.04, .5, .5, 0.))
        self.track = _DynamicCropV3PriorTrack(
            song_id='TRAIN-A', reference_id='train-ref', sequence_id='seq', token_index=1,
            source_id='observed-source', values=((1.03, .5, .5, 0.), (1.04, .5, .5, 0.)),
            normalized_times=(0., .7), motion_label='dynamic', motion_label_loss_masked=False,
            context_embedding=np.array([.2, .3], np.float32), source_embedding=np.array([-.1, .4], np.float32),
            original_observed_keyframes=points, shot_duration_seconds=1.)
        bias = np.zeros(76, np.float32)
        bias[:32].reshape(8, 4)[:, 0] = _PRIOR_ZOOM_LOGIT
        bias[33] = 5.  # Actual count2, while the original observation has3points.
        self.snapshot = dict(context_weight=mx.zeros((76, 2)), context_bias=mx.array(bias),
            source_weight=mx.zeros((76, 2)), context_center=mx.array([.01, .02]), source_center=mx.zeros(2))

    def evaluate(self, tracks=None):
        return evaluate_materialized_prior_fit(tracks or (self.track,), self.snapshot,
                                               reference_song_ids={'train-ref': 'TRAIN-A'})

    def test_real_original_clock_metric_does_not_score_only_retained_knots(self):
        result = self.evaluate()
        curve = result['decoded_observed_time']
        self.assertTrue(curve['complete'])
        self.assertEqual(curve['observed_keyframe_count'], 3)
        self.assertEqual(result['count_exact_accuracy'], 1.)
        self.assertEqual(result['encoder_forward_count'], 0)
        self.assertFalse(result['checkpoint_selection_or_qualification_authority'])
        self.assertFalse(result['validation_or_heldout_data_consumed'])
        self.assertEqual(curve, result['per_song_decoded_observed_time']['TRAIN-A'])
        self.assertAlmostEqual(curve['normalized_mae'], (0.01 + 0.04 + 0.02) / 3 / .18 / 4, delta=2e-6)

    def test_function_uses_the_actual_centered_merged_affine_expression(self):
        rng = np.random.default_rng(327)
        for name in ('context_weight', 'source_weight'):
            self.snapshot[name] = mx.array(rng.normal(0., .1, (76, 2)).astype(np.float32))
        head = _DynamicCropV3PriorHeadModule(
            self.snapshot['context_weight'], self.snapshot['context_bias'], self.snapshot['source_weight'],
            context_center=self.snapshot['context_center'], source_center=self.snapshot['source_center'])
        raw = np.asarray(head(mx.array(self.track.context_embedding), mx.array(self.track.source_embedding)))
        values = np.asarray(renderer_safe_values_from_raw_v3(mx.array(raw[:32].reshape(8, 4)))).astype(float)
        decoded = DynamicCropHeadPredictionV3(source_id=self.track.source_id, source_index=0,
            values=tuple(map(tuple, values)), count_logits=tuple(raw[32:40].astype(float)),
            time_interval_logits=tuple(raw[40:75].astype(float)), motion_logit=float(raw[75])).decode(1.)
        expected = decoded_observation_error(decoded, self.track.original_observed_keyframes,
                                             expected_source_id=self.track.source_id)
        self.assertAlmostEqual(self.evaluate()['decoded_observed_time']['normalized_mae'],
                               float(expected['normalized_absolute_error'].mean()), delta=2e-6)

    def test_read_only_diagnosis_preserves_rng_and_snapshot(self):
        before = {k: np.array(v) for k, v in self.snapshot.items()}
        mx.random.seed(714)
        expected = np.asarray(mx.random.normal((11,)))
        mx.random.seed(714)
        self.evaluate()
        np.testing.assert_array_equal(np.asarray(mx.random.normal((11,))), expected)
        for name, value in self.snapshot.items():
            np.testing.assert_array_equal(np.asarray(value), before[name])

    def test_foreign_reference_and_missing_original_observations_are_rejected(self):
        for track in (replace(self.track, reference_id='validation-ref'),
                      replace(self.track, song_id='HELDOUT'),
                      replace(self.track, original_observed_keyframes=()),
                      replace(self.track, context_embedding=None),
                      replace(self.track, shot_duration_seconds=0.)):
            with self.subTest(track=track):
                with self.assertRaisesRegex(ValueError, 'TRAIN'):
                    self.evaluate((track,))


if __name__ == '__main__':
    unittest.main()
