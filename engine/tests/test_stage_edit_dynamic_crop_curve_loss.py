from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from crosscut.stage_edit_dynamic_crop_head_v3 import (
    DynamicCropV3LossContract,
    DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION,
    renderer_safe_values_from_raw_v3,
    dynamic_crop_v3_times_by_count,
    _original_observation_curve_loss,
)
from crosscut.stage_edit_dynamic_crop_curve_loss import decoded_curve_risk
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import DynamicCropHeadPredictionV3
from crosscut.stage_edit_production_training import _dynamic_crop_v3_prior_loss_from_raw

try:
    import mlx.core as mx
except ImportError:
    mx = None


@unittest.skipIf(mx is None, "Apple MLX is required")
class DynamicCropDecodedCurveLossTests(unittest.TestCase):
    def test_every_conditional_curve_agrees_with_the_real_renderer_decoder(self) -> None:
        rng = np.random.default_rng(41179)
        for _ in range(5):
            values = renderer_safe_values_from_raw_v3(mx.array(rng.normal(0., 2., (8, 4)).astype(np.float32)))
            raw_times = rng.normal(0., 2., 35).astype(np.float32)
            times = dynamic_crop_v3_times_by_count(mx.array(raw_times))
            observed = np.array([0., .013, .2, .7, 1.], np.float32)
            target = np.tile([1.02, .5, .5, 0.], (len(observed), 1))
            _, conditional = decoded_curve_risk(values, times, mx.zeros(8), mx.array(observed), mx.array(target))
            for count in range(1, 9):
                logits = np.full(8, -20.)
                logits[count - 1] = 20.
                decoded = DynamicCropHeadPredictionV3(
                    source_id='actual-source', source_index=2,
                    values=tuple(map(tuple, np.asarray(values).astype(float))),
                    count_logits=tuple(logits), time_interval_logits=tuple(raw_times.astype(float)), motion_logit=0.,
                ).decode(3.7)
                self.assertTrue(decoded.eligible)
                self.assertEqual(decoded.minimum_safe_zoom_projection_count, 0)
                actual = np.array([[p.zoom, p.center_x, p.center_y, p.roll_degrees]
                                   for p in [decoded.track.evaluate(float(t) * 3.7) for t in observed]])
                expected = np.abs((actual - target) / [.18, .04, .04, 1.60]).mean()
                self.assertAlmostEqual(float(conditional[count - 1].item()), float(expected), delta=2e-6)

    def test_count_time_and_value_gradients_are_finite_and_nonzero(self) -> None:
        rng = np.random.default_rng(392)
        raw_values = mx.array(rng.normal(0., .3, (8, 4)).astype(np.float32))
        raw_times = mx.array(rng.normal(0., .4, 35).astype(np.float32))
        logits = mx.array(rng.normal(0., .3, 8).astype(np.float32))
        times = mx.array([0., .13, .32, .8, 1.])
        targets = mx.array([[1.02, .5, .5, 0.]] * 5)
        def loss(value, clock, count):
            return decoded_curve_risk(renderer_safe_values_from_raw_v3(value),
                                      dynamic_crop_v3_times_by_count(clock), count, times, targets)[0]
        gradients = mx.grad(loss, argnums=(0, 1, 2))(raw_values, raw_times, logits)
        for gradient in gradients:
            actual = np.asarray(gradient)
            self.assertTrue(np.isfinite(actual).all())
            self.assertGreater(float(np.abs(actual).sum()), 0.)

    def test_masked_original_points_do_not_change_the_risk(self) -> None:
        values = renderer_safe_values_from_raw_v3(mx.zeros((2, 8, 4)))
        times = dynamic_crop_v3_times_by_count(mx.zeros((2, 35)))
        targets = mx.array([[[1.02, .5, .5, 0.], [1.02, .5, .5, 0.]],
                            [[1.02, .5, .5, 0.], [100., 100., 100., 100.]]])
        risk, _ = decoded_curve_risk(values, times, mx.zeros((2, 8)),
                                    mx.array([[0., 1.], [0., 1.]]), targets,
                                    mx.array([[True, True], [True, False]]))
        expected = np.abs((np.asarray(values[0, 0]) - [1.02, .5, .5, 0.])
                          / [.18, .04, .04, 1.60]).mean()
        np.testing.assert_allclose(np.asarray(risk), [expected, expected], atol=1e-6)

    def test_joint_loss_gathers_only_the_target_source_and_requires_original_points(self) -> None:
        from tests import test_stage_edit_dynamic_crop_head_v3 as fixtures
        from crosscut.stage_edit_dynamic_crop_head_v3 import StageEditTransformerDynamicCropV3
        helper = fixtures.StageEditDynamicCropHeadV3Tests()
        helper.setUp()
        batch, _, target = helper._batches(fixtures._trajectory())
        model = StageEditTransformerDynamicCropV3(helper.config)
        model.eval()
        output = model(batch)
        before = _original_observation_curve_loss(output, target)
        values = np.array(output.dynamic_crop_values)
        values[0, 0, 0, :, 0] = 1.18
        changed_other_source = replace(output, dynamic_crop_values=mx.array(values))
        self.assertEqual(float(before.item()), float(_original_observation_curve_loss(changed_other_source, target).item()))
        with self.assertRaisesRegex(ValueError, 'original observed'):
            _original_observation_curve_loss(output, replace(target, original_observed_keyframes=None))

    def test_wrong_count_curves_do_not_alias_in_the_real_prior_loss(self) -> None:
        # Both predictions assign exactly the same probability to target
        # count2. Index-wise coordinates/times therefore have identical old
        # losses, while the actual count8 curve exposes six zoomed tail knots.
        raw = np.zeros(DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION, np.float32)
        zooms = np.array([1.02, 1.06] + [1.17] * 6)
        floor = float(renderer_safe_values_from_raw_v3(mx.array([-80., 0., 0., 0.]))[0].item())
        amount = (zooms - floor) / (1.18 - floor)
        raw[:32].reshape(8, 4)[:, 0] = np.log(amount / (1. - amount))
        raw[32:40] = -8.
        raw[33] = 0.
        good, bad = raw.copy(), raw.copy()
        good[32], bad[39] = 8., 8.
        targets = mx.array([[1.02, .5, .5, 0.], [1.06, .5, .5, 0.]])
        arguments = (targets, mx.array([0., .5]), 2, 0., False,
                     DynamicCropV3LossContract(base_loss_weight=0.))
        good_loss, good_parts = _dynamic_crop_v3_prior_loss_from_raw(mx.array(good), *arguments)
        bad_loss, bad_parts = _dynamic_crop_v3_prior_loss_from_raw(mx.array(bad), *arguments)
        for name in ("values_normalized", "count", "times", "motion"):
            self.assertAlmostEqual(float(good_parts[name].item()), float(bad_parts[name].item()), places=6)
        self.assertGreater(float(bad_loss.item()), float(good_loss.item()) + .01)


if __name__ == "__main__":
    unittest.main()
