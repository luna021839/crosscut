"""Regression for a frozen-value warmup that could never learn a crop curve.

These are isolated gradient/merge tests, not production quality evidence.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
import numpy as np
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError:
    mx = nn = optim = None

from crosscut.stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_MAX_KEYFRAMES,
    DynamicCropV3LossContract,
    StageEditTransformerDynamicCropV3,
    renderer_safe_values_from_raw_v3,
)
from crosscut.stage_edit_transformer import StageEditTransformerConfig, DEFAULT_STAGE_EDIT_SCHEMA
from crosscut.stage_edit_production_training import (
    _DynamicCropV3StructuredPriorAdapterModule,
    _DynamicCropV3PriorHeadModule,
    _DynamicCropV3PriorProjectionNormalizer,
    _dynamic_crop_v3_prior_head_snapshot,
    _dynamic_crop_v3_prior_loss_from_raw,
    _dynamic_crop_v3_prior_minibatch_loss,
    _dynamic_crop_v3_structured_prior_head_snapshot,
    _dynamic_crop_v3_structured_prior_raw,
)


@unittest.skipUnless(mx is not None, 'MLX is optional')
class TrajectoryPriorRegressionTests(unittest.TestCase):
    def test_minibatch_averages_each_original_ragged_loss_and_gradient_before_update(self):
        dimension=4;adapter=_DynamicCropV3StructuredPriorAdapterModule(2)
        initial=dict(context_weight=mx.zeros((76,dimension)),context_bias=mx.zeros(76),
            source_weight=mx.zeros((76,dimension)),context_center=mx.zeros(dimension),source_center=mx.zeros(dimension))
        normalizer=SimpleNamespace(context_mean=mx.zeros(2),source_mean=mx.zeros(2),
            context_scale=mx.ones(2),source_scale=mx.ones(2))
        projection=mx.eye(dimension)[:2];contract=DynamicCropV3LossContract()
        balance=dict(count_class_weights_1_to_8=[1.]*8,motion_class_weights_static_dynamic=[1.2,.8])
        tracks=[]
        for count,masked in ((1,False),(3,True)):
            values=np.asarray(renderer_safe_values_from_raw_v3(mx.full((count,4),.1)))
            tracks.append(SimpleNamespace(context_embedding=np.array([.2,.1,0.,0.]),
                source_embedding=np.array([-.1,.3,0.,0.]),values=values,
                normalized_times=np.arange(count)/count,motion_label='dynamic',motion_label_loss_masked=masked))
        def loss(module,batch):return _dynamic_crop_v3_prior_minibatch_loss(module,batch,initial,
            projection,projection,normalizer,contract,balance)
        total,parts=loss(adapter,tracks)
        individual=[loss(adapter,[t])[1] for t in tracks]
        for key in parts:
            self.assertAlmostEqual(float(parts[key].item()),sum(float(v[key].item()) for v in individual)/2,places=6)
        fn=nn.value_and_grad(adapter,lambda m,batch:loss(m,batch)[0])
        _,combined=fn(adapter,tracks);single=[fn(adapter,[t])[1] for t in tracks]
        for name in combined:
            for leaf in combined[name]:
                np.testing.assert_allclose(np.asarray(combined[name][leaf]),
                    (np.asarray(single[0][name][leaf])+np.asarray(single[1][name][leaf]))/2,atol=1e-6)
        self.assertTrue(np.isfinite(float(total.item())))

    def test_value_gradients_learn_source_conditioned_moving_path_and_merge_exactly(self):
        # Small dimension only isolates the same production prior functions.
        config = StageEditTransformerConfig.from_schema(
            DEFAULT_STAGE_EDIT_SCHEMA,
            model_dimension=16, num_heads=2, num_layers=1,
            feed_forward_dimension=32, dropout=0.0,
            max_sequence_length=8, max_sources=2,
        )
        with mx.stream(mx.cpu):
            model = StageEditTransformerDynamicCropV3(config)
            detached = _DynamicCropV3PriorHeadModule(
                model.dynamic_crop_context_head.weight,
                model.dynamic_crop_context_head.bias,
                model.dynamic_crop_source_head.weight,
            )
            initial = _dynamic_crop_v3_prior_head_snapshot(detached)
            initial_arrays = {key: np.array(value) for key, value in initial.items()}
            adapter = _DynamicCropV3StructuredPriorAdapterModule(2)
            projection = mx.eye(16)[:2]
            normalizer = _DynamicCropV3PriorProjectionNormalizer(
                context_mean=mx.array([0.2, -0.1]),
                context_scale=mx.array([0.5, 2.0]),
                source_mean=mx.array([-0.1, 0.2]),
                source_scale=mx.array([1.5, 0.7]),
                context_center=mx.zeros(16),source_center=mx.zeros(16),evidence={},
            )
            context = mx.array([0.4, 0.3] + [0.0] * 14)
            sources = [mx.array([sign, 0.2] + [0.0] * 14) for sign in (-1.0, 1.0)]
            targets = [renderer_safe_values_from_raw_v3(mx.array([
                [-1.5, sign * -0.2, 0.08, -0.05],
                [-1.0, sign * 0.1, -0.10, 0.08],
                [-1.2, sign * 0.25, 0.06, -0.04],
            ])) for sign in (-1.0, 1.0)]
            contract = DynamicCropV3LossContract(
                base_loss_weight=0.0, count_weight=0.0,
                time_weight=0.0, motion_weight=0.0,
            )

            def raw(module, source):
                return _dynamic_crop_v3_structured_prior_raw(
                    module, initial, projection, projection, normalizer,
                    context, source,
                )

            def loss(module):
                return sum(_dynamic_crop_v3_prior_loss_from_raw(
                    raw(module, source), target, mx.array([0.1, 0.5, 0.9]),
                    3, 1.0, True, contract,
                )[0] for source, target in zip(sources, targets)) / 2.0

            before = float(loss(adapter).item())
            value_and_grad = nn.value_and_grad(adapter, loss)
            _, gradients = value_and_grad(adapter)
            value_rows = DYNAMIC_CROP_V3_MAX_KEYFRAMES * 4
            for head in ('context_adapter', 'source_adapter'):
                numeric = np.asarray(gradients[head]['weight'][:value_rows])
                self.assertTrue(np.isfinite(numeric).all())
                self.assertGreater(float(np.abs(numeric).sum()), 0.0)
            optimizer = optim.Adam(learning_rate=0.02)
            for _ in range(180):
                current_loss, gradients = value_and_grad(adapter)
                optimizer.update(adapter, gradients)
                mx.eval(current_loss, adapter.parameters(), optimizer.state)
            self.assertLess(float(loss(adapter).item()), before * 0.15)

            snapshot = _dynamic_crop_v3_structured_prior_head_snapshot(
                adapter, initial, projection, projection, normalizer,
            )
            paths = []
            for source in sources:
                merged = (snapshot['context_weight'] @ context
                          + snapshot['context_bias']
                          + snapshot['source_weight'] @ source)
                np.testing.assert_allclose(np.asarray(merged), np.asarray(raw(adapter, source)),
                                           atol=2e-6, rtol=2e-6)
                values = np.asarray(renderer_safe_values_from_raw_v3(
                    merged[:value_rows].reshape((DYNAMIC_CROP_V3_MAX_KEYFRAMES, 4))
                ))[:3]
                self.assertGreater(float(np.ptp(values[:, 1])), 0.01)
                paths.append(values)
            self.assertLess(paths[0][-1, 1], paths[0][0, 1])
            self.assertGreater(paths[1][-1, 1], paths[1][0, 1])
            for key, expected in initial_arrays.items():
                np.testing.assert_array_equal(expected, np.asarray(initial[key]))


if __name__ == '__main__':
    unittest.main()
