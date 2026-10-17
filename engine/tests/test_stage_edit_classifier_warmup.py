from __future__ import annotations

import unittest

import numpy as np

from crosscut.stage_edit_classifier_warmup import begin_classifier_warmup, finish_classifier_warmup
from crosscut.stage_edit_dynamic_crop_head_v3 import StageEditTransformerDynamicCropV3, DynamicCropV3LossContract, stage_edit_dynamic_crop_v3_loss
from crosscut.stage_edit_production_training import _cut_only_pretraining_loss_contract, _source_only_pretraining_loss_contract
from crosscut.stage_edit_transformer import StageEditLossContract, stage_edit_loss
from crosscut.stage_edit_gradient_checks import feature_groups, measured_feature_delta, norm

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map, tree_flatten
except ImportError:
    mx = None


@unittest.skipIf(mx is None, 'Apple MLX is required')
class ClassifierWarmupTests(unittest.TestCase):
    def setUp(self):
        from tests import test_stage_edit_dynamic_crop_head_v3 as fixtures
        helper = fixtures.StageEditDynamicCropHeadV3Tests()
        helper.setUp()
        self.batch, self.base_targets, self.dynamic = helper._batches(fixtures._trajectory())
        self.model = StageEditTransformerDynamicCropV3(helper.config)
        self.model.eval()

    def test_real_classifier_optimizer_preserves_features_then_full_joint_updates_them(self):
        before = np.asarray(self.model.base._hidden_and_sources(self.batch)[0]).copy()
        base_contract = StageEditLossContract()
        for phase, selected_contract in (
            ('cut', _cut_only_pretraining_loss_contract(base_contract)),
            ('source', _source_only_pretraining_loss_contract(base_contract)),
        ):
            state = begin_classifier_warmup(self.model, phase)
            optimizer = optim.AdamW(learning_rate=.01, weight_decay=.1)
            grad_fn = nn.value_and_grad(self.model,
                lambda m: stage_edit_loss(m, self.batch, self.base_targets, selected_contract)[0])
            for _ in range(3):
                loss, gradients = grad_fn(self.model)
                gradient_keys = {name for name, _ in tree_flatten(gradients)}
                self.assertFalse(any(name.startswith('base.encoder.') for name in gradient_keys))
                self.assertFalse(any(name.startswith('base.source_projection.') for name in gradient_keys))
                optimizer.update(self.model, gradients)
                mx.eval(loss, self.model.parameters(), optimizer.state)
            result = finish_classifier_warmup(self.model, state)
            self.assertTrue(result['all_nonclassifier_parameters_byte_identical'])
            self.assertTrue(result['classifier_parameters_changed'])
            self.assertTrue(result['full_joint_trainability_restored'])
            self.assertEqual(result['frozen_parameter_sha256_before'], result['frozen_parameter_sha256_after'])
            np.testing.assert_array_equal(np.asarray(self.model.base._hidden_and_sources(self.batch)[0]), before)
        weights_before = {name: tree_map(lambda a: mx.array(np.array(a, copy=True)), self.model.base.parameters()[name])
                          for name in ('encoder', 'source_projection')}
        loss, gradients = nn.value_and_grad(self.model, lambda m: stage_edit_dynamic_crop_v3_loss(
            m, self.batch, self.base_targets, self.dynamic, DynamicCropV3LossContract())[0])(self.model)
        for value in feature_groups(gradients).values():
            self.assertGreater(value, 0.)
        optimizer = optim.AdamW(learning_rate=.001, weight_decay=0.)
        optimizer.update(self.model, gradients)
        mx.eval(loss, self.model.parameters(), optimizer.state)
        for delta in measured_feature_delta(self.model, weights_before).values():
            self.assertGreater(norm(delta), 0.)

    def test_frozen_parameter_tampering_is_detected_and_trainability_restored(self):
        original_keys = set(name for name, _ in tree_flatten(self.model.trainable_parameters()))
        state = begin_classifier_warmup(self.model, 'source')
        self.model.base.context_projection.weight = self.model.base.context_projection.weight + .001
        with self.assertRaisesRegex(RuntimeError, 'frozen'):
            finish_classifier_warmup(self.model, state)
        self.assertEqual(set(name for name, _ in tree_flatten(self.model.trainable_parameters())), original_keys)

    def test_unknown_phase_and_preexisting_freeze_are_not_reinterpreted(self):
        with self.assertRaisesRegex(ValueError, 'phase'):
            begin_classifier_warmup(self.model, 'joint')
        self.model.base.encoder.freeze()
        with self.assertRaisesRegex(ValueError, 'fully trainable'):
            begin_classifier_warmup(self.model, 'cut')


if __name__ == '__main__':
    unittest.main()
