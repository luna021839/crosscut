"""Actual MLX train/serve temporal input regression, not quality acceptance."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA as SCHEMA, SourceCandidateToken,
    StageEditToken, StageEditSequence, StageEditTemporalPoseSidecar,
    StageEditTransformer, StageEditTransformerConfig, smoke_stage_edit_config,
    STAGE_EDIT_TEMPORAL_INPUT_MODE,
    STAGE_EDIT_TEMPORAL_POSE_VISUAL_CACHE_CURRENT_NEXT_CONTRACT as NEXT,
    STAGE_EDIT_TEMPORAL_POSE_TIMELINE_ENTRY_EXIT_CONTRACT as EXIT,
    stage_edit_parameter_count, tensorize_stage_edit_batch,
    stage_edit_source_scoring_contract, stage_edit_cut_scoring_contract,
)
from crosscut.stage_edit_dynamic_crop_head_v3 import (
    StageEditTransformerDynamicCropV3,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v3,
    save_stage_edit_dynamic_crop_checkpoint_v3,
    load_stage_edit_dynamic_crop_checkpoint_v3,
)


def pair(sign=1.0, interval=1/30, contract=NEXT):
    return StageEditTemporalPoseSidecar(
        embeddings=((0.1,) * 60, (sign * 0.2,) * 60),
        interval_seconds=interval, sampling_contract=contract,
    )


def sequence(sidecar=None, times=(10.0, 10.1, 12.0)):
    candidates = tuple(SourceCandidateToken(
        source_id=name, features=(0.2,) * SCHEMA.raw_source_dimension,
        temporal_pose_sidecar=sidecar if name == 'a' else pair(-1),
    ) for name in ('a', 'b'))
    return StageEditSequence('s', tuple(StageEditToken(
        master_time_seconds=t,
        music_features=(0.,) * len(SCHEMA.music_feature_names),
        section_features=(0.,) * len(SCHEMA.section_feature_names),
        source_candidates=candidates,
        recent_edit_features=(0.,) * len(SCHEMA.recent_edit_feature_names),
        style_features=(0.,) * len(SCHEMA.style_feature_names),
    ) for t in times))


class TemporalInputTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(42)
        self.legacy = smoke_stage_edit_config()
        self.config = replace(self.legacy, config_version=4,
                              temporal_input_mode=STAGE_EDIT_TEMPORAL_INPUT_MODE)

    def batch(self, seq, config=None):
        return tensorize_stage_edit_batch((seq,), SCHEMA, config or self.config, device=mx.cpu)

    def test_requires_explicit_new_config_and_roundtrips(self):
        with self.assertRaises(ValueError):
            replace(self.legacy, temporal_input_mode=STAGE_EDIT_TEMPORAL_INPUT_MODE)
        with self.assertRaises(ValueError):
            replace(self.legacy, config_version=4)
        self.assertEqual(self.config, StageEditTransformerConfig.from_dict(self.config.to_dict()))
        raw = self.legacy.to_dict()
        self.assertNotIn('temporal_input_mode', raw)
        self.assertEqual(self.legacy, StageEditTransformerConfig.from_dict(raw))

    def test_exact_pair_velocity_sampling_kind_missing_and_gap(self):
        batch = self.batch(sequence(pair()))
        values = np.asarray(batch.temporal_source_features)[0, 0, 0]
        np.testing.assert_allclose(values[:60], .1)
        np.testing.assert_allclose(values[60:120], .2)
        np.testing.assert_allclose(values[120:180], .75)
        np.testing.assert_allclose(values[-4:], [np.log1p(1/30), 1, 0, 1])
        np.testing.assert_allclose(np.asarray(batch.temporal_context_features)[0],
                                   [[0, 0], [np.log1p(.1), 1], [np.log1p(1.9), 1]], atol=1e-6)
        missing = np.asarray(self.batch(sequence()).temporal_source_features)
        self.assertTrue((missing[0, :, 0] == 0).all())
        other = np.asarray(self.batch(sequence(pair(contract=EXIT))).temporal_source_features)
        np.testing.assert_allclose(other[0, 0, 0, -3:], [0, 1, 1])
        tiny = np.asarray(self.batch(sequence(pair(interval=1e-300))).temporal_source_features)
        self.assertTrue(np.isfinite(tiny).all())

    def test_absolute_clock_translation_and_future_token_do_not_change_prefix(self):
        original = self.batch(sequence(pair()))
        shifted = self.batch(sequence(pair(), times=(110, 110.1, 112)))
        np.testing.assert_allclose(np.asarray(original.temporal_context_features),
                                   np.asarray(shifted.temporal_context_features), atol=1e-6)
        changed = self.batch(sequence(pair(), times=(10, 10.1, 99)))
        with mx.stream(mx.cpu):
            model = StageEditTransformer(self.config)
            model.eval()
            a, b = model(original), model(changed)
            np.testing.assert_allclose(np.asarray(a.source_selection_logits)[:, :2],
                                       np.asarray(b.source_selection_logits)[:, :2], atol=1e-6)

    def test_legacy_excludes_sidecars_and_rejects_mixed_batches(self):
        a = self.batch(sequence(), self.legacy)
        b = self.batch(sequence(pair()), self.legacy)
        self.assertIsNone(a.temporal_source_features)
        np.testing.assert_array_equal(np.asarray(a.source_features), np.asarray(b.source_features))
        with mx.stream(mx.cpu):
            old = StageEditTransformer(self.legacy)
            new = StageEditTransformer(self.config)
            with self.assertRaises(ValueError):
                old(self.batch(sequence(pair())))
            with self.assertRaises(ValueError):
                new(a)

    def test_permutation_equivariance_and_hard_gate_isolation(self):
        seq = sequence(pair())
        perm = replace(seq, tokens=tuple(replace(t, source_candidates=t.source_candidates[::-1]) for t in seq.tokens))
        with mx.stream(mx.cpu):
            model = StageEditTransformer(self.config)
            model.eval()
            a, b = model(self.batch(seq)), model(self.batch(perm))
            np.testing.assert_allclose(np.asarray(a.source_selection_logits)[..., :2],
                                       np.asarray(b.source_selection_logits)[..., :2][..., ::-1], atol=2e-5)
            np.testing.assert_allclose(np.asarray(a.cut_timing_logits), np.asarray(b.cut_timing_logits), atol=2e-5)
            def gated(sidecar):
                return replace(seq, tokens=tuple(replace(t, source_candidates=(
                    t.source_candidates[0], replace(t.source_candidates[1],
                    hard_gate_passed=False, temporal_pose_sidecar=sidecar))) for t in seq.tokens))
            a, b = model(self.batch(gated(pair()))), model(self.batch(gated(pair(-1, 2.))))
            np.testing.assert_array_equal(np.asarray(a.cut_timing_logits), np.asarray(b.cut_timing_logits))
            np.testing.assert_array_equal(np.asarray(a.crop), np.asarray(b.crop))

    def test_real_forward_gradients_parameter_count_and_dynamic_crop_path(self):
        with mx.stream(mx.cpu):
            model = StageEditTransformer(self.config)
            batch = self.batch(sequence(pair()))
            count = sum(v.size for _, v in tree_flatten(model.parameters()))
            self.assertEqual(stage_edit_parameter_count(self.config), count)
            def loss(m):
                return mx.square(m(batch).source_selection_logits[..., :2]).mean()
            _, grads = nn.value_and_grad(model, loss)(model)
            # Temporal channels are at the end of every raw relational block.
            temporal_grad = np.asarray(grads['source_projection']['weight'])[:, SCHEMA.encoded_source_dimension:self.config.model_source_input_dimension]
            self.assertGreater(float(np.abs(temporal_grad).sum()), 0.)
            dynamic = StageEditTransformerDynamicCropV3(self.config)
            out = dynamic(batch)
            self.assertTrue(np.isfinite(np.asarray(out.dynamic_crop_values)).all())
            source_contract = stage_edit_source_scoring_contract(self.config)
            cut_contract = stage_edit_cut_scoring_contract(self.config)
            self.assertEqual(4*self.config.model_source_input_dimension, source_contract['relational_source_input_dimension'])
            self.assertEqual(model.cut_linear_baseline.weight.shape[1], cut_contract['linear_input_dimension'])
            self.assertTrue(source_contract['temporal_input_contract']['enabled'])
            self.assertTrue(cut_contract['temporal_input_contract']['enabled'])

    def test_real_checkpoint_roundtrip_preserves_temporal_predictions(self):
        with mx.stream(mx.cpu), tempfile.TemporaryDirectory() as directory:
            model = StageEditTransformerDynamicCropV3(self.config)
            model.eval()
            metadata = new_stage_edit_dynamic_crop_checkpoint_metadata_v3(
                'temporal-test-only', SCHEMA, self.config, training_steps=0,
                weights_trained=False,
            )
            save_stage_edit_dynamic_crop_checkpoint_v3(Path(directory), model, metadata)
            loaded, restored = load_stage_edit_dynamic_crop_checkpoint_v3(
                Path(directory), expected_schema=SCHEMA, require_trained=False,
            )
            loaded.eval()
            self.assertEqual(self.config, restored.base_model_config)
            batch = self.batch(sequence(pair()))
            a, b = model(batch), loaded(batch)
            np.testing.assert_array_equal(np.asarray(a.source_selection_logits), np.asarray(b.source_selection_logits))
            np.testing.assert_array_equal(np.asarray(a.dynamic_crop_values), np.asarray(b.dynamic_crop_values))
            self.assertFalse(restored.production_activation_authorized)


if __name__ == '__main__':
    unittest.main()
