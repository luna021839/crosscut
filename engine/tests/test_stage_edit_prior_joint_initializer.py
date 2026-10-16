"""A TRAIN-only warm start cannot qualify a checkpoint or change split gates."""
from dataclasses import replace
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from crosscut.stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_MAX_KEYFRAMES, DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION,
    DynamicCropV3LossContract, renderer_safe_values_from_raw_v3,
)
from crosscut.stage_edit_production_training import (
    _DynamicCropV3PriorInventory, _DynamicCropV3PriorTrack,
    _dynamic_crop_v3_train_only_joint_initializer,
)


def fixture():
    values = tuple(tuple(map(float, row)) for row in
        np.asarray(renderer_safe_values_from_raw_v3(mx.zeros((2, 4)))))
    tracks = tuple(_DynamicCropV3PriorTrack(
        song_id=song, reference_id=ref, sequence_id=ref, token_index=0,
        source_id='s', values=values, normalized_times=(0., 1.),
        motion_label='static', motion_label_loss_masked=False,
        context_embedding=np.ones(4, np.float32), source_embedding=np.zeros(4, np.float32),
    ) for song, ref in [('a', 'a1'), ('a', 'a2'), ('b', 'b1')])
    inventory = _DynamicCropV3PriorInventory(tracks,
        {t.reference_id: 1 for t in tracks}, (), {t.reference_id: t.song_id for t in tracks}, 'detached-train')
    initial = dict(context_weight=mx.zeros((DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION, 4)),
        context_bias=mx.zeros(DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION),
        source_weight=mx.zeros((DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION, 4)),
        context_center=mx.zeros(4), source_center=mx.zeros(4))
    bias = np.zeros(DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION, np.float32)
    bias[DYNAMIC_CROP_V3_MAX_KEYFRAMES*4+1] = 6.  # actual target count2
    bias[-1] = -6.  # actual supervised static motion
    learned = dict(initial, context_bias=mx.array(bias))
    balance = dict(count_class_weights_1_to_8=[1.]*8, motion_class_weights_static_dynamic=[1., 1.])
    return inventory, {0: initial, 50: learned, 500: learned}, balance


def test_best_complete_train_loss_is_only_a_joint_initializer_and_ties_choose_earlier_step():
    inventory, snapshots, balance = fixture()
    step, receipt = _dynamic_crop_v3_train_only_joint_initializer(inventory, snapshots, DynamicCropV3LossContract(), balance)
    assert step == 50
    assert receipt['selected_training_loss'] < receipt['initial_training_loss']
    assert receipt['training_trajectory_count'] == 3
    assert receipt['training_reference_ids'] == ['a1', 'a2', 'b1']
    assert not receipt['validation_or_heldout_inputs_used']
    assert not receipt['checkpoint_qualification_or_retention_authorized']
    assert not receipt['gradients_computed'] and receipt['new_encoder_forwards'] == 0
    assert receipt['evaluations'][1]['training_loss'] == receipt['evaluations'][2]['training_loss']


def test_an_unchanged_snapshot_cannot_replace_the_initial_head():
    inventory, snapshots, balance = fixture()
    step, _ = _dynamic_crop_v3_train_only_joint_initializer(
        inventory, {0: snapshots[0], 50: snapshots[0]}, DynamicCropV3LossContract(), balance)
    assert step == 0


def test_repeating_one_reference_does_not_change_hierarchical_train_comparison():
    inventory, snapshots, balance = fixture()
    # Give b1 a different actual count so references have unequal losses.
    last = replace(inventory.tracks[-1], values=inventory.tracks[-1].values[:1], normalized_times=(0.,))
    inventory = replace(inventory, tracks=(*inventory.tracks[:-1], last))
    _, original = _dynamic_crop_v3_train_only_joint_initializer(inventory, snapshots, DynamicCropV3LossContract(), balance)
    duplicated = replace(inventory, tracks=(*inventory.tracks, *([last]*9)),
        per_reference_track_counts={**inventory.per_reference_track_counts, 'b1': 10})
    _, repeated = _dynamic_crop_v3_train_only_joint_initializer(duplicated, snapshots, DynamicCropV3LossContract(), balance)
    np.testing.assert_allclose([r['training_loss'] for r in original['evaluations']],
        [r['training_loss'] for r in repeated['evaluations']], rtol=1e-6)


def test_nonfinite_learned_snapshot_is_not_an_initializer():
    inventory, snapshots, balance = fixture()
    broken = dict(snapshots[50], context_bias=mx.full((DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION,), float('nan')))
    step, receipt = _dynamic_crop_v3_train_only_joint_initializer(
        inventory, {0: snapshots[0], 50: broken}, DynamicCropV3LossContract(), balance)
    assert step == 0 and receipt['evaluations'][1]['training_loss'] is None


def test_missing_detached_features_or_initial_head_are_rejected():
    inventory, snapshots, balance = fixture()
    with pytest.raises(ValueError, match='initial head'):
        _dynamic_crop_v3_train_only_joint_initializer(inventory, {50: snapshots[50]}, DynamicCropV3LossContract(), balance)
    inventory = replace(inventory, tracks=(replace(inventory.tracks[0], context_embedding=None),))
    with pytest.raises(ValueError, match='already detached'):
        _dynamic_crop_v3_train_only_joint_initializer(inventory, snapshots, DynamicCropV3LossContract(), balance)
