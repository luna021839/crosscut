from unittest import mock
import numpy as np
import pytest

mx=pytest.importorskip('mlx.core')
import mlx.optimizers as optim
from mlx.utils import tree_map
from crosscut.stage_edit_gradient_checks import (separated_loss_gradients,feature_groups,norm,
    measured_feature_delta,require_actual_feature_update,row_supervision,head_row_norms)
from crosscut.stage_edit_dynamic_crop_head_v3 import StageEditTransformerDynamicCropV3,DynamicCropV3LossContract,_FixedCenteredLinear
from tests import test_stage_edit_dynamic_crop_head_v3 as fixtures


def fixture():
    helper=fixtures.StageEditDynamicCropHeadV3Tests();helper.setUp()
    batch,base,dynamic=helper._batches(fixtures._trajectory())
    mx.random.seed(1731);model=StageEditTransformerDynamicCropV3(helper.config)
    for name in ('dynamic_crop_context_head','dynamic_crop_source_head'):
        head=getattr(model,name);head.weight=.02*mx.random.normal(head.weight.shape)
    model.train()
    return model,batch,base,dynamic,DynamicCropV3LossContract()


def test_base_crop_combined_gradients_use_same_rng_and_crop_reaches_real_features():
    model,batch,base,dynamic,contract=fixture()
    loss,gradients,evidence=separated_loss_gradients(model,batch,base,dynamic,contract)
    assert evidence['crop_reaches_features']
    assert len({evidence[key]['matched_rng_sha256'] for key in ('base_only','crop_only','combined')})==1
    assert float(loss.item())==pytest.approx(evidence['combined']['loss'])
    assert all(value>0 for value in feature_groups(gradients).values())
    rows=head_row_norms(gradients);support=row_supervision(3)
    assert rows[support].max()>0 and not rows[~support].any()


def test_serving_mode_features_keep_real_encoder_gradients_and_rematerialized_updates():
    from crosscut.stage_edit_production_training import (
        _enable_dynamic_crop_v3_activation_checkpointing, _set_dynamic_crop_v3_joint_feature_mode,
    )

    model, batch, base, dynamic, contract = fixture()
    _enable_dynamic_crop_v3_activation_checkpointing(model)
    _set_dynamic_crop_v3_joint_feature_mode(model, 'serving')
    assert not model.training and not model.base.encoder.training
    assert model.base.encoder.checkpoint
    before = {name: tree_map(lambda value: value, model.base.parameters()[name])
              for name in ('encoder', 'source_projection')}
    loss, gradients, evidence = separated_loss_gradients(model, batch, base, dynamic, contract)
    assert evidence['crop_reaches_features']
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)
    deltas = {name: norm(value) for name, value in measured_feature_delta(model, before).items()}
    require_actual_feature_update(feature_groups(gradients), deltas)
    _set_dynamic_crop_v3_joint_feature_mode(model, 'stochastic')
    assert model.training and model.base.encoder.training
    assert model.base.encoder.checkpoint
    with pytest.raises(ValueError):
        _set_dynamic_crop_v3_joint_feature_mode(model, 'silent-freeze')


def test_crop_branch_stop_gradient_cannot_hide_behind_positive_base_loss_gradient():
    model,batch,base,dynamic,contract=fixture();records=[]
    original=_FixedCenteredLinear.__call__
    def detached(instance,value):return original(instance,mx.stop_gradient(value))
    with mock.patch.object(_FixedCenteredLinear,'__call__',detached):
        with pytest.raises(AssertionError,match='crop-only loss does not reach'):
            separated_loss_gradients(model,batch,base,dynamic,contract,record=records.append)
    assert len(records)==1
    assert all(value>0 for value in records[0]['base_only']['feature_gradient_l2'].values())
    assert all(value==0 for value in records[0]['crop_only']['feature_gradient_l2'].values())
    assert all(value>0 for value in records[0]['combined']['feature_gradient_l2'].values())


def test_real_feature_delta_and_noop_and_weight_decay_only_mutations():
    model,batch,base,dynamic,contract=fixture()
    loss,gradients,_=separated_loss_gradients(model,batch,base,dynamic,contract)
    def before():return {name:tree_map(lambda value:value,model.base.parameters()[name])
        for name in ('encoder','source_projection')}
    original=before();optimizer=optim.AdamW(learning_rate=1e-3,weight_decay=.1)
    optimizer.update(model,gradients);mx.eval(model.parameters(),optimizer.state)
    changed={name:norm(value) for name,value in measured_feature_delta(model,original).items()}
    require_actual_feature_update(feature_groups(gradients),changed)
    noop={name:norm(value) for name,value in measured_feature_delta(model,before()).items()}
    with pytest.raises(AssertionError,match='update missing'):
        require_actual_feature_update(feature_groups(gradients),noop)
    original=before();zero=tree_map(mx.zeros_like,gradients)
    decay=optim.AdamW(learning_rate=1e-3,weight_decay=.1)
    decay.update(model,zero);mx.eval(model.parameters(),decay.state)
    changed={name:norm(value) for name,value in measured_feature_delta(model,original).items()}
    assert all(value>0 for value in changed.values())
    with pytest.raises(AssertionError,match='gradient/update missing'):
        require_actual_feature_update(feature_groups(zero),changed)


def test_unsupervised_value_and_time_rows_are_explicit_but_all_count_logits_are_supervised():
    for count in range(1,7):
        mask=row_supervision(count)
        assert mask[:count*4].all() and not mask[24:32].any()
        assert mask[32:40].all() and not mask[60:75].any() and mask[75]
    assert not row_supervision(3,motion_masked=True)[75]
