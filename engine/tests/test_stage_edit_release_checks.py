import copy
from types import SimpleNamespace as Item
import numpy as np
import pytest

mx=pytest.importorskip('mlx.core')
import mlx.optimizers as optim
from mlx.utils import tree_map
from crosscut.stage_edit_release_checks import LIMITS,compare_drift,snapshot_transaction,restore_transaction
from crosscut.stage_edit_production_training import _parameter_tree_sha256
from tests.test_stage_edit_gradient_checks import fixture


def test_limits_preexist_and_changed_source_or_large_drift_cannot_pass():
    original=[dict(source_id='real',token_index=2,raw=np.zeros(76),value=np.zeros((8,4)),features=np.zeros(32))]
    assert compare_drift(original,original)['passed']
    for key,amount in (('raw',LIMITS['max_raw76_step_drift']+1),
            ('value',LIMITS['max_normalized_value_step_drift']+1),
            ('features',LIMITS['max_projected_train_sigma_step_drift']+1)):
        changed=copy.deepcopy(original);changed[0][key]+=amount
        assert not compare_drift(original,changed)['passed']
    changed=copy.deepcopy(original);changed[0]['source_id']='wrong'
    with pytest.raises(ValueError,match='source/token changed'):compare_drift(original,changed)


def test_model_optimizer_and_rng_rollback_reproduces_the_identical_next_update():
    model,*_=fixture();optimizer=optim.AdamW(learning_rate=1e-3,weight_decay=.01)
    gradients=tree_map(mx.ones_like,model.trainable_parameters())
    optimizer.update(model,gradients);mx.eval(model.parameters(),optimizer.state)
    before=snapshot_transaction(model,optimizer)
    before_rng=[np.array(k) for k in mx.random.state]
    optimizer.update(model,gradients);mx.eval(model.parameters(),optimizer.state)
    expected_model=_parameter_tree_sha256(model.parameters());expected_optimizer=_parameter_tree_sha256(optimizer.state)
    mx.random.normal((3,))
    restore_transaction(model,optimizer,before)
    for actual,expected in zip(mx.random.state,before_rng):np.testing.assert_array_equal(np.asarray(actual),expected)
    optimizer.update(model,gradients);mx.eval(model.parameters(),optimizer.state)
    assert _parameter_tree_sha256(model.parameters())==expected_model
    assert _parameter_tree_sha256(optimizer.state)==expected_optimizer
