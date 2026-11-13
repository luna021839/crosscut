import copy
from types import SimpleNamespace as Item
import numpy as np
import pytest

mx=pytest.importorskip('mlx.core')
import mlx.optimizers as optim
from mlx.utils import tree_map
from crosscut.stage_edit_release_checks import (LIMITS,compare_drift,snapshot_transaction,restore_transaction,
    FEATURE_RELEASE_SCALES,apply_bounded_feature_release,ReleaseProposalRejected)
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


def _probe(model):
    value=float(model.base.source_projection.weight[0,0].item())
    return [dict(source_id='numeric-test-only',token_index=2,raw=np.full(76,value),
        value=np.full((8,4),value/10),features=np.full(32,value))]


def _proposal():
    model,*_=fixture();model.base.input_dropout.eval()
    optimizer=optim.AdamW(learning_rate=2.,weight_decay=0.)
    before=snapshot_transaction(model,optimizer);sources=_probe(model)
    # One genuine AdamW proposal. The synthetic probe is a numeric contract,
    # not real-data supervision, identity or a positive audiovisual label.
    mx.eval(mx.random.normal((7,)))
    optimizer.update(model,tree_map(mx.ones_like,model.trainable_parameters()))
    mx.eval(model.parameters(),optimizer.state)
    return model,optimizer,before,sources,snapshot_transaction(model,optimizer)


def test_finite_candidates_share_original_tree_heads_remain_full_and_optimizer_runs_once():
    from crosscut.stage_edit_production_training import DYNAMIC_CROP_V3_PRIOR_FEATURE_MODULES
    from mlx.utils import tree_flatten
    model,optimizer,before,sources,proposal=_proposal();calls=[]
    def capture():
        calls.append(_parameter_tree_sha256(model.parameters()))
        mx.eval(mx.random.normal((3,)))  # candidate forwards cannot advance training RNG
        return _probe(model)
    result=apply_bounded_feature_release(model,optimizer,before=before,before_sources=sources,capture=capture)
    assert FEATURE_RELEASE_SCALES==(1.,.1,.01,.001)
    assert result['accepted_scale']==.1 and len(calls)==2
    assert int(optimizer.state['step'].item())==1
    assert _parameter_tree_sha256(optimizer.state)==_parameter_tree_sha256(proposal['optimizer'])
    current=model.parameters()
    for name in DYNAMIC_CROP_V3_PRIOR_FEATURE_MODULES:
        expected=tree_map(lambda a,b:a+.1*(b-a),before['parameters']['base'][name],proposal['parameters']['base'][name])
        for (_,a),(_,b) in zip(tree_flatten(current['base'][name]),tree_flatten(expected),strict=True):
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
    for name in ('dynamic_crop_context_head','dynamic_crop_source_head'):
        assert _parameter_tree_sha256(current[name])==_parameter_tree_sha256(proposal['parameters'][name])
    for a,b in zip(mx.random.state,proposal['rng'],strict=True):
        np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
    assert tuple((n,m.training) for n,m in model.named_modules())==proposal['training_modes']


def test_all_four_rejections_restore_parameters_optimizer_rng_and_mixed_modes_exactly():
    model,optimizer,before,sources,_=_proposal()
    def capture():
        result=copy.deepcopy(sources);result[0]['raw']+=2
        return result
    with pytest.raises(ReleaseProposalRejected) as error:
        apply_bounded_feature_release(model,optimizer,before=before,before_sources=sources,capture=capture)
    assert [a['scale'] for a in error.value.receipt['attempts']]==list(FEATURE_RELEASE_SCALES)
    assert error.value.receipt['rollback_performed']
    assert _parameter_tree_sha256(model.parameters())==_parameter_tree_sha256(before['parameters'])
    assert _parameter_tree_sha256(optimizer.state)==_parameter_tree_sha256(before['optimizer'])
    for a,b in zip(mx.random.state,before['rng'],strict=True):
        np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
    assert tuple((n,m.training) for n,m in model.named_modules())==before['training_modes']


def test_deadline_or_nonfinite_candidate_rolls_back_instead_of_trying_more_scales():
    for reason in ('deadline','nonfinite'):
        model,optimizer,before,sources,_=_proposal();calls=[]
        def capture():
            calls.append(1)
            if reason=='deadline':raise TimeoutError('shared deadline')
            value=copy.deepcopy(sources);value[0]['raw'][0]=np.nan
            return value
        with pytest.raises((ValueError,TimeoutError)):
            apply_bounded_feature_release(model,optimizer,before=before,before_sources=sources,capture=capture)
        assert len(calls)==1
        assert _parameter_tree_sha256(model.parameters())==_parameter_tree_sha256(before['parameters'])
        assert _parameter_tree_sha256(optimizer.state)==_parameter_tree_sha256(before['optimizer'])
        assert tuple((n,m.training) for n,m in model.named_modules())==before['training_modes']
