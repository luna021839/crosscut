"""Actual opt-in MLX crop-expression contracts, not trained/movie quality."""
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
import mlx.nn as nn
from mlx.utils import tree_flatten
from crosscut.stage_edit_dynamic_crop_head_v3 import (
    DynamicCropHeadV3Config, DYNAMIC_CROP_V3_INTERACTION_SOURCE_CONDITIONING,
    DYNAMIC_CROP_V3_INTERACTION_HEAD_REPRESENTATION,
    StageEditTransformerDynamicCropV3, StageEditDynamicCropCheckpointMetadataV3,
    stage_edit_dynamic_crop_v3_parameter_count, new_stage_edit_dynamic_crop_checkpoint_metadata_v3,
    save_stage_edit_dynamic_crop_checkpoint_v3, load_stage_edit_dynamic_crop_checkpoint_v3)
from crosscut.stage_edit_adoption_checks import native_source_raw, verify_public_outputs
from crosscut.stage_edit_production_training import (
    _assign_dynamic_crop_v3_prior_head_snapshot, _dynamic_crop_v3_prior_parameter_digests)
from crosscut.stage_edit_gradient_checks import head_row_norms
from crosscut.stage_edit_transformer import smoke_stage_edit_config, tensorize_stage_edit_batch, stage_edit_readiness_report
from tests.test_stage_edit_transformer import SCHEMA
from tests.test_stage_edit_dynamic_crop_head_v3 import _permuted_sequence
from tests.test_stage_edit_adoption_checks import material


HEAD = DynamicCropHeadV3Config(source_conditioning=DYNAMIC_CROP_V3_INTERACTION_SOURCE_CONDITIONING)


def actual():
    config, legacy, snapshot, probe = material()
    model = StageEditTransformerDynamicCropV3(config, HEAD)
    # Explicit opt-in transfer for this diagnostic, not checkpoint reinterpretation.
    model.base.update(legacy.base.parameters())
    snapshot = dict(snapshot, interaction_weight=mx.zeros_like(model.dynamic_crop_interaction_head.weight))
    _assign_dynamic_crop_v3_prior_head_snapshot(model, snapshot)
    model.eval()
    return config, legacy, model, snapshot, probe


def test_opt_in_and_zero_initialization_preserve_legacy_real_public_outputs_exactly():
    config, legacy, model, snapshot, probe = actual()
    _assign_dynamic_crop_v3_prior_head_snapshot(legacy, {k:v for k,v in snapshot.items() if k!='interaction_weight'})
    a,b = legacy(probe['batch']),model(probe['batch'])
    for name in ('dynamic_crop_values','dynamic_crop_count_logits','dynamic_crop_time_interval_logits',
                 'dynamic_crop_motion_logits','source_selection_logits','cut_timing_logits'):
        np.testing.assert_array_equal(np.asarray(getattr(a,name)), np.asarray(getattr(b,name)))
    legacy_count=stage_edit_dynamic_crop_v3_parameter_count(config)
    count=stage_edit_dynamic_crop_v3_parameter_count(config,HEAD)
    assert count-legacy_count == 76*config.model_dimension
    assert count==sum(int(np.prod(v.shape)) for _,v in tree_flatten(model.parameters()))
    assert DynamicCropHeadV3Config().checkpoint_format_version==2 and HEAD.checkpoint_format_version==3
    assert not model.checkpoint_contract['production_activation_authorized']


def test_actual_raw_head_can_reverse_source_motion_order_with_context_only():
    config=smoke_stage_edit_config(SCHEMA)
    model=StageEditTransformerDynamicCropV3(config,HEAD)
    w=np.zeros(model.dynamic_crop_interaction_head.weight.shape,np.float32);w[75,0]=1.
    model.dynamic_crop_interaction_head.weight=mx.array(w)
    hidden=np.zeros((2,1,config.model_dimension),np.float32);hidden[:,0,0]=[1.,-1.]
    sources=np.zeros((2,1,2,config.model_dimension),np.float32);sources[:,:,0,0]=1.;sources[:,:,1,0]=-1.
    motion=np.asarray(model.dynamic_crop_raw(mx.array(hidden),mx.array(sources)))[...,75]
    np.testing.assert_array_equal(motion[:,0],[[1.,-1.],[-1.,1.]])
    # This is an expression test, NOT a real training/AV target.


def test_nonzero_interaction_public_native_source_decode_and_permutation_are_consistent():
    config,_,model,snapshot,probe=actual()
    rng=np.random.default_rng(831)
    w=rng.normal(0,.01,model.dynamic_crop_interaction_head.weight.shape).astype(np.float32)
    model.dynamic_crop_interaction_head.weight=mx.array(w)
    raw,h,s,slot=native_source_raw(model,probe['batch'],probe['token_index'],probe['source_id'])
    expected=probe['expected_raw']+w@((np.asarray(h)-np.asarray(snapshot['context_center']))*
                                   (np.asarray(s)-np.asarray(snapshot['source_center'])))
    np.testing.assert_allclose(np.asarray(raw),expected,atol=2e-4,rtol=2e-5)
    p=dict(probe,expected_raw=expected)
    assert verify_public_outputs(model,[p])['passed']
    perm=tensorize_stage_edit_batch((_permuted_sequence(reverse=True),),SCHEMA,config)
    assert verify_public_outputs(model,[dict(p,batch=perm)])['passed']
    original=probe['batch']
    for mask in (~np.asarray(original.source_candidate_mask),
                 np.asarray(original.source_candidate_mask)&~np.asarray(original.hard_source_gate_mask)):
        changed=replace(original,source_features=mx.where(mx.array(mask)[...,None],12345.,original.source_features),
            temporal_source_features=None if original.temporal_source_features is None else
                mx.where(mx.array(mask)[...,None],12345.,original.temporal_source_features))
        assert verify_public_outputs(model,[dict(p,batch=changed)])['passed']
    with pytest.raises(ValueError,match='hard-gated'):
        native_source_raw(model,original,probe['token_index'],'B')


def test_interaction_gradient_reaches_both_feature_inputs_and_row_instruments():
    config=smoke_stage_edit_config(SCHEMA);model=StageEditTransformerDynamicCropV3(config,HEAD)
    h=mx.ones((1,1,config.model_dimension));s=mx.ones((1,1,2,config.model_dimension))
    # Trainable interaction starts at zero but receives a nonzero real gradient.
    _,g=nn.value_and_grad(model,lambda m:mx.square(m.dynamic_crop_raw(h,s)[...,75]-1.).mean())(model)
    assert float(mx.abs(g['dynamic_crop_interaction_head']['weight'][75]).sum())>0
    assert head_row_norms(g)[75]>0
    w=np.zeros(model.dynamic_crop_interaction_head.weight.shape,np.float32);w[75,:]=.1
    model.dynamic_crop_interaction_head.weight=mx.array(w)
    gh=mx.grad(lambda v:model.dynamic_crop_raw(v,s)[...,75].sum())(h)
    gs=mx.grad(lambda v:model.dynamic_crop_raw(h,v)[...,75].sum())(s)
    assert float(mx.abs(gh).sum())>0 and float(mx.abs(gs).sum())>0


def test_interaction_snapshot_is_complete_atomic_and_hash_bound():
    _,legacy,model,snapshot,_=actual()
    before=_dynamic_crop_v3_prior_parameter_digests(model)
    missing={k:v for k,v in snapshot.items() if k!='interaction_weight'}
    with pytest.raises(ValueError,match='interaction snapshot weight is missing'):
        _assign_dynamic_crop_v3_prior_head_snapshot(model,missing)
    with pytest.raises(ValueError,match='cannot be assigned'):
        _assign_dynamic_crop_v3_prior_head_snapshot(legacy,snapshot)
    with pytest.raises(ValueError,match='interaction_weight is invalid'):
        _assign_dynamic_crop_v3_prior_head_snapshot(model,dict(snapshot,interaction_weight=mx.zeros((1,1))))
    assert _dynamic_crop_v3_prior_parameter_digests(model)==before
    _assign_dynamic_crop_v3_prior_head_snapshot(model,dict(snapshot,interaction_weight=snapshot['interaction_weight']+.01))
    assert _dynamic_crop_v3_prior_parameter_digests(model)['dynamic_head_parameter_tree_sha256']!=before['dynamic_head_parameter_tree_sha256']


def test_new_expression_checkpoint_cannot_be_silently_reinterpreted_and_reload_is_exact(tmp_path):
    config,_,model,snapshot,probe=actual()
    model.dynamic_crop_interaction_head.weight=mx.ones_like(model.dynamic_crop_interaction_head.weight)*.002
    raw,*_=native_source_raw(model,probe['batch'],probe['token_index'],probe['source_id'])
    metadata=new_stage_edit_dynamic_crop_checkpoint_metadata_v3('interaction-smoke',SCHEMA,config,
        training_steps=1,weights_trained=True,dynamic_crop_config=HEAD)
    assert metadata.format_version==3 and metadata.affine_head_representation==DYNAMIC_CROP_V3_INTERACTION_HEAD_REPRESENTATION
    saved=save_stage_edit_dynamic_crop_checkpoint_v3(tmp_path,model,metadata)
    loaded,restored=load_stage_edit_dynamic_crop_checkpoint_v3(tmp_path)
    loaded.eval();actual_raw,*_=native_source_raw(loaded,probe['batch'],probe['token_index'],probe['source_id'])
    np.testing.assert_array_equal(np.asarray(actual_raw),np.asarray(raw))
    assert restored.contract_fingerprint_sha256==saved.contract_fingerprint_sha256
    assert verify_public_outputs(loaded,[dict(probe,expected_raw=np.asarray(raw))])['passed']
    assert not stage_edit_readiness_report(tmp_path).learned_inference_ready
    payload=saved.to_dict()
    for bad in (dict(payload,format_version=2),dict(payload,affine_head_representation='fixed_train_centered_affine_full_rank_v1'),
                dict(payload,source_conditioning=DynamicCropHeadV3Config().source_conditioning)):
        with pytest.raises(ValueError,match='format|contract'):
            StageEditDynamicCropCheckpointMetadataV3.from_dict(bad)
    original_path=tmp_path/saved.weight_filename
    # Do not truncate a live MLX-mapped safetensors file: both loaded models
    # still own the original arrays. Corrupt an independent derived file.
    weights={key:mx.array(np.array(value,copy=True)) for key,value in mx.load(str(original_path)).items()}
    weights.pop('dynamic_crop_interaction_head.weight')
    path=tmp_path/'missing-interaction.safetensors'
    mx.save_safetensors(str(path),weights)
    payload['weight_filename']=path.name
    payload['weight_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'dynamic-crop-v3-metadata.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError):load_stage_edit_dynamic_crop_checkpoint_v3(tmp_path)
