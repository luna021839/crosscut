"""Adoption instrument mutations; not evidence of a fitted production model."""
from dataclasses import replace
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
mx=pytest.importorskip('mlx.core')

from crosscut.stage_edit_adoption_checks import adopt_and_verify,native_source_raw,verify_public_outputs
from crosscut.stage_edit_dynamic_crop_head_v3 import (
    StageEditTransformerDynamicCropV3,new_stage_edit_dynamic_crop_checkpoint_metadata_v3,
    save_stage_edit_dynamic_crop_checkpoint_v3,load_stage_edit_dynamic_crop_checkpoint_v3)
from crosscut.stage_edit_production_training import _assign_dynamic_crop_v3_prior_head_snapshot
from crosscut.stage_edit_transformer import smoke_stage_edit_config,tensorize_stage_edit_batch
from tests.test_stage_edit_dynamic_crop_head_v3 import _permuted_sequence
from tests.test_stage_edit_transformer import SCHEMA


def material():
    config=smoke_stage_edit_config(SCHEMA)
    mx.random.seed(1731)
    model=StageEditTransformerDynamicCropV3(config);model.eval()
    batch=tensorize_stage_edit_batch((_permuted_sequence(reverse=False),),SCHEMA,config)
    rng=np.random.default_rng(29);d=config.model_dimension
    snapshot=dict(context_weight=mx.array(rng.normal(0,.02,(76,d)).astype(np.float32)),
        context_bias=mx.array(rng.normal(0,.1,76).astype(np.float32)),
        source_weight=mx.array(rng.normal(0,.04,(76,d)).astype(np.float32)),
        context_center=mx.array(rng.normal(0,.1,d).astype(np.float32)),
        source_center=mx.array(rng.normal(0,.1,d).astype(np.float32)))
    hidden,_,sources,_=model.base._hidden_and_sources(batch)
    h,e=np.asarray(hidden[0,1]),np.asarray(sources[0,1,batch.source_ids[0][1].index('C')])
    # Independent NumPy affine expression, not the native/public probe itself.
    raw=(np.asarray(snapshot['context_weight'])@(h-np.asarray(snapshot['context_center']))
        +np.asarray(snapshot['context_bias'])
        +np.asarray(snapshot['source_weight'])@(e-np.asarray(snapshot['source_center'])))
    probe=dict(batch=batch,token_index=1,source_id='C',expected_raw=raw,
        duration_seconds=1.,observed_times=(0.,.2,.7,1.))
    return config,model,snapshot,probe


def test_real_native_rank_public_outputs_and_nonzero_center_reload():
    config,model,snapshot,probe=material()
    evidence=adopt_and_verify(model,snapshot,[probe])
    assert evidence['passed'] and evidence['track_count']==1
    raw,*_=native_source_raw(model,probe['batch'],1,'C')
    np.testing.assert_allclose(np.asarray(raw),probe['expected_raw'],atol=2e-4,rtol=2e-5)
    with tempfile.TemporaryDirectory() as folder:
        metadata=new_stage_edit_dynamic_crop_checkpoint_metadata_v3(
            'adoption-test',SCHEMA,config,training_steps=1,weights_trained=True)
        save_stage_edit_dynamic_crop_checkpoint_v3(Path(folder),model,metadata)
        restored,_=load_stage_edit_dynamic_crop_checkpoint_v3(Path(folder))
        assert verify_public_outputs(restored,[probe])['passed']


@pytest.mark.parametrize('mutation',['no_op','partial_rows','wrong_source_weight'])
def test_assignment_mutations_fail_at_public_adoption_and_preserve_all_probe_evidence(mutation):
    _,model,snapshot,probe=material();records=[]
    def broken(target,candidate):
        if mutation=='no_op':return
        candidate=dict(candidate)
        if mutation=='partial_rows':
            for name in ('context_weight','context_bias','source_weight'):
                value=candidate[name]
                candidate[name]=mx.concatenate((value[:32],mx.zeros_like(value[32:])),axis=0)
        else:candidate['source_weight']=mx.zeros_like(candidate['source_weight'])
        _assign_dynamic_crop_v3_prior_head_snapshot(target,candidate)
    with pytest.raises(AssertionError,match='public-output candidate adoption'):
        adopt_and_verify(model,snapshot,[probe,dict(probe,source_id='missing')],assign=broken,record=records.append)
    assert len(records)==1 and len(records[0]['tracks'])==2
    assert not any(row['passed'] for row in records[0]['tracks'])


def test_wrong_real_source_slot_and_hard_gated_source_cannot_pass():
    _,model,snapshot,probe=material()
    _assign_dynamic_crop_v3_prior_head_snapshot(model,snapshot)
    assert not verify_public_outputs(model,[dict(probe,source_id='A')])['passed']
    assert not verify_public_outputs(model,[dict(probe,source_id='B')])['passed']
    with pytest.raises(ValueError,match='hard-gated'):
        native_source_raw(model,probe['batch'],1,'B')


def test_mutated_public_source_axis_is_detected_even_when_private_heads_are_correct():
    _,model,snapshot,probe=material()
    _assign_dynamic_crop_v3_prior_head_snapshot(model,snapshot)
    original=StageEditTransformerDynamicCropV3.__call__
    def switched(instance,batch):
        output=original(instance,batch)
        return replace(output,dynamic_crop_values=mx.roll(output.dynamic_crop_values,1,axis=2))
    with mock.patch.object(StageEditTransformerDynamicCropV3,'__call__',switched):
        assert not verify_public_outputs(model,[probe])['passed']


def test_source_permutation_padding_and_present_hard_gate_perturbations_preserve_target():
    config,model,snapshot,probe=material()
    _assign_dynamic_crop_v3_prior_head_snapshot(model,snapshot)
    batch=tensorize_stage_edit_batch((_permuted_sequence(reverse=True),),SCHEMA,config)
    assert verify_public_outputs(model,[dict(probe,batch=batch)])['passed']
    original=probe['batch'];gated=np.asarray(original.source_candidate_mask)&~np.asarray(original.hard_source_gate_mask)
    assert gated.any() and np.asarray(original.source_candidate_mask).any(axis=0).any()
    for mask in (gated,~np.asarray(original.source_candidate_mask)):
        changed=replace(original,
            source_features=mx.where(mx.array(mask)[...,None],12345.,original.source_features),
            temporal_source_features=None if original.temporal_source_features is None else
                mx.where(mx.array(mask)[...,None],12345.,original.temporal_source_features))
        assert verify_public_outputs(model,[dict(probe,batch=changed)])['passed']
