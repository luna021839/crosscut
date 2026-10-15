import json
import numpy as np

from crosscut2.recurrent_member_hints import recover_query_snapshots


def save(path,binding,*,value=1.):
    vector=np.zeros((1,512),np.float32);vector[0,0]=value
    np.savez_compressed(path,binding=json.dumps(binding),keys=np.array(['k']),vectors=vector,valid=np.array([True]))


def test_old_runtime_queries_reuse_same_source_model_inputs_without_rebinding_or_writes(tmp_path):
    old=dict(source_sha256='source',model='model',adapter='adapter',ort='old',opencv='old')
    path=tmp_path/'old.npz';save(path,old);before=path.read_bytes()
    current=dict(old,ort='new',opencv='new')
    values,receipt=recover_query_snapshots(tmp_path,current)
    assert values['k'][1] and receipt['parent_snapshots'][0]['producer_ort']=='old'
    assert not receipt['new_inference_performed']
    assert path.read_bytes()==before and len(list(tmp_path.iterdir()))==1


def test_model_source_or_preprocess_change_does_not_reuse_prior_runtime_queries(tmp_path):
    original=dict(source_sha256='source',model='model',adapter='adapter',ort='old',opencv='old')
    save(tmp_path/'old.npz',original)
    for key in ('source_sha256','model','adapter'):
        values,_=recover_query_snapshots(tmp_path,dict(original,**{key:'different'}))
        assert not values


def test_conflicting_prior_runtime_values_remain_ambiguous_not_favourable_member_votes(tmp_path):
    binding=dict(source_sha256='source',model='model',adapter='adapter',ort='old',opencv='old')
    save(tmp_path/'a.npz',binding);save(tmp_path/'b.npz',dict(binding,ort='other'),value=-1.)
    values,receipt=recover_query_snapshots(tmp_path,dict(binding,ort='now'))
    assert not values and receipt['ambiguous_query_inputs']==1


def test_incomplete_checkpoint_and_symlink_are_not_query_snapshots(tmp_path):
    binding=dict(source_sha256='source',model='model',adapter='adapter')
    (tmp_path/'.member-query-interrupted.npz').write_bytes(b'incomplete')
    (tmp_path/'external.npz').symlink_to(tmp_path/'.member-query-interrupted.npz')
    values,receipt=recover_query_snapshots(tmp_path,binding)
    assert not values and not receipt['parent_snapshots']
