import hashlib
import numpy as np
import pytest

from crosscut.native_face_descriptor import ARCFACE_R50_SHA256
from crosscut2.direct_face_queries import (query_key,descriptor_family,validate_values,
    read_snapshot,publish_snapshot,fields)


def binding():
    return dict(model=ARCFACE_R50_SHA256,raster=[960,540],source_sha256='1'*64,
        adapter='2'*64,detector='3'*64,association='4'*64,decoder_implementation='5'*64,
        ort='original-producer',opencv='original-producer')


def test_query_address_is_the_original_native_and_rectangle_not_a_close_neighbour():
    head=np.array([.5,.4,.1,.99],np.float32)
    assert query_key(12,head)==hashlib.sha256(np.array([12,*head],np.float64).tobytes()).hexdigest()
    assert query_key(12,head)!=query_key(13,head)
    changed=head.copy();changed[0]+=1e-5
    assert query_key(12,head)!=query_key(12,changed)


def test_snapshot_preserves_its_producer_and_512D_space(tmp_path):
    vector=np.eye(512,dtype=np.float32)[0];key='1'*64
    path=tmp_path/'source.npz';record=binding()
    publish_snapshot(path,{key:(vector,True)},binding=record)
    recovered=read_snapshot(path,binding=record)
    np.testing.assert_array_equal(recovered[key][0],vector)
    assert not recovered[key][0].flags.writeable
    with pytest.raises(ValueError,match='producer'):
        read_snapshot(path,binding=dict(record,ort='different-runtime'))
    assert descriptor_family(record)==descriptor_family(dict(record,source_sha256='f'*64,ort='different-runtime'))
    with pytest.raises(ValueError,match='pinned'):
        descriptor_family(dict(record,model='0'*64))


def test_invalid_and_missing_current_queries_do_not_turn_into_positives():
    head=np.array([.5,.4,.1,.99],np.float32)
    bank=dict(heads=np.tile(head,(2,2,3,1)),native_indices=np.array([[10,11],[20,21]]))
    vector=np.eye(512,dtype=np.float32)[3]
    value={query_key(11,head):(vector,True),query_key(20,head):(np.zeros(512,np.float32),False)}
    before=bank['heads'].copy();result,known=fields(bank,value)
    assert known.sum()==3 and known[0,1].all()
    assert not known[1,0].any() and not result[1,0].any()
    assert not result.flags.writeable and not known.flags.writeable
    np.testing.assert_array_equal(bank['heads'],before)


@pytest.mark.parametrize('error',['wrong-family','nan','not-unit','failed-nonzero','duplicate'])
def test_malformed_snapshots_are_rejected_before_selection(error):
    keys=np.array(['a'*64]);vectors=np.eye(512,dtype=np.float32)[:1];valid=np.array([True])
    if error=='wrong-family':vectors=np.eye(128,dtype=np.float32)[:1]
    if error=='nan':vectors[0,0]=np.nan
    if error=='not-unit':vectors*=2.
    if error=='failed-nonzero':valid[:]=False
    if error=='duplicate':keys=np.repeat(keys,2);vectors=np.repeat(vectors,2,axis=0);valid=np.repeat(valid,2)
    with pytest.raises(ValueError):validate_values(keys,vectors,valid)
