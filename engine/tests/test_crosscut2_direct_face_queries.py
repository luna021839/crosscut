import hashlib
import numpy as np
import pytest

from crosscut.native_face_descriptor import ARCFACE_R50_SHA256
from crosscut2.direct_face_queries import (query_key,descriptor_family,validate_values,
    read_snapshot,publish_snapshot,fields,bind_banks,bind_manifest,refresh_member_scores)


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


def test_bound_bank_recovers_actual_current_queries_without_touching_old_observations(tmp_path):
    old=tmp_path/'old';current=tmp_path/'current';old.mkdir();current.mkdir()
    head=np.array([.5,.4,.1,.99],np.float32)
    bank=dict(heads=np.tile(head,(2,2,1,1)),native_indices=np.array([[10,11],[20,21]]))
    original=bank['heads'].copy();record=binding();producer=dict(record,contract='actual_native_face_pair_queries_v2')
    publish_snapshot(current/'one.npz',{query_key(11,head):(np.eye(512,dtype=np.float32)[0],True)},binding=producer)
    result=bind_banks([bank],[record],source_hashes=['1'*64],legacy_cache=old,cache=current)
    assert result['sources'][0]['matched_current_face_slots']==1 and result['query_recovery_NN']==0
    assert bank['direct_face_known'][0,1,0] and not bank['direct_face_known'][0,0,0]
    np.testing.assert_array_equal(bank['heads'],original)
    wrong=dict(heads=original.copy(),native_indices=bank['native_indices'])
    with pytest.raises(ValueError,match='physical source'):
        bind_banks([wrong],[record],source_hashes=['f'*64],legacy_cache=old,cache=current)
    assert 'direct_face_vectors' not in wrong


def test_member_profiles_use_exact_current_queries_and_retain_unknown_cluster_hints():
    from crosscut.ive import IVEMember
    from crosscut.ive_local_arcface_gallery import LocalIVEArcFaceGallery
    from crosscut2.source_member_scores import MemberScoreSeeds,bank_signature
    basis=np.eye(512,dtype=np.float32)
    gallery=LocalIVEArcFaceGallery(tuple((member,(tuple(map(float,basis[m])),))
        for m,member in enumerate(IVEMember)),('a'*64,))
    head=np.array([.5,.4,.1,.99],np.float32);record=binding()
    bank=dict(heads=np.tile(head,(2,2,1,1)),native_indices=np.array([[10,11],[20,21]]),
        vectors=np.tile(np.eye(128,dtype=np.float32)[0],(2,2,1,1)))
    old=MemberScoreSeeds(np.array([-1],np.int16),np.empty((0,128)),np.empty((0,6)),
        bank_signature(bank),gallery.identity)
    values={query_key(11,head):(basis[2],True)}
    bank['direct_face_vectors'],bank['direct_face_known']=fields(bank,values)
    bank['direct_face_descriptor_family']=descriptor_family(record)
    before=bank['heads'].copy()
    seeds,receipt=refresh_member_scores([bank],[record],[old],gallery=gallery)
    assert seeds[0].cluster_hints.tolist()==[-1]
    assert len(seeds[0].direct_queries)==1 and seeds[0].direct_queries[0][0]==11
    scores,support,direct=seeds[0].fields(bank['heads'],bank['vectors'],native_indices=bank['native_indices'])
    assert direct.sum()==1 and direct[0,1,0] and scores[0,1,0,2]==pytest.approx(1.)
    assert support[0,1,0]==pytest.approx(1.)
    np.testing.assert_array_equal(before,bank['heads'])
    assert receipt['source_NN']==0 and not receipt['source_exclusion']
    bank['direct_face_descriptor_family']='f'*64
    with pytest.raises(ValueError,match='bound actual'):
        refresh_member_scores([bank],[record],[old],gallery=gallery)


def test_cli_manifest_rejects_another_physical_source_or_clock_before_attaching(tmp_path):
    import json
    from crosscut2.grid_perception import file_identity
    source=tmp_path/'source.mp4';source.write_bytes(b'original')
    old=tmp_path/'old';current=tmp_path/'current';old.mkdir();current.mkdir()
    nodes=np.array([0,30],np.int64);head=np.array([.5,.4,.1,.99],np.float32)
    def bank():return dict(heads=np.tile(head,(2,2,1,1)),native_indices=np.array([[10,11],[20,21]]))
    manifest=tmp_path/'manifest.json'
    record=dict(contract='all_preselection_actor_proposal_faces_on_fixed_source_grid_v1',
        source_paths=[str(source)],source_file_identities=[list(file_identity(source))],
        source_sha256=['1'*64],bindings=[binding()],grid_nodes_sha256=hashlib.sha256(nodes.tobytes()).hexdigest(),
        output_fps=30.,master_start=0.,original_node_count=2)
    manifest.write_text(json.dumps(record))
    kwargs=dict(nodes=nodes,sources=[source],source_hashes=['1'*64],master_start=0.,fps=30.,
        authorized_root=tmp_path,legacy_cache=old,cache=current)
    banks=[bank()];bindings,receipt=bind_manifest(manifest,banks=banks,**kwargs)
    assert bindings==[binding()] and receipt['query_recovery_NN']==0
    assert 'direct_face_vectors' in banks[0]
    for changed in (dict(kwargs,fps=25.),dict(kwargs,source_hashes=['f'*64]),
                    dict(kwargs,nodes=np.array([0,31],np.int64))):
        banks=[bank()]
        with pytest.raises(ValueError,match='physical source, clock or candidate grid'):
            bind_manifest(manifest,banks=banks,**changed)
        assert 'direct_face_vectors' not in banks[0]
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='physical source, clock or candidate grid'):
        bind_manifest(manifest,banks=[bank()],**kwargs)
