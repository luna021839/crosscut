import numpy as np
import pytest
from crosscut2.sparse_face_completion import requests,apply,grid_requests


def inputs():
    from test_crosscut2_fine_subject_grid import inputs as original
    nodes,banks,poses,styles,states,*_=original()
    for bank in banks:
        bank['face_observation_valid']=np.ones((3,2),bool)
        bank['face_observation_valid'][1]=False
        bank['heads'][1]=0.;bank['vectors'][1]=0.
    return nodes,banks,poses,styles,states


def row(source=0,native=99):
    heads=np.zeros((8,4),np.float32);heads[0]=[.5,.3,.15,.99]
    return dict(source=source,native_frame=native,source_sha256='a'*64,raster_sha256='b'*64,
        heads=heads,vectors=np.zeros((3,128)),native_readability=np.zeros((8,4)),
        native_readability_valid=np.zeros(8,bool),colour=np.zeros((8,2,40)),
        colour_valid=np.zeros(8,bool),presence=np.zeros(2))


def test_requests_target_missing_selected_endpoints_once_not_observed_empty_frames():
    nodes,banks,*_=inputs();seam=dict(output_frame=100,left_source=0,right_source=1,played_geometry_rate=1.)
    wanted,receipt=requests([seam,seam],nodes,banks,maximum_frames=4)
    assert wanted==[dict(source=s,native_frame=k) for s in range(2) for k in (99,100)]
    assert receipt['selected_centres']==[100] and receipt['bounded_single_pass']
    banks[0]['face_observation_valid'][1,0]=True
    assert dict(source=0,native_frame=99) not in requests([seam],nodes,banks)[0]
    assert requests([seam],nodes,banks,maximum_frames=1)[0]==[]
    for bank in banks:bank['face_observation_valid'][:]=True
    assert requests([seam],nodes,banks)[0]==[]


def test_complete_grid_requests_missing_alternatives_even_if_selected_sources_are_observed():
    nodes,banks,*_=inputs()
    banks[0]['face_observation_valid'][:]=True
    # A known selected face must not conceal unobserved alternative sources.
    wanted,receipt=grid_requests(nodes,banks)
    assert wanted==[dict(source=1,native_frame=k) for k in (99,100)]
    assert not receipt['selected_source_sequence_used']
    assert receipt['sources'][0]['unique_native_frames']==0
    # A sampled empty original viewport is already observed, not a new probe.
    banks[1]['face_observation_valid'][:]=True
    assert grid_requests(nodes,banks)[0]==[]


def test_complete_grid_deduplicates_native_aliases_and_cannot_truncate_to_a_budget():
    nodes,banks,*_=inputs()
    banks[0]['native_indices'][1]=99
    wanted,receipt=grid_requests(nodes,banks,maximum_frames=3)
    assert len(wanted)==3 and receipt['actual_frame_requests']==3
    with pytest.raises(ValueError,match='needs 3 frames'):
        grid_requests(nodes,banks,maximum_frames=2)
    for bad in (True,-1,8193):
        with pytest.raises(ValueError,match='allowance'):grid_requests(nodes,banks,maximum_frames=bad)


def test_larger_explicit_grid_merge_preserves_small_default_budget():
    _,banks,poses,styles,states=inputs()
    rows=[row(native=k) for k in range(129)]
    with pytest.raises(ValueError,match='Bounded'):
        apply(banks,poses,styles,states,rows,source_hashes=['a'*64]*2)
    bs,_,_,_,receipt=apply(banks,poses,styles,states,rows,source_hashes=['a'*64]*2,maximum_records=129)
    assert receipt['source_bound_native_records']==129
    assert bs[0]['face_observation_valid'][1].all()


def test_completion_never_attaches_old_body_to_a_new_face_or_changes_observed_input():
    _,banks,poses,styles,states=inputs();poses[0][1,:2]=9.
    old_heads=banks[0]['heads'].copy();old_pose=poses[0].copy()
    bs,ps,ss,st,receipt=apply(banks,poses,styles,states,[row()],source_hashes=['a'*64]*2)
    assert receipt['new_observed_endpoints']==1
    assert bs[0]['face_observation_valid'][1,0]
    np.testing.assert_array_equal(bs[0]['heads'][1,0],row()['heads'])
    assert not ps[0][1,:2].any() and not st[0][1,:2].any()
    np.testing.assert_array_equal(bs[0]['heads'][[0,2]],old_heads[[0,2]])
    np.testing.assert_array_equal(banks[0]['heads'],old_heads)
    np.testing.assert_array_equal(poses[0],old_pose)
    assert not receipt['old_input_arrays_rewritten']


def test_actual_observed_empty_endpoint_is_preserved_not_filled_by_a_new_hypothesis():
    _,banks,poses,styles,states=inputs();banks[0]['face_observation_valid'][1,0]=True
    bs,_,_,_,receipt=apply(banks,poses,styles,states,[row()],source_hashes=['a'*64]*2)
    assert not bs[0]['heads'][1,0].any()
    assert receipt['new_observed_endpoints']==0 and receipt['already_observed_endpoints_preserved']==1


def test_source_binding_duplicate_and_conflicting_unobserved_slot_are_not_silently_repaired():
    _,banks,poses,styles,states=inputs()
    with pytest.raises(ValueError,match='binding'):
        apply(banks,poses,styles,states,[row()],source_hashes=['c'*64]*2)
    with pytest.raises(ValueError,match='Duplicate'):
        apply(banks,poses,styles,states,[row(),row()],source_hashes=['a'*64]*2)
    banks[0]['heads'][1,0,0]=[.5,.3,.2,.99]
    with pytest.raises(ValueError,match='conflicting'):
        apply(banks,poses,styles,states,[row()],source_hashes=['a'*64]*2)


def test_no_body_bank_is_still_unknown_after_a_face_only_completion():
    _,banks,_,styles,_=inputs()
    _,poses,_,states,_=apply(banks,None,styles,None,[row()],source_hashes=['a'*64]*2)
    assert all(not p.any() for p in poses) and all(not s.any() for s in states)


def test_old_member_occurrences_remain_exact_and_new_faces_do_not_borrow_an_old_label():
    from crosscut2.fixed_holds import local_actor_hypotheses
    from crosscut2.fine_subject_grid import rebind_hints
    _,banks,poses,styles,states=inputs()
    hints=[np.zeros(local_actor_hypotheses(b)[1],np.int16) for b in banks]
    bs,*_=apply(banks,poses,styles,states,[row()],source_hashes=['a'*64]*2)
    with pytest.raises(ValueError,match='original physical'):
        rebind_hints(banks,bs,hints)
    result=rebind_hints(banks,bs,hints,allow_previously_unobserved=True)
    labels,_=local_actor_hypotheses(bs[0]);new_actor=int(labels[1,0,0])
    assert new_actor==-1  # No observed appearance supports an old actor label.
    assert result[0][int(labels[0,0,0])]==0
    bs[0]['heads'][0,0,0,0]+=.1
    with pytest.raises(ValueError,match='original physical'):
        rebind_hints(banks,bs,hints,allow_previously_unobserved=True)


def test_completion_cli_rejects_an_unrelated_editor_path_before_opening_media(monkeypatch):
    from crosscut2.stage_mix import main
    monkeypatch.setattr('sys.argv',['stage_mix','--source-plan','unused','--analysis','unused',
        '--cache','unused','--output','unused','--models','unused','--face-endpoint-observations','unused'])
    with pytest.raises(ValueError,match='actual played-pose'):main()


@pytest.mark.parametrize('extra',[['--face-endpoint-record-limit','512'],
    ['--face-endpoint-observations','unused','--played-pose-port-camera-prior','--face-endpoint-record-limit','8193']])
def test_larger_cli_face_allowance_is_explicit_and_bounded_before_any_source_io(monkeypatch,extra):
    from crosscut2.stage_mix import main
    monkeypatch.setattr('sys.argv',['stage_mix','--source-plan','unused','--analysis','unused',
        '--cache','unused','--output','unused','--models','unused',*extra])
    with pytest.raises(ValueError,match='face-record allowance'):main()


def test_explicit_cached_reference_validates_source_and_native_binding_without_media_models(tmp_path):
    import json
    from crosscut2.grid_perception import file_identity
    from crosscut2.sparse_face_completion import load
    source=tmp_path/'source.mp4';source.write_bytes(b'fixture-not-decoded')
    identity=list(file_identity(source));data=row();producer=dict(detector='fixture')
    binding=json.dumps(dict(source_sha256='a'*64,source_identity=identity,native_frame=99,producer=producer),sort_keys=True)
    cache=tmp_path/'endpoint.npz'
    with cache.open('xb') as f:np.savez_compressed(f,binding=binding,**data)
    record=dict(sources=[str(source)],source_sha256=['a'*64],source_file_identities=[identity],
        fps=30.,master_audio_start=0.,records=[dict(source=0,native_frame=99,path=str(cache))],producer=producer,
        used=dict(face=1,pose=0,recognition=0,raster=1),elapsed_seconds=.1)
    reference=tmp_path/'reference.json';reference.write_text(json.dumps(record))
    loaded,receipt=load(reference,sources=[source],source_hashes=['a'*64],master_start=0.,fps=30.,authorized_root=tmp_path)
    np.testing.assert_array_equal(loaded[0]['heads'],data['heads'])
    assert receipt['records_loaded']==1 and not receipt['new_NN_or_decode_in_loader']
    with pytest.raises(ValueError,match='domain changed'):
        load(reference,sources=[source],source_hashes=['c'*64],master_start=0.,fps=30.,authorized_root=tmp_path)
    record['records'][0]['native_frame']=100;reference.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='producer/ordinal changed'):
        load(reference,sources=[source],source_hashes=['a'*64],master_start=0.,fps=30.,authorized_root=tmp_path)


def test_separate_completed_batches_union_with_their_actual_producers_and_no_duplicate_work(tmp_path):
    import json
    from crosscut2.grid_perception import file_identity
    from crosscut2.sparse_face_completion import load
    source=tmp_path/'source.mp4';source.write_bytes(b'not-decoded')
    identity=list(file_identity(source))
    def reference(name,native,*,conflict=False):
        data=row(native=native);producer=dict(actual_producer=name)
        if conflict:data['heads'][0,0]+=.1
        binding=json.dumps(dict(source_sha256='a'*64,source_identity=identity,native_frame=native,
            producer=producer),sort_keys=True)
        cache=tmp_path/(name+'.npz')
        with cache.open('xb') as f:np.savez_compressed(f,binding=binding,**data)
        path=tmp_path/(name+'.json')
        path.write_text(json.dumps(dict(sources=[str(source)],source_sha256=['a'*64],source_file_identities=[identity],
            fps=30.,master_audio_start=0.,records=[dict(source=0,native_frame=native,path=str(cache))],
            producer=producer,used=dict(face=1),elapsed_seconds=.1)))
        return path
    a,b=reference('actual-first',99),reference('actual-complement',100)
    kw=dict(sources=[source],source_hashes=['a'*64],master_start=0.,fps=30.,authorized_root=tmp_path)
    rows,receipt=load([a,b,a],**kw)
    assert [r['native_frame'] for r in rows]==[99,100]
    assert len(receipt['source_references'])==2 and not receipt['new_NN_or_decode_in_loader']
    with pytest.raises(ValueError,match='Combined'):load([a,b],maximum_records=1,**kw)
    with pytest.raises(ValueError,match='Conflicting'):
        load([a,reference('conflicting-observation',99,conflict=True)],**kw)
