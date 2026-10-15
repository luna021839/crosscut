from dataclasses import replace
import numpy as np
import pytest
from crosscut2.dense_subject_catalogue import DenseSubjects,OFFSETS
from crosscut2.subject_window_bank import SubjectWindow
from crosscut2.played_partial_pose import features,matrix
from test_crosscut2_source_clock_action import catalogue,set_style,obs


def prepared(source=0,*,bump=0.,missing=False,cut=()):
    cat=catalogue()
    indices=np.maximum(0,np.minimum(199,np.array([0,100,200])[:,None]+OFFSETS))
    cat.bank['pose_native_indices']=indices[:,[0,6,7,13]]
    cat.bank['native_indices']=indices[:,[6,7]]
    cat.native=cat.bank['pose_native_indices']
    ordinals=np.arange(86,115);N=len(ordinals)
    body=np.tile(cat.body[1,1,0],(N,1,1))
    body[ordinals==101,4:6,0]+=bump
    support=np.ones((N,8),bool)
    if missing:support[:,2:]=False
    vectors=np.zeros((N,128));vectors[:,0]=1.
    window=SubjectWindow(source,str(source),'actual-track',ordinals,
        np.tile(cat.bank['heads'][1,0,0],(N,1)),vectors,body,support,
        np.ones(N,bool),tuple(np.zeros((0,4)) for _ in range(N)),cut)
    cat.subjects=DenseSubjects(source=source,binding=str(source),indices=indices,windows=[window],bank=cat.bank,fps=30.)
    set_style(cat,source)
    return cat,window,indices


def detail(a,b):return matrix(features([obs(a,0)],fps=30.),features([obs(b,1)],fps=30.),details=True)


def test_actual_contemporary_body_replaces_even_old_normal_observations():
    cat,window,_=prepared()
    cat.body[1,:,0,:,0]=.99  # old successful but mis-attributed ROI
    value=obs(cat,0)
    assert value['contemporary_subject_window_used']
    np.testing.assert_array_equal(value['body'],window.body[[7,13]])
    assert value['dense_subject_context']['clock'].native_indices.tolist()==list(range(93,107))
    assert not value['source_clock_subject_associated']
    assert not value['source_clock_context_is_sparse_model_hypothesis']
    assert value['evidence'].max()==1  # MODEL_ONLY, never pixel certification


def test_opposite_intermediate_motion_is_not_aliased_by_identical_sparse_endpoints():
    a,_,_=prepared(0,bump=.08);same,_,_=prepared(1,bump=.08);opposite,_,_=prepared(1,bump=-.08)
    # -7,-1,0,+6 poses are identical; only +1 distinguishes these inputs.
    for b in (same,opposite):np.testing.assert_array_equal(a.body,b.body)
    good,bad=detail(a,same),detail(a,opposite)
    assert good['contemporary_dense_action_used'][0,0] and bad['contemporary_dense_action_used'][0,0]
    assert good['pose_rate'][0,0]<bad['pose_rate'][0,0]
    assert good['matched_action_change_density'][0,0]>0
    assert good['face_rate'][0,0]==bad['face_rate'][0,0]==0.


def test_one_stationary_source_cannot_borrow_the_others_activity():
    a,_,_=prepared(0,bump=.08);b,_,_=prepared(1,bump=0.)
    assert detail(a,b)['matched_action_change_density'][0,0]==0.


def test_missing_dense_arms_do_not_recover_old_falsely_supported_pose():
    a,_,_=prepared(0);b,_,_=prepared(1,missing=True)
    value=detail(a,b)
    assert value['contemporary_dense_action_observed'][0,0]
    assert not value['contemporary_dense_action_used'][0,0]
    assert value['pose_rate'][0,0]==.5
    assert value['matched_action_change_density'][0,0]==0.


def test_wrong_source_clock_hash_or_actual_endpoint_cannot_relabel_cached_window():
    cat,window,indices=prepared()
    with pytest.raises(ValueError,match='source/hash'):
        DenseSubjects(source=1,binding='0',indices=indices,windows=[window],bank=cat.bank,fps=30.)
    shifted=indices.copy();shifted[1]+=1
    with pytest.raises(ValueError,match='actual grid'):
        DenseSubjects(source=0,binding='0',indices=shifted,windows=[window],bank=cat.bank,fps=30.)


def test_conflicting_actual_pose_is_unknown_independent_of_cache_file_order():
    cat,window,indices=prepared()
    body=window.body.copy();body[:,4,0]+=.04;other=replace(window,body=body,track='other-seed')
    for windows in ([window,other],[other,window]):
        cat.subjects=DenseSubjects(source=0,binding='0',indices=indices,windows=windows,bank=cat.bank,fps=30.)
        assert cat.subjects.sample(1,0,0) is None
        assert cat.subjects.receipt()['conflicting_contexts']==1


def test_fourth_grid_face_does_not_inherit_the_first_tracks_body():
    cat,_,_=prepared();cat.bank['heads'][1,:,3]=[.8,.3,.2,.99]
    value=cat.observation(1,0,np.tile([1.,.5,.5],(7,1)),target_head_index=3)
    assert 'dense_subject_context' not in value
    assert not value['body'].any()


def test_original_native_cut_stops_context_without_excluding_the_source():
    a,_,_=prepared(0,bump=.08);b,_,_=prepared(1,bump=.08,cut=(100,))
    value=detail(a,b)
    assert value['allowed'][0,0]
    assert value['contemporary_dense_action_used'][0,0]  # actual common POST remains
    assert not obs(b,1)['dense_subject_context']['same_shot'][:7].any()


def test_bound_saved_windows_load_without_inference_or_globbing_other_experiments(tmp_path):
    import json
    from types import SimpleNamespace
    from crosscut2.grid_perception import file_identity
    from crosscut2.subject_window_bank import save
    from crosscut2.dense_subject_catalogue import load_reused
    cat,window,_=prepared();source=tmp_path/'source.mp4';source.touch()
    target=tmp_path/'window.npz';save(window,target,provenance=dict(actual_test_window=True))
    record=dict(render_plan=dict(sources=[str(source)],fps=30.,master_audio_start=0.),
        source_sha256=['0'],source_file_identities=[file_identity(source)],
        subject_moment_refinement_receipt=dict(window_receipts=[dict(source=0,path=str(target)),dict(source=0,path=str(target))]))
    path=tmp_path/'observations.json';path.write_text(json.dumps(record))
    kw=dict(sources=[source],source_hashes=['0'],banks=[cat.bank],nodes=[0,100,200],
        warps=[SimpleNamespace(master_to_source=lambda t:t)],clocks=[dict(fps=30.,frame_count=200)],
        master_start=0.,fps=30.,authorized_root=tmp_path)
    loaded=load_reused(path,**kw)
    assert len(loaded)==1 and loaded[0].sample(1,0,0) is not None
    assert loaded[0].receipt()['new_neural_calls']==0
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source domain changed'):load_reused(path,**kw)


def test_optional_dense_evidence_cannot_silently_enter_other_cli_mode(monkeypatch):
    from crosscut2.stage_mix import main
    monkeypatch.setattr('sys.argv',['stage_mix','--source-plan','unused','--analysis','unused','--cache','unused',
        '--output','unused','--models','unused','--reuse-subject-observations','unused'])
    with pytest.raises(ValueError,match='explicit played port-camera'):main()
