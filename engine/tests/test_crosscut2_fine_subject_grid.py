from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from crosscut2.fine_subject_grid import propose, expand, rebind_hints, schedule, observe_style
from crosscut2.source_member_scores import MemberScoreSeeds, bank_signature
from crosscut2.fixed_holds import local_actor_hypotheses
from crosscut2.dense_subject_catalogue import bind_windows
from test_crosscut2_subject_cut_moments import window


class IdentityClock:
    def master_to_source(self,t):return t
    def source_to_master(self,t):return t


def inputs(count=2):
    nodes=np.array([0,100,200]);warps=[IdentityClock() for _ in range(count)]
    clocks=[dict(fps=30.,frame_count=201) for _ in range(count)];groups=[[window(s)] for s in range(count)]
    banks=[];poses=[];styles=[];states=[]
    for s in range(count):
        heads=np.zeros((3,2,8,4),np.float32);heads[:,:,0]=[.5,.3,.2,.99]
        vectors=np.zeros((3,2,3,128),np.float32);vectors[:,:,0,0]=1.
        banks.append(dict(heads=heads,vectors=vectors,zoom_cap=2.,
            native_indices=schedule(warps[s],clocks[s],nodes[:,None]+[-1,0],master_start=0.,fps=30.,final=200),
            pose_native_indices=schedule(warps[s],clocks[s],nodes[:,None]+[-7,-1,0,6],master_start=0.,fps=30.,final=200),
            native_readability=np.ones((3,2,8,4),np.float32),native_readability_valid=np.ones((3,2,8),bool)))
        poses.append(np.zeros((3,4,3,8,3),np.float32));states.append(np.zeros((3,4,3),np.uint8))
        styles.append(dict(colour=np.ones((3,2,8,2,40),np.float32),colour_valid=np.ones((3,2,8),bool),
            presence=np.zeros((3,2,2),np.float32)))
    return nodes,banks,poses,styles,states,groups,warps,clocks


def test_fine_proposals_come_from_common_native_observations_not_original_cut_times():
    nodes,banks,_,_,_,groups,warps,clocks=inputs()
    new,receipt=propose(nodes,groups,warps,clocks,[2.,2.],master_start=0.,fps=30.,maximum_new_nodes=8)
    assert len(new)==len(nodes)+8 and np.isin(nodes,new).all()
    assert receipt['tested_pair_moments']>8 and receipt['observed_regions']>0
    assert receipt['proposals_from_source_windows_not_finished_edit'] and not receipt['source_retimed']
    _,none=propose(nodes,[groups[0],[]],warps,clocks,[2.,2.],master_start=0.,fps=30.)
    assert none['new_nodes']==0


def test_projected_old_rows_are_byte_exact_and_fine_rows_are_actual_not_nearest_old_faces():
    nodes,banks,poses,styles,states,groups,warps,clocks=inputs()
    new_nodes=np.array([0,98,100,200])
    output=expand(nodes,new_nodes,banks,poses,styles,states,groups,warps,clocks,master_start=0.,fps=30.)
    bs,ps,ss,st,receipt=output
    for old,new in zip(banks,bs):
        np.testing.assert_array_equal(new['heads'][[0,2,3]],old['heads'])
        np.testing.assert_array_equal(new['native_indices'][1],[97,98])
        assert not new['native_readability_valid'][1].any()
    assert ps[0][1,:,0,:,2].min()>.3 and not ss[0]['colour_valid'][1].any()
    assert receipt['sources'][0]['fine_observed_faces']==2
    dense=bind_windows(groups,source_hashes=['0','1'],banks=bs,nodes=new_nodes,warps=warps,
        clocks=clocks,master_start=0.,fps=30.)
    assert dense[0].sample(1,0,0) is not None and dense[0].sample(1,1,0) is not None


def test_missing_source_windows_never_borrow_another_person_or_erase_the_source():
    nodes,banks,poses,styles,states,groups,warps,clocks=inputs();groups[1]=[]
    bs,ps,_,_,_=expand(nodes,np.array([0,98,100,200]),banks,poses,styles,states,groups,warps,clocks,master_start=0.,fps=30.)
    assert not bs[1]['heads'][1].any() and not ps[1][1].any()
    assert bs[1]['zoom_cap']==banks[1]['zoom_cap'] and len(bs)==2
    np.testing.assert_array_equal(bs[1]['heads'][[0,2,3]],banks[1]['heads'])


def test_unobserved_fine_node_neither_erases_real_content_samples_nor_claims_a_face():
    from crosscut2.display_source_catalogue import SourceCatalogue
    nodes,banks,poses,styles,states,groups,warps,clocks=inputs();groups[1]=[]
    ns=np.array([0,98,100,200])
    bs,ps,ss,_,_=expand(nodes,ns,banks,poses,styles,states,groups,warps,clocks,master_start=0.,fps=30.)
    before=SourceCatalogue(nodes,banks[1],poses[1],[],styles[1],[],fps=30.)
    after=SourceCatalogue(ns,bs[1],ps[1],[],ss[1],[],fps=30.)
    assert not bs[1]['face_observation_valid'][1].any()
    assert bs[0]['face_observation_valid'][1].all()
    frames=np.arange(94,107);camera=np.tile([1.,.5,.5],(len(frames),1))
    np.testing.assert_array_equal(after.content_scope,before.content_scope)
    np.testing.assert_array_equal(after.rates(frames,camera),before.rates(frames,camera))
    # Nearby samples are only the existing integration rule. They never turn
    # the unsampled fine endpoint into exact face/pose/appearance evidence.
    assert after.observation(1,0,np.tile([1.,.5,.5],(7,1)))['head_index']==-1


def test_conflicting_same_face_body_observations_cannot_win_by_input_order():
    nodes,banks,poses,styles,states,groups,warps,clocks=inputs()
    original=groups[0][0];body=original.body.copy();body[:,4,0]+=.1
    conflict=replace(original,body=body,track='other-ROI')
    for group in ([original,conflict],[conflict,original]):
        bs,ps,*_=expand(nodes,np.array([0,98,100,200]),banks,poses,styles,states,
            [group,groups[1]],warps,clocks,master_start=0.,fps=30.)
        assert bs[0]['heads'][1,:,0,2].min()>0
        assert not ps[0][1].any()


def test_gallery_query_profiles_remain_bound_while_new_cluster_numbers_can_change():
    nodes,banks,poses,styles,states,groups,warps,clocks=inputs()
    bs,*_=expand(nodes,np.array([0,98,100,200]),banks,poses,styles,states,groups,warps,clocks,master_start=0.,fps=30.)
    hints=[]
    for old in banks:
        _,count=local_actor_hypotheses(old)
        hints.append(MemberScoreSeeds(np.zeros(count,np.int16),old['vectors'][1,1,:1],
            np.asarray([[.6,.1,.1,.1,.1,.1]]),bank_signature(old),'gallery'))
    new=rebind_hints(banks,bs,hints)
    for h,before,bank in zip(new,hints,bs):
        h.verify(bank)
        np.testing.assert_array_equal(h.vectors,before.vectors)
        np.testing.assert_array_equal(h.scores,before.scores)
        assert h.gallery_identity==before.gallery_identity
    bad=dict(bs[0],heads=bs[0]['heads'].copy());bad['heads'][0,0,0,0]+=.1
    with pytest.raises(ValueError,match='physical occurrence'):
        rebind_hints(banks,[bad,bs[1]],hints)


def test_native_style_observes_only_new_actual_face_rows_and_keeps_original_values(monkeypatch,tmp_path):
    from crosscut2 import source_style
    nodes,banks,poses,styles,states,groups,warps,clocks=inputs();groups[1]=[]
    bs,_,ss,_,_=expand(nodes,np.array([0,98,100,200]),banks,poses,styles,states,
        groups,warps,clocks,master_start=0.,fps=30.)
    calls=[]
    def measure(*,sources,banks,cache,check,maximum_seconds):
        calls.append(sources)
        assert sources==['source-a'] and len(banks)==1
        np.testing.assert_array_equal(banks[0]['native_indices'],[[97,98]])
        return [dict(colour=np.full((1,2,8,2,40),.2,np.float32),
            colour_valid=np.ones((1,2,8),bool),presence=np.full((1,2,2),.3,np.float32))],\
            [dict(source=0,decoded_native_inputs=2,new_neural_calls=0)]
    monkeypatch.setattr(source_style,'observe',measure)
    result,receipt=observe_style(['source-a','source-b'],bs,ss,cache=tmp_path)
    assert len(calls)==1 and receipt['decoded_native_inputs']==2
    assert result[0]['colour_valid'][1].all() and not result[1]['colour_valid'][1].any()
    assert not ss[0]['colour_valid'][1].any()  # old observation object not mutated
    for old,new in zip(ss,result):
        for key in old:np.testing.assert_array_equal(old[key][[0,2,3]],new[key][[0,2,3]])
