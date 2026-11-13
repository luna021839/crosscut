"""Counterexample separation is representation proof, not AV quality."""
from dataclasses import replace
import numpy as np
import pytest
from crosscut2.ordered_context import project,merge,NAMES,PERSON_NAMES,OrderedEvidence
from crosscut2.playback_blocks import BlockPool,InputKey,flatten


def observed_scene(T=120):
    from crosscut2.scene_targets import TargetScene,FEATURE
    head=np.tile([[[.5,.375,.15625,1.]]],(T,1,1));known=np.ones((T,1),bool)
    body=np.ones((T,1,8,3));body[...,:2]=np.array([[.46,.48],[.54,.48],[.42,.56],[.58,.56],
        [.40,.63],[.60,.63],[.47,.72],[.53,.72]])
    return TargetScene('source','actor',np.arange(T),np.arange(T),FEATURE,head,known,body,
        np.ones((T,1,8),np.uint8),np.zeros((T,1,8),bool),np.ones((T,1)),known,head,known)


def test_actual_body_trajectory_and_camera_order_are_not_pooled_away():
    from crosscut2.sequence_context import project as old_project,windows
    s=observed_scene();body=s.body.copy();body[:,0,4,0]+=.1875*np.sin(2*np.pi*np.arange(120)/119)
    other=replace(s,body=body);camera=np.tile([1.,.5,.5],(120,1))
    assert np.array_equal(windows(old_project(s,camera),final_frame=120,fps=30.)['features'],
        windows(old_project(other,camera),final_frame=120,fps=30.)['features'])
    a,b=project(s,camera),project(other,camera)
    k=PERSON_NAMES.index('left_wrist.screen_x')
    assert a.people_observed[:,k].all() and b.people_observed[:,k].all()
    assert not np.array_equal(a.people[:,k],b.people[:,k])
    ca=np.tile([2.,.5,.5],(120,1));cb=ca.copy()
    ca[:,1]+=np.tile([0,-4,-4,1,-2,0],20)/64.;cb[:,1]+=np.tile([0,-4,-2,-4,1,0],20)/64.
    assert np.array_equal(windows(old_project(s,ca),final_frame=120,fps=30.)['features'],
        windows(old_project(s,cb),final_frame=120,fps=30.)['features'])
    assert not np.array_equal(project(s,ca).values,project(s,cb).values)


def test_partial_arm_and_missing_face_keep_own_masks_without_lending_a_body():
    s=observed_scene(8);kinds=s.body_evidence.copy();kinds[:,:,5]=0
    ambiguity=s.body_ambiguity.copy();ambiguity[:,:,4]=True
    s=replace(s,body_evidence=kinds,body_ambiguity=ambiguity,head_observed=np.zeros((8,1),bool),
        roi_measured=np.zeros((8,1),bool))
    a=project(s,np.tile([1.,.5,.5],(8,1)))
    assert a.people_observed[:,PERSON_NAMES.index('left_elbow.screen_x')].all()
    assert not a.people_observed[:,PERSON_NAMES.index('right_wrist.screen_x')].any()
    assert not a.people_observed[:,PERSON_NAMES.index('left_wrist.screen_x')].any()
    assert a.people[:,PERSON_NAMES.index('left_wrist.ownership_ambiguous')].all()
    assert not a.people_observed[:,PERSON_NAMES.index('head.screen_x')].any()
    assert np.diff(a.person_splits).tolist()==[1]*8
    assert not a.observed[:,NAMES.index('measured_background_flow')].any()


def test_ordered_conflict_persists_and_virtual_splits_do_not_change_input():
    s=observed_scene(12);a=project(s,np.tile([1.,.5,.5],(12,1)))
    def cut(a,i,j):return a.take(np.arange(i,j))
    joined=merge([cut(a,0,5),cut(a,5,12)])
    assert np.array_equal(a.values,joined.values) and a.scopes==joined.scopes and np.array_equal(a.people,joined.people)
    different=replace(s,track='unassociated-competitor')
    bad=merge([a,project(different,np.tile([1.,.5,.5],(12,1)))])
    final=merge([bad,a]);k=PERSON_NAMES.index('left_elbow.screen_x')
    assert final.people_conflicted[:,k].all() and not final.people_observed[:,k].any()


@pytest.mark.parametrize('changed',['native','camera','source','intent','version'])
def test_block_cache_reuses_only_original_exact_inputs_even_if_hashes_collide(monkeypatch,changed):
    monkeypatch.setattr(InputKey,'__hash__',lambda self:1)
    pool=BlockPool();s=observed_scene(12);calls=[]
    def observe(t,n,c):calls.append(t.copy());return project(replace(s,frames=t,native_indices=n,
        **{name:getattr(s,name)[t] for name in ('heads','head_observed','body','body_evidence',
            'body_ambiguity','roi_readability','roi_measured','original_heads','original_observed')}),c)
    args=dict(source='a'*64,observation_version=0,model_schema=('schema','model'),frames=np.arange(12),native=np.arange(12),
        cameras=np.tile([1.,.5,.5],(12,1)),intents=((),)*12,project=observe,
        encode=lambda e:e.people[:,[PERSON_NAMES.index('left_wrist.screen_x')]])
    a=pool.compile(**args);b=pool.compile(**args)
    assert all(x is y for x,y in zip(a,b)) and len(calls)==1
    other=dict(args)
    if changed=='native':other['native']=np.arange(12)+1
    elif changed=='camera':other['cameras']=args['cameras'].copy();other['cameras'][0,1]=np.nextafter(.5,1.)
    elif changed=='source':other['source']='b'*64
    elif changed=='intent':other['intents']=((('other-target',),),)*12
    else:other['observation_version']=1
    c=pool.compile(**other)
    assert any(x.key!=y.key for x,y in zip(a,c)) and len(calls)==2
    assert pool.receipt()['candidates_or_camera_codes_removed'] is False


def test_global_storage_blocks_keep_each_frame_and_exact_split_invariant_identity():
    pool=BlockPool();s=observed_scene(12)
    def compile(first,final):
        frames=np.arange(first,final);cameras=np.tile([1.,.5,.5],(len(frames),1))
        cameras[:,1]+=np.arange(first,final)/1000.
        return pool.compile(source='a'*64,observation_version=0,model_schema=('s','m'),frames=frames,native=frames,
            cameras=cameras,intents=((),)*len(frames),project=lambda t,n,c:project(replace(s,frames=t,native_indices=n,
                **{name:getattr(s,name)[t] for name in ('heads','head_observed','body','body_evidence',
                    'body_ambiguity','roi_readability','roi_measured','original_heads','original_observed')}),c),
            encode=lambda e:e.values[:,[NAMES.index('additional_center_x')]])
    whole=flatten(compile(0,12));split=flatten((*compile(0,5),*compile(5,12)))
    assert np.array_equal(whole[0],split[0]) and np.array_equal(whole[1],split[1]) and whole[2]==split[2]
    assert len(np.unique(whole[1]))==12


def test_more_than_eight_observed_people_are_retained_without_a_source_veto_or_person_cap():
    from crosscut2.scene_targets import FORMATION
    s=observed_scene(12);H=11
    many=replace(s,role=FORMATION,**{name:np.repeat(getattr(s,name),H,axis=1) for name in
        ('heads','head_observed','body','body_evidence','body_ambiguity','roi_readability','roi_measured',
            'original_heads','original_observed')})
    a=project(many,np.tile([1.,.5,.5],(12,1)))
    assert np.diff(a.person_splits).tolist()==[H]*12 and len(a.people)==12*H
    assert a.people_observed[:,PERSON_NAMES.index('left_wrist.screen_x')].all()
    joined=merge([a.take(np.arange(5)),a.take(np.arange(5,12))])
    assert np.array_equal(a.people,joined.people) and np.array_equal(a.person_splits,joined.person_splits)
