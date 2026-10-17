import numpy as np
import pytest

from crosscut2.focal_image_motion import TICKS,describe,matrix,static_camera_pairs
from crosscut2.boundary_port_domain import make
from crosscut2.shot_camera_paths import make_paths
from test_crosscut2_focal_image_motion import observation
from test_crosscut2_pair_camera_options import counterexample


def test_proposal_batch_matches_explicit_constant_camera_replay_and_not_native_only_velocity():
    a=[observation(.5+.003*TICKS,side=0),observation(.5+.001*np.abs(TICKS),side=0)]
    b=[observation(.5-.002*TICKS,side=1),observation(.5+.004*TICKS,side=1)]
    ai=np.array([0,1,0]);bi=np.array([1,0,0]);za=np.array([1.,1.6,2.]);zb=np.array([1.8,1.,1.4])
    actual=static_camera_pairs([describe(o,fps=30.) for o in a],[describe(o,fps=30.) for o in b],ai,bi,za,zb)
    for k,(i,j,x,y) in enumerate(zip(ai,bi,za,zb)):
        left=dict(a[i],camera_strip=np.tile([x,.45,.55],(7,1)))
        right=dict(b[j],camera_strip=np.tile([y,.55,.45],(7,1)))
        expected=matrix([describe(left,fps=30.)],[describe(right,fps=30.)])
        for p,q in zip(actual,expected):assert p[k]==pytest.approx(q[0,0],abs=1e-12)
    native=matrix([describe(a[0],fps=30.)],[describe(b[1],fps=30.)])[0][0,0]
    assert actual[0][0]!=pytest.approx(native)


def test_unknown_proposal_motion_is_not_stationary_or_an_extra_candidate():
    a,b=observation(.5+.005*TICKS,side=0),observation(.5-.005*TICKS,side=1)
    a['dense_subject_context']['contiguous'][5]=False
    score,known,count=static_camera_pairs([describe(a,fps=30.)],[describe(b,fps=30.)],
        np.array([0]),np.array([0]),np.array([2.]),np.array([1.]))
    assert score.tolist()==[.5] and not known.any() and not count.any()
    nodes,cats,paths=motion_catalogues(False)
    old=make(nodes,cats,paths,fps=30.,retain_pair_cameras=True)
    new=make(nodes,cats,paths,fps=30.,retain_pair_cameras=True,focal_motion=True)
    for before,after in zip(old[:2],new[:2]):
        for p,q in zip(before,after):assert physical(p)==physical(q)
    assert new[3]['additional_motion_actor_pairs']==[]


def test_static_proposal_hypothesis_rejects_a_previously_moving_camera_or_bad_indices():
    a=observation(.5+.005*TICKS,side=0,camera=np.c_[np.ones(7),np.linspace(.5,.51,7),np.full(7,.5)])
    b=observation(.5+.005*TICKS,side=1)
    with pytest.raises(ValueError,match='identity native static'):
        static_camera_pairs([describe(a,fps=30.)],[describe(b,fps=30.)],np.array([0]),np.array([0]),np.ones(1),np.ones(1))
    a=observation(.5+.005*TICKS,side=0)
    for indices,zoom in ((np.array([1]),np.ones(1)),(np.array([0]),np.array([np.nan]))):
        with pytest.raises(ValueError,match='paired native indices'):
            static_camera_pairs([describe(a,fps=30.)],[describe(b,fps=30.)],indices,np.array([0]),zoom,np.ones(1))


def motion_catalogues(measured):
    _,_,_,_,arcs,_=counterexample(False);cats=arcs.cats
    for s,cat in enumerate(cats):
        cat.bank['heads'][:,:,0]=[.5,.4,.1,.99]
        original=cat.observation
        def observed(n,side,camera_strip,*,target_head_index=None,_original=original,_s=s,_nodes=arcs.nodes):
            value=_original(n,side,camera_strip,target_head_index=target_head_index)
            if measured and n in (1,2):
                velocity=[.009,-.009,.001,.001][_s]
                context=observation(.5+velocity*(TICKS+(side==0)),side=side)['dense_subject_context']
                context['window'].heads[:,1]=.4
                context['window'].native_cuts=np.array([],int)
                context['window'].source=context['clock'].source=_s
                context['window'].binding=context['clock'].binding='unbound-catalogue-source-'+str(_s)
                context['window'].ordinals=context['clock'].native_indices=_nodes[n]+TICKS
                context['window'].body=np.zeros((14,8,3))
                context['window'].track='actual-test-source-'+str(_s)
                context['window'].competitors=tuple(np.zeros((0,4)) for _ in TICKS)
                context['supported']=np.zeros((14,8),bool)
                value=dict(value,dense_subject_context=context)
            return value
        cat.observation=observed
    paths,_=make_paths(arcs.nodes,[c.bank for c in cats],[[]]*4)
    return arcs.nodes,cats,paths


def physical(ports):
    return {(p.source,p.geometry_hash,p.observation['head_index'],p.proposal_ids) for p in ports}


def test_motion_can_recover_pruned_actor_pairs_without_replacing_old_ports_or_clocks():
    nodes,cats,paths=motion_catalogues(True)
    old_in,old_out,old_dense,_=make(nodes,cats,paths,fps=30.,retain_pair_cameras=True)
    entry,exit,dense,receipt=make(nodes,cats,paths,fps=30.,retain_pair_cameras=True,focal_motion=True)
    np.testing.assert_array_equal(old_dense,dense)
    for before,after in ((old_in,entry),(old_out,exit)):
        for p,q in zip(before,after):assert physical(p)<=physical(q)
    assert receipt['additional_motion_actor_pair_limit_per_node']==2
    assert receipt['additional_motion_actor_pairs']
    assert receipt['measured_motion_camera_options_compared']>0
    assert max(map(len,entry))<=4*4+40 and max(map(len,exit))<=4*4+40
    for n in (1,2):
        added=[r for r in receipt['additional_motion_actor_pairs'] if r['node']==n]
        assert len(added)<=2
        assert any(r['actor_pair'][:2] in ([2,3],[3,2]) for r in added)
        left={i for p in exit[n] for i in p.proposal_ids if i[0]=='paired'}
        right={i for p in entry[n] for i in p.proposal_ids if i[0]=='paired'}
        assert left==right
        assert any(len(entry[n])>len(old_in[n]) for n in (1,2))
        for p in (*exit[n],*entry[n]):
            if any(i[0]=='paired' for i in p.proposal_ids):assert p.observation['target_is_displayed_focal']


def test_disabled_pairs_and_one_source_never_expand_or_invent_motion():
    nodes,cats,paths=motion_catalogues(True)
    for kwargs in (dict(maximum_pairs=0),dict(source_cohort=(0,))):
        entry,exit,_,receipt=make(nodes,cats,paths,fps=30.,focal_motion=True,retain_pair_cameras=True,**kwargs)
        assert not receipt['additional_motion_actor_pairs']
        assert not any(i[0]=='paired' for row in (*entry,*exit) for p in row for i in p.proposal_ids)
