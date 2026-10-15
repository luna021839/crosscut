import numpy as np
import pytest
from crosscut2.editorial_roles import FeatureRoles,apply,event_energy,ordinary_pairs
from crosscut2.featured_member_state import consensus
from crosscut2.played_partial_pose import features,matrix
from crosscut2.boundary_arc_costs import ArcCosts
from test_crosscut2_event_holds import inputs
from test_crosscut2_source_clock_action import catalogue,obs
from test_crosscut2_boundary_arc_costs import arc_fixture


def evidence(*,families=3,changed=True,unknown=False):
    nodes,banks,_,_=inputs(count=families);members=[]
    for s,bank in enumerate(banks):
        bank['consensus_family']=s
        names=np.full(bank['heads'].shape[:3],-1,int)
        if not unknown:
            names[:,:,0]=0
            if changed:names[1,1,0]=1
        members.append(names)
    factors,receipt,raw=consensus(banks,members,return_evidence=True)
    return FeatureRoles(nodes,raw),factors,receipt,raw


def test_roles_are_fixed_original_family_evidence_and_bind_both_people():
    roles,_,_,_=evidence()
    assert roles.at(18,0,1)>0
    assert roles.at(18,1,0)==roles.at(18,0,0)==roles.at(19,0,1)==0
    event=roles.receipt()['events'][0]
    assert event['kind']=='FEATURE_HANDOFF' and event['members']==[0,1]
    assert event['independent_family_support']==[3,3]
    assert roles.receipt()['authored_cut_bonus']==0.


@pytest.mark.parametrize('kw',[dict(families=1),dict(changed=False),dict(unknown=True)])
def test_no_original_role_evidence_cannot_turn_wrong_identity_into_free_handoff(kw):
    roles,_,_,_=evidence(**kw)
    assert not roles.strength.any() and roles.at(18,0,1)==0


def test_a_clone_family_cannot_certify_a_role_change():
    nodes,banks,_,_=inputs(count=3);members=[]
    for bank in banks:
        bank['consensus_family']='same'
        names=np.full(bank['heads'].shape[:3],-1,int);names[:,:,0]=0;names[1,1,0]=1;members.append(names)
    _,_,raw=consensus(banks,members,return_evidence=True)
    assert not FeatureRoles(nodes,raw).strength.any()


def test_supported_handoff_keeps_frame_and_flow_cost_without_matching_different_faces():
    a,b=catalogue(),catalogue();b.bank['heads'][1,1,0,:3]=[.7,.3,.4]
    detail=matrix(features([obs(a,0)],fps=30.),features([obs(b,1)],fps=30.),details=True)
    assert not detail['allowed'][0,0]
    handoff=apply(detail,1.)
    assert handoff['allowed'][0,0] and handoff['geometry_rate'][0,0]>0
    assert handoff['pose_rate'][0,0]==detail['motion_flow_rate'][0,0]
    assert handoff['appearance_rate'][0,0]==0
    unchanged=apply(detail,0.)
    for name in ('geometry_rate','pose_rate','appearance_rate','allowed'):
        np.testing.assert_array_equal(unchanged[name],detail[name])


def test_native_and_authored_use_the_same_role_cost_and_missing_body_risk():
    roles,_,_,_=evidence();cat=catalogue();a,b=obs(cat,0),obs(cat,1)
    for o in (a,b):o['body'][:]=0;o['evidence'][:]=0
    a['recurrent_member_hint']=0;b['recurrent_member_hint']=1
    detail=matrix(features([a],fps=30.),features([b],fps=30.),details=True)
    detail=apply(detail,roles.at(18,0,1))
    arcs=ArcCosts.__new__(ArcCosts);arcs.roles=roles;arcs.point_geometry=True;arcs.fps=30.
    heads=np.tile(cat.bank['heads'][1],(1,1,1,1));vectors=cat.bank['vectors'][1:2].copy()
    names=np.full(heads.shape[:3],-1,int);names[0,0,0]=0;names[0,1,0]=1
    arcs.native=[dict(frames=np.array([18]),heads=heads,vectors=vectors,member_hints=names)]
    camera=np.tile([1.,.5,.5],(1,1,1))
    body,point=arcs.native_rates(0,np.array([18]),camera,camera,components=True)
    for window in (1/60.,6/30.):
        actual=point+window*body
        expected=event_energy(detail,window,fps=30.,point_geometry=True)
        np.testing.assert_allclose(actual,expected)
    assert body[0,0]==.5  # no fabricated native pose/flow support


def test_adjacent_events_cannot_erase_instantaneous_face_or_appearance_jump():
    detail=dict(geometry_rate=np.array([[2.]]),appearance_rate=np.array([[.5]]),pose_rate=np.array([[.5]]))
    short=event_energy(detail,1/60.,fps=30.,point_geometry=True)
    isolated=event_energy(detail,.2,fps=30.,point_geometry=True)
    assert short[0,0]>.5 and isolated[0,0]-short[0,0]==pytest.approx(.5*(.2-1/60.))
    legacy=event_energy(detail,1/60.,fps=30.,point_geometry=False)
    assert legacy[0,0]==pytest.approx(3/60.)


def test_native_point_terms_are_exact_in_prefix_patch_and_replayed_whole_cost():
    # Include closely neighbouring native events and truncated endpoint
    # patches; fixed point terms must not be lost to support clipping.
    nodes,entry,exit,old,fragments=arc_fixture(True)
    arcs=ArcCosts(nodes,entry,exit,old.cats,old.dense,old.native,old.support,fragments,fps=30.,point_geometry=True)
    compared=0
    for i in range(len(nodes)-1):
        for j in range(i+1,len(nodes)):
            value,allowed,codes=arcs(i,j)
            for p,q in zip(*np.nonzero(allowed)):
                path=arcs.replay(i,j,p,q,int(codes[p,q]));s=entry[i][p].source
                assert value[p,q]==pytest.approx(arcs.direct_cost(s,int(nodes[i]),path[None])[0]+fragments(i,j)[s])
                compared+=1
    assert compared>0 and sum(p.sum() for p,_ in arcs.native_points)>0


def test_native_role_fulfillment_has_no_authored_cut_reward_and_missing_role_has_deficit():
    roles,factors,_,_=evidence()
    # Showing the same old person after the original consensus handoff has
    # a deficit; showing the new one in an existing native shot does not.
    assert roles.at(18,0,1)>0 and factors[0][1,1,0]==pytest.approx(1.)
    _,banks,_,_=inputs(count=3)
    members=[]
    for s,bank in enumerate(banks):
        bank['consensus_family']=s;names=np.full(bank['heads'].shape[:3],-1,int)
        names[:,:,0]=0
        if s<2:names[1,1,0]=1
        members.append(names)
    f,_,_=consensus(banks,members,return_evidence=True)
    assert f[2][1,1,0]<f[0][1,1,0]
    assert roles.receipt()['authored_cut_bonus']==0


def test_ordinary_scene_change_does_not_need_an_unknown_face_bridge_to_evade_match_gate():
    from types import SimpleNamespace
    from crosscut2.boundary_port_dp import solve
    static=lambda s:SimpleNamespace(source=s,proposal_ids=(('static',0),))
    paired=lambda s:SimpleNamespace(source=s,proposal_ids=(('paired',1,0,1,0,0,0),))
    assert ordinary_pairs([static(0),paired(0)],[static(1),paired(1)]).tolist()==[[True,False],[False,False]]
    nodes=np.array([0,30,42,72]);ids=[np.arange(3) for n in nodes]
    def hold(i,j):
        value=np.full((3,3),20.);allowed=np.eye(3,dtype=bool)
        if i==0:value[0,0]=0.
        if i==1 and j==2:value[2,2]=.4  # short empty/legs scene
        if i>=1:value[1,1]=0.
        return value,allowed
    risks=np.array([[0.,1.8,1.],[1.8,0.,1.],[1.,1.,0.]])
    def choose(ordinary):
        return solve(nodes,ids,ids,start_cost=[0.,20.,20.],hold_cost=hold,
            cut_cost=lambda n:(.2*risks,(risks<=1.)|ordinary),final_cost=[20.,0.,20.])
    legacy,_,_=choose(False);actual,_,_=choose(True)
    assert [ids[i][p] for i,j,p,q in legacy]==[0,2,1]
    assert [ids[i][p] for i,j,p,q in actual]==[0,1]
    # The direct scene change still pays 1.8 times the common point unit;
    # it is not certified as a matching-pose transition or rewarded.


def test_actual_producer_keeps_ordinary_change_playable_without_claiming_a_face_match(monkeypatch):
    from test_crosscut2_preference_integration import producer
    from crosscut2.event_holds import select
    from crosscut2 import played_pose_port_programme as programme
    original=programme.solve
    def declared_endpoints(nodes,entries,exits,**kw):
        # Force endpoint sources in this mechanical fixture only. Both
        # cameras are observed, but even their largest feasible crop cannot
        # turn the 3.2x native face jump into a matched-face transition.
        return original(nodes,entries,exits,**kw,
            start_allowed=np.asarray(entries[0])==0,final_allowed=np.asarray(exits[-1])==1)
    monkeypatch.setattr(programme,'solve',declared_endpoints)
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    args[1][1]['heads'][:,:,0,2]=.8
    runs,_,_,_,receipt,ports=select(*args,**kw,native_ports=native,music=[0,1,1,0],
        connection_prior=programme.PlayedPosePortPrior(editorial_opportunities=True))
    assert len(runs)>=2
    changes=[ports.selected(i,k-1,k) for k,(i,j,s) in enumerate(runs) if k]
    assert any(s['editorial_transition_kind']=='SCENE_CHANGE' and s['played_face_rate']>1. for s in changes)
    assert all(not s['successful_face_match_claimed'] for s in changes if s['editorial_transition_kind']=='SCENE_CHANGE')
    assert receipt['event_preference']['fixed_connection_prior']['maximum_measured_face_height_ratio'] is None
