import numpy as np
import pytest
from crosscut2.display_source_catalogue import SourceCatalogue
from crosscut2.shot_camera_paths import make_paths
from crosscut2.boundary_port_domain import make
from crosscut2.boundary_arc_costs import ArcCosts
from crosscut2.played_event_support import PlayedEventSupport
from crosscut2.event_holds import fragment_costs
from test_crosscut2_event_holds import inputs


def counterexample(retain):
    nodes=np.array([0,30,120,150]);_,banks,poses,styles=inputs(count=4);cats=[]
    for s,bank in enumerate(banks):
        bank['heads'][:]=0;bank['heads'][:,:,0]=[.5,.4,[.1,.1,.12,.15][s],.99]
        bank['zoom_cap']=2.
        bank['native_indices']=np.clip(nodes[:,None]+[-1,0],0,149)
        bank['pose_native_indices']=np.clip(nodes[:,None]+[-7,-1,0,6],0,149)
        bank['programme_render_clock']=dict(fps=30.,count=150,source_times=nodes/30.)
        cats.append(SourceCatalogue(nodes,bank,poses[s],[],styles[s],[],fps=30.,editorial_content=True))
    paths,_=make_paths(nodes,banks,[[]]*4)
    entry,exit,dense,receipt=make(nodes,cats,paths,fps=30.,articulated_actions=True,
        recycle_static_capacity=True,retain_pair_cameras=retain)
    support=PlayedEventSupport([[]]*4,final=150,fps=30.)
    fragment=fragment_costs(nodes,[[]]*4,fps=30.,support_seconds=.4)
    native=[dict(frames=np.array([],int),heads=np.zeros((0,2,8,4))) for _ in cats]
    arcs=ArcCosts(nodes,entry,exit,cats,dense,native,support,fragment,fps=30.,
        opportunity_episodes=np.arange(4),preserve_camera_alternatives=True)
    return entry,exit,dense,receipt,arcs,fragment


def pair_ports(ports):
    return [(i,p) for i,p in enumerate(ports) if any(k[0]=='paired' and k[2:4]==(0,1) for k in p.proposal_ids)]


def test_no_alias_domain_keeps_actual_lower_cost_camera_for_interval_choice():
    old_in,_,_,old,_,_=counterexample(False)
    entry,exit,_,receipt,arcs,fragment=counterexample(True)
    assert old['identical_static_port_aliases_coalesced']==receipt['identical_static_port_aliases_coalesced']==0
    assert {p.strip[0,0] for _,p in pair_ports(old_in[1])}=={1.}
    assert {p.strip[0,0] for _,p in pair_ports(entry[1])}=={1.,1.5,2.}
    q=next(i for i,p in enumerate(exit[2]) if p.source==1 and np.allclose(p.strip[0],[2.,.5,.45]))
    costs,allowed=arcs.camera_alternatives(1,2);best={}
    for p,port in pair_ports(entry[1]):
        code=int(np.argmin(np.where(allowed[p,q],costs[p,q],np.inf)))
        path=arcs.replay(1,2,p,q,code)
        actual=arcs.direct_cost(1,30,path[None])[0]+fragment(1,2)[1]
        best[port.strip[0,0]]=costs[p,q,code]
        assert actual==pytest.approx(best[port.strip[0,0]],abs=1e-12)
    assert best[1.5]<best[1.]-.09 and best[1.5]<best[2.]
    assert receipt['additional_primary_pair_camera_options']>0
    assert not receipt['previous_work_bound_unchanged']


def test_expansion_keeps_old_physical_ports_baselines_pair_owners_and_an_explicit_bound():
    previous_in,previous_out,old_dense,_,_,_=counterexample(False)
    entry,exit,dense,receipt,_,_=counterexample(True)
    np.testing.assert_array_equal(dense,old_dense)
    physical=lambda ports:{(p.source,p.geometry_hash,p.observation['head_index'],p.proposal_ids) for p in ports}
    for before,after in ((previous_in,entry),(previous_out,exit)):
        for a,b in zip(before,after):assert physical(a)<=physical(b)
    assert max(map(len,entry))<=4*4+32 and max(map(len,exit))<=4*4+32
    for n in range(1,len(entry)-1):
        left={k for p in exit[n] for k in p.proposal_ids if k[0]=='paired'}
        right={k for p in entry[n] for k in p.proposal_ids if k[0]=='paired'}
        assert left==right
        for ports,source_index,head_index in ((exit[n],2,4),(entry[n],3,5)):
            for p in ports:
                ids=[k for k in p.proposal_ids if k[0]=='paired']
                if not ids:continue
                assert len(ids)==1
                assert p.source==ids[0][source_index]
                assert p.observation['head_index']==ids[0][head_index]
    assert receipt['all_baseline_interior_camera_families_retained']


def test_expansion_requires_real_interval_consumer_and_is_opt_in(monkeypatch):
    from crosscut2.played_pose_port_programme import PlayedPosePortPrior
    from crosscut2.stage_mix import main
    assert not PlayedPosePortPrior().retain_pair_cameras
    with pytest.raises(ValueError,match='actual interval camera'):
        PlayedPosePortPrior(retain_pair_cameras=True)
    value=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,retain_pair_cameras=True)
    assert not value.artifact['same_total_port_bound']
    assert value.artifact['maximum_cameras_per_primary_actor_pair']==4
    monkeypatch.setattr('sys.argv',['stage_mix','--source-plan','unused','--analysis','unused','--cache','unused',
        '--output','unused','--models','unused','--retain-pair-cameras'])
    with pytest.raises(ValueError,match='actual played interval utility'):main()


def test_real_programme_selector_consumes_expanded_domain_and_freezes_actual_cameras():
    from crosscut2.played_pose_port_programme import PlayedPosePortPrior,select
    _,_,_,_,arcs,_=counterexample(False)
    prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,
        recycle_static_capacity=True,retain_pair_cameras=True)
    _,_,_,_,receipt,_=select(arcs.nodes,[c.bank for c in arcs.cats],[c.body for c in arcs.cats],
        [c.style for c in arcs.cats],fps=30.,source_events=[[]]*4,body_states=None,
        native_ports=arcs.native,music=np.array([0.,1.,1.,0.]),prior=prior)
    domain=receipt['boundary_port_domain']
    assert domain['primary_actor_pair_camera_variants_retained']
    assert domain['additional_primary_pair_camera_options']>0
    assert receipt['frozen_selected_camera_arcs']
