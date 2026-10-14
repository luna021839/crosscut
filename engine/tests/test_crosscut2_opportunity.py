import numpy as np
import pytest
from crosscut2.opportunity import music_on_nodes
from crosscut2.opportunity_dp import choose_opportunities
from crosscut2.fixed_holds import State, make_states, port_cost, select_fixed_holds
from crosscut2.stage_mix import native_event_cost


def graph(nodes, unary, matrix, sources, *, through=None, events=None, same=None):
    nodes=np.asarray(nodes);sources=np.asarray(sources);q=len(sources)
    return dict(nodes=nodes,fps=10.,unary=np.asarray(unary),ports=lambda n:matrix[n],
        state_sources=sources,through=np.zeros((q,len(nodes))) if through is None else through,
        source_events=[np.array([],int) for _ in range(int(sources.max())+1)] if events is None else events,
        same_show=np.zeros((q,q)) if same is None else same)


def test_no_opportunity_keeps_real_long_hold_even_with_a_bad_second_source():
    nodes=np.arange(0,61,10)
    args=graph(nodes,np.array([np.zeros(6),np.full(6,100.)]),np.full((7,2,2),100.),[0,1])
    runs,score,diversity=choose_opportunities(**args)
    assert runs==((0,6,0),) and score==diversity==0.


def test_good_pair_only_consumes_its_own_opportunity_and_music_changes_timing():
    nodes=np.arange(0,41,10);base=np.full((5,2,2),0.)
    for strong,weak in ((1,2),(2,1)):
        ports=base.copy();ports[strong,0,1]=-.8;ports[weak,0,1]=-.5
        runs,score,_=choose_opportunities(**graph(nodes,np.zeros((2,4)),ports,[0,1]))
        assert runs==((0,strong,0),(strong,4,1)) and score==pytest.approx(-.45)


def test_diversity_is_secondary_and_cannot_discard_unique_good_same_show_match():
    nodes=np.array([0,18,36]);unary=np.array([[0.,10.],[10.,0.],[10.,0.]])
    same=np.zeros((3,3));same[0,1]=.9
    ports=np.zeros((3,3,3));ports[1,0,1]=ports[1,0,2]=-.8
    a=graph(nodes,unary,ports,[0,1,2],same=same)
    runs,_,_=choose_opportunities(**a);assert runs[-1][2]==2
    ports[1,0,2]=0.
    runs,_,secondary=choose_opportunities(**a)
    assert runs[-1][2]==1 and secondary==pytest.approx(.9)


def test_adding_empty_grid_nodes_does_not_invent_cut_or_duration_cost():
    results=[]
    for nodes in (np.array([0,20,40]),np.array([0,10,20,30,40])):
        n=len(nodes);a=graph(nodes,np.full((2,n-1),.2),np.full((n,2,2),10.),[0,1])
        results.append(choose_opportunities(**a))
    assert all(len(r[0])==1 and r[1]==pytest.approx(.8) for r in results)


def test_native_event_stays_in_long_hold_and_deadline_is_checked_in_dp():
    nodes=np.arange(0,61,10)
    a=graph(nodes,np.array([np.zeros(6),np.full(6,100.)]),np.full((7,2,2),100.),[0,1],
            events=[np.array([30]),np.array([],int)])
    runs,score,_=choose_opportunities(**a)
    assert len(runs)==1 and score==pytest.approx(.25)
    def stop():raise TimeoutError('predeclared selection deadline')
    with pytest.raises(TimeoutError,match='predeclared'):choose_opportunities(**a,check=stop)


def test_exact_lexicographic_dp_matches_independent_exhaustive_tiny_graph():
    rng=np.random.default_rng(181)
    nodes=np.array([0,10,20,30,40]);sources=np.array([0,0,1,2]);q=len(sources)
    unary=rng.uniform(0,.4,(q,4));ports=rng.uniform(-.8,2.,(5,q,q))
    through=rng.uniform(0,.1,(q,5));through[:,[0,-1]]=0
    same=np.zeros((q,q));same[:2,2]=same[2,:2]=.9
    events=[np.array([15]),np.array([],int),np.array([25])]
    candidates=[]
    def visit(first,path,p,d):
        if first==4:
            candidates.append((p,d,tuple(path)));return
        for end in range(first+1,5):
            duration=(nodes[end]-nodes[first])/10
            for state in range(q):
                if path and sources[path[-1][2]]==sources[state]:continue
                cost=float(unary[state,first:end].sum()+through[state,first+1:end].sum())
                cost+=native_event_cost(nodes[first],nodes[end],events[sources[state]],10)
                diversity=0.
                if path:
                    old=path[-1][2];cost+=ports[first,old,state]+.35;diversity+=same[old,state]
                    if len(path)>=2 and duration<3 and sources[path[-2][2]]==sources[state]:diversity+=.7
                visit(end,[*path,(first,end,state)],p+cost,d+diversity)
    visit(0,[],0.,0.)
    expected=min(candidates)
    runs,p,d=choose_opportunities(**graph(nodes,unary,ports,sources,through=through,events=events,same=same))
    assert p==pytest.approx(expected[0]) and d==pytest.approx(expected[1]) and runs==expected[2]


def test_music_strength_is_bound_without_altering_grid_or_inventing_downbeats():
    nodes=np.array([0,10,20,30]);before=nodes.copy()
    values=music_on_nodes(nodes,fps=10,master_start=5.,rhythm={'beat_grid':[
        dict(time_seconds=6.,strength=.2),dict(time_seconds=7.,strength=.9)]})
    assert values.tolist()==[0.,.2,.9,0.] and np.array_equal(nodes,before)
    assert not music_on_nodes(nodes,fps=10,master_start=5.,rhythm={}).any()
    with pytest.raises(ValueError):music_on_nodes(nodes,fps=10,master_start=5.,rhythm={'beat_grid':[
        dict(time_seconds=6.,strength=float('nan'))]})


def moving_bank_and_poses(wrong_wrist=False):
    nodes=np.array([0,18,36]);heads=np.zeros((3,2,8,4),np.float32);heads[:,:,0]=[.45,.3,.16,.95]
    vectors=np.zeros((3,2,3,128),np.float32);vectors[:,:,0,0]=1.
    bank=dict(heads=heads,vectors=vectors,zoom_cap=1.35,
        native_indices=np.clip(nodes[:,None]+[-1,0],0,35),
        pose_native_indices=np.clip(nodes[:,None]+[-7,-1,0,6],0,35))
    pose=np.array([[.4,.4,.9],[.5,.4,.9],[.38,.5,.9],[.52,.5,.9],
                   [.37,.6,.9],[.53,.6,.9],[.41,.7,.9],[.49,.7,.9]])
    windows=np.tile(pose,(4,1,1));windows[0,2:6,1]-=.06;windows[3,2:6,1]+=.06
    if wrong_wrist:windows[3,4,1]-=.12
    poses=np.zeros((3,4,3,8,3));poses[:,:,0]=windows[None]
    return nodes,bank,poses


def test_actual_pose_to_port_producer_opposed_wrist_loses_credit_not_source():
    nodes,bank,good=moving_bank_and_poses();_,_,wrong=moving_bank_and_poses(True)
    states,rows,_=make_states(nodes,[bank,bank,bank],[good,good,wrong],fps=10)
    left=rows[states.index(State(0,0,'native'))]
    good_row=rows[states.index(State(1,0,'native'))];bad_row=rows[states.index(State(2,0,'native'))]
    exact=port_cost(1,left,good_row,fps=10,music_strength=1.)
    bad=port_cost(1,left,bad_row,fps=10,music_strength=1.)
    assert exact['connection_opportunity']['credit']>0
    assert bad['connection_opportunity']['credit']==0 and np.isfinite(bad['nonnegative_risk_cost'])
    assert State(2,0,'native') in states
    good_row['pose_native_indices']=None
    assert port_cost(1,left,good_row,fps=10,music_strength=1.)['connection_opportunity']['credit']==0


def test_real_selector_consumes_music_and_reports_exact_primary_secondary_costs():
    nodes,bank,poses=moving_bank_and_poses()
    _,_,_,score,receipt=select_fixed_holds(nodes,[bank,bank],[poses,poses],fps=10,
        source_events=[np.array([],int),np.array([],int)],broadcast_slots=['same','same'],
        check=lambda:None,music_strengths=np.array([0.,1.,0.]))
    assert receipt['experimental_opportunity_mode'] is True
    assert receipt['editorial_maximum_hold_removed'] is True
    assert receipt['selected_hold_cost_decomposition'][-1]['cumulative_cost']==pytest.approx(score)
    assert receipt['port_matrices_built_once_per_node'] is True
