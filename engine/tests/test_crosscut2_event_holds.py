import itertools
import numpy as np
import pytest
from crosscut2.event_ports import Ports,event_widths
from crosscut2.event_states import build
from crosscut2.event_holds import fragment_costs,select,realize
from crosscut2.fixed_holds import State
from crosscut2.opportunity_dp import choose_opportunities


def inputs(count=2,one_arm=False,opposed_source=None):
    nodes=np.array([0,18,36,54]);banks=[];poses=[];styles=[]
    for source in range(count):
        heads=np.zeros((4,2,8,4),np.float32);heads[:,:,0]=[.5,.3,.25,.95]
        vectors=np.zeros((4,2,3,128),np.float32);vectors[:,:,0,0]=1.
        native=np.clip(nodes[:,None]+[-1,0],0,53)
        banks.append(dict(heads=heads,vectors=vectors,native_indices=native,zoom_cap=1.35,
            pose_native_indices=np.clip(nodes[:,None]+[-7,-1,0,6],0,53),
            native_readability=np.tile([.1,.8,0.,200.],(4,2,8,1)),
            native_readability_valid=np.ones((4,2,8),bool)))
        base=np.array([[.4,.4,.9],[.6,.4,.9],[.35,.5,.9],[.65,.5,.9],
            [.3,.55,.9],[.7,.55,.9],[.43,.7,.9],[.57,.7,.9]])
        four=np.tile(base,(4,1,1));four[0,2:6,1]-=.06;four[3,2:6,1]+=.06
        if one_arm:four[:,[1,3,5,6,7],2]=0.
        if source==opposed_source:four[3,4,1]-=.16
        value=np.zeros((4,4,3,8,3));value[:,:,0]=four;poses.append(value)
        colour=np.zeros((4,2,8,2,40),np.float32);colour[:,:,0,:,source%40]=1.
        valid=np.zeros((4,2,8),bool);valid[:,:,0]=True
        styles.append(dict(colour=colour,colour_valid=valid,presence=np.tile([0.,.2],(4,2,1))))
    return nodes,banks,poses,styles


def prepared(**kw):
    nodes,banks,poses,styles=inputs(**kw);events=[np.array([],int)]*len(banks)
    states,data,receipt=build(nodes,banks,poses,styles,fps=10.,source_events=events,
        body_states=[np.ones((4,4,3),int)]*len(banks))
    ports=Ports(nodes,states,data,fps=10.,music=np.array([0.,1.,.8,0.]))
    return nodes,banks,poses,styles,events,states,data,ports,receipt


def test_one_complete_moving_arm_has_match_benefit_without_legacy_four_joint_gate():
    _,_,_,_,_,states,_,ports,_=prepared(one_arm=True)
    a=states.index(State(0,0,'native'));b=states.index(State(1,0,'native'))
    result=ports.selected(1,a,b)
    assert result['moving_connection_supported'] and result['benefit']>0
    assert result['worst_phase_loss']==pytest.approx(0.) and result['worst_motion_loss']==pytest.approx(0.)
    assert result['loss']==pytest.approx(0.) and result['cost']<0


def test_natural_one_frame_movement_is_not_a_cut_error_and_copy_has_no_style_gain():
    nodes,banks,poses,styles=inputs();poses=[p.copy() for p in poses]
    for p in poses:p[:,2:,0,2:6,1]+=.005
    styles[1]={k:v.copy() for k,v in styles[0].items()}
    states,data,_=build(nodes,banks,poses,styles,fps=10.,source_events=[np.array([],int)]*2,
        body_states=[np.ones((4,4,3),int)]*2)
    ports=Ports(nodes,states,data,fps=10.,music=[0.,1.,1.,0.]);a=states.index(State(0,0,'native'));b=states.index(State(1,0,'native'))
    result=ports.selected(1,a,b)
    assert result['loss']==pytest.approx(0.) and result['benefit']==pytest.approx(0.)
    assert result['style_distance']==pytest.approx(0.)


def test_one_opposed_wrist_window_cannot_be_averaged_away_or_rewarded():
    _,_,_,_,_,states,_,ports,_=prepared(count=3,opposed_source=2)
    a=states.index(State(0,0,'native'));b=states.index(State(2,0,'native'))
    result=ports.selected(1,a,b)
    assert result['opposed_motion'] and not result['moving_connection_supported']
    assert result['benefit']==0 and result['worst_motion_loss']>=.5 and np.isfinite(result['cost'])


def test_missing_pose_is_finite_not_stationary_and_does_not_remove_source():
    nodes,banks,poses,styles=inputs();poses[1][:]=0
    states,data,_=build(nodes,banks,poses,styles,fps=10.,source_events=[np.array([],int)]*2,
        body_states=[np.ones((4,4,3),int)]*2)
    ports=Ports(nodes,states,data,fps=10.,music=[0.,1.,1.,0.]);a=states.index(State(0,0,'native'));b=states.index(State(1,0,'native'))
    result=ports.selected(1,a,b)
    assert result['missing_fraction']>=.5 and not result['moving_connection_supported']
    assert np.isfinite(ports.matrix(1)).all() and State(1,None,'native') in states


def test_event_duration_does_not_change_on_neutral_node_insertion():
    a=event_widths([0,10,20,30],[0,1.,.7,0],fps=10)
    b=event_widths([0,5,10,15,20,25,30],[0,0,1.,0,.7,0,0],fps=10)
    assert np.array_equal(a[[1,2]],b[[2,4]])


def test_exact_signed_event_dp_against_exhaustive_hold_enumeration():
    rng=np.random.default_rng(132);nodes=np.array([0,10,20,30,40]);sources=np.array([0,0,1])
    unary=rng.uniform(0,.3,(3,4));through=rng.uniform(-.1,.1,(3,5));ports=rng.uniform(-.2,.5,(5,3,3))
    candidates=[]
    def visit(first,path,cost):
        if first==4:candidates.append((cost,tuple(path)));return
        for end in range(first+1,5):
            for state in range(3):
                if path and sources[path[-1][2]]==sources[state]:continue
                value=float(unary[state,first:end].sum()+through[state,first+1:end].sum())
                if path:value+=ports[first,path[-1][2],state]+.1
                visit(end,[*path,(first,end,state)],cost+value)
    visit(0,[],0.)
    runs,cost,_=choose_opportunities(nodes,fps=10.,unary=unary,ports=lambda n:ports[n],state_sources=sources,
        source_events=[np.array([],int)]*2,same_show=np.zeros((3,3)),through=through,
        minimum_hold=.4,event_seconds=True,native_cost=lambda a,b:np.zeros(2))
    assert cost==pytest.approx(min(candidates)[0]) and runs==min(candidates)[1]


def test_source_copy_renaming_and_full_producer_dp_keep_coherent_decomposition():
    nodes,banks,poses,styles=inputs();events=[np.array([],int)]*2
    native=[dict(frames=np.array([],int),heads=np.zeros((0,2,8,4)))]*2
    runs,states,data,score,receipt,ports=select(nodes,banks,poses,styles,fps=10.,source_events=events,
        body_states=[np.ones((4,4,3),int)]*2,native_ports=native,music=np.array([0.,1.,.8,0.]))
    assert len(runs)>1 and receipt['selected_hold_cost_decomposition'][-1]['cumulative_cost']==pytest.approx(score)
    assert all('unary' not in row and 'through' not in row for row in data)
    for row in data:
        assert np.allclose(row['curve'],row['curve'][0])
        assert np.isfinite(row['event_loss_rate']).all()


def test_native_fragment_loss_is_seconds_of_missing_support_only():
    nodes=np.array([0,2,10,18,20]);f=fragment_costs(nodes,[np.array([10])],fps=10.,support_seconds=.4)
    assert f(0,4)[0]==0 and f(1,3)[0]==0
    assert f(0,1)[0]==pytest.approx(.2) and f(3,4)[0]==pytest.approx(.2)
