import numpy as np
import pytest
from crosscut2.preference_dp import choose


def args():
    return dict(nodes=np.array([0,20,40,60]),fps=10.,unary=np.zeros((2,3)),through=np.zeros((2,4)),
        preference_holds=np.array([[0.,.3,.3,0.],[0.,.3,.3,0.]]),
        preference_intervals=np.zeros((2,3)),preference_ports=lambda n:np.array([[0.,.1],[.1,0.]]),
        state_sources=np.array([0,1]),source_events=[[],[]],minimum_hold=.4,native_cost=lambda a,b:np.zeros(2))


def test_visibility_cannot_buy_a_preference_inferior_seam_but_good_cut_can_beat_hold():
    kw=args();good,score=choose(**kw)
    assert len(good)>1 and score[0]<.6
    kw['preference_ports']=lambda n:np.ones((2,2))
    baseline,score=choose(**kw)
    assert len(baseline)==1
    for gain in (1.,1e4,1e12):
        kw['unary']=np.array([[0.,gain,gain],[gain,0.,0.]])
        result,cost=choose(**kw)
        assert len(result)==1 and cost[0]==score[0]


@pytest.mark.parametrize('fragment_in_primary',[False,True])
def test_exact_two_objective_solution_matches_complete_hold_enumeration(fragment_in_primary):
    kw=args();rng=np.random.default_rng(171)
    kw['fragment_in_primary']=fragment_in_primary
    kw['native_cost']=lambda a,b:np.array([.03*(b-a),.02*(a+1)])
    kw['unary']=rng.uniform(0,1,(2,3));kw['through']=rng.uniform(0,.3,(2,4))
    kw['preference_holds']=rng.uniform(0,1,(2,4));kw['preference_intervals']=rng.uniform(0,.2,(2,3))
    matrix=rng.uniform(0,1,(4,2,2));kw['preference_ports']=lambda n:matrix[n]
    candidates=[]
    def visit(first,path,cost):
        if first==3:candidates.append((cost,tuple(path)));return
        for end in range(first+1,4):
            for state in range(2):
                if path and path[-1][2]==state:continue
                p=kw['preference_intervals'][state,first:end].sum()+kw['preference_holds'][state,first+1:end].sum()
                s=(2*kw['unary'][state,first:end].sum()+kw['through'][state,first+1:end].sum())
                if fragment_in_primary:p+=kw['native_cost'](first,end)[state]
                else:s+=kw['native_cost'](first,end)[state]
                if path:p+=matrix[first,path[-1][2],state];s+=.1
                visit(end,[*path,(first,end,state)],(cost[0]+p,cost[1]+s))
    visit(0,[],(0.,0.));runs,score=choose(**kw)
    assert score==pytest.approx(min(candidates)[0]) and runs==min(candidates)[1]


def test_same_event_hold_replaced_by_cut_without_double_native_event_charge():
    kw=args();kw['preference_holds'][:]=0.;kw['preference_holds'][:,1]=3.
    kw['preference_ports']=lambda n:np.full((2,2),.2 if n==1 else 10.)
    runs,score=choose(**kw)
    assert len(runs)==2 and runs[0][1]==1
    assert score[0]==pytest.approx(.2)


def test_identical_state_copy_does_not_change_event_objective_or_add_reward():
    kw=args();_,score=choose(**kw)
    for key in ('unary','through','preference_holds','preference_intervals'):
        kw[key]=kw[key][[0,0,1]]
    kw['state_sources']=np.array([0,0,1]);kw['preference_ports']=lambda n:np.full((3,3),.1)
    _,duplicated=choose(**kw)
    assert duplicated==pytest.approx(score)


def test_explicit_local_admission_cannot_be_reopened_by_arbitrary_future_content():
    for magnitude in (6.,60.,1e12):
        kw=args();kw['preference_holds'][:]=0.;kw['unified_content']=True;kw['fragment_in_primary']=True
        kw['unary']=np.array([[0,magnitude,magnitude],[magnitude,0.,0.]])
        kw['cut_allowed']=lambda n:np.zeros((2,2),bool)
        runs,_=choose(**kw)
        assert len(runs)==1


def test_signed_good_connection_and_state_sized_native_hold_are_in_unified_objective():
    kw=args();kw['preference_holds'][:]=0.;kw['unified_content']=True;kw['fragment_in_primary']=True
    kw['preference_ports']=lambda n:np.array([[0.,-.1],[-.1,0.]])
    kw['cut_allowed']=lambda n:np.ones((2,2),bool)
    kw['native_hold_cost']=lambda a,b:np.full(2,.02*(b-a))
    runs,score=choose(**kw)
    assert len(runs)==3 and score[0]==pytest.approx(-.14) and score[1]==pytest.approx(.2)


def test_identical_physical_content_and_zero_cut_gain_does_not_create_floating_phantom_cuts():
    kw=args();kw['preference_holds'][:]=0.;kw['unified_content']=True;kw['fragment_in_primary']=True
    kw['unary']=np.tile([.038278319,.267393837,.715310993],(2,1))
    kw['preference_ports']=lambda n:np.zeros((2,2));kw['cut_allowed']=lambda n:np.ones((2,2),bool)
    runs,score=choose(**kw)
    assert len(runs)==1 and score[1]==0.


def test_unified_masked_signed_graph_matches_exhaustive_paths_with_state_native_holds():
    rng=np.random.default_rng(173)
    for _ in range(20):
        nodes=np.arange(5)*12;q=3;sources=np.array([0,1,0]);rates=rng.uniform(-.1,1.,(q,4))
        K=rng.uniform(-.1,.2,(5,q,q));mask=rng.random((5,q,q))>.4
        native=rng.uniform(0,.1,(5,5,q));found=[]
        fragment=lambda a,b:np.array([.01*(b-a),.02*(a+1)])
        def walk(first,path,cost,tie):
            if first==4:found.append(((cost,tie),tuple(path)));return
            for end in range(first+1,5):
                for state in range(q):
                    if path and (sources[path[-1][2]]==sources[state] or not mask[first,path[-1][2],state]):continue
                    entry=0. if not path else K[first,path[-1][2],state]
                    content=float((rates[state,first:end]*.4).sum())
                    value=cost+entry+content+native[first,end,state]+fragment(first,end)[sources[state]]
                    walk(end,[*path,(first,end,state)],value,tie+(1/30 if path else 0.))
        walk(0,[],0.,0.)
        runs,score=choose(nodes,fps=30.,unary=rates,through=np.zeros((q,5)),
            preference_holds=np.zeros((q,5)),preference_intervals=np.zeros((q,4)),
            preference_ports=lambda n:K[n],state_sources=sources,source_events=[[],[]],
            minimum_hold=.4,native_cost=fragment,fragment_in_primary=True,
            cut_allowed=lambda n:mask[n],native_hold_cost=lambda a,b:native[a,b],unified_content=True)
        expected=min(found)
        assert score==pytest.approx(expected[0],abs=1e-12) and runs==expected[1]
