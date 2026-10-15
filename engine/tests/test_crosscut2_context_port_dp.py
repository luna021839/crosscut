import numpy as np
import pytest
from crosscut2.context_port_dp import ArcRef,solve
from crosscut2.sequence_context import Evidence,PRIMITIVES,FEATURE_NAMES,windows


def domains(nodes):return [np.arange(2)]*len(nodes)


def zero_cut(n):return np.zeros((2,2)),~np.eye(2,dtype=bool)


def zero_credit(i,j):return np.zeros(2)


def test_more_expensive_camera_prefix_survives_until_actual_future_context_is_known():
    nodes=np.array([0,30,60]);ids=domains(nodes)
    def hold(i,j):
        cost=np.zeros((2,2,2));cost[:,:,1]=.1 if i==0 else 0.
        valid=np.broadcast_to(np.eye(2,dtype=bool)[...,None],cost.shape).copy()
        if i==0 and j==2:valid[:]=False
        return cost,valid
    def context(histories,arc):
        return np.array([float(arc.first==1 and h[0].camera==0) for h in histories])
    args=dict(start_cost=[0.,0.],start_allowed=[True,False],final_allowed=[False,True],
        hold_cost=hold,cut_cost=zero_cut,context_cost=context,episodes=[0,0,1],
        credit=zero_credit,history_frames=120)
    runs,objective,receipt=solve(nodes,ids,ids,label_limit=2,**args)
    assert len(runs)==2 and objective[0]==pytest.approx(.1)
    assert receipt['selected_arc_codes']==[1,0]
    assert not receipt['scalar_cost_or_debt_dominance']
    assert not receipt['full_frame_arrays_per_label']
    assert sum(receipt['selected_context_costs'])==0.
    # An explicit one-label approximation loses this future-dependent choice;
    # the receipt must disclose it rather than claim an exact global optimum.
    _,narrow,bound=solve(nodes,ids,ids,label_limit=1,**args)
    assert narrow[0]==pytest.approx(1.)
    assert bound['labels_discarded_by_bound_events']>0 and not bound['exact_global_optimum_claimed']


def test_fixed_global_time_cost_does_not_increase_or_pay_more_after_alias_hold_splits():
    nodes=np.array([0,30,60]);ids=domains(nodes);ends=np.arange(6,61,6)
    def hold(i,j):return np.zeros((2,2,1)),np.eye(2,dtype=bool)[...,None]
    def context(histories,arc):
        cost=.75*6/30.*((ends>nodes[arc.first])&(ends<=nodes[arc.final])).sum()
        return np.full(len(histories),cost)
    kwargs=dict(start_cost=[0.,0.],start_allowed=[True,False],hold_cost=hold,
        cut_cost=zero_cut,context_cost=context,episodes=[0,0,1],credit=zero_credit,history_frames=120)
    a,av,_=solve(nodes,ids,ids,final_allowed=[True,False],**kwargs)
    b,bv,receipt=solve(nodes,ids,ids,final_allowed=[False,True],**kwargs)
    assert len(a)==1 and len(b)==2 and av[0]==bv[0]==pytest.approx(1.5)
    assert sum(receipt['selected_context_costs'])==pytest.approx(1.5)


def test_once_only_action_opportunity_bit_is_not_reset_by_an_extra_cut():
    nodes=np.array([0,12,24,36]);ids=domains(nodes)
    def hold(i,j):return np.zeros((2,2,1)),np.eye(2,dtype=bool)[...,None]
    _,objective,receipt=solve(nodes,ids,ids,start_cost=[0.,0.],start_allowed=[True,False],
        hold_cost=hold,cut_cost=zero_cut,context_cost=lambda past,arc:np.zeros(len(past)),
        episodes=[0,0,0,1],credit=lambda i,j:np.ones(2) if i else np.zeros(2),history_frames=120)
    assert objective==(-1.,1)
    claims=[v for v in receipt['opportunity_claims'] if v['claimed']]
    assert len(claims)==1 and claims[0]['episode']==0 and claims[0]['credit']==1.


def test_expired_history_is_not_copied_into_every_future_label():
    nodes=np.arange(0,61,12);ids=domains(nodes);seen=[]
    def hold(i,j):
        valid=np.eye(2,dtype=bool)[...,None].copy()
        if j!=i+1:valid[:]=False
        return np.zeros((2,2,1)),valid
    def context(histories,arc):
        for h in histories:
            assert all(nodes[r.final]>nodes[arc.first]-24 for r in h)
            seen.append(len(h))
        return np.zeros(len(histories))
    runs,_,receipt=solve(nodes,ids,ids,start_cost=[0.,0.],start_allowed=[True,False],
        hold_cost=hold,cut_cost=zero_cut,context_cost=context,episodes=np.zeros(len(nodes),int),
        credit=zero_credit,history_frames=24)
    assert len(runs)==5 and max(seen)<=2 and receipt['shared_suffix_histories']>1


@pytest.mark.parametrize('bounded',[False,True])
def test_numeric_whole_sequence_objective_matches_independent_exhaustive_enumeration(bounded):
    nodes=np.arange(0,49,12);ids=domains(nodes);rng=np.random.default_rng(58)
    base=rng.uniform(0.,.2,(len(nodes),len(nodes),2,2))
    cut=rng.uniform(0.,.1,(len(nodes),2,2))
    def hold(i,j):
        values=np.zeros((2,2,2));values[0,0]=base[i,j,0];values[1,1]=base[i,j,1]
        return values,np.broadcast_to(np.eye(2,dtype=bool)[...,None],values.shape)
    def cut_cost(j):return cut[j],~np.eye(2,dtype=bool)
    def whole_context(path,final):
        frames=np.concatenate([np.arange(nodes[r.first],nodes[r.final]) for r in path])
        values=np.zeros((len(frames),len(PRIMITIVES)));known=np.zeros(values.shape,bool)
        k=PRIMITIVES.index('additional_log_zoom');offset=0
        for r in path:
            n=int(nodes[r.final]-nodes[r.first]);values[offset:offset+n,k]=.05*(r.entry+1)+.1*r.camera
            known[offset:offset+n,k]=True;offset+=n
        report=windows(Evidence(frames,values,known),final_frame=final,fps=30.)
        lo=report['features'][:,FEATURE_NAMES.index('additional_log_zoom.observed_minimum')]
        hi=report['features'][:,FEATURE_NAMES.index('additional_log_zoom.observed_maximum')]
        mean=report['features'][:,FEATURE_NAMES.index('additional_log_zoom.observed_mean')]
        # A numeric nonlinear context fixture, NOT fitted aesthetic weights.
        return report,mean**2+4*(hi-lo)**2
    def context(histories,arc):
        result=[]
        for history in histories:
            report,cost=whole_context((*history,arc),int(nodes[arc.final]))
            fresh=report['interval_final']>nodes[arc.first]
            result.append(float(cost[fresh]@report['interval_seconds'][fresh]))
        return np.asarray(result)
    all_paths=[]
    def enumerate_paths(i,source,path,primary):
        for j in range(i+1,len(nodes)):
            for camera in range(2):
                arc=ArcRef(i,j,source,source,camera);new=(*path,arc)
                cost=primary+base[i,j,source,camera]
                if j==len(nodes)-1:
                    if source==1:
                        report,sequence=whole_context(new,int(nodes[-1]))
                        all_paths.append((cost+float(sequence@report['interval_seconds']),len(new)-1,new))
                else:enumerate_paths(j,1-source,new,cost+cut[j,source,1-source])
    enumerate_paths(0,0,(),0.)
    best=min(all_paths,key=lambda row:row[:2])
    runs,objective,receipt=solve(nodes,ids,ids,start_cost=[0.,0.],start_allowed=[True,False],
        final_allowed=[False,True],hold_cost=hold,cut_cost=cut_cost,context_cost=context,
        episodes=np.zeros(len(nodes),int),credit=zero_credit,history_frames=120,label_limit=32,
        context_lower_bound=(lambda arc:0.) if bounded else None)
    assert objective[0]==pytest.approx(best[0],abs=1e-12) and objective[1]==best[1]
    assert receipt['no_beam_pruning_in_this_finite_domain']
    actual=tuple(ArcRef(*run,code) for run,code in zip(runs,receipt['selected_arc_codes']))
    report,sequence=whole_context(actual,int(nodes[-1]))
    assert sum(receipt['selected_context_costs'])==pytest.approx(float(sequence@report['interval_seconds']),abs=1e-12)


def test_identical_physical_suffix_keys_may_merge_but_source_and_paid_ports_stay_separate():
    nodes=np.array([0,30,60]);ids=domains(nodes)
    def hold(i,j):return np.zeros((2,2,2)),np.broadcast_to(np.eye(2,dtype=bool)[...,None],(2,2,2))
    # In this fixture the two codes produce exactly the same physical camera.
    def same_physical(history,end):return tuple((r.first,r.final,r.entry,r.exit) for r in history)
    _,objective,receipt=solve(nodes,ids,ids,start_cost=[0.,0.],hold_cost=hold,cut_cost=zero_cut,
        context_cost=lambda past,arc:np.zeros(len(past)),episodes=[0,0,1],credit=zero_credit,
        history_frames=120,history_key=same_physical)
    assert objective==(0.,0) and receipt['equivalent_histories_merged']>0
    assert receipt['history_merge_policy']=='caller_supplied_future_equivalent_physical_suffix'


def test_invalid_context_or_workspace_is_not_silently_retried_or_reduced():
    nodes=np.array([0,30]);ids=domains(nodes)
    def hold(i,j):return np.zeros((2,2,1)),np.eye(2,dtype=bool)[...,None]
    kw=dict(start_cost=[0.,0.],hold_cost=hold,cut_cost=zero_cut,
        context_cost=lambda past,arc:np.full(len(past),np.nan),episodes=[0,1],credit=zero_credit,history_frames=120)
    with pytest.raises(ValueError,match='finite incremental'):solve(nodes,ids,ids,**kw)
    with pytest.raises(MemoryError,match='workspace exceeded'):
        solve(nodes,ids,ids,workspace_limit_bytes=64,**kw)


def test_proved_nonnegative_context_bound_preserves_same_beam_result_and_once_paid_credit():
    nodes=np.arange(0,73,12);ids=domains(nodes);rng=np.random.default_rng(92)
    values=rng.uniform(0.,1.,(len(nodes),len(nodes),2,2,3))
    def hold(i,j):return values[i,j],np.broadcast_to(np.eye(2,dtype=bool)[...,None],(2,2,3))
    def context(histories,arc):
        return np.array([.1*arc.camera+.03*sum(r.camera for r in history) for history in histories])
    args=dict(start_cost=[0.,0.],start_allowed=[True,False],final_allowed=[False,True],
        hold_cost=hold,cut_cost=zero_cut,context_cost=context,history_frames=120,label_limit=4,
        episodes=[0,0,0,1,1,1,2],credit=lambda i,j:np.full(2,.15 if i else 0.))
    plain=solve(nodes,ids,ids,**args);bounded=solve(nodes,ids,ids,context_lower_bound=lambda arc:0.,**args)
    assert plain[:2]==bounded[:2]
    for name in ('selected_arc_codes','selected_context_costs','opportunity_claims'):
        assert plain[2][name]==bounded[2][name]
    assert bounded[2]['prefix_context_evaluations_skipped_by_explicit_lower_bound']>0
    assert bounded[2]['context_prefix_evaluations']<plain[2]['context_prefix_evaluations']
    assert not bounded[2]['scalar_cost_or_debt_dominance']
    with pytest.raises(ValueError,match='finite explicit context lower bound'):
        solve(nodes,ids,ids,context_lower_bound=lambda arc:np.nan,**args)
    with pytest.raises(ValueError,match='violated its explicit lower bound'):
        solve(nodes,ids,ids,context_lower_bound=lambda arc:100.,**args)
