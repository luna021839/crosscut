import json
import numpy as np
import pytest
from crosscut2.scene_readout import IDENTITY,roi_readability,then,change,recovery,apply,TransferTree,SceneReadout
from crosscut2.display_source_catalogue import SourceCatalogue
from crosscut2.played_partial_pose import features,matrix
from crosscut2.editorial_roles import event_energy
from crosscut2.boundary_port_dp import solve


def catalogue(nodes, *, height=.12, black=.8, edge=.5, known=True):
    n=len(nodes);heads=np.zeros((n,2,8,4));heads[:,:,0]=[.5,.35,height,.99]
    vectors=np.zeros((n,2,3,128));vectors[:,:,0,0]=1.
    values=np.tile([.075,edge,0.,120.],(n,2,8,1))
    bank=dict(heads=heads,vectors=vectors,zoom_cap=1.5,
        native_indices=np.clip(np.asarray(nodes)[:,None]+[-1,0],0,nodes[-1]-1),
        pose_native_indices=np.clip(np.asarray(nodes)[:,None]+[-7,-1,0,6],0,nodes[-1]-1),
        native_readability=values,native_readability_valid=np.full((n,2,8),known))
    presence=np.zeros((n,2,2));presence[:,:,0]=black
    return SourceCatalogue(nodes,bank,None,[],dict(presence=presence),[],fps=30.,editorial_content=True)


def test_T1_actual_readable_ROI_cannot_purchase_3_7x_jump_with_background_blackness():
    nodes=np.unique(np.r_[np.arange(0,631,6),30]);n=int(np.where(nodes==30)[0][0])
    a=catalogue(nodes,black=.81);b=catalogue(nodes,height=.12*1.5**3.232)
    strip=np.tile([1.,.5,.5],(7,1))
    detail=matrix(features([a.observation(n,0,strip)],fps=30.),
        features([b.observation(n,1,strip)],fps=30.),details=True)
    k=event_energy(detail,.2,fps=30.,point_geometry=True)[0,0]
    assert detail['matched_action_change_density'][0,0]==0 and k==pytest.approx(.7464)
    ticks=np.array([0,30,630]);ids=[np.arange(2)]*3
    def hold(i,j):
        frames=np.arange(ticks[i],ticks[j]);cam=np.tile([1.,.5,.5],(len(frames),1))
        cost=np.array([c.rates(frames,cam).sum()/30. for c in (a,b)])
        return np.broadcast_to(cost[:,None],(2,2)),np.eye(2,dtype=bool)
    runs,_,_=solve(ticks,ids,ids,start_cost=[0.,0.],start_allowed=[True,False],
        hold_cost=hold,cut_cost=lambda n:(np.full((2,2),k),~np.eye(2,dtype=bool)))
    assert len(runs)==1 and runs[0][2]==0
    good=catalogue(nodes);blur=catalogue(nodes,edge=.125);unknown=catalogue(nodes,known=False)
    assert good.roi_readability[1,0,0]==1.
    assert blur.roi_readability[1,0,0]<unknown.roi_readability[1,0,0]<1.
    assert not unknown.roi_readability_valid.any()
    good.bank['native_readability'][:,:,0,2]=1.
    assert roi_readability(good.bank)[0][1,0,0]<1.  # real target exposure deficit retained


def test_T3_transfer_composition_recovery_and_UNKNOWN_do_not_reset_at_virtual_splits():
    first=change(2.,unit=.2);last=change(2.,unit=.2)
    fast=then(then(first,recovery(.5,tau=2.)),last)
    slow=then(then(first,recovery(5.,tau=2.)),last)
    assert apply(fast,0.)[1]>apply(slow,0.)[1]>0.
    unknown=then(then(first,recovery(0.,tau=2.)),last)
    assert apply(unknown,0.)[1]>apply(fast,0.)[1]
    rng=np.random.default_rng(2);steps=[]
    for amount,ell in zip(rng.uniform(0,1,100),rng.uniform(0,.04,100)):
        steps.append(then(change(amount,unit=.2),recovery(ell,tau=2.)))
    steps=np.asarray(steps)[None];tree=TransferTree(steps)
    for i,j,k in ((0,1,100),(0,50,100),(7,31,85)):
        np.testing.assert_allclose(tree.query(i,k),then(tree.query(i,j),tree.query(j,k)),rtol=1e-12)
        sequential=IDENTITY
        for t in range(i,k):sequential=then(sequential,steps[0,t])
        np.testing.assert_allclose(tree.query(i,k)[0],sequential,rtol=1e-12)
    # No episode/native boundary parameter can erase state in this algebra.
    np.testing.assert_array_equal(then(first,IDENTITY),first)
    # Exhaustively enumerable future-camera counterexample: a cheap camera
    # leaves debt5; paying .1 for the other camera saves1 at the next cut.
    from crosscut2.readout_port_dp import solve as stateful
    nodes=np.array([0,30,60]);ids=[np.arange(2)]*3
    def hold(i,j):
        cost=np.zeros((2,2,2));cost[:,:,1]=.1 if i==0 else 0.
        allowed=np.broadcast_to(np.eye(2,dtype=bool)[...,None],cost.shape).copy()
        transfer=np.broadcast_to(IDENTITY,cost.shape+(4,)).copy()
        if i==0 and j==1:transfer[:,:,0]=change(5.,unit=.2)
        if i==0 and j==2:allowed[:]=False
        return cost,allowed,transfer
    def cut(n):return np.zeros((2,2)),~np.eye(2,dtype=bool)
    runs,objective,receipt=stateful(nodes,ids,ids,start_cost=[0.,0.],start_allowed=[True,False],
        hold_cost=hold,cut_cost=cut,cut_transfer=lambda n:change(np.ones((2,2)),unit=.2),
        episodes=[0,0,1],credit=lambda i,j:np.array([0.,.5]) if i else np.zeros(2),final_allowed=[False,True])
    assert receipt['selected_arc_codes'][0]==1
    enumerated=[(cost+apply(change(1.,unit=.2),debt)[1]-.5,code)
        for code,cost,debt in ((0,0.,5.),(1,.1,0.))]
    assert objective[0]==pytest.approx(min(enumerated)[0])
    assert objective[0]==pytest.approx(-.4) and len(runs)==2
    assert sum(v['claimed'] for v in receipt['opportunity_claims'])==1
    assert not receipt['exact_global_optimum_claimed'] and not receipt['full_frame_arrays_per_label']


def test_T2_original_head_reframing_not_arm_motion_or_invented_dense_recovery():
    nodes=np.arange(0,121,6);still=catalogue(nodes);moving=catalogue(nodes);arms=catalogue(nodes)
    pattern=np.arange(len(nodes))%2
    moving.bank['heads'][:,:,0,0]=(.3+.4*pattern)[:,None]
    moving.bank['heads'][:,:,0,2]=(.12+.2*pattern)[:,None]
    arms.body[:,:,:,4,0]=np.arange(len(nodes))[:,None,None]%2
    paths=np.tile([1.,.5,.5],(4,120,1))
    native=dict(frames=np.array([],int),heads=np.zeros((0,2,8,4)))
    fields=[SceneReadout(c,native,[],paths,tau=2.) for c in (still,moving,arms)]
    summaries=[f.summary(0,paths[:1],hold_first=0)[0] for f in fields]
    from crosscut2.scene_readout import compose_steps
    np.testing.assert_allclose(summaries[1],compose_steps(fields[1].steps(0,paths[:1],hold_first=0))[0])
    assert summaries[0][1]==summaries[2][1]==0. and summaries[1][1]>0.
    unknown=catalogue(nodes,known=False);f=SceneReadout(unknown,native,[],paths,tau=2.)
    assert f.summary(0,paths[:1],hold_first=0)[0,0]==1.  # no measured recovery
    assert not f.receipt()['dense_head_track_or_true_camera_cause_certified']
    assert f.receipt()['target_changes_are_attention_impulses_not_actor_velocity']
    # Split only the composition, NOT the physical programme/observation field.
    np.testing.assert_allclose(fields[1].base(0,120),
        then(fields[1].base(0,59),fields[1].base(59,120)),rtol=1e-12)
    # Actual original-cut head flanks use the same impulse as an authored
    # change, followed by that same one observed output frame of exposure.
    cat=catalogue(nodes);ticks=np.clip(nodes[:,None]+[-1,0],0,119)
    cat.bank['heads'][:,:,0,2]=np.where(ticks>=60,.24,.12)
    n=int(np.where(nodes==60)[0][0]);strip=np.tile([1.,.5,.5],(7,1))
    raw=dict(frames=np.array([60]),heads=cat.bank['heads'][n:n+1].copy(),observed=np.array([True]))
    native=SceneReadout(cat,raw,[60],paths,tau=2.)
    detail=matrix(features([cat.observation(n,0,strip)],fps=30.),
        features([cat.observation(n,1,strip)],fps=30.),details=True)
    expected=then(change(detail['face_rate'][0,0],unit=.2),recovery(1/30.,tau=2.))
    np.testing.assert_allclose(native.steps(60,paths[:1,:1],hold_first=0,baseline=paths[0])[0,0],expected)


def test_T4_all_camera_alternatives_transfer_and_cost_equal_exact_replay():
    from crosscut2.boundary_port_domain import make
    from crosscut2.boundary_arc_costs import ArcCosts
    from crosscut2.played_event_support import PlayedEventSupport
    nodes=np.array([0,60,120,180]);cats=[catalogue(nodes),catalogue(nodes,height=.14)]
    paths=[[(nodes,np.tile([z,.5,.5],(4,1))) for z in (1.,1.1,1.2,1.3)] for c in cats]
    entries,exits,dense,_=make(nodes,cats,paths,fps=30.)
    native=[dict(frames=np.array([t]),heads=c.bank['heads'][1:2].copy(),observed=np.array([True]))
        for t,c in zip((65,85),cats)]
    for bank in native:bank['heads'][:,1,0,2]*=.8
    support=PlayedEventSupport([[65],[85]],final=180,fps=30.)
    arcs=ArcCosts(nodes,entries,exits,cats,dense,native,support,lambda i,j:np.zeros(2),fps=30.,readout_tau=2.,
        opportunity_episodes=np.array([0,0,1,2]))
    tested=0;whole=0
    for i in range(len(nodes)-1):
        for j in range(i+1,len(nodes)):
            cost,allowed,transfer=arcs.alternatives(i,j)
            for p,q,code in zip(*np.nonzero(allowed)):
                s=entries[i][p].source;path=arcs.replay(i,j,p,q,int(code))
                assert cost[p,q,code]==pytest.approx(arcs.direct_cost(s,int(nodes[i]),path[None])[0])
                direct=arcs.readouts[s].summary(int(nodes[i]),path[None],hold_first=int(nodes[i]))[0]
                np.testing.assert_allclose(transfer[p,q,code],direct,rtol=1e-10,atol=1e-10)
                tested+=1
                whole+=int(code==5)
    assert tested>0 and whole>0
    # Run the actual opt-in producer/consumer, not only the algebra fixture.
    from test_crosscut2_preference_integration import producer
    from crosscut2.event_holds import select
    from crosscut2.played_pose_port_programme import PlayedPosePortPrior
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    runs,_,rows,score,receipt,_=select(*args,**kw,native_ports=native,music=[0,1,1,0],
        connection_prior=PlayedPosePortPrior(editorial_opportunities=True,scene_readout=True))
    assert receipt['scene_readout_consumed'] and not receipt['pre_DP_scalar_camera_argmin_used']
    assert receipt['boundary_port_DP']['label_limit_per_port_paid_bit']==8
    assert receipt['selected_hold_cost_decomposition'][-1]['cumulative_cost']==pytest.approx(score)
    assert len(runs)==len(rows) and all(row['scored_camera_values'].shape[1]==3 for row in rows)
    assert receipt['source_NN_and_analysis_decode_during_new_selection']==0
    json.dumps(receipt,allow_nan=False)  # actual programme persistence boundary
