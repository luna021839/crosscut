from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from crosscut2.boundary_port_dp import solve
from crosscut2.boundary_arc_costs import camera_samples_many
from crosscut2.matched_entry_delivery import MatchedEntryDelivery
from crosscut2.played_interval_utility import PlayedIntervalUtility
from test_crosscut2_played_interval_utility import samples


def fixture(*,clock_gap=False,missing=False,event_utility=False):
    nodes=np.array([0,30,60,120]);table=samples(120)
    support=table.support.copy()
    if missing:support[37:,2:6]=False
    window=SimpleNamespace(ordinals=table.native,heads=table.heads,body=table.body,support=support,
        associated=np.ones(120,bool),native_cuts=np.array([],int),competitors=table.competitors)
    entries=[];exits=[]
    for n,tick in enumerate(nodes):
        offsets=np.arange(-7,7);at=np.clip(tick+offsets,0,119)
        context=dict(window=window,at=at,clock=SimpleNamespace(output_frames=offsets))
        paired=SimpleNamespace(source=0,strip=np.tile([1.2,.5,.5],(7,1)),
            proposal_ids=(('paired',n,1,0,0,0,0),),observation=dict(dense_subject_context=context))
        static=SimpleNamespace(source=0,strip=np.tile([1.,.5,.5],(7,1)),
            proposal_ids=(('static',0),),observation=dict(dense_subject_context=context))
        entries.append([paired,static])
        exits.append([SimpleNamespace(strip=np.tile([z,.5,.5],(7,1))) for z in (1.,1.35)])
    field=PlayedIntervalUtility(nodes=nodes,delivery=[table],fps=30.,image_records=[[]],delivery_in_base=False,
        clocks=[dict(fps=30.,count=121,source_times=nodes/30.)])
    if clock_gap:
        actual=field.native
        def changed(source,first,final,frames):
            result=actual(source,first,final,frames);result[np.asarray(frames)%11==0]+=1
            return result
        field.native=changed
    arcs=SimpleNamespace(nodes=nodes,entries=entries,exits=exits,fps=30.,
        dense=[np.array([np.tile([z,.5,.5],(120,1)) for z in (1.,1.15,1.3,1.4)])],
        cats=[SimpleNamespace(bank=dict(zoom_cap=1.4))],played_interval_utility=field)
    events=SimpleNamespace(episodes=np.array([0,1,1,2]),boundaries=np.array([0,30,120]))
    if event_utility:
        from crosscut2.editorial_opportunities import Opportunities
        events=Opportunities(nodes,[0,1,1,0],fps=30.,families=['source'])
        events.utility[:]=1.
    return arcs,MatchedEntryDelivery(arcs,events,event_utility=event_utility)


@pytest.mark.parametrize('clock_gap',[False,True])
@pytest.mark.parametrize('event_utility',[False,True])
def test_compiled_claim_equals_raw_actual_camera_for_all_families_and_lengths(clock_gap,event_utility):
    arcs,claim=fixture(clock_gap=clock_gap,event_utility=event_utility);allowed=np.ones((2,2,6),bool)
    for i,j in ((1,2),(1,3),(2,3)):
        density=np.array([.8,.8])
        direct=claim.benefits(i,j,density,allowed,direct=True)
        compiled=claim.benefits(i,j,density,allowed)
        np.testing.assert_allclose(compiled,direct,rtol=1e-12,atol=1e-12)
        assert not compiled[1].any()  # a static port cannot borrow its paired neighbour's target
        first,final=map(int,arcs.nodes[[i,j]])
        for q in range(2):
            for code in range(6):
                curve=camera_samples_many(arcs.entries[i][0].strip[None],arcs.exits[j][q].strip[None],
                    arcs.dense[0][0 if code>=4 else code][first:final],np.arange(final-first),code=code)[0]
                assert compiled[0,q,code]==pytest.approx(claim.replay(i,j,0,curve,.8),abs=1e-12)


def test_claim_uses_actual_long_hold_action_not_only_seven_frame_entry_or_duration():
    arcs,claim=fixture();allowed=np.ones((2,2,6),bool)
    density=np.array([1.,0.]);long=claim.benefits(1,3,density,allowed)
    missing_arcs,missing=fixture(missing=True)
    short_evidence=missing.benefits(1,3,density,allowed)
    assert long.max()>short_evidence.max()
    assert short_evidence.max()<=7/30.+1e-12
    assert not claim.benefits(0,3,density,allowed).any()  # source continuation/intro is not an edit
    assert long.max()<=3.  # at most the actual90frames, no event-utility inflation


def test_current_episode_clips_benefit_even_if_entry_track_continues():
    arcs,claim=fixture();claim.events.boundaries=np.array([0,30,45,120])
    claim.events.episodes=np.array([0,1,2,3])
    benefit=claim.benefits(1,3,np.array([1.,0.]),np.ones((2,2,6),bool))
    assert benefit.max()<=15/30.+1e-12


def test_event_value_uses_actual_coverage_but_never_creates_unobserved_body_persistence():
    arcs,claim=fixture(missing=True,event_utility=True)
    allowed=np.ones((2,2,6),bool);density=np.array([.8,0.])
    benefits=claim.benefits(1,3,density,allowed)
    _,old=fixture(missing=True)
    seconds_reward=old.benefits(1,3,density,allowed)
    # Same actual seven-frame strip; only the fixed musical EVENT unit
    # differs. A longer observation window is not required to value a cut.
    np.testing.assert_allclose(benefits,seconds_reward*(3./(7./30.)))
    assert benefits.max()<=3.*.8 and not benefits[1].any()
    assert not claim.benefits(0,3,density,allowed).any()
    # Neither long holds nor extra camera variants can exceed this one
    # episode's fixed raw utility. Entirely missing incoming arms earn0.
    _,complete=fixture(event_utility=True)
    assert complete.benefits(1,3,density,allowed).max()<=3.*.8
    arcs,missing=fixture(missing=True,event_utility=True)
    arcs.entries[1][0].observation['dense_subject_context']['window'].support[:]=False
    assert not missing.benefits(1,3,density,allowed).any()


def test_event_normalization_is_explicit_and_requires_actual_interval_consumer():
    from crosscut2.played_pose_port_programme import PlayedPosePortPrior
    with pytest.raises(ValueError,match='actual pair-bound interval'):PlayedPosePortPrior(matched_event_utility=True)
    prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,matched_event_utility=True)
    assert prior.artifact['matched_musical_event_utility']
    assert not PlayedPosePortPrior().artifact['matched_musical_event_utility']


def test_claim_camera_argmin_can_differ_from_base_camera_and_replay_is_frozen():
    nodes=np.array([0,12,24]);domain=[np.arange(2) for _ in nodes];same=np.eye(2,dtype=bool)
    def hold(i,j):
        return np.full((2,2),0. if i==0 else .1),same,np.zeros((2,2),int)
    def claim(i,j):
        benefit=np.full((2,2),0. if i==0 else .8)
        # Camera1 costs.3 before its real.8 benefit. Reusing the cheaper
        # unclaimed camera0 would lose this genuinely delivered action.
        return np.full((2,2),.3)-benefit,same,np.ones((2,2),int),benefit
    runs,score,receipt=solve(nodes,domain,domain,start_cost=np.zeros(2),hold_cost=hold,
        cut_cost=lambda n:(np.full((2,2),.1),~same),
        opportunity_episodes=[0,0,1],claim_hold_cost=claim)
    assert score==pytest.approx((-.4,1))
    assert receipt['selected_arc_codes']==[0,1]
    assert [v['credit'] for v in receipt['opportunity_claims']]==[0.,.8]


def test_camera_dependent_episode_claim_DP_matches_exhaustive_oracle_and_no_ABA_farming():
    for seed in range(40):
        rng=np.random.default_rng(seed);N=5;nodes=np.arange(N)*12
        ids=[np.arange(2) for _ in nodes];episodes=np.array([0,0,0,1,2]);same=np.eye(2,dtype=bool)
        H={(i,j):rng.uniform(0,2,(2,2,3)) for i in range(N) for j in range(i+1,N)}
        C={key:rng.uniform(0,2,(2,2,3))*(key[0]>0) for key in H}
        K={n:rng.uniform(0,1,(2,2)) for n in range(1,N-1)}
        best=[(float('inf'),0)]
        def visit(i,p,total,cuts,paid):
            for j in range(i+1,N):
                for code in range(3):
                    base=total+H[i,j][p,p,code]
                    choices=[(base,dict(paid))]
                    e=int(episodes[i]);benefit=C[i,j][p,p,code]
                    if e not in paid and benefit>0:
                        claimed=dict(paid);claimed[e]=True;choices.append((base-benefit,claimed))
                    for value,claims in choices:
                        if j==N-1:best[0]=min(best[0],(value,cuts))
                        else:visit(j,1-p,value+K[j][p,1-p],cuts+1,claims)
        for source in range(2):visit(0,source,0.,0,{})
        def hold(i,j):
            code=np.argmin(H[i,j],axis=-1)
            return np.take_along_axis(H[i,j],code[...,None],axis=-1).squeeze(-1),same,code
        def claim(i,j):
            value=H[i,j]-C[i,j];code=np.argmin(value,axis=-1)
            return (np.take_along_axis(value,code[...,None],axis=-1).squeeze(-1),same,code,
                np.take_along_axis(C[i,j],code[...,None],axis=-1).squeeze(-1))
        runs,score,receipt=solve(nodes,ids,ids,start_cost=np.zeros(2),hold_cost=hold,
            cut_cost=lambda n:(K[n],~same),opportunity_episodes=episodes,claim_hold_cost=claim)
        assert score==pytest.approx(best[0])
        claimed=[v['episode'] for v in receipt['opportunity_claims'] if v['claimed']]
        assert len(claimed)==len(set(claimed))
        actual=sum(H[i,j][p,q,code] for (i,j,p,q),code in zip(runs,receipt['selected_arc_codes']))
        actual+=sum(K[a[1]][a[3],b[2]] for a,b in zip(runs,runs[1:]))
        actual-=sum(v['credit'] for v in receipt['opportunity_claims'])
        assert score[0]==pytest.approx(actual)
