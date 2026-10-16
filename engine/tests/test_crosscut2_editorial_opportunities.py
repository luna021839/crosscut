from dataclasses import replace
import numpy as np
import pytest
from crosscut2.boundary_port_dp import solve
from crosscut2.editorial_opportunities import Opportunities
from test_crosscut2_dense_subject_catalogue import prepared
from test_crosscut2_source_clock_action import obs
from crosscut2.played_partial_pose import features,matrix


def graph(credits,episodes):
    n=len(episodes);sources=[np.arange(2) for _ in range(n)];nodes=np.arange(n)*12
    def hold(i,j):return np.zeros((2,2)),np.eye(2,dtype=bool),np.full((2,2),3,int)
    def cut(j):return np.full((2,2),.1),np.ones((2,2),bool)
    return solve(nodes,sources,sources,start_cost=np.zeros(2),hold_cost=hold,cut_cost=cut,
        opportunity_episodes=episodes,hold_credit=lambda i,j:np.full(2,credits[i]))


def test_no_measured_opportunity_does_not_force_any_cut():
    runs,score,receipt=graph([0,0,0,0],[0,0,0,1])
    assert len(runs)==1 and score==(0.,0)
    assert not any(p['claimed'] for p in receipt['opportunity_claims'])


def test_A_B_A_cannot_farm_a_fixed_episode_and_can_defer_weak_early_claim():
    runs,score,receipt=graph([0,.2,1.,.5,0],[0,0,0,0,1])
    assert score==pytest.approx((-.9,1))
    assert runs[-1][0]==2  # choose the actually better opportunity, not the first cut
    claims=[p for p in receipt['opportunity_claims'] if p['claimed']]
    assert len(claims)==1 and claims[0]['credit']==1.
    assert receipt['current_event_paid_bit'] and not receipt['all_history_subsets_stored']


def test_new_episode_resets_bit_without_a_global_cut_limit():
    runs,score,receipt=graph([0,1,0,1,0],[0,0,1,1,2])
    assert score==pytest.approx((-1.8,2))
    assert [p['episode'] for p in receipt['opportunity_claims'] if p['claimed']]==[0,1]
    assert all(code==3 for code in receipt['selected_arc_codes'])


def test_small_random_episode_graphs_match_exhaustive_whole_programme_oracle():
    for seed in range(40):
        rng=np.random.default_rng(seed);N=5;nodes=np.arange(N)*12
        ids=[np.arange(2) for _ in nodes];episodes=np.array([0,0,0,1,2])
        H={(i,j):rng.uniform(0,2,(2,2)) for i in range(N) for j in range(i+1,N)}
        K={i:rng.uniform(0,1,(2,2)) for i in range(1,N-1)}
        C={(i,j):rng.uniform(0,1,(2,2))*(i>0) for i in range(N) for j in range(i+1,N)}
        best=[(float('inf'),0)]
        def visit(i,p,total,cuts,benefits):
            for j in range(i+1,N):
                q=p;paid=dict(benefits);e=int(episodes[i]);paid[e]=max(paid.get(e,0.),C[i,j][p,q])
                value=total+H[i,j][p,q]
                if j==N-1:
                    best[0]=min(best[0],(value-sum(paid.values()),cuts));continue
                visit(j,1-q,value+K[j][q,1-q],cuts+1,paid)
        for p in range(2):visit(0,p,0.,0,{})
        runs,score,receipt=solve(nodes,ids,ids,start_cost=np.zeros(2),
            hold_cost=lambda i,j:(H[i,j],np.eye(2,dtype=bool)),
            cut_cost=lambda j:(K[j],~np.eye(2,dtype=bool)),
            opportunity_episodes=episodes,hold_credit=lambda i,j:C[i,j])
        assert score==pytest.approx(best[0])
        actual=sum(H[i,j][p,q] for i,j,p,q in runs)
        actual+=sum(K[a[1]][a[3],b[2]] for a,b in zip(runs,runs[1:]))
        actual-=sum(p['credit'] for p in receipt['opportunity_claims'])
        assert actual==pytest.approx(score[0])


def test_duplicate_source_family_cannot_create_extra_raw_utility():
    a,_,_=prepared(0,bump=.08);b,_,_=prepared(1,bump=.08)
    left,right=obs(a,0),obs(b,1)
    detail=matrix(features([left],fps=30.),features([right],fps=30.),details=True)
    same=Opportunities([0,100,200],[0,1,0],fps=30.,families=['show','show'])
    same.observe(1,[left],[right],detail,[0],[1])
    assert same.utility.sum()==0.
    unique=Opportunities([0,100,200],[0,1,0],fps=30.,families=['showA','showB'])
    unique.observe(1,[left],[right],detail,[0],[1]);before=unique.utility.copy()
    unique.observe(1,[left],[right],detail,[0],[1])
    np.testing.assert_array_equal(unique.utility,before)
    assert before.sum()>0 and unique.receipt()['raw_source_actor_bindings']==1


def test_exposure_never_continues_after_target_gap_native_cut_or_missing_arm():
    cat,w,indices=prepared();original=obs(cat,1)
    opportunity=Opportunities([0,100,200],[0,1,0],fps=30.,families=['a'])
    assert opportunity.exposure(1,original)==pytest.approx(7/30.)
    for field in ('gap','arm','cut'):
        context=dict(original['dense_subject_context'])
        if field=='gap':
            associated=w.associated.copy();associated[w.ordinals>=103]=False
            support=w.support.copy();support[~associated]=False
            context['window']=replace(w,associated=associated,support=support)
        elif field=='arm':
            support=w.support.copy();support[w.ordinals>=103,2:]=False
            context['window']=replace(w,support=support)
        else:
            same=context['same_shot'].copy();same[context['clock'].output_frames>=3]=False
            context['same_shot']=same
        actual=dict(original,dense_subject_context=context)
        assert opportunity.exposure(1,actual)==pytest.approx(3/30.)
    sparse=dict(original);sparse.pop('dense_subject_context')
    assert opportunity.exposure(1,sparse)==0.
    # The incoming source's cut BEFORE the played strip is not target loss
    # INSIDE that strip. A retained post-cut source remains a legal candidate.
    post,_,_=prepared(cut=(100,))
    assert opportunity.exposure(1,obs(post,1))==pytest.approx(7/30.)


def test_fixed_utility_uses_actual_strip_coverage_not_fictitious_whole_hold_exposure():
    opportunity=Opportunities([0,30,60,90],[0,1,1,0],fps=30.,families=['a'])
    opportunity.utility[opportunity.episodes[1]]=.4
    assert opportunity.credit(1,.4,7/30.)==pytest.approx(.8)
    assert opportunity.credit(1,.4,3/30.)==pytest.approx(.8*3/7)
    assert opportunity.credit(1,.4,300.)==pytest.approx(.8)
    assert opportunity.credit(1,.4,0.)==0.


def test_readability_plateau_removes_continuous_activity_and_extra_zoom_reward():
    from test_crosscut2_focal_activity_binding import two_faces
    still=two_faces(first_moves=False);moving=two_faces(first_moves=True)
    frames=np.array([18,19]);native=np.tile([1.,.5,.5],(2,1))
    for cat in (still,moving):cat.editorial_content=True
    np.testing.assert_allclose(still.rates(frames,native),moving.rates(frames,native))
    still.quality[:]=.36;one=still.rates(frames,native)
    still.quality[:]=.81;two=still.rates(frames,native)
    np.testing.assert_allclose(one,two)
    zoomed=np.tile([1.35,.5,.5],(2,1))
    assert (still.rates(frames,zoomed)>two).all()


def test_new_consumer_is_explicit_and_keeps_existing_main_path_unchanged():
    from crosscut2.played_pose_port_programme import PlayedPosePortPrior
    from crosscut2.event_holds import select
    from test_crosscut2_preference_integration import producer
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    baseline=PlayedPosePortPrior();trial=PlayedPosePortPrior(editorial_opportunities=True)
    assert not baseline.artifact['fixed_editorial_opportunities']
    runs,states,data,score,receipt,_=select(*args,**kw,native_ports=native,music=[0,1,1,0],connection_prior=trial)
    assert receipt['boundary_port_DP']['opportunity_credit_once_only']
    assert receipt['boundary_port_domain']['raw_fixed_opportunity_used_before_global_pruning']
    assert receipt['editorial_opportunities']['role_handoff_objective_implemented'] is False
    assert score==pytest.approx(sum(sum(x['components'].values()) for x in receipt['selected_hold_cost_decomposition']))


def test_continuous_strength_does_not_turn_fine_moments_into_extra_beat_episodes():
    from crosscut2.opportunity import continuous_music_context
    nodes=np.array([0,10,15,20,25,30,40])
    rhythm=dict(beat_grid=[dict(time_seconds=t/10.,strength=s)
        for t,s in ((10,.8),(20,.4),(30,.9))])
    strength,events=continuous_music_context(nodes,fps=10.,master_start=0.,rhythm=rhythm)
    np.testing.assert_allclose(strength,[0,.8,.6,.4,.65,.9,0])
    event=Opportunities(nodes,strength,fps=10.,families=['a','b'],beat_frames=events)
    np.testing.assert_array_equal(event.boundaries,[0,10,30,40])
    assert event.episodes[2]==event.episodes[3]==event.episodes[4]
    a,_,_=prepared(0,bump=.08);b,_,_=prepared(1,bump=.08)
    left,right=obs(a,0),obs(b,1)
    detail=matrix(features([left],fps=30.),features([right],fps=30.),details=True)
    event.observe(2,[left],[right],detail,[0],[1])
    assert event.bindings[(2,0,1,0,0)]>0  # real pose opportunity need not be an exact beat
    assert event.receipt()['music_strength_continuous_not_extra_beat_events']
    for bad in ([10,10],[-1,10],[40],np.array([10.])):
        with pytest.raises(ValueError):Opportunities(nodes,strength,fps=10.,families=['a'],beat_frames=bad)


def test_empty_or_single_beat_does_not_invent_a_continuous_music_region():
    from crosscut2.opportunity import continuous_music_context
    for rhythm in ({},dict(beat_grid=[dict(time_seconds=1.,strength=.7)])):
        strength,events=continuous_music_context([0,5,10,15,20],fps=10.,master_start=0.,rhythm=rhythm)
        assert strength[1]==strength[3]==0.
        assert len(events)==len(rhythm.get('beat_grid',()))
    legacy=Opportunities([0,100,200],[1,1,1],fps=30.,families=['a'])
    np.testing.assert_array_equal(legacy.boundaries,[0,200])
    explicit_empty=Opportunities([0,100,200],[0,0,0],fps=30.,families=['a'],beat_frames=[])
    np.testing.assert_array_equal(explicit_empty.boundaries,[0,200])
