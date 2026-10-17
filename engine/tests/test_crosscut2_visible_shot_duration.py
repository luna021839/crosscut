import hashlib
import math

import numpy as np
import pytest

from crosscut2.visible_shot_duration import Model, fit, tail_artifact,exposure_tail_artifact


def reference(identity, durations, split='train'):
    return dict(reference_id=identity, split=split, input_sha256=hashlib.sha256(identity.encode()).hexdigest(),
        cut_seconds=np.r_[1., 1.+np.cumsum(durations)].tolist())


def model():
    return Model(fit([reference('TRAIN_A', [1., 2., 3., 4.]),
        reference('TRAIN_B', [2., 3., 4., 5.])]))


def test_only_distinct_train_videos_fit_equal_mass_and_no_same_song_clock():
    a=reference('A',[1.,2.]);b=reference('B',[3.,4.,5.,6.])
    result=fit([a,b]);actual=Model(result)
    expected=np.mean([np.log([1.,2.]).mean(),np.log([3.,4.,5.,6.]).mean()])
    assert actual.mean==pytest.approx(expected)
    assert 'cut_seconds' not in str(result['references'])
    for rows in ([a,a],[a,reference('dev',[1.,2.],split='dev')],[]):
        with pytest.raises(ValueError):fit(rows)
    with pytest.raises(ValueError):fit([a,dict(a,reference_id='duplicate_content')])
    result['log_seconds_mean']+=1.
    with pytest.raises(ValueError):Model(result)


def test_short_highlights_remain_finite_without_requiring_more_cuts():
    value=model();peak=math.exp(value.mean)
    assert value.loss(peak)==pytest.approx(0.)
    assert 0.<value.loss(.4)<.4 and 0.<value.loss(20.)<20.
    assert value.loss(.4,censored=True)==0.
    assert value.loss(20.,censored=True)==value.loss(20.)
    assert value.loss(0.)==0.


@pytest.mark.parametrize('tails',[False,True])
def test_native_cut_prefix_table_matches_every_actual_visible_fragment(tails):
    value=model();nodes=np.array([0,7,12,35,80,120]);fps=30.
    if tails:value=Model(tail_artifact(value.artifact,fps=fps))
    events=[[],np.array([0,10,25,60,120]),np.array([7,12,35,80])]
    table=value.table(nodes,events,fps=fps)
    assert not table.flags.writeable and table.nbytes<2**20
    for s,xs in enumerate(events):
        for i,a in enumerate(nodes):
            for j,b in enumerate(nodes):
                if i>=j:
                    assert table[i,j,s]==0.
                    continue
                edges=[int(a),*(int(t) for t in xs if a<t<b),int(b)]
                direct=sum(float(value.loss((right-left)/fps,
                    censored=(left==0 or right==nodes[-1])))
                    for left,right in zip(edges[:-1],edges[1:]))
                assert table[i,j,s]==pytest.approx(direct,abs=1e-14)


def test_soft_prior_never_masks_a_strong_short_motion_match():
    from crosscut2.boundary_port_dp import solve
    value=model();nodes=np.array([0,15,30,45]);sources=[np.array([0,1])]*4
    table=value.table(nodes,[[],[]],fps=30.)
    def hold(i,j):
        # The long alternative visibly loses the action, while every short
        # alternative has actual finite cost. Pacing must not forbid it.
        base=0. if j-i==1 else 10.
        return base+np.diag(table[i,j]),np.eye(2,dtype=bool)
    runs,objective,_=solve(nodes,sources,sources,start_cost=np.zeros(2),hold_cost=hold,
        cut_cost=lambda n:(np.zeros((2,2)),~np.eye(2,dtype=bool)))
    assert len(runs)==3 and objective[1]==2


@pytest.mark.parametrize('recycle',[False,True])
def test_real_programme_scores_duration_before_both_camera_argmins_and_replays_it(recycle):
    from crosscut2.played_pose_port_programme import PlayedPosePortPrior,select
    from test_crosscut2_preference_integration import producer
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    nodes,banks,_,_=args
    for bank in banks:
        bank['programme_render_clock']=dict(fps=30.,count=int(nodes[-1]),source_times=nodes/30.)
    duration=model()
    prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,
        shot_duration_model=duration,recycle_static_capacity=recycle)
    result=select(*args,**kw,native_ports=native,music=[0,1,1,0],prior=prior)
    receipt=result[4];table=duration.table(nodes,kw['source_events'],fps=30.)
    for run in receipt['selected_hold_cost_decomposition']:
        i=int(np.flatnonzero(nodes==run['first_frame'])[0]);j=int(np.flatnonzero(nodes==run['final_frame'])[0])
        assert run['components']['visible_shot_duration_prior']==pytest.approx(table[i,j,run['source']])
    assert result[3]==pytest.approx(sum(sum(p['components'].values()) for p in receipt['selected_hold_cost_decomposition']))
    assert receipt['visible_shot_duration_prior']['includes_retained_native_cuts']
    assert receipt['boundary_port_domain']['baseline_paired_proposal_limit_per_node']==8
    assert receipt['boundary_port_domain']['maximum_entry_ports']<=4*len(banks)+8
    assert receipt['boundary_port_DP']['claim_camera_chosen_before_argmin']


def test_reference_quartile_tails_have_a_flat_middle_not_a_preferred_single_length():
    original=model();fitted=dict(original.artifact)
    value=Model(tail_artifact(fitted,fps=30.))
    middle=np.exp(original.mean+original.scale*np.array([-.5,0.,.5]))
    np.testing.assert_allclose(value.loss(middle),0.,atol=1e-14)
    short,long=np.exp(original.mean+original.scale*np.array([-2.,2.]))
    assert 0.<value.loss(short)<value.loss(short/2.)
    assert 0.<value.loss(long)<value.loss(long*2.)
    assert value.loss(short,censored=True)==0.
    assert value.loss(long,censored=True)==value.loss(long)
    assert value.loss(0.)==0.
    assert original.artifact==fitted and value.artifact['references']==fitted['references']
    assert value.artifact['fit_content_sha256']==fitted['content_sha256']
    assert not value.artifact['preferred_single_duration'] and not value.artifact['aesthetic_weight_fitted']
    with pytest.raises(ValueError,match='clock changed'):value.table([0,30,60],[[]],fps=25.)
    with pytest.raises(ValueError):tail_artifact(value.artifact,fps=30.)


def test_cli_requires_explicit_compatible_interval_path_before_reading_a_model(monkeypatch):
    from crosscut2.stage_mix import main
    args=['stage_mix','--source-plan','unused','--analysis','unused','--cache','unused',
        '--output','unused','--models','unused','--visible-shot-duration-model','unused']
    monkeypatch.setattr('sys.argv',args)
    with pytest.raises(ValueError,match='requires the actual E6'):main()
    args.extend(['--played-interval-utility'])
    with pytest.raises(ValueError,match='requires E4'):main()


def test_long_tail_exposure_matches_screen_time_integral_without_changing_short_or_middle_shots():
    from scipy.integrate import quad
    old=Model(tail_artifact(model().artifact,fps=30.));saved=dict(old.artifact)
    value=Model(exposure_tail_artifact(old.artifact,fps=30.));a=value.upper_quartile
    short=math.exp(value.mean-2*value.scale)
    assert value.loss(short)==old.loss(short) and value.loss(short,censored=True)==0.
    np.testing.assert_allclose(value.loss(np.exp(value.mean+value.scale*np.array([-.5,0.,.5]))),0.)
    assert value.loss(a)==pytest.approx(0.) and value.loss(0.)==0.
    for seconds in (a+1e-5,a+1.,a+8.,a+20.):
        expected=quad(lambda t:1.-.5*math.erfc((math.log(t)-value.mean)/value.scale/math.sqrt(2.))/.25,
            a,seconds,epsabs=1e-10)[0]
        assert value.loss(seconds)==pytest.approx(expected,abs=1e-10)
        assert value.loss(seconds,censored=True)==value.loss(seconds)
    assert value.loss(a+16.)>8. and old.artifact==saved
    assert value.artifact['references']==old.artifact['references']
    assert value.artifact['fit_content_sha256']==old.artifact['fit_content_sha256']
    assert value.artifact['hard_maximum_hold_seconds'] is None
    with pytest.raises(ValueError,match='clock changed'):exposure_tail_artifact(old.artifact,fps=25.)


def test_exposure_pacing_counts_real_native_fragments_not_an_uncut_source_hold():
    value=Model(exposure_tail_artifact(model().artifact,fps=30.))
    nodes=np.array([0,120,240,360,480]);events=[[],[120,240,360]]
    table=value.table(nodes,events,fps=30.)
    assert table[0,-1,0]>table[0,-1,1]
    assert table[0,-1,1]==pytest.approx(4.*value.loss(4.))
    for source,xs in enumerate(events):
        for i,a in enumerate(nodes):
            for j,b in enumerate(nodes):
                if i>=j:continue
                edges=[a,*(x for x in xs if a<x<b),b]
                expected=sum(value.loss((right-left)/30.,censored=(left==0 or right==nodes[-1]))
                    for left,right in zip(edges[:-1],edges[1:]))
                assert table[i,j,source]==pytest.approx(expected,abs=1e-12)


def test_actual_interval_pacing_includes_native_ports_missing_from_the_older_source_list(monkeypatch):
    from crosscut2.played_pose_port_programme import PlayedPosePortPrior,select
    from test_crosscut2_preference_integration import producer
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    nodes,banks,_,_=args
    for bank in banks:
        bank['programme_render_clock']=dict(fps=30.,count=int(nodes[-1]),source_times=nodes/30.)
    native[0]=dict(native[0],frames=np.array([20]),heads=np.zeros((1,2,8,4)))
    duration=Model(exposure_tail_artifact(model().artifact,fps=30.));observed=[];original=duration.table
    def table(nodes,events,**kwargs):observed.append([list(v) for v in events]);return original(nodes,events,**kwargs)
    monkeypatch.setattr(duration,'table',table)
    prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,shot_duration_model=duration)
    result=select(*args,**kw,native_ports=native,music=[0,1,1,0],prior=prior)
    assert observed and 20 in observed[0][0]
    assert result[4]['visible_shot_duration_prior']['physical_native_event_union_used']
