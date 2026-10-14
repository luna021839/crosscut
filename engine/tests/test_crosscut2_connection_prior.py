"""Mechanical anchors, not fabricated human AV labels or video approval."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
from crosscut2.connection_prior import ConnectionPrior,quality,physical_weights,activity
from crosscut2.event_holds import select
from crosscut2.preference_model import partial_choice,fit,Model
from test_crosscut2_preference_features import setup
from test_crosscut2_preference_integration import producer
from test_crosscut2_native_preference import native_bank
from test_crosscut2_preference_model import weak


def test_weak_source_choice_does_not_identify_cut_vs_hold_and_is_rejected_as_primary(tmp_path):
    assert partial_choice([0,5,-5],[1,2])[0]==pytest.approx(partial_choice([0,-5,5],[1,2])[0])
    args,kw,native,_,_,_=producer(opposed=False)
    with pytest.raises(ValueError,match='diagnostic'):
        select(*args,**kw,native_ports=native,music=[0,1,1,0],
            preference_model=Model(fit([weak()])),preference_workspace=tmp_path)


def test_fixed_quality_is_monotone_and_cannot_reward_faceless_or_opposed_action():
    args=dict(position=.1,scale=.1,phase=.1,motion=.1,crop=.1,visibility=.8,own=.5,connection=.5,music=1.,style=.5)
    before=quality(**args)
    for k in ('position','scale','phase','motion','crop'):
        assert quality(**dict(args,**{k:.2}))<before
    assert quality(**dict(args,visibility=.9))>before
    assert quality(**dict(args,visibility=0.,own=1.,connection=1.,style=1.))==0.
    assert quality(**dict(args,motion=1.,own=1.,connection=1.,style=1.))==0.
    assert quality(**dict(args,identity_conflict=True))<before


def test_real_producer_face_position_increase_cannot_earn_the_old_positive_reward():
    p,a,b=setup();prior=ConnectionPrior()
    before=prior.scores_at(p,1,p.preference_features(1))[a,b]
    p.head[b,1,:,0]+=.2
    after=prior.scores_at(p,1,p.preference_features(1))[a,b]
    assert after<before


def test_actual_own_motion_survives_style_zero_and_unknown_part_caps():
    p,a,b=setup();p.colour[:,1,1]=p.colour[0,1,0];prior=ConnectionPrior()
    moving=prior.scores_at(p,1,p.preference_features(1))[a,b]
    own,_=activity(p,1);assert own[a,b]>0
    p.velocity[:]=0.
    p.projected[:,1,0,:,:2]=p.projected[:,1,1,:,:2]
    p.projected[:,1,3,:,:2]=p.projected[:,1,2,:,:2]
    stationary=prior.scores_at(p,1,p.preference_features(1))[a,b]
    assert moving>stationary


def test_real_one_arm_producer_can_select_a_connection_with_exact_primary_fragment_decomposition(tmp_path):
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    runs,_,_,score,receipt,_=select(*args,**kw,native_ports=native,music=[0,1,1,0],
        connection_prior=ConnectionPrior(),preference_workspace=tmp_path)
    assert len(runs)>1
    assert score==pytest.approx(sum(sum(r.values()) for r in receipt['selected_event_preferences']))
    assert all(r['components']['native_fragment_support']==0 for r in receipt['selected_hold_cost_decomposition'])
    assert receipt['objective_contract']=='declared_monotone_connection_primary_fragment_v1'
    assert list(tmp_path.iterdir())==[]


def test_long_tail_visibility_cannot_buy_an_observed_inferior_connection(tmp_path):
    for tail in (180,1800):
        args,kw,native,_,_,_=producer(opposed=True,tail=tail)
        # Both sources have real readable flanks at the compared event; only
        # the subsequent interval visibility changes. No votes/labels fitted.
        for bank in args[1]:bank['native_readability'][1]=[.1,.8,0.,200.]
        runs,_,_,_,_,_=select(*args,**kw,native_ports=native,music=[0,1,1,0],
            connection_prior=ConnectionPrior(),preference_workspace=tmp_path)
        assert all(args[0][first]!=60 for first,_,_ in runs)


def test_local_native_cost_cannot_punish_a_continuous_source_for_another_sources_event():
    _,_,native,p,_,_=producer(opposed=False);native=list(native);native[1]=native_bank(90)
    events=[np.array([],int),np.array([90])];prior=ConnectionPrior();prior.bind(p,native_ports=native,source_events=events)
    costs,receipt=prior.native_intervals(p,native_ports=native,source_events=events,check=lambda:None)
    mask=np.array([s.source==0 for s in p.states]);assert not costs[mask].any()
    assert (costs[~mask,1]>0).all() and not costs[:,[0,2]].any()
    assert not receipt['weak_linear_worst_regret_used']
    native[1]=native_bank(60);events[1]=np.array([60]);prior.bind(p,native_ports=native,source_events=events)
    costs,_=prior.native_intervals(p,native_ports=native,source_events=events,check=lambda:None)
    assert not costs.any()  # exact authored/native overlap handled only by H/K


def test_physical_support_deduplicates_events_and_neutral_observation_partition_does_not_add_weight():
    assert physical_weights([59,60,60,120],final=180,fps=30)==physical_weights([59,60,120],final=180,fps=30)
    base=SimpleNamespace(nodes=np.array([0,60,120,180]),strength=np.array([0,1,1,0]),fps=30.)
    refined=SimpleNamespace(nodes=np.array([0,58,60,120,180]),strength=np.array([0,0,1,1,0]),fps=30.)
    a,b=ConnectionPrior(),ConnectionPrior();native=[dict(frames=[])];events=[[]]
    a.bind(base,native_ports=native,source_events=events);b.bind(refined,native_ports=native,source_events=events)
    assert a.weights==b.weights and b.hold_weight[1]==0.


def test_real_identical_state_copy_cannot_create_preference_or_style_reward():
    p,_,_=setup();a=ConnectionPrior();native=[dict(frames=[])]*2;events=[[],[]]
    a.bind(p,native_ports=native,source_events=events);h,k=a.losses_at(p,1,p.preference_features(1))
    q=len(p.states);idx=np.r_[np.arange(q),q-1];cloned=copy.copy(p)
    cloned.states=(*p.states,p.states[-1]);cloned.data=(*p.data,p.data[-1])
    for name in ('head','camera','body','projected','vectors','continuous','formation','availability','colour',
        'colour_valid','crop','dominant','member_hint','phase_certainty','motion_certainty','bones','velocity'):
        setattr(cloned,name,getattr(p,name)[idx])
    b=ConnectionPrior();b.bind(cloned,native_ports=native,source_events=events)
    hh,kk=b.losses_at(cloned,1,cloned.preference_features(1))
    np.testing.assert_array_equal(hh,h[idx]);np.testing.assert_array_equal(kk,k[idx][:,idx])


@pytest.mark.parametrize('condition',['ordinary','opposed','missing','uncertain_path','central_cut','contradicted'])
def test_fast_projection_is_bit_exact_to_all59channel_prior_on_real_producer(condition):
    p,a,b=setup(opposed=condition=='opposed');prior=ConnectionPrior()
    if condition=='missing':p.body[:,1,:,[1,3,5,6,7],2]=0.
    if condition in ('uncertain_path','central_cut'):p.continuous[[a,b],1]=False
    if condition=='central_cut':
        p.data[a]['known_center_native_cut'][1]=True;p.head[a,1,1,0]+=.2
    if condition=='contradicted':
        p.phase_certainty[:]=.5;p.motion_certainty[:]=.5
        p.phase_certainty[a,1,1,0]=0.;p.motion_certainty[b,1,0,1]=0.
        p.member_hint[b,1]=4;p.member_hint[a,1]=2
    for n in (1,2):
        reference=prior.scores_at(p,n,p.preference_features(n))
        np.testing.assert_array_equal(prior.scores_from_ports(p,n),reference)
