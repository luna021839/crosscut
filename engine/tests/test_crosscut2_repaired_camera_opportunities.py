"""A failed original view cannot erase a real proposed crop's action match."""
from types import SimpleNamespace
import numpy as np
import pytest
from crosscut2.editorial_opportunities import Opportunities
from crosscut2.played_partial_pose import features,matrix
from test_crosscut2_dense_subject_catalogue import prepared
from test_crosscut2_source_clock_action import obs


def pair():
    a,_,_=prepared(0,bump=.08);b,_,_=prepared(1,bump=.08)
    key=('paired',1,0,1,0,0,0)
    return tuple(SimpleNamespace(source=s,observation=obs(cat,side),proposal_ids=(key,))
        for s,cat,side in ((0,a,0),(1,b,1)))


def test_offered_crop_can_recover_credit_but_bad_actual_playback_still_earns_zero():
    left,right=pair();event=Opportunities([0,100,200],[0,1,0],fps=30.,families=['a','b'])
    original=matrix(features([left.observation],fps=30.),features([right.observation],fps=30.),details=True)
    original['matched_action_change_density'][:]=0.  # original-view placement failed
    event.observe(1,[left.observation],[right.observation],original,[0],[1])
    assert not event.bindings
    actual=event.observe_played_pair(1,left,right)
    density=float(actual['matched_action_change_density'][0,0])
    assert density>0 and event.density(1,left.observation,right.observation,0,1,density)==density
    assert event.density(1,left.observation,right.observation,0,1,0.)==0.
    assert event.credit(1,density,0.)==0.
    assert event.receipt()['source_actor_bindings_improved_by_actual_fitted_views']==1
    before=event.utility.copy();event.observe_played_pair(1,left,right)
    np.testing.assert_array_equal(event.utility,before)
    assert len(event.bindings)==1


def test_unpaired_continuation_and_duplicate_families_cannot_manufacture_credit():
    left,right=pair();event=Opportunities([0,100,200],[0,1,0],fps=30.,families=['same','same'])
    event.observe_played_pair(1,left,right)
    assert not event.bindings and not event.utility.any()
    right.proposal_ids=(('static',1),)
    with pytest.raises(ValueError,match='physical camera pair'):event.observe_played_pair(1,left,right)


def test_aligned_faces_without_action_do_not_gain_an_observed_match():
    from test_crosscut2_focal_appearance_risk import observation,vector
    key=('paired',1,0,1,0,0,0);ports=[]
    for source,side in ((0,0),(1,1)):
        value=observation(side,vector());value.update(head_index=0,displayed_native_quality_proxy=.9)
        ports.append(SimpleNamespace(source=source,observation=value,proposal_ids=(key,)))
    event=Opportunities([0,100,200],[0,1,0],fps=30.,families=['a','b'])
    actual=event.observe_played_pair(1,*ports)
    assert actual['matched_action_change_density'][0,0]==0.
    assert not event.bindings and not event.utility.any()


def test_actual_domain_registers_fitted_views_before_interval_selection(monkeypatch):
    from crosscut2 import played_pose_port_programme as programme
    from test_crosscut2_preference_integration import producer
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    nodes,banks,_,_=args
    for bank in banks:
        bank['programme_render_clock']=dict(fps=30.,count=int(nodes[-1]),source_times=nodes/30.)
    original=Opportunities.observe_played_pair;calls=[]
    def observed(self,n,left,right,**kw):
        calls.append((n,left.source,right.source));return original(self,n,left,right,**kw)
    monkeypatch.setattr(Opportunities,'observe_played_pair',observed)
    result=programme.select(*args,**kw,native_ports=native,music=[0,1,1,0],
        prior=programme.PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True))
    assert calls and all(a!=b for _,a,b in calls)
    assert result[4]['editorial_opportunities']['played_camera_options_observed_before_selection']==len(calls)
