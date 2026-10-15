import numpy as np
import pytest
from crosscut2.played_pose_port_programme import PlayedPosePortPrior
from crosscut2.event_holds import select
from test_crosscut2_preference_integration import producer


def test_actual_producer_uses_physical_pose_and_saved_two_ended_camera_codes(tmp_path):
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    runs,states,data,score,receipt,ports=select(*args,**kw,native_ports=native,music=[0,1,1,0],
        connection_prior=PlayedPosePortPrior(),preference_workspace=tmp_path)
    assert receipt['played_pose_cost_used'] and receipt['physical_group_focal_bound']
    assert not receipt['actor_alias_changes_cost'] and not receipt['ownership_certified']
    assert all(s.actor is None for s in states)
    assert receipt['authored_cuts']==len(runs)-1
    assert len(receipt['boundary_port_DP']['selected_arc_codes'])==len(runs)
    assert score==pytest.approx(sum(sum(x['components'].values()) for x in receipt['selected_hold_cost_decomposition']))
    for k,(i,j,index) in enumerate(runs):
        assert len(data[index]['scored_camera_values'])==args[0][j]-args[0][i]
        if k:assert ports.selected(i,k-1,k)['partial_pose_cost_used']
    assert not list(tmp_path.iterdir())


def test_additive_observer_runs_once_before_costs_without_changing_E2(tmp_path,monkeypatch):
    from crosscut2 import played_pose_port_programme as module
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    baseline=select(*args,**kw,native_ports=native,music=[0,1,1,0],connection_prior=PlayedPosePortPrior())
    called=[];original=module.ArcCosts
    def observer(**domain):
        called.append(domain)
        assert domain['entries'] and domain['exits'] and domain['catalogues']
        return dict(phase='preparation_only',new_person_forwards=0)
    def costs(*a,**kw):
        assert len(called)==1
        return original(*a,**kw)
    monkeypatch.setattr(module,'ArcCosts',costs)
    actual=select(*args,**kw,native_ports=native,music=[0,1,1,0],connection_prior=PlayedPosePortPrior(),
        person_observer=observer)
    assert baseline[0]==actual[0] and baseline[3]==actual[3]
    assert actual[4]['person_rectangle_observations']['new_person_forwards']==0
    for a,b in zip(baseline[2],actual[2]):assert np.array_equal(a['scored_camera_values'],b['scored_camera_values'])


def test_observer_cannot_silently_enter_an_unrelated_legacy_selector():
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    with pytest.raises(ValueError,match='actual played port-camera'):
        select(*args,**kw,native_ports=native,music=[0,1,1,0],person_observer=lambda **kw:None)
