from types import SimpleNamespace
import numpy as np
from crosscut2.subject_opportunity_probe import candidates


def test_preselection_probe_ranks_fixed_raw_opportunities_not_selected_movie_cuts():
    cats=[SimpleNamespace(subjects=SimpleNamespace(sample=lambda *a:None)) for _ in range(3)]
    bindings={(1,0,1,0,0):.8,(2,0,2,0,0):.9,(3,1,2,0,0):.7}
    events=SimpleNamespace(bindings=bindings,episodes=np.array([0,1,1,2]))
    assert candidates(cats,events,maximum=2)==[(2,0,2,0,0),(3,1,2,0,0)]
    # Repeated camera variants / pairs in one music episode do not consume
    # several observation centres. No final programme even exists here.
    events.bindings[(2,2,0,0,0)]=.89
    assert candidates(cats,events,maximum=2)==[(2,0,2,0,0),(3,1,2,0,0)]


def test_already_observed_pair_does_not_trigger_new_source_inference():
    cats=[SimpleNamespace(subjects=SimpleNamespace(sample=lambda *a:object())) for _ in range(2)]
    events=SimpleNamespace(bindings={(1,0,1,0,0):.8},episodes=np.array([0,1,2]))
    assert candidates(cats,events)==[]
    assert candidates(cats,events,include_observed=True)==[(1,0,1,0,0)]


def test_opportunity_observer_finishes_before_final_domain_and_actual_costs(monkeypatch):
    from crosscut2 import played_pose_port_programme as module
    from test_crosscut2_preference_integration import producer
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    stages=[];old_make=module.make;old_arcs=module.ArcCosts
    def make(*args,**kw):
        stages.append('domain');return old_make(*args,**kw)
    def observed(**kw):
        assert kw['opportunities'].bindings is not None and kw['catalogues']
        stages.append('raw_observations');return dict(source_NN_calls=0,fixture_only=True)
    def arcs(*args,**kw):
        assert stages==['domain','raw_observations','domain']
        return old_arcs(*args,**kw)
    monkeypatch.setattr(module,'make',make);monkeypatch.setattr(module,'ArcCosts',arcs)
    result=module.select(*args,**kw,native_ports=native,music=[0,1,1,0],
        prior=module.PlayedPosePortPrior(editorial_opportunities=True),opportunity_observer=observed)
    assert result[4]['opportunity_subject_observations']['fixture_only']
