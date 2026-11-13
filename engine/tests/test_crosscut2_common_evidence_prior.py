import copy
import numpy as np
import pytest
from crosscut2.common_evidence_connection import scores
from crosscut2.common_evidence_prior import CommonEvidencePrior
from crosscut2.event_holds import select
from crosscut2.observed_motion_utility import integrals
from test_crosscut2_preference_features import setup
from test_crosscut2_preference_integration import producer


def aligned(p,a,b):
    p.head[b]=p.head[a];p.availability[:]=.8;p.crop[:]=0.


def test_unknown_only_pose_uses_symmetric_prior_and_cannot_itself_veto_a_cut():
    p,a,b=setup();aligned(p,a,b);p.body[:]=0.
    K,base,evidence=scores(p,1)
    assert K[a,b]==pytest.approx(base[a,b])
    assert not evidence['common_phase'][a,b] and not evidence['common_motion'][a,b]


def test_one_corroborated_opposed_arm_is_not_hidden_by_missing_rest_of_body():
    p,a,b=setup(opposed=True);aligned(p,a,b)
    p.body[:,:,:, [1,3,5,6,7],2]=0.
    p.phase_certainty[:]=1.;p.motion_certainty[:]=1.
    for d in p.data:d['pixel_part_scope']=np.ones(len(p.nodes),bool)
    K,base,evidence=scores(p,1)
    assert evidence['common_motion'][a,b] and K[a,b]<base[a,b]
    # Unknown MODEL opposition is not the same evidentiary statement.
    p.phase_certainty[:]=.5;p.motion_certainty[:]=.5
    _,_,evidence=scores(p,1)
    assert not evidence['common_motion'][a,b]


def test_real_producer_uses_unified_content_explicit_mask_and_exact_decomposition(tmp_path):
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    prior=CommonEvidencePrior()
    runs,_,_,value,receipt,_=select(*args,**kw,native_ports=native,music=[0,1,1,0],
        connection_prior=prior,preference_workspace=tmp_path)
    assert len(runs)>1
    assert receipt['objective_contract']=='common_evidence_admitted_unified_played_events_v1'
    assert value==pytest.approx(sum(sum(x.values()) for x in receipt['selected_event_preferences']))
    assert all(x['internal_event_preferences']==0 for x in receipt['selected_event_preferences'])
    assert not receipt['event_preference']['observed_own_motion']['unobserved_hold_motion_extrapolated']
    assert list(tmp_path.iterdir())==[]


def test_physical_movement_integral_deduplicates_windows_and_never_fills_a_long_unobserved_hold():
    p,_,_=setup()
    for d in p.data:d['physical_window_valid'][[0,-1]]=False
    first,_=integrals(p);long=copy.copy(p);long.nodes=p.nodes.copy();long.nodes[-1]=3000
    after,_=integrals(long)
    np.testing.assert_allclose(first.sum(1),after.sum(1),atol=1e-14)
    duplicated=copy.copy(p);idx=[0,1,1,2,3];duplicated.nodes=p.nodes[idx]
    for name in ('body','projected','head','camera','availability','velocity','phase_certainty','motion_certainty'):
        setattr(duplicated,name,getattr(p,name)[:,idx])
    duplicated.data=[dict(d,physical_window_valid=d['physical_window_valid'][idx],
        pixel_part_scope=d.get('pixel_part_scope',np.zeros(4,bool))[idx]) for d in p.data]
    extra,_=integrals(duplicated)
    np.testing.assert_allclose(first.sum(1),extra.sum(1),atol=1e-14)
    assert (first.sum(1)>0).any()


def test_fixed_pair_cost_not_discounted_by_an_unused_source_native_event():
    p,_,_=setup();native=[dict(frames=[])]*3
    a,b=CommonEvidencePrior(),CommonEvidencePrior()
    a.bind(p,native_ports=native,source_events=[[],[],[]])
    b.bind(p,native_ports=native,source_events=[[],[],[9,11]])
    ka,ma=a.costs_and_mask(p,1);kb,mb=b.costs_and_mask(p,1)
    np.testing.assert_array_equal(ka,kb);np.testing.assert_array_equal(ma,mb)
