import copy
import math
import numpy as np
import pytest
from crosscut2.event_states import displayed_focal_faces
from crosscut2.native_preference import _focal
from crosscut2.played_face_connection import extract
from crosscut2.played_face_prior import PlayedFacePrior
from crosscut2.connection_observations import extract as counterfactual
from crosscut2.event_holds import select
from test_crosscut2_preference_features import setup
from test_crosscut2_preference_integration import producer


def aligned():
    p,a,b=setup();p.camera[:]=[1.,.5,.5]
    for i,row in enumerate(p.data):
        row['displayed_focal_heads'][:]=[.5,.4,.2,.99]
        p.head[i]=row['displayed_focal_heads']
    return p,a,b


def prior(p):
    model=PlayedFacePrior();model.bind(p,native_ports=[dict(frames=[])]*2,source_events=[[],[]])
    return model


def test_played_endpoints_not_unplayed_same_time_pairs():
    p,a,b=aligned()
    # The virtual same-time pairs agree perfectly. The PLAYED pair does not.
    for i in (a,b):
        p.head[i,1,0,2]=.15;p.head[i,1,1,2]=.26
        p.data[i]['displayed_focal_heads'][1]=p.head[i,1]
    assert counterfactual(p,1)['scale'][a,b]==0.
    assert extract(p,1)['log_scale'][a,b]==pytest.approx(math.log(.26/.15))


def test_unplayed_flanks_tracking_flags_and_native_H_cannot_change_played_geometry():
    p,a,b=aligned();model=prior(p)
    before=extract(p,1);cost,mask=model.costs_and_mask(p,1)
    p.data[a]['displayed_focal_heads'][1,1]=[.8,.8,.8,.99]
    p.data[b]['displayed_focal_heads'][1,0]=[.2,.2,.01,.99]
    p.continuous[:,1]=False
    for row in p.data:row['known_center_native_cut'][1]=True
    p.head[:,1]=0.;p.availability[:,1]=0.
    after=extract(p,1);new_cost,new_mask=model.costs_and_mask(p,1)
    for key in ('measured','log_scale','face_height_displacement'):
        np.testing.assert_array_equal(before[key][a,b],after[key][a,b])
    assert new_mask[a,b]==mask[a,b] and new_cost[a,b]==cost[a,b]


def test_unknown_allowed_not_rewarded_and_large_visible_jump_not_bought_by_bad_H():
    p,a,b=aligned();model=prior(p)
    good,allowed=model.costs_and_mask(p,1);assert allowed[a,b] and good[a,b]==0.
    p.data[b]['displayed_focal_heads'][1,1,2]=.8
    cost,allowed=model.costs_and_mask(p,1)
    assert not allowed[a,b] and cost[a,b]>0.
    p.data[a]['known_center_native_cut'][1]=True;p.availability[a,1]=0.
    after,mask=model.costs_and_mask(p,1)
    assert not mask[a,b] and after[a,b]==cost[a,b]
    p.data[b]['displayed_focal_heads'][1,1]=0.
    unknown,mask=model.costs_and_mask(p,1)
    assert mask[a,b] and unknown[a,b]==pytest.approx(model.support.cut_weights(int(p.nodes[1]),model.sources)[a,b])


def test_same_physical_GROUP_pixels_are_invariant_to_selected_absent_actor_label():
    p,a,b=aligned();before=extract(p,1)
    p.head[b]=0.;p.member_hint[b]=-1;p.availability[b]=0.;p.dominant[b]=False
    after=extract(p,1)
    for key in before:np.testing.assert_array_equal(before[key],after[key])
    assert extract(p,1,intent='member')['measured'].sum()==0
    with pytest.raises(ValueError):extract(p,1,intent='member',target_measured=np.ones(3))


def test_vectorized_physical_focal_equals_independent_native_pixel_selector():
    rng=np.random.default_rng(174);heads=rng.uniform(.05,.95,(5,2,3,4));heads[...,2]*=.4
    heads[...,3]=.99;heads[1,0]=0.;cameras=np.tile([1.,.5,.5],(5,2,1));cameras[2:,0]=[1.6,.45,.55]
    actual,indices=displayed_focal_faces(heads,cameras)
    for n in range(5):
        for side in (0,1):
            focal=_focal(heads[n,side],cameras[n,side])
            if focal is None:assert indices[n,side]==-1 and not actual[n,side].any()
            else:
                assert indices[n,side]==focal[0]
                np.testing.assert_array_equal(actual[n,side],heads[n,side,focal[0]])


def test_actual_producer_selects_D_with_hold_integrity_and_exact_objective_decomposition(tmp_path):
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    model=PlayedFacePrior()
    runs,_,_,value,receipt,_=select(*args,**kw,native_ports=native,music=[0,1,1,0],
        connection_prior=model,preference_workspace=tmp_path)
    assert receipt['objective_contract']=='played_face_geometry_unified_native_integrity_v1'
    assert value==pytest.approx(sum(sum(x.values()) for x in receipt['selected_event_preferences']))
    assert all(model.hold_allowed(a,b)[s] for a,b,s in runs)
    assert all(x['entry_preference']>=0. for x in receipt['selected_event_preferences'])
    assert not model.artifact['measured_body_action_cost_used']
    assert list(tmp_path.iterdir())==[]
