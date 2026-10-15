import numpy as np
import pytest

from crosscut2.focal_appearance import discrepancy
from crosscut2.played_partial_pose import features,matrix
from crosscut2.boundary_arc_costs import ArcCosts


def vector(value=1.):
    result=np.zeros(128);result[0]=value
    return result


def observation(side,value):
    return dict(head=np.array([.5,.3,.2,.99]),body=np.zeros((2,8,3)),
        evidence=np.zeros((2,8),np.uint8),ownership_ambiguity=np.zeros((2,8),bool),
        camera_strip=np.tile([1.,.5,.5],(7,1)),side=side,physical_window_valid=True,
        native_indices=np.array([10,16]) if side==0 else np.array([17,23]),appearance_vector=value)


def test_matching_and_different_appearance_affect_actual_played_cost_not_only_reports():
    a=features([observation(0,vector())],fps=30.)
    same=matrix(a,features([observation(1,vector())],fps=30.))[0]
    other=matrix(a,features([observation(1,vector(-1.))],fps=30.))[0]
    assert other[0,0]-same[0,0]==pytest.approx(1.)


def test_unknown_is_not_free_identity_and_soft_risk_never_vetoes_source():
    a=features([observation(0,vector())],fps=30.)
    detail=matrix(a,features([observation(1,np.zeros(128))],fps=30.),details=True)
    assert detail['appearance_rate'][0,0]==.5 and not detail['appearance_measured'][0,0]
    assert detail['allowed'][0,0]


def test_authored_and_native_events_have_identical_appearance_risk_and_fourth_face_cannot_borrow():
    heads=np.zeros((1,2,8,4));heads[:,:,0]=[.5,.3,.2,.99]
    v=np.zeros((1,2,3,128));v[0,0,0]=vector();v[0,1,0]=vector(-1.)
    arcs=ArcCosts.__new__(ArcCosts);arcs.native=[dict(frames=np.array([20]),heads=heads,vectors=v)]
    cameras=np.tile([1.,.5,.5],(1,1,1))
    native=arcs.native_rates(0,np.array([20]),cameras,cameras)
    authored=matrix(features([observation(0,vector())],fps=30.),
                    features([observation(1,vector(-1.))],fps=30.))[0]
    assert native[0,0]==pytest.approx(authored[0,0])
    heads[:,:,3]=[.5,.3,.4,.99]
    assert arcs.native_rates(0,np.array([20]),cameras,cameras)[0,0]==pytest.approx(1.)
    arcs.native[0]['observed']=np.array([False])
    # An unobserved registry row cannot reuse a real-looking stale vector.
    assert arcs.native_rates(0,np.array([20]),cameras,cameras)[0,0]==pytest.approx(2.)


def test_single_cosine_is_not_a_named_member_threshold():
    a=vector();b=np.zeros(128);b[:2]=[.42,np.sqrt(1-.42**2)]
    risk,measured=discrepancy(a,b)
    assert measured and risk==pytest.approx(.29)
