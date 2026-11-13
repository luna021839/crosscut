import numpy as np
import pytest

from crosscut2.boundary_arc_costs import ArcCosts
from crosscut2.played_partial_pose import features, matrix, out_of_view_face_risk
from test_crosscut2_focal_appearance import catalogue, sample


def pair():
    cat=catalogue();cat.bank['heads'][:,:,1:]=0.
    left=sample(cat,0,[2.,.5,.75]);right=sample(cat,1,[1.,.5,.5])
    assert left['head_index']==-1 and right['head_index']==0
    return cat,left,right


def test_known_face_cropped_out_does_not_receive_a_cheaper_unknown_jump():
    _,left,right=pair()
    a,b=features([left],fps=30.),features([right],fps=30.)
    detail=matrix(a,b,details=True)
    assert not detail['measured_face'][0,0] and detail['face_rate'][0,0]==1.
    assert detail['cropped_original_face_geometry_measured'][0,0]
    assert detail['geometry_rate'][0,0]>detail['face_rate'][0,0]
    assert detail['allowed'][0,0]  # finite cost, no new source/camera gate
    assert detail['matched_action_change_density'][0,0]==0.


def test_missing_original_observation_stays_unknown_and_cannot_borrow_a_nearby_face():
    _,left,right=pair()
    left=dict(left,source_clock_original_heads=np.zeros((4,8,4)))
    detail=matrix(features([left],fps=30.),features([right],fps=30.),details=True)
    assert detail['geometry_rate'][0,0]==1.
    assert not detail['cropped_original_face_geometry_measured'][0,0]
    wrong=np.zeros((4,8,4));wrong[0,0]=[.5,.3,.2,.99]
    left['source_clock_original_heads']=wrong  # preceding pose sample is NOT the endpoint
    assert not matrix(features([left],fps=30.),features([right],fps=30.),details=True)[
        'cropped_original_face_geometry_measured'].any()


def test_visible_focal_cannot_be_reassigned_to_an_easier_hidden_other_face():
    raw=np.zeros((1,8,4));raw[0,0]=[.5,.3,.2,.99]
    other=raw.copy();other[0,0]=[.5,.8,.2,.99];other[0,1]=raw[0,0]
    camera=np.array([[2.,.5,.75]])
    risk,known=out_of_view_face_risk(raw,np.zeros((1,4)),camera,np.array([False]),
        other,other[:,0],np.array([[1.,.5,.5]]),np.array([True]))
    assert known[0] and risk[0]>3.
    unchanged,known=out_of_view_face_risk(raw,raw[:,0],camera,np.array([True]),
        other,other[:,0],camera,np.array([True]))
    assert unchanged[0]==0. and not known[0]


def test_retained_native_and_authored_boundaries_pay_the_same_known_crop_risk():
    cat,left,right=pair()
    arc=object.__new__(ArcCosts)
    arc.native=[dict(frames=np.array([100]),heads=cat.bank['heads'][1:2],
        vectors=cat.bank['vectors'][1:2],observed=np.array([True]))]
    before=np.array([[[2.,.5,.75]]]);after=np.array([[[1.,.5,.5]]])
    native=arc.native_rates(0,np.array([100]),before,after)
    detail=matrix(features([left],fps=30.),features([right],fps=30.),details=True)
    expected=detail['geometry_rate']+detail['pose_rate']+detail['appearance_rate']
    np.testing.assert_allclose(native,expected)
    assert np.isfinite(native).all()
    arc.native[0]['observed'][:]=False
    np.testing.assert_allclose(arc.native_rates(0,np.array([100]),before,after),2.)
