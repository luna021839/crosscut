from types import SimpleNamespace
import numpy as np
import pytest
from crosscut2.partial_pose import corroborate,face_proposals
from crosscut2.action_geometry import _arm_segments


HEAD=np.array([.5,.3,.15,.95])
def evidence():
    raw=np.zeros((17,3));raw[:,2]=.9;raw[0,:2]=[.5,.3]
    raw[[5,6,7,8,9,10,11,12],:2]=[[.42,.44],[.58,.44],[.35,.6],[.65,.6],
        [.3,.39],[.7,.39],[.45,.85],[.55,.85]]
    return SimpleNamespace(reason='observed',raw_keypoints_xyc=raw)


def test_fixed_crop_proposals_do_not_inspect_detector_ambiguity_or_pose_match():
    crops=face_proposals(HEAD)
    assert len(crops)==2 and crops[0]!=crops[1]
    assert all(min(b[:2])>=0 and min(b[2:])>0 and b[0]+b[2]<=1 and b[1]+b[3]<=1 for b in crops)
    good=evidence();partial=corroborate([good,good],HEAD,np.zeros((0,4)))
    assert partial.supported.all()
    assert np.array_equal(partial.points,good.raw_keypoints_xyc[[5,6,7,8,9,10,11,12]].astype(np.float32))


def test_neighbor_nose_admitted_by_old_ellipse_is_not_focal_evidence():
    # Exact geometry of the real probe, without an image or identity label.
    head=np.array([.515025,.497137,.332885,.95])
    wrong=evidence();wrong.raw_keypoints_xyc[0]=[.635988,.394196,.31]
    result=corroborate([wrong,wrong],head,np.zeros((0,4)))
    assert result.reason=='nose_outside_selected_face' and not result.supported.any()
    # A different detected face containing this nose is also explicit ambiguity.
    good=evidence();result=corroborate([good,good],HEAD,np.array([[.51,.30,.15,.96]]))
    assert result.reason=='competing_face_anchor' and not result.supported.any()


def test_one_forearm_disagreement_preserves_other_measured_complete_arm():
    primary,other=evidence(),evidence()
    other.raw_keypoints_xyc[7,:2]=[.35,.3]  # one crop grabs a background elbow
    result=corroborate([primary,other],HEAD,np.zeros((0,4)))
    assert result.supported.tolist()==[True,True,False,True,True,True,True,True]
    _,arms=_arm_segments(result.points)
    assert arms.tolist()==[False,False,True,True]
    assert np.array_equal(result.points[:,:2],primary.raw_keypoints_xyc[[5,6,7,8,9,10,11,12],:2].astype(np.float32))
    # A genuinely opposed but crop-agreed wrist is NOT erased or corrected.
    primary.raw_keypoints_xyc[10,:2]=[.6,.8];other.raw_keypoints_xyc[10,:2]=[.6,.8]
    result=corroborate([primary,other],HEAD,np.zeros((0,4)))
    assert result.supported[5] and result.points[5,1]==pytest.approx(.8)


def test_missing_crop_or_raw_joint_is_never_stationary_or_filled():
    a,b=evidence(),evidence();b.raw_keypoints_xyc[9,2]=.1
    result=corroborate([a,b],HEAD,np.zeros((0,4)))
    assert not result.supported[4] and result.points[4,2]==0
    assert result.points[4,0]==pytest.approx(.3)  # measured coordinate retained
    missing=corroborate([a,None],HEAD,np.zeros((0,4)))
    assert not missing.supported.any() and not missing.points.any()
    b.raw_keypoints_xyc[0,0]=np.nan
    with pytest.raises(ValueError,match='raw17'):corroborate([a,b],HEAD,np.zeros((0,4)))


def test_raw_simcc_peaks_above_one_survive_without_probability_claim():
    a,b=evidence(),evidence();a.raw_keypoints_xyc[:,2]=1.2;b.raw_keypoints_xyc[:,2]=1.4
    result=corroborate([a,b],HEAD,np.zeros((0,4)))
    assert result.supported.all() and (result.points[:,2]==1.).all()
    assert (a.raw_keypoints_xyc[:,2]==1.2).all() and (b.raw_keypoints_xyc[:,2]==1.4).all()
    b.raw_keypoints_xyc[9,2]=-.2
    result=corroborate([a,b],HEAD,np.zeros((0,4)))
    assert not result.supported[4] and result.points[4,2]==0
    assert b.raw_keypoints_xyc[9,2]==-.2
