import copy
import numpy as np
import pytest
from crosscut2.member_body_context import framing_loss


def body():
    points=np.zeros((1,17,3));points[0,[5,6,11,12]]=[(.4,.3,.9),(.6,.3,.9),(.4,.7,.9),(.6,.7,.9)]
    return dict(keypoints17=points,scores=np.array([.9]))


def test_unmeasured_body_is_not_a_detector_observed_empty_frame():
    crops=[[0.,0.,1.,1.]]
    loss,known=framing_loss(None,crops)
    empty,observed=framing_loss(dict(keypoints17=np.zeros((0,17,3)),scores=np.zeros(0)),crops)
    assert loss[0]==.9 and not known[0] and empty[0]==4. and observed[0]


def test_visible_back_torso_does_not_require_an_invented_face_or_named_identity():
    actual=body();before=copy.deepcopy(actual)
    loss,known=framing_loss(actual,[[0.,0.,1.,1.]])
    assert known[0] and .55<loss[0]<.9
    np.testing.assert_array_equal(actual['keypoints17'],before['keypoints17'])
    assert not actual['keypoints17'][0,:5].any()


def test_camera_pointed_at_legs_without_shoulders_is_not_equal_to_a_back_view():
    actual=body();actual['keypoints17'][0,[5,6],1]=-.1
    loss,known=framing_loss(actual,[[0.,0.,1.,1.]])
    assert known[0] and loss[0]==4.


def test_partial_occlusion_is_soft_and_actual_crop_can_worsen_body_visibility():
    actual=body();actual['keypoints17'][0,5,2]=0.
    partial,_=framing_loss(actual,[[0.,0.,1.,1.],[0.,0.,1.,.4]])
    complete,_=framing_loss(body(),[[0.,0.,1.,1.]])
    assert complete[0]<partial[0]<4. and partial[1]==4.


def test_candidates_use_their_own_physical_view_not_a_shared_best_crop():
    loss,_=framing_loss(body(),[[0.,0.,1.,1.],[0.,0.,.3,1.]])
    assert loss[0]<.9 and loss[1]==4.
    with pytest.raises(ValueError):framing_loss(body(),[[0.,0.,0.,1.]])
