"""Real LK on deterministic nonhuman textures, not human identity evidence."""
import cv2
import numpy as np
import pytest

from crosscut.group_focal_attention import focal_landmarks_track


def chart(dx=40):
    rng=np.random.default_rng(983)
    first=cv2.GaussianBlur(rng.integers(0,256,(540,960,3),dtype=np.uint8),(7,7),1.5)
    second=cv2.warpAffine(first,np.array([[1,0,dx],[0,1,8]],np.float32),(960,540))
    points=np.array([[400,220],[460,220],[430,245],[405,265],[455,265]],np.float32)
    return first,second,points/[960,540],(points+[dx,8])/[960,540]


@pytest.mark.parametrize('dx',[20,40,60])
def test_native_coarse_to_fine_search_recovers_measured_translation(dx):
    assert focal_landmarks_track(*chart(dx))


def test_real_pixels_do_not_match_another_landmark_destination():
    a,b,before,after=chart()
    assert not focal_landmarks_track(a,b,before,after+[.1,0])


def test_flat_pixels_cannot_gain_a_path_from_deeper_pyramid():
    a,b,before,after=chart()
    assert not focal_landmarks_track(np.zeros_like(a),np.zeros_like(b),before,after)


def test_only_three_matching_points_remain_insufficient():
    a,b,before,after=chart()
    after[:2]+=[.1,0]
    assert not focal_landmarks_track(a,b,before,after)


def test_unrelated_texture_does_not_match_a_claimed_translation():
    a,_,before,after=chart()
    b=np.random.default_rng(777).integers(0,256,a.shape,dtype=np.uint8)
    assert not focal_landmarks_track(a,b,before,after)


@pytest.mark.parametrize('invalid',[float('nan'),float('inf'),-1.,2.])
def test_nonfinite_or_outside_native_points_remain_rejected(invalid):
    a,b,before,after=chart()
    after[0,0]=invalid
    assert not focal_landmarks_track(a,b,before,after)
