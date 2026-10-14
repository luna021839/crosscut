import cv2
import numpy as np
import pytest
from crosscut2.rtmw_pose import prepare,decode,WholeBodyPose


def test_graph_sized_rgb_input_and_actual_affine_inverse_are_consistent():
    image=np.full((540,960,3),[10,20,30],np.uint8)
    tensor,center,scale,affine,crop=prepare(image,(.25,.1,.5,.8))
    assert tensor.shape==(1,3,384,288) and tensor.dtype==np.float32
    assert np.allclose(tensor[0,:,192,144],(np.array([30,20,10])-[123.675,116.28,103.53])/[58.395,57.12,57.375])
    assert np.allclose(np.r_[center,1.]@affine.T,[144,192])
    assert crop[2]>0 and crop[3]>0


def test_raw_wholebody_simcc_scores_are_not_probabilities_or_body17_aliases():
    sx=np.zeros((1,133,576),np.float32);sy=np.zeros((1,133,768),np.float32)
    sx[:,:,288]=4.2;sy[:,:,384]=3.7
    raw=decode([sx,sy],np.array([480,270]),np.array([600,800]),raster_wh=(960,540))
    assert raw.shape==(133,3) and np.allclose(raw[:,:2],[.5,.5])
    assert (raw[:,2]==np.float32(3.7)).all() and not raw.flags.writeable
    evidence=WholeBodyPose('observed',raw,(0.,0.,1.,1.),np.zeros((2,3)))
    assert evidence.raw_keypoints_xyc.shape==(17,3) and evidence.wholebody_xyc.shape==(133,3)
    sx[0,0,288]=np.nan
    with pytest.raises(ValueError,match='actual finite'):decode([sx,sy],[480,270],[600,800],raster_wh=(960,540))
