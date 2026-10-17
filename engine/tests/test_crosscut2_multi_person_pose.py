import numpy as np
import pytest
from crosscut2.multi_person_pose import tensor,decode,bind_faces,BODY8_COCO17


@pytest.mark.parametrize('shape',[(720,1280),(1280,720)])
def test_full_frame_letterbox_keeps_portrait_and_landscape_geometry_and_BGR(shape):
    image=np.zeros((*shape,3),np.uint8);image[:]=[7,13,29]
    x,ratio=tensor(image)
    assert x.shape==(1,3,640,640) and x.dtype==np.float32 and x.flags.c_contiguous
    assert ratio==640/max(shape)
    np.testing.assert_array_equal(x[0,:,0,0],[7,13,29])
    if shape[0]<shape[1]:np.testing.assert_array_equal(x[0,:,-1,0],[114]*3)
    else:np.testing.assert_array_equal(x[0,:,0,-1],[114]*3)


def test_duplicate_body_hypotheses_do_not_invent_more_people_and_no_input_is_mutated():
    det=np.array([[[10,20,90,200,.9],[11,21,91,201,.8],[120,20,200,200,.7]]],np.float32)
    pose=np.zeros((1,3,17,3),np.float32);pose[...,:2]=[80,160];pose[...,2]=.7
    before=pose.copy();value=decode((det,pose),ratio=.5,raster=(640,480))
    assert value['raw_detection_indices'].tolist()==[0,2]
    assert value['above_score_before_nms']==3 and value['body8_indices']==BODY8_COCO17
    np.testing.assert_allclose(value['keypoints17'][...,:2],np.broadcast_to([.25,2/3],(2,17,2)))
    np.testing.assert_array_equal(pose,before)
    assert value['evidence']=='MODEL_ONLY' and not value['named_identity_certified']


def test_overlapping_distinct_neighbours_remain_and_low_score_is_not_a_named_identity_decision():
    det=np.array([[[0,0,80,160,.8],[40,0,120,160,.8],[200,0,280,160,.2]]],np.float32)
    pose=np.zeros((1,3,17,3),np.float32);pose[...,2]=.8
    value=decode((det,pose),ratio=1.,raster=(640,640))
    assert value['raw_detection_indices'].tolist()==[0,1]
    # Keep an out-of-image prediction as raw model output; do not forge it
    # into a measured on-screen hand by clamping to the image border.
    pose[0,0,9,0]=700
    assert decode((det,pose),ratio=1.,raster=(640,640))['keypoints17'][0,9,0]>1.


def test_empty_scene_and_wrong_model_family_are_distinct():
    det=np.zeros((1,0,5),np.float32);pose=np.zeros((1,0,17,3),np.float32)
    assert decode((det,pose),ratio=1.,raster=(640,640))['keypoints17'].shape==(0,17,3)
    with pytest.raises(ValueError,match='tensor family'):
        decode((det,np.zeros((1,0,133,3))),ratio=1.,raster=(640,640))


def test_one_body_cannot_supply_the_limbs_of_two_neighbouring_faces():
    heads=np.array([[.4,.4,.12,.99],[.43,.4,.12,.99],[.8,.4,.12,.99]])
    points=np.zeros((2,17,3),np.float32);points[...,2]=.9
    points[0,:,:2]=[.4,.4];points[1,:,:2]=[.8,.4]
    result=bind_faces(heads,dict(keypoints17=points,scores=np.array([.9,.8])),source_aspect=16/9)
    assert result['person_index'].tolist()==[0,-1,1]
    assert not result['supported'][1].any()
    assert len(set(result['person_index'][result['person_index']>=0]))==2
    assert not result['anatomical_ownership_certified']


def test_a_body_without_a_visible_head_anchor_does_not_invent_an_actor():
    points=np.zeros((1,17,3));points[:,5:,2]=.9
    result=bind_faces([[.5,.4,.1,.99]],dict(keypoints17=points,scores=np.array([.9])),source_aspect=9/16)
    assert result['person_index'].tolist()==[-1]


def test_the_sensor_uses_the_shared_inverse_exactly_once_without_opening_a_model():
    from crosscut2.rtmo_original_people import Sensor
    from types import SimpleNamespace
    det=np.array([[[10,20,100,300,.9]]],np.float32)
    pose=np.zeros((1,1,17,3),np.float32);pose[...,:2]=[320,180];pose[...,2]=.8
    sensor=Sensor.__new__(Sensor);sensor.minimum_score=.35;sensor.forwards=0
    sensor.session=SimpleNamespace(run=lambda outputs,inputs:(det,pose))
    result=sensor.describe(np.zeros((720,1280,3),np.uint8))
    np.testing.assert_allclose(result['keypoints17'][0,:,:2],np.tile([.5,.5],(17,1)))
    assert sensor.forwards==1
