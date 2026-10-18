import numpy as np
import pytest
from crosscut2.member_pair_camera import fit,score


def test_joint_member_cameras_match_observed_face_size_in_actual_portrait_metric():
    a=np.array([.45,.2,.1,.1]);b=np.array([.42,.2,.16,.16]);pose=np.zeros((8,3))
    cameras,receipt=fit(a,b,pose,pose,existing=[[1.,.5,.5]]*2,output_size=(1080,1920))
    assert receipt['after']['cost']<receipt['before']['cost']
    assert cameras[0,0]>.99 and cameras[1,0]>=1.
    assert abs(np.log(a[3]*cameras[0,0]/(b[3]*cameras[1,0])))<1e-8
    for z,x,y in cameras:
        assert .5/z-1e-9<=min(x,y) and max(x,y)<=1.-.5/z+1e-9
        assert np.linalg.norm(np.array([x,y])-.5)<=.25+1e-9


def test_a_good_existing_member_pair_is_not_changed_just_to_add_zoom():
    face=np.array([.4,.2,.15,.15]);pose=np.zeros((8,3));old=np.array([[1.,.5,.5]]*2)
    cameras,receipt=fit(face,face,pose,pose,existing=old,output_size=(1080,1920))
    np.testing.assert_array_equal(cameras,old);assert receipt['original_camera_retained']


def test_sampling_capacity_and_measured_arm_clipping_are_not_disguised_as_face_quality():
    face=np.array([.45,.2,.1,.1]);pose=np.zeros((8,3));pose[4]=(.01,.5,.9)
    a=score([2.,.5,.5],[2.,.5,.5],face,face,pose,pose,output_size=(1080,1920),sampling=(1.,1.))
    b=score([2.,.5,.5],[2.,.5,.5],face,face,pose,pose,output_size=(1080,1920),sampling=(2.,2.))
    assert a['added_arm_crop_fraction']>0 and a['added_sampling_effort']>b['added_sampling_effort']
    assert a['geometry']==b['geometry']


def test_missing_face_does_not_become_a_certain_member_camera_fit():
    with pytest.raises(ValueError,match='observed faces'):
        fit([0,0,0,0],[.4,.2,.1,.1],np.zeros((8,3)),np.zeros((8,3)),existing=[[1.,.5,.5]]*2,
            output_size=(1080,1920))
