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


def test_equal_heads_do_not_conceal_a_displaced_observed_wrist():
    face=np.array([.45,.2,.1,.1]);a=np.zeros((8,3));a[:6]=[
        [.4,.4,.9],[.6,.4,.9],[.35,.5,.9],[.65,.5,.9],[.3,.6,.9],[.7,.6,.9]]
    b=a.copy();b[4,0]=.6
    same=score([1.,.5,.5],[1.,.5,.5],face,face,a,a,output_size=(1080,1920))
    wrong=score([1.,.5,.5],[1.,.5,.5],face,face,a,b,output_size=(1080,1920))
    assert same['geometry']==wrong['geometry']==0.
    assert same['measured_arm_position_in_face_heights']==0.
    assert wrong['measured_arm_position_in_face_heights']>1 and wrong['cost']>same['cost']


def test_joint_registration_is_an_actual_camera_candidate_not_invented_matching_limbs():
    face_a=np.array([.45,.2,.1,.1]);face_b=np.array([.45,.2,.1,.1])
    a=np.zeros((8,3));a[:6]=[[.4,.4,.9],[.6,.4,.9],[.35,.5,.9],[.65,.5,.9],[.3,.6,.9],[.7,.6,.9]]
    b=a.copy();b[:6,:2]=[.5,.25]+1.5*(a[:6,:2]-[.5,.25]);before=a.copy(),b.copy()
    cameras,receipt=fit(face_a,face_b,a,b,existing=[[1.,.5,.5]]*2,output_size=(1080,1920))
    assert receipt['after']['cost']<receipt['before']['cost']
    assert receipt['after']['measured_arm_position_in_face_heights']<receipt['before']['measured_arm_position_in_face_heights']
    np.testing.assert_array_equal(a,before[0]);np.testing.assert_array_equal(b,before[1])
