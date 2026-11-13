import numpy as np
import pytest
from crosscut2.member_clothing_replace import reveal,composite


def test_replaced_clothing_is_opaque_not_a_whole_outfit_crossfade():
    mask=np.zeros((100,80),np.float32);mask[20:90,20:60]=1
    alpha=reveal(mask,.4)
    assert alpha[30,40]==1 and alpha[85,40]==0
    assert np.sum((alpha>0)&(alpha<1))<.15*np.count_nonzero(mask)
    assert np.array_equal(reveal(mask,1),mask)
    assert not reveal(mask,0).any()


def test_face_and_stage_remain_separate_from_opaque_clothes():
    a=np.full((100,80,3),(210,20,15),np.uint8)
    b=np.full_like(a,(10,210,25))
    garment=np.zeros((100,80),np.float32);garment[25:90,20:60]=1
    person=np.zeros_like(garment);person[8:90,20:60]=1
    args=dict(incoming_garment=garment,outgoing_person=person,incoming_person=person,
        incoming_to_outgoing=np.eye(2,3,dtype=np.float32))
    value=composite(a,b,progress=.4,**args)
    assert np.array_equal(value[30,40],b[30,40])
    assert np.array_equal(value[12,40],a[12,40])
    assert np.array_equal(value[50,5],a[50,5])
    assert np.array_equal(composite(a,b,progress=0.,**args),a)
    assert np.array_equal(composite(a,b,progress=1.,**args),b)
    with pytest.raises(ValueError):composite(a,b,progress=2.,**args)


def test_new_neckline_can_replace_old_fabric_with_actual_incoming_skin():
    a=np.full((100,80,3),(210,20,15),np.uint8);b=np.full_like(a,(10,210,25))
    garment=np.zeros((100,80),np.float32);garment[45:90,20:60]=1
    person=np.zeros_like(garment);person[8:90,20:60]=1
    points=np.array([[32,12],[48,12],[40,16],[35,20],[45,20]],float)
    result=composite(a,b,incoming_garment=garment,outgoing_person=person,incoming_person=person,
        incoming_to_outgoing=np.eye(2,3,dtype=np.float32),incoming_face_landmarks=points,progress=.5)
    assert np.array_equal(result[35,40],b[35,40])
    assert np.array_equal(result[12,40],a[12,40])
