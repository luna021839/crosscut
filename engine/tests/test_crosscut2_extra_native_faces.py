import numpy as np
from crosscut2.extra_native_faces import slots


def test_unrepresented_foreground_is_added_without_reordering_old_heads():
    old=np.zeros((8,4));old[0]=[.5,.4,.09,.9];before=old.copy()
    detected=np.array([[.5,.4,.10,.95],[.35,.6,.23,.9]])
    assert slots(old,detected,source_aspect=16/9)==((1,1),)
    np.testing.assert_array_equal(old,before)


def test_weak_but_real_old_measurement_is_not_overwritten_or_promoted():
    old=np.zeros((8,4));old[0]=[.5,.4,.09,.5]
    assert slots(old,np.array([[.3,.3,.2,.95]]),source_aspect=16/9)==((1,0),)
    assert old[0,3]==.5


def test_full_bank_keeps_every_old_source_observation_even_with_a_large_new_face():
    old=np.tile([.5,.4,.1,.9],(8,1))
    assert slots(old,np.array([[.2,.7,.3,.95]]),source_aspect=16/9)==()


def test_duplicate_new_detections_and_low_confidence_cannot_fill_extra_slots():
    new=np.array([[.5,.4,.1,.9],[.501,.401,.102,.95],[.2,.4,.15,.4]])
    assert len(slots(np.zeros((8,4)),new,source_aspect=16/9))==1
