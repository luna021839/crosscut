import cv2
import numpy as np
import pytest
from crosscut2.temporal_parts import observe


def scene(steps=1,shift=4):
    image=np.random.default_rng(417).integers(0,256,(240,400,3),dtype=np.uint8)
    points=np.array([[.3,.25,.9],[.65,.25,.9],[.25,.4,.9],[.7,.4,.9],
        [.5,.6,.9],[.74,.6,.9],[.4,.8,.9],[.6,.8,.9]])
    # Explicit local texture, rather than assuming every random13px patch
    # happens to have at least three independently trackable corners.
    for x,y,_ in points:
        x,y=round(x*400),round(y*240)
        cv2.rectangle(image,(x-6,y-6),(x+6,y+6),(0,0,0),-1)
        cv2.rectangle(image,(x-3,y-3),(x+3,y+3),(255,255,255),-1)
    frames=[cv2.warpAffine(image,np.float32([[1,0,shift*k],[0,1,0]]),(400,240)) for k in range(steps+1)]
    final=points.copy();final[:,0]+=shift*steps/400
    return frames,points,final


def test_actual_raw_wrist_can_survive_other_wrist_switching_to_background():
    frames,a,b=scene();old_a=a.copy();old_b=b.copy();b[5,:2]+=[-.2,-.25]
    result=observe(frames,a,b,head_heights=[.3,.3])
    assert result.shoulder_anchor_observed and result.supported[4] and not result.supported[5]
    assert result.model_pixel_disagreement[5] and result.worst_error_face_heights[5]>.5
    assert np.array_equal(a,old_a) and np.array_equal(b[:5],old_b[:5])


def test_absent_texture_is_unknown_not_stationary_or_measured_mismatch():
    frames,a,b=scene();frames=[np.zeros_like(f) for f in frames]
    result=observe(frames,a,b,head_heights=[.3,.3])
    assert not result.supported.any() and not result.pixel_path_observed.any()
    assert not result.model_pixel_disagreement.any() and np.isnan(result.worst_error_face_heights).all()


def test_fast_six_step_gesture_has_no_total_displacement_point_zero_eight_veto():
    frames,a,b=scene(steps=6,shift=8)
    result=observe(frames,a,b,head_heights=[.3,.3])
    assert (b[4,0]-a[4,0])>.08 and result.supported[4]


def test_matching_wrist_without_a_source_connected_shoulder_stays_unknown():
    frames,a,b=scene();a[:2,2]=b[:2,2]=0.
    result=observe(frames,a,b,head_heights=[.3,.3])
    assert result.pixel_path_observed[4] and not result.shoulder_anchor_observed and not result.supported.any()


def test_frame_budget_and_raster_changes_fail_closed_not_interpolated():
    frames,a,b=scene()
    with pytest.raises(ValueError):observe(frames*7,a,b,head_heights=[.3,.3])
    with pytest.raises(ValueError):observe([frames[0],frames[1][:100]],a,b,head_heights=[.3,.3])
