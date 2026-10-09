"""Decoded-head feedback geometry, not real-media editorial approval."""
from copy import deepcopy

import numpy as np
import pytest

from crosscut.group_review_camera_repair import _rotated_point
from crosscut.group_review_roll_focal_camera import coupled_rendered_face_height_cameras
from crosscut.ive import AspectRatio
from crosscut.render import EditorialTransformKeyframe,_rotated_crop_is_inside_frame


def face(x=.45,y=.38,h=.25):
    return {'box_xywh':[x-.07,y-h/2,.14,h]}


def displayed_after(face,old,new):
    box=np.asarray(face['box_xywh']);center=box[:2]+box[2:]/2
    native=_rotated_point(np.array(old[1:3])+(center-.5)/old[0],-old[3])
    rotated=np.asarray(_rotated_point(native,new[3]))
    return (rotated-np.asarray(new[1:3]))*new[0]+.5,box[3]*new[0]/old[0]


@pytest.mark.parametrize('rolls',[(0.,0.),(6.,-6.),(-4.6,4.6)])
def test_displayed_height_and_center_match_without_shrinking_or_changing_roll(rolls):
    faces=(face(.46,.42,.25),face(.51,.45,.29))
    old=([1.5,.49,.5,rolls[0]],[1.5,.51,.5,rolls[1]])
    untouched=deepcopy((faces,old))
    cameras=coupled_rendered_face_height_cameras(faces,old)
    assert cameras is not None
    views=[displayed_after(f,o,n) for f,o,n in zip(faces,old,cameras)]
    assert np.linalg.norm(views[0][0]-views[1][0])<1e-9
    assert views[0][1]==pytest.approx(views[1][1])
    assert views[0][1]>=.29-1e-9
    assert [v[3] for v in cameras]==list(rolls)
    assert all(v[0]>=o[0]-1e-9 for v,o in zip(cameras,old))
    assert (faces,old)==untouched
    for camera in cameras:
        EditorialTransformKeyframe(0.,*camera)
        assert _rotated_crop_is_inside_frame(zoom=camera[0],center_x=camera[1],center_y=camera[2],
            roll_degrees=camera[3],aspect_ratio=AspectRatio.LANDSCAPE)


def test_matching_pair_keeps_actual_original_cameras():
    faces=(face(.47,.42),face(.47,.42))
    old=([1.5,.48,.49,0.],[1.6,.52,.54,0.])
    cameras=coupled_rendered_face_height_cameras(faces,old)
    assert np.allclose(cameras,old,rtol=0,atol=1e-12)


def test_unreachable_shared_height_is_unknown_not_a_clamp_or_shrink():
    assert coupled_rendered_face_height_cameras((face(h=.07),face(h=.4)),
        ([1.5,.5,.5,0.],[1.5,.5,.5,0.])) is None


@pytest.mark.parametrize('bad',[None,[.5,.3,0.,.2],[.5,.3,.2,float('nan')],[-.1,.3,.2,.2],[.9,.3,.2,.2]])
def test_invalid_or_outside_measured_box_does_not_enter_geometry(bad):
    with pytest.raises(ValueError,match='finite contained'):
        coupled_rendered_face_height_cameras(({'box_xywh':bad},face()),
            ([1.5,.5,.5,0.],[1.5,.5,.5,0.]))


def test_uncovered_source_and_relaxed_capacity_are_rejected():
    with pytest.raises(ValueError,match='uncovered'):
        coupled_rendered_face_height_cameras((face(),face()),([1.,.5,.5,6.],[1.5,.5,.5,0.]))
    with pytest.raises(ValueError,match='finite selected'):
        coupled_rendered_face_height_cameras((face(),face()),([1.5,.5,.5,0.],)*2,maximum_zoom=3.)
