"""Decoded-head feedback geometry, not real-media editorial approval."""
from copy import deepcopy
import math

import numpy as np
import pytest

from crosscut.group_review_camera_repair import _rotated_point
from crosscut.group_review_roll_focal_camera import (coupled_rendered_face_height_cameras,
    coupled_native_face_height_cameras,cover_interpolated_native_camera,cover_native_camera_sequence)
from crosscut.ive import AspectRatio
from crosscut.render import EditorialTransformKeyframe,EditorialTransformTrack,_rotated_crop_is_inside_frame


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


@pytest.mark.parametrize('rolls',[(0.,0.),(6.,-6.)])
def test_native_candidate_height_target_is_realized_before_cost_without_relabelling_pixels(rolls):
    rows=({'face':face(.4,.35,.2)},{'face':face(.6,.38,.24)})
    original=deepcopy(rows)
    cameras=coupled_native_face_height_cameras(*rows,rolls=rolls,minimum_face_height=.34)
    assert cameras is not None
    centers=[];heights=[]
    for row,camera in zip(rows,cameras):
        box=row['face']['box_xywh'];z,cx,cy,roll=camera
        center=_rotated_point((box[0]+box[2]/2,box[1]+box[3]/2),roll)
        centers.append((np.asarray(center)-[cx,cy])*z+.5)
        radians=math.radians(roll)
        heights.append(z*(box[3]*abs(math.cos(radians))+box[2]*16/9*abs(math.sin(radians))))
    assert heights[0]==pytest.approx(heights[1])
    assert min(heights)>=.34-1e-9
    assert np.linalg.norm(centers[0]-centers[1])<1e-9
    assert rows==original


def test_native_candidate_cannot_shrink_or_relax_to_fit_a_wide_head():
    assert coupled_native_face_height_cameras({'face':face(h=.03)},{'face':face(h=.24)},
        rolls=(0.,0.),minimum_face_height=.34) is None


def test_interpolated_rotated_canvas_is_covered_by_physical_zoom_not_fill_or_pan_clamp():
    raw=[1.23,.404,.583,.4];original=raw.copy()
    result=cover_interpolated_native_camera(raw)
    assert result is not None
    assert result[0]>=raw[0]
    assert result[1:]==raw[1:]
    assert raw==original
    EditorialTransformKeyframe(0.,*result)
    assert _rotated_crop_is_inside_frame(zoom=result[0],center_x=result[1],center_y=result[2],
        roll_degrees=result[3],aspect_ratio=AspectRatio.LANDSCAPE)


def test_already_covered_camera_remains_exactly_unchanged_and_capacity_cannot_expand():
    raw=[1.5,.5,.5,0.]
    assert cover_interpolated_native_camera(raw)==raw
    with pytest.raises(ValueError,match='unchanged production envelope'):
        cover_interpolated_native_camera([2.5,.5,.5,0.])


def test_continuous_native_sequence_keeps_both_endpoints_and_renderer_subdivisions_covered():
    raw=[[1.3,.45,.52,0.],cover_interpolated_native_camera([1.15,.45,.52,3.]),
        [1.5,.46,.5,6.]]
    original=deepcopy(raw);result=cover_native_camera_sequence(raw)
    assert result is not None
    assert result[0]==raw[0] and result[-1]==raw[-1]
    assert all(a[0]>=b[0] and a[1:]==b[1:] for a,b in zip(result,raw))
    track=EditorialTransformTrack(tuple(EditorialTransformKeyframe(float(i),*v) for i,v in enumerate(result)))
    assert track.covers_aspect_ratio(AspectRatio.LANDSCAPE)
    assert raw==original


def test_two_endpoint_sequence_cannot_change_frozen_ends_to_cover_a_missing_interior():
    assert cover_native_camera_sequence([[1.5,.5,.5,0.],[1.5,.5,.5,6.]]) is not None
    with pytest.raises(ValueError,match='already cover'):
        cover_native_camera_sequence([[1.,.5,.5,0.],[1.,.5,.5,6.]])
