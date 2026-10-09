"""Keep camera representation capacity separate from actual output coverage."""
from pathlib import Path

import pytest

from crosscut.ive import AspectRatio
from crosscut.render import (
    EDITORIAL_TRANSFORM_MAXIMUM_ROLL, EditorialTransformKeyframe,
    EditorialTransformTrack, RenderPlan, RenderSegment, _rotated_crop_is_inside_frame,
    minimum_roll_safe_zoom,
)
from crosscut.render_plan_serialization import render_plan_from_json


def plan_for(track, aspect=AspectRatio.LANDSCAPE):
    return RenderPlan(sources=(Path('input.mp4'),),
        segments=(RenderSegment(0, 0., 1., 0., editorial_transform=track),),
        reference_source=0, aspect_ratio=aspect, output=Path('output.mp4'),
        width=1920 if aspect is AspectRatio.LANDSCAPE else 1080,
        height=1080 if aspect is AspectRatio.LANDSCAPE else 1920)


@pytest.mark.parametrize('roll', (-6., 6.))
def test_physically_covered_six_degree_camera_can_be_represented(roll):
    key = EditorialTransformKeyframe(0., 1.5, roll_degrees=roll)
    track = EditorialTransformTrack((key,))
    assert key.roll_degrees == roll
    assert EDITORIAL_TRANSFORM_MAXIMUM_ROLL == 6.
    assert plan_for(track).segments[0].editorial_transform is track


@pytest.mark.parametrize('roll', (-6.001, 6.001))
def test_representation_still_has_a_physical_roll_cap(roll):
    with pytest.raises(ValueError, match='physical-roll'):
        EditorialTransformKeyframe(0., 2., roll_degrees=roll)


@pytest.mark.parametrize('roll', (-6., 6.))
def test_wider_roll_never_waives_real_frame_edge_coverage(roll):
    with pytest.raises(ValueError, match='frame edges'):
        EditorialTransformKeyframe(0., 1.05, roll_degrees=roll)


@pytest.mark.parametrize('camera', (
    (1.375759654023746, .38380804203898966, .46232565572877815, -6.),
    (1.2807258066532015, .5886776638305836, .5661658484259574, 4.603508566316332),
))
def test_observed_landscape_camera_with_sampling_margin_uses_its_actual_aspect(camera):
    # Review raster and production chroma sampling have different edge margins.
    # One explicit one-percent output overscan preserves paired head matching.
    zoom, x, y, roll = camera
    padded = (zoom*1.01, x, y-(1-1/1.01)*.08/zoom, roll)
    track = EditorialTransformTrack((EditorialTransformKeyframe(0., *padded),))
    assert track.covers_aspect_ratio(AspectRatio.LANDSCAPE)
    assert not track.covers_aspect_ratio(AspectRatio.PORTRAIT)
    plan_for(track)
    with pytest.raises(ValueError, match='output aspect'):
        plan_for(track, AspectRatio.PORTRAIT)


def test_portrait_specific_camera_remains_bound_to_its_own_actual_output():
    # Reciprocal geometry of the actual landscape-only incoming camera.
    key = EditorialTransformKeyframe(0., 1.375759654023746,
                                    .46232565572877815, .38380804203898966, 6.)
    track = EditorialTransformTrack((key,))
    assert track.covers_aspect_ratio(AspectRatio.PORTRAIT)
    assert not track.covers_aspect_ratio(AspectRatio.LANDSCAPE)
    plan_for(track, AspectRatio.PORTRAIT)
    with pytest.raises(ValueError, match='output aspect'):
        plan_for(track)


def test_one_track_cannot_mix_incompatible_orientation_specific_cameras():
    landscape = EditorialTransformKeyframe(0., 1.375759654023746,
                                          .38380804203898966, .46232565572877815, -6.)
    portrait = EditorialTransformKeyframe(.5, 1.375759654023746,
                                         .46232565572877815, .38380804203898966, 6.)
    with pytest.raises(ValueError, match='frame edges'):
        EditorialTransformTrack((landscape, portrait))


def test_interpolated_rotation_still_checks_edges_not_just_endpoints():
    # Safe endpoints do not prove intermediate rotation coverage. The small
    # rotation's overscan function is not the linear chord of these endpoints.
    first = EditorialTransformKeyframe(0., 1.)
    last = EditorialTransformKeyframe(.5, minimum_roll_safe_zoom(6.), roll_degrees=6.)
    with pytest.raises(ValueError, match='frame edges'):
        EditorialTransformTrack((first, last))


def test_review_raster_only_margin_is_not_silently_accepted_for_production():
    # This actual r21 endpoint has a two-pixel review margin, insufficient for
    # the unchanged production chroma footprint. Capacity must not waive it.
    with pytest.raises(ValueError, match='frame edges'):
        EditorialTransformKeyframe(0., 1.2807258066532015,
                                  .5886776638305836, .5661658484259574, 4.603508566316332)


@pytest.mark.parametrize('roll', (float('nan'), float('inf')))
def test_invalid_roll_is_not_an_unknown_but_rejected(roll):
    with pytest.raises(ValueError, match='finite'):
        EditorialTransformKeyframe(0., 2., roll_degrees=roll)


@pytest.mark.parametrize('roll', (0., 1.6, 6.))
def test_exact_serialized_plan_has_no_new_orientation_or_capacity_default(roll):
    track = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.5, roll_degrees=roll),))
    plan = plan_for(track)
    payload = plan.to_json()
    key = payload['segments'][0]['editorial_transform']['keyframes'][0]
    assert set(key) == {'time_seconds', 'zoom', 'center_x', 'center_y', 'roll_degrees'}
    assert render_plan_from_json(payload).to_json() == payload


def test_shared_coverage_helper_default_stays_conservative():
    camera = dict(zoom=1.375759654023746, center_x=.38380804203898966,
                  center_y=.46232565572877815, roll_degrees=-6.)
    assert not _rotated_crop_is_inside_frame(**camera)
    assert _rotated_crop_is_inside_frame(**camera, aspect_ratio=AspectRatio.LANDSCAPE)
