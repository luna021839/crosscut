import numpy as np
import pytest
from crosscut.focal_selector_geometry import FocalSelectorGeometry
from crosscut2.aspect_viewport import aspect_crop,project_points
from crosscut2.member_viewport_observations import project,compose_camera,face_centered_candidate


def observation():
    return FocalSelectorGeometry(2,42.,(.76,.3,.08,.10),pose_screen_xyc=tuple(
        (x,y,c) for x,y,c in ((.7,.4,.9),(.8,.4,.8),(.6,.5,.7),(.9,.5,.6),
                            (.2,.6,.5),(.95,.6,.4),(.7,.8,.3),(.8,.8,.2))))


@pytest.mark.parametrize('size',[(2160,3840),(3840,2160)])
def test_source_face_and_screen_pose_use_actual_portrait_crop_without_altering_evidence(size):
    original=observation();before=original.to_dict();raw=original.pose_screen_xyc
    crop=aspect_crop(size,(1080,1920),center=(.8,.5))
    result=project(original,crop,source_size=size,output_size=(1080,1920))
    assert result.source is original and original.to_dict()==before and original.pose_screen_xyc==raw
    expected=project_points(np.asarray(raw)[:,:2],crop)
    np.testing.assert_allclose(np.asarray(result.pose_screen_xyc)[:,:2],expected)
    np.testing.assert_array_equal(np.asarray(result.pose_screen_xyc)[:,2],np.asarray(raw)[:,2])
    assert result.native_face_height_pixels==pytest.approx(.1*size[1])
    assert result.face_bbox_xywh[3]==pytest.approx(.1/crop.height)


def test_offscreen_arm_is_not_clamped_or_replaced_by_torso_normalized_shape():
    original=observation();crop=aspect_crop((3840,2160),(1080,1920),center=(.8,.5))
    result=project(original,crop,source_size=(3840,2160),output_size=(1080,1920))
    assert result.pose_screen_xyc[4][0]<0 and result.pose_screen_xyc[4][2]==.5
    # A record without genuine screen joints stays unmeasured.
    no_pose=FocalSelectorGeometry(2,42.,(.76,.3,.08,.10))
    assert project(no_pose,crop,source_size=(3840,2160),output_size=(1080,1920)).pose_screen_xyc==()


def test_physical_member_motion_does_not_use_landscape_metric_on_portrait_output():
    original=observation();crop=aspect_crop((2160,3840),(1080,1920))
    result=project(original,crop,source_size=(2160,3840),output_size=(1080,1920))
    points=np.asarray(result.pose_screen_xyc);metric=np.asarray(result.metric_pose_xyc)
    np.testing.assert_allclose(metric[:,0],points[:,0]*9/16)
    np.testing.assert_array_equal(metric[:,1:],points[:,1:])


def test_composed_camera_matches_sequential_point_projection_and_keeps_native_detail():
    original=observation();size=(3840,2160);output=(1080,1920)
    base=aspect_crop(size,output,center=(.8,.5));camera=np.array([1.5,.55,.45])
    crop=compose_camera(base,camera);result=project(original,crop,source_size=size,output_size=output)
    a=project_points(np.asarray(original.pose_screen_xyc)[:,:2],base)
    expected=.5+camera[0]*(a-camera[1:])
    np.testing.assert_allclose(np.asarray(result.pose_screen_xyc)[:,:2],expected)
    assert result.native_face_height_pixels==pytest.approx(.1*size[1])
    assert result.source is original and result.face_visible_fraction==pytest.approx(1.)


def test_invalid_camera_or_output_aspect_is_not_silently_stretched_or_recentered():
    crop=aspect_crop((3840,2160),(1080,1920))
    with pytest.raises(ValueError,match='escapes'):compose_camera(crop,[1.,.8,.5])
    with pytest.raises(ValueError,match='physical output aspect'):
        project(observation(),crop,source_size=(3840,2160),output_size=(1920,1080))


def test_real_member_face_gets_an_additive_landscape_pan_candidate_not_a_source_veto():
    original=observation();size=(3840,2160);output=(1080,1920)
    baseline=aspect_crop(size,output)
    chosen=face_centered_candidate(original,source_size=size,output_size=output)
    assert chosen.width==baseline.width and chosen.height==baseline.height
    assert chosen.x>baseline.x
    view=project(original,chosen,source_size=size,output_size=output)
    assert view.face_visible_fraction==pytest.approx(1.)
    assert baseline.x<chosen.x and project(original,baseline,source_size=size,output_size=output).face_visible_fraction<1.


def test_missing_face_uses_the_declared_baseline_and_does_not_copy_a_torso_shape_target():
    original=FocalSelectorGeometry(0,42.,None)
    size=(3840,2160);output=(1080,1920);center=(.3,.5)
    assert face_centered_candidate(original,source_size=size,output_size=output,fallback_center=center)==aspect_crop(size,output,center=center)
    assert face_centered_candidate(observation(),source_size=(2160,3840),output_size=output)==aspect_crop((2160,3840),output)


def test_feasible_camera_at_source_edge_does_not_create_negative_ulp_crop():
    from crosscut.render import NormalizedCrop
    z=1.3;half=.5/z
    crop=compose_camera(NormalizedCrop(0.,0.,1.,1.),[z,np.nextafter(half,0.),np.nextafter(half,0.)])
    assert crop.x==0. and crop.y==0.
    assert crop.width==pytest.approx(1./z)
