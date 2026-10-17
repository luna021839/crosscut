import numpy as np
import pytest
import shutil
import subprocess

from crosscut2.aspect_viewport import aspect_crop,project_points,native_sampling_capacity


@pytest.mark.parametrize('source,output,expected',[
    ((2160,3840),(1080,1920),(1.,1.,2.)),
    ((1080,1920),(1080,1920),(1.,1.,1.)),
    ((3840,2160),(1080,1920),(81/256,1.,1.125)),
    ((1920,1080),(1080,1920),(81/256,1.,.5625)),
    ((1080,1920),(1920,1080),(1.,81/256,.5625)),
])
def test_mixed_source_aspects_use_real_crops_without_stretch_or_source_exclusion(source,output,expected):
    crop=aspect_crop(source,output)
    assert (crop.width,crop.height)==pytest.approx(expected[:2])
    assert crop.width*source[0]/(crop.height*source[1])==pytest.approx(output[0]/output[1])
    assert native_sampling_capacity(source,output)==pytest.approx(expected[2])


def test_wide_source_can_pan_to_the_actual_solo_subject_instead_of_losing_it_in_center_fill():
    crop=aspect_crop((3840,2160),(1080,1920),center=(.8,.4))
    np.testing.assert_allclose(project_points([.8,.4],crop),[.5,.4])
    assert crop.x>.6
    # A real offscreen limb is not falsely clamped to a visible edge.
    assert project_points([.2,.4],crop)[0]<0


@pytest.mark.parametrize('source',[(2160,3840),(3840,2160)])
def test_legal_edge_clamp_and_zoom_are_used_by_projection(source):
    crop=aspect_crop(source,(1080,1920),center=(1.,0.),zoom=1.5)
    assert crop.x+crop.width==pytest.approx(1.) and crop.y==0.
    corners=np.array([[crop.x,crop.y],[crop.x+crop.width,crop.y+crop.height]])
    np.testing.assert_allclose(project_points(corners,crop),[[0.,0.],[1.,1.]])


@pytest.mark.parametrize('bad',[0.,float('nan'),float('inf'),True])
def test_invalid_zoom_is_not_silently_replaced_by_an_identity_camera(bad):
    with pytest.raises(ValueError):aspect_crop((3840,2160),(1080,1920),zoom=bad)


def test_source_coordinates_and_confidence_arrays_are_never_rewritten():
    raw=np.array([[.8,.4,.9],[.2,.5,.3]])
    original=raw.copy()
    project_points(raw[:,:2],aspect_crop((3840,2160),(1080,1920),center=(.8,.4)))
    np.testing.assert_array_equal(raw,original)


@pytest.mark.parametrize('source',[(640,360),(360,640)])
@pytest.mark.parametrize('zoom',[1.,1.5])
def test_projected_marker_matches_actual_FFmpeg_member_crop_without_squashing(source,zoom):
    from crosscut.render import _video_post_clock_filters
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg is None:pytest.skip('actual FFmpeg unavailable')
    width,height=source;output=(180,320);point=(.8,.4)
    crop=aspect_crop(source,output,center=point,zoom=zoom)
    image=np.zeros((height,width,3),np.uint8)
    x,y=round(point[0]*width),round(point[1]*height)
    image[y-4:y+5,x-4:x+5]=[255,0,0]
    filters=_video_post_clock_filters(crop=crop,pan_crop=None,editorial_transform=None,
        look_adjustment=None,boundary_look_track=None,width=output[0],height=output[1],
        fps=30.,frame_count=1,input_clock_is_canonical=True)
    result=subprocess.run([ffmpeg,'-nostdin','-v','error','-threads','1','-filter_threads','1',
        '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
        '-framerate','30','-i','pipe:0','-vf',','.join(filters),'-frames:v','1',
        '-pix_fmt','rgb24','-f','rawvideo','pipe:1'],input=image.tobytes(),capture_output=True,
        timeout=10,check=True)
    actual=np.frombuffer(result.stdout,np.uint8).reshape(output[1],output[0],3)
    yy,xx=np.nonzero((actual[:,:,0]>120)&(actual[:,:,1]<80)&(actual[:,:,2]<80))
    assert len(xx)>0
    expected=project_points(point,crop)*output
    np.testing.assert_allclose([xx.mean()+.5,yy.mean()+.5],expected,atol=3.)
    assert abs((xx.max()-xx.min())-(yy.max()-yy.min()))<=2
