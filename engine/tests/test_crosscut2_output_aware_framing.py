import numpy as np
import pytest
from crosscut2.display_source_catalogue import SourceCatalogue,upper_body_crop_loss,added_pixel_upscale_effort


def test_hip_crop_is_not_an_arm_crop_and_observed_hands_remain_protected():
    valid=np.ones((2,8),bool);points=np.full((2,8,2),.5)
    points[0,6:,1]=1.1;points[1,4,0]=1.1
    np.testing.assert_allclose(upper_body_crop_loss(valid,points),[0.,1/6])
    valid[:]=False
    np.testing.assert_array_equal(upper_body_crop_loss(valid,points),[0.,0.])


def test_lossless_4k_crop_is_not_penalized_as_1080p_upscaling_or_quality_certification():
    # Same displayed216px face: original216px /108px /54px source faces.
    actual=added_pixel_upscale_effort(np.array([.2]*4),np.array([216.,108.,54.,216.]),
        np.array([True,True,True,False]),np.array([2.]*4),output_height=1080)
    np.testing.assert_allclose(actual,[0.,1.,1.,1.])
    # The original54→108px view already needed interpolation. Do not bill it
    # twice as a new digital crop; native readability owns that source detail.
    assert added_pixel_upscale_effort(.1,54.,True,1.,output_height=1080)==0.


def catalogue(output_height,native_height=2160,measured=True):
    from test_crosscut2_pair_camera_options import counterexample
    _,_,_,_,arcs,_=counterexample(False);old=arcs.cats[0];bank=dict(old.bank)
    metrics=np.zeros(bank['heads'].shape[:3]+(4,));metrics[...,:3]=[.1,.6,0.]
    metrics[...,3]=bank['heads'][...,2]*native_height
    bank.update(native_readability=metrics,native_readability_valid=(bank['heads'][...,2]>.019)&measured)
    body=old.body.copy();body[:,:,:,:,0]=np.array([.4,.6,.4,.6,.4,.6,.4,.6])
    body[:,:,:,:,1]=[.4,.4,.5,.5,.55,.55,.8,.8];body[...,2]=.9
    return SourceCatalogue(arcs.nodes,bank,body,[],old.style,[],fps=30.,editorial_content=True,output_height=output_height)


@pytest.mark.parametrize('native_height,measured,saved',[(2160,True,.35),(1080,True,.25),(2160,False,.25)])
def test_actual_interval_content_uses_output_pixels_without_touching_native_pose_or_original_crop(native_height,measured,saved):
    legacy=catalogue(None,native_height,measured);actual=catalogue(1080,native_height,measured)
    frames=np.array([30]);crop=np.array([[2.,.5,.5]]);native=np.array([[1.,.5,.5]])
    before=actual.body.copy()
    np.testing.assert_allclose(legacy.rates(frames,crop)-actual.rates(frames,crop),saved,atol=1e-12)
    np.testing.assert_allclose(legacy.rates(frames,native),actual.rates(frames,native),atol=1e-12)
    np.testing.assert_array_equal(before,actual.body)


@pytest.mark.parametrize('height',[0,True,-1,8193,1080.5])
def test_output_height_is_actual_data_not_an_unchecked_quality_setting(height):
    from crosscut2.played_pose_port_programme import PlayedPosePortPrior
    with pytest.raises(ValueError,match='output height'):
        PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,output_height=height)


def test_cli_uses_the_requested_render_height_before_loading_media(monkeypatch,tmp_path):
    from crosscut2 import played_pose_port_programme as programme,stage_mix
    class PriorBuilt(Exception):pass
    def prior(**kwargs):
        assert kwargs['output_height']==1080
        raise PriorBuilt()
    monkeypatch.setattr(programme,'PlayedPosePortPrior',prior)
    monkeypatch.setattr('sys.argv',['stage_mix','--source-plan','unused','--analysis','unused',
        '--cache',str(tmp_path/'cache'),'--output',str(tmp_path/'not-rendered.mp4'),'--models','unused',
        '--body-pose-model','unused','--fixed-actor-camera-holds','--native-source-readability',
        '--native-camera-ports','--joint-motion-event-seconds','--native-part-pixel-support',
        '--played-pose-port-camera-prior','--editorial-opportunity-exposure','--played-interval-utility',
        '--preview-height','1080'])
    with pytest.raises(PriorBuilt):stage_mix.main()
