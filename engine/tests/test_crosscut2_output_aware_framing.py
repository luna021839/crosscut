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


def test_prepruning_candidate_uses_same_actual_pixel_effort_as_realized_interval():
    from crosscut2.boundary_port_domain import proposal_zoom_effort
    observation=dict(head=np.array([.5,.3,.1,.99]),actual_output_pixel_height=1080,
        native_face_pixels=216.,native_face_pixels_measured=True)
    assert proposal_zoom_effort(observation,2.)==0.
    observation['native_face_pixels']=108.
    assert proposal_zoom_effort(observation,2.)==pytest.approx(np.log(2.))
    observation['native_face_pixels']=54.
    # Already low-resolution native framing must not be charged twice.
    assert proposal_zoom_effort(observation,2.)==pytest.approx(np.log(2.))
    observation['native_face_pixels_measured']=False
    assert proposal_zoom_effort(observation,2.)==pytest.approx(np.log(2.))
    observation.pop('actual_output_pixel_height')
    assert proposal_zoom_effort(observation,2.)==np.log(2.)


def test_scalar_proposal_effort_matches_interval_sampling_without_hot_loop_arrays():
    from crosscut2.boundary_port_domain import proposal_zoom_effort
    for height in (720,1080,2160):
        for native in (0.,54.,108.,216.,432.):
            for zoom in (1.,1.15,1.5,2.):
                obs=dict(head=np.array([.5,.3,.1,.99]),actual_output_pixel_height=height,
                    native_face_pixels=native,native_face_pixels_measured=True)
                exact=np.log(2.)*added_pixel_upscale_effort(.1*zoom,native,True,zoom,output_height=height)
                assert proposal_zoom_effort(obs,zoom)==pytest.approx(exact,abs=1e-12)


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


def test_output_and_native_face_sampling_follow_the_actual_pinned_subject():
    cat=catalogue(1080)
    for side in (0,1):
        obs=cat.observation(1,side,np.tile([1.,.5,.5],(7,1)),target_head_index=0)
        assert obs['actual_output_pixel_height']==1080
        assert obs['native_face_pixels']==cat.native_face_pixels[1,side,0]
        assert obs['native_face_pixels_measured']


def test_candidate_pruner_consumes_sampling_before_the_global_port_limit(monkeypatch):
    from crosscut2 import boundary_port_domain as ports
    from crosscut2.shot_camera_paths import make_paths
    cat=catalogue(1080);calls=[];original=ports.proposal_zoom_effort
    def measured(obs,zoom):
        calls.append((obs['actual_output_pixel_height'],obs['native_face_pixels'],zoom))
        return original(obs,zoom)
    monkeypatch.setattr(ports,'proposal_zoom_effort',measured)
    banks=[cat.bank,cat.bank];paths,_=make_paths(cat.nodes,banks,[[],[]])
    entries,exits,_,receipt=ports.make(cat.nodes,[cat,cat],paths,fps=30.)
    assert calls and all(height==1080 for height,_,_ in calls)
    assert any(zoom>1. and pixels>0 for _,pixels,zoom in calls)
    assert receipt['proposal_zoom_cost_uses_native_output_pixels']
    assert receipt['unknown_native_sampling_keeps_legacy_proposal_zoom_cost']
    assert max(map(len,entries))<=receipt['global_port_limit_per_side']
    assert max(map(len,exits))<=receipt['global_port_limit_per_side']


def test_known_native_sampling_does_not_require_a_blur_measurement_at_every_fine_node():
    from crosscut2.display_source_catalogue import native_sampling_pixels
    from crosscut2.scene_readout import roi_readability
    cat=catalogue(1080);bank=dict(cat.bank)
    bank['native_readability']=bank['native_readability'].copy()
    bank['native_readability_valid']=bank['native_readability_valid'].copy()
    bank['native_readability'][1]=0.;bank['native_readability_valid'][1]=False
    pixels,known=native_sampling_pixels(bank)
    assert known[1,1,0]
    assert pixels[1,1,0]==pytest.approx(bank['heads'][1,1,0,2]*2160)
    readable,valid=roi_readability(bank)
    assert not valid[1].any() and (readable[1]==.5).all()
    actual=SourceCatalogue(cat.nodes,bank,cat.body,[],cat.style,[],fps=30.,
        editorial_content=True,output_height=1080)
    obs=actual.observation(1,1,np.tile([1.,.5,.5],(7,1)),target_head_index=0)
    assert obs['native_face_pixels_measured'] and not obs['displayed_native_quality_measured']
    from crosscut2.boundary_port_domain import proposal_zoom_effort
    assert proposal_zoom_effort(obs,2.)==0.


@pytest.mark.parametrize('conflict',[True,False])
def test_missing_or_contradictory_source_shape_never_certifies_fine_pixel_sampling(conflict):
    from crosscut2.display_source_catalogue import native_sampling_pixels
    cat=catalogue(1080);bank=dict(cat.bank)
    bank['native_readability']=bank['native_readability'].copy()
    bank['native_readability_valid']=bank['native_readability_valid'].copy()
    bank['native_readability'][1]=0.;bank['native_readability_valid'][1]=False
    if conflict:bank['native_readability'][0,...,3]*=.5
    else:bank['native_readability_valid'][:]=False
    _,known=native_sampling_pixels(bank)
    assert not known[1].any()


def test_source_shape_cannot_create_unobserved_faces_or_change_cached_arrays():
    from crosscut2.display_source_catalogue import native_sampling_pixels
    cat=catalogue(1080);bank=dict(cat.bank)
    bank['face_observation_valid']=np.ones(bank['heads'].shape[:2],bool)
    bank['face_observation_valid'][1]=False
    old_pixels=bank['native_readability'].copy();old_heads=bank['heads'].copy()
    _,known=native_sampling_pixels(bank)
    assert not known[1].any()
    np.testing.assert_array_equal(bank['heads'],old_heads)
    np.testing.assert_array_equal(bank['native_readability'],old_pixels)


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
