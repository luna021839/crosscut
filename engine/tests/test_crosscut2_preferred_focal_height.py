import numpy as np
import pytest
from crosscut2.display_source_catalogue import SourceCatalogue
from crosscut2.boundary_port_domain import proposal_display_quality
from crosscut2.editorial_roles import focal_height_preference
from crosscut2.played_pose_port_programme import PlayedPosePortPrior
from test_crosscut2_output_aware_framing import catalogue


def make(preferred):
    cat=catalogue(1080)
    return SourceCatalogue(cat.nodes,cat.bank,cat.body,[],cat.style,[],fps=30.,
        editorial_content=True,output_height=1080,preferred_focal_height=preferred)


def test_default_content_and_proposal_scores_remain_exactly_unchanged():
    old=catalogue(1080);explicit=make(.12);frames=np.array([29,30,31])
    for camera in ([1.,.5,.5],[1.6,.5,.5],[2.,.6,.5]):
        path=np.tile(camera,(len(frames),1))
        np.testing.assert_array_equal(old.rates(frames,path),explicit.rates(frames,path))


def test_closeup_preference_is_not_fabricated_native_detail_or_a_wide_shot_ban():
    old=make(.12);close=make(.22)
    np.testing.assert_array_equal(old.roi_readability,close.roi_readability)
    np.testing.assert_array_equal(old.roi_readability_valid,close.roi_readability_valid)
    np.testing.assert_array_equal(old.body,close.body)
    for zoom in (1.,1.5,2.):
        strip=np.tile([zoom,.5,.5],(7,1))
        a=old.observation(1,1,strip,target_head_index=0)
        b=close.observation(1,1,strip,target_head_index=0)
        assert b['displayed_native_quality_proxy']<=a['displayed_native_quality_proxy']
        assert b['target_native_ROI_readability']==a['target_native_ROI_readability']
        assert b['native_face_pixels']==a['native_face_pixels']
        assert b['displayed_native_quality_measured']==a['displayed_native_quality_measured']
        assert np.isfinite(close.rates(np.array([30]),np.array([[zoom,.5,.5]]))).all()


def test_native_and_cropped_views_use_the_same_preference_before_candidate_pruning():
    cat=make(.22);native=cat.observation(1,1,np.tile([1.,.5,.5],(7,1)),target_head_index=0)
    assert native['editorial_preferred_focal_height']==.22
    for camera in ([1.,.5,.5],[1.6,.5,.5],[2.,.75,.5]):
        actual=cat.observation(1,1,np.tile(camera,(7,1)),target_head_index=0)
        assert proposal_display_quality(native,camera)==pytest.approx(actual['displayed_native_quality_proxy'],abs=1e-12)


def test_preference_prior_is_explicit_and_does_not_change_source_or_cut_admission():
    with pytest.raises(ValueError,match='output-bound played intervals'):
        PlayedPosePortPrior(preferred_focal_height=.22)
    prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,
        output_height=1080,preferred_focal_height=.22)
    assert prior.artifact['preferred_displayed_focal_height']==.22
    assert not prior.artifact['focal_size_preference_changes_source_or_cut_admission']
    assert not prior.artifact['focal_size_preference_fitted']


@pytest.mark.parametrize('value',[True,.1,.31,float('nan'),float('inf')])
def test_size_preference_is_bounded_and_finite(value):
    with pytest.raises(ValueError,match='Preferred focal height'):focal_height_preference(value)


def test_cli_does_not_silently_ignore_a_closeup_request(monkeypatch,tmp_path):
    from crosscut2.stage_mix import main
    argv=['stage_mix','--preferred-focal-height','.22']
    for key in ('source-plan','analysis','cache','output','models'):argv.extend(['--'+key,str(tmp_path/key)])
    monkeypatch.setattr('sys.argv',argv)
    with pytest.raises(ValueError,match='output-bound played intervals'):main()


def test_cli_binds_the_requested_preference_to_the_actual_programme_prior(monkeypatch,tmp_path):
    from crosscut2 import played_pose_port_programme as programme,stage_mix
    class PriorBuilt(Exception):pass
    def prior(**kwargs):
        assert kwargs['preferred_focal_height']==.22
        assert kwargs['output_height']==1080
        raise PriorBuilt()
    monkeypatch.setattr(programme,'PlayedPosePortPrior',prior)
    monkeypatch.setattr('sys.argv',['stage_mix','--source-plan','unused','--analysis','unused',
        '--cache',str(tmp_path/'cache'),'--output',str(tmp_path/'not-rendered.mp4'),'--models','unused',
        '--body-pose-model','unused','--fixed-actor-camera-holds','--native-source-readability',
        '--native-camera-ports','--joint-motion-event-seconds','--native-part-pixel-support',
        '--played-pose-port-camera-prior','--editorial-opportunity-exposure','--played-interval-utility',
        '--preferred-focal-height','.22','--preview-height','1080'])
    with pytest.raises(PriorBuilt):stage_mix.main()
