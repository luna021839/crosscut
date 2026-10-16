import json
import cv2
import numpy as np
import pytest
from crosscut2.observed_reframing import estimate,displayed_energy,measure


def texture(seed=3):
    rng=np.random.default_rng(seed)
    gray=rng.integers(0,256,(180,320),np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(gray,(3,3),.4),cv2.COLOR_GRAY2BGR)


def test_translated_textured_image_is_observed_without_claiming_a_camera_cause():
    first=texture();last=cv2.warpAffine(first,np.float32([[1,0,7],[0,1,-3]]),(320,180))
    obs=estimate(first,last)
    assert obs['state']=='OBSERVED_IMAGE_REFRAMING'
    np.testing.assert_allclose(np.asarray(obs['affine'])[:2,2],[7/320.,-3/180.],atol=.002)
    assert not obs['camera_cause_certified'] and not obs['actor_motion_or_identity_certified']
    assert displayed_energy(obs,[1,.5,.5],[1,.5,.5],seconds=.2)>0
    assert obs['occupied_cells']>=4 and obs['neural_calls']==0


def test_stationary_background_is_not_called_a_pan_because_a_small_foreground_patch_moves():
    first=texture();last=first.copy()
    first[50:100,110:150]=texture(5)[:50,:40]
    last[50:100,135:175]=texture(5)[:50,:40]
    obs=estimate(first,last)
    assert obs['state']=='OBSERVED_IMAGE_REFRAMING'
    assert displayed_energy(obs,[1,.5,.5],[1,.5,.5],seconds=.2)<1e-5


def test_untextured_images_are_unknown_not_a_perfect_stationary_hold():
    black=np.zeros((180,320,3),np.uint8);obs=estimate(black,black)
    assert obs['state']=='UNOBSERVED'
    assert displayed_energy(obs,[1,.5,.5],[1,.5,.5],seconds=.2) is None


def test_native_discontinuity_never_becomes_an_inferred_camera_velocity():
    first=texture();last=texture(8)
    obs=estimate(first,last,native_cut_crossed=True)
    assert obs['state']=='NATIVE_DISCONTINUITY'
    assert obs['consistent_tracks']==0
    assert displayed_energy(obs,[1,.5,.5],[1,.5,.5],seconds=.2) is None


def test_native_and_digital_motion_are_composed_not_blindly_added():
    p=np.array([(x,y) for x in (.2,.4,.6,.8) for y in (.2,.4,.6,.8)])
    obs=dict(state='OBSERVED_IMAGE_REFRAMING',affine=[[1,0,.02],[0,1,0],[0,0,1]],
        inlier_samples=np.column_stack((p,p+[.02,0])).tolist())
    native=displayed_energy(obs,[1.2,.5,.5],[1.2,.5,.5],seconds=.2)
    compensated=displayed_energy(obs,[1.2,.5,.5],[1.2,.52,.5],seconds=.2)
    assert native>0 and compensated==pytest.approx(0,abs=1e-15)
    bad=dict(obs,inlier_samples=[[float('nan'),.5,.5,.5]])
    with pytest.raises(ValueError,match='normalized correspondence'):
        displayed_energy(bad,[1,.5,.5],[1,.5,.5],seconds=.2)


def test_unobserved_crop_cannot_borrow_global_background_stillness():
    points=np.array([(x,y) for x in (.02,.06,.94,.98) for y in (.02,.06,.94,.98)])
    obs=dict(state='OBSERVED_IMAGE_REFRAMING',affine=np.eye(3).tolist(),
        inlier_samples=np.column_stack((points,points)).tolist())
    assert displayed_energy(obs,[1,.5,.5],[1,.5,.5],seconds=.2)==0.
    assert displayed_energy(obs,[2,.5,.5],[2,.5,.5],seconds=.2) is None


def test_failed_correspondence_is_not_the_same_evidence_as_absent_texture():
    first=texture();last=texture(9)
    failed=estimate(first,last);empty=estimate(first*0,last*0)
    assert failed['feature_count']>=12
    assert failed['correspondence_loss'] is not None
    assert empty['correspondence_loss'] is None


def test_complete_warm_return_precedes_any_source_probe_or_raster_decoder(tmp_path,monkeypatch):
    from crosscut2 import sparse_native_frames as sparse
    from crosscut2.grid_perception import file_identity
    from crosscut2.subject_window_bank import ProbeBudget
    source=tmp_path/'source.mp4';source.write_bytes(b'unit-test-not-a-real-movie')
    original=texture();other=cv2.warpAffine(original,np.float32([[1,0,4],[0,1,0]]),(320,180))
    first=cv2.resize(original,(960,540));last=cv2.resize(other,(960,540))
    decoded=[]
    monkeypatch.setattr(sparse,'probe',lambda *a,**kw:object())
    def frames(path,natives,**kw):
        decoded.extend(natives);return iter([(n,first if i==0 else last) for i,n in enumerate(natives)])
    monkeypatch.setattr(sparse,'frames',frames)
    def budget():return ProbeBudget(face_limit=0,recognition_limit=0,pose_limit=0,raster_limit=4,seconds=10.,check=lambda:None)
    identity=file_identity(source);kw=dict(source_sha256='0'*64,source_identity=identity,native_cuts=[20],cache=tmp_path/'cache')
    cold,receipt=measure(source,[(3,9),(18,24)],budget=budget(),**kw)
    assert decoded==[3,9] and receipt['returned_source_rasters']==2
    assert cold[1]['observation']['state']=='NATIVE_DISCONTINUITY'
    def forbidden(*a,**kw):raise AssertionError('warm bank must not initialize a source decoder')
    monkeypatch.setattr(sparse,'probe',forbidden);monkeypatch.setattr(sparse,'frames',forbidden)
    warm_budget=budget();warm,receipt=measure(source,[(3,9),(18,24)],budget=warm_budget,**kw)
    assert receipt['reused_pairs']==2 and receipt['returned_source_rasters']==0
    assert warm_budget.used['raster']==0
    assert all(v['observation']==c['observation'] for v,c in zip(warm,cold))


@pytest.mark.parametrize('pairs',[[[2.2,8.]],[[3,3]],[[4,2]],[[1,2],[1,2]]])
def test_bad_native_domains_cannot_be_silently_rounded_or_duplicated(tmp_path,pairs):
    with pytest.raises(ValueError):
        measure(tmp_path/'missing',pairs,source_sha256='0'*64,source_identity=(),native_cuts=[],cache=tmp_path/'cache',budget=None)
