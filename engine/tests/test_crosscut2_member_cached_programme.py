import numpy as np
import pytest
from crosscut2.member_cached_programme import follow_crops,projected,pair_cost,solve


def test_fixed_zoom_camera_keeps_real_portrait_aspect_without_size_breathing():
    face=np.tile([.7,.2,.1,.1],(60,1));face[30:,0]=.6
    crop=follow_crops(face,np.ones(60,bool),source_size=(3840,2160),output_size=(1080,1920),zoom=1.2,fps=30.)
    assert crop.shape==(60,4) and (crop>=0).all() and (crop[:,:2]+crop[:,2:]<=1.000001).all()
    np.testing.assert_allclose(crop[:,2]*3840/(crop[:,3]*2160),9/16)
    assert np.max(np.abs(np.diff(crop[:,0])))<.01
    np.testing.assert_array_equal(crop[:,2:],np.tile(crop[0,2:],(60,1)))


def test_portrait_body_and_face_use_the_same_actual_camera_and_preserve_confidence():
    face=np.tile([.7,.2,.1,.1],(30,1));pose=np.tile([[[.75,.4,.8]]*8],(30,1,1))
    crop=follow_crops(face,np.ones(30,bool),source_size=(3840,2160),output_size=(1080,1920),zoom=1.2,fps=30.)
    f,p=projected(face,pose,crop,output_size=(1080,1920))
    assert np.all(f[:,0]>=0) and np.all(f[:,0]+f[:,2]<=1.)
    np.testing.assert_array_equal(p[:,:,2],pose[:,:,2])
    np.testing.assert_allclose(p[:,:,:2],(pose[:,:,:2]-crop[:,None,:2])/crop[:,None,2:])


def test_matching_arms_beat_opposed_arms_and_missing_faces_do_not_get_a_perfect_match():
    face=np.array([[.45,.2,.1,.1]])
    pose=np.array([[[.4,.4,.9],[.6,.4,.9],[.3,.5,.9],[.7,.5,.9],[.2,.6,.9],[.8,.6,.9],[.4,.8,.9],[.6,.8,.9]]])
    wrong=pose.copy();wrong[:,[2,4],1]=[.3,.2]
    cost=lambda p,k=True:pair_cost(face,face,pose,p,[True],[k],output_size=(1080,1920))[0,0]
    assert cost(pose)<cost(wrong) and cost(pose)<cost(pose,False)
    assert np.isfinite(cost(pose,False))


def test_whole_film_dp_does_not_skip_coverage_or_fake_cuts_with_same_source_zoom_aliases():
    nodes=np.arange(0,301,10);loss=np.full((4,300),.4);source=np.array([0,0,1,1])
    holds,value=solve(nodes,loss,source,lambda f:np.full((4,4),.3),fps=30.)
    assert holds[0][0]==0 and holds[-1][1]==300 and np.isfinite(value)
    assert all(a[1]==b[0] and source[a[2]]!=source[b[2]] for a,b in zip(holds,holds[1:]))
    assert all(1.1<=(b-a)/30<=5 for a,b,_ in holds[:-1])


def test_nonfinite_input_never_becomes_a_best_member_film():
    with pytest.raises(ValueError,match='finite complete'):
        solve(np.array([0,30]),np.full((2,30),np.nan),[0,1],lambda f:np.zeros((2,2)),fps=30.)


def test_explicit_unknown_risk_changes_no_measured_pair_or_finite_source_admission():
    box=np.array([[.4,.2,.1,.1]]);pose=np.zeros((1,8,3))
    cost=lambda known,risk:pair_cost(box,box,pose,pose,[known],[known],output_size=(1080,1920),unknown_geometry=risk)
    np.testing.assert_array_equal(cost(True,2.),cost(True,4.))
    assert np.isfinite(cost(False,4.)).all() and cost(False,4.)[0,0]>cost(False,2.)[0,0]


def test_soft_train_pacing_can_keep_a_long_matching_shot_instead_of_forcing_a_bad_cut():
    from crosscut2.visible_shot_duration import Model,fit,exposure_tail_artifact
    model=Model(exposure_tail_artifact(fit([dict(reference_id='editor',split='train',input_sha256='a'*64,
        cut_seconds=[0.,1.,3.,6.,10.])]),fps=30.))
    nodes=np.arange(0,241,15);loss=np.zeros((2,240))
    transition=lambda f:np.full((2,2),100.)
    legacy,_=solve(nodes,loss,[0,1],transition,fps=30.)
    actual,value=solve(nodes,loss,[0,1],transition,fps=30.,minimum_seconds=1/30,
        maximum_seconds=8.,duration_model=model)
    assert len(legacy)>1 and actual in (((0,240,0),),((0,240,1),))
    assert value==pytest.approx(model.loss(8.,censored=True))


def test_soft_pacing_still_switches_at_a_useful_real_source_improvement():
    from crosscut2.visible_shot_duration import Model,fit,exposure_tail_artifact
    model=Model(exposure_tail_artifact(fit([dict(reference_id='editor',split='train',input_sha256='a'*64,
        cut_seconds=[0.,1.,3.,6.,10.])]),fps=30.))
    loss=np.full((2,240),10.);loss[0,:120]=0.;loss[1,120:]=0.
    holds,_=solve(np.arange(0,241,15),loss,[0,1],lambda f:np.full((2,2),.3),
        fps=30.,minimum_seconds=1/30,maximum_seconds=8.,duration_model=model)
    assert holds==((0,120,0),(120,240,1))
