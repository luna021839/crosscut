import copy
import numpy as np
import pytest
from crosscut2.played_partial_pose import features, matrix, source_activity
from crosscut2.physical_display_observations import MODEL_ONLY, PIXEL_SUPPORTED, UNMEASURED, CONTRADICTED


def observation(side, *, kind=MODEL_ONLY):
    base=np.array([[.4,.4,.9],[.6,.4,.9],[.35,.5,.9],[.65,.5,.9],
        [.3,.55,.9],[.7,.55,.9],[.43,.7,.9],[.57,.7,.9]])
    body=np.tile(base,(2,1,1))
    return dict(head=np.array([.5,.3,.25,.99]),body=body,evidence=np.full((2,8),kind,np.uint8),
        ownership_ambiguity=np.zeros((2,8),bool),camera_strip=np.tile([1.,.5,.5],(7,1)),
        physical_window_valid=True,native_indices=np.array([10,16]) if side==0 else np.array([17,23]),side=side)


def compare(a,b):return matrix(features([a],fps=30.),features([b],fps=30.),details=True)


def test_real_pose_changes_K_with_face_fixed_and_direction_reversal_not_hard_veto():
    a,b=observation(0),observation(1)
    good=compare(a,b)
    b['body'][:,4,1]-=.25;b['body'][1,4,0]+=.15
    bad=compare(a,b)
    assert good['face_rate'][0,0]==bad['face_rate'][0,0]==0.
    assert bad['pose_rate'][0,0]>good['pose_rate'][0,0]
    assert bad['allowed'][0,0] and np.isfinite(bad['pose_rate']).all()


def test_missing_contradicted_and_ambiguous_are_neutral_or_tentative_not_perfect():
    a,b=observation(0),observation(1)
    model=compare(a,b)['pose_rate'][0,0]
    assert model==pytest.approx(.25)
    a['evidence'][:]=PIXEL_SUPPORTED;b['evidence'][:]=PIXEL_SUPPORTED
    assert compare(a,b)['pose_rate'][0,0]==0.
    a['ownership_ambiguity'][:]=True
    assert compare(a,b)['pose_rate'][0,0]==pytest.approx(model)
    a['evidence'][:]=CONTRADICTED
    assert compare(a,b)['pose_rate'][0,0]==pytest.approx(.5)
    a['body'][:]=0.;a['evidence'][:]=UNMEASURED
    assert compare(a,b)['pose_rate'][0,0]==pytest.approx(.5)
    assert compare(a,b)['allowed'][0,0]


def test_partial_one_arm_useful_and_missing_channels_not_deleted():
    a,b=observation(0,kind=PIXEL_SUPPORTED),observation(1,kind=PIXEL_SUPPORTED)
    for o in (a,b):
        o['body'][:,[1,3,5,6,7],2]=0.;o['evidence'][:,[1,3,5,6,7]]=UNMEASURED
    r=compare(a,b)
    assert r['pose_rate'][0,0]==pytest.approx(1/3)
    assert r['fixed_semantic_channel_rates'][0,0,0]==0.
    assert (r['fixed_semantic_channel_rates'][0,0,1:]==.5).all()


def test_own_motion_cannot_be_created_by_digital_pan_zoom():
    a=observation(0);before=source_activity(a,fps=30.)
    a['camera_strip'][:,0]=np.linspace(1.,1.5,7);a['camera_strip'][:,1]=np.linspace(.5,.6,7)
    assert before==source_activity(a,fps=30.)==0.
    a['body'][1,4,1]-=.2
    assert source_activity(a,fps=30.)>0.


def test_native_cut_or_duplicate_clock_makes_velocity_unknown_not_phase_missing():
    a,b=observation(0,kind=PIXEL_SUPPORTED),observation(1,kind=PIXEL_SUPPORTED)
    a['physical_window_valid']=False
    r=compare(a,b)
    assert (r['phase_evidence_kind']==PIXEL_SUPPORTED).all()
    assert (r['motion_evidence_kind']==UNMEASURED).all()
    assert r['pose_rate'][0,0]==pytest.approx(1/3)
    assert source_activity(a,fps=30.)==0.


def test_features_snapshot_is_bound_to_exposed_observations_only():
    a,b=observation(0),observation(1);left=features([a],fps=30.)
    before=matrix(left,features([b],fps=30.))[0]
    a['unplayed_other_side']=np.full((2,8,3),99.)
    after=matrix(features([a],fps=30.),features([b],fps=30.))[0]
    np.testing.assert_array_equal(before,after)
