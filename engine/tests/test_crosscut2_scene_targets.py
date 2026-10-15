from dataclasses import replace
import numpy as np
import pytest
from crosscut2.scene_targets import TargetScene, FEATURE, FORMATION, UNKNOWN, from_boundary_observation
from crosscut2.scene_readout import SceneReadout
from test_crosscut2_scene_readout import catalogue
from test_crosscut2_dense_subject_catalogue import prepared


def scene(heads, *, role=FEATURE, original=None):
    heads=np.asarray(heads,float);t,k=heads.shape[:2]
    original=heads if original is None else np.asarray(original,float)
    known=(heads[...,2]>.019)&(heads[...,3]>=.75)
    return TargetScene('actual-source-hash','pre-camera-target',np.arange(t),np.arange(t),role,
        heads,known,np.zeros((t,k,8,3)),np.zeros((t,k,8),np.uint8),np.zeros((t,k,8),bool),
        np.ones((t,k)),known,original,(original[...,2]>.019)&(original[...,3]>=.75))


def test_current_argmax_counterexample_cannot_move_a_pre_camera_target():
    nodes=np.arange(0,181,18);cat=catalogue(nodes)
    cat.bank['heads'][:,:,0]=[.3,.35,.12,.99]
    cat.bank['heads'][:,:,1]=[.7,.35,.12,.99]
    cat.bank['heads'][:,0,0,2]+=1e-5
    cat.bank['heads'][:,1,1,2]+=1e-5
    cat.bank['heads'][0,0]=cat.bank['heads'][0,1]
    cat.bank['heads'][-1,1]=cat.bank['heads'][-1,0]
    native=dict(frames=np.array([],int),heads=np.zeros((0,2,8,4)))
    old=SceneReadout(cat,native,[],np.tile([1.,.5,.5],(4,180,1)),tau=2.)
    frames=np.flatnonzero(old.valid);camera=np.tile([1.,.5,.5],(len(frames),1))
    # This is the actual old producer, not a hand-coded argmax surrogate.
    assert np.ptp(old.view(frames,camera)[0][:,0])>.7
    target=scene(old.heads[frames,:1],original=old.heads[frames])
    projected=target.project(camera)
    assert np.ptp(projected['centers'][:,0,0])==0.
    assert np.ptp(projected['heights'])<1.1e-5
    assert not projected['target_reselected_after_camera']


def test_crop_cannot_delete_a_competing_face_or_replace_the_evaluated_target():
    original=np.tile([[.3,.35,.12,.99],[.7,.35,.12,.99]],(8,1,1))
    target=scene(original[:,:1],original=original)
    native=target.project(np.tile([1.,.5,.5],(8,1)))
    cropped=target.project(np.tile([2.,.7,.5],(8,1)))
    assert native['head_observed'].all() and cropped['head_observed'].all()
    assert cropped['known_crop_out'].all() and not cropped['visible_fraction'].any()
    # The right face remains visible in the actual source but is NOT borrowed.
    assert (cropped['centers'][:,0,0]<0).all()
    np.testing.assert_array_equal(cropped['original_heads'],original)
    np.testing.assert_array_equal(native['roi_readability'],cropped['roi_readability'])
    assert cropped['roi_measured'].all()


def test_formation_keeps_its_observed_set_and_unknown_does_not_become_a_feature():
    original=np.tile([[.2,.4,.08,.99],[.5,.4,.08,.99],[.8,.4,.08,.99]],(6,1,1))
    formation=scene(original,role=FORMATION)
    result=formation.project(np.tile([2.,.5,.5],(6,1)))
    assert result['role']==FORMATION and result['heights'].shape==(6,3)
    assert result['known_crop_out'][:,[0,2]].all()
    assert not result['known_crop_out'][:,1].any()
    unknown=replace(formation,role=UNKNOWN)
    assert not unknown.project(np.tile([1.,.5,.5],(6,1)))['role_known']


def test_missing_target_cannot_borrow_visible_body_or_roi_and_ambiguity_stays_unknown():
    target=scene(np.tile([[[.5,.35,.12,.99]]],(5,1,1)))
    body=target.body.copy();body[:,:,:,0:2]=.5;body[:,:,:,2]=.9
    kinds=np.ones(body.shape[:-1],np.uint8);ambiguity=np.ones(kinds.shape,bool)
    target=replace(target,head_observed=np.zeros((5,1),bool),roi_measured=np.zeros((5,1),bool),
        body=body,body_evidence=kinds,body_ambiguity=ambiguity)
    view=target.project(np.tile([2.,.5,.5],(5,1)))
    assert not view['head_observed'].any() and not view['known_crop_out'].any()
    assert not view['body_observed'].any() and not view['roi_measured'].any()
    assert not view['anatomical_ownership_certified']


def test_dense_adapter_keeps_exact_subject_clock_and_actual_contemporary_body():
    cat,window,_=prepared(0,bump=.08)
    observed=cat.observation(1,1,np.tile([1.,.5,.5],(7,1)),target_head_index=0)
    result=from_boundary_observation(observed,frame=20,source_binding='0')
    assert result.frames.tolist()==list(range(13,27))
    np.testing.assert_array_equal(result.native_indices,np.arange(93,107))
    np.testing.assert_array_equal(result.body[:,0],window.body[7:21])
    assert result.head_observed.all() and result.body_evidence.max()==1
    assert not result.roi_measured.any()  # old grid ROI cannot rename a new ROI
    assert result.track==window.track
    with pytest.raises(ValueError,match='binding changed'):
        from_boundary_observation(observed,frame=20,source_binding='other-source')
    with pytest.raises(ValueError,match='boundary clock changed'):
        from_boundary_observation(observed,frame=21,source_binding='0')


def test_dense_native_cut_keeps_available_post_samples_without_inventing_pre_track():
    cat,_,_=prepared(0,bump=.08,cut=(100,))
    observed=cat.observation(1,1,np.tile([1.,.5,.5],(7,1)),target_head_index=0)
    result=from_boundary_observation(observed,frame=20,source_binding='0')
    assert not result.head_observed[:7].any() and result.head_observed[7:].all()
    assert not result.body_evidence[:7].any() and result.body_evidence[7:].any()


def test_sparse_adapter_uses_actual_four_native_ordinals_not_output_fps_or_head_proxies():
    cat=catalogue(np.array([0,60,120]))
    cat.bank['pose_native_indices']=np.array([[0,0,0,12],[106,118,120,132],[226,238,238,238]])
    cat.bank['native_indices']=cat.bank['pose_native_indices'][:,[1,2]]
    cat.native=cat.bank['pose_native_indices']
    observed=cat.observation(1,1,np.tile([1.,.5,.5],(7,1)),target_head_index=0)
    result=from_boundary_observation(observed,frame=60,source_binding='native-60fps')
    assert result.frames.tolist()==[53,59,60,66]
    assert result.native_indices.tolist()==[106,118,120,132]
    assert result.head_observed[:,0].tolist()==[False,True,True,False]
    assert result.roi_measured[:,0].tolist()==[False,True,True,False]
    assert not result.original_observed[[0,3]].any()


def test_unpinned_focal_cannot_enter_and_fourth_face_does_not_borrow_first_body():
    cat=catalogue(np.array([0,60,120]));strip=np.tile([1.,.5,.5],(7,1))
    with pytest.raises(ValueError,match='before proposing'):
        from_boundary_observation(cat.observation(1,1,strip),frame=60,source_binding='hash')
    cat.bank['heads'][1,:,3]=[.8,.3,.2,.99]
    result=from_boundary_observation(cat.observation(1,1,strip,target_head_index=3),frame=60,source_binding='hash')
    assert not result.body_evidence.any()
    assert result.head_observed[:,0].tolist()==[False,False,True,False]
    assert result.original_observed[:,3].any()  # original fourth detection remains


def test_shared_inputs_are_frozen_camera_batches_and_virtual_splits_do_not_change_projection():
    original=np.tile([[[.5,.35,.12,.99]]],(12,1,1));target=scene(original)
    original[:]=0
    assert target.heads[0,0,2]==.12
    with pytest.raises(ValueError):target.heads[0,0,2]=.9
    camera=np.tile([1.,.5,.5],(2,12,1));camera[1,:,0]=1.3
    together=target.project(camera)
    for i,j in ((0,5),(5,12)):
        part=replace(target,**{name:getattr(target,name)[i:j] for name in
            ('frames','native_indices','heads','head_observed','body','body_evidence','body_ambiguity',
             'roi_readability','roi_measured','original_heads','original_observed')})
        view=part.project(camera[:,i:j])
        for key in ('centers','heights','head_observed','known_crop_out','roi_readability'):
            np.testing.assert_array_equal(view[key],together[key][:,i:j])


def test_clipped_duplicate_sample_is_not_independent_exposure_or_a_fabricated_track():
    cat=catalogue(np.array([0,60,120]));strip=np.tile([1.,.5,.5],(7,1))
    result=from_boundary_observation(cat.observation(0,1,strip,target_head_index=0),frame=0,source_binding='hash')
    assert result.frames.tolist()==[0,6]
    assert result.head_observed[:,0].tolist()==[True,False]
    np.testing.assert_array_equal(result.native_indices,[0,6])
    with pytest.raises(ValueError,match='camera paths'):
        result.project(np.array(1.))
