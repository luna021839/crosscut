import copy
from dataclasses import replace
import hashlib
import numpy as np
import pytest
from crosscut2.subject_window_bank import SubjectWindow
from crosscut2.subject_action_match import clock,compare,rank,incoming_camera


TICKS=np.arange(-12,13)
BASE=np.array([[.4,.4,.9],[.6,.4,.9],[.35,.5,.9],[.65,.5,.9],
               [.3,.55,.9],[.7,.55,.9],[.43,.7,.9],[.57,.7,.9]])


def window(source=0, *, missing=False, opposite=False, identity=1., competitors=None, cut=()):
    ordinals=np.arange(86,115);n=len(ordinals)
    heads=np.tile([.5,.3,.2,.99],(n,1));body=np.tile(BASE,(n,1,1))
    motion=.08*np.sin(2*np.pi*(ordinals-100)/6)
    if opposite:motion=-motion
    body[:,4,1]+=motion;body[:,5,1]+=motion
    support=np.ones((n,8),bool)
    if missing:support[:]=False;body[:]=0.
    vectors=np.zeros((n,128),np.float32);vectors[:,0]=identity
    others=tuple(np.zeros((0,4)) if competitors is None else np.array(competitors,float) for _ in ordinals)
    return SubjectWindow(source,'source'+str(source),'subject'+str(source),ordinals,heads,vectors,body,
        support,np.ones(n,bool),others,cut)


def hypothesis(w,phase=0,ticks=TICKS):
    return clock(source=w.source,binding=w.binding,relative_frames=ticks,source_start=100/30,
        playback_rate=1.,output_fps=30.,native_fps=30.,native_count=500,phase_frames=phase)


def matched(a,b,phase=0,**kwargs):
    return compare(a,b,hypothesis(a),hypothesis(b,phase),
        left_cameras=np.tile([1.,.5,.5],(len(TICKS),1)),incoming_cap=2.,**kwargs)


def test_opposite_intermediate_motion_not_aliased_by_equal_endpoints():
    a,b,bad=window(0),window(1),window(1,opposite=True)
    assert np.array_equal(b.body[[14,20]],bad.body[[14,20]])
    good=matched(a,b);wrong=matched(a,bad)
    assert good['status']=='SUPPORTED_CANDIDATE' and good['motion_residual']<1e-12
    assert wrong['motion_residual']>good['motion_residual']+.1
    assert wrong['rank_score']>good['rank_score']


def test_matching_synchronous_dance_reversal_does_not_need_constant_cross_seam_velocity():
    a,b=window(0),window(1)
    # Both source trajectories reverse together at the same music time.
    good=matched(a,b)
    assert good['phase_residual']<1e-12 and good['motion_residual']<1e-12


def test_real_producer_sparse_pose_samples_are_compared_at_their_actual_common_times():
    a,b=window(0),window(1)
    def sparse(w):
        selected=np.isin(w.ordinals-100,[-12,-8,-4,-1,0,4,8,10,12])
        support=w.support.copy();support[~selected]=False
        body=w.body.copy();body[~selected]=0.
        return replace(w,body=body,support=support)
    value=matched(sparse(a),sparse(b))
    assert value['status']=='SUPPORTED_CANDIDATE'
    assert value['motion_residual']<1e-12
    assert value['comparable_motion_pairs']>=6


def test_appearance_is_actual_retrieval_evidence_not_an_unused_report_or_identity_certificate():
    a,b=window(0),window(1);wrong=window(1,identity=-1.)
    good,bad=matched(a,b),matched(a,wrong)
    assert good['rank_score']<bad['rank_score']
    assert not good['matched_identity_certified'] and not good['anatomical_ownership_certified']


def test_unobserved_action_is_separate_not_a_neutral_cost_winner_or_good_hold():
    good=matched(window(0),window(1));unknown=matched(window(0,missing=True),window(1,missing=True))
    assert unknown['status']=='NO_SUPPORTED_MATCH' and unknown['rank_score'] is None
    result=rank([unknown,good]);assert result['supported']==[good]
    assert rank([unknown])['status']=='NO_SUPPORTED_MATCH'
    assert rank([unknown])['H_hold_is_not_a_success']


def test_pinned_subject_remains_target_when_camera_makes_another_face_dominant():
    a=window(0);b=window(1,competitors=[[.5,.3,.5,.99]])
    value=matched(a,b)
    assert value['track_B']==b.track and not value['target_reselected_after_camera']
    assert value['target_attention_ratio']<.6
    assert value['status']=='TARGET_OCCLUDED_OR_NOT_ATTENDED'


def test_exact_independent_pro_focal_reassignment_counterexample_through_real_modules():
    from crosscut2.display_source_catalogue import SourceCatalogue
    from crosscut2.boundary_port_domain import make
    faces=[np.array([[.5575650354576569,.5491560207020963,.2175005940828223,.95],
        [.4456923394988387,.5100103527186727,.08817220382883184,.95],
        [.4554007332066811,.44569336146181326,.19103173154659386,.95]]),
        np.array([[.45415629261341817,.3779578475785671,.20620328644050506,.95],
        [.4461833298814043,.5300756462555543,.20620881294213955,.95],
        [.47890973364706096,.4123221992536122,.09952321152433102,.95]])]
    nodes=np.array([0,20,40]);catalogues=[];paths=[]
    for rows in faces:
        heads=np.zeros((3,2,8,4));heads[:,:,:3]=rows
        bank=dict(heads=heads,pose_native_indices=np.array([[0,1,2,3],[13,19,20,26],[30,36,37,39]]),
            zoom_cap=2.,vectors=np.zeros((3,2,3,128)))
        catalogues.append(SourceCatalogue(nodes,bank,None,[],dict(presence=np.zeros((3,2,1))),[],fps=30.))
        paths.append([(np.array([0,39]),np.array([[1.,.5,.5],[1.,.5,.5]])) for _ in range(4)])
    # Reproduce the old unbound evaluation first: camera k1 changes B's
    # dominant face from the proposed face0 to face1.
    ca=np.array([1.4480584975413202,.5535177790481345,.5743228671530826])
    cb=np.array([1.527393616846104,.45031925678264184,.40181749021274227])
    assert catalogues[0].observation(1,0,np.tile(ca,(7,1)))['head_index']==0
    assert catalogues[1].observation(1,1,np.tile(cb,(7,1)))['head_index']==1
    pinned=catalogues[1].observation(1,1,np.tile(cb,(7,1)),target_head_index=0)
    assert pinned['head_index']==0 and not pinned['target_is_displayed_focal']
    entries,exits,_,domain=make(nodes,catalogues,paths,fps=30.)
    # A different actor can now be retrieved before camera construction.
    # Regardless of its rank, the old face0->camera->face1 substitution may
    # not survive with that frozen target/camera combination.
    assert not any(p.source==1 and p.observation['head_index']==0
        and np.allclose(p.strip,np.tile(cb,(7,1))) for p in entries[1])
    assert domain['physical_actor_pairs_compared_before_camera']==18
    assert all(p.observation['target_is_displayed_focal']
        for side in (entries[1],exits[1]) for p in side
        if any(i[0]=='paired' for i in p.proposal_ids))
    assert all(p.observation['head_index']==identity[4 if side==0 else 5]
        for side,items in enumerate((exits[1],entries[1])) for p in items
        for identity in p.proposal_ids if identity[0]=='paired')
    # The replacement retrieves/pins face0 FIRST. Camera evaluation cannot
    # turn its cached body into face1's body even though old admission did.
    a,b=window(0),window(1)
    a=replace(a,heads=np.tile(faces[0][0],(len(a.ordinals),1)),
        competitors=tuple(faces[0][1:] for _ in a.ordinals))
    b=replace(b,heads=np.tile(faces[1][0],(len(b.ordinals),1)),
        competitors=tuple(faces[1][1:] for _ in b.ordinals))
    result=compare(a,b,hypothesis(a),hypothesis(b),left_cameras=np.tile(ca,(25,1)),incoming_cap=2.)
    assert result['track_B']==b.track and not result['target_reselected_after_camera']
    assert result['face_residual']<1e-10


def test_outgoing_camera_is_fixed_and_incoming_transform_is_legal_no_padding():
    a=np.array([.55,.4,.2,.99]);b=np.array([.25,.35,.15,.99]);ca=np.array([1.2,.55,.5]);before=ca.copy()
    cb=incoming_camera(a,b,ca,cap=1.35)
    np.testing.assert_array_equal(ca,before)
    assert 1<=cb[0]<=1.35 and np.all(cb[1:]>=.5/cb[0]) and np.all(cb[1:]<=1.-.5/cb[0])
    assert np.linalg.norm(cb[1:]-.5)<=.25+1e-12


def test_new_joint_proposals_can_match_scales_without_pretending_old_outgoing_camera_was_unchanged():
    a,b=window(0),window(1)
    heads=b.heads.copy();heads[:,2]=.30;b=replace(b,heads=heads)
    fixed=matched(a,b)
    joint=matched(a,b,outgoing_locked=False,outgoing_cap=2.)
    assert fixed['face_residual']>joint['face_residual']
    assert joint['face_residual']<1e-10 and joint['new_joint_camera_before_selection']
    assert not joint['outgoing_camera_locked'] and fixed['outgoing_camera_locked']


@pytest.mark.parametrize('phase',[-2,-1,1,2])
def test_nonzero_phase_has_distinct_actual_ordinals_not_posthoc_retiming_or_automatic_approval(phase):
    a,b=window(0),window(1);zero=hypothesis(b);shift=hypothesis(b,phase)
    assert np.array_equal(shift.native_indices,zero.native_indices+phase)
    assert shift.schedule_sha256!=zero.schedule_sha256
    value=matched(a,b,phase=phase,following_four_seconds_verified=True)
    assert value['diagnostic_phase_only'] and not value['preview_eligible']


def test_lower_unverified_phase_score_cannot_displace_zero_phase_as_playable_candidate():
    a,b=window(0),window(1);zero=matched(a,b);shift=matched(a,b,phase=1)
    shift['rank_score']=-100.  # Even an artificially perfect numeric ranking
    value=rank([shift,zero])
    assert value['supported']==[zero] and value['diagnostic_phase_candidates']==[shift]
    assert rank([shift])['status']=='NO_SUPPORTED_MATCH'
    assert rank([shift],include_diagnostic_phases=True)['supported']==[shift]


def test_low_engineering_residuals_and_followup_flag_never_mean_editorial_quality_approval():
    value=matched(window(0),window(1),following_four_seconds_verified=True)
    assert value['rank_score']<1e-10
    assert value['diagnostic_render_eligible'] and value['requires_actual_visual_review']
    assert not value['preview_eligible'] and not value['professional_quality_approved']


def test_cannot_borrow_wrong_source_unobserved_native_frame_or_change_outgoing_clock():
    a,b=window(0),window(1)
    with pytest.raises(ValueError,match='binding'):
        compare(a,b,hypothesis(b),hypothesis(b),left_cameras=np.tile([1.,.5,.5],(25,1)),incoming_cap=2.)
    out=clock(source=1,binding=b.binding,relative_frames=TICKS,source_start=130/30,
        playback_rate=1.,output_fps=30.,native_fps=30.,native_count=500)
    with pytest.raises(ValueError,match='unobserved'):
        compare(a,b,hypothesis(a),out,left_cameras=np.tile([1.,.5,.5],(25,1)),incoming_cap=2.)
    with pytest.raises(ValueError,match='fixed outgoing'):
        compare(a,b,hypothesis(a,1),hypothesis(b),left_cameras=np.tile([1.,.5,.5],(25,1)),incoming_cap=2.)


def test_followup_native_cut_and_missing_four_seconds_never_claim_preview_approval():
    good=matched(window(0),window(1))
    assert not good['preview_eligible'] and not good['following_four_seconds_verified']
    cut=matched(window(0),window(1,cut=(100,)))
    assert cut['status']=='SUPPORTED_CANDIDATE'
    assert not any(cut['common_source_shot_sample_mask'][:12])
    assert all(cut['common_source_shot_sample_mask'][12:])
    assert not cut['professional_quality_approved'] and not cut['normal_app_activation_approved']


def test_window_arrays_are_immutable_and_comparison_does_not_mutate_original_observations():
    a,b=window(0),window(1)
    before=hashlib.sha256(a.body.tobytes()+b.body.tobytes()).hexdigest()
    matched(a,b)
    assert hashlib.sha256(a.body.tobytes()+b.body.tobytes()).hexdigest()==before
    with pytest.raises(ValueError):b.body[0,0,0]=1.
