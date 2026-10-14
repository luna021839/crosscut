import numpy as np
from crosscut2.preference_features import extract,FEATURE_NAMES
from crosscut2.event_ports import Ports
from crosscut2.event_states import build
from crosscut2.fixed_holds import State
from test_crosscut2_event_holds import inputs,provisional_parts,prepared


def setup(*,opposed=False):
    nodes,banks,poses,styles=inputs(opposed_source=1 if opposed else None)
    states,data,_=build(nodes,banks,poses,styles,fps=10.,source_events=[np.array([],int)]*2,
        body_states=[np.ones((4,4,3),int)]*2,part_observer=provisional_parts)
    p=Ports(nodes,states,data,fps=10.,music=[0.,1.,1.,0.])
    return p,states.index(State(0,0,'native')),states.index(State(1,0,'native'))


def test_unknown_evidence_does_not_scale_down_raw_motion_or_create_pixel_credit():
    p,a,b=setup(opposed=True);feature=extract(p,1)[a,b]
    assert feature[FEATURE_NAMES.index('motion')]>p.selected(1,a,b)['worst_motion_loss']
    assert feature[FEATURE_NAMES.index('pixel_agreed_active_novelty')]==0
    assert feature[FEATURE_NAMES.index('motion_unmeasured')]==1
    assert np.isfinite(extract(p,1)).all()


def test_real_hip_phase_has_its_own_channel_not_face_or_arm_error():
    p,a,b=setup();before=extract(p,1)[a,b].copy()
    p.projected[b,1,:,6:8,0]+=.15
    after=extract(p,1)[a,b]
    assert after[FEATURE_NAMES.index('torso_phase')]>before[FEATURE_NAMES.index('torso_phase')]
    assert np.array_equal(after[:3],before[:3])


def test_stationary_pixel_parts_do_not_lend_active_novelty_to_an_unknown_arm():
    p,a,b=setup()
    # Fully known left-side evidence, but all physical movement belongs to an
    # unmeasured right side. The raw right motion remains available separately.
    p.velocity[[a,b],1,:,0:2]=0.
    p.projected[[a,b],1,0,2:5,:2]=p.projected[[a,b],1,1,2:5,:2]
    p.projected[[a,b],1,3,2:5,:2]=p.projected[[a,b],1,2,2:5,:2]
    p.phase_certainty[[a,b],1,:,0:2]=1.;p.motion_certainty[[a,b],1,:,0:2]=1.
    assert extract(p,1)[a,b,FEATURE_NAMES.index('pixel_agreed_active_novelty')]==0


def test_missing_identity_is_separate_from_appearance_mismatch_and_hint_is_not_certification():
    *_,p,_=prepared();f=extract(p,1)
    p.vectors[:]=0.;missing=extract(p,1)
    assert np.all(missing[:,:,FEATURE_NAMES.index('appearance_unmeasured')]==1)
    assert np.all(missing[:,:,FEATURE_NAMES.index('anonymous_appearance_cosine')]==0)
    assert np.any(f[:,:,FEATURE_NAMES.index('appearance_unmeasured')]==0)


def test_real_producer_retains_moving_connection_instead_of_collapsing_it_to_stationary_pose():
    observations=[]
    for moving in (True,False):
        nodes,banks,poses,styles=inputs()
        if not moving:
            for pose in poses:pose[:]=pose[:,2:3]
        states,data,_=build(nodes,banks,poses,styles,fps=10.,source_events=[np.array([],int)]*2,
            body_states=[np.ones((4,4,3),int)]*2,part_observer=provisional_parts)
        p=Ports(nodes,states,data,fps=10.,music=[0.,1.,1.,0.])
        a=states.index(State(0,0,'native'));b=states.index(State(1,0,'native'))
        observations.append(extract(p,1)[a,b])
    moving,stationary=observations
    # The previous19 channels collapsed these different real pose sequences.
    np.testing.assert_allclose(moving[:19],stationary[:19],atol=1e-7,rtol=0.)
    for name in ('raw_model_angular_activity','raw_model_screen_activity',
                 'raw_model_angular_style','raw_model_screen_style'):
        k=FEATURE_NAMES.index(name)
        assert moving[k]>.05 and stationary[k]==0.
    assert moving[FEATURE_NAMES.index('pixel_agreed_active_novelty')]==0.
    assert moving[FEATURE_NAMES.index('motion_unmeasured')]==1.


def test_raw_activity_requires_the_same_primitive_on_both_sides_and_physical_window():
    p,a,b=setup()
    p.body[[a,b],1,:,:,2]=0.
    # A left-arm estimate and B right-arm estimate must not be combined into
    # a moving connection. Neither the common shoulders nor missing hips move.
    p.body[a,1,:,[0,2,4],2]=1.;p.body[b,1,:,[1,3,5],2]=1.
    values=extract(p,1)[a,b]
    assert values[19]==values[20]==values[21]==values[22]==0.
    p, a, b=setup()
    for row in p.data:row['physical_window_valid'][1]=False
    values=extract(p,1)[a,b]
    assert values[19]==values[20]==values[21]==values[22]==0.


def test_which_actual_arm_has_pixel_agreement_cannot_disappear_into_global_fractions():
    nodes,banks,poses,styles=inputs()
    for pose in poses:pose[:]=pose[:,2:3]
    poses[1][:,:,:,5,0]+=.15  # only the right wrist estimate disagrees
    states,data,_=build(nodes,banks,poses,styles,fps=10.,source_events=[np.array([],int)]*2,
        body_states=[np.ones((4,4,3),int)]*2,part_observer=provisional_parts)
    p=Ports(nodes,states,data,fps=10.,music=[0.,1.,1.,0.])
    a=states.index(State(0,0,'native'));b=states.index(State(1,0,'native'));values=[]
    for agreed in ([0,1,4],[2,3,5]):
        p.phase_certainty[:]=.5;p.motion_certainty[:]=.5
        for i in (a,b):p.phase_certainty[i,1,:,agreed]=1.
        values.append(extract(p,1)[a,b])
    # Identical geometry AND identical confidence fractions, but agreement is
    # attached to a different actual limb. A global max plus fractions lost it.
    np.testing.assert_array_equal(values[0][:23],values[1][:23])
    assert not np.array_equal(values[0],values[1])


def test_body_absence_cannot_change_the_same_continuous_face_residual():
    p,a,b=setup()
    p.head[:,1,1,0]+=.02
    before=extract(p,1)
    p.body[:]=0.
    after=extract(p,1)
    np.testing.assert_array_equal(before[:,:,:2],after[:,:,:2])
    assert after[a,a,FEATURE_NAMES.index('face_position')]==0.
    assert after[a,a,FEATURE_NAMES.index('face_comparison_observed')]==1.


def test_actual_native_face_discontinuity_uses_the_exposed_boundary_without_bodies():
    p,a,b=setup();p.body[:]=0.
    p.data[a]['known_center_native_cut'][1]=True;p.continuous[a,1]=False
    p.head[a,1,1,0]+=.2
    f=extract(p,1)[a,a]
    assert f[FEATURE_NAMES.index('face_position')]>.3
    assert f[FEATURE_NAMES.index('face_comparison_observed')]==1.
    assert f[FEATURE_NAMES.index('outgoing_known_native_cut')]==1.


def test_intrinsic_played_motion_survives_zero_style_and_unavailable_counterfactual_source():
    p,a,b=setup();p.colour[:,1,1]=p.colour[0,1,0]
    f=extract(p,1)[a,b]
    assert f[FEATURE_NAMES.index('style_novelty')]==0.
    assert f[FEATURE_NAMES.index('played_pre_angular_activity')]>.05
    assert f[FEATURE_NAMES.index('played_post_angular_activity')]>.05
    # A native cut hides the counterfactual pre-B and post-A views, not the
    # actual played pre-A/post-B portions. Their own activity remains present.
    p.continuous[[a,b],1]=False
    f=extract(p,1)[a,b]
    assert f[FEATURE_NAMES.index('played_pre_angular_activity')]>.05
    assert f[FEATURE_NAMES.index('played_post_angular_activity')]>.05
    assert f[FEATURE_NAMES.index('raw_model_angular_activity')]==0.


def test_unmeasured_real_hips_and_absent_hips_have_different_geometry_presence():
    p,a,b=setup();before=extract(p,1)[a,b]
    p.body[:,1,:,6:8,2]=0.
    after=extract(p,1)[a,b]
    k=FEATURE_NAMES.index('unmeasured_torso_phase_geometry')
    assert before[k]>after[k]
    assert before[FEATURE_NAMES.index('torso_phase')]==after[FEATURE_NAMES.index('torso_phase')]==0.


def test_actual_added_camera_context_and_candidate_dependent_music_are_not_discarded():
    p,a,b=setup()
    for row in p.data:row['ticks']=np.array([0,5,10,15,20,30]);row['curve']=np.tile([1.,.5,.5],(6,1))
    p.data[a]['curve'][1,0]=1.2
    f=extract(p,1)
    assert f[a,b,FEATURE_NAMES.index('pre_added_camera_zoom_travel')]>0
    assert f[b,b,FEATURE_NAMES.index('pre_added_camera_zoom_travel')]==0
    assert f[a,b,FEATURE_NAMES.index('event_active_style')]>0
    assert f[a,a,FEATURE_NAMES.index('event_active_style')]==0
