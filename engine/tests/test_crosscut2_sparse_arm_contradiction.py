import copy
import numpy as np
import pytest
from crosscut2.played_partial_pose import features,matrix
from crosscut2.sparse_arm_contradiction import compare,bound_credit
from crosscut2.editorial_roles import event_energy
from crosscut2.played_pose_port_programme import PlayedPosePortPrior
from test_crosscut2_played_partial_pose import observation


def pair(*,opposed=False,enabled=True):
    a,b=observation(0),observation(1)
    for o in (a,b):o['original_full_image_people_used']=True
    if opposed:
        # One visible arm reverses. The other arm and face remain identical.
        b['body'][:,2,:2]=[.45,.3];b['body'][:,4,:2]=[.5,.25]
    return a,b,features([a],fps=30.,sparse_arm_contradictions=enabled),features(
        [b],fps=30.,sparse_arm_contradictions=enabled)


def test_visible_opposed_arm_not_diluted_by_matching_face_other_arm_or_hips():
    a,b,left,right=pair(opposed=True)
    result=matrix(left,right,details=True)
    assert result['geometry_rate'][0,0]==0.
    assert result['sparse_arm_direction_residual'][0,0]>.99
    assert result['sparse_arm_contradiction_relative_cost'][0,0]==pytest.approx(.25)
    assert result['sparse_arm_action_credit_cap'][0,0]==0.
    assert result['allowed'][0,0]


def test_matching_arm_earns_no_new_bonus_or_extra_penalty_and_default_unchanged():
    a,b,left,right=pair()
    new=matrix(left,right,details=True)
    old=matrix(features([a],fps=30.),features([b],fps=30.),details=True)
    assert new['sparse_arm_contradiction_relative_cost'][0,0]==0.
    assert 'sparse_arm_contradiction_relative_cost' not in old
    for key in ('pose_rate','geometry_rate','allowed','matched_action_change_density'):
        np.testing.assert_array_equal(new[key],old[key])


@pytest.mark.parametrize('unknown',['missing','contradicted','ambiguous','outside','not_original'])
def test_unseen_or_ambiguous_opposition_never_becomes_measured_bad_arm(unknown):
    a,b,_,_=pair(opposed=True)
    if unknown=='missing':b['evidence'][:,[0,2,4]]=0
    if unknown=='contradicted':b['evidence'][:,[0,2,4]]=3
    if unknown=='ambiguous':b['ownership_ambiguity'][:,[0,2,4]]=True
    if unknown=='outside':b['body'][:,[0,2,4],0]=1.1
    if unknown=='not_original':b['original_full_image_people_used']=False
    value=compare(features([a],fps=30.),features([b],fps=30.))
    assert value['sparse_arm_contradiction_relative_cost'][0,0]==0.
    assert value['sparse_arm_action_credit_cap'][0,0]==pytest.approx(1.)


def test_one_actual_exposed_bone_suffices_but_missing_wrist_is_not_filled():
    a,b,_,_=pair(opposed=True)
    for o in (a,b):o['evidence'][:,4]=0
    detail=compare(features([a],fps=30.),features([b],fps=30.))
    assert detail['sparse_arm_bones_measured'][0,0].tolist()==[True,False,True,True]
    assert detail['sparse_arm_contradiction_relative_cost'][0,0]>.24


def test_symmetric_pair_and_viewport_visibility_and_no_camera_motion_reward():
    a,b,left,right=pair(opposed=True)
    first=compare(left,right)
    second=compare(right,left)
    np.testing.assert_array_equal(first['sparse_arm_contradiction_relative_cost'],second['sparse_arm_contradiction_relative_cost'].T)
    for o in (a,b):o['camera_strip'][:,0]=1.2
    value=compare(features([a],fps=30.),features([b],fps=30.))
    assert value['sparse_arm_contradiction_relative_cost'][0,0]==pytest.approx(.25)


def test_one_sided_option_is_rejected_and_prior_requires_real_consumer():
    a,b,left,right=pair(opposed=True)
    with pytest.raises(ValueError,match='Both sides'):
        matrix(left,features([b],fps=30.),details=True)
    assert not PlayedPosePortPrior().sparse_arm_contradictions
    with pytest.raises(ValueError,match='original full-image'):
        PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,sparse_arm_contradictions=True)
    prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,person_attention=True,
        original_person_attention=True,sparse_arm_contradictions=True)
    assert prior.artifact['sparse_visible_arm_contradictions']
    assert not prior.artifact['sparse_arm_scoring_changes_source_or_cut_admission']


def test_cli_rejects_unused_arm_option_before_reading_any_source(monkeypatch,tmp_path):
    from crosscut2.stage_mix import main
    argv=['stage_mix','--sparse-arm-contradictions']
    for key in ('source-plan','analysis','cache','output','models'):
        argv.extend(['--'+key,str(tmp_path/key)])
    monkeypatch.setattr('sys.argv',argv)
    with pytest.raises(ValueError,match='sparse arm contradictions require'):
        main()


def test_six_frame_point_units_do_not_vanish_at_short_observation_support():
    _,_,left,right=pair(opposed=True)
    value=matrix(left,right,details=True)
    old=dict(value);old.pop('sparse_arm_contradiction_relative_cost')
    for seconds in (.001,.2,1.):
        delta=event_energy(value,seconds,fps=30.,point_geometry=True)-event_energy(old,seconds,fps=30.,point_geometry=True)
        assert delta[0,0]==pytest.approx(.05)


def test_existing_dense_arm_score_cannot_restore_opposed_sparse_bone_credit():
    from crosscut2.played_interval_utility import articulated_seam_evidence
    a,b,left,right=pair(opposed=True)
    detail=matrix(left,right,details=True)
    detail['matched_action_evidence_density']=np.ones((1,1))
    after=articulated_seam_evidence(detail,[a],[b])
    assert after['matched_action_change_density'][0,0]==0.
    assert after['sparse_arm_contradiction_relative_cost'][0,0]==pytest.approx(.25)


def test_unequal_candidate_counts_stay_pair_local_not_max_over_actors():
    a,b,_,_=pair(opposed=True)
    good=copy.deepcopy(a);good['side']=1
    left=features([a],fps=30.,sparse_arm_contradictions=True)
    right=features([b,good],fps=30.,sparse_arm_contradictions=True)
    result=matrix(left,right,details=True)
    assert result['sparse_arm_contradiction_relative_cost'].shape==(1,2)
    assert result['sparse_arm_contradiction_relative_cost'][0,0]>.24
    assert result['sparse_arm_contradiction_relative_cost'][0,1]==0.


def test_digital_crop_cannot_hide_known_opposed_arm_to_get_clean_match():
    a,b,left,right=pair(opposed=True)
    before=compare(left,right)
    for o in (a,b):o['camera_strip'][:]=[2.,.62,.5]
    after=compare(features([a],fps=30.),features([b],fps=30.))
    assert after['sparse_arm_contradiction_relative_cost'][0,0]==before['sparse_arm_contradiction_relative_cost'][0,0]
    assert after['sparse_arm_action_credit_cap'][0,0]==0.


def test_both_wholly_hidden_arm_has_no_added_visual_cost_but_no_false_match_credit():
    a,b,_,_=pair(opposed=True)
    for o in (a,b):
        o['evidence'][:,[1,3,5]]=0
        o['camera_strip'][:]=[6.,.62,.5]
    after=compare(features([a],fps=30.),features([b],fps=30.))
    assert after['sparse_arm_contradiction_relative_cost'][0,0]==0.
    assert after['sparse_arm_action_credit_cap'][0,0]==0.
