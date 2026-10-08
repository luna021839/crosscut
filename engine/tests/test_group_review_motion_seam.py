from dataclasses import replace
from copy import deepcopy
import pytest
from crosscut.group_review_camera_repair import FramingDescriptor
from crosscut.group_review_motion_seam import (score_review_motion_seam,compare_aligned_review_motion_windows,
    compare_aligned_review_head_windows,score_review_head_seam)


def view(tick,phase):
    vector=[1.]+[0.]*127
    shape=tuple((k,k*.1,phase) for k in range(4))
    d=FramingDescriptor((.5,.35),.3,.15,shape,7,True)
    return dict(row=dict(output_tick=tick,native_observation=dict(request_source_index=2,
        native_frame_index=tick,source_clock_identity='clock',source_file_identity=[1,2,3,4,5]),
        projection=dict(camera_sha256='camera',heads=[
        dict(observation_id=7,projected_landmarks_xy=[[.45,.3],[.55,.3],[.5,.36],[.46,.4],[.54,.4]])])),
        descriptor=d,vector=vector,near=1.,appearance=dict(heads=[dict(observation_id=7,embedding=vector)]))


def test_same_pose_but_reversed_action_is_worse_than_continuing_action():
    left=[view(0,0.),view(2,.2)]
    good=[view(3,.3),view(5,.5)]
    reverse=[view(3,.3),view(5,.1)]
    a,b=[score_review_motion_seam(left,right,fps=30.) for right in (good,reverse)]
    assert a['state']==b['state']=='OBSERVED_BODY_PHASE_HYPOTHESIS'
    assert a['pose_phase_error']==pytest.approx(0.)
    assert b['direction_error'] > a['direction_error']
    assert b['exploration_cost'] > a['exploration_cost']


def test_native_actor_motion_stays_connected_across_authored_camera_queries():
    from crosscut.group_review_motion_seam import review_window_connected
    rows=[view(t,t*.1) for t in range(4)]
    for t,v in enumerate(rows):
        v['native_camera_unit_index']=9
        v['row']['projection']['camera_sha256']=f'authored-pan-query-{t}'
    assert review_window_connected(rows,require_native_motion=True)
    rows[2]['native_camera_unit_index']=10
    assert not review_window_connected(rows,require_native_motion=True)
    rows[2].pop('native_camera_unit_index')
    assert not review_window_connected(rows,require_native_motion=True)


def head_view(tick,angle):
    v=view(tick,0.);v['descriptor']=replace(v['descriptor'],pose_shape=None)
    v['row']['projection']['heads'][0]['projected_landmarks_xy'][2][0]=.5+angle*.1
    return v


def test_actual_head_seam_measures_orientation_not_mouth_or_body_choreography():
    left=[head_view(0,0.),head_view(1,.1)]
    a=score_review_head_seam(left,[head_view(2,.2),head_view(3,.3)],fps=30.)
    b=score_review_head_seam(left,[head_view(2,.2),head_view(3,.1)],fps=30.)
    assert a['state']=='OBSERVED_OUTPUT_HEAD_POSE_SEAM_HYPOTHESIS'
    assert a['pose_phase_error']==pytest.approx(0.,abs=1e-10)
    assert b['exploration_cost']>a['exploration_cost']
    assert not a['body_motion_certified'] and not a['mouth_points_used']


def test_actual_head_seam_does_not_replace_a_missing_selected_occurrence():
    assert score_review_head_seam([head_view(0,0.),None],
        [head_view(2,.2),head_view(3,.3)],fps=30.)['state']=='INCONCLUSIVE'


def test_close_face_with_mouth_motion_is_not_a_body_phase_observation():
    a=[view(0,0.),view(2,.2)]
    b=[view(3,.3),view(5,.5)]
    b[1]['descriptor']=replace(b[1]['descriptor'],pose_shape=None)
    assert score_review_motion_seam(a,b,fps=30.)['state']=='INCONCLUSIVE'


def test_explicit_phase_mapping_keeps_raw_observations_and_uses_output_dt():
    a=[view(0,0.),view(2,.2)]
    b=[view(8,.3),view(10,.5)]
    original=deepcopy(b)
    wrong=score_review_motion_seam(a,b,fps=30.)
    mapped=score_review_motion_seam(a,b,fps=30.,incoming_output_ticks=(3,5),maximum_phase_frames=6)
    assert wrong['pose_phase_error'] > mapped['pose_phase_error']
    assert b==original
    assert mapped['measured_incoming_ticks']==(8,10)
    assert mapped['proposed_incoming_output_ticks']==(3,5)
    assert mapped['phase_proposal_requires_actual_renderer_recompilation']
    assert not mapped['renderer_clock_authority_granted'] and not mapped['quality_approved']


def test_missing_interior_pose_or_incoming_frame_cannot_buy_a_better_score():
    left=[view(0,0.),view(1,.1),view(2,.2)]
    right=[view(3,.3),view(4,.4),view(5,.5)]
    right[1]['descriptor']=replace(right[1]['descriptor'],pose_shape=None)
    assert score_review_motion_seam(left,right,fps=30.)['state']=='INCONCLUSIVE'
    right[1]=None
    assert score_review_motion_seam(left,right,fps=30.)['state']=='INCONCLUSIVE'


def test_reversed_or_overlapping_output_clock_is_rejected():
    with pytest.raises(ValueError,match='order'):
        score_review_motion_seam([view(2,.2),view(0,0.)],[view(3,.3),view(5,.5)],fps=30.)
    with pytest.raises(ValueError,match='join'):
        score_review_motion_seam([view(0,0.),view(2,.2)],[view(3,.3),view(5,.5)],fps=30.,
            incoming_output_ticks=(2,4))


def test_matching_endpoints_cannot_hide_a_different_middle_actor_or_camera():
    left=[view(0,0.),view(1,.1),view(2,.2)]
    right=[view(3,.3),view(4,.4),view(5,.5)]
    good=score_review_motion_seam(left,right,fps=30.)
    for kind in ('actor','camera'):
        changed=deepcopy(right)
        if kind=='actor':
            changed[1]['vector']=[0.,1.]+[0.]*126
        else:
            changed[1]['row']['projection']['camera_sha256']='another-camera'
        result=score_review_motion_seam(left,changed,fps=30.)
        assert result['state']=='INCONCLUSIVE'
        assert not result['all_sampled_window_edges_connected']
        assert result['exploration_cost']>good['exploration_cost']


def test_window_cannot_borrow_another_source_or_reverse_native_time():
    left=[view(0,0.),view(1,.1),view(2,.2)]
    right=[view(3,.3),view(4,.4),view(5,.5)]
    for key,value in (('request_source_index',3),('source_clock_identity','other'),
                      ('source_file_identity',[1,2,3,4,6]),('native_frame_index',8)):
        changed=deepcopy(right)
        changed[1]['row']['native_observation'][key]=value
        with pytest.raises(ValueError,match='source/native/clock'):
            score_review_motion_seam(left,changed,fps=30.)


def test_equal_whole_window_endpoints_do_not_hide_reversed_cut_tangent():
    left=[view(0,0.),view(1,.1),view(2,.2),view(3,.3)]
    continuing=[view(4,.4),view(5,.5),view(6,.6),view(7,.7),view(8,.4)]
    reversing=[view(4,.4),view(5,.3),view(6,.2),view(7,.1),view(8,.4)]
    a,b=[score_review_motion_seam(left,r,fps=30.) for r in (continuing,reversing)]
    assert a['direction_error']==pytest.approx(0.)
    assert b['direction_error']==pytest.approx(1.)
    assert b['exploration_cost']>a['exploration_cost']


def test_phase_mapping_cannot_silently_change_video_tempo():
    with pytest.raises(ValueError,match='constant video phase'):
        score_review_motion_seam([view(0,0.),view(2,.2)],[view(8,.3),view(10,.5)],
            fps=30.,incoming_output_ticks=(3,6))


def test_unmeasured_large_gap_and_conflicting_native_alias_stay_explicit():
    left=[view(0,0.),view(2,.2)]
    right=[view(3,.3),view(16,.5)]
    assert score_review_motion_seam(left,right,fps=30.)['state']=='INCONCLUSIVE'
    right=[view(3,.3),view(4,.4)]
    right[1]['row']['native_observation']['native_frame_index']=3
    right[1]['row']['native_observation']['different_raw']='not same observation'
    with pytest.raises(ValueError,match='conflicting'):
        score_review_motion_seam(left,right,fps=30.)


def test_dropping_a_bad_wrist_cannot_manufacture_a_low_error_body_phase():
    left=[view(0,0.),view(2,.2)]
    right=[view(3,.3),view(5,.5)]
    for sequence in (left,right):
        for v in sequence:
            d=v['descriptor']
            v['descriptor']=replace(d,pose_shape=(*d.pose_shape,(4,.4,2.),(5,.5,3.)))
    right[0]['descriptor']=replace(right[0]['descriptor'],
        pose_shape=right[0]['descriptor'].pose_shape[:-1])
    result=score_review_motion_seam(left,right,fps=30.)
    assert result['state']=='INCONCLUSIVE' and result['pose_phase_error'] is None
    assert result['unpaired_observed_joint_indices']==[5]
    assert result['evidence_loss_prevents_body_phase_comparison']


def test_consistently_observed_partial_body_is_not_a_full_body_requirement():
    result=score_review_motion_seam([view(0,0.),view(2,.2)],[view(3,.3),view(5,.5)],fps=30.)
    assert result['state']=='OBSERVED_BODY_PHASE_HYPOTHESIS'
    assert result['observed_joint_indices']==[0,1,2,3]


def test_near_boundary_middle_motion_cannot_hide_behind_symmetric_endpoints():
    left=[view(0,0.),view(1,.1),view(2,.2)]
    good=[view(3,.3),view(4,.4),view(5,.5)]
    changed=[view(3,.3),view(4,3.),view(5,.5)]
    a,b=[score_review_motion_seam(left,right,fps=30.) for right in (good,changed)]
    assert b['velocity_error']>a['velocity_error']
    assert b['exploration_cost']>a['exploration_cost']


def aligned(a,b):
    return compare_aligned_review_motion_windows(a,b,fps=30.,
        requested_output_ticks=(0,1,2,3,4),requested_joint_indices=(0,1,2,3))


def test_interior_aligned_trajectory_and_adjacent_velocities_are_actually_compared():
    a=[view(i,i*.1) for i in range(5)]
    b=deepcopy(a);b[2]=view(2,3.)
    baseline=aligned(a,a);changed=aligned(a,b)
    assert baseline['exploration_cost']==pytest.approx(0.)
    assert changed['trajectory_error']>0. and changed['velocity_error']>0.
    assert changed['interior_pose_observations']==5


def test_same_fast_accelerating_choreography_on_both_sources_is_not_a_smoothness_failure():
    a=[view(i,p) for i,p in enumerate((0.,3.,-2.,5.,0.))]
    assert aligned(a,deepcopy(a))['exploration_cost']==pytest.approx(0.)


def test_frozen_wrist_request_cannot_be_erased_to_improve_aligned_trajectory():
    a=[view(i,i*.1) for i in range(5)];b=deepcopy(a)
    b[2]['descriptor']=replace(b[2]['descriptor'],pose_shape=b[2]['descriptor'].pose_shape[:-1])
    result=aligned(a,b)
    assert result['state']=='INCONCLUSIVE' and result['exploration_cost'] is None
    assert result['missing_requested_joint_observations']==((1,2,3),)


def test_constant_phase_exceeding_the_explicit_batch_bound_is_rejected():
    with pytest.raises(ValueError,match='frame bound'):
        score_review_motion_seam([view(0,0.),view(2,.2)],[view(8,.3),view(10,.5)],
            fps=30.,incoming_output_ticks=(3,5))


def test_head_connection_is_independent_of_mouth_animation_and_is_not_body_motion():
    a=[view(t,0.) for t in range(5)];b=deepcopy(a)
    for v in b:
        v['descriptor']=replace(v['descriptor'],pose_shape=None)
        v['row']['projection']['heads'][0]['projected_landmarks_xy'][3:]=[[.2,.8],[.8,.9]]
    result=compare_aligned_review_head_windows(a,b,fps=30.,requested_output_ticks=range(5))
    assert result['exploration_cost']==pytest.approx(0.)
    assert not result['choreography_motion_observed'] and not result['mouth_points_used']


def test_head_path_reads_middle_orientation_and_keeps_missing_occurrence_unresolved():
    a=[view(t,0.) for t in range(5)];b=deepcopy(a)
    b[2]['row']['projection']['heads'][0]['projected_landmarks_xy'][2]=[.7,.5]
    result=compare_aligned_review_head_windows(a,b,fps=30.,requested_output_ticks=range(5))
    assert result['trajectory_error']>0. and result['velocity_error']>0.
    b[2]=None
    assert compare_aligned_review_head_windows(a,b,fps=30.,requested_output_ticks=range(5))['state']=='INCONCLUSIVE'
