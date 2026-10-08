from dataclasses import replace
import json
import math
import pytest
from crosscut.group_review_camera_repair import (FramingDescriptor,fit_whole_camera_track,
    transformed_descriptor,geometry_cost,project_review_editorial_point)
from crosscut.group_review_two_shot_camera import (realize_review_two_shot_camera_pair,
    FrozenRealizedTwoShotCandidate,select_review_two_shot_diagnostic,fit_review_coupled_boundary_keys,
    fit_review_temporal_camera_corridor,measure_review_partial_joint_trajectory)
from crosscut.render import EditorialTransformKeyframe
from crosscut.group_review_projected_action import measure_review_partial_screen_trajectory
from crosscut.group_review_two_shot_camera import realize_review_focal_pair
from crosscut.group_review_two_shot_camera import (measure_review_focal_incident,
    compare_review_focal_incident_inventory)


def incident_fixture():
    points={0:(.4,.45),1:(.6,.45),2:(.3,.55),3:(.7,.55),4:(.3,.35),5:(.7,.35)}
    frames=[{j:(x+.005*t,y) for j,(x,y) in points.items()} for t in range(6)]
    face=dict(landmarks_xy=((.45,.3),(.55,.3),(.5,.35),(.46,.4),(.54,.4)))
    return dict(left_faces=[face]*3,right_faces=[face]*3,left_points=frames[:3],right_points=frames[3:],
        left_ticks=(0,1,2),right_ticks=(3,4,5),fps=30.,
        left_native=[(0,'c','file',7,t) for t in range(3)],
        right_native=[(1,'d','other-file',3,t) for t in range(3,6)])


def test_actual_entry_and_return_use_measured_face_and_motion_not_perfect_continuation_badge():
    data=incident_fixture();good=measure_review_focal_incident(**data)
    assert good['exploration_score']==pytest.approx(0.,abs=1e-12)
    data['right_points']=[{j:(x+.2,y) for j,(x,y) in f.items()} for f in data['right_points']]
    bad=measure_review_focal_incident(**data)
    assert bad['exploration_score']>good['exploration_score']
    assert bad['actual_input_identity']!=good['actual_input_identity']
    assert not bad['full_action_certified']


def test_incident_erasing_bad_observations_cannot_improve_fixed_upper_cost():
    data=incident_fixture()
    data['right_points']=[{j:(-10.*x,y) for j,(x,y) in f.items()} for f in data['right_points']]
    observed=measure_review_focal_incident(**data)
    data['right_points']=[{}]*3;data['right_faces']=[None]*3
    missing=measure_review_focal_incident(**data)
    assert missing['exploration_score']>=observed['exploration_score']
    assert missing['face_scale_log_error'] is None


def test_incident_cannot_call_a_native_cut_or_repeated_raster_a_measured_tangent():
    for native in (None,[(1,'d','other-file',3,3),(1,'d','other-file',4,4),(1,'d','other-file',4,5)],
            [(1,'d','other-file',3,3)]*3):
        data=incident_fixture();data['right_native']=native
        result=measure_review_focal_incident(**data)
        assert not result['physical_tangents_observed']
        assert result['action_cost_intervals']==[(0.,1.)]*3


def test_good_middle_cannot_buy_a_worse_return_or_discard_known_action():
    from copy import deepcopy
    baseline=measure_review_focal_incident(**incident_fixture())
    original=[deepcopy(baseline) for _ in range(3)]
    original[1]['face_scale_log_error']=.5
    candidate=deepcopy(original);candidate[1]['face_scale_log_error']=0.
    candidate[2]['face_scale_log_error']=.9
    result=compare_review_focal_incident_inventory(original,candidate)
    assert not result['eligible_for_further_review']
    assert result['failures'][0]['incident']==2
    candidate=deepcopy(original);candidate[0]['joint_requests'][0]['state']='UNOBSERVED'
    assert not compare_review_focal_incident_inventory(original,candidate)['eligible_for_further_review']
    assert compare_review_focal_incident_inventory(original,original)['eligible_for_further_review']


def test_focal_pair_scores_the_exact_render_tracks_without_automatic_zoom_release():
    points={0:(.4,.5),1:(.6,.5),2:(.4,.6),3:(.6,.6),4:(.4,.4),5:(.6,.4)}
    ticks=(13,14,15,16,17)
    key=EditorialTransformKeyframe(0.,1.5)
    tracks,receipt=realize_review_focal_pair(left_key=key,right_key=key,
        left_support=[tuple(points.values())]*5,right_support=[tuple(points.values())]*5,
        left_motion=[points]*5,right_motion=[points]*5,motion_ticks=ticks,
        first=10,boundary=15,end=20,fps=30.,boundary_fit=dict(geometry_score=0.,
            log_face_scale_error=0.,projected_head_center_error=0.))
    assert tracks and all(k.zoom==1.5 for t in tracks for k in t.keyframes)
    assert receipt['screen_motion']['exploration_score']==pytest.approx(0.)
    assert receipt['exploration_cost']==pytest.approx(0.)
    assert not receipt['automatic_zoom_release']
    assert receipt['left_camera']['keyframes'][0]['zoom']==tracks[0].keyframes[0].zoom
    assert not receipt['camera_refit_after_selection_permitted']


def test_focal_pair_keeps_unknown_support_and_face_size_jump_in_its_score():
    points={0:(.4,.5),1:(.6,.5),4:(.4,.4)}
    args=dict(left_key=EditorialTransformKeyframe(0.,1.5),right_key=EditorialTransformKeyframe(0.,1.5),
        left_support=[None]*5,right_support=[None]*5,left_motion=[points]*5,right_motion=[points]*5,
        motion_ticks=(13,14,15,16,17),first=10,boundary=15,end=20,fps=30.)
    _,good=realize_review_focal_pair(**args,boundary_fit=dict(geometry_score=0.,log_face_scale_error=0.,projected_head_center_error=0.))
    _,bad=realize_review_focal_pair(**args,boundary_fit=dict(geometry_score=.25,log_face_scale_error=.4,projected_head_center_error=.1))
    assert bad['exploration_cost']==pytest.approx(good['exploration_cost']+2.2)
    assert all(not r['complete_support_certified'] for r in bad['support'])
    assert not bad['full_action_certified'] and not bad['quality_approved']


def test_opposite_screen_root_motion_is_not_erased_by_equal_relative_arm_shapes():
    ticks=(0,1,2,3,4)
    shape={0:(.4,.45),1:(.6,.45),2:(.3,.55),3:(.7,.55),4:(.3,.35),5:(.7,.35)}
    a=[{j:(x+.005*t,y) for j,(x,y) in shape.items()} for t in ticks]
    b=[{j:(x-.005*t,y) for j,(x,y) in shape.items()} for t in ticks]
    relative=measure_review_partial_joint_trajectory(a,b,ticks=ticks,fps=30.)
    screen=measure_review_partial_screen_trajectory(a,b,ticks=ticks,fps=30.)
    same=measure_review_partial_screen_trajectory(a,a,ticks=ticks,fps=30.)
    assert relative['exploration_score']==pytest.approx(0.,abs=1e-10)
    assert screen['observed_root_direction_error']==pytest.approx(1.)
    assert screen['exploration_score']>same['exploration_score']+.5
    assert not screen['actual_screen_root_translation_removed']


def test_deleting_bad_screen_motion_observations_cannot_improve_upper_cost():
    ticks=(0,1,2,3,4)
    a=[{0:(.4,.4),1:(.6,.4),4:(.2+t*.03,.2)} for t in ticks]
    b=[{0:(.4,.4),1:(.6,.4),4:(.8-t*.03,.8)} for t in ticks]
    observed=measure_review_partial_screen_trajectory(a,b,ticks=ticks,fps=30.)
    removed=measure_review_partial_screen_trajectory(a,[{j:p for j,p in r.items() if j!=4} for r in b],ticks=ticks,fps=30.)
    assert removed['exploration_score']>=observed['exploration_score']
    assert not removed['full_action_certified'] and not removed['missing_observations_interpolated']


def test_one_sided_wrist_motion_has_no_measured_common_direction():
    ticks=(0,1,2,3,4)
    shape={0:(.4,.45),1:(.6,.45),2:(.3,.55),3:(.7,.55),4:(.3,.35),5:(.7,.35)}
    still=[dict(shape) for _ in ticks]
    moving=[dict(shape) for _ in ticks]
    for t,frame in zip(ticks,moving):frame[4]=(.3+.01*t,.35)
    relative=measure_review_partial_joint_trajectory(still,moving,ticks=ticks,fps=30.)
    assert not relative['nonstationary_common_action_observed']
    assert relative['observed_direction_error'] is None
    assert relative['bounded_direction_cost_interval'][1]>0.
    assert all(r['direction_error'] is None for r in relative['direction_requests'])
    assert any(r['applicability']=='ONE_SIDED_MOTION' for r in relative['direction_requests'])
    screen=measure_review_partial_screen_trajectory(still,moving,ticks=ticks,fps=30.)
    assert screen['observed_root_direction_error'] is None
    assert screen['direction_cost_interval'][1]>0.
    assert screen['measured_direction_requests']==0
    assert all(r['direction_applicability']=='BOTH_STATIONARY' for r in screen['root_velocity_samples'])


def test_observed_stillness_is_not_a_measured_angle_or_common_moving_action():
    ticks=(0,1,2)
    points={0:(.4,.45),1:(.6,.45),2:(.3,.55),3:(.7,.55),4:(.3,.35),5:(.7,.35)}
    for measure in (measure_review_partial_joint_trajectory,measure_review_partial_screen_trajectory):
        result=measure([points]*3,[points]*3,ticks=ticks,fps=30.)
        assert result['exploration_score']==pytest.approx(0.)
        assert result['measured_direction_requests']==0
        assert all(r['applicability']=='BOTH_STATIONARY' and r['direction_error'] is None
            for r in result['direction_requests'])


def view(t,x,h):
    box=[x-h/4,.35-h/2,h/2,h]
    row=dict(output_tick=t,native_observation=dict(heads=[dict(observation_id=0,box_xywh=box)],people=[],
        frame_size_wh=[1920,1080]),
        projection=dict(heads=[dict(observation_id=0,projected_box_xywh=box,visible_fraction=1.,
            projected_landmarks_xy=[])],people=[]),query=dict(camera=dict(output_size=[1920,1080])))
    return dict(row=row,descriptor=FramingDescriptor((x,.35),h,h/2,None,0,False))


def test_actual_donor_joint_fit_beats_two_donors_independently_fitted_to_old_neighbours():
    a=[view(t,.42,.15) for t in range(10,16)];b=[view(t,.60,.18) for t in range(16,22)]
    old=replace(a[0]['descriptor'],head_center=(.5,.45),face_height=.08,face_width=.04)
    la,lb,receipt=realize_review_two_shot_camera_pair(a,b,first=10,boundary=16,end=22,fps=30.,
        original_entry=old,original_exit=old,required_support_check=lambda side,v,key:'PASS')
    def oldfit(values):
        return fit_whole_camera_track([v['row'] for v in values],old,old,duration=5/30.,
            sample_times=tuple(i/30. for i in range(6)),head_ids=(0,)*6,
            descriptors=tuple(v['descriptor'] for v in values),required_support_check=lambda row,key:True)
    aa,bb=oldfit(a),oldfit(b)
    before=geometry_cost(transformed_descriptor(a[-1]['descriptor'],aa.evaluate(5/30.)),
        transformed_descriptor(b[0]['descriptor'],bb.evaluate(0.)))
    assert receipt['state']=='REALIZED' and receipt['geometry_costs']['internal']<before
    assert receipt['camera_fitted_before_selection'] and not receipt['post_selection_refit_permitted']


def candidate(boundary,motion,geometry=0.,state='PASS',closeups=1.):
    return FrozenRealizedTwoShotCandidate(json.dumps(dict(contract='sealed_realized_two_shot_candidate_v1',
        boundary=boundary,technical_state=state,changed_frames=48,camera_fitted_before_selection=True,
        post_selection_camera_refit=False,native_motion_error=motion,projected_geometry_error=geometry,
        camera_instability=0.,closeup_fraction=closeups)))


def test_predeclared_boundary_alternative_and_motion_take_priority_over_closeup_reward():
    a=candidate(3665,2.,closeups=1.);b=candidate(3671,.1,closeups=.1)
    assert select_review_two_shot_diagnostic([a,b])==b
    assert select_review_two_shot_diagnostic([candidate(3677,0.,state='UNKNOWN')]) is None


def test_selected_camera_cannot_be_changed_after_the_realization_was_ranked():
    a=candidate(3671,.1);p=a.payload;p['camera_instability']=100.
    with pytest.raises(ValueError,match='after sealing'):a.assert_unchanged(p)


def test_joint_fit_does_not_claim_unknown_required_support_is_pass():
    a=[view(t,.42,.15) for t in range(10,16)];b=[view(t,.60,.18) for t in range(16,22)]
    _,_,r=realize_review_two_shot_camera_pair(a,b,first=10,boundary=16,end=22,fps=30.,
        original_entry=a[0]['descriptor'],original_exit=b[-1]['descriptor'],
        required_support_check=lambda side,v,key:'UNKNOWN')
    assert r['state']=='REALIZED' and r['actual_support_state']=='UNKNOWN'


def test_coupled_boundary_can_move_both_actual_cameras_without_inventing_missing_wrists():
    a=dict(center=(.38,.60),width=.09,height=.18)
    b=dict(center=(.66,.58),width=.1,height=.20)
    pa={0:(.34,.73),1:(.45,.73),3:(.48,.86),5:(.52,.79)}
    pb={0:(.62,.72),1:(.74,.72),3:(.77,.85),5:(.81,.77)}
    before=math.dist(a['center'],b['center'])
    values=fit_review_coupled_boundary_keys(left_head=a,right_head=b,left_joint_points=pa,right_joint_points=pb,
        left_required_visible_points=tuple(pa.values()),right_required_visible_points=tuple(pb.values()))
    assert values and len(values)<=8
    left,right,receipt=values[0]
    assert math.dist(project_review_editorial_point(a['center'],left),project_review_editorial_point(b['center'],right))<before
    assert receipt['missing_joint_requests']=={'left':[2,4],'right':[2,4]}
    assert not receipt['complete_requested_action_basis'] and not receipt['quality_approved']
    assert not receipt['partial_geometry_is_complete_motion_match']


def test_coupled_boundary_retains_known_joints_not_shared_by_both_sides():
    a=dict(center=(.4,.4),width=.1,height=.2);b=dict(center=(.6,.4),width=.1,height=.2)
    pa={0:(.35,.6),1:(.45,.6),4:(.0,.95)};pb={0:(.55,.6),1:(.65,.6)}
    options=fit_review_coupled_boundary_keys(left_head=a,right_head=b,left_joint_points=pa,right_joint_points=pb,
        left_required_visible_points=tuple(pa.values()),right_required_visible_points=tuple(pb.values()))
    assert options
    for ka,kb,r in options:
        assert all(0<=x<=1 for x in project_review_editorial_point(pa[4],ka))
        assert 4 not in r['missing_joint_requests']['left'] and 4 in r['missing_joint_requests']['right']


def test_coupled_boundary_rejects_nonphysical_head_and_unbounded_recipe_inventory():
    for count in (0,9):
        with pytest.raises(ValueError,match='bounded'):
            fit_review_coupled_boundary_keys(left_head={},right_head={},left_joint_points={},right_joint_points={},maximum_candidates=count)
    with pytest.raises(ValueError,match='geometry'):
        fit_review_coupled_boundary_keys(left_head=dict(center=(.5,.5),width=-.1,height=.2),
            right_head=dict(center=(.5,.5),width=.1,height=.2),left_joint_points={},right_joint_points={})


def test_temporal_corridor_does_not_require_a_moving_hand_inside_every_other_frames_crop():
    keys=tuple(EditorialTransformKeyframe(i/30.,2.,.25+.5*i/10,.5) for i in range(11))
    points=[[(.10+.8*i/10,.5)] for i in range(11)]
    track,receipt=fit_review_temporal_camera_corridor(keys,observed_points_by_frame=points,
        locked_indices=(0,10),fps=30.)
    assert track and not receipt['cross_time_coordinate_union_used']
    assert track.keyframes[0]==keys[0] and track.keyframes[-1]==keys[-1]
    for key,support in zip(track.keyframes,points):
        assert all(0<=x<=1 for x in project_review_editorial_point(support[0],key))
    assert not all(0<=x<=1 for x in project_review_editorial_point(points[-1][0],track.keyframes[0]))


def test_temporal_corridor_keeps_missing_support_unknown_and_locked_geometry_unmodified():
    keys=(EditorialTransformKeyframe(0.,1.5),EditorialTransformKeyframe(1/30.,1.5))
    track,receipt=fit_review_temporal_camera_corridor(keys,observed_points_by_frame=[None,[(.5,.5)]],
        locked_indices=(0,1),fps=30.)
    assert track.keyframes==keys and receipt['unmeasured_support_frame_indices']==[0]
    assert not receipt['complete_support_certified'] and not receipt['quality_approved']


def test_temporal_corridor_does_not_soften_an_illegal_locked_action_anchor():
    keys=(EditorialTransformKeyframe(0.,2.),EditorialTransformKeyframe(1/30.,2.))
    track,receipt=fit_review_temporal_camera_corridor(keys,observed_points_by_frame=[[(0.,.5)],[(.5,.5)]],
        locked_indices=(0,),fps=30.)
    assert track is None and receipt['frame_index']==0


def test_temporal_corridor_can_pan_to_a_descending_hand_without_zooming_out_and_back():
    keys=tuple(EditorialTransformKeyframe(i/30.,1.65,.65,.65) for i in range(13))
    points=[[(.70,.65),(.62,.90+.095*math.sin(math.pi*i/12))] for i in range(13)]
    track,receipt=fit_review_temporal_camera_corridor(keys,observed_points_by_frame=points,
        locked_indices=(0,12),fps=30.)
    assert track and min(k.zoom for k in track.keyframes)>1.60
    assert max(k.center_y for k in track.keyframes)>.68
    assert receipt['support_constrained_pan_is_distinct_from_zoom_release']
    assert receipt['second_order_camera_path'] and receipt['maximum_states_per_frame']<=14
    assert receipt['maximum_log_zoom_step']<.03
    for k,support in zip(track.keyframes,points):
        assert all(all(-1e-9<=v<=1+1e-9 for v in project_review_editorial_point(p,k)) for p in support)
    assert not receipt['quality_approved']


def test_actual_single_frame_support_does_not_force_a_one_frame_pan_impulse():
    keys=tuple(EditorialTransformKeyframe(i/30.,1.65,.65,.65) for i in range(13))
    points=[[(.70,.65),(.62,.90)] for _ in keys];points[6]=[(.70,.65),(.62,.995)]
    track,receipt=fit_review_temporal_camera_corridor(keys,observed_points_by_frame=points,
        locked_indices=(0,12),fps=30.)
    assert track and all(k.zoom==1.65 for k in track.keyframes)
    assert receipt['selected_camera_path_family']=='continuous_fixed_authored_zoom_pan'
    assert receipt['maximum_camera_vector_second_difference']<.01
    assert track.keyframes[5].center_y>keys[5].center_y
    assert not receipt['continuous_pan_proposal']['anatomical_observations_interpolated']
    assert not receipt['continuous_pan_proposal']['unknown_support_filled']
    for k,support in zip(track.keyframes,points):
        assert all(all(-1e-9<=v<=1+1e-9 for v in project_review_editorial_point(p,k)) for p in support)


def test_partial_trajectory_keeps_occlusion_missing_not_a_perfect_action_match():
    points={0:(.4,.4),1:(.6,.4),3:(.65,.65),5:(.7,.7)}
    rows=[dict(points,**{}) for _ in range(5)]
    r=measure_review_partial_joint_trajectory(rows,rows,ticks=(10,11,12,13,14),fps=30.)
    assert r['exploration_score']>0 and r['coverage']<1
    assert len(r['missing_joint_requests'])==20 and not r['full_action_certified']
    assert not r['missing_imputed_as_zero'] and not r['quality_approved']


def test_partial_trajectory_rejects_opposite_moving_actions_with_equal_midpoint_pose():
    rows=[];opposite=[]
    for i in range(5):
        p={0:(.4,.4),1:(.6,.4),2:(.3,.6),3:(.7,.6),4:(.2+.01*i,.7),5:(.8-.01*i,.7)}
        rows.append(p);opposite.append(dict(p,**{}))
        opposite[-1][4]=(.24-.01*i,.7);opposite[-1][5]=(.76+.01*i,.7)
    same=measure_review_partial_joint_trajectory(rows,rows,ticks=(10,11,12,13,14),fps=30.)
    bad=measure_review_partial_joint_trajectory(rows,opposite,ticks=(10,11,12,13,14),fps=30.)
    assert same['exploration_score']==pytest.approx(0.)
    assert bad['exploration_score']>.3


def test_partial_trajectory_does_not_invent_motion_through_a_missing_wrist_frame():
    p={0:(.4,.4),1:(.6,.4),3:(.65,.65),5:(.7,.7)}
    rows=[dict(p) for _ in range(5)];rows[2].pop(5)
    r=measure_review_partial_joint_trajectory(rows,rows,ticks=(10,11,12,13,14),fps=30.)
    assert r['measured_velocity_requests']==14
    assert not r['missing_joint_trajectory_interpolated']


def test_erasing_badly_matching_wrists_cannot_improve_exploratory_rank():
    left=[];right=[]
    for i in range(5):
        p={0:(.4,.4),1:(.6,.4),2:(.3,.6),3:(.7,.6),4:(.2+.01*i,.7),5:(.8-.01*i,.7)}
        left.append(p);right.append(dict(p))
        right[-1][4]=(.9-.01*i,.2);right[-1][5]=(.1+.01*i,.2)
    measured=measure_review_partial_joint_trajectory(left,right,ticks=(10,11,12,13,14),fps=30.)
    sparse=[{j:p for j,p in row.items() if j not in (4,5)} for row in right]
    dropped=measure_review_partial_joint_trajectory(left,sparse,ticks=(10,11,12,13,14),fps=30.)
    assert dropped['exploration_score']>=measured['exploration_score']
    assert dropped['bounded_position_cost_interval'][1]>=measured['bounded_position_cost_interval'][1]
    assert dropped['bounded_velocity_cost_interval'][1]>=measured['bounded_velocity_cost_interval'][1]
    assert dropped['coverage']<measured['coverage'] and not dropped['missing_cost_bounds_are_observations']
    assert dropped['ranking_basis']=='fixed_requested_joint_time_upper_cost_not_observed_subset_mean'


def test_one_articulated_snapshot_is_not_a_measured_moving_trajectory():
    p={0:(.4,.4),1:(.6,.4),2:(.3,.6),3:(.7,.6),4:(.2,.7),5:(.8,.7)}
    rows=[{}, {},p,{},{}]
    r=measure_review_partial_joint_trajectory(rows,rows,ticks=(10,11,12,13,14),fps=30.)
    assert r['exploration_score'] is None and r['state']=='UNMEASURED'
    assert r['measured_articulated_velocity_requests']==0
    assert r['exploratory_cost_interval'][1]>0 and not r['quality_approved']


def test_complete_same_motion_has_zero_cost_interval_not_unknown_reward():
    rows=[{0:(.4,.4),1:(.6,.4),2:(.3,.6),3:(.7,.6),4:(.2+.01*i,.7),5:(.8-.01*i,.7)} for i in range(5)]
    r=measure_review_partial_joint_trajectory(rows,rows,ticks=(10,11,12,13,14),fps=30.)
    assert r['exploratory_cost_interval']==pytest.approx((0.,0.))
    assert r['coverage']==1 and r['nonstationary_common_action_observed']
    assert not r['full_action_certified'] and not r['actor_identity_certified']


def test_equal_detector_boxes_do_not_override_twofold_actual_facial_landmark_scale():
    points=((.48,.43),(.52,.43),(.5,.45),(.485,.47),(.515,.47))
    a=dict(center=(.5,.45),width=.12,height=.20,landmarks_xy=points)
    b=dict(a,landmarks_xy=tuple((.5+2*(x-.5),.45+2*(y-.45)) for x,y in points))
    pairs=fit_review_coupled_boundary_keys(left_head=a,right_head=b,left_joint_points={},right_joint_points={})
    assert pairs and pairs[0][0].zoom/pairs[0][1].zoom==pytest.approx(2.,rel=.02)
    receipt=pairs[0][2]
    assert receipt['actual_facial_landmarks_used']
    assert receipt['face_scale_basis']=='projected_five_anatomical_landmarks_rms'
    assert receipt['log_face_scale_error']<.02 and not receipt['quality_approved']
    assert not receipt['detector_box_height_is_verified_facial_size']


def test_absent_or_malformed_facial_landmarks_are_not_a_verified_face_size():
    a=dict(center=(.5,.45),width=.12,height=.20)
    pairs=fit_review_coupled_boundary_keys(left_head=a,right_head=a,left_joint_points={},right_joint_points={})
    assert pairs[0][2]['face_scale_basis']=='detector_box_height_proxy'
    assert not pairs[0][2]['actual_facial_landmarks_used']
    with pytest.raises(ValueError,match='five actually projected'):
        fit_review_coupled_boundary_keys(left_head=dict(a,landmarks_xy=((.5,.4),)),right_head=a,
            left_joint_points={},right_joint_points={})


def test_coupled_focal_destination_is_optimized_instead_of_forcing_dead_center():
    a=dict(center=(.46856,.71637),width=.09,height=.16212)
    b=dict(center=(.70774,.68049),width=.08,height=.14114)
    pa={0:(.56548,.87192),1:(.43520,.83883),3:(.39798,.99008),5:(.33289,.87419)}
    pb={0:(.74509,.82412),1:(.63716,.81674),2:(.75505,.96728),3:(.60146,.98942),5:(.58071,.84921)}
    def fitted(**extra):
        return fit_review_coupled_boundary_keys(left_head=a,right_head=b,left_joint_points=pa,right_joint_points=pb,
            left_required_visible_points=tuple(pa.values()),right_required_visible_points=tuple(pb.values()),**extra)
    old=fitted(target_centers=((.5,.5),));new=fitted()
    assert old and new and new[0][2]['geometry_score']<=old[0][2]['geometry_score']
    assert not new[0][2]['shared_target_must_be_dead_center']
    assert new[0][2]['camera_pair_count']<=new[0][2]['maximum_geometric_camera_pairs']
    for ka,kb,receipt in new:
        assert all(all(0<=v<=1 for v in project_review_editorial_point(p,k))
            for k,points in ((ka,pa),(kb,pb)) for p in points.values())


def test_face_anchor_family_does_not_let_viewpoint_changed_arms_drag_the_face_match():
    a=dict(center=(.51,.61),width=.09,height=.172)
    b=dict(center=(.71,.68),width=.08,height=.141)
    pa={0:(.606,.726),1:(.470,.739),2:(.632,.951),3:(.409,.935)}
    pb={0:(.745,.824),1:(.637,.817),2:(.755,.967),3:(.601,.989),5:(.581,.849)}
    def fit(**extra):
        return fit_review_coupled_boundary_keys(left_head=a,right_head=b,left_joint_points=pa,right_joint_points=pb,
            left_required_visible_points=(),right_required_visible_points=(pb[5],),**extra)
    old=fit(camera_fit_bases=('face_and_body',));new=fit()
    assert old and new and new[0][2]['geometry_score']<=old[0][2]['geometry_score']
    assert new[0][2]['projected_head_center_error']<.01
    assert new[0][2]['measured_common_joint_indices']==[0,1,2,3]
    assert new[0][2]['missing_joint_requests']=={'left':[4,5],'right':[4]}
    assert not new[0][2]['body_requests_removed_to_fit_face']
    assert not new[0][2]['quality_approved']
    for _,right,receipt in new:
        assert all(0<=v<=1 for v in project_review_editorial_point(pb[5],right))
        assert receipt['camera_pair_count']<=receipt['maximum_geometric_camera_pairs']


def test_coupled_family_does_not_spend_its_recipe_budget_on_float_noise_duplicates():
    options=fit_review_coupled_boundary_keys(left_head=dict(center=(.51,.61),width=.09,height=.172),
        right_head=dict(center=(.71,.68),width=.08,height=.141),left_joint_points={},right_joint_points={})
    tokens=[tuple(round(v,10) for k in (a,b) for v in (k.zoom,k.center_x,k.center_y,k.roll_degrees))
        for a,b,_ in options]
    assert len(tokens)==len(set(tokens))
