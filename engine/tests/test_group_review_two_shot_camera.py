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
