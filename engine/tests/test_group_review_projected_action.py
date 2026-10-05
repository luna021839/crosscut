from copy import deepcopy
from dataclasses import replace
import pytest
from engine.tests.test_scene_owned_program import fixture
from crosscut.group_scene_owned_program import freeze_owned_scene_program
from crosscut.group_scene_epoch_program import compile_common_clock_camera_queries
from crosscut.group_review_camera_repair import FramingDescriptor
from crosscut.group_review_motion_contract import freeze_selected_review_motion_intent,freeze_review_native_motion_route
from crosscut.group_review_declared_action import freeze_review_action_declaration
from crosscut.group_review_projected_action import (project_observed_review_joints,measure_review_projected_action,
    fit_review_anatomical_key,project_observed_review_face,measure_review_projected_face)
from crosscut.group_review_camera_repair import project_review_editorial_point
from crosscut.group_performer_family_owner import digest
from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe

TICKS=(8,9,10,11,12)

def setup(zoom=1.):
    context,domain=fixture();plan=context.incumbent.render_plan;original=freeze_owned_scene_program(context,domain)
    camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,zoom),EditorialTransformKeyframe(1.,zoom)),reason='predeclared')
    candidate=replace(plan,segments=(replace(plan.segments[0],editorial_transform=camera),*plan.segments[1:]))
    queries=compile_common_clock_camera_queries(context=context,domain=domain,original_program=original,
        candidate_plan=candidate,output_intervals=((0,30),)).queries
    mapping={t:q for q in queries for t in range(q.first_output_frame,q.end_output_frame)}
    views=[];signatures=[]
    for t in TICKS:
        q=mapping[t];native=q.sample_at(t)[2]
        raw=dict(request_source_index=0,source_clock_identity=q.source_clock_identity,
            source_file_identity=[1,2,3,4,5],native_frame_index=native)
        provenance={j:dict(raw_parent_sha256=digest(raw),basis='PRIMARY_NATIVE_MODEL',model_confidence=.9,
            xy_native=[.45+(j%2)*.1,.4+(j//2)*.05+t*.002]) for j in range(4)}
        v=dict(native_camera_unit_index=0,row=dict(output_tick=t,native_observation=raw,
            query=dict(registered_query_sha256=q.key),projection=dict(camera_sha256=q.camera.key)),
            descriptor=FramingDescriptor((.5,.3),.2,.1,tuple((j,.1*j,t*.1) for j in range(4)),0,True),
            review_joint_provenance=provenance,review_joint_evidence_identity=digest(provenance))
        views.append(v);signatures.append((0,q.source_clock_identity,digest(raw['source_file_identity']),0,native))
    intent=freeze_selected_review_motion_intent(views,requested_ticks=TICKS,original_actor_identity='frozen')
    declaration=freeze_review_action_declaration(schema='bilateral_upper_arm_v1',ticks=TICKS,
        original_motion_intent=intent,original_preservation_duty_identity='a'*64,declared_scope_identity='b'*64)
    route=freeze_review_native_motion_route(ticks=TICKS,boundary_tick=10,left_native=signatures,right_native=signatures)
    return views,{t:mapping[t] for t in TICKS},declaration,route


def test_actual_zoom_changes_projected_anatomical_cost_while_normalized_pose_is_identical():
    a,qa,declaration,route=setup();b,qb,_,_=setup(1.5)
    r=measure_review_projected_action(a,b,left_queries=qa,right_queries=qb,fps=30.,declaration=declaration,route=route)
    assert r['state']=='MEASURED' and r['projected_action_error']>0.
    identical=measure_review_projected_action(a,a,left_queries=qa,right_queries=qa,fps=30.,declaration=declaration,route=route)
    assert identical['projected_action_error']==pytest.approx(0.)
    assert identical['measured_velocity_windows']==4


def test_projected_anatomy_needs_actual_native_provenance_not_normalized_pose_only():
    a,qa,declaration,route=setup();b=deepcopy(a);b[2]['review_joint_provenance'].pop(3)
    r=measure_review_projected_action(a,b,left_queries=qa,right_queries=qa,fps=30.,declaration=declaration,route=route)
    assert r['state']=='UNKNOWN' and r['projected_action_error'] is None


@pytest.mark.parametrize('field,value',(('raw_parent_sha256','c'*64),('model_confidence',.1)))
def test_native_anatomical_parent_or_confidence_cannot_be_borrowed(field,value):
    a,qa,_,_=setup();a[0]['review_joint_provenance'][3][field]=value
    with pytest.raises(ValueError,match='provenance'):project_observed_review_joints(a[0],qa[8],8)


def test_pose_query_cannot_use_old_neutral_view_under_new_camera():
    a,qa,_,_=setup();_,qb,_,_=setup(1.5)
    with pytest.raises(ValueError,match='actual camera'):project_observed_review_joints(a[0],qb[8],8)


def test_screen_joint_fit_uses_actual_anatomy_not_only_face_center_and_size():
    points={0:(.4,.45),1:(.6,.45),2:(.35,.65),3:(.65,.65)}
    expected=EditorialTransformKeyframe(0.,1.4,.52,.48,1.3)
    target={j:project_review_editorial_point(xy,expected) for j,xy in points.items()}
    key=fit_review_anatomical_key(points,target,requested_joints=(0,1,2,3),time_seconds=0.)
    for j in points:assert project_review_editorial_point(points[j],key)==pytest.approx(target[j])
    missing=dict(points);missing.pop(3)
    assert fit_review_anatomical_key(missing,target,requested_joints=(0,1,2,3),time_seconds=0.) is None


def test_infeasible_unconstrained_roll_does_not_discard_measured_joint_request():
    points={0:(.4,.45),1:(.6,.45),2:(.35,.65),3:(.65,.65)}
    # Rotation at unit zoom would expose corners. A legal constrained fit is
    # still measured, and no joint is dropped to manufacture a zero error.
    import math
    a=math.radians(1.6);c,s=math.cos(a),math.sin(a)
    target={j:(.5+c*(x-.5)-s*(y-.5)*9/16,.5+s*(x-.5)*16/9+c*(y-.5))
        for j,(x,y) in points.items()}
    key=fit_review_anatomical_key(points,target,requested_joints=(0,1,2,3),time_seconds=0.)
    assert key is not None
    before=sum(math.dist(points[j],target[j])**2 for j in points)
    after=sum(math.dist(project_review_editorial_point(points[j],key),target[j])**2 for j in points)
    assert after<=before+1e-12


def faces(zoom=1.):
    from crosscut.group_source_changing_owner import project_box
    views,queries,declaration,route=setup(zoom)
    for t,v in zip(TICKS,views):
        box=(.44,.27+t*.001,.12,.20)
        v['row']['native_observation']['heads']=[dict(observation_id=0,box_xywh=box)]
        projected,fraction=project_box(box,queries[t],t)
        v['row']['projection']['heads']=[dict(observation_id=0,projected_box_xywh=projected,visible_fraction=fraction)]
    return views,queries,route


def test_actual_face_scale_and_position_are_separate_from_normalized_body_pose():
    a,qa,route=faces();b,qb,_=faces(1.5)
    result=measure_review_projected_face(a,b,left_queries=qa,right_queries=qb,fps=30.,route=route)
    import math
    assert result['state']=='MEASURED'
    assert result['face_scale_log_error']==pytest.approx(math.log(1.5))
    assert result['face_center_position_error']>0
    assert result['measured_position_samples']==5 and result['measured_velocity_windows']==4
    assert not result['quality_approved'] and not result['actor_correspondence_certified']
    same=measure_review_projected_face(a,a,left_queries=qa,right_queries=qa,fps=30.,route=route)
    assert same['face_scale_log_error']==0 and same['face_center_velocity_error']==0


def test_face_request_cannot_drop_missing_middle_or_ambiguous_actor():
    a,qa,route=faces();b=deepcopy(a);b[2]['row']['projection']['heads']=[]
    result=measure_review_projected_face(a,b,left_queries=qa,right_queries=qa,fps=30.,route=route)
    assert result['state']=='UNKNOWN' and result['face_scale_log_error'] is None
    b=deepcopy(a);b[2]['reacquisition_ambiguous']=True
    assert measure_review_projected_face(a,b,left_queries=qa,right_queries=qa,fps=30.,route=route)['state']=='UNKNOWN'


def test_face_descriptor_cannot_supply_an_unobserved_smaller_scale():
    a,qa,route=faces();b=deepcopy(a)
    # Scores are reconstructed from the actual raw head, not this DTO field.
    b[2]['descriptor']=replace(b[2]['descriptor'],face_height=.001,head_center=(.9,.9))
    result=measure_review_projected_face(a,b,left_queries=qa,right_queries=qa,fps=30.,route=route)
    assert result['face_scale_log_error']==0 and result['face_center_position_error']==0


def test_forged_projected_face_box_or_neutral_camera_cannot_be_consumed():
    a,qa,_=faces();b=deepcopy(a)
    box=list(b[0]['row']['projection']['heads'][0]['projected_box_xywh']);box[3]*=2
    b[0]['row']['projection']['heads'][0]['projected_box_xywh']=box
    with pytest.raises(ValueError,match='actual raw'):project_observed_review_face(b[0],qa[8],8)
    _,qb,_=faces(1.5)
    with pytest.raises(ValueError,match='actual camera'):project_observed_review_face(a[0],qb[8],8)


def test_face_measurement_keeps_actual_native_unit_and_output_clock():
    a,qa,route=faces();b=deepcopy(a);b[2]['native_camera_unit_index']=99
    with pytest.raises(ValueError,match='frozen native route'):
        measure_review_projected_face(a,b,left_queries=qa,right_queries=qa,fps=30.,route=route)
    with pytest.raises(ValueError,match='output clock'):
        measure_review_projected_face(a,a,left_queries=qa,right_queries=qa,fps=29.97,route=route)
