from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import pytest
from crosscut.group_review_continuous_shot import (ContinuousReviewShotScope,continuous_review_native_runs,
    measure_review_added_camera_motion,joint_review_shot_measurements,select_continuous_review_shot,
    classify_review_continuation_incident,enumerate_continuous_review_intervals,ReviewCameraRealizationBudget)
from crosscut.render import EditorialTransformKeyframe as Key
from crosscut.render import RenderPlan, RenderSegment, SourceLookAdjustment, EditorialTransformTrack
from crosscut.render_frame_schedule import VerifiedRenderFrameClock


def test_continuous_shot_rejects_three_frame_flash_and_freezes_two_contexts():
    with pytest.raises(ValueError,match='hold'):ContinuousReviewShotScope(50,53,200)
    s=ContinuousReviewShotScope(50,80,200)
    assert s.incidents==(50,80) and s.comparison_context==(38,92)
    assert s.changed_ticks==tuple(range(50,80))
    assert s.measurement_ticks==(48,49,50,51,52,78,79,80,81,82)
    assert not s.to_dict()['duration_is_quality_certificate']


def test_native_runs_exist_without_actor_coverage_and_split_actual_physical_cuts():
    rows={t:(0,'clock','file',0,t) for t in range(40,90)}
    rows[60]=(0,'clock','file',1,60)
    for t in range(61,90):rows[t]=(0,'clock','file',1,t)
    result=continuous_review_native_runs(rows)
    assert [(r['first_tick'],r['end_tick']) for r in result]==[(40,60),(60,90)]
    assert all(not r['actor_correspondence_certified'] and not r['known_joint_duties_waived'] for r in result)


def test_native_run_cannot_bridge_missing_addresses_or_reverse_frame_clock():
    rows={0:(0,'clock','file',0,10),1:None,2:(0,'clock','file',0,11),3:(0,'clock','file',0,2)}
    assert [(r['first_tick'],r['end_tick']) for r in continuous_review_native_runs(rows)]==[(0,1),(2,3),(3,4)]


def test_physical_subinterval_is_generated_without_actor_or_body_observations():
    rows={t:(0,'clock','file',0 if t<2715 else 1,t) for t in range(2664,2724)}
    result=enumerate_continuous_review_intervals(runs_by_source={0:continuous_review_native_runs(rows)},
        independent_boundaries=(2664,2724),output_frame_count=4357)
    assert result['state']=='COMPLETE' and len(result['hypotheses'])==1
    scope=result['hypotheses'][0]['scope']
    assert (scope['entry_tick'],scope['return_tick'])==(2664,2715)
    assert not result['hypotheses'][0]['actor_correspondence_certified']
    assert not result['pose_costs_used_to_choose_boundaries']


def test_interval_budget_defers_fairly_instead_of_losing_earlier_hypotheses():
    runs={s:continuous_review_native_runs({t:(s,'clock','file',0,t) for t in range(50,100)}) for s in range(3)}
    result=enumerate_continuous_review_intervals(runs_by_source=runs,independent_boundaries=(50,70,100),
        output_frame_count=200,maximum_returned=2)
    assert result['state']=='BUDGET_LIMITED' and [r['source'] for r in result['hypotheses']]==[0,1]
    assert result['deferred'] and result['total_physical_hypotheses']==9


def test_camera_recipe_budget_counts_unique_cameras_not_action_schemas():
    budget=ReviewCameraRealizationBudget(56)
    for i in range(56):
        for _schema in range(3):assert budget.reserve(dict(camera=i,source=0))
    assert not budget.reserve(dict(camera=56,source=0))
    assert budget.reserve(dict(camera=0,source=0))
    receipt=budget.receipt()
    assert receipt['state']=='BUDGET_LIMITED' and len(receipt['reserved_recipe_identities'])==56
    assert len(receipt['deferred_recipe_identities'])==1 and not receipt['pre_existing_candidates_erased']


def test_different_constant_cameras_and_original_return_are_measured():
    samples={t:Key(0.,1.8 if 50<=t<80 else 1.) for t in range(48,83)}
    r=measure_review_added_camera_motion(samples,fps=30.,boundary_ticks=(50,80))
    assert r['maximum_within_shot_velocity']==0
    assert r['maximum_boundary_transform_change']>0
    assert all(x['transform_change_norm']>0 for x in r['boundaries'])
    assert not r['body_motion_velocity_penalized']
    samples.pop(80)
    with pytest.raises(ValueError,match='return'):measure_review_added_camera_motion(samples,fps=30.,boundary_ticks=(50,80))


def test_same_added_transform_over_fast_and_slow_clocks_has_different_rates():
    a={t:Key(0.,1.+.01*t) for t in range(10)}
    b={10*t:k for t,k in a.items()}
    b[89]=Key(0.,1.089)
    fast=measure_review_added_camera_motion(a,fps=30.,boundary_ticks=(9,))
    slow=measure_review_added_camera_motion(b,fps=30.,boundary_ticks=(90,))
    assert fast['maximum_within_shot_velocity']==pytest.approx(10*slow['maximum_within_shot_velocity'])


def candidate(head_scale=.01,*,state='MEASURED',benefit=True,program='a'*64):
    keys=('native_action','projected_position','projected_velocity','head_center','head_scale','head_velocity')
    rows=[dict(boundary_tick=t,**{k:.01 for k in keys}) for t in (50,80)]
    for r in rows:r['head_scale']=head_scale
    return dict(measurement=dict(state=state,program_identity=program,incidents=rows,
        camera_motion=dict(maximum_within_shot_velocity=0.,maximum_within_shot_acceleration=0.,maximum_boundary_transform_change=0.)),
        predeclared_editorial_benefit=benefit)


def limits():return {k:1. for k in ('native_action','projected_position','projected_velocity','head_center','head_scale','head_velocity',
    'added_camera_velocity','added_camera_acceleration','boundary_transform_change')}


def test_same_native_motion_different_head_framing_changes_joint_rank():
    bad=candidate(.8,program='b'*64);good=candidate(.02)
    r=select_continuous_review_shot([bad,good],independent_acceptable_limits=limits(),limits_calibrated=True)
    assert r['selected']==good and r['result']=='SELECTED_REVIEW_HYPOTHESIS'
    assert not r['quality_approved']


def test_unknown_unacceptable_or_uncalibrated_candidates_keep_original():
    for candidates,calibrated in (([candidate(state='UNKNOWN')],True),([candidate(2.)],True),
            ([candidate()],False),([candidate(benefit=False)],True),([],True)):
        r=select_continuous_review_shot(candidates,independent_acceptable_limits=limits(),limits_calibrated=calibrated)
        assert r['result']=='KEEP_ORIGINAL' and not r['forced_technical_winner']


def test_joint_measurement_rejects_different_camera_and_missing_return():
    s=ContinuousReviewShotScope(50,80,200)
    c=measure_review_added_camera_motion({t:Key(0.,1.) for t in range(48,83)},fps=30.,boundary_ticks=(50,80))
    rows=[]
    for t in s.incidents:
        rows.append(dict(boundary_tick=t,program_identity='a'*64,
            native_action=dict(route_identity='route',same_time_action_supported=True,exploration_cost=.1),
            projected_action=dict(route_identity='route',state='MEASURED',projected_position_error=.1,
                projected_velocity_error=.1,actual_projected_inputs=[dict(side='left',tick=t,registered_query_identity='camera-a',
                    native_parent='raw',selected_head=10,native_head_identity='head')]),
            head_connection=dict(route_identity='route',state='MEASURED',face_center_position_error=.1,
                face_scale_log_error=.1,face_center_velocity_error=.1,
                actual_inputs=[dict(side='left',tick=t,face=dict(registered_query_identity='camera-a',
                    native_parent='raw',selected_head=10,native_head_identity='head'))])))
    assert joint_review_shot_measurements(scope=s,program_identity='a'*64,incidents=rows,camera_motion=c,
        known_joint_state='PASS',actor_state='PASS')['state']=='MEASURED'
    wrong=deepcopy(rows);wrong[1]['head_connection']['actual_inputs'][0]['face']['registered_query_identity']='camera-b'
    with pytest.raises(ValueError,match='different actual cameras'):
        joint_review_shot_measurements(scope=s,program_identity='a'*64,incidents=wrong,camera_motion=c,known_joint_state='PASS',actor_state='PASS')
    with pytest.raises(ValueError,match='both declared incidents'):
        joint_review_shot_measurements(scope=s,program_identity='a'*64,incidents=rows[:1],camera_motion=c,known_joint_state='PASS',actor_state='PASS')
    for field,value in (('native_parent','other-raw'),('selected_head',20),('native_head_identity','other-head')):
        wrong=deepcopy(rows);wrong[0]['head_connection']['actual_inputs'][0]['face'][field]=value
        with pytest.raises(ValueError,match='actor/raw'):
            joint_review_shot_measurements(scope=s,program_identity='a'*64,incidents=wrong,camera_motion=c,
                known_joint_state='PASS',actor_state='PASS')
    wrong=deepcopy(rows);wrong[0]['head_connection']['actual_inputs']*=2
    with pytest.raises(ValueError,match='reused'):
        joint_review_shot_measurements(scope=s,program_identity='a'*64,incidents=wrong,camera_motion=c,
            known_joint_state='PASS',actor_state='PASS')


def test_actual_original_camera_return_is_consumed_not_just_reported():
    stable=candidate(program='b'*64);jump=candidate(program='a'*64)
    jump['measurement']['camera_motion']['maximum_boundary_transform_change']=.9
    r=select_continuous_review_shot([jump,stable],independent_acceptable_limits=limits(),limits_calibrated=True)
    assert r['selected']==stable
    jump['measurement']['camera_motion']['maximum_boundary_transform_change']=2.
    assert select_continuous_review_shot([jump],independent_acceptable_limits=limits(),limits_calibrated=True)['result']=='KEEP_ORIGINAL'


def continuation_fixture():
    p=RenderPlan((Path('a.mp4'),Path('b.mp4')),(RenderSegment(0,0.,3.,0.),),0,'16:9',Path('unused.mp4'),fps=30.)
    clocks={s:VerifiedRenderFrameClock(Fraction(30),300,str(s)*64) for s in (0,1)}
    native={t:(0,'clock','file',0,t) for t in range(48,53)}
    return p,clocks,native


def classify(p,clocks,native,**kwargs):
    return classify_review_continuation_incident(candidate=kwargs.pop('candidate',p),continuation=p,
        source_clocks=clocks,candidate_native_signatures=kwargs.pop('actual',native),
        continuation_native_signatures=native,boundary_tick=50,**kwargs)


def test_same_native_render_continuation_does_not_invent_unknown_body_measurements():
    p,clocks,native=continuation_fixture();r=classify(p,clocks,native)
    assert r['kind']=='EXACT_NATIVE_RENDER_CONTINUATION'
    assert not r['new_cross_performance_action_comparison_required']
    assert r['anatomical_measurement_state']=='NOT_PROMOTED'
    assert not r['actor_correspondence_certified'] and not r['known_joint_duties_waived']
    assert not r['candidate_admitted'] and not r['quality_approved']


def test_same_native_new_grade_and_camera_remain_separate_visible_interventions():
    p,clocks,native=continuation_fixture()
    grade=replace(p,segments=(replace(p.segments[0],look_adjustment=SourceLookAdjustment(brightness=.03)),))
    r=classify(p,clocks,native,candidate=grade)
    assert r['kind']=='SAME_NATIVE_FRAMING_WITH_LOOK_CHANGE' and r['look_change_review_required']
    camera=replace(p,segments=(replace(p.segments[0],editorial_transform=EditorialTransformTrack((Key(0.,1.3),))),))
    r=classify(p,clocks,native,candidate=camera)
    assert r['kind']=='SAME_NATIVE_WITH_NEW_FRAMING' and r['projected_layout_comparison_required']
    assert not r['new_cross_performance_action_comparison_required']


def test_same_index_different_path_sampler_camera_unit_or_missing_slot_cannot_claim_continuation():
    p,clocks,native=continuation_fixture()
    for candidate in (replace(p,sources=(Path('different-a.mp4'),p.sources[1])),
            replace(p,segments=(replace(p.segments[0],source_start=1/30),))):
        assert classify(p,clocks,native,candidate=candidate)['kind']=='NEW_NATIVE_COMPARISON_REQUIRED'
    wrong=dict(native);wrong[50]=(0,'clock','file',1,50)
    assert classify(p,clocks,native,actual=wrong)['kind']=='NEW_NATIVE_COMPARISON_REQUIRED'
    wrong=dict(native);wrong[50]=None
    assert classify(p,clocks,native,actual=wrong)['kind']=='NEW_NATIVE_COMPARISON_REQUIRED'
    wrong=dict(native);wrong.pop(50)
    with pytest.raises(ValueError,match='omitted'):classify(p,clocks,native,actual=wrong)
