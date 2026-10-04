from copy import deepcopy
from dataclasses import replace
import pytest

from crosscut.group_review_motion_contract import (freeze_selected_review_motion_intent,
    freeze_review_native_motion_route,compare_frozen_review_motion_route,same_native_review_continuation)
from crosscut.group_review_motion_seam import compare_aligned_review_motion_windows
from crosscut.group_review_camera_repair import FramingDescriptor


def view(tick,phase):
    vector=[1.]+[0.]*127
    d=FramingDescriptor((.5,.35),.3,.15,tuple((k,k*.1,phase) for k in range(4)),7,True)
    return dict(row=dict(output_tick=tick,native_observation=dict(request_source_index=2,
        native_frame_index=tick,source_clock_identity='clock',source_file_identity=[1,2,3,4,5]),
        projection=dict(camera_sha256='camera',heads=[dict(observation_id=7,
            projected_landmarks_xy=[[.45,.3],[.55,.3],[.5,.36],[.46,.4],[.54,.4]])])),
        descriptor=d,vector=vector,near=1.,appearance=dict(heads=[dict(observation_id=7,embedding=vector)]))


TICKS=(8,9,10,11,12)


def native(units=None):
    units=(0,)*5 if units is None else units
    return [(2,'clock','file',u,t) for t,u in zip(TICKS,units)]


def route(a=None,b=None):
    return freeze_review_native_motion_route(ticks=TICKS,boundary_tick=10,
        left_native=native(a),right_native=native(b))


def intent(a,head=False):
    return freeze_selected_review_motion_intent(a,requested_ticks=TICKS,
        original_actor_identity='frozen-original-actor',head_only_declared=head)


def compare(a,b,r=None):
    return compare_frozen_review_motion_route(a,b,fps=30.,intent=intent(a),route=route() if r is None else r)


def test_non_output_future_native_cut_uses_fixed_past_and_incoming_continuation():
    a=[view(t,t*.1) for t in TICKS];b=deepcopy(a)
    for i in (2,3,4):a[i]['row']['projection']['camera_sha256']='another-native-shot'
    r=route((0,0,1,1,1))
    assert r['route']=='PAST_COMMON_INCOMING_CONTINUES'
    assert compare(a,b,r)['same_time_action_supported']
    assert compare(a,b,r)['exploration_cost']==pytest.approx(0.)


def test_route_is_not_shortened_because_a_bad_wrist_is_missing():
    a=[view(t,t*.1) for t in TICKS];b=deepcopy(a)
    b[0]['descriptor']=replace(b[0]['descriptor'],pose_shape=b[0]['descriptor'].pose_shape[:-1])
    r=route((0,0,1,1,1))
    result=compare(a,b,r)
    assert result['state']=='INCONCLUSIVE' and result['evidence_loss_prevents_head_fallback']
    assert r['comparison_indices']==[0,1]


def test_same_fast_natural_turn_has_zero_common_time_discrepancy():
    a=[view(t,p) for t,p in zip(TICKS,(0.,3.,-2.,5.,0.))]
    result=compare(a,deepcopy(a))
    assert result['exploration_cost']==pytest.approx(0.)
    assert not result['actual_cut_tangent_penalizes_natural_turn']
    b=[view(t,p) for t,p in zip(TICKS,(3.,-2.,5.,0.,0.))]
    assert compare(a,b)['exploration_cost']>0.


def test_original_transient_pose_loss_cannot_switch_known_body_intent_to_head():
    a=[view(t,t*.1) for t in TICKS]
    a[2]['descriptor']=replace(a[2]['descriptor'],pose_shape=None)
    frozen=intent(a,True)
    assert frozen.payload['kind']=='BODY'
    assert frozen.payload['joint_requests'][2]==[0,1,2,3]
    b=[view(t,0.) for t in TICKS]
    for v in b:v['descriptor']=replace(v['descriptor'],pose_shape=None)
    result=compare_frozen_review_motion_route(b,b,fps=30.,intent=frozen,route=route())
    assert result['state']=='INCONCLUSIVE' and result['intent_kind']=='BODY'


def test_part_visibility_is_per_tick_not_whole_arm_for_whole_window():
    a=[view(t,t*.1) for t in TICKS]
    a[0]['descriptor']=replace(a[0]['descriptor'],pose_shape=(*a[0]['descriptor'].pose_shape,(5,.5,.8)))
    frozen=intent(a)
    assert frozen.payload['joint_requests'][0]==[0,1,2,3,5]
    assert frozen.payload['joint_requests'][1]==[0,1,2,3]
    assert compare(a,deepcopy(a))['same_time_action_supported']


def test_selected_middle_reacquisition_ambiguity_cannot_certify_equal_pose():
    a=[view(t,t*.1) for t in TICKS];b=deepcopy(a)
    b[2]['reacquisition_ambiguous']=True
    assert compare(a,b)['state']=='INCONCLUSIVE'
    assert compare_aligned_review_motion_windows(a,b,fps=30.,requested_output_ticks=TICKS,
        requested_joint_indices=(0,1,2,3))['state']=='INCONCLUSIVE'


def test_background_inventory_truncation_is_not_selected_actor_ambiguity():
    a=[view(t,t*.1) for t in TICKS]
    for v in a:v['background_head_inventory_truncated']=True
    assert compare(a,deepcopy(a))['same_time_action_supported']


def test_repeated_native_frame_does_not_create_body_motion_observation():
    aliases=[(2,'clock','file',0,8)]*5
    r=freeze_review_native_motion_route(ticks=TICKS,boundary_tick=10,
        left_native=aliases,right_native=aliases)
    assert r['route']=='ACTUAL_ONLY_UNRESOLVED'


def test_actor_loss_in_crossing_continuation_is_not_hidden_by_past_match():
    a=[view(t,t*.1) for t in TICKS];b=deepcopy(a);b[3]=None
    result=compare(a,b,route((0,0,1,1,1)))
    assert result['state']=='INCONCLUSIVE' and result['unresolved_crossing_continuation']=='right'


@pytest.mark.parametrize('lost_joint',[None,3])
def test_past_match_requires_body_on_actual_incoming_future(lost_joint):
    a=[view(t,t*.1) for t in TICKS];b=deepcopy(a)
    for i in (2,3,4):
        shape=None if lost_joint is None else tuple(p for p in b[i]['descriptor'].pose_shape if p[0]!=lost_joint)
        b[i]['descriptor']=replace(b[i]['descriptor'],pose_shape=shape)
    result=compare(a,b,route((0,0,1,1,1)))
    assert not result['same_time_action_supported']
    assert result['missing_actual_output_joint_requests']


def test_future_match_requires_body_on_actual_outgoing_past():
    a=[view(t,t*.1) for t in TICKS];b=deepcopy(a)
    frozen=intent(a)
    for i in (0,1):a[i]['descriptor']=replace(a[i]['descriptor'],pose_shape=None)
    r=route(b=(0,0,1,1,1))
    assert r['route']=='FUTURE_COMMON_OUTGOING_CONTINUES'
    result=compare_frozen_review_motion_route(a,b,fps=30.,intent=frozen,route=r)
    assert not result['same_time_action_supported'] and result['missing_actual_output_joint_requests']


def test_body_in_a_different_original_shot_does_not_invent_a_past_visible_wrist():
    a=[view(t,t*.1) for t in TICKS]
    for v in a[:2]:v['descriptor']=replace(v['descriptor'],pose_shape=None)
    frozen=freeze_selected_review_motion_intent(a,requested_ticks=TICKS,original_actor_identity='original',
        reference_native_units=[('head-shot',)]*2+[('body-shot',)]*3)
    assert frozen.payload['joint_requests'][:2]==[[],[]]
    assert frozen.payload['kind']=='BODY'
    a[3]['descriptor']=replace(a[3]['descriptor'],pose_shape=None)
    frozen=freeze_selected_review_motion_intent(a,requested_ticks=TICKS,original_actor_identity='original',
        reference_native_units=[('head-shot',)]*2+[('body-shot',)]*3)
    assert frozen.payload['joint_requests'][3]==[0,1,2,3]


def test_same_native_continuation_survives_missing_body_but_not_wrong_sampler_or_actor():
    a=[view(t,t*.1) for t in TICKS]
    a[2]['descriptor']=replace(a[2]['descriptor'],pose_shape=None)
    expected=[(2,'clock',(1,2,3,4,5),t) for t in TICKS]
    result=same_native_review_continuation(a[:2],a[2:],expected_native_addresses=expected)
    assert result['continuity_supported'] and not result['motion_improvement_claimed']
    wrong=list(expected);wrong[2]=(*expected[2][:-1],100)
    assert not same_native_review_continuation(a[:2],a[2:],expected_native_addresses=wrong)['continuity_supported']
    a[3]['reacquisition_ambiguous']=True
    assert not same_native_review_continuation(a[:2],a[2:],expected_native_addresses=expected)['continuity_supported']
