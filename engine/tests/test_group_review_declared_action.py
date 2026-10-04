from copy import deepcopy
from dataclasses import replace
import json
import pytest
from crosscut.group_review_declared_action import (freeze_review_action_declaration,
    compare_declared_review_action_route,FrozenReviewActionDeclaration)
from crosscut.group_review_motion_contract import freeze_selected_review_motion_intent,freeze_review_native_motion_route
from crosscut.group_review_camera_repair import FramingDescriptor

TICKS=(8,9,10,11,12)

def view(t,shape=True):
    d=FramingDescriptor((.5,.35),.3,.15,tuple((k,k*.1,t*.1) for k in (0,1,3,5)) if shape else None,7,True)
    return dict(row=dict(output_tick=t,native_observation=dict(request_source_index=2,native_frame_index=t,
        source_clock_identity='clock',source_file_identity=[1,2,3,4,5]),
        projection=dict(camera_sha256='camera')),descriptor=d,vector=[1.]+[0.]*127)

def setup():
    old=freeze_selected_review_motion_intent([None]*5,requested_ticks=TICKS,original_actor_identity='actual-old')
    duty=freeze_review_action_declaration(schema='right_arm_with_shoulders_v1',ticks=TICKS,
        original_motion_intent=old,original_preservation_duty_identity='a'*64,declared_scope_identity='b'*64)
    raw=[(2,'clock','file',0,t) for t in TICKS]
    route=freeze_review_native_motion_route(ticks=TICKS,boundary_tick=10,left_native=raw,right_native=raw)
    return old,duty,route

def test_unknown_original_does_not_prevent_separate_candidate_measurement_or_become_a_pass():
    old,duty,route=setup();a=[view(t) for t in TICKS]
    result=compare_declared_review_action_route(a,deepcopy(a),fps=30.,declaration=duty,route=route,
        original_motion_intent=old)
    assert result['same_time_action_supported'] and result['exploration_cost']==pytest.approx(0.)
    assert old.payload['kind']=='UNKNOWN' and result['original_intent_kind']=='UNKNOWN'
    assert not result['original_motion_state_promoted'] and not result['original_known_duties_waived']
    assert not result['adoption_permitted'] and not result['review_encode_permitted']

def test_missing_declared_wrist_cannot_become_a_smaller_successful_intersection():
    old,duty,route=setup();a=[view(t) for t in TICKS];b=deepcopy(a)
    b[2]['descriptor']=replace(b[2]['descriptor'],pose_shape=b[2]['descriptor'].pose_shape[:-1])
    result=compare_declared_review_action_route(a,b,fps=30.,declaration=duty,route=route,original_motion_intent=old)
    assert not result['same_time_action_supported'] and result['exploration_cost'] is None
    assert duty.payload['requested_joint_indices']==[0,1,3,5]

def test_original_intent_identity_cannot_be_replaced_after_candidate_success():
    old,duty,route=setup();other=freeze_selected_review_motion_intent([view(t) for t in TICKS],
        requested_ticks=TICKS,original_actor_identity='different-old')
    with pytest.raises(ValueError,match='rewrote'):
        compare_declared_review_action_route([view(t) for t in TICKS],[view(t) for t in TICKS],fps=30.,
            declaration=duty,route=route,original_motion_intent=other)

def test_declared_catalogue_cannot_drop_a_joint_or_claim_candidate_based_selection():
    _,duty,_=setup();p=duty.payload;p['requested_joint_indices']=[0,1,3]
    with pytest.raises(ValueError,match='catalogue'):FrozenReviewActionDeclaration(json.dumps(p))
    p=duty.payload;p['candidate_joint_success_used_to_declare']=True
    with pytest.raises(ValueError,match='catalogue'):FrozenReviewActionDeclaration(json.dumps(p))

def test_partial_elbow_basis_is_distinct_from_complete_body():
    old,_,route=setup();duty=freeze_review_action_declaration(schema='right_elbow_with_shoulders_v1',ticks=TICKS,
        original_motion_intent=old,original_preservation_duty_identity='a'*64,declared_scope_identity='b'*64)
    a=[view(t) for t in TICKS]
    result=compare_declared_review_action_route(a,deepcopy(a),fps=30.,declaration=duty,route=route,original_motion_intent=old)
    assert result['intent_kind']=='PARTIAL_ACTION' and not result['complete_body_observation']


def test_four_joint_upper_arm_basis_cannot_masquerade_as_full_body():
    old,_,route=setup()
    duty=freeze_review_action_declaration(schema='bilateral_upper_arm_v1',ticks=TICKS,
        original_motion_intent=old,original_preservation_duty_identity='a'*64,declared_scope_identity='b'*64)
    a=[view(t) for t in TICKS]
    for v in a:v['descriptor']=replace(v['descriptor'],pose_shape=tuple((k,k*.1,v['row']['output_tick']*.1)
        for k in range(4)))
    result=compare_declared_review_action_route(a,deepcopy(a),fps=30.,declaration=duty,route=route,
        original_motion_intent=old)
    assert result['intent_kind']=='PARTIAL_ACTION' and result['complete_on_declared_basis']
    assert not result['full_BODY_contract_complete'] and not result['complete_body_observation']
