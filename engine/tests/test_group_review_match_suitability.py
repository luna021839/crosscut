import pytest
from crosscut.group_review_match_suitability import (freeze_control_calibrated_review_match_policy,
    evaluate_review_match_suitability)


def controls():
    return [dict(label=label,evidence_identity=str(i)*64,output_ticks=[20,21,22],
        appearance_cosine=cos,motion_cost=cost) for i,(label,cos,cost) in enumerate((
        ('POSITIVE_MOVING_MATCH',.7,.2),('WRONG_ACTOR',.2,.1),('WRONG_PHASE',.7,1.2)),1)]


def policy():return freeze_control_calibrated_review_match_policy(controls(),target_interval=(100,160))


def evaluate(cos=.7,cost=.2,**kwargs):
    return evaluate_review_match_suitability(policy=policy(),appearance_cosine=cos,mutual_appearance=True,
        motion=dict(intent_kind='BODY',same_time_action_supported=True,exploration_cost=cost,
            nonstationary_native_action_observed=True,complete_on_declared_basis=True,full_BODY_contract_complete=True),
        actual_required_support_state=kwargs.pop('actual_required_support_state','PASS'),**kwargs)


def test_observed_sole_mutual_faces_with_zero_cosine_are_not_same_actor_match():
    assert evaluate(cos=0.)['state']=='UNSUITABLE'


def test_same_actor_wrong_phase_and_large_pose_error_cannot_be_bought_by_style():
    assert evaluate(cost=5.)['state']=='UNSUITABLE'


def test_observed_positive_match_still_needs_complete_final_replay():
    result=evaluate()
    assert result['match_suitable'] and not result['review_encode_permitted'] and not result['quality_approved']


@pytest.mark.parametrize('state',('UNKNOWN','FAIL'))
def test_actual_camera_missing_wrist_does_not_become_match_suitable(state):
    assert not evaluate(actual_required_support_state=state)['match_suitable']


def test_repeated_native_raster_cannot_calibrate_motion_from_an_alias():
    assert evaluate(actual_native_alias=True)['state']=='UNSUITABLE'


def test_nonseparable_real_controls_do_not_lower_threshold_to_get_a_pass():
    data=controls();data[1]['appearance_cosine']=.9
    p=freeze_control_calibrated_review_match_policy(data,target_interval=(100,160))
    assert p.payload['state']=='CONTROLS_NOT_SEPARATED'
    assert p.payload['minimum_appearance_cosine'] is None


def test_target_result_is_not_an_independent_calibration_control():
    data=controls();data[0]['output_ticks']=[120]
    with pytest.raises(ValueError,match='borrowed target'):
        freeze_control_calibrated_review_match_policy(data,target_interval=(100,160))


def test_identity_negative_does_not_invent_an_unmeasured_body_cost():
    data=controls();data[1]['motion_cost']=None
    p=freeze_control_calibrated_review_match_policy(data,target_interval=(100,160))
    assert p.payload['state']=='CONTROL_SEPARATED'
    assert p.payload['controls'][1]['motion_cost'] is None
    data[0]['motion_cost']=None
    with pytest.raises(ValueError,match='actual control measurements'):
        freeze_control_calibrated_review_match_policy(data,target_interval=(100,160))


def test_serialized_policy_cannot_replace_the_control_separation_threshold():
    import json
    from crosscut.group_review_match_suitability import FrozenReviewMatchPolicy
    value=policy().payload;value['maximum_motion_cost']=100.
    with pytest.raises(ValueError,match='thresholds'):
        FrozenReviewMatchPolicy(json.dumps(value))


def partial_policy(state='FAIL'):
    data=controls();data[2].update(motion_cost=None,control_role='BODY_COVERAGE_ADVERSARY',
        expected_body_coverage_state='FAIL',actual_body_coverage_state=state)
    return freeze_control_calibrated_review_match_policy(data,target_interval=(100,160))


def test_offscreen_phase_control_does_not_erase_measured_identity_or_invent_motion_cost():
    p=partial_policy();assert p.payload['state']=='CONTROLS_PARTIAL'
    assert p.payload['identity_control_state']=='CONTROL_SEPARATED'
    assert p.payload['minimum_appearance_cosine']==pytest.approx(.45)
    assert p.payload['numeric_phase_control_state']=='UNMEASURED'
    assert p.payload['maximum_motion_cost'] is None
    assert p.payload['body_coverage_adversary_state']=='COVERAGE_NEGATIVE_CONFIRMED'
    assert p.payload['controls'][2]['motion_cost'] is None
    result=evaluate_review_match_suitability(policy=p,appearance_cosine=.7,mutual_appearance=True,
        motion=dict(intent_kind='BODY',same_time_action_supported=True,exploration_cost=.2),
        actual_required_support_state='PASS')
    assert not result['match_suitable'] and result['state']=='UNMEASURED'
    result=evaluate_review_match_suitability(policy=p,appearance_cosine=.1,mutual_appearance=True,
        motion={},actual_required_support_state='PASS')
    assert result['state']=='UNSUITABLE'


def test_unknown_coverage_is_not_confirmed_negative_and_failed_coverage_stays_failed():
    assert partial_policy('UNKNOWN').payload['body_coverage_adversary_state']=='UNMEASURED'
    assert partial_policy('PASS').payload['state']=='CONTROLS_NOT_SEPARATED'


def test_coverage_negative_has_no_silent_finite_cost_imputation():
    data=controls();data[2].update(control_role='BODY_COVERAGE_ADVERSARY',
        expected_body_coverage_state='FAIL',actual_body_coverage_state='FAIL')
    with pytest.raises(ValueError):freeze_control_calibrated_review_match_policy(data,target_interval=(100,160))


def test_one_technical_diagnostic_is_separate_from_uncalibrated_suitability():
    from crosscut.group_review_match_suitability import evaluate_review_diagnostic_encode_permission
    kwargs=dict(policy=partial_policy(),appearance_cosine=.7,mutual_appearance=True,
        motion=dict(intent_kind='BODY',same_time_action_supported=True,exploration_cost=.2,
            nonstationary_native_action_observed=True,full_BODY_contract_complete=True),
        actual_required_support_state='PASS',complete_keyed_incident_state='PASS',
        protected_reference_state='PASS',candidate_full_output_replay_state='PASS',
        original_known_duty_state='PASS',actor_relation_state='PASS')
    r=evaluate_review_diagnostic_encode_permission(**kwargs)
    assert r['diagnostic_encode_permitted'] and r['maximum_encodes']==1
    assert not r['match_suitable'] and not r['adoption_permitted']
    for field in ('actual_required_support_state','complete_keyed_incident_state',
            'protected_reference_state','candidate_full_output_replay_state'):
        assert not evaluate_review_diagnostic_encode_permission(**dict(kwargs,**{field:'UNKNOWN'}))['diagnostic_encode_permitted']
    assert not evaluate_review_diagnostic_encode_permission(**dict(kwargs,policy=partial_policy('PASS')))['diagnostic_encode_permitted']


@pytest.mark.parametrize('field',('original_known_duty_state','actor_relation_state'))
@pytest.mark.parametrize('state',('UNKNOWN','FAIL'))
def test_body_diagnostic_cannot_bypass_original_or_actor_duty(field,state):
    from crosscut.group_review_match_suitability import evaluate_review_diagnostic_encode_permission
    kwargs=dict(policy=partial_policy(),appearance_cosine=.7,mutual_appearance=True,
        motion=dict(intent_kind='BODY',same_time_action_supported=True,exploration_cost=.2,
            nonstationary_native_action_observed=True,full_BODY_contract_complete=True),
        actual_required_support_state='PASS',complete_keyed_incident_state='PASS',
        protected_reference_state='PASS',candidate_full_output_replay_state='PASS',
        original_known_duty_state='PASS',actor_relation_state='PASS')
    kwargs[field]=state
    assert not evaluate_review_diagnostic_encode_permission(**kwargs)['diagnostic_encode_permitted']


def test_typed_partial_diagnostic_keeps_full_body_false_and_needs_every_original_actor_duty():
    from crosscut.group_review_match_suitability import evaluate_review_diagnostic_encode_permission
    from crosscut.group_review_declared_action import freeze_review_action_declaration
    from crosscut.group_review_motion_contract import freeze_selected_review_motion_intent
    old=freeze_selected_review_motion_intent([None]*5,requested_ticks=(8,9,10,11,12),original_actor_identity='original')
    declaration=freeze_review_action_declaration(schema='bilateral_upper_arm_v1',ticks=(8,9,10,11,12),
        original_motion_intent=old,original_preservation_duty_identity='a'*64,declared_scope_identity='b'*64)
    motion=dict(intent_kind='PARTIAL_ACTION',declaration_identity=declaration.identity,
        declared_action_schema='bilateral_upper_arm_v1',complete_on_declared_basis=True,
        full_BODY_contract_complete=False,all_declared_comparison_velocity_windows_measured=True,
        nonstationary_native_action_observed=True,same_time_action_supported=True,exploration_cost=.2)
    data=controls()
    for c in data:c['action_schema']='bilateral_upper_arm_v1'
    data[2].update(motion_cost=None,control_role='BODY_COVERAGE_ADVERSARY',
        expected_body_coverage_state='FAIL',actual_body_coverage_state='FAIL')
    scoped_policy=freeze_control_calibrated_review_match_policy(data,target_interval=(100,160),
        action_schema='bilateral_upper_arm_v1')
    kwargs=dict(policy=scoped_policy,appearance_cosine=.7,mutual_appearance=True,motion=motion,
        actual_required_support_state='PASS',complete_keyed_incident_state='PASS',protected_reference_state='PASS',
        candidate_full_output_replay_state='PASS',partial_action_declaration=declaration,
        original_known_duty_state='PASS',actor_relation_state='PASS')
    result=evaluate_review_diagnostic_encode_permission(**kwargs)
    assert result['diagnostic_encode_permitted'] and not result['full_BODY_contract_complete']
    assert not result['adoption_permitted'] and not result['match_suitable']
    for field in ('original_known_duty_state','actor_relation_state'):
        assert not evaluate_review_diagnostic_encode_permission(**dict(kwargs,**{field:'UNKNOWN'}))['diagnostic_encode_permitted']
    for field in ('nonstationary_native_action_observed','all_declared_comparison_velocity_windows_measured'):
        assert not evaluate_review_diagnostic_encode_permission(**dict(kwargs,motion=dict(motion,**{field:False})))['diagnostic_encode_permitted']
    assert not evaluate_review_diagnostic_encode_permission(**dict(kwargs,partial_action_declaration=None))['diagnostic_encode_permitted']
    assert not evaluate_review_diagnostic_encode_permission(**dict(kwargs,policy=partial_policy()))['diagnostic_encode_permitted']


def test_partial_control_policy_cannot_reuse_full_body_costs_or_stationary_suitability():
    with pytest.raises(ValueError,match='control measurements'):
        freeze_control_calibrated_review_match_policy(controls(),target_interval=(100,160),
            action_schema='bilateral_upper_arm_v1')
    result=evaluate_review_match_suitability(policy=policy(),appearance_cosine=.7,mutual_appearance=True,
        motion=dict(intent_kind='BODY',same_time_action_supported=True,exploration_cost=0.,
            complete_on_declared_basis=True,full_BODY_contract_complete=True,
            nonstationary_native_action_observed=False),actual_required_support_state='PASS')
    assert not result['match_suitable']
