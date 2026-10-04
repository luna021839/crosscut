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
        motion=dict(intent_kind='BODY',same_time_action_supported=True,exploration_cost=cost),
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
