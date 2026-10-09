"""Fixed observed orientation requests, not real-stage quality approval."""
from copy import deepcopy
import math

import pytest

from crosscut.group_review_head_action import (observed_head_eye_angle,measure_observed_head_incident,
    bind_source_local_head_candidate)


def row(angle,native,source):
    radians=math.radians(angle)
    dx,dy=.03*math.cos(radians),.03*math.sin(radians)*16/9
    raw=[[.5,.4,.9],[.5+dx,.4+dy,.9],[.5-dx,.4-dy,.9]]+[[.5,.5,.1]]*14
    return dict(source_index=source,native_ordinal=native,body=dict(reason='observed',
        face_anchor_consistent=True,raw_keypoints_xyc=raw))


def incident(left=(20.,20.),right=(20.,20.)):
    return dict(left_rows=[row(a,100+i,0) for i,a in enumerate(left)],
        right_rows=[row(a,200+i,1) for i,a in enumerate(right)],
        left_cameras=([1.5,.5,.5,0.],)*2,right_cameras=([1.5,.5,.5,0.],)*2,
        left_ticks=(10,11),right_ticks=(12,13),fps=30.)


def test_opposite_head_inclination_is_not_hidden_by_matching_face_centers():
    good=measure_observed_head_incident(**incident())
    bad=measure_observed_head_incident(**incident(right=(-35.,-35.)))
    assert good['exploration_score']==pytest.approx(0.)
    assert bad['head_tilt_difference_degrees']==pytest.approx(55.)
    assert bad['exploration_score']>.4
    assert bad['quality_approved'] is False


def test_rotation_uses_clockwise_renderer_sign_and_does_not_mutate_observations():
    value=row(20.,100,0);original=deepcopy(value)
    assert observed_head_eye_angle(value,[1.5,.5,.5,6.])==pytest.approx(26.)
    assert value==original


def test_missing_ears_do_not_discard_observed_eyes_or_get_fabricated():
    value=row(20.,100,0)
    assert value['body']['raw_keypoints_xyc'][3][2]==.1
    assert observed_head_eye_angle(value,[1.5,.5,.5,0.])==pytest.approx(20.)


def test_missing_or_wrong_anchor_is_unknown_not_perfect_match():
    args=incident();args['right_rows'][0]['body']['face_anchor_consistent']=False
    result=measure_observed_head_incident(**args)
    assert result['head_tilt_difference_degrees'] is None
    assert result['exploration_cost_interval']==[0.,1.]
    assert result['missing_imputed_as_zero'] is False


def test_measured_reversed_head_tangent_has_a_positive_cost():
    result=measure_observed_head_incident(**incident(left=(20.,23.),right=(23.,20.)))
    assert result['head_tilt_difference_degrees']==pytest.approx(0.)
    assert result['exploration_score']>.2


def test_nonphysical_native_sequence_never_claims_observed_velocity():
    args=incident();args['right_rows'][1]['native_ordinal']=args['right_rows'][0]['native_ordinal']
    result=measure_observed_head_incident(**args)
    assert result['angular_velocity_cost_interval']==[0.,1.]
    assert result['physical_tangents_observed'] is False


def test_nonfinite_raw_head_rejects_instead_of_silently_filling_a_valid_angle():
    value=row(20.,100,0);value['body']['raw_keypoints_xyc'][1][0]=float('nan')
    with pytest.raises(ValueError,match='raw17'):observed_head_eye_angle(value,[1.5,.5,.5,0.])


def test_source_local_occurrence_cannot_flip_to_a_neighbour_with_slightly_higher_appearance_score():
    target={'box_xywh':[.248,.329,.123,.285]}
    lead={'face':{'box_xywh':[.241,.308,.132,.313]},'anchor_similarity':.3287}
    neighbour={'face':{'box_xywh':[.884,.578,.113,.291]},'anchor_similarity':.3397}
    inputs=deepcopy((target,lead,neighbour))
    assert bind_source_local_head_candidate([neighbour,lead],target) is lead
    assert (target,lead,neighbour)==inputs


def test_ambiguous_or_absent_occurrence_stays_unknown_instead_of_using_appearance_or_largest_face():
    target={'box_xywh':[.3,.3,.1,.2]}
    candidates=[{'face':deepcopy(target),'anchor_similarity':.9},
        {'face':{'box_xywh':[.302,.3,.1,.2]},'anchor_similarity':.1}]
    assert bind_source_local_head_candidate(candidates,target) is None
    assert bind_source_local_head_candidate([],target) is None
    assert bind_source_local_head_candidate([{'face':{'box_xywh':[.7,.3,.25,.5]}}],target) is None


@pytest.mark.parametrize('bad',[[.3,.3,0.,.2],[.3,.3,.1,float('nan')],[-.1,.3,.1,.2],[.95,.3,.1,.2]])
def test_native_head_binding_rejects_invalid_or_outside_observation(bad):
    with pytest.raises(ValueError,match='contained measured'):
        bind_source_local_head_candidate([{'face':{'box_xywh':bad}}],{'box_xywh':[.3,.3,.1,.2]})
