from copy import deepcopy
from types import SimpleNamespace
import pytest
from crosscut.group_review_actor_relation import (freeze_review_actor_relation_duty,
    original_actor_relation_request,evaluate_review_actor_relation)


def view(t,vector):
    return dict(row=dict(output_tick=t,native_observation=dict(source=2,native=t)),
        descriptor=SimpleNamespace(frame_local_head_id=0),vector=vector)


def duty():
    a=[1.]+[0.]*127
    return freeze_review_actor_relation_duty(requests=[original_actor_relation_request(t,view(t,a)) for t in (8,9,10)],
        original_actor_inventory_identity='a'*64,independent_control_identity='b'*64,minimum_cosine=.4)


def test_mutual_candidate_b_b_does_not_preserve_original_a():
    b=[0.,1.]+[0.]*126
    result=evaluate_review_actor_relation(duty=duty(),observations={t:view(t,b) for t in (8,9,10)})
    assert result['state']=='FAIL' and not result['known_original_actor_preserved']


def test_actual_same_original_relation_is_separate_from_motion_and_named_identity():
    a=[1.]+[0.]*127
    result=evaluate_review_actor_relation(duty=duty(),observations={t:view(t,a) for t in (8,9,10)})
    assert result['state']=='PASS' and not result['motion_suitability_certified']
    assert not result['named_actor_certified']


def test_reacquisition_ambiguity_and_missing_native_occurrence_stay_unknown():
    a=[1.]+[0.]*127;rows={t:view(t,a) for t in (8,9,10)}
    rows[9]['reacquisition_ambiguous']=True
    assert evaluate_review_actor_relation(duty=duty(),observations=rows)['state']=='UNKNOWN'
    rows[9]=None
    assert evaluate_review_actor_relation(duty=duty(),observations=rows)['state']=='UNKNOWN'


def test_anonymous_continuation_cannot_replace_a_known_original_request():
    p=duty().payload;p['requests'][1]['relation']='ANONYMOUS_CONTINUATION'
    with pytest.raises(ValueError,match='known actor'):
        freeze_review_actor_relation_duty(requests=p['requests'],original_actor_inventory_identity='a'*64,
            independent_control_identity='b'*64,minimum_cosine=.4)


def test_full_requested_tick_inventory_and_explicit_handoff_are_required():
    with pytest.raises(ValueError,match='instant'):
        evaluate_review_actor_relation(duty=duty(),observations={8:None,9:None})
    p=duty().payload;p['requests'][1]['relation']='EXPLICIT_HANDOFF'
    with pytest.raises(ValueError,match='handoff'):
        freeze_review_actor_relation_duty(requests=p['requests'],original_actor_inventory_identity='a'*64,
            independent_control_identity='b'*64,minimum_cosine=.4)
