from copy import deepcopy
import pytest
from crosscut.group_review_camera_repair import FramingDescriptor
from crosscut.group_review_actor_relation import (freeze_review_actor_relation_duty,
    original_actor_relation_request,evaluate_review_actor_relation)


def view(t,vector):
    return dict(row=dict(output_tick=t,native_observation=dict(request_source_index=2,native_frame_index=t,
        source_clock_identity='clock',source_file_identity=[1,2,3,4,5]),projection=dict(camera_sha256='camera')),
        descriptor=FramingDescriptor((.5,.35),.2,.1,None,0,False),vector=vector,native_camera_unit_index=0)


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


def test_ambiguous_original_is_unresolved_not_known_original_authority():
    v=view(8,[1.]+[0.]*127);v['reacquisition_ambiguous']=True
    request=original_actor_relation_request(8,v)
    assert request['relation']=='UNRESOLVED' and request['original_actor_known'] is False


@pytest.mark.parametrize('state',('UNKNOWN','UNMATCHED','AMBIGUOUS'))
def test_unassociated_candidate_cannot_pass_by_identical_embedding(state):
    a=[1.]+[0.]*127;rows={t:view(t,a) for t in (8,9,10)}
    rows[9]['selected_actor_association_state']=state
    assert evaluate_review_actor_relation(duty=duty(),observations=rows)['state']=='UNKNOWN'


def test_anonymous_only_does_not_report_vacuous_known_original_preservation():
    d=freeze_review_actor_relation_duty(requests=[dict(tick=8,relation='ANONYMOUS_CONTINUATION',
        original_actor_known=False)],original_actor_inventory_identity='a'*64,
        independent_control_identity='b'*64,minimum_cosine=.4)
    result=evaluate_review_actor_relation(duty=d,observations={8:view(8,[1.]+[0.]*127)},
        anonymous_continuation_state='PASS')
    assert result['state']=='PASS' and not result['known_original_actor_preserved']
    assert result['known_original_request_count']==0


def test_actual_declared_a_to_b_handoff_has_distinct_scopes_and_bounded_original_requests():
    import json
    from crosscut.group_review_actor_relation import FrozenReviewFocalHandoff
    handoff=FrozenReviewFocalHandoff(json.dumps(dict(contract='independently_declared_focal_handoff_v1',
        from_focal_scope='c'*64,to_focal_scope='d'*64,transition_interval=[9,11],
        original_actor_inventory_identity='a'*64,original_preservation_duty_identity='e'*64,
        editorial_reason='independent original focal role change',
        permitted_transition_relation='EXPLICIT_FOCAL_CHANGE')))
    a=[1.]+[0.]*127;b=[0.,1.]+[0.]*126
    requests=[original_actor_relation_request(8,view(8,a)),
        original_actor_relation_request(9,view(9,b),relation='EXPLICIT_HANDOFF',
            handoff_declaration_identity=handoff.identity,handoff_declaration=handoff)]
    d=freeze_review_actor_relation_duty(requests=requests,original_actor_inventory_identity='a'*64,
        independent_control_identity='b'*64,minimum_cosine=.4)
    r=evaluate_review_actor_relation(duty=d,observations={8:view(8,a),9:view(9,b)})
    assert r['state']=='PASS' and r['known_original_request_count']==2
    bad=deepcopy(requests);bad[1]['tick']=11
    with pytest.raises(ValueError,match='handoff'):
        freeze_review_actor_relation_duty(requests=bad,original_actor_inventory_identity='a'*64,
            independent_control_identity='b'*64,minimum_cosine=.4)


def test_independent_anchor_and_two_complete_paths_are_not_per_frame_frontal_comparison():
    from crosscut.group_review_actor_relation import evaluate_review_control_anchored_actor_relation
    a=[1.]+[0.]*127
    original={t:view(t,a) for t in (8,9,10)}
    rows={8:view(8,a),9:view(9,[.8,.6]+[0.]*126),10:view(10,[.2,.98]+[0.]*126)}
    d=freeze_review_actor_relation_duty(requests=[original_actor_relation_request(t,original[t]) for t in original],
        original_actor_inventory_identity='a'*64,independent_control_identity='b'*64,minimum_cosine=.4,
        independently_declared_anchor_tick=8)
    assert evaluate_review_actor_relation(duty=d,observations=rows)['state']=='FAIL'
    kwargs=dict(duty=d,original_observations=original,observations=rows,
        original_anchor_alternatives=[original[8]],candidate_anchor_alternatives=[rows[8]])
    r=evaluate_review_control_anchored_actor_relation(**kwargs)
    assert r['state']=='PASS' and r['per_time_original_occurrences_consumed']
    assert not r['pixel_flow_certified'] and not r['named_actor_certified']
    rows[9]['selected_actor_association_state']='UNKNOWN'
    assert evaluate_review_control_anchored_actor_relation(**kwargs)['state']=='UNKNOWN'


def test_control_anchor_b_b_cannot_preserve_known_a_or_skip_native_path_loss():
    from crosscut.group_review_actor_relation import evaluate_review_control_anchored_actor_relation
    a=[1.]+[0.]*127;b=[0.,1.]+[0.]*126;original={t:view(t,a) for t in (8,9,10)}
    rows={t:view(t,b) for t in original}
    d=freeze_review_actor_relation_duty(requests=[original_actor_relation_request(t,original[t]) for t in original],
        original_actor_inventory_identity='a'*64,independent_control_identity='b'*64,minimum_cosine=.4,
        independently_declared_anchor_tick=8)
    kwargs=dict(duty=d,original_observations=original,observations=rows,
        original_anchor_alternatives=[original[8]],candidate_anchor_alternatives=[rows[8]])
    assert evaluate_review_control_anchored_actor_relation(**kwargs)['state']=='FAIL'
    rows[9]=None
    assert evaluate_review_control_anchored_actor_relation(**kwargs)['state']=='UNKNOWN'
