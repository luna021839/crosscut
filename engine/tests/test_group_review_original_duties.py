from types import SimpleNamespace as NS
from copy import deepcopy
import pytest

from crosscut.group_review_original_duties import freeze_review_original_duties
from crosscut.group_original_visible_support import ARM_INDICES


def world(role='focal_head'):
    unit=NS(unit_id='unit',role_program=dict(epochs=[dict(first_output_frame=0,end_output_frame=3,role=role)]))
    original=NS(program_sha256='a'*64,global_cell_frame_counts=(3,),units=(unit,))
    joints=[[.4,.3,.9],[.6,.3,.9],[.35,.5,.9],[.65,.5,.9],[.3,.7,.9],[.7,.7,.9],[.4,.8,.9],[.6,.8,.9]]
    person=dict(observation_id=5,unique_head_observation_id=7,part_states=[('left_arm_chain','observed')],upper_body_xyc=joints)
    row=dict(native_observation=dict(people=[person]),projection=dict(original_unit_id='unit',
        original_epoch_index=0,original_role=role,people=[dict(observation_id=5,projected_upper_body_xyc=joints)]))
    measured=dict(original_program_sha256='a'*64,requested_output_ticks=[0,2],rows={'0':row,
        '2':dict(native_observation=dict(people=[]),projection=dict(original_unit_id='unit',original_epoch_index=0,original_role=role,people=[]))},
        original_linked_heads={'0':7,'2':None})
    return original,measured


def test_unrequested_and_missing_actor_do_not_erase_known_original_arm():
    original,data=world();duties=freeze_review_original_duties(original,data)
    name=next(iter(ARM_INDICES))
    # Use the actual arm part vocabulary, not a renamed application role.
    data['rows']['0']['native_observation']['people'][0]['part_states']=[(name,'observed')]
    duties=freeze_review_original_duties(original,data)
    assert name in duties['rows']['0']['parts']
    assert duties['rows']['1']['parts']==duties['rows']['0']['parts']==duties['rows']['2']['parts']
    assert duties['rows']['1']['observation']=='not_requested'
    assert duties['full_original_output_ticks_observed'] is False


def test_declared_focal_head_and_observed_arms_is_not_accidentally_face_only():
    original,data=world('focal_head_and_observed_arms')
    duties=freeze_review_original_duties(original,data)
    assert all(r['head_required'] and set(r['parts'])==set(ARM_INDICES) for r in duties['rows'].values())
    assert not duties['quality_approved']


def test_formation_and_unresolved_roles_do_not_manufacture_a_cast_solution():
    for role in ('required_formation',None):
        original,data=world(role);duties=freeze_review_original_duties(original,data)
        assert all(r['unresolved_role_or_formation'] for r in duties['rows'].values())
        assert duties['original_cast_correspondence_certified'] is False


def test_candidate_relabel_or_omitted_requested_tick_cannot_weaken_original():
    original,data=world();data['rows']['0']['projection']['original_role']='observed_body_composition'
    with pytest.raises(ValueError,match='external unit'):freeze_review_original_duties(original,data)
    original,data=world();del data['rows']['2']
    with pytest.raises(ValueError,match='requested tick'):freeze_review_original_duties(original,data)


def test_new_positive_original_support_updates_while_missing_keeps_it():
    original,data=world();name=next(iter(ARM_INDICES))
    data['rows']['0']['native_observation']['people'][0]['part_states']=[(name,'observed')]
    new=deepcopy(data['rows']['0'])
    points=new['native_observation']['people'][0]['upper_body_xyc']
    for i in ARM_INDICES[name]:points[i][1]+=.1
    new['projection']['people'][0]['projected_upper_body_xyc']=deepcopy(points)
    data['rows']['1']=new;data['requested_output_ticks']=[0,1,2];data['original_linked_heads']['1']=7
    duties=freeze_review_original_duties(original,data)
    assert duties['rows']['0']['parts'][name]!=duties['rows']['1']['parts'][name]
    assert duties['rows']['1']['parts'][name]==duties['rows']['2']['parts'][name]


def test_explicit_whole_arm_duty_is_not_weakened_to_new_partial_support():
    original,data=world('focal_head_and_observed_arms');name=next(iter(ARM_INDICES))
    data['rows']['0']['native_observation']['people'][0]['part_states']=[(name,'observed')]
    duties=freeze_review_original_duties(original,data)
    assert duties['rows']['0']['parts'][name] is None


def test_automatic_role_does_not_promote_unobserved_arms_to_human_locks():
    original,data=world('focal_head_and_observed_arms')
    data['rows']['0']['native_observation']['people']=[]
    duties=freeze_review_original_duties(original,data,role_origin='automatic_observed_support_v1')
    assert all(not r['parts'] for r in duties['rows'].values())
    assert duties['automatic_role_policy']=='automatic_observed_support_v1'


def test_automatic_known_arm_remains_required_when_original_detection_disappears():
    original,data=world('focal_head_and_observed_arms');name=next(iter(ARM_INDICES))
    data['rows']['0']['native_observation']['people'][0]['part_states']=[(name,'observed')]
    duties=freeze_review_original_duties(original,data,role_origin='automatic_observed_support_v1')
    assert duties['rows']['0']['parts'][name] is not None
    assert duties['rows']['2']['parts']==duties['rows']['0']['parts']
