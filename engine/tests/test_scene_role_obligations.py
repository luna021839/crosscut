"""Required subjects are pre-camera; raw context is not a quality veto or erased."""
from copy import deepcopy

import pytest

from crosscut.group_scene_role_obligations import scene_role_obligations
from crosscut.group_joint_source_camera import GroupOutputRole as Role


def native():
    return dict(heads=[dict(observation_id=0),dict(observation_id=1)],people=[
        dict(observation_id=10,unique_head_observation_id=0,
            part_states=[['left_arm','observed'],['right_arm','observed']]),
        dict(observation_id=11,unique_head_observation_id=None,
            part_states=[['left_arm','observed'],['right_arm','observed']])])


def test_unbound_background_is_context_not_required_focal_arms():
    raw=native(); before=deepcopy(raw)
    duties=scene_role_obligations(raw,Role.FOCAL_HEAD_ARMS,0)
    assert duties.required_head_ids==(0,)
    assert duties.required_person_ids==(10,)
    assert duties.required_parts==((10,'left_arm'),(10,'right_arm'))
    assert duties.context_head_ids==(1,) and duties.context_person_ids==(11,)
    assert raw==before
    assert duties.to_dict()['raw_context_must_be_retained']


@pytest.mark.parametrize('part_state',['unknown','ambiguous'])
def test_missing_required_focal_arm_is_not_completed_by_background_arm(part_state):
    raw=native();raw['people'][0]['part_states'][1][1]=part_state
    with pytest.raises(ValueError,match='required arms'):
        scene_role_obligations(raw,Role.FOCAL_HEAD_ARMS,0)


def test_head_only_can_have_unknown_body_without_fabricating_ownership():
    duties=scene_role_obligations(native(),Role.FOCAL_HEAD,1)
    assert duties.focal_owner_id is None and duties.focal_owner_state=='unknown'
    assert duties.required_parts==() and duties.context_person_ids==(10,11)
    with pytest.raises(ValueError,match='required arms'):
        scene_role_obligations(native(),Role.FOCAL_HEAD_ARMS,1)


def test_known_selected_owner_arms_are_not_discarded_on_head_role():
    assert scene_role_obligations(native(),Role.FOCAL_HEAD,0).required_parts==(
        (10,'left_arm'),(10,'right_arm'))


def test_formation_preserves_every_declared_native_person_and_both_arms():
    duties=scene_role_obligations(native(),Role.FORMATION,None)
    assert duties.required_head_ids==(0,1)
    assert duties.required_person_ids==(10,11)
    assert len(duties.required_parts)==4 and not duties.context_person_ids
    with pytest.raises(ValueError,match='formation duty'):
        scene_role_obligations(native(),Role.FORMATION,0)


def test_ambiguous_body_owners_do_not_become_unique_by_selection():
    raw=native();raw['people'][1]['unique_head_observation_id']=0
    duties=scene_role_obligations(raw,Role.FOCAL_HEAD,0)
    assert duties.focal_owner_state=='ambiguous' and duties.focal_owner_id is None
    with pytest.raises(ValueError,match='required arms'):
        scene_role_obligations(raw,Role.FOCAL_HEAD_ARMS,0)


def test_missing_or_duplicate_observations_are_not_assumed_valid():
    raw=native()
    with pytest.raises(ValueError,match='missing'):
        scene_role_obligations(raw,Role.FOCAL_HEAD,5)
    raw['people'][1]['observation_id']=10
    with pytest.raises(ValueError,match='duplicated'):
        scene_role_obligations(raw,Role.FOCAL_HEAD,0)
