from types import SimpleNamespace
import pytest
from crosscut.group_review_known_joint_duty import (
    freeze_review_known_joint_duty,consume_review_known_joint_duty)


def view(tick,joints):
    return dict(row=dict(output_tick=tick,native_observation=dict(native_frame_index=tick)),
        descriptor=SimpleNamespace(pose_shape=tuple((k,0.,0.) for k in joints),frame_local_head_id=0))


def duty():
    return freeze_review_known_joint_duty({10:view(10,[0,1,3]),11:view(11,[0,1,3,5])},
        original_actor_inventory_identity='a'*64,original_preservation_duty_identity='b'*64)


def consume(values,points):
    return consume_review_known_joint_duty(duty=duty(),observations=values,projected_joint_points=points)


def test_original_wrist_outside_action_window_is_still_requested():
    result=consume({10:view(10,[0,1,3]),11:view(11,[0,1,3])},
        {10:{k:(.5,.5) for k in [0,1,3]},11:{k:(.5,.5) for k in [0,1,3]}})
    assert result['state']=='UNKNOWN'
    assert result['requests'][1]['missing_joint_indices']==[5]


def test_original_lost_detection_is_not_inferred_whole_arm_visibility():
    assert duty().payload['requests'][0]['known_joint_indices']==[0,1,3]


def test_all_actual_known_joints_preserved_is_not_motion_certificate():
    result=consume({t:view(t,[0,1,3,5]) for t in [10,11]},
        {t:{k:(.5,.5) for k in [0,1,3,5]} for t in [10,11]})
    assert result['state']=='PASS'
    assert not result['action_phase_certified'] and not result['actor_relation_certified']


def test_crop_is_fail_even_when_joint_exists():
    result=consume({t:view(t,[0,1,3,5]) for t in [10,11]},
        {10:{k:(.5,.5) for k in [0,1,3,5]},11:{k:(1.01,.5) for k in [0,1,3,5]}})
    assert result['state']=='FAIL'


def test_missing_whole_shot_tick_is_not_success():
    with pytest.raises(ValueError,match='whole-shot'):
        consume({10:view(10,[0,1,3])},{10:{}})


def test_unknown_original_actor_remains_unknown():
    frozen=freeze_review_known_joint_duty({10:None},original_actor_inventory_identity='a'*64,
        original_preservation_duty_identity='b'*64)
    result=consume_review_known_joint_duty(duty=frozen,observations={10:view(10,[0,1,3,5])},
        projected_joint_points={10:{k:(.5,.5) for k in [0,1,3,5]}})
    assert result['state']=='UNKNOWN'
