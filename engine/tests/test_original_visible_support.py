"""Nonhuman coordinate fixtures; no actual video quality is certified."""
from copy import deepcopy
from dataclasses import replace
import pytest

from crosscut.group_original_visible_support import (measured_visible_arm_support,
    support_preserved, validate_support)
from crosscut import group_scene_original_duties as duties
from crosscut import group_scene_observations as scene
from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe
from engine.tests.test_scene_focal_pose_motion import scene_world
from engine.tests.test_scene_original_duties import freeze


def test_partial_support_keeps_visible_shoulder_and_bone_without_requiring_outside_elbow():
    points=((.45,.40),(.45,.52),(.45,.57))
    project=lambda p:(p[0]*2-.5,p[1]*2)
    support=measured_visible_arm_support(native_points=points,project_point=project)
    assert support['visibility']=='partial'
    assert support['joint_indices']==[0]
    assert support['bone_portions'][0]['end_parameter']==pytest.approx(5/6)
    assert support_preserved(support,native_points=points,project_point=project)
    assert not support_preserved(support,native_points=points,
        project_point=lambda p:(p[0]*2-.5,p[1]*2+.03))


def test_outside_is_positive_projection_not_unknown_or_anatomical_occlusion():
    support=measured_visible_arm_support(native_points=((.2,.2),(.3,.3),(.4,.4)),
        project_point=lambda p:(p[0]+2,p[1]))
    assert support['visibility']=='outside'
    assert not support['joint_indices'] and not support['bone_portions']
    assert not support['occlusion_inferred'] and not support['anatomical_mask_certified']
    with pytest.raises(ValueError):measured_visible_arm_support(native_points=((.2,.2),(.3,float('nan')),(.4,.4)),
        project_point=lambda p:p)


def test_base_crop_loss_cannot_be_recovered_by_a_later_transform():
    points=((.2,.2),(.3,.3),(.4,.4))
    support=measured_visible_arm_support(native_points=points,project_point=lambda p:p)
    assert support['visibility']=='visible'
    assert not support_preserved(support,native_points=points,project_point=lambda p:p,
        base_crop=(.25,0.,1.,1.))


def test_visible_crossing_bone_is_required_even_with_both_endpoints_outside():
    support=measured_visible_arm_support(native_points=((-1.,.5),(2.,.5),(3.,.5)),
        project_point=lambda p:p)
    assert support['visibility']=='partial' and not support['joint_indices']
    assert len(support['bone_portions'])==1
    assert support_preserved(support,native_points=((-1.,.5),(2.,.5),(3.,.5)),project_point=lambda p:p)
    assert not support_preserved(support,native_points=((-1.,.5),(2.,.5),(3.,.5)),
        project_point=lambda p:(p[0]+.01,p[1]))


def test_rehashed_support_cannot_change_its_observed_projection_mask():
    support=measured_visible_arm_support(native_points=((.2,.2),(.3,.3),(.4,.4)),project_point=lambda p:p)
    bad=deepcopy(support);bad['joint_indices']=[]
    with pytest.raises(ValueError,match='changed after projection'):validate_support(bad)


def test_original_camera_changes_the_frozen_requirement_not_the_native_observation():
    sensors,state,h,rows,_=scene_world()
    for tick,row in tuple(rows.items()):
        rows[tick]=replace(row,people=tuple(replace(p,upper_body_xyc=tuple(
            (x,.20,c) for x,_,c in p.upper_body_xyc)) for p in row.people))
    original=freeze(sensors,state,h)
    # Exact original camera owns the visibility calculation. This explicit
    # synthetic camera translates all native arms outside the output.
    changed=replace(state,segments=tuple(replace(s,editorial_transform=EditorialTransformTrack(
        (EditorialTransformKeyframe(0.,2.,.5,.75),))) for s in state.segments))
    changed_program=freeze(sensors,changed,h)
    assert original['units'][0]['rows'][0]['required_part_names']==['left_arm','right_arm']
    assert changed_program['units'][0]['rows'][0]['required_part_names']==[]
    assert all(p['observation']=='observed' and p['original_output_visibility']=='outside'
        for p in changed_program['units'][0]['branches'][0]['rows'][0]['parts'].values())
    assert not changed_program['quality_approved']


def test_partial_original_requirement_survives_a_later_missing_measurement():
    sensors,state,h,rows,_=scene_world()
    original=freeze(sensors,state,h)
    first=original['units'][0]['rows'][0]['required_support_by_part']['left_arm']
    person=rows[2].people[0]
    rows[2]=replace(rows[2],people=(replace(person,part_states=(('left_arm','unknown'),('right_arm','unknown'))),))
    program=freeze(sensors,state,h)
    row=program['units'][0]['rows'][2]
    assert row['required_part_names']==['left_arm','right_arm']
    assert row['required_support_by_part']['left_arm'][0]['support']==first[0]['support']
    assert program['units'][0]['branches'][0]['rows'][2]['parts']['left_arm']['observation']=='missing'
