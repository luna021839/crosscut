"""Actual numeric operation classification; NONMEDIA/noneditorial fixtures."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from engine.tests.test_scene_owned_program import fixture
from crosscut.group_scene_edit_operations import (derive_scene_edit_operations,
    validate_scene_edit_operations, segment_edit_operation, path_has_numeric_edit)
from crosscut.render import EditorialTransformTrack, EditorialTransformKeyframe, SourceLookAdjustment
from crosscut.group_performer_family_owner import digest


def programme():
    context, domain = fixture()
    plan = context.incumbent.render_plan
    # Four different operations on distant original whole shots. Mixed-role
    # unit 30:90 stays one camera and is not edited in this numeric fixture.
    segments = list(plan.segments)
    track = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.15),))
    segments[0] = replace(segments[0], editorial_transform=track)
    segments[3] = replace(segments[3], source_index=7)
    segments[5] = replace(segments[5], source_index=7, editorial_transform=track)
    proposed = replace(plan, segments=tuple(segments))
    intervals = ((0, 30), (90, 120), (150, 180))
    return plan, proposed, domain, intervals


def test_four_operations_are_one_global_original_actual_shot_inventory():
    original, proposed, domain, intervals = programme()
    value = derive_scene_edit_operations(original=original, proposed=proposed,
        domain=domain, mutable_intervals=intervals)
    assert [row['operation'] for row in value['actual_shot_operations']] == [
        'CAMERA', 'KEEP', 'SOURCE', 'KEEP', 'SOURCE_AND_CAMERA', 'KEEP']
    assert len({row['actual_shot_occurrence_id'] for row in value['actual_shot_operations']}) == 6
    assert value['has_numeric_edit'] is True
    assert value['camera_only_program'] is False
    assert value['distinct_camera_transition_count_not_inferred'] is True
    assert value['quality_approved'] is False
    validate_scene_edit_operations(value, original=original, proposed=proposed,
        source_domain=domain.to_dict(), mutable_intervals=intervals)


def test_same_source_camera_is_an_edit_without_inventing_a_new_camera_cut():
    original, _, domain, _ = programme()
    segments = list(original.segments)
    segments[3] = replace(segments[3], editorial_transform=
        EditorialTransformTrack((EditorialTransformKeyframe(0., 1.1),)))
    proposed = replace(original, segments=tuple(segments))
    value = derive_scene_edit_operations(original=original, proposed=proposed,
        domain=domain, mutable_intervals=((90, 120),))
    assert value['camera_only_program'] is True
    edited = [row for row in value['actual_shot_operations'] if row['operation'] != 'KEEP']
    assert len(edited) == 1 and edited[0]['operation'] == 'CAMERA'
    assert edited[0]['request_source_index_changed'] is False
    assert edited[0]['source_occurrence_changed'] is False
    validate_scene_edit_operations(value, original=original, proposed=proposed,
        source_domain=domain.to_dict(), mutable_intervals=((90, 120),))


def test_unchanged_camera_and_color_only_are_not_a_new_numeric_edit():
    original, _, domain, _ = programme()
    proposed = replace(original, segments=tuple(replace(s,
        look_adjustment=SourceLookAdjustment(brightness=.01)) for s in original.segments))
    value = derive_scene_edit_operations(original=original, proposed=proposed,
        domain=domain, mutable_intervals=((120, 150),))
    assert value['has_numeric_edit'] is False
    assert all(row['operation'] == 'KEEP' for row in value['actual_shot_operations'])
    with pytest.raises(ValueError, match='KEEP'):
        validate_scene_edit_operations(value, original=original, proposed=proposed,
            source_domain=domain.to_dict(), mutable_intervals=((120, 150),))


def test_hypothesis_binding_change_is_not_eligibility_for_camera_work():
    original, proposed, _, _ = programme()
    a = NS(first_output_frame=0, end_output_frame=30, segments=(original.segments[0],), key='a')
    b = NS(first_output_frame=0, end_output_frame=30, segments=(original.segments[0],), key='b')
    assert path_has_numeric_edit((a,), (b,)) is False
    b.segments = (proposed.segments[0],)
    assert path_has_numeric_edit((a,), (b,)) is True


@pytest.mark.parametrize('change', ['camera_kind', 'mutable', 'source', 'quality', 'grid', 'native'])
def test_operations_cannot_be_relabelled_or_move_unowned_pixels(change):
    original, proposed, domain, intervals = programme()
    value = derive_scene_edit_operations(original=original, proposed=proposed,
        domain=domain, mutable_intervals=intervals)
    altered = deepcopy(value)
    if change == 'camera_kind': altered['actual_shot_operations'][0]['operation'] = 'SOURCE'
    elif change == 'mutable': altered['actual_shot_operations'][0]['mutable'] = False
    elif change == 'source': altered['actual_shot_operations'][0]['selected_request_source_index'] = 7
    elif change == 'quality': altered['quality_approved'] = True
    elif change == 'grid': proposed = replace(proposed, fps=29.97)
    elif change == 'native': proposed = replace(proposed, segments=(replace(proposed.segments[0],
        source_start=proposed.segments[0].source_start+1/30), *proposed.segments[1:]))
    payload=dict(altered);payload.pop('operations_sha256')
    altered['operations_sha256']=digest(payload)
    with pytest.raises(ValueError):
        validate_scene_edit_operations(altered, original=original, proposed=proposed,
            source_domain=domain.to_dict(), mutable_intervals=intervals)


def test_an_unowned_camera_edit_and_incident_subcell_leak_are_rejected():
    original, proposed, domain, intervals = programme()
    with pytest.raises(ValueError, match='unowned'):
        derive_scene_edit_operations(original=original, proposed=proposed,
            domain=domain, mutable_intervals=intervals[1:])
    context, domain = fixture(native_cut=1.51)
    original = context.incumbent.render_plan
    proposed = replace(original, segments=(original.segments[0],
        *(replace(segment, editorial_transform=EditorialTransformTrack((EditorialTransformKeyframe(0., 1.1),)))
            for segment in original.segments[1:3]), *original.segments[3:]))
    with pytest.raises(ValueError, match='unowned'):
        derive_scene_edit_operations(original=original, proposed=proposed,
            domain=domain, mutable_intervals=((30, 46),))


def test_effective_retiming_is_source_occurrence_not_camera_or_diversity():
    original, _, domain, _ = programme()
    segment = replace(original.segments[3], source_duration=1.002)
    assert segment_edit_operation(original.segments[3], segment).value == 'SOURCE'
    proposed = replace(original, segments=(*original.segments[:3], segment, *original.segments[4:]))
    value = derive_scene_edit_operations(original=original, proposed=proposed,
        domain=domain, mutable_intervals=((90, 120),))
    assert value['actual_shot_operations'][2]['request_source_index_changed'] is False
    assert value['actual_shot_operations'][2]['source_occurrence_changed'] is True


def test_one_whole_same_source_camera_is_not_two_operations_at_a_role_epoch():
    context,domain=fixture();original=context.incumbent.render_plan
    track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),))
    proposed=replace(original,segments=(original.segments[0],
        *(replace(segment,editorial_transform=track) for segment in original.segments[1:3]),
        *original.segments[3:]))
    value=derive_scene_edit_operations(original=original,proposed=proposed,
        domain=domain,mutable_intervals=((30,90),))
    edited=[row for row in value['actual_shot_operations'] if row['operation']!='KEEP']
    assert len(edited)==1 and edited[0]['operation']=='CAMERA'
    assert (edited[0]['first_output_frame'],edited[0]['end_output_frame'])==(30,90)
    assert value['camera_only_program'] is True


def test_keep_completion_is_rejected_before_expensive_descriptors():
    from crosscut import group_source_changing_owner as source
    from crosscut.group_joint_source_camera import GroupOutputRole
    from crosscut.group_actual_shots import visible_group_shots
    context,domain=fixture();context.domain=domain;context.source_derivative_binding=None
    path=source._states(context.incumbent.render_plan,domain,
        visible_group_shots(context.incumbent.render_plan,domain)[:3],(GroupOutputRole.FOCAL_HEAD,)*3)
    with patch('crosscut.group_joint_render_observations.refresh_joint_group_render_observations') as descriptor:
        with pytest.raises(ValueError,match='before descriptor work'):
            source.complete_source_candidate(context,path,NS(scene_family=True),None,{},[],[],{})
        descriptor.assert_not_called()
