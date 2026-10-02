"""A missing legacy crop fit cannot manufacture mandatory arm observations."""
from types import SimpleNamespace as NS

import pytest

from crosscut.group_joint_source_camera import GroupOutputRole as Role
from crosscut.group_source_changing_owner import _shot_roles


def fixture(shot_type='medium',family=None,decisions=()):
    shot=NS(first_segment_index=0,last_segment_index=0)
    context=NS(incumbent=NS(
        editorial_motion_plan=NS(boundary_subject_alignment_plan=NS(decisions=decisions)),
        visual_timeline=NS(choices=(NS(shot_type=shot_type),)),
        face_match_plan=NS(segments=(NS(alignment_subject=family,applied=False,
            fallback_reason='ambiguous_multiple_faces'),))))
    return context,(shot,)


@pytest.mark.parametrize('shot_type',['medium','close_up','closeup'])
def test_disabled_legacy_face_fit_keeps_declared_close_medium_head_intent(shot_type):
    assert _shot_roles(*fixture(shot_type))==(Role.FOCAL_HEAD,)


@pytest.mark.parametrize('family',['upper_body','silhouette'])
def test_explicit_nonface_family_is_not_downgraded_by_close_label(family):
    assert _shot_roles(*fixture('medium',family))==(Role.FOCAL_HEAD_ARMS,)


def test_wide_unknown_is_not_relabelled_a_head_only_shot():
    assert _shot_roles(*fixture('wide'))==(Role.FOCAL_HEAD_ARMS,)


def test_existing_eligible_formation_declaration_remains_required():
    decision=NS(eligible=True,boundary_index=0,subject_kind='formation')
    assert _shot_roles(*fixture(decisions=(decision,)))==(Role.FORMATION,)


def test_existing_eligible_upper_body_declaration_remains_required():
    decision=NS(eligible=True,boundary_index=0,subject_kind='upper_body')
    assert _shot_roles(*fixture(decisions=(decision,)))==(Role.FOCAL_HEAD_ARMS,)


def test_failed_formation_measurement_does_not_invent_an_output_duty():
    decision=NS(eligible=False,boundary_index=0,subject_kind='formation')
    assert _shot_roles(*fixture(decisions=(decision,)))==(Role.FOCAL_HEAD,)
