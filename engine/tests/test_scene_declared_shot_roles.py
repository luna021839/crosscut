"""A missing legacy crop fit cannot manufacture mandatory arm observations."""
from types import SimpleNamespace as NS
from pathlib import Path

import pytest

from crosscut.group_joint_source_camera import GroupOutputRole as Role
from crosscut.group_source_changing_owner import _shot_roles
from crosscut.render import RenderPlan, RenderSegment
from crosscut.ive import AspectRatio


def fixture(shot_type='medium',family=None,decisions=()):
    shot=NS(first_segment_index=0,last_segment_index=0,
        first_output_frame=0,end_output_frame=30)
    plan=RenderPlan(sources=(Path('declared-intent-fixture.mp4'),),
        segments=(RenderSegment(0,0.,1.,0.),),reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,output=Path('unused.mp4'),fps=30.)
    context=NS(incumbent=NS(
        render_plan=plan,
        editorial_motion_plan=NS(boundary_subject_alignment_plan=NS(decisions=decisions)),
        visual_timeline=NS(choices=(NS(shot_type=shot_type),)),
        face_match_plan=NS(segments=(NS(alignment_subject=family,applied=False,
            fallback_reason='ambiguous_multiple_faces'),))))
    return context,(shot,)


@pytest.mark.parametrize('shot_type',['medium','close_up','closeup'])
def test_disabled_legacy_face_fit_keeps_declared_close_medium_head_intent(shot_type):
    assert _shot_roles(*fixture(shot_type))==(Role.FOCAL_HEAD,)


@pytest.mark.parametrize('family,expected',[
    ('upper_body',Role.FOCAL_HEAD_ARMS),('silhouette',Role.FORMATION)])
def test_explicit_nonface_family_is_not_downgraded_by_close_label(family,expected):
    assert _shot_roles(*fixture('medium',family))==(expected,)


def test_wide_unknown_is_not_relabelled_a_head_only_shot():
    assert _shot_roles(*fixture('wide'))==(Role.COMPOSITION,)


def test_incident_formation_observation_does_not_own_entire_output_cell():
    decision=NS(eligible=True,boundary_index=0,subject_kind='formation')
    assert _shot_roles(*fixture(decisions=(decision,)))==(Role.FOCAL_HEAD,)


def test_incident_upper_body_observation_does_not_own_entire_output_cell():
    decision=NS(eligible=True,boundary_index=0,subject_kind='upper_body')
    assert _shot_roles(*fixture(decisions=(decision,)))==(Role.FOCAL_HEAD,)


@pytest.mark.parametrize('family,expected',[
    ('face',Role.FOCAL_HEAD),('upper_body',Role.FOCAL_HEAD_ARMS),
    ('formation',Role.FORMATION),('silhouette',Role.FORMATION)])
def test_stronger_cell_owned_duty_survives_a_wide_label(family,expected):
    assert _shot_roles(*fixture('wide',family))==(expected,)


def test_failed_formation_measurement_does_not_invent_an_output_duty():
    decision=NS(eligible=False,boundary_index=0,subject_kind='formation')
    assert _shot_roles(*fixture(decisions=(decision,)))==(Role.FOCAL_HEAD,)
