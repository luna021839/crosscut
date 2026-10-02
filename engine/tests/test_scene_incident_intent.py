"""Own-side fixed pixels must not inherit duties from distant unchanged cells."""
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from crosscut.group_scene_intent import declared_shot_intent, validate_intent
from crosscut.group_joint_source_camera import GroupOutputRole as Role


def context(kinds, *, subjects=None):
    subjects=subjects or [None]*len(kinds)
    return NS(incumbent=NS(render_plan=NS(),
        visual_timeline=NS(choices=[NS(shot_type=k) for k in kinds]),
        face_match_plan=NS(segments=[NS(alignment_subject=s,applied=False) for s in subjects])))


def intent(kinds, scope, *, subjects=None, first=0, end=None):
    shot=NS(first_segment_index=0,last_segment_index=len(kinds)-1,
        first_output_frame=first,end_output_frame=30*len(kinds) if end is None else end)
    with patch('crosscut.group_scene_intent.segment_frame_counts',return_value=(30,)*len(kinds)):
        return declared_shot_intent(context(kinds,subjects=subjects),shot,endpoint_scope=scope)


@pytest.mark.parametrize('kinds,scope',[(['close_up','medium','group_wide'],'exit'),
    (['group_wide','medium','close_up'],'entry')])
def test_wide_fixed_incident_does_not_inherit_a_distant_focal_duty(kinds,scope):
    value=intent(kinds,scope)
    assert value['role']==Role.COMPOSITION.value
    assert len(value['cells'])==1 and len(value['retained_output_cells'])==3
    assert value['fixed_nonincident_pixels_unchanged'] is True
    validate_intent(value,role=Role.COMPOSITION.value,first=0,end=90,endpoint_scope=scope)


def test_mutable_whole_shot_still_cannot_discard_a_stronger_cell():
    with pytest.raises(ValueError,match='mixed'):
        intent(['close_up','group_wide'],'whole')


def test_explicit_focal_duty_on_actual_fixed_pixels_remains_required():
    value=intent(['close_up','group_wide'],'exit',subjects=[None,'face'])
    assert value['role']==Role.FOCAL_HEAD.value
    validate_intent(value,role=Role.FOCAL_HEAD.value,first=0,end=60,endpoint_scope='exit')


def test_incident_measurement_crossing_role_boundary_still_rejects():
    with pytest.raises(ValueError,match='mixed'):
        intent(['close_up','group_wide'],'entry',first=28)


def test_fixed_scope_cannot_be_reused_as_whole_mutable_authority():
    value=intent(['close_up','group_wide'],'exit')
    with pytest.raises(ValueError):
        validate_intent(value,role=Role.COMPOSITION.value,first=0,end=60,endpoint_scope='whole')


def test_removed_retained_cells_and_relocated_interval_cannot_pass():
    value=intent(['close_up','medium','group_wide'],'exit')
    for mode in ('cell','interval','borrow'):
        bad=deepcopy(value)
        if mode=='cell':bad['retained_output_cells'].pop(1)
        elif mode=='interval':bad['required_output_interval']=[30,60]
        else:bad['cells']=[bad['retained_output_cells'][0]]
        with pytest.raises(ValueError):
            validate_intent(bad,role=Role.COMPOSITION.value,first=0,end=90,endpoint_scope='exit')


def test_distinct_incident_scopes_have_distinct_frozen_cell_inventories():
    kinds=['group_wide','medium','close_up']
    entry,exit=(intent(kinds,scope) for scope in ('entry','exit'))
    assert entry['role']==Role.COMPOSITION.value and exit['role']==Role.FOCAL_HEAD.value
    assert entry['retained_output_cells']==exit['retained_output_cells']
    assert entry['cells']!=exit['cells']
