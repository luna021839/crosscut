"""Known geometry survives missing motion; observability is not editing gain.

These numbers and frame dictionaries are explicit non-human sensor fixtures.
They do not establish real pose accuracy or professional editing quality.
"""
from types import SimpleNamespace

import pytest

from crosscut import group_scene_observations as scene
from crosscut.group_scene_editorial_policy import evaluate_composition_path
from crosscut.group_joint_source_camera import GroupOutputRole


def edge(value, *, available=True):
    return dict(available=available, required_semantics_available=available,
        metrics={key:value for key in ('occupancy','attention','scale','span','screen_motion')})


class MissingEndpointSensor:
    def frame(self, state, tick):
        incoming=state.first_output_frame==10
        return dict(scene_geometry=dict(occupancy=.3 if incoming else .2,
            attention_xy=[.5,.4],subject_height_quantiles=[.3,.4,.5],horizontal_span=.6),
            projected_face_roi_xywh=[.45,.35,.1,.15])

    def endpoint(self, state, *, outgoing):
        raise ValueError('EXPLICIT_REQUIRED_ARM_MEASUREMENT_UNKNOWN')


@pytest.mark.parametrize('role',[GroupOutputRole.FOCAL_HEAD,GroupOutputRole.COMPOSITION,GroupOutputRole.FORMATION])
def test_failed_required_motion_retains_independently_measured_geometry(role):
    left=SimpleNamespace(first_output_frame=0,end_output_frame=10,role=role,key='left',output_fps=30.)
    right=SimpleNamespace(first_output_frame=10,end_output_frame=20,role=role,key='right',output_fps=30.)
    value=scene.compare_scene_edge(MissingEndpointSensor(),left,right)
    assert not value['available']
    assert value['required_semantics_available'] is False
    assert value['metrics']['occupancy']==pytest.approx(.1)
    assert value['metrics']['screen_motion']==0.
    assert value['reason']=='EXPLICIT_REQUIRED_ARM_MEASUREMENT_UNKNOWN'
    assert 'arm_predicted_0' not in value['metrics']
    known=scene.coherent_baseline([value])
    assert known['metrics']['occupancy']==pytest.approx(.1)
    assert not scene.preserves_known(known,edge(.2))


@pytest.mark.parametrize('before,after,expected',[
    (edge(.2,available=False),edge(.2),False),
    (edge(.2,available=False),edge(.3),False),
    (edge(.2,available=False),edge(.1),True),
    (dict(available=False,metrics={}),edge(.0),False),
    (dict(available=False,metrics={'occupancy':None}),edge(.0),False),
    (edge(.2),edge(.2,available=False),False),
])
def test_focal_gain_needs_a_strictly_better_known_value_not_new_availability(before,after,expected):
    assert scene.has_editorial_gain(before,after) is expected


def test_new_arm_measurements_alone_are_coverage_not_known_editing_gain():
    before,after=edge(.2,available=False),edge(.2)
    after['metrics'].update(arm_predicted_0=0.,arm_velocity_0=0.)
    assert scene.has_editorial_gain(before,after) is False


@pytest.mark.parametrize('before,candidate,reason',[
    (edge(.1,available=False),edge(.2),'known_worst_boundary_composition_utility_worse'),
    (edge(.2,available=False),edge(.2),'no_editorial_gain'),
    (dict(available=False,metrics=None),edge(.2),'no_editorial_gain'),
])
def test_composition_does_not_drop_partial_geometry_or_buy_gain_with_coverage(before,candidate,reason):
    result=evaluate_composition_path(dict(edges=[before]),[candidate])
    assert not result['admitted']
    assert result['reason']==reason
    assert result['unknown_imputed_as_zero'] is False


def test_partial_known_composition_geometry_can_improve_without_old_motion():
    result=evaluate_composition_path(dict(edges=[edge(.3,available=False)]),[edge(.2)])
    assert result['admitted']
    assert result['comparison_kind']=='partial_observed_baseline'
    assert result['known_baseline_cost'] is not None


def test_one_unknown_original_metric_does_not_erase_other_known_geometry():
    before=edge(.2,available=False)
    before['metrics']['screen_motion']=None
    after=edge(.3)
    result=evaluate_composition_path(dict(edges=[before]),[after])
    assert not result['admitted']
    assert result['reason']=='known_partial_geometry_lost_or_worse'


def test_partial_known_non_geometry_cannot_be_bought_by_better_composition():
    before,after=edge(.5,available=False),edge(.1)
    before['metrics']['arm_predicted_0']=.2
    result=evaluate_composition_path(dict(edges=[before]),[after])
    assert not result['admitted']
    assert result['reason']=='known_non_geometry_evidence_lost_or_worse'
