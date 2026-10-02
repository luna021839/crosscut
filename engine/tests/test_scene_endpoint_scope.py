"""Fixed-neighbor incident endpoints do not certify mutable whole shots."""
from types import SimpleNamespace

import pytest

from crosscut.group_joint_source_camera import GroupOutputRole
from crosscut.group_scene_observations import SceneSensors, scene_anchor_ticks


def fixture(unknown_ticks):
    state = SimpleNamespace(eligible_source_index=0, first_output_frame=100,
        end_output_frame=120, native_frame_indices=tuple(range(200,220)),
        role=GroupOutputRole.FOCAL_HEAD)
    sensors = object.__new__(SceneSensors)
    sensors.hypotheses = {}
    sensors.discoveries = {}
    calls = []

    def raw(state,tick,roll):
        calls.append(tick)
        return SimpleNamespace(sha256=str(tick),tick=tick,people=(),
            heads=() if tick in unknown_ticks else (SimpleNamespace(observation_id=tick),))

    sensors.raw = raw
    sensors._head_links = lambda a,b: {'unique_links': [[a.tick,b.tick]] if a.heads and b.heads else []}
    return state,sensors,calls


@pytest.mark.parametrize('scope,unknown,expected', [
    ('entry',{116,119},[100,103]), ('exit',{100,103},[116,119])])
def test_unknown_far_endpoint_does_not_erase_fixed_incident_motion(scope,unknown,expected):
    state,sensors,calls=fixture(unknown)
    found=sensors.discover(state,rolls=(None,),endpoint_scope=scope)
    assert len(found)==1
    assert found[0]['endpoint_scope']==scope
    assert found[0]['anchor_output_ticks']==expected
    assert calls==expected
    # Cache scope must not relabel the same partial hypothesis as a whole shot.
    assert sensors.discover(state,rolls=(None,),endpoint_scope='whole')==()
    assert sensors.discover(state,rolls=(None,),endpoint_scope=scope)==found


def test_unknown_incident_endpoint_still_rejects():
    state,sensors,_=fixture({116,119})
    assert sensors.discover(state,rolls=(None,),endpoint_scope='exit')==()


def test_mutable_dense_path_cannot_use_partial_endpoint_hypothesis():
    state,sensors,_=fixture(set())
    partial=sensors.discover(state,rolls=(None,),endpoint_scope='entry')[0]
    sensors.hypothesis=lambda state:partial
    with pytest.raises(ValueError,match='cannot borrow'):
        sensors.dense((state,state,state))


def test_whole_mutable_hypothesis_keeps_all_four_anchors():
    state,sensors,calls=fixture(set())
    found=sensors.discover(state,rolls=(None,))
    assert len(found)==1 and found[0]['endpoint_scope']=='whole'
    assert calls==[100,103,116,119]


def test_endpoint_scope_and_span_length_are_validated():
    assert scene_anchor_ticks(10,14,'entry')==(10,13)
    assert scene_anchor_ticks(10,14,'exit')==(10,13)
    for args in ((10,14,'whole'),(10,13,'entry'),(10,20,'unchecked')):
        with pytest.raises(ValueError):scene_anchor_ticks(*args)
