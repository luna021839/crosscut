"""Ordered discovery-query ownership with explicit NON-HUMAN sensor rows.

These isolate cache semantics, not tilted human detection or editing quality.
Native/model inference is substituted only by the existing declared fixture.
"""
from dataclasses import replace
from copy import deepcopy

import pytest

from crosscut import group_scene_observations as scene
from crosscut.group_joint_source_camera import GroupOutputRole
from engine.tests.test_scene_known_part_motion import fixture as native_fixture


def fixture():
    sensors, state = native_fixture()
    sensors.articulation_raw_rows = {}
    sensors.articulation_endpoint_cache = {}
    return sensors, state


def rolls(rows):
    return tuple(row['analysis_roll'] for row in rows)


def test_warm_all_then_upright_is_identical_to_fresh_upright():
    sensors, state = fixture()
    assert rolls(sensors.discover(state, rolls=scene.VARIANTS)) == scene.VARIANTS
    fresh, fresh_state = fixture()
    expected = fresh.discover(fresh_state, rolls=(None,))
    assert sensors.discover(state, rolls=(None,)) == expected
    assert rolls(expected) == (None,)


def test_overlapping_queries_are_exact_ordered_views_without_erasing_archives():
    sensors, state = fixture()
    first = sensors.discover(state, rolls=(None, -60., 30.))
    second = sensors.discover(state, rolls=(30., None))
    reversed_query = sensors.discover(state, rolls=(None, 30.))
    assert rolls(first) == (None, -60., 30.)
    assert rolls(second) == (30., None)
    assert rolls(reversed_query) == (None, 30.)
    assert sensors.discover(state, rolls=(30., None)) == second
    assert sensors.discover(state, rolls=(None, -60., 30.)) == first
    archive = sensors.archive()
    assert len(archive['discoveries']) == 4  # Includes fixture's upright query.
    assert set(archive['hypotheses']) == {
        scene.source.family.digest(row) for row in (*first, *second, *reversed_query)}
    assert all(key in archive['hypotheses']
        for query in archive['discoveries'] for key in query['hypothesis_ids'])


def test_equivalent_numeric_rolls_share_one_query_and_raw_rows():
    sensors, state = fixture()
    integer_query = sensors.discover(state, rolls=(30, None))
    archive_before = deepcopy(sensors.archive())
    assert sensors.discover(state, rolls=(30., None)) == integer_query
    assert sensors.archive() == archive_before
    assert rolls(integer_query) == (30., None)


@pytest.mark.parametrize('query', [
    None, (), [], [None], 'upright', iter((None,)), (True,), (False,),
    (0.,), (90.,), (float('nan'),), (float('inf'),), (None, None),
    (30, 30.), (None, -60., -30., 30., 60., None),
])
@pytest.mark.parametrize('composition', [False, True])
def test_invalid_query_is_rejected_before_any_sensor_work(query, composition):
    sensors, state = fixture()
    if composition:
        state = replace(state, role=GroupOutputRole.COMPOSITION)
    before = deepcopy(sensors.archive())
    with pytest.raises(ValueError, match='roll'):
        sensors.discover(state, rolls=query)
    assert sensors.archive() == before


def test_composition_keeps_one_effective_unrotated_query():
    sensors, state = fixture()
    state = replace(state, role=GroupOutputRole.COMPOSITION)
    assert sensors.discover(state, rolls=scene.VARIANTS) == ()
    archive_before = deepcopy(sensors.archive())
    assert sensors.discover(state, rolls=(30., None)) == ()
    assert sensors.archive() == archive_before
