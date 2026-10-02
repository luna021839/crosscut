"""Pure work allocation: later song windows cannot disappear behind intro retries."""
from types import SimpleNamespace

import pytest

from crosscut.group_scene_source_owner import time_stratified_window_frontier


def window(first):
    states = tuple(SimpleNamespace(first_output_frame=first + 30*i,
        end_output_frame=first + 30*(i+1)) for i in range(3))
    return states, (), {'first': first}


def test_frontier_covers_beginning_middle_and_end_without_changing_inventory():
    prepared = tuple(window(90*i) for i in range(30))
    before = tuple(prepared)
    chosen = time_stratified_window_frontier(prepared, 3)
    assert [i for i, _ in chosen] == [4, 14, 24]
    assert tuple(prepared) == before
    assert all(item is prepared[i] for i, item in chosen)


def test_frontier_is_output_time_stratified_not_index_stratified():
    prepared = tuple(window(i*3) for i in range(20)) + (window(1200), window(2400))
    chosen = time_stratified_window_frontier(prepared, 3)
    assert len(chosen) == 3
    assert [item[2]['first'] for _, item in chosen][-2:] == [1200, 2400]


def test_small_or_empty_inventory_is_preserved_and_duplicate_windows_not_created():
    for count in range(4):
        prepared = tuple(window(i*90) for i in range(count))
        chosen = time_stratified_window_frontier(prepared, 3)
        assert len({i for i, _ in chosen}) == count
        assert tuple(item for _, item in chosen) == prepared


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_invalid_work_budget_is_rejected(limit):
    with pytest.raises(ValueError):
        time_stratified_window_frontier((window(0),), limit)
