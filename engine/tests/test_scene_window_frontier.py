"""Pure work allocation: later song windows cannot disappear behind intro retries."""
from types import SimpleNamespace

import pytest

from crosscut.group_scene_source_owner import (time_stratified_window_frontier,
    camera_balanced_source_frontier,round_robin,repair_window_frontier,
    observed_wide_repair_priority,fixed_window_probe_order)
from crosscut.group_joint_source_camera import JointGroupWorkBudget,JointGroupBudgetExhausted


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


def test_source_quota_reaches_fitted_cameras_before_many_native_head_hypotheses():
    groups=[]
    for source in range(8):
        native=tuple(SimpleNamespace(key=f'{source}:native:{i}') for i in range(12))
        fitted=tuple(SimpleNamespace(key=f'{source}:fitted:{i}') for i in range(12))
        groups.append(camera_balanced_source_frontier(native,fitted))
    chosen=round_robin(groups,16)
    assert [s.key for s in chosen]==([f'{source}:native:0' for source in range(8)]+
        [f'{source}:fitted:0' for source in range(8)])


def test_camera_interleave_keeps_all_unique_states_and_does_not_create_a_camera():
    a,b,c=(SimpleNamespace(key=name) for name in ('a','b','c'))
    assert camera_balanced_source_frontier((a,b),(a,c))==(a,b,c)
    assert camera_balanced_source_frontier((a,b),())==(a,b)
    assert camera_balanced_source_frontier((),(c,))==(c,)
    assert camera_balanced_source_frontier((),())==()


def test_one_distant_existing_wide_precedes_uniform_middle_song_revisit():
    prepared=tuple(window(90*i) for i in range(30))
    prepared[10][2]['observed_wide_repair_priority']=.8
    chosen=repair_window_frontier(prepared,3)
    assert [i for i,_ in chosen]==[10,4,14]
    assert repair_window_frontier(prepared,1)[0][0]==10
    assert len({i for i,_ in chosen})==3


def test_without_observed_wide_priority_time_allocation_is_unchanged():
    prepared=tuple(window(90*i) for i in range(30))
    assert repair_window_frontier(prepared,3)==time_stratified_window_frontier(prepared,3)
    assert repair_window_frontier((),3)==()


def test_wide_work_priority_never_fills_unknown_scale_with_zero():
    shot=SimpleNamespace(first_segment_index=0,last_segment_index=0)
    def timeline(kind,scale):
        return SimpleNamespace(choices=(SimpleNamespace(canonical_master_start=0.,
            canonical_master_end=2.,shot_type=kind,entry_shot_scale=scale,exit_shot_scale=scale),))
    assert observed_wide_repair_priority(timeline('group_wide',.1),(shot,))==pytest.approx(.9)
    assert observed_wide_repair_priority(timeline('medium',.1),(shot,))==0.
    for scale in (None,float('nan'),-1,2):
        assert observed_wide_repair_priority(timeline('group_wide',scale),(shot,))==0.


def test_failed_primary_windows_leave_distinct_weak_wide_probe_work_reachable():
    prepared=tuple(window(90*i) for i in range(30))
    prepared[10][2]['observed_wide_repair_priority']=.9
    prepared[18][2]['observed_wide_repair_priority']=.8
    chosen=fixed_window_probe_order(prepared,6)
    assert [i for i,_ in chosen][:4]==[10,4,14,18]
    assert len({i for i,_ in chosen})==6
    assert all(item is prepared[i] for i,item in chosen)
    assert fixed_window_probe_order((),6)==()


def test_multiple_frontiers_have_explicit_request_cap_not_a_single_graph_cap():
    budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.,
        maximum_frontiers=3,maximum_states=72,maximum_edges=384)
    for index in range(3):
        budget.begin_frontier(label=index,maximum_states=24,maximum_edges=128)
        budget.spend('states',24)
        budget.spend('edges',128)
    assert budget.counts['states']==72
    assert budget.counts['edges']==384
    assert budget.to_dict()['maximum_frontiers']==3
    with pytest.raises(JointGroupBudgetExhausted):budget.spend('states')
    with pytest.raises(JointGroupBudgetExhausted):budget.spend('edges')
    with pytest.raises(JointGroupBudgetExhausted):
        budget.begin_frontier(label=3,maximum_states=24,maximum_edges=128)


def test_one_frontier_cannot_spend_a_later_graphs_states_or_edges():
    budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.,
        maximum_frontiers=3,maximum_states=72,maximum_edges=384)
    with pytest.raises(ValueError):budget.spend('states')
    budget.begin_frontier(label='first',maximum_states=24,maximum_edges=128)
    budget.spend('states',24);budget.spend('edges',128)
    with pytest.raises(JointGroupBudgetExhausted):budget.spend('states')
    with pytest.raises(JointGroupBudgetExhausted):budget.spend('edges')
    budget.spend('model_calls',2400,model='actual_named_model')
    budget.begin_frontier(label='later',maximum_states=24,maximum_edges=128)
    assert budget.frontier_remaining('states')==24
    assert budget.counts['states']==24
    assert budget.counts['model_calls']==2400
    with pytest.raises(JointGroupBudgetExhausted):
        budget.spend('model_calls',model='actual_named_model')


@pytest.mark.parametrize('frontiers',[0,4,True,1.5])
def test_request_frontier_count_cannot_expand_work_without_a_finite_contract(frontiers):
    with pytest.raises(ValueError):
        JointGroupWorkBudget(absolute_deadline=10.,maximum_frontiers=frontiers)


def test_default_single_frontier_and_native_work_caps_stay_unchanged():
    with pytest.raises(ValueError):
        JointGroupWorkBudget(absolute_deadline=10.,maximum_states=25)
    budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.,
        maximum_frontiers=3,maximum_states=72,maximum_edges=384)
    assert budget.limits['new_decodes']==1200
    assert budget.limits['model_calls']==2400
    assert budget.limits['dense_paths']==4
    budget.spend('new_decodes',1200)
    with pytest.raises(JointGroupBudgetExhausted):budget.spend('new_decodes')
