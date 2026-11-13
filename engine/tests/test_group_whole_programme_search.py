"""Small exact-search counterexamples; no model, sensor or media substitutes."""
import pytest

from crosscut.group_whole_programme_search import (
    ProgrammeOpportunity, ProgrammeRun, ProgrammeSearchBudget,
    programme_opportunities, select_whole_programme,
)


def run(identity, first, end, source=0, cost=0., actor=None, camera='camera', phase=0,
        protected=False):
    return ProgrammeRun(identity, first, end, source, 0, actor, camera, phase, cost, protected)


def nodes(*ticks):
    return tuple(ProgrammeOpportunity(t, ('test',)) for t in ticks)


def select(opportunities, runs, edge=lambda history, incoming: 0., **kwargs):
    return select_whole_programme(opportunities=opportunities, runs=runs,
        transition_cost=edge, budget=ProgrammeSearchBudget(beam_width=None), **kwargs)


def test_native_offbeat_node_is_not_snapped_to_a_beat_or_a_forced_cut():
    opportunities = programme_opportunities(output_frame_count=60, fps=30,
        beat_seconds=(0., 1., 2.), native_cut_ticks=(19,), action_ticks=(43,))
    assert tuple(o.output_tick for o in opportunities) == (0, 19, 30, 43, 60)
    result = select(opportunities, (run('hold', 0, 60),
        run('early', 0, 19, cost=1), run('later', 19, 60, cost=1)))
    assert [r.identity for r in result.runs] == ['hold']
    assert result.to_dict()['optional_nodes_are_not_forced_cuts']
    assert not result.to_dict()['approximate']


def test_full_history_prevents_wrong_a_b_a_dominance():
    inventory = (run('A-first', 0, 2, source=0), run('C-first', 0, 2, source=2, cost=1),
        run('B', 2, 4, source=1), run('A-last', 4, 6, source=0))

    def edge(history, incoming):
        return 100. if len(history) > 1 and history[-2].request_source_index == incoming.request_source_index else 0.

    result = select(nodes(0, 2, 4, 6), inventory, edge)
    assert [r.identity for r in result.runs] == ['C-first', 'B', 'A-last']
    assert result.cost == 1.


@pytest.mark.parametrize('field,value', [
    ('actor_path_identity', 'different-actor'),
    ('camera_recipe_identity', 'different-camera'),
    ('phase_native_frames', 1),
])
def test_same_source_and_duration_do_not_merge_actor_camera_or_phase(field, value):
    from dataclasses import replace
    cheap = run('cheap', 0, 2, actor='actor')
    useful = replace(cheap, identity='useful', unary_cost=1., **{field: value})
    last = run('last', 2, 4)
    result = select(nodes(0, 2, 4), (cheap, useful, last),
        lambda history, incoming: 10. if history and getattr(history[-1], field) != value else 0.)
    assert result.runs[0] == useful


def test_unpruned_search_matches_brute_force_variable_duration_graph():
    inventory = tuple(run(f'{first}-{end}-{source}', first, end, source,
        cost=(first + 3*end + source) % 7 / 10.)
        for first, end in ((0, 2), (0, 4), (2, 4), (2, 6), (4, 6))
        for source in (0, 1))

    def edge(history, incoming):
        return (0.3 if history and history[-1].request_source_index == incoming.request_source_index else 0.) + (
            1.7 if len(history) > 1 and history[-2].request_source_index == incoming.request_source_index else 0.)

    complete = []

    def visit(tick, path, cost):
        if tick == 6:
            complete.append((cost, tuple(r.identity for r in path)))
        for candidate in inventory:
            if candidate.first_output_frame == tick:
                visit(candidate.end_output_frame, (*path, candidate),
                    cost+candidate.unary_cost+edge(path, candidate))

    visit(0, (), 0.)
    result = select(nodes(0, 2, 4, 6), inventory, edge)
    assert (result.cost, tuple(r.identity for r in result.runs)) == min(complete)


def test_beam_reports_approximation_instead_of_claiming_exact_dp():
    result = select_whole_programme(opportunities=nodes(0, 2, 4),
        runs=(run('a', 0, 2), run('b', 0, 2), run('end', 2, 4)),
        transition_cost=lambda history, incoming: 0.,
        budget=ProgrammeSearchBudget(beam_width=1))
    assert result.beam_truncated
    assert result.to_dict()['approximate']


def test_only_frozen_protected_recipes_can_cover_protected_interval():
    result = select(nodes(0, 2, 4, 6), (run('prefix', 0, 2),
        run('cheap-new', 2, 4, cost=-100), run('reference', 2, 4, protected=True),
        run('false-bridge', 0, 6, cost=-100, protected=True), run('suffix', 4, 6)),
        protected_spans=((2, 4),))
    assert [r.identity for r in result.runs] == ['prefix', 'reference', 'suffix']
    with pytest.raises(ValueError, match='protected recipe flag'):
        select(nodes(0, 2, 4, 6), (run('bad-flag', 0, 2, protected=True),), protected_spans=((2, 4),))


def test_missing_measurement_is_a_finite_cost_not_a_global_source_veto():
    result = select(nodes(0, 2), (run('unknown-actor', 0, 2, actor=None),),
        lambda history, incoming: 1.25 if incoming.actor_path_identity is None else 0.)
    assert result.cost == 1.25


def test_cancellation_and_exhaustion_are_terminal_without_retry():
    inventory = (run('first', 0, 2), run('last', 2, 4))
    with pytest.raises(InterruptedError):
        select(nodes(0, 2, 4), inventory, cancelled=lambda: True)
    with pytest.raises(TimeoutError, match='no retry'):
        select_whole_programme(opportunities=nodes(0, 2, 4), runs=inventory,
            transition_cost=lambda history, incoming: 0.,
            budget=ProgrammeSearchBudget(maximum_expansions=1))


@pytest.mark.parametrize('tick,reasons', [(True, ('beat',)), (-1, ('beat',)), (2, ()),
    (2, ('beat', 'beat')), (2, ('',)), (2, ['beat'])])
def test_malformed_opportunities_are_rejected(tick, reasons):
    with pytest.raises(ValueError):
        ProgrammeOpportunity(tick, reasons)


@pytest.mark.parametrize('bad_edge', [None, True, float('nan'), float('inf')])
def test_nonfinite_or_boolean_edges_cannot_be_unknown_vetoes(bad_edge):
    with pytest.raises(ValueError, match='finite cost'):
        select(nodes(0, 2), (run('only', 0, 2),), lambda history, incoming: bad_edge)
