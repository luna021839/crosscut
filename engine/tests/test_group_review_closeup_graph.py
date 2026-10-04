from dataclasses import replace

import pytest

from crosscut.group_review_closeup_graph import CloseupReviewStyle, select_closeup_review_path


def alt(cell, source, *, keep=False, near=1.):
    return dict(identity=f'cell-{cell}-content-{source}-{keep}', source=source, keep=keep,
        near_fraction=near, blur_risk=0., uncertainty_risk=0.)


def run(cells, edge, **kwargs):
    return select_closeup_review_path(cells=cells, edge_at=edge, fps=30.,
        style=kwargs.get('style', CloseupReviewStyle()), maximum_changed_frames=kwargs.get('cap', 300))


def test_closeup_duration_preference_without_source_number_bias():
    cells = [dict(first=i*30, end=(i+1)*30, original_source=0,
        alternatives=[alt(i, 0, keep=True, near=.1), alt(i, 7)]) for i in range(3)]
    result = run(cells, lambda i, a, b: dict(risk=0., moving_match=0.))
    assert [a['source'] for a in result['alternatives']] == [7, 7, 7]
    assert result['added_source_switches'] == 0
    assert result['quality_approved'] is False


def test_measured_moving_near_match_can_interrupt_long_near_hold():
    cells = [dict(first=i*30, end=(i+1)*30, original_source=0,
        alternatives=[alt(i, 0, keep=True), alt(i, 1)]) for i in range(5)]
    def edge(i, a, b):
        return dict(risk=.1 if a['source'] != b['source'] else 0.,
            moving_match=.85 if a['source'] != b['source'] else 0.)
    result = run(cells, edge)
    assert result['added_source_switches'] >= 3
    assert result['whole_program_selected_once']


def test_source_change_without_match_gets_no_reward():
    cells = [dict(first=i*30, end=(i+1)*30, original_source=0,
        alternatives=[alt(i, 0, keep=True), alt(i, 1)]) for i in range(2)]
    result = run(cells, lambda i, a, b: dict(risk=2. if a['source'] != b['source'] else 0., moving_match=0.))
    assert all(a['keep'] for a in result['alternatives'])


def test_protected_keep_and_changed_frame_cap_survive_whole_graph():
    cells = [dict(first=i*30, end=(i+1)*30, original_source=0, protected=(i == 1),
        alternatives=[alt(i, 0, keep=True, near=0.)] if i == 1 else
            [alt(i, 0, keep=True, near=0.), alt(i, 2)]) for i in range(3)]
    result = run(cells, lambda i, a, b: dict(risk=.1, moving_match=0.), cap=30)
    assert result['changed_output_frames'] == 30
    assert result['alternatives'][1]['keep']


def test_cell_seam_with_same_source_cannot_claim_a_crosscut():
    cells = [dict(first=i*30, end=(i+1)*30, original_source=0,
        alternatives=[alt(i, 0, keep=True)]) for i in range(2)]
    with pytest.raises(ValueError, match='unchanged camera'):
        run(cells, lambda i, a, b: dict(risk=0., moving_match=1.))


def test_graph_keeps_explicit_feasible_keep_instead_of_retrying_subset():
    cells = [dict(first=0, end=30, original_source=0,
        alternatives=[alt(0, 0, keep=True, near=0.), alt(0, 1)])]
    assert run(cells, lambda *args: None, cap=0)['alternatives'][0]['keep']
