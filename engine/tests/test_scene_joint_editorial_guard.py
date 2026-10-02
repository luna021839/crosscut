"""Two-sided normal composition costs; synthetic numbers are not human QA."""
from crosscut.group_scene_editorial_policy import evaluate_composition_path


def edge(value):
    return dict(available=True, required_semantics_available=True,
        metrics=dict(occupancy=value, attention=value, scale=value,
            span=value, screen_motion=value))


def test_better_entry_cannot_hide_a_new_worst_exit():
    result = evaluate_composition_path(dict(edges=[edge(.3), edge(.3)]),
        [edge(0.), edge(.4)])
    assert not result['admitted']
    assert result['reason'] == 'known_worst_boundary_composition_utility_worse'


def test_real_negative_texture_seam_costs_do_not_count_as_gain():
    # Exact recorded scalar counterexample; these are texture costs, NOT poses.
    before = [edge(.108629), edge(.003683)]
    after = [edge(.109570), edge(.110049)]
    assert not evaluate_composition_path(dict(edges=before), after)['admitted']


def test_both_edges_improve_with_no_changed_semantic_guards():
    result = evaluate_composition_path(dict(edges=[edge(.4), edge(.3)]),
        [edge(.2), edge(.1)])
    assert result['admitted']
    assert not result['quality_approved']
    assert not result['calibrated_professional_utility']


def test_unknown_incumbent_edge_does_not_erase_known_worst_regression():
    result = evaluate_composition_path(dict(edges=[None, edge(.3)]),
        [edge(0.), edge(.4)])
    assert not result['admitted']
    assert result['unknown_imputed_as_zero'] is False


def test_all_edge_ties_keep_the_incumbent():
    result = evaluate_composition_path(dict(edges=[edge(.3), edge(.3)]),
        [edge(.3), edge(.3)])
    assert not result['admitted']


def test_two_sided_utility_allows_tradeoff_without_a_new_worst_seam():
    result = evaluate_composition_path(dict(edges=[edge(.5), edge(.2)]),
        [edge(.4), edge(.21)])
    assert result['admitted']
    assert result['matched_candidate_worst_boundary_cost'] < result['known_baseline_worst_boundary_cost']


def test_geometry_cannot_buy_known_articulated_evidence_loss():
    before = edge(.5)
    before['metrics']['pose_velocity'] = .1
    after = edge(.01)
    assert not evaluate_composition_path(dict(edges=[before]), [after])['admitted']
    after['metrics']['pose_velocity'] = .2
    assert not evaluate_composition_path(dict(edges=[before]), [after])['admitted']
