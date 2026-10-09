"""Numeric physical sampler inventory, not detector/model/quality tests."""
from dataclasses import replace

import pytest

from crosscut.group_cached_native_observations import CachedNativeInventory
from crosscut.group_cached_programme_inventory import (
    CachedProgrammeRecipe, build_cached_programme_inventory, cached_composition_edge_cost,
    cached_programme_physical_compatibility,
)
from crosscut.group_native_motion_programme import realize_native_motion_programme
from crosscut.group_whole_programme_search import (
    ProgrammeOpportunity, ProgrammeRun, ProgrammeSearchBudget, programme_opportunities, select_whole_programme,
)
from crosscut.render import RenderSegment
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.render_frame_schedule import sampler_frame_schedule
from tests.test_scene_role_epoch_program import fixture


def empty_inventory(domain):
    return {source: CachedNativeInventory((), 0, 0, 0, 0) for source in domain.eligible_indices}


def inventory(*, protected=(), extra=(), cuts=None):
    context, domain, _ = fixture()
    if cuts is not None:
        domain = replace(domain, scene_cut_times=cuts)
    plan = context.incumbent.render_plan
    nodes = programme_opportunities(output_frame_count=plan.output_frame_count, fps=plan.fps,
        beat_seconds=(0., .6, 1.2, 1.8, 2.4, 3., 3.6, 4.),
        protected_spans=protected, protected_recipe_boundaries=extra)
    result = build_cached_programme_inventory(original=plan, domain=domain,
        inventories=empty_inventory(domain), opportunities=nodes, protected_spans=protected)
    return plan, domain, result


def test_unknown_cache_stays_eligible_and_does_not_inherit_old_cell_grid():
    plan, domain, result = inventory()
    assert result.recipes
    assert {r.run.request_source_index for r in result.recipes} == set(domain.eligible_indices)
    assert any(r.run.end_output_frame == 18 for r in result.recipes)
    assert all(r.run.actor_path_identity is None for r in result.recipes)
    assert all(r.observed_fraction == 0 for r in result.recipes)
    assert result.evidence()['pose_is_face_owned'] is False
    assert result.evidence()['original_cells_are_not_the_free_edit_grid']


def test_candidate_grid_is_the_actual_selected_renderer_grid():
    plan, domain, result = inventory()
    recipe = next(r for r in result.recipes if r.run.first_output_frame == 18
        and r.run.end_output_frame == 72 and r.run.request_source_index == 2)
    realized = realize_native_motion_programme(original=plan, domain=domain,
        sides=(recipe.side,), selection_scope='variable_programme')
    actual = native_frame_clocks_for_output(realized.render_plan, domain.clocks_by_request,
        range(recipe.run.first_output_frame, recipe.run.end_output_frame))
    assert tuple(s.native_frame_index for s in actual.values()) == recipe.native_frame_indices


def test_broadcaster_cut_is_physical_and_not_an_unknown_actor_veto():
    context, domain, _ = fixture()
    cuts = list(domain.scene_cut_times)
    cuts[domain.eligible_index(2)] = (1.,)
    _, _, result = inventory(cuts=tuple(cuts))
    assert result.physical_cut_crossings > 0
    cut = 60  # Verified native clock is 60 fps.
    assert all(not (r.native_frame_indices[0] < cut <= r.native_frame_indices[-1])
        for r in result.recipes if r.run.request_source_index == 2)
    assert any(r.run.request_source_index == 2 for r in result.recipes)


def test_protected_graph_edges_keep_old_recipes_only_inside_protection():
    plan, domain, result = inventory(protected=((40, 80),), extra=(60,))
    frozen = [r for r in result.recipes if r.run.protected_reference]
    assert [(r.run.first_output_frame, r.run.end_output_frame) for r in frozen] == [(40, 60), (60, 80)]
    assert all(r.side is None for r in frozen)
    selection = select_whole_programme(opportunities=result.opportunities, runs=result.runs,
        transition_cost=cached_composition_edge_cost(result.recipes), protected_spans=((40, 80),),
        budget=ProgrammeSearchBudget(beam_width=4))
    assert [r.identity for r in selection.runs if r.protected_reference] == [r.run.identity for r in frozen]


def test_inventory_cancellation_and_declared_bound_do_not_trigger_retries():
    context, domain, _ = fixture()
    args = dict(original=context.incumbent.render_plan, domain=domain,
        inventories=empty_inventory(domain), opportunities=programme_opportunities(
            output_frame_count=120, fps=30, beat_seconds=(0., 1., 2., 3., 4.)))
    with pytest.raises(InterruptedError):
        build_cached_programme_inventory(**args, cancelled=lambda: True)
    with pytest.raises(ValueError, match='original bound'):
        build_cached_programme_inventory(**args, maximum_runs=1)


def test_protected_old_cell_boundary_must_not_be_fabricated_after_search():
    with pytest.raises(ValueError, match='protected source recipe'):
        inventory(protected=((40, 80),))


def test_coalescing_cannot_rewrite_frozen_candidate_pixels_after_selection():
    context, domain, _ = fixture()
    original = context.incumbent.render_plan
    # Two individually valid nearly linear spans straddle native half frames.
    # Joining them changes their exact native IDs despite a tiny clock error.
    first = RenderSegment(0, .0083328, 10/30, 0., source_duration=10/30+.0000333333333)
    second = RenderSegment(0, first.source_end, 10/30, 10/30, source_duration=10/30-.0000333333333)
    combined = RenderSegment(0, first.source_start, 20/30, 0., source_duration=20/30)

    def numeric_recipe(name, low, high, segment, cost=0.):
        grid = sampler_frame_schedule(source_start=segment.source_start,
            source_playback_rate=segment.source_playback_rate, native_fps=60., output_fps=30.,
            frame_count=high-low, source_frame_count=domain.clocks_by_request[0].frame_count).source_frame_indices
        return CachedProgrammeRecipe(ProgrammeRun(name, low, high, 0, 0, None, 'neutral', 0, cost),
            None, grid, (None,)*(high-low), aligned_segment=segment)

    recipes = (numeric_recipe('first', 0, 10, first), numeric_recipe('second', 10, 20, second),
        numeric_recipe('combined', 0, 20, combined, cost=1.))
    compatible = cached_programme_physical_compatibility(original=original, domain=domain, recipes=recipes)
    assert compatible((), recipes[0].run)
    assert not compatible((recipes[0].run,), recipes[1].run)
    selection = select_whole_programme(opportunities=tuple(ProgrammeOpportunity(t, ('test',)) for t in (0,10,20)),
        runs=tuple(r.run for r in recipes), transition_cost=lambda history, incoming: 0.,
        physically_compatible=compatible, budget=ProgrammeSearchBudget(beam_width=None))
    assert [r.identity for r in selection.runs] == ['combined']
    assert selection.physically_incompatible_paths == 1
