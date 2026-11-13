"""Finite graph, request lifetime and filesystem tests; no NN/media stand-ins."""
from dataclasses import replace
from contextlib import nullcontext
from time import monotonic
import math
from types import SimpleNamespace as NS
from unittest.mock import patch
import json

import pytest

from crosscut.group_joint_observations import GroupJointObservationSession
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.group_request_work import GroupRequestWork, GroupWorkPolicy
from crosscut.group_scene_epoch_program import compile_common_clock_camera_queries
from crosscut.group_scene_owned_program import freeze_owned_scene_program
from crosscut.group_source_changing_owner import NativeFamilySensors, NativeMotionDemand, NativeMotionInventory, NativeMotionViews
from crosscut.group_review_recorded_index import RecordedGroupReviewIndex
from crosscut.group_whole_native_cache import WholeNativeFactCache
from crosscut.group_whole_programme_native import (
    _camera_key, _recipe_projection_plan, _unknown_flank, WholeNativeRecipe,
    WholeNativeInventory, whole_native_edge_cost, freeze_native_projection_frontier,
    freeze_whole_native_inventory, ordered_whole_actor_seeds,
)
from crosscut.group_cached_programme_inventory import build_cached_programme_inventory
from crosscut.group_native_motion_programme import realize_native_motion_programme
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.group_whole_programme_owner import (
    freeze_whole_native_demands, freeze_whole_sensing_recipes, freeze_whole_completion_packages,
    freeze_whole_completion_recipe_groups,
    union_cached_whole_demands,
)
from crosscut.group_whole_programme_search import (
    ProgrammeOpportunity, ProgrammeRun, ProgrammeSearchBudget, select_whole_programme,
)
from tests.test_group_cached_programme_inventory import inventory
from tests.test_group_request_work import controller, rows
from tests.test_scene_role_epoch_program import fixture


def test_sensing_frontier_is_source_balanced_and_order_independent():
    plan, domain, cached = inventory()
    selected = freeze_whole_sensing_recipes(cached, fps=plan.fps, maximum_recipes=8)
    assert {r.run.request_source_index for r in selected} == set(domain.eligible_indices)
    reverse = replace(cached, recipes=tuple(reversed(cached.recipes)))
    assert freeze_whole_sensing_recipes(reverse, fps=plan.fps, maximum_recipes=8) == selected
    assert len(cached.recipes) > len(selected)  # Sensing does not delete search runs.


def test_sensing_frontier_contains_measurable_cross_source_incident_pairs():
    plan, _, cached=inventory()
    frontier=freeze_whole_sensing_recipes(cached,fps=plan.fps,maximum_recipes=24)
    assert any(a.run.end_output_frame==b.run.first_output_frame
        and a.run.request_source_index!=b.run.request_source_index
        for a in frontier for b in frontier)
    assert len({r.run.identity for r in frontier})==len(frontier)
    assert all(r in cached.recipes for r in frontier)


@pytest.mark.parametrize('fps,expected', [(24., '53'), (30., '66'), (60., '132')])
def test_sensing_duration_uses_actual_output_fps(fps, expected):
    def numeric(end):
        return NS(run=ProgrammeRun(str(end), 0, end, 0, 0, None, 'neutral', 0, 0.))
    cached = NS(recipes=tuple(numeric(n) for n in (53, 66, 132)))
    assert freeze_whole_sensing_recipes(cached, fps=fps, maximum_recipes=1)[0].run.identity == expected


@pytest.mark.parametrize('fps', [True, 0., -1., float('nan'), float('inf')])
def test_invalid_fps_is_rejected_before_sensing(fps):
    with pytest.raises(ValueError):
        freeze_whole_sensing_recipes(NS(recipes=()), fps=fps)


def test_frozen_native_union_keeps_exact_addresses_and_unknown_unrequested_count():
    plan, domain, cached = inventory()
    sensing = freeze_whole_sensing_recipes(cached, fps=plan.fps, maximum_recipes=12)
    demands, census = freeze_whole_native_demands(cached, sensing_recipes=sensing, maximum_demands=8)
    assert len(demands) == len({d.native_key for d in demands}) == 8
    assert {d.source_index for d in demands} == set(domain.eligible_indices)
    assert census['unrequested_native_inputs'] == census['physical_union_count'] - 8
    assert census['selection_scores_used'] is False
    for d in demands:
        assert any(r.run.request_source_index == d.source_index
            and r.run.first_output_frame <= d.output_tick < r.run.end_output_frame
            and r.native_frame_indices[d.output_tick-r.run.first_output_frame] == d.native_frame_index
            for r in sensing)


def test_coherent_sensing_bridges_exact_native_samples_not_invented_poses():
    plan, _, cached = inventory()
    sensing=freeze_whole_sensing_recipes(cached,fps=plan.fps,maximum_recipes=12)
    demands,_=freeze_whole_native_demands(cached,sensing_recipes=sensing)
    packages=freeze_whole_completion_packages(sensing,demands)
    assert len(packages)==len(sensing)
    by_id={r.run.identity:r for r in sensing}
    for package in packages:
        recipe=by_id[package.identity]
        native=tuple(k[1] for k in package.native_keys)
        assert native[0]==recipe.native_frame_indices[0]
        assert native[-1]==recipe.native_frame_indices[-1]
        assert max((b-a for a,b in zip(native,native[1:])),default=0)<=12
        assert all(n in recipe.native_frame_indices for n in native)


def numeric_joint_recipe_fixture():
    # Two source samplers and adjacent 66-frame holds; genuinely UNKNOWN data.
    def recipe(source,a,b):
        run=ProgrammeRun(f'physical-{source}-{a}-{b}',a,b,source,0,None,'neutral',0,0.)
        return NS(run=run,native_frame_indices=tuple(range(a,b)))
    holds=tuple(recipe(source,a,b) for source in range(2) for a,b in ((0,66),(66,132)))
    comparisons=tuple(recipe(source,0,132) for source in range(2))
    return NS(recipes=(*holds,*comparisons)),holds


def test_joint_cut_and_continuous_comparator_share_one_precap_dependency_union():
    cached,sensing=numeric_joint_recipe_fixture()
    groups=freeze_whole_completion_recipe_groups(cached,sensing)
    assert len(groups)==2
    reverse=NS(recipes=tuple(reversed(cached.recipes)))
    assert groups==freeze_whole_completion_recipe_groups(reverse,tuple(reversed(sensing)))
    demands,census=freeze_whole_native_demands(cached,sensing_recipes=sensing,
        completion_recipe_groups=groups)
    packages=freeze_whole_completion_packages(sensing,demands,recipe_groups=groups)
    for package,(_,recipes) in zip(packages,groups,strict=True):
        left,right,hold=recipes
        assert left.run.end_output_frame==right.run.first_output_frame==66
        assert left.run.request_source_index!=right.run.request_source_index
        assert hold.run.first_output_frame==0 and hold.run.end_output_frame==132
        assert set(package.native_keys[:6])=={
            *((left.run.request_source_index,n,None) for n in left.native_frame_indices[-3:]),
            *((right.run.request_source_index,n,None) for n in right.native_frame_indices[:3])}
        assert not package.unrequested_native_keys
        assert set(package.native_keys)<={d.native_key for d in demands}
    assert census['unrequested_native_inputs']==0


def test_input_cap_does_not_shrink_original_package_requirements():
    cached,sensing=numeric_joint_recipe_fixture()
    groups=freeze_whole_completion_recipe_groups(cached,sensing)
    all_inputs,_=freeze_whole_native_demands(cached,sensing_recipes=sensing,
        completion_recipe_groups=groups)
    capped,census=freeze_whole_native_demands(cached,sensing_recipes=sensing,
        completion_recipe_groups=groups,maximum_demands=1)
    full=freeze_whole_completion_packages(sensing,all_inputs,recipe_groups=groups)
    partial=freeze_whole_completion_packages(sensing,capped,recipe_groups=groups)
    assert census['unrequested_native_inputs']>0
    for a,b in zip(full,partial,strict=True):
        assert a.identity==b.identity and a.native_keys==b.native_keys
        assert len(b.native_keys)>1 and b.unrequested_native_keys
        assert set(b.unrequested_native_keys)==set(b.native_keys)-{d.native_key for d in capped}


def test_search_journals_each_numeric_layer_not_every_edge(tmp_path):
    _, owner, _ = controller(tmp_path)
    opportunities = tuple(ProgrammeOpportunity(t, ('test',)) for t in (0, 2, 4, 6))
    candidates = tuple(ProgrammeRun(f'{a}-{b}-{s}', a, b, s, 0, None, 'neutral', 0, 0.)
        for a, b in ((0, 2), (0, 4), (2, 4), (2, 6), (4, 6)) for s in range(8))
    with patch('crosscut.first_result_controller.enforce_review_first_deadline', return_value=nullcontext()), \
            GroupRequestWork(owner, journal_path=tmp_path/'work.jsonl',
                policy=GroupWorkPolicy(whole_programme=True)) as work:
        def operation():
            result = select_whole_programme(opportunities=opportunities, runs=candidates,
                transition_cost=lambda history, incoming: 0.,
                budget=ProgrammeSearchBudget(beam_width=8), work_port=work.lease.active)
            assert work.lease.active.charged['edges'] == result.expansions
            assert work.lease.active.charged['native_completion_passes'] == 0
            assert work.lease.active.charged['model_calls'] == 0
            return result
        result = work.run('selected_completion', operation)
        assert result.expansions > 100
    charges = [r for r in rows(work) if r['kind'] == 'work_admitted']
    assert len(charges) == 2 * (len(opportunities)-1)


def test_whole_policy_changes_only_numeric_edge_capacity():
    local = GroupWorkPolicy().allocations(600.)
    whole = GroupWorkPolicy(whole_programme=True).allocations(600.)
    for phase in local:
        assert local[phase].wall_seconds == whole[phase].wall_seconds
        for resource, count in local[phase].resources.items():
            assert whole[phase].resources[resource] == (
                2000000 if phase == 'selected_completion' and resource == 'edges' else count)


def unknown_cost_bank():
    # Pure numeric UNKNOWN recipes. This is cost accounting, not fake NN data.
    plan, domain, cached=inventory()
    recipes=tuple(WholeNativeRecipe(r,r.run,r.side,None,None,
        _unknown_flank(r.run.first_output_frame,r.run.end_output_frame,True),
        _unknown_flank(r.run.first_output_frame,r.run.end_output_frame,False),1.,0.)
        for r in cached.recipes)
    return WholeNativeInventory(plan,domain,cached.opportunities,recipes,(),0,0)


def test_unknown_hold_cost_includes_the_opening_in_both_conditions():
    bank=unknown_cost_bank();plan=bank.original
    run=next(r.run for r in bank.recipes if r.run.first_output_frame==0)
    control=whole_native_edge_cost(bank,actor_action_enabled=False)
    enabled=whole_native_edge_cost(bank,actor_action_enabled=True)
    assert enabled((),run)==control((),run)==pytest.approx(
        .35*(run.end_output_frame-run.first_output_frame)/plan.fps)
    assert bank.evidence()['connected_actor_paths']==0


def test_all_unknown_ab_has_identical_every_edge_and_whole_plan():
    bank=unknown_cost_bank()
    control=whole_native_edge_cost(bank,actor_action_enabled=False)
    enabled=whole_native_edge_cost(bank,actor_action_enabled=True)
    for incoming in bank.recipes:
        assert control((),incoming.run)==enabled((),incoming.run)
        for previous in bank.recipes:
            if previous.run.end_output_frame==incoming.run.first_output_frame:
                assert control((previous.run,),incoming.run)==enabled((previous.run,),incoming.run)
    args=dict(opportunities=bank.opportunities,runs=tuple(r.run for r in bank.recipes),
        budget=ProgrammeSearchBudget(beam_width=8))
    a=select_whole_programme(**args,transition_cost=control)
    b=select_whole_programme(**args,transition_cost=enabled)
    assert a.runs==b.runs and a.cost==b.cost


def test_removing_a_difficult_numeric_joint_cannot_improve_action_cost():
    # A numerical cost counterexample, not fabricated model/media evidence.
    bank=unknown_cost_bank()
    left=next(r for r in bank.recipes if r.run.first_output_frame==0)
    right=next(r for r in bank.recipes if r.run.first_output_frame==left.run.end_output_frame
        and r.run.request_source_index!=left.run.request_source_index)
    changed={left.run.identity,right.run.identity}
    def flank(recipe,entry,drop):
        rows=recipe.entry if entry else recipe.exit
        return tuple((tick,None,({} if drop else {5:(float(tick%7)/10.,.2)}),
            ('numeric-clock','numeric-source','numeric-camera',0,tick),None,False)
            for tick,*_ in rows)
    observed=replace(bank,recipes=tuple(replace(r,entry=flank(r,True,False),
        exit=flank(r,False,False)) if r.run.identity in changed else r for r in bank.recipes))
    missing=replace(observed,recipes=tuple(replace(r,entry=flank(r,True,True),
        exit=flank(r,False,True)) if r.run.identity in changed else r for r in observed.recipes))
    a=whole_native_edge_cost(observed,actor_action_enabled=False)
    b=whole_native_edge_cost(observed,actor_action_enabled=True)
    erased=whole_native_edge_cost(missing,actor_action_enabled=True)
    assert b((left.run,),right.run)<=erased((left.run,),right.run)==a((left.run,),right.run)


def test_numeric_focal_seed_policy_is_detector_id_permutation_equivariant():
    # Numeric sorting policy only, NOT synthetic detector/model observations.
    candidates=tuple(dict(descriptor=NS(frame_local_head_id=i,face_height=.03*(i+1),
        face_width=.02*(i+1),head_center=(i/10.,.4),pose_shape=()),vector=(float(i),))
        for i in range(6))
    selected=ordered_whole_actor_seeds(candidates,maximum_actor_seeds=6)
    assert len(selected)==6 and selected[0]['descriptor'].frame_local_head_id==5
    permutation=(5,0,4,1,3,2)
    changed=tuple(dict(v,descriptor=NS(**{**vars(v['descriptor']),
        'frame_local_head_id':permutation[i]})) for i,v in enumerate(candidates))
    reordered=ordered_whole_actor_seeds(tuple(reversed(changed)),maximum_actor_seeds=6)
    assert tuple(v['vector'] for v in selected)==tuple(v['vector'] for v in reordered)
    assert ordered_whole_actor_seeds((),maximum_actor_seeds=6)==()


@pytest.mark.parametrize('center', [(0.95, 0.95), (0.05, 0.05), (0.5, 0.5)])
def test_declared_camera_target_is_inside_the_real_renderer_domain(center):
    # Numeric target geometry only, not an invented model detection.
    key=_camera_key(dict(height=.10,landmarks_xy=(center,)*5),.30)
    assert .25<=key.center_x<=.75 and .25<=key.center_y<=.75
    assert .5/key.zoom<=key.center_x<=1-.5/key.zoom
    assert .5/key.zoom<=key.center_y<=1-.5/key.zoom
    assert math.hypot(key.center_x-.5,key.center_y-.5)<=.25+1e-9


def test_readonly_run_observation_is_not_vetoed_by_obsolete_one_frame_neighbours():
    plan, domain, cached = inventory()
    nodes=tuple(ProgrammeOpportunity(t, ('native-camera',)) for t in (0,31,59,120))
    from tests.test_group_cached_programme_inventory import empty_inventory
    cached=build_cached_programme_inventory(original=plan,domain=domain,
        inventories=empty_inventory(domain),opportunities=nodes)
    recipe=next(r for r in cached.recipes if r.run.first_output_frame==31
        and r.run.end_output_frame==59 and r.run.request_source_index==3)
    with pytest.raises(ValueError,match='one-frame'):
        realize_native_motion_programme(original=plan,domain=domain,
            sides=(recipe.side,),selection_scope='variable_programme')
    observed=_recipe_projection_plan(plan,domain,recipe,recipe.side)
    actual=native_frame_clocks_for_output(observed,domain.clocks_by_request,range(31,59))
    assert tuple(actual[t].native_frame_index for t in range(31,59))==recipe.native_frame_indices
    assert all(actual[t].source_index==3 for t in actual)
    assert observed.output_frame_count==plan.output_frame_count
    assert observed.duration==pytest.approx(plan.duration)
    # This observation plan is never handed to final materialization. A joint
    # fully authored programme has no obsolete neighbours and remains valid.
    before=next(r for r in cached.recipes if r.run.first_output_frame==0
        and r.run.end_output_frame==31 and r.run.request_source_index==0)
    after=next(r for r in cached.recipes if r.run.first_output_frame==59
        and r.run.end_output_frame==120 and r.run.request_source_index==5)
    final=realize_native_motion_programme(original=plan,domain=domain,
        sides=(before.side,recipe.side,after.side),selection_scope='variable_programme')
    assert final.render_plan.output_frame_count==plan.output_frame_count


def live_empty_native_fixture(tmp_path):
    context, domain, _ = fixture()
    sources = tuple(tmp_path/f'nonmedia-source-{i}' for i in range(len(domain.request_sources)))
    for source in sources:
        source.write_bytes(b'Filesystem identity fixture only; never decoded.')
    plan = replace(context.incumbent.render_plan, sources=sources)
    context.incumbent.render_plan = plan
    domain = replace(domain, request_sources=sources)
    session = GroupJointObservationSession(domain=domain,
        budget=JointGroupWorkBudget(absolute_deadline=monotonic()+30.))
    sensors = NS(session=session)
    completed = NativeMotionInventory((), (), True, 'TEST_EMPTY_CONSUMED_PASS')
    views = NativeMotionViews(sensors, completed)
    queries = compile_common_clock_camera_queries(context=context, domain=domain,
        original_program=freeze_owned_scene_program(context, domain), candidate_plan=plan,
        output_intervals=((30, 60),)).queries
    return domain, session, sensors, completed, views, queries[0]


def empty_whole_native_fixture(tmp_path):
    # Filesystem and sampler fixtures only: no model results are synthesized.
    domain,session,_,_,views,_=live_empty_native_fixture(tmp_path)
    context,_,_=fixture()
    context.incumbent.render_plan=replace(context.incumbent.render_plan,sources=domain.request_sources)
    from tests.test_group_cached_programme_inventory import empty_inventory
    cached=build_cached_programme_inventory(original=context.incumbent.render_plan,domain=domain,
        inventories=empty_inventory(domain),opportunities=tuple(
            ProgrammeOpportunity(t,('test',)) for t in (0,30,60,90,120)))
    return context,domain,session,views,cached


def test_local_projection_cutoff_keeps_every_physical_duration_and_source(tmp_path):
    context,domain,session,views,cached=empty_whole_native_fixture(tmp_path)
    with session,patch('crosscut.group_whole_programme_native.monotonic',side_effect=(0.,1.)):
        bank=freeze_whole_native_inventory(context=context,domain=domain,cached=cached,views=views,
            sensed_recipe_ids=frozenset(r.run.identity for r in cached.recipes),maximum_seconds=.1)
    assert tuple(r.run for r in bank.recipes)==cached.runs
    assert bank.projection_budget_exhausted is True
    assert bank.evidence()['projection_budget_exclusion_is_actor_unsuitability'] is False
    assert all(r.actor_path is None and r.actor_receipt is None for r in bank.recipes)
    assert {r.run.request_source_index for r in bank.recipes}==set(domain.eligible_indices)
    assert session.budget.counts['model_calls']==0


def test_parent_native_deadline_is_not_a_settleable_local_projection_cutoff(tmp_path):
    context,domain,session,views,cached=empty_whole_native_fixture(tmp_path)
    with session,patch.object(session.budget,'check',side_effect=TimeoutError('parent deadline expired')):
        with pytest.raises(TimeoutError,match='parent deadline expired'):
            freeze_whole_native_inventory(context=context,domain=domain,cached=cached,views=views)
    assert views._signature_source_bindings is None
    assert session.budget.counts['model_calls']==0


def test_completed_projection_claim_cannot_replace_missing_native_facts(tmp_path):
    context,domain,session,views,cached=empty_whole_native_fixture(tmp_path)
    identity=cached.recipes[0].run.identity
    with session:
        with pytest.raises(ValueError,match='unobserved original native requirements'):
            freeze_whole_native_inventory(context=context,domain=domain,cached=cached,views=views,
                completed_joint_recipe_ids=(identity,))
    assert session.budget.counts['model_calls']==0
    assert not views.paths and not views.views


def test_signature_epoch_checks_live_sources_at_both_ends_not_every_tick(tmp_path):
    domain, session, _, _, views, query = live_empty_native_fixture(tmp_path)
    with session, patch.object(session, '_assert_source_identity', wraps=session._assert_source_identity) as actual_stat:
        with views.native_signature_scope():
            for tick in range(30, 60):
                signature = views.native_signature(query, tick)
                assert signature[0] == domain.request_index(query.eligible_source_index)
                assert signature[4] == query.sample_at(tick)[2]
        assert actual_stat.call_count == 2*len(domain.eligible_indices)
        views.native_signature(query, 30)
        assert actual_stat.call_count == 2*len(domain.eligible_indices)+1


def test_absent_recorded_addresses_do_not_exhaust_model_file_checks(tmp_path):
    domain, session, _, _, _, _ = live_empty_native_fixture(tmp_path)
    # Empty real index and real filesystem session, with no models or NN
    # results installed. Absence is not synthetic detector success.
    sensors=NativeFamilySensors.__new__(NativeFamilySensors)
    sensors.session=session
    sensors.raw_rows={};sensors.appearance_rows={};sensors.member_gallery=None
    with session:
        index=RecordedGroupReviewIndex()
        sensors.bind_recorded_motion_index(index,
            source_identities={i:session._assert_source_identity(i) for i in domain.eligible_indices},
            source_content_sha256={i:None for i in domain.eligible_indices})
        demands=tuple(NativeMotionDemand(i,n,n,f'source-{i}','other',require_appearance=True)
            for i in domain.eligible_indices for n in range(30))
        with patch.object(session,'_assert_model_assets',wraps=session._assert_model_assets) as model_stat:
            result=sensors.motion_inventory(demands)
        assert model_stat.call_count==0
        assert all(s.observation is None and s.appearance is None for s in result.supplies)
        assert session.budget.counts['model_calls']==0
    with pytest.raises(ValueError,match='closed'):
        sensors._recorded_motion_supply(demands[0])


def test_recorded_address_presence_is_exact_and_does_not_select_variants():
    index=RecordedGroupReviewIndex()
    args=dict(source=2,native=80,clock='c'*64,identity=(1,2,3,4,5))
    assert not index.contains_address(**args)
    # Address presence is an index-only test, not a fabricated native row.
    index.rows[(2,80,'c'*64,(1,2,3,4,5))]={}
    assert index.contains_address(**args)
    assert not index.contains_address(**dict(args,native=81))
    assert not index.contains_address(**dict(args,identity=(1,2,3,4,6)))
    with pytest.raises(ValueError,match='physical'):
        index.contains_address(**dict(args,native=True))


def test_native_cache_census_keeps_adverse_addresses_and_exact_bindings():
    # Presence-only index fixtures: no raw payload or invented NN result.
    index=RecordedGroupReviewIndex()
    index.rows[(2,80,'c'*64,(1,2,3,4,5))]={}
    index.rows[(2,81,'c'*64,(1,2,3,4,5))]={}
    index.rows[(2,82,'d'*64,(1,2,3,4,5))]={}
    index.rows[(2,83,'c'*64,(1,2,3,4,6))]={}
    args=dict(source=2,clock='c'*64,identity=(1,2,3,4,5))
    assert index.native_addresses_for_binding(**args)==(80,81)
    assert index.native_addresses_for_binding(**dict(args,source=3))==()
    with pytest.raises(ValueError,match='physical'):
        index.native_addresses_for_binding(**dict(args,source=True))


def test_whole_cache_union_reuses_physical_samples_across_duration_ids():
    _,_,cached=inventory()
    short=next(r for r in cached.recipes if r.run.first_output_frame==0)
    source=short.run.request_source_index
    native=short.native_frame_indices[-1]
    long=next(r for r in cached.recipes if r.run.request_source_index==source
        and r.run.first_output_frame==0 and r.run.end_output_frame>short.run.end_output_frame
        and native in r.native_frame_indices)
    addresses={(source,native),(source,999999)}
    demands,census=union_cached_whole_demands(cached,(),indexed_addresses=addresses)
    assert len(demands)==1 and demands[0].native_key==(source,native,None)
    assert census['added_cached_native_addresses']==1
    assert census['cache_address_presence_is_observation_validation'] is False
    assert short.run.identity!=long.run.identity
    reversed_cache=replace(cached,recipes=tuple(reversed(cached.recipes)))
    assert union_cached_whole_demands(reversed_cache,(),indexed_addresses=addresses)==(demands,census)
    again,_=union_cached_whole_demands(cached,demands,indexed_addresses=addresses)
    assert again==demands


def test_native_projection_frontier_keeps_source_and_duration_options_without_nn_scores():
    # Numeric native-sampler keys only, not fabricated detections/appearance.
    plan,domain,cached=inventory()
    keys={(r.run.request_source_index,n) for r in cached.recipes for n in r.native_frame_indices}
    chosen=freeze_native_projection_frontier(cached,available_native_addresses=keys,
        fps=plan.fps,maximum_recipes=12)
    selected=[r for r in cached.recipes if r.run.identity in chosen]
    assert {r.run.request_source_index for r in selected}==set(domain.eligible_indices)
    assert len({r.run.end_output_frame-r.run.first_output_frame for r in selected})>1
    assert freeze_native_projection_frontier(replace(cached,recipes=tuple(reversed(cached.recipes))),
        available_native_addresses=keys,fps=plan.fps,maximum_recipes=12)==chosen
    assert freeze_native_projection_frontier(cached,available_native_addresses=set(),fps=plan.fps)==()
    assert len(cached.recipes)>len(chosen)


def test_signature_epoch_invalidates_changed_source_and_clears_scope(tmp_path):
    domain, session, _, _, views, _ = live_empty_native_fixture(tmp_path)
    with session:
        with pytest.raises(ValueError, match='source'):
            with views.native_signature_scope():
                domain.request_sources[domain.eligible_indices[0]].write_bytes(b'changed source')
        assert views._signature_source_bindings is None


def test_signature_epoch_is_non_nested_and_requires_live_session(tmp_path):
    _, session, _, _, views, _ = live_empty_native_fixture(tmp_path)
    with session:
        with views.native_signature_scope():
            with pytest.raises(ValueError, match='non-nested'):
                with views.native_signature_scope():
                    pytest.fail('nested source binding must not enter')
    with pytest.raises(ValueError, match='live'):
        with views.native_signature_scope():
            pytest.fail('closed session must not reopen an epoch')


def test_empty_completion_does_not_publish_unreadable_fact_archive(tmp_path):
    _, session, sensors, completed, _, _ = live_empty_native_fixture(tmp_path)
    with session:
        cache = WholeNativeFactCache(root=tmp_path, sensors=sensors)
        assert cache.bind_existing() is False
        assert cache.preserve_completed(completed) is False
        assert not cache.path.exists()


def test_fact_cache_does_not_follow_or_change_the_separate_derived_cache_link(tmp_path):
    _, session, sensors, _, _, _ = live_empty_native_fixture(tmp_path)
    target=tmp_path/'another-cache-owner'
    target.mkdir()
    (tmp_path/'derived').symlink_to(target, target_is_directory=True)
    with session:
        cache=WholeNativeFactCache(root=tmp_path,sensors=sensors)
        assert cache.path.parent==tmp_path/'whole-native-facts-v1'
        assert (tmp_path/'derived').is_symlink()
        assert tuple(target.iterdir())==()


def test_fact_cache_rejects_a_changed_source_model_header(tmp_path):
    _, session, sensors, _, _, _ = live_empty_native_fixture(tmp_path)
    with session:
        cache = WholeNativeFactCache(root=tmp_path, sensors=sensors)
        payload = dict(header=dict(cache.header, quality_approved=True), records=[], decode_failures=[])
        cache.path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match='header'):
            cache.bind_existing()
