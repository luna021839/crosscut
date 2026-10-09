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
from crosscut.group_whole_programme_native import _camera_key, _recipe_projection_plan
from crosscut.group_cached_programme_inventory import build_cached_programme_inventory
from crosscut.group_native_motion_programme import realize_native_motion_programme
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.group_whole_programme_owner import (
    freeze_whole_native_demands, freeze_whole_sensing_recipes, freeze_whole_completion_packages,
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
