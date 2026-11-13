"""Global interval ownership tests; explicit NONMEDIA numeric fixtures.

These test the actual programme/union arithmetic, not subject admission,
completion, an app run or the quality of a human performance.
"""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from crosscut.group_actual_shots import visible_group_cut_inventory
from crosscut.render import (RenderPlan, RenderSegment, RenderTransition,
    EditorialTransformTrack, EditorialTransformKeyframe, segment_frame_counts)
from crosscut.group_source_domain import GroupSourceDomain
from crosscut.render_frame_schedule import VerifiedRenderFrameClock


def fixture(*, native_cut=None, fps=30., length=7):
    paths = tuple(Path(f'EXPLICIT_NONMEDIA_OWNED_SOURCE_{i}.mp4') for i in range(8))
    clocks = (VerifiedRenderFrameClock(Fraction(60000, 1001), 10000, 'a'*64),)*8
    indices = (0, 1, 1, 2, 3, 4, 5) + tuple(i % 6 for i in range(length-7))
    plan = RenderPlan(sources=paths, segments=tuple(RenderSegment(source, float(i), 1., float(i))
        for i, source in enumerate(indices)), reference_source=0, aspect_ratio='16:9',
        output=Path('EXPLICIT_NEVER_RENDERED_GLOBAL_UNION.mp4'), fps=fps, width=320, height=180,
        transitions=tuple(RenderTransition(tier='D' if a == b else 'C')
            for a, b in zip(indices, indices[1:])))
    cuts = tuple((native_cut,) if i == 1 and native_cut is not None else () for i in range(8))
    domain = GroupSourceDomain(paths, tuple(range(8)), (0.,)*8, (None,)*8, cuts, clocks)
    context = NS(incumbent=NS(render_plan=plan,
        visual_timeline=NS(choices=[NS(shot_type=kind) for kind in
            ('medium', 'group_wide', 'close_up', 'medium', 'medium', 'group_wide', 'medium')
                + ('medium',)*(length-7)]),
        face_match_plan=NS(segments=[NS(alignment_subject=None, applied=False) for _ in indices]),
        boundary_look_match=NS(decisions=())))
    return context, domain


def api():
    from crosscut import group_scene_owned_program as owned
    return owned


def test_mixed_roles_are_epochs_of_one_globally_identified_actual_shot():
    context, domain = fixture()
    program = api().freeze_owned_scene_program(context, domain)
    assert len(program.units) == 6
    mixed = program.units[1]
    assert (mixed.first_output_frame, mixed.end_output_frame) == (30, 90)
    assert [e['role'] for e in mixed.role_program['epochs']] == [
        'observed_body_composition', 'focal_head']
    assert len({u.unit_id for u in program.units}) == 6
    data = program.to_dict()
    assert data['original_native_part_lifecycle_frozen'] is False
    assert data['candidate_admitted'] is False
    assert data['quality_approved'] is False


def test_actual_subcell_cuts_have_exact_units_without_splitting_render_cells():
    context, domain = fixture(native_cut=1.51)
    program = api().freeze_owned_scene_program(context, domain)
    plan = context.incumbent.render_plan
    assert len(program.units) == 7
    assert [(u.first_output_frame, u.end_output_frame) for u in program.units[1:3]] == [(30, 46), (46, 90)]
    assert segment_frame_counts(plan) == (30,)*7
    assert program.to_dict()['render_topology_materialized'] is False


def test_mutable_ids_are_explicit_global_and_not_positional_middle_slots():
    context, domain = fixture()
    program = api().freeze_owned_scene_program(context, domain)
    ids = (program.units[0].unit_id, program.units[3].unit_id, program.units[-1].unit_id)
    allocation = api().own_scene_intervals(program, mutable_unit_ids=ids,
        maximum_mutable_units=6, maximum_mutable_frames=180)
    assert allocation.mutable_unit_ids == ids
    assert allocation.mutable_output_intervals == ((0, 30), (120, 150), (180, 210))
    assert allocation.mutable_frame_count == 90
    assert allocation.to_dict()['positional_fixed_endpoints_assumed'] is False


def test_failed_fixed_neighbor_can_become_mutable_under_same_explicit_caps():
    context, domain = fixture()
    program = api().freeze_owned_scene_program(context, domain)
    allocation = api().own_scene_intervals(program, mutable_unit_ids=(program.units[2].unit_id,),
        maximum_mutable_units=3, maximum_mutable_frames=120)
    expanded = api().expand_owned_scene_intervals(allocation,
        blocked_fixed_unit_ids=(program.units[1].unit_id, program.units[3].unit_id))
    assert expanded.mutable_unit_ids == tuple(u.unit_id for u in program.units[1:4])
    assert expanded.mutable_frame_count == 120
    assert expanded.program is program
    assert expanded.maximum_mutable_frames == allocation.maximum_mutable_frames
    assert expanded.to_dict()['original_duties_need_freezing_before_donor_sensing'] is True
    with pytest.raises(ValueError, match='budget'):
        api().expand_owned_scene_intervals(expanded, blocked_fixed_unit_ids=(program.units[4].unit_id,))
    with pytest.raises(ValueError, match='incident fixed'):
        api().expand_owned_scene_intervals(allocation, blocked_fixed_unit_ids=(program.units[-1].unit_id,))


@pytest.mark.parametrize('ids', [(), ('missing',), ('duplicate', 'duplicate')])
def test_missing_duplicate_or_empty_mutable_domain_cannot_buy_ownership(ids):
    context, domain = fixture()
    program = api().freeze_owned_scene_program(context, domain)
    if ids and ids[0] == 'duplicate': ids = (program.units[1].unit_id,)*2
    with pytest.raises(ValueError):
        api().own_scene_intervals(program, mutable_unit_ids=ids,
            maximum_mutable_units=6, maximum_mutable_frames=180)


def test_external_original_role_and_clock_preimage_is_rederived_not_self_signed():
    context, domain = fixture()
    program = api().freeze_owned_scene_program(context, domain)
    data = deepcopy(program.to_dict())
    data['units'][1]['original_role_program']['epochs'][0]['role'] = 'focal_head'
    with pytest.raises(ValueError, match='external original'):
        api().validate_owned_scene_program(data, context, domain)
    context.incumbent.face_match_plan.segments[1].alignment_subject = 'upper_body'
    with pytest.raises(ValueError, match='external original'):
        api().validate_owned_scene_program(program.to_dict(), context, domain)


def camera_plan(context, *, change_cells, source_index=7):
    old = context.incumbent.render_plan
    return replace(old, segments=tuple(replace(s, source_index=source_index)
        if i in change_cells else s for i, s in enumerate(old.segments)))


def test_three_distant_whole_cameras_are_union_before_one_support_derivation():
    context, domain = fixture(length=13)
    owned = api(); program = owned.freeze_owned_scene_program(context, domain)
    ids = tuple(program.units[i].unit_id for i in (0, 5, 11))
    allocation = owned.own_scene_intervals(program, mutable_unit_ids=ids,
        maximum_mutable_units=6, maximum_mutable_frames=180)
    candidate = camera_plan(context, change_cells=(0, 6, 12))
    cameras = tuple(owned.camera_unit_from_plan(program=program, unit_id=unit_id,
        plan=candidate, domain=domain) for unit_id in ids)
    result = owned.assemble_owned_scene_union(context=context, domain=domain,
        allocation=allocation, cameras=cameras)
    assert result.changed_segment_indices == (0, 6, 12)
    assert [result.plan.segments[i].source_index for i in (0, 6, 12)] == [7, 7, 7]
    assert all(s == context.incumbent.render_plan.segments[i]
        for i, s in enumerate(result.plan.segments) if i not in (0, 6, 12))
    assert result.dependency_support['numeric_write_cells'] == [0, 6, 12]
    assert result.dependency_support['descriptor_refresh_intervals'] == [[0, 2], [5, 8], [11, 13]]
    assert result.to_dict()['one_numeric_union_before_completion'] is True
    assert result.to_dict()['full_bundle_created'] is False
    assert result.to_dict()['quality_approved'] is False
    assert [(e.first_incoming_output_frame, e.kind) for e in visible_group_cut_inventory(result.plan, domain)] == \
        [(e.first_incoming_output_frame, e.kind) for e in visible_group_cut_inventory(context.incumbent.render_plan, domain)]


def test_whole_mixed_camera_keeps_camera_phase_canvas_and_native_clock_across_roles():
    from crosscut.shot_context_camera import slice_shot_camera_track
    context, domain = fixture(fps=30000/1001)
    owned = api(); program = owned.freeze_owned_scene_program(context, domain)
    unit = program.units[1]; old = context.incumbent.render_plan
    allocation = owned.own_scene_intervals(program, mutable_unit_ids=(unit.unit_id,),
        maximum_mutable_units=1, maximum_mutable_frames=90)
    track = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.1, .5, .5),
        EditorialTransformKeyframe(2.01, 1.35, .54, .47, .4)), sampling_canvas_zoom=1.4)
    segments = list(old.segments)
    for index, offset in ((1, 0), (2, 30)):
        segments[index] = replace(segments[index], source_index=7,
            editorial_transform=slice_shot_camera_track(track, frame_offset=offset,
                frame_count=30, fps=old.fps))
    candidate = replace(old, segments=tuple(segments))
    camera = owned.camera_unit_from_plan(program=program, unit_id=unit.unit_id,
        plan=candidate, domain=domain)
    assert camera.role_epoch_intervals == ((30, 60), (60, 90))
    result = owned.assemble_owned_scene_union(context=context, domain=domain,
        allocation=allocation, cameras=(camera,))
    assert result.changed_segment_indices == (1, 2)
    assert result.cameras[0].native_frame_indices == camera.native_frame_indices
    for index, offset in ((1, 0), (2, 30)):
        for tick in range(30):
            actual = result.plan.segments[index].editorial_transform.evaluate(tick/old.fps)
            expected = track.evaluate((offset+tick)/old.fps)
            assert actual.zoom == pytest.approx(expected.zoom, abs=1e-9)
            assert result.plan.segments[index].editorial_transform.effective_sampling_canvas_zoom == 1.4


@pytest.mark.parametrize('mutation', ['native', 'fixed_pixels', 'missing', 'duplicate', 'cadence', 'header'])
def test_global_union_rejects_unowned_or_inconsistent_camera_preimages(mutation):
    context, domain = fixture()
    owned = api(); program = owned.freeze_owned_scene_program(context, domain)
    ids = (program.units[2].unit_id,)
    allocation = owned.own_scene_intervals(program, mutable_unit_ids=ids,
        maximum_mutable_units=1, maximum_mutable_frames=90)
    candidate = camera_plan(context, change_cells=(3,))
    if mutation == 'fixed_pixels':
        context.incumbent.render_plan = replace(context.incumbent.render_plan,
            segments=(replace(context.incumbent.render_plan.segments[0], editorial_transform=
                EditorialTransformTrack((EditorialTransformKeyframe(0., 1.2),))),
                *context.incumbent.render_plan.segments[1:]))
    elif mutation == 'cadence': candidate = camera_plan(context, change_cells=(3,), source_index=3)
    elif mutation == 'header': candidate = replace(candidate, master_audio_start=.2)
    if mutation == 'header':
        with pytest.raises(ValueError, match='header'):
            owned.camera_unit_from_plan(program=program, unit_id=ids[0], plan=candidate, domain=domain)
        return
    camera = owned.camera_unit_from_plan(program=program, unit_id=ids[0], plan=candidate, domain=domain)
    cameras = (camera,)
    if mutation == 'native': cameras = (replace(camera, native_frame_indices=tuple(n+1 for n in camera.native_frame_indices)),)
    elif mutation == 'missing': cameras = ()
    elif mutation == 'duplicate': cameras = (camera, camera)
    with pytest.raises(ValueError):
        owned.assemble_owned_scene_union(context=context, domain=domain, allocation=allocation, cameras=cameras)


def test_subcell_numeric_ownership_is_not_promoted_to_cell_aligned_mutation():
    context, domain = fixture(native_cut=1.51)
    owned = api(); program = owned.freeze_owned_scene_program(context, domain)
    allocation = owned.own_scene_intervals(program, mutable_unit_ids=(program.units[1].unit_id,),
        maximum_mutable_units=1, maximum_mutable_frames=90)
    with pytest.raises(ValueError, match='topology-aware'):
        owned.camera_unit_from_plan(program=program, unit_id=allocation.mutable_unit_ids[0],
            plan=context.incumbent.render_plan, domain=domain)


def test_unknown_original_roles_are_retained_not_donor_relabelled():
    context, domain = fixture()
    context.incumbent.face_match_plan.segments[1].applied = True
    program = api().freeze_owned_scene_program(context, domain)
    epoch = program.units[1].role_program['epochs'][0]
    assert epoch['role'] is None and epoch['unresolved_reason']
    assert program.to_dict()['original_native_part_lifecycle_frozen'] is False


def test_source_clock_or_role_changes_in_external_original_invalidate_unit_ids():
    context, domain = fixture()
    owned = api(); before = owned.freeze_owned_scene_program(context, domain)
    context.incumbent.face_match_plan.segments[1].alignment_subject = 'upper_body'
    after = owned.freeze_owned_scene_program(context, domain)
    assert before.units[1].unit_id != after.units[1].unit_id
    changed_domain = replace(domain, audio_offsets=(.001, *domain.audio_offsets[1:]))
    with pytest.raises(ValueError, match='external original'):
        owned.validate_owned_scene_program(after.to_dict(), context, changed_domain)


def test_adjacent_cameras_can_swap_atomically_without_valid_solo_cuts():
    context, domain = fixture()
    owned = api(); program = owned.freeze_owned_scene_program(context, domain)
    ids = tuple(program.units[i].unit_id for i in (2, 3))
    allocation = owned.own_scene_intervals(program, mutable_unit_ids=ids,
        maximum_mutable_units=2, maximum_mutable_frames=90)
    old = context.incumbent.render_plan
    candidate = replace(old, segments=tuple(replace(s, source_index=3 if i == 3 else 2)
        if i in (3, 4) else s for i, s in enumerate(old.segments)))
    cameras = tuple(owned.camera_unit_from_plan(program=program, unit_id=unit_id,
        plan=candidate, domain=domain) for unit_id in ids)
    result = owned.assemble_owned_scene_union(context=context, domain=domain,
        allocation=allocation, cameras=cameras)
    assert result.changed_segment_indices == (3, 4)
    assert [s.source_index for s in result.plan.segments[3:5]] == [3, 2]


def test_retained_look_reads_are_part_of_global_union_scope_before_completion():
    context, domain = fixture(length=13)
    context.incumbent.boundary_look_match.decisions = (NS(boundary_index=8, applied=True),)
    owned = api(); program = owned.freeze_owned_scene_program(context, domain)
    ids = (program.units[0].unit_id,)
    allocation = owned.own_scene_intervals(program, mutable_unit_ids=ids,
        maximum_mutable_units=1, maximum_mutable_frames=90)
    candidate = camera_plan(context, change_cells=(0,))
    camera = owned.camera_unit_from_plan(program=program, unit_id=ids[0], plan=candidate, domain=domain)
    result = owned.assemble_owned_scene_union(context=context, domain=domain,
        allocation=allocation, cameras=(camera,))
    assert result.dependency_support['retained_look_proof_edges'] == [8]
    # Reading original cell 1 closes over its unchanged continuous (1,2)
    # renderer unit. This is a read, not an additional mutable/refresh cell.
    assert set(result.dependency_support['read_cells']) == {0, 1, 2, 8, 9}
    assert set(result.dependency_support['write_cells']) == {0, 1}
