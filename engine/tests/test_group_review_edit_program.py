"""Numeric review topology tests. Never real member/group media quality."""
from dataclasses import replace

import pytest

from engine.tests.test_scene_owned_program import fixture
from crosscut.group_review_edit_program import own_review_edit_program
from crosscut.group_scene_owned_program import freeze_owned_scene_program
from crosscut.render import RenderSegment


def own(context, domain, plan, *, mutable=((30, 90),), protected=((150, 180),), cap=60):
    return own_review_edit_program(context=context, domain=domain,
        original_program=freeze_owned_scene_program(context, domain), candidate_plan=plan,
        mutable_output_intervals=mutable, protected_output_intervals=protected,
        maximum_changed_frames=cap, maximum_visible_cuts=20)


def test_review_can_insert_cut_inside_original_mixed_role_cell():
    context, domain = fixture()
    old = context.incumbent.render_plan
    parts = (RenderSegment(7, 1., .5, 1.), RenderSegment(6, 1.5, .5, 1.5))
    candidate = replace(old, segments=(old.segments[0], *parts, *old.segments[2:]), transitions=())
    result = own(context, domain, candidate)
    data = result.to_dict()
    assert result.changed_output_intervals == ((30, 60),)
    assert any(r['first_incoming_output_frame'] == 45 for r in data['candidate_visible_events_added'])
    queries = [q for q in data['original_role_queries'] if 30 <= q['first'] < 90]
    assert [(q['first'], q['end'], q['role']) for q in queries] == [
        (30, 45, 'observed_body_composition'), (45, 60, 'observed_body_composition'),
        (60, 90, 'focal_head')]
    assert data['normal_complete_bundle_created'] is False
    assert data['quality_approved'] is False


def test_removed_visible_native_wide_insert_remains_original_source_fact():
    context, domain = fixture(native_cut=1.51)
    old = context.incumbent.render_plan
    candidate = replace(old, segments=(old.segments[0],
        *[replace(s, source_index=7) for s in old.segments[1:3]], *old.segments[3:]))
    result = own(context, domain, candidate)
    removed = result.to_dict()['original_visible_events_removed']
    assert any(r['first_incoming_output_frame'] == 46 and r['kind'] == 'native_camera_cut' for r in removed)
    assert domain.scene_cut_times[1] == (1.51,)
    assert result.to_dict()['original_source_native_inventory_changed'] is False


def test_keep_split_is_not_counted_as_changed_pixels_or_a_new_cut():
    context, domain = fixture()
    old = context.incumbent.render_plan
    candidate = replace(old, segments=(RenderSegment(0, 0., .5, 0.),
        RenderSegment(0, .5, .5, .5), *old.segments[1:]), transitions=())
    data = own(context, domain, candidate).to_dict()
    assert data['changed_output_frame_count'] == 0
    assert data['candidate_visible_events_added'] == []


@pytest.mark.parametrize('mutation', ['outside', 'protected_scope', 'header', 'budget'])
def test_unowned_protected_header_and_budget_changes_cannot_be_hidden_by_new_grid(mutation):
    context, domain = fixture()
    old = context.incumbent.render_plan
    candidate = replace(old, segments=tuple(replace(s, source_index=7) if i == 1 else s
        for i, s in enumerate(old.segments)), transitions=())
    if mutation == 'outside':
        candidate = replace(candidate, segments=(replace(candidate.segments[0], source_index=7),
            *candidate.segments[1:]))
    elif mutation == 'header': candidate = replace(candidate, master_audio_start=.1)
    with pytest.raises(ValueError):
        own(context, domain, candidate,
            mutable=((30, 180),) if mutation == 'protected_scope' else ((30, 90),),
            cap=20 if mutation == 'budget' else 60)
