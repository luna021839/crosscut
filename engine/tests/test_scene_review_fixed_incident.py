"""Fixed far-away fragments are not new mutable evidence cells."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from crosscut.group_scene_review_proposal import _review_states
from crosscut.group_joint_source_camera import GroupOutputRole
from crosscut.group_joint_state_factory import candidate_joint_group_state, assemble_single_group_source_camera_state
from crosscut.group_source_changing_owner import _states
from crosscut.render import RenderSegment, RenderTransition, EditorialTransformTrack, EditorialTransformKeyframe
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from engine.tests import test_group_joint_state_factory as numeric_fixture


def fixture():
    plan, domain, _ = numeric_fixture.JointStateFactoryTests().fixture()
    camera = EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),
        EditorialTransformKeyframe(1.,1.2)), sampling_canvas_zoom=1.3)
    plan = replace(plan, segments=tuple(RenderSegment(s,float(i),1.,float(i),
        editorial_transform=camera if i==1 else None)
        for i,s in enumerate((0,0,2,3,3,5))), transitions=(RenderTransition(),)*5)
    domain = replace(domain, scene_cut_times=((1/30.,),(),(149/30.,),()))
    shots = tuple(NS(first_output_frame=a,end_output_frame=b) for a,b in ((1,60),(60,90),(90,149)))
    return NS(incumbent=NS(render_plan=plan),domain=domain), shots


def test_fixed_far_one_frame_fragment_does_not_block_original_incident_support():
    context, shots = fixture()
    with pytest.raises(ValueError, match='one-frame'):
        _states(context.incumbent.render_plan, context.domain, shots,
            (GroupOutputRole.COMPOSITION,)*3, fixed_subcell_views=True)
    path = _review_states(context, shots)
    assert [(s.first_output_frame,s.end_output_frame) for s in path] == [(30,60),(60,90),(90,120)]
    assert path[0].segments == context.incumbent.render_plan.segments[1:2]
    original = native_frame_clocks_for_output(context.incumbent.render_plan,
        context.domain.clocks_by_request, range(180))
    for state in path:
        assert state.native_frame_indices == tuple(original[t].native_frame_index
            for t in range(state.first_output_frame,state.end_output_frame))


def test_shorter_fixed_view_changes_no_outside_source_camera_or_actual_cuts():
    context, shots = fixture()
    path = _review_states(context, shots)
    mutable = candidate_joint_group_state(plan=context.incumbent.render_plan,domain=context.domain,
        incumbent_state=path[1],eligible_source_index=3)
    plan,changed = assemble_single_group_source_camera_state(plan=context.incumbent.render_plan,
        domain=context.domain,path=(path[0],mutable,path[-1]))
    assert changed == (2,)
    assert all(a==b for i,(a,b) in enumerate(zip(plan.segments,context.incumbent.render_plan.segments)) if i!=2)


def test_review_cannot_hide_an_incident_native_cut_or_mutate_an_unowned_cell():
    context, shots = fixture()
    domain = replace(context.domain,scene_cut_times=((59/30.,),(),(149/30.,),()))
    with pytest.raises(ValueError,match='native camera cut'):
        _review_states(NS(incumbent=context.incumbent,domain=domain),shots)
    unowned = (shots[0],NS(first_output_frame=61,end_output_frame=90),shots[-1])
    with pytest.raises(ValueError,match='owned splitting'):
        _review_states(context,unowned)


def test_review_can_replace_existing_clock_jump_but_not_change_its_tick():
    from crosscut.group_actual_shots import visible_group_cut_inventory
    context, shots = fixture()
    original = context.incumbent.render_plan
    # A real old same-source seek is an existing seam, not a native cut.
    segments = list(original.segments)
    segments[3] = replace(segments[3],source_index=2,source_start=4.)
    plan = replace(original,segments=tuple(segments))
    context = NS(incumbent=NS(render_plan=plan),domain=context.domain)
    path = _review_states(context,shots)
    middle = candidate_joint_group_state(plan=plan,domain=context.domain,
        incumbent_state=path[1],eligible_source_index=3)
    changed_path = (path[0],middle,path[-1])
    with pytest.raises(ValueError,match='visible cut cadence'):
        assemble_single_group_source_camera_state(plan=plan,domain=context.domain,path=changed_path)
    changed,_ = assemble_single_group_source_camera_state(plan=plan,domain=context.domain,
        path=changed_path,replace_incident_clock_jumps=True)
    old = visible_group_cut_inventory(plan,context.domain)
    new = visible_group_cut_inventory(changed,context.domain)
    assert [e.first_incoming_output_frame for e in new]==[e.first_incoming_output_frame for e in old]
    assert [(e.kind,e.first_incoming_output_frame) for e in old if e.first_incoming_output_frame==90]==[('same_source_clock_jump',90)]
    assert [(e.kind,e.first_incoming_output_frame) for e in new if e.first_incoming_output_frame==90]==[('source_switch',90)]
    with pytest.raises(ValueError,match='explicit'):
        assemble_single_group_source_camera_state(plan=plan,domain=context.domain,
            path=changed_path,replace_incident_clock_jumps=1)
