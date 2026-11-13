"""Read-only fixed subcells do not authorize mutable topology changes."""
from dataclasses import replace

import pytest

from crosscut.group_joint_source_camera import GroupOutputRole as Role,joint_state_from_render_plan
from crosscut.group_joint_state_factory import (fixed_neighbor_state_from_render_plan,
    candidate_joint_group_state,assemble_single_group_source_camera_state)
from crosscut.group_actual_shots import visible_group_cut_inventory
from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from engine.tests import test_group_joint_state_factory as factory_fixture


def fixture():
    plan,domain,_=factory_fixture.JointStateFactoryTests().fixture()
    plan=replace(plan,segments=plan.segments[:3],transitions=plan.transitions[:2])
    domain=replace(domain,scene_cut_times=((.5,),(),(2.5,),()))
    def state(slot,a,b,fixed):
        builder=fixed_neighbor_state_from_render_plan if fixed else joint_state_from_render_plan
        return builder(plan=plan,domain=domain,slot=slot,first_output_frame=a,
            end_output_frame=b,role=Role.FOCAL_HEAD)
    return plan,domain,(state(0,15,30,True),state(1,30,60,False),state(2,60,75,True))


def test_fixed_subcell_states_keep_the_original_global_native_sampler():
    plan,domain,path=fixture()
    original=native_frame_clocks_for_output(plan,domain.clocks_by_request,range(90))
    for state in (path[0],path[-1]):
        assert state.native_frame_indices==tuple(original[t].native_frame_index
            for t in range(state.first_output_frame,state.end_output_frame))
    assert len(plan.segments)==3
    with pytest.raises(ValueError,match='owned splitting'):
        joint_state_from_render_plan(plan=plan,domain=domain,slot=0,
            first_output_frame=15,end_output_frame=30,role=Role.FOCAL_HEAD)


def test_aligned_middle_can_change_without_reindexing_any_full_plan_family():
    plan,domain,path=fixture()
    middle=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=path[1],
        eligible_source_index=3)
    candidate,changed=assemble_single_group_source_camera_state(plan=plan,domain=domain,
        path=(path[0],middle,path[2]))
    assert changed==(1,) and len(candidate.segments)==len(plan.segments)==3
    assert candidate.segments[0]==plan.segments[0] and candidate.segments[2]==plan.segments[2]
    assert [(r.first_incoming_output_frame,r.kind) for r in visible_group_cut_inventory(candidate,domain)]==[
        (r.first_incoming_output_frame,r.kind) for r in visible_group_cut_inventory(plan,domain)]
    before=native_frame_clocks_for_output(plan,domain.clocks_by_request,range(90))
    after=native_frame_clocks_for_output(candidate,domain.clocks_by_request,range(90))
    assert all((before[t].source_index,before[t].native_frame_index)==
        (after[t].source_index,after[t].native_frame_index) for t in (*range(30),*range(60,90)))


def test_neighbor_camera_is_sliced_without_restart_and_cannot_be_tampered():
    plan,domain,path=fixture()
    camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),
        EditorialTransformKeyframe(1.,1.3)),sampling_canvas_zoom=1.4)
    plan=replace(plan,segments=(replace(plan.segments[0],editorial_transform=camera),*plan.segments[1:]))
    left=fixed_neighbor_state_from_render_plan(plan=plan,domain=domain,slot=0,
        first_output_frame=15,end_output_frame=30,role=Role.FOCAL_HEAD)
    sliced=left.segments[0].editorial_transform
    assert sliced.effective_sampling_canvas_zoom==1.4
    for i in range(15):
        assert sliced.evaluate(i/30).zoom==pytest.approx(camera.evaluate((15+i)/30).zoom)
    tampered=replace(left,segments=(replace(left.segments[0],editorial_transform=None),))
    with pytest.raises(ValueError,match='fixed outer neighbor'):
        assemble_single_group_source_camera_state(plan=plan,domain=domain,
            path=(tampered,path[1],path[2]))


def test_fixed_view_cannot_be_borrowed_to_mutate_a_partial_cell():
    plan,domain,path=fixture()
    with pytest.raises(ValueError,match='owned cell splitting'):
        candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=path[2],
            eligible_source_index=3)
    with pytest.raises(ValueError,match='fixed outer'):
        fixed_neighbor_state_from_render_plan(plan=plan,domain=domain,slot=1,
            first_output_frame=15,end_output_frame=30,role=Role.FOCAL_HEAD)


def test_fixed_view_cannot_hide_a_native_cut_or_one_frame_fragment():
    plan,domain,_=fixture()
    with pytest.raises(ValueError,match='native camera cut'):
        fixed_neighbor_state_from_render_plan(plan=plan,domain=domain,slot=0,
            first_output_frame=0,end_output_frame=30,role=Role.FOCAL_HEAD)
    with pytest.raises(ValueError,match='one-frame'):
        fixed_neighbor_state_from_render_plan(plan=plan,domain=domain,slot=0,
            first_output_frame=29,end_output_frame=30,role=Role.FOCAL_HEAD)
