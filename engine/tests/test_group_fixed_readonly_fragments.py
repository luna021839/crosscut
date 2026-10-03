"""An unchanged observation view must not create a one-frame render cell."""
from dataclasses import replace
from fractions import Fraction

import pytest

from crosscut.group_joint_state_factory import fixed_neighbor_state_from_render_plan
from crosscut.group_joint_source_camera import GroupOutputRole
from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe,segment_frame_counts
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from engine.tests import test_group_joint_state_factory as factory_fixture


@pytest.mark.parametrize('fps,native',[(30.,Fraction(30)),(29.97,Fraction(60000,1001))])
@pytest.mark.parametrize('first,end',[(0,29),(1,30),(1,29)])
def test_fixed_subcell_view_keeps_global_native_phase_without_a_new_render_cell(fps,native,first,end):
    plan,domain,_=factory_fixture.JointStateFactoryTests().fixture(fps,native)
    camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1,.49,.5),
        EditorialTransformKeyframe(1.,1.3,.52,.5,.6)),sampling_canvas_zoom=1.4)
    plan=replace(plan,segments=(replace(plan.segments[0],editorial_transform=camera),*plan.segments[1:]))
    before=plan.to_json();counts=segment_frame_counts(plan)
    state=fixed_neighbor_state_from_render_plan(plan=plan,domain=domain,slot=0,
        first_output_frame=first,end_output_frame=end,role=GroupOutputRole.FOCAL_HEAD)
    actual=native_frame_clocks_for_output(plan,domain.clocks_by_request,range(first,end))
    assert state.native_frame_indices==tuple(sample.native_frame_index for sample in actual.values())
    assert state.output_size==(plan.width,plan.height)
    assert sum(state.segment_output_frame_counts)==end-first
    assert plan.to_json()==before and segment_frame_counts(plan)==counts
    for relative in range(end-first):
        seen=state.segments[0].editorial_transform.evaluate(relative/fps)
        expected=camera.evaluate((first+relative)/fps)
        for name in ('zoom','center_x','center_y','roll_degrees'):
            assert getattr(seen,name)==pytest.approx(getattr(expected,name),abs=1e-9)
        assert state.segments[0].editorial_transform.effective_sampling_canvas_zoom==1.4


def test_partial_cell_view_is_not_permission_to_make_a_mutable_replacement():
    from crosscut.group_joint_state_factory import candidate_joint_group_state
    plan,domain,_=factory_fixture.JointStateFactoryTests().fixture()
    state=fixed_neighbor_state_from_render_plan(plan=plan,domain=domain,slot=0,
        first_output_frame=0,end_output_frame=29,role=GroupOutputRole.FOCAL_HEAD)
    with pytest.raises(ValueError):candidate_joint_group_state(plan=plan,domain=domain,
        incumbent_state=state,eligible_source_index=1)
    with pytest.raises(ValueError):fixed_neighbor_state_from_render_plan(plan=plan,domain=domain,slot=1,
        first_output_frame=0,end_output_frame=29,role=GroupOutputRole.FOCAL_HEAD)


@pytest.mark.parametrize('first,end,counts',[(29,60,(1,30)),(0,31,(30,1))])
def test_one_tick_internal_observation_fragment_is_not_a_one_frame_render_cell(first,end,counts):
    from crosscut.group_joint_source_camera import FixedGroupNeighborObservationState
    from crosscut.group_joint_state_factory import candidate_joint_group_state
    plan,domain,_=factory_fixture.JointStateFactoryTests().fixture(29.97,Fraction(60000,1001))
    plan=replace(plan,segments=(plan.segments[0],replace(plan.segments[1],source_index=0),*plan.segments[2:]))
    state=fixed_neighbor_state_from_render_plan(plan=plan,domain=domain,slot=2,
        first_output_frame=first,end_output_frame=end,role=GroupOutputRole.FOCAL_HEAD)
    assert isinstance(state,FixedGroupNeighborObservationState)
    assert state.segment_output_frame_counts==counts
    assert state.to_dict()['render_topology_materialized'] is False
    assert segment_frame_counts(plan)==(30,30,30,30)
    actual=native_frame_clocks_for_output(plan,domain.clocks_by_request,range(first,end))
    assert state.native_frame_indices==tuple(row.native_frame_index for row in actual.values())
    with pytest.raises(ValueError):candidate_joint_group_state(plan=plan,domain=domain,
        incumbent_state=state,eligible_source_index=2)


def test_fixed_fragment_cannot_bridge_an_actual_native_cut():
    plan,domain,_=factory_fixture.JointStateFactoryTests().fixture()
    domain=replace(domain,scene_cut_times=((.5,),(),(),()))
    with pytest.raises(ValueError,match='native camera cut'):
        fixed_neighbor_state_from_render_plan(plan=plan,domain=domain,slot=0,
            first_output_frame=1,end_output_frame=29,role=GroupOutputRole.FOCAL_HEAD)


def test_fixed_fragment_registered_pose_consumer_restores_the_exact_new_state_type():
    from pathlib import Path
    from types import SimpleNamespace as NS
    from crosscut.group_scene_intent import declared_shot_intent
    from crosscut.group_scene_focal_pose_motion import restore_view_state
    from crosscut.group_joint_source_camera import FixedGroupNeighborObservationState
    from crosscut.render import RenderPlan,RenderSegment
    from engine.tests.test_scene_focal_pose_motion import scene_world
    sensors,_,_,_,_=scene_world()
    domain=sensors.session.domain
    plan=RenderPlan(sources=domain.request_sources,segments=(RenderSegment(0,0.,8/30,0.),),
        reference_source=0,aspect_ratio='16:9',output=Path('NONHUMAN_NOT_RENDERED.mp4'),fps=30.)
    state=fixed_neighbor_state_from_render_plan(plan=plan,domain=domain,slot=0,
        first_output_frame=0,end_output_frame=7,role=GroupOutputRole.FOCAL_HEAD)
    context=NS(incumbent=NS(render_plan=plan,visual_timeline=NS(choices=[NS(shot_type='medium')]),
        face_match_plan=NS(segments=[NS(alignment_subject=None,applied=False)])))
    shot=NS(first_output_frame=0,end_output_frame=7,first_segment_index=0,last_segment_index=0)
    sensors.shot_intents[0,7,'exit']=declared_shot_intent(context,shot,endpoint_scope='exit')
    hypothesis=sensors.discover(state,rolls=(None,),endpoint_scope='exit')[0]
    state=sensors.bind(state,hypothesis)
    restored=restore_view_state(state.to_dict(),domain,hypothesis,state.key)
    assert isinstance(restored,FixedGroupNeighborObservationState)
    assert restored==state and restored.key==state.key
    endpoint=sensors.endpoint(state,outgoing=True)
    assert endpoint['registered_pose_sequence']['complete_state_preimage']==state.to_dict()
