"""Actual numeric renderer clocks; no real media or quality approval."""
from dataclasses import replace
from fractions import Fraction

import pytest

from crosscut.group_native_motion_programme import NativeMotionSide,realize_native_motion_programme
from crosscut.group_scene_owned_program import freeze_owned_scene_program
from crosscut.group_scene_epoch_program import compile_common_clock_camera_queries
from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe,segment_frame_counts,RenderSegment,RenderTransition
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.shot_video_clock import validate_authorized_shot_video_clocks
from tests.test_scene_role_epoch_program import fixture


@pytest.mark.parametrize('boundary',[57,59,60,61,63])
@pytest.mark.parametrize('native_fps',[Fraction(30),Fraction(60)])
def test_moved_topology_frozen_before_two_sided_native_phase(boundary,native_fps):
    context,domain,_=fixture(native_fps=native_fps)
    original=context.incumbent.render_plan
    sides=(NativeMotionSide(30,boundary,2,1),NativeMotionSide(boundary,90,3,-1))
    result=realize_native_motion_programme(original=original,domain=domain,sides=sides)
    assert result.shot_clock_context.incumbent_plan is result.aligned_plan
    assert result.render_plan.output_frame_count==original.output_frame_count
    assert len(result.render_plan.authorized_shot_video_clocks)==2
    validate_authorized_shot_video_clocks(plan=result.render_plan,context=result.shot_clock_context)
    old=native_frame_clocks_for_output(result.aligned_plan,domain.clocks_by_request,range(30,90))
    new=native_frame_clocks_for_output(result.render_plan,domain.clocks_by_request,range(30,90))
    assert new[30].native_frame_index-old[30].native_frame_index==1
    assert new[boundary].native_frame_index-old[boundary].native_frame_index==-1
    assert tuple(segment_frame_counts(result.render_plan))==(30,boundary-30,90-boundary,30)
    assert result.preservation['actual_native_rasters_preserved']
    assert result.preservation['camera_refit_after_selection'] is False


def test_selected_camera_reaches_every_actual_query_without_a_new_time_origin():
    context,domain,_=fixture(native_fps=Fraction(60))
    camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.12,.5,.5),
        EditorialTransformKeyframe(1.,1.18,.5,.5)),sampling_canvas_zoom=1.2)
    sides=(NativeMotionSide(30,61,2,1,camera),NativeMotionSide(61,90,3,-1,camera))
    original=freeze_owned_scene_program(context,domain)
    result=realize_native_motion_programme(original=context.incumbent.render_plan,domain=domain,sides=sides,
        protected_output_intervals=((0,20),(100,120)))
    queries=compile_common_clock_camera_queries(context=context,domain=domain,original_program=original,
        candidate_plan=result.render_plan,output_intervals=((30,90),))
    for q in queries.queries:
        for tick in range(q.first_output_frame,q.end_output_frame):
            segment,time,native=q.sample_at(tick)
            expected=camera.evaluate((tick-(30 if tick<61 else 61))/result.render_plan.fps)
            actual=segment.editorial_transform.evaluate(time)
            assert (actual.zoom,actual.center_x,actual.center_y)==pytest.approx(
                (expected.zoom,expected.center_x,expected.center_y))
    import json
    value=json.loads(queries.original_program_json)
    assert value==original.to_dict()
    assert value['original_output_role_epochs_frozen']


def test_protected_pixels_and_actual_native_camera_cuts_are_not_waived():
    context,domain,_=fixture(native_fps=Fraction(60))
    original=context.incumbent.render_plan
    sides=(NativeMotionSide(30,60,2),NativeMotionSide(60,90,3))
    with pytest.raises(ValueError,match='protected'):
        realize_native_motion_programme(original=original,domain=domain,sides=sides,
            protected_output_intervals=((50,70),))
    native=native_frame_clocks_for_output(original,domain.clocks_by_request,range(30,60))
    cuts=list(domain.scene_cut_times);cuts[domain.eligible_index(2)]=(native[45].native_frame_index/60,)
    with pytest.raises(ValueError,match='native.*cut'):
        realize_native_motion_programme(original=original,domain=replace(domain,scene_cut_times=tuple(cuts)),sides=sides)


def test_excluded_request_source_is_not_an_eligible_local_index():
    context,domain,_=fixture()
    with pytest.raises(ValueError,match='excluded'):
        realize_native_motion_programme(original=context.incumbent.render_plan,domain=domain,
            sides=(NativeMotionSide(30,60,1),NativeMotionSide(60,90,2)))


def test_false_policy_is_not_implicit_permission_to_use_default_bounds():
    context,domain,_=fixture()
    with pytest.raises(ValueError,match='policy'):
        realize_native_motion_programme(original=context.incumbent.render_plan,domain=domain,
            sides=(NativeMotionSide(30,60,2),NativeMotionSide(60,90,3)),phase_policy=False)


def test_fractional_original_tier_d_units_are_not_independently_reset_per_cell():
    context,domain,_=fixture(native_fps=Fraction(60))
    original=replace(context.incumbent.render_plan,segments=(
        RenderSegment(0,0.,.97,0.),RenderSegment(0,.97,1.03,.97),
        RenderSegment(0,2.,.98,2.),RenderSegment(0,2.98,1.02,2.98)),
        transitions=(RenderTransition(tier='D',reason='same_native_run'),)*3)
    result=realize_native_motion_programme(original=original,domain=domain,
        sides=(NativeMotionSide(35,60,2,1),NativeMotionSide(60,85,3,-1)))
    before=native_frame_clocks_for_output(original,domain.clocks_by_request,range(120))
    after=native_frame_clocks_for_output(result.render_plan,domain.clocks_by_request,range(120))
    for tick in (*range(35),*range(85,120)):
        assert (after[tick].source_index,after[tick].native_frame_index)==(
            before[tick].source_index,before[tick].native_frame_index)
    assert result.render_plan.transitions[0].tier=='D'
    assert result.render_plan.transitions[-1].tier=='D'
