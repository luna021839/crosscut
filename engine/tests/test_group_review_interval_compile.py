from dataclasses import replace
import pytest
from engine.tests.test_scene_owned_program import fixture
from crosscut.group_review_interval_compile import (clip_review_segments,assemble_review_intervals,
    shifted_review_source_plan,realize_interval_in_plan,verify_realized_sampler)
from crosscut.group_review_edit_program import own_review_edit_program
from crosscut.group_scene_owned_program import freeze_owned_scene_program
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe


def test_shift_zero_is_original_and_nonzero_changes_actual_native_samples_not_tempo():
    context,domain=fixture();plan=context.incumbent.render_plan
    assert shifted_review_source_plan(plan,phase_frames=0) is plan
    candidate=shifted_review_source_plan(plan,phase_frames=2)
    a,b=[native_frame_clocks_for_output(p,domain.clocks_by_request,(40,41)) for p in (plan,candidate)]
    assert b[40].source_request_seconds-a[40].source_request_seconds==pytest.approx(2/plan.fps)
    assert all(s.source_playback_rate==t.source_playback_rate for s,t in zip(plan.segments,candidate.segments))
    with pytest.raises(ValueError,match='bounded'):shifted_review_source_plan(plan,phase_frames=3)


@pytest.mark.parametrize('bounds',[(28,62),(32,58),(2,178)])
def test_keep_split_keeps_every_source_native_address(bounds):
    context,domain=fixture();plan=context.incumbent.render_plan
    candidate=realize_interval_in_plan(plan,*bounds)
    result=verify_realized_sampler(plan,candidate,domain,range(plan.output_frame_count))
    assert result['selected_sampler_preserved']
    owner=own_review_edit_program(context=context,domain=domain,
        original_program=freeze_owned_scene_program(context,domain),candidate_plan=candidate,
        mutable_output_intervals=((0,150),),protected_output_intervals=((150,180),),
        maximum_changed_frames=150,maximum_visible_cuts=20)
    assert not owner.changed_output_intervals


def test_realized_camera_serialization_preserves_sampler_but_not_original_geometry_claim():
    context,domain=fixture();plan=context.incumbent.render_plan
    camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.2),
        EditorialTransformKeyframe(1.,1.3)),reason='realized_before_search')
    candidate=realize_interval_in_plan(plan,30,60,camera=camera)
    assert verify_realized_sampler(plan,candidate,domain,range(30,60))['selected_sampler_preserved']
    assert any(s.editorial_transform is not None for s in candidate.segments)


def test_overlap_gap_or_quantized_boundary_drift_cannot_be_serialized():
    context,domain=fixture();plan=context.incumbent.render_plan
    with pytest.raises(ValueError,match='omit or overlap'):
        assemble_review_intervals(plan,[(0,30,clip_review_segments(plan,0,30)),
            (31,180,clip_review_segments(plan,31,180))])


def test_selected_phase_cannot_claim_original_sampler_identity():
    context,domain=fixture();plan=context.incumbent.render_plan
    candidate=shifted_review_source_plan(plan,phase_frames=1)
    with pytest.raises(ValueError,match='selected source/native'):
        verify_realized_sampler(plan,candidate,domain,range(plan.output_frame_count))


def test_boundary_lattice_must_not_create_a_one_frame_render_piece():
    context,domain=fixture()
    with pytest.raises(ValueError,match='one-frame'):
        realize_interval_in_plan(context.incumbent.render_plan,29,61)


def test_late_local_negative_phase_does_not_require_shifting_unused_source_zero_preroll():
    context,domain=fixture();plan=context.incumbent.render_plan
    candidate=shifted_review_source_plan(plan,phase_frames=-2,output_interval=(30,60))
    a,b=[native_frame_clocks_for_output(p,domain.clocks_by_request,(0,40,90)) for p in (plan,candidate)]
    assert b[0]==a[0] and b[90]==a[90]
    assert b[40].source_request_seconds==pytest.approx(a[40].source_request_seconds-2/plan.fps)


def test_exact_rasters_do_not_claim_identical_unquantized_request_times():
    context,domain=fixture();plan=context.incumbent.render_plan
    candidate=replace(plan,segments=tuple(replace(s,source_start=s.source_start+.00001)
        for s in plan.segments))
    with pytest.raises(ValueError,match='selected source/native'):
        verify_realized_sampler(plan,candidate,domain,range(plan.output_frame_count))
    report=verify_realized_sampler(plan,candidate,domain,range(plan.output_frame_count),
        require_source_request_time=False)
    assert report['actual_native_rasters_preserved']
    assert not report['unquantized_source_request_times_preserved']
    assert report['source_request_time_difference_count']==plan.output_frame_count
    assert report['maximum_source_request_time_difference_seconds']==pytest.approx(.00001)
    assert report['quality_approved'] is False


def test_raster_only_contract_still_rejects_a_changed_native_frame():
    context,domain=fixture();plan=context.incumbent.render_plan
    candidate=shifted_review_source_plan(plan,phase_frames=1)
    with pytest.raises(ValueError,match='selected source/native'):
        verify_realized_sampler(plan,candidate,domain,range(plan.output_frame_count),
            require_source_request_time=False)
