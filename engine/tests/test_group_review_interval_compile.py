from dataclasses import replace
import pytest
from engine.tests.test_scene_owned_program import fixture
from crosscut.group_review_interval_compile import (clip_review_segments,assemble_review_intervals,
    shifted_review_source_plan,realize_interval_in_plan,verify_realized_sampler,cover_review_sampling_interval,
    retain_whole_review_units)
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


def test_counterfactual_window_cover_does_not_create_one_frame_render_cells_or_move_real_edit():
    context,domain=fixture();plan=context.incumbent.render_plan
    assert cover_review_sampling_interval(plan,31,59)==(30,60)
    assert cover_review_sampling_interval(plan,30,60)==(30,60)
    first,end=cover_review_sampling_interval(plan,29,31)
    assert (first,end)==(0,60)
    pieces=clip_review_segments(plan,first,end)
    assert all(s.duration*plan.fps>=2 for s in pieces)
    with pytest.raises(ValueError,match='one-frame'):clip_review_segments(plan,0,31)


def test_sliced_fractional_duration_cell_uses_the_original_shared_render_clock():
    from crosscut.render import RenderSegment,RenderTransition
    context,domain=fixture(fps=29.97)
    plan=replace(context.incumbent.render_plan,segments=(
        RenderSegment(0,.009,1.005,0.),RenderSegment(0,1.014,1.005,1.005)),
        transitions=(RenderTransition(tier='D'),))
    camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.2),),reason='fixed_preselection')
    candidate=realize_interval_in_plan(plan,40,50,camera=camera)
    result=verify_realized_sampler(plan,candidate,domain,range(plan.output_frame_count),
        require_source_request_time=False)
    assert result['actual_native_rasters_preserved']
    # The authored second cell starts at 1.014s, but its source raster clock
    # begins after 30 quantized ticks of the shared first cell, not 1.005s.
    requested=native_frame_clocks_for_output(plan,domain.clocks_by_request,(40,))[40]
    selected=next(s for s in candidate.segments if abs(s.master_start-(1.005+10/29.97))<1e-8)
    assert selected.source_start==pytest.approx(requested.source_request_seconds)


def test_whole_cell_borrowed_after_source_change_keeps_its_old_shared_clock():
    from crosscut.render import RenderSegment,RenderTransition
    context,domain=fixture(fps=29.97)
    source=replace(context.incumbent.render_plan,segments=(
        RenderSegment(0,.029,1.005,0.),RenderSegment(0,1.034,1.005,1.005)),
        transitions=(RenderTransition(tier='D'),))
    counts=__import__('crosscut.render',fromlist=['segment_frame_counts']).segment_frame_counts(source)
    boundary=counts[0]
    prefix=tuple(replace(s,source_index=1) for s in clip_review_segments(source,0,boundary))
    borrowed=clip_review_segments(source,boundary,source.output_frame_count)
    candidate=assemble_review_intervals(source,[(0,boundary,prefix),
        (boundary,source.output_frame_count,borrowed)])
    result=verify_realized_sampler(source,candidate,domain,range(boundary,source.output_frame_count),
        require_source_request_time=False)
    assert result['actual_native_rasters_preserved']
    assert borrowed[0].source_start==pytest.approx(.029+boundary/source.fps)
    assert borrowed[0].editorial_transform is source.segments[1].editorial_transform


def test_materializing_all_whole_fractional_cells_keeps_every_physical_raster():
    from crosscut.render import RenderSegment,RenderTransition
    context,domain=fixture(fps=29.97)
    source=replace(context.incumbent.render_plan,segments=tuple(
        RenderSegment(0,.029+i*1.005,1.005,i*1.005) for i in range(4)),
        transitions=tuple(RenderTransition(tier='D') for _ in range(3)))
    whole=assemble_review_intervals(source,[(0,source.output_frame_count,
        clip_review_segments(source,0,source.output_frame_count))])
    assert verify_realized_sampler(source,whole,domain,range(source.output_frame_count),
        require_source_request_time=False)['actual_native_rasters_preserved']


def test_retaining_complete_fractional_decode_units_keeps_tier_d_and_exact_request_times():
    from crosscut.render import RenderSegment,RenderTransition,segment_frame_counts
    context,domain=fixture(fps=29.97)
    source=replace(context.incumbent.render_plan,segments=(
        RenderSegment(0,.029,1.005,0.),RenderSegment(0,1.034,1.005,1.005),
        RenderSegment(1,.029,1.005,2.010),RenderSegment(1,1.034,1.005,3.015)),
        transitions=(RenderTransition(tier='D'),RenderTransition(tier='C'),RenderTransition(tier='D')))
    boundary=sum(segment_frame_counts(source)[:2])
    prefix=retain_whole_review_units(source,0,boundary)
    suffix=retain_whole_review_units(source,boundary,source.output_frame_count)
    assert all(a is b for a,b in zip(prefix+suffix,source.segments,strict=True))
    rebuilt=assemble_review_intervals(source,[(0,boundary,prefix),(boundary,source.output_frame_count,suffix)])
    rebuilt=replace(rebuilt,transitions=source.transitions)
    result=verify_realized_sampler(source,rebuilt,domain,range(source.output_frame_count))
    assert result['unquantized_source_request_times_preserved']
    assert result['source_request_time_contract']=='exact'
    assert rebuilt.to_json()==source.to_json()


def test_retaining_authored_cell_cannot_detach_it_from_its_shared_decoder_clock():
    from crosscut.render import RenderSegment,RenderTransition,segment_frame_counts
    context,domain=fixture(fps=29.97)
    source=replace(context.incumbent.render_plan,segments=(
        RenderSegment(0,.029,1.005,0.),RenderSegment(0,1.034,1.005,1.005)),
        transitions=(RenderTransition(tier='D'),))
    boundary=segment_frame_counts(source)[0]
    with pytest.raises(ValueError,match='partial decoder unit'):
        retain_whole_review_units(source,boundary,source.output_frame_count)
    assert retain_whole_review_units(source,0,source.output_frame_count)==source.segments


@pytest.mark.parametrize('bounds',[(True,60),(-1,60),(0,'past_end'),(60,60)])
def test_whole_decoder_retention_rejects_invalid_output_intervals(bounds):
    context,domain=fixture()
    if bounds[1]=='past_end':bounds=(bounds[0],context.incumbent.render_plan.output_frame_count+1)
    with pytest.raises(ValueError,match='exact hard-cut'):
        retain_whole_review_units(context.incumbent.render_plan,*bounds)
