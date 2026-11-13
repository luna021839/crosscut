"""Complete numeric metadata transport; no detector/model or quality proof."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from crosscut.group_joint_native_preparation import _plan_identity
from crosscut.group_joint_bundle_metadata import SelectedNativeActorFrame, SelectedNativeMotionShot
from crosscut.group_joint_source_camera import GroupOutputRole
from crosscut.group_native_motion_programme import NativeMotionSide, realize_native_motion_programme
from crosscut.group_native_motion_camera_producer import NativeMotionCameraCandidate
from crosscut.group_native_motion_timeline import NativeMotionTimelineMetadata, metadata_cells, _fresh_choice
from crosscut.group_native_motion_metadata import complete_native_motion_metadata
from crosscut.pipeline import _validate_review_first_group_bundle
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.visual_score import VisualTimelinePlan
from tests.test_group_native_motion_bundle import synthetic_bundle
from tests.test_group_native_motion_timeline import sample


def variable_programme_complete():
    """Nine fresh cells, three runs, nonidentity mapping and UNKNOWN actors."""
    from crosscut.render import RenderSegment, RenderTransition
    from crosscut.group_review_interval_compile import output_offsets
    old, domain = synthetic_bundle()
    old = replace(old,dynamic_crop_application=None,boundary_look_match=None)
    counts = (13,)*8+(16,)
    offsets = (0,13,26,39,52,65,78,91,104,120)
    segments = tuple(RenderSegment((0,2,3)[i//3],offsets[i]/30,counts[i]/30,offsets[i]/30)
        for i in range(9))
    plan = replace(old.render_plan,segments=segments,transitions=(RenderTransition(),)*8)
    clocks = native_frame_clocks_for_output(plan,domain.clocks_by_request,range(120))
    shots = tuple(SelectedNativeMotionShot(a,b,s,(0,10000),GroupOutputRole.FOCAL_HEAD,0,None,
        tuple(SelectedNativeActorFrame(t,s,clocks[t].native_frame_index,None,None) for t in range(a,b)),
        tuple(range(first,last))) for a,b,s,first,last in ((0,39,0,0,3),(39,78,2,3,6),(78,120,3,6,9)))
    selected = NativeMotionCameraCandidate(plan,None,shots,
        json.dumps(dict(contract='NUMERIC_VARIABLE_PROGRAMME_TRANSPORT_ONLY',quality_approved=False)),
        selection_scope='variable_programme')
    cells = metadata_cells(original=old.render_plan,selected=plan,domain=domain,
        selected_shots=shots,selection_scope='variable_programme')
    samples = {t:sample(t) for c in cells for t in c.taps}
    choices = tuple(_fresh_choice(c,s,domain,plan,samples,'numeric-variable',())
        for c,s in zip(cells,segments,strict=True))
    metadata = NativeMotionTimelineMetadata(VisualTimelinePlan(choices,1.),cells,json.dumps(dict(
        original_plan_identity=_plan_identity(old.render_plan),selected_plan_identity=_plan_identity(plan),
        selected_shot_count=3,selection_scope='variable_programme',samples=list(samples.values()))))
    return SimpleNamespace(incumbent=old),selected,metadata,domain


def test_variable_programme_complete_transports_three_runs_and_nine_changed_cells():
    from crosscut.group_performer_family_owner import validate_serialized_metadata
    context,selected,metadata,domain = variable_programme_complete()
    assert len(metadata.cells)==9 and all(c.refresh_source_observations for c in metadata.cells)
    bundle = complete_native_motion_metadata(context=context,selected=selected,metadata=metadata,domain=domain)
    assert bundle.native_motion_production.selection_scope=='variable_programme'
    assert len(bundle.native_motion_production.selected_shots)==3
    assert all(row.frame_local_head_id is None for s in selected.selected_shots for row in s.actor_frames)
    assert all(c.source_index==domain.eligible_index(s.source_index)
        for c,s in zip(bundle.visual_timeline.choices,bundle.render_plan.segments,strict=True))
    assert validate_serialized_metadata(bundle.to_dict())['quality_approved'] is False
    assert dict(bundle.editorial_motion_plan.authored_event_counts)['separately_bound_selected_native_cameras']==3


def test_variable_programme_does_not_silently_change_legacy_two_shot_contract():
    context,selected,metadata,domain = variable_programme_complete()
    with pytest.raises(ValueError,match='both exact selected shots'):
        metadata_cells(original=context.incumbent.render_plan,selected=selected.render_plan,
            domain=domain,selected_shots=selected.selected_shots)
    with pytest.raises(ValueError):
        metadata_cells(original=context.incumbent.render_plan,selected=selected.render_plan,
            domain=domain,selected_shots=tuple(reversed(selected.selected_shots)),selection_scope='variable_programme')
    with pytest.raises(ValueError):
        # Omitting a run is not permission to change its exterior sampler.
        metadata_cells(original=context.incumbent.render_plan,selected=selected.render_plan,
            domain=domain,selected_shots=selected.selected_shots[1:],selection_scope='variable_programme')


def numeric_complete(*, first=30, cut=61, end=90, fractional=False, phases=(1, -1)):
    old, domain = synthetic_bundle()
    old = replace(old, dynamic_crop_application=None, boundary_look_match=None)
    if fractional:
        from crosscut.render import RenderSegment, RenderTransition
        from crosscut.editorial_motion import EditorialMotionSegment
        times = (0., .97, 2., 2.98, 4.)
        parts = tuple(RenderSegment(0, times[i], times[i+1]-times[i], times[i],
            pan_crop=old.render_plan.segments[i].pan_crop) for i in range(4))
        timeline = replace(old.visual_timeline, choices=tuple(replace(c, source_index=0,
            canonical_master_start=times[i], canonical_master_end=times[i+1],
            source_local_start=times[i], source_local_end=times[i+1]) for i,c in enumerate(old.visual_timeline.choices)))
        old = replace(old, visual_timeline=timeline,
            render_plan=replace(old.render_plan, segments=parts,
                transitions=(RenderTransition(tier='D', reason='same_original_native_unit'),)*3),
            editorial_motion_plan=replace(old.editorial_motion_plan, segments=tuple(
                EditorialMotionSegment(i, 0, None, ()) for i in range(4))),
            face_match_plan=replace(old.face_match_plan, segments=tuple(replace(c, source_index=0)
                for c in old.face_match_plan.segments)), boundary_repairs=None)
    result = realize_native_motion_programme(original=old.render_plan, domain=domain,
        sides=(NativeMotionSide(first, cut, 2, phases[0]), NativeMotionSide(cut, end, 3, phases[1])))
    from crosscut.group_review_interval_compile import output_offsets
    offsets = output_offsets(result.render_plan)
    rows = native_frame_clocks_for_output(result.render_plan, domain.clocks_by_request, range(first, end))
    shots = tuple(SelectedNativeMotionShot(a, b, s, (0, 10000), GroupOutputRole.FOCAL_HEAD,
        phase, None, tuple(SelectedNativeActorFrame(t, s, rows[t].native_frame_index, None, None)
            for t in range(a, b)), tuple(range(offsets.index(a), offsets.index(b))))
        for a, b, s, phase in ((first, cut, 2, phases[0]), (cut, end, 3, phases[1])))
    selected = NativeMotionCameraCandidate(result.render_plan, result.shot_clock_context, shots,
        json.dumps({'contract':'NUMERIC_CONSTRUCTION_TEST_ONLY', 'quality_approved':False}))
    cells = metadata_cells(original=old.render_plan, selected=selected.render_plan, domain=domain, selected_shots=shots)
    samples = {t:sample(t) for c in cells if c.refresh_source_observations for t in c.taps}
    choices = tuple(_fresh_choice(c, segment, domain, selected.render_plan, samples, 'numeric-new', ())
        if c.refresh_source_observations else replace(old.visual_timeline.choices[c.retained_old_segment_index],
            segment_index=c.segment_index, canonical_master_start=segment.master_start,
            canonical_master_end=segment.master_start+segment.duration,
            source_local_start=segment.source_start, source_local_end=segment.source_end)
        for c, segment in zip(cells, selected.render_plan.segments, strict=True))
    metadata = NativeMotionTimelineMetadata(VisualTimelinePlan(choices, 1.), cells,
        json.dumps(dict(original_plan_identity=_plan_identity(old.render_plan),
            selected_plan_identity=_plan_identity(selected.render_plan), samples=list(samples.values()))))
    context = SimpleNamespace(incumbent=old)
    return context, selected, metadata, domain


@pytest.mark.parametrize('cut', [57, 60, 61, 63])
def test_complete_metadata_owns_both_moved_shots_and_retained_face_family(cut):
    context, selected, metadata, domain = numeric_complete(cut=cut)
    bundle = complete_native_motion_metadata(context=context, selected=selected, metadata=metadata, domain=domain)
    assert bundle.render_plan is selected.render_plan
    assert len(bundle.visual_timeline.choices) == len(bundle.face_match_plan.segments) == len(bundle.render_plan.segments)
    assert bundle.face_match_plan.segments[0] == context.incumbent.face_match_plan.segments[0]
    assert bundle.face_match_plan.segments[-1] == context.incumbent.face_match_plan.segments[-1]
    assert bundle.face_match_plan.segments[1].fallback_reason == 'selected_editorial_camera_no_new_face_pan_recipe'
    assert bundle.face_match_plan.segments[1].observed_face_height is None
    assert bundle.native_motion_production.complete_bundle.native_motion_production is None
    assert bundle.native_motion_production.selected_shots == selected.selected_shots
    assert bundle.native_motion_production.to_dict()['quality_approved'] is False
    _validate_review_first_group_bundle(bundle, eligible_indices=domain.eligible_indices,
        offsets=domain.audio_offsets, timewarps=domain.source_timewarps, common_start=0., phrase_times=bundle.phrase_times)


def test_partial_old_cells_change_cardinality_without_old_face_source_or_clock_labels():
    context, selected, metadata, domain = numeric_complete(first=35, end=85)
    bundle = complete_native_motion_metadata(context=context, selected=selected, metadata=metadata, domain=domain)
    assert len(bundle.render_plan.segments) == 6
    assert len(bundle.editorial_motion_plan.segments) == len(bundle.face_match_plan.segments) == 6
    for i, (choice, segment, face, motion) in enumerate(zip(bundle.visual_timeline.choices,
            bundle.render_plan.segments, bundle.face_match_plan.segments, bundle.editorial_motion_plan.segments, strict=True)):
        assert choice.segment_index == face.segment_index == motion.segment_index == i
        assert choice.source_index == face.source_index == motion.source_index == domain.eligible_index(segment.source_index)
        assert face.pan_crop == segment.pan_crop and motion.track == segment.editorial_transform
        assert (choice.source_local_start, choice.source_local_end) == (segment.source_start, segment.source_end)


@pytest.mark.parametrize('name,value', [('dynamic_crop_application', {'active':True}),
    ('cut_local_sync', {'active':True}), ('splice_graph_authorization', {'active':True}),
    ('boundary_look_match', object())])
def test_unsupported_active_authority_is_declined_not_erased(name, value):
    context, selected, metadata, domain = numeric_complete()
    old = replace(context.incumbent, **{name:value})
    context = SimpleNamespace(incumbent=old)
    with pytest.raises(ValueError, match='retained owner support refresh'):
        complete_native_motion_metadata(context=context, selected=selected, metadata=metadata, domain=domain)
    assert getattr(context.incumbent, name) is not None


def test_foreign_predecessor_cannot_borrow_fresh_metadata_for_the_same_source_array():
    context, selected, metadata, domain = numeric_complete()
    context.incumbent = replace(context.incumbent,
        render_plan=replace(context.incumbent.render_plan, master_audio_start=.1))
    with pytest.raises(ValueError, match='different selected programme'):
        complete_native_motion_metadata(context=context, selected=selected, metadata=metadata, domain=domain)


def test_fractional_original_decoder_unit_reaches_full_bundle_without_global_clock_reset():
    context, selected, metadata, domain = numeric_complete(first=35, cut=60, end=85, fractional=True)
    bundle = complete_native_motion_metadata(context=context, selected=selected, metadata=metadata, domain=domain)
    assert len(bundle.render_plan.segments) == 6
    # This real numeric counterexample kept native rasters but previously
    # failed the complete consumer's independent-per-cell audio/warp clock.
    partial = bundle.render_plan.segments[-2]
    assert abs(partial.source_start-partial.master_start) > 1e-6
    _validate_review_first_group_bundle(bundle, eligible_indices=domain.eligible_indices,
        offsets=domain.audio_offsets, timewarps=domain.source_timewarps, common_start=0., phrase_times=bundle.phrase_times)


def test_removing_new_family_receipt_is_not_treated_as_an_ordinary_legacy_bundle():
    context, selected, metadata, domain = numeric_complete()
    bundle = complete_native_motion_metadata(context=context, selected=selected, metadata=metadata, domain=domain)
    with pytest.raises(ValueError, match='receipt was removed'):
        _validate_review_first_group_bundle(replace(bundle, performer_family_review=None),
            eligible_indices=domain.eligible_indices, offsets=domain.audio_offsets,
            timewarps=domain.source_timewarps, common_start=0., phrase_times=bundle.phrase_times)


def test_metadata_content_binding_cannot_promote_unknown_actor_to_quality_approval():
    context, selected, metadata, domain = numeric_complete()
    bundle = complete_native_motion_metadata(context=context, selected=selected, metadata=metadata, domain=domain)
    from crosscut.group_performer_family_owner import plain, validate_bundle_binding
    receipt = plain(bundle.performer_family_review)
    receipt['quality_approved'] = True
    with pytest.raises(ValueError, match='metadata/family binding changed'):
        validate_bundle_binding(replace(bundle, performer_family_review=receipt))


def test_native_metadata_receipt_does_not_allow_a_late_camera_mutation():
    context, selected, metadata, domain = numeric_complete()
    bundle = complete_native_motion_metadata(context=context, selected=selected, metadata=metadata, domain=domain)
    from crosscut.render import EditorialTransformTrack, EditorialTransformKeyframe
    from crosscut.group_performer_family_owner import validate_bundle_binding
    plan = bundle.render_plan
    changed = replace(plan, segments=(plan.segments[0], replace(plan.segments[1], editorial_transform=
        EditorialTransformTrack((EditorialTransformKeyframe(0., 1.2),))), *plan.segments[2:]))
    with pytest.raises(ValueError, match='metadata/family binding changed'):
        validate_bundle_binding(replace(bundle, render_plan=changed))


def test_phase_zero_preserved_clock_cannot_borrow_a_foreign_external_alignment():
    context, selected, metadata, domain = numeric_complete(phases=(0, 0))
    bundle = complete_native_motion_metadata(context=context, selected=selected, metadata=metadata, domain=domain)
    assert not bundle.render_plan.authorized_shot_video_clocks
    with pytest.raises(ValueError, match='typed external root context'):
        _validate_review_first_group_bundle(bundle, eligible_indices=domain.eligible_indices,
            offsets=(.1, *domain.audio_offsets[1:]), timewarps=domain.source_timewarps,
            common_start=0., phrase_times=bundle.phrase_times)
