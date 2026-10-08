"""Synthetic root/CAS/final-clock contracts; no model, media or quality proof."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json

import pytest

from crosscut.editorial_motion import EditorialMotionPlan,EditorialMotionSegment
from crosscut.first_result_controller import FirstResultController
from crosscut.group_external_original_reference import (GroupExternalOriginalReference,
    validate_external_original_reference_payload)
from crosscut.group_joint_bundle_metadata import (SelectedNativeActorFrame,SelectedNativeMotionShot,
    seal_native_motion_production)
from crosscut.group_joint_native_preparation import _plan_identity
from crosscut.group_joint_source_camera import GroupOutputRole
from crosscut.group_native_motion_programme import NativeMotionSide,realize_native_motion_programme
from crosscut.group_reference_interval import compare_group_reference_interval
from crosscut.group_review_interval_compile import output_offsets
from crosscut.group_source_domain import GroupSourceDomain
from crosscut.group_transition import GroupTransitionPlan
from crosscut.pipeline import _deterministic_exact_shot_clock_from_timeline
from crosscut.render import (NormalizedCrop,EditorialTransformTrack,EditorialTransformKeyframe,
    SourceLookAdjustment)
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.review_first_runtime import ReviewFirstRuntime
from crosscut.shot_video_clock import frozen_shot_video_clock_context_from_dict
from crosscut.timeline import ShotType
from crosscut.timeline_review import review_timeline
from crosscut.visual_score import VisualTimelineChoice,VisualTimelinePlan
from tests.test_group_native_motion_bundle import synthetic_bundle


def fixture(tmp_path,*,boundary=60,native_fps=30,phases=(1,-1)):
    baseline,old_domain=synthetic_bundle()
    sources=tuple(tmp_path/f'synthetic-not-media-{i}.mp4' for i in range(6))
    for path in sources:path.write_bytes(b'SYNTHETIC NUMERIC FILE IDENTITY ONLY - NOT A VIDEO')
    clock=VerifiedRenderFrameClock(Fraction(native_fps),10000,'a'*64)
    domain=GroupSourceDomain(sources,old_domain.eligible_indices,old_domain.audio_offsets,
        old_domain.source_timewarps,old_domain.scene_cut_times,(clock,)*4)
    plan=replace(baseline.render_plan,sources=sources,output=tmp_path/'never-rendered-root.mp4',fps=30.,
        segments=tuple(replace(s,pan_crop=None) for s in baseline.render_plan.segments))
    baseline=replace(baseline,render_plan=plan,face_match_plan=None,boundary_repairs=None,
        boundary_look_match=None,dynamic_crop_application=None,
        diagnostics=('SYNTHETIC_NUMERIC_NO_MEDIA_OR_MODEL',))
    protection=((0,30),(90,120))
    sides=(NativeMotionSide(30,boundary,2,phases[0]),NativeMotionSide(boundary,90,3,phases[1]))
    realised=realize_native_motion_programme(original=plan,domain=domain,sides=sides,
        protected_output_intervals=protection)
    selected=realised.render_plan
    # Fresh synthetic numeric rows, explicitly not source image descriptors.
    timeline=VisualTimelinePlan(tuple(VisualTimelineChoice(segment_index=i,
        source_index=domain.eligible_index(s.source_index),canonical_master_start=s.master_start,
        canonical_master_end=s.master_start+s.duration,source_local_start=s.source_start,
        source_local_end=s.source_end,quality_score=0.,composition_score=None,boundary_score=None,
        shot_scale=None,shot_type=ShotType.MEDIUM,shot_id=f'SYNTHETIC_UNOBSERVED-{i}')
        for i,s in enumerate(selected.segments)),0.,analysis_basis='SYNTHETIC_NUMERIC_ONLY_NO_PIXEL_OBSERVATIONS')
    review=review_timeline(timeline,available_source_count=4)
    motion=EditorialMotionPlan(tuple(EditorialMotionSegment(i,c.source_index,s.editorial_transform,
        ('SYNTHETIC_NUMERIC_ONLY_NO_ACTION_OBSERVATIONS',))
        for i,(c,s) in enumerate(zip(timeline.choices,selected.segments,strict=True))),
        0,0,0,0,0,analysis_basis='SYNTHETIC_NUMERIC_TRANSPORT',eligible_request_indices=domain.eligible_indices)
    complete=replace(baseline,render_plan=selected,visual_timeline=timeline,timeline_review=review,
        pre_render_score=review.weighted_score,editorial_motion_plan=motion,
        group_transition_plan=GroupTransitionPlan(selected.transitions,analysis_basis='SYNTHETIC_NUMERIC_TRANSPORT'),
        stage_edit_exact_shot_clock=_deterministic_exact_shot_clock_from_timeline(timeline,
            section_plan=baseline.section_plan,phrase_times=baseline.phrase_times))
    offsets=output_offsets(selected)
    schedule=native_frame_clocks_for_output(selected,domain.clocks_by_request,range(30,90))
    shots=tuple(SelectedNativeMotionShot(side.first_output_frame,side.end_output_frame,side.request_source_index,
        (0,10000),GroupOutputRole.FOCAL_HEAD,side.phase_native_frames,None,
        tuple(SelectedNativeActorFrame(tick,side.request_source_index,schedule[tick].native_frame_index,None,None)
            for tick in range(side.first_output_frame,side.end_output_frame)),
        tuple(range(offsets.index(side.first_output_frame),offsets.index(side.end_output_frame)))) for side in sides)
    production=seal_native_motion_production(complete_bundle=complete,source_domain=domain,
        shot_clock_context=realised.shot_clock_context,selected_shots=shots,
        original_plan_identity=_plan_identity(plan),protected_output_intervals=protection,
        state='SELECTED_FOR_REVIEW',selection_evidence={'contract':'SYNTHETIC_NUMERIC_CAS_ONLY',
            'actor_paths_observed':False,'quality_approved':False})
    attached=replace(complete,native_motion_production=production)
    reference=GroupExternalOriginalReference(baseline,domain,protection)
    return reference,attached,realised


def validate_serialized(reference,candidate,context_payload=None):
    policy=reference.to_dict()
    if context_payload is None and candidate.native_motion_production is not None:
        context_payload=candidate.native_motion_production.shot_clock_context.to_dict()
    validate_external_original_reference_payload(policy,plan=candidate.render_plan,
        look=candidate.boundary_look_match,source_domain_payload=policy['original_source_domain'],
        source_identities=policy['source_identities'],live_clocks=reference.source_domain.clocks_by_request,
        shot_clock_context_payload=None if context_payload is None else json.loads(json.dumps(context_payload)))


@pytest.mark.parametrize('boundary',(57,60,61))
@pytest.mark.parametrize('native_fps',(30,60))
def test_authorized_moved_cut_reaches_live_cas_and_serialized_final_policy(tmp_path,boundary,native_fps):
    reference,candidate,result=fixture(tmp_path,boundary=boundary,native_fps=native_fps)
    owner=FirstResultController(ReviewFirstRuntime(enabled=True),sources=reference.source_domain.request_sources,
        source_durations=(10.,)*6,master_start=0.,duration=4.,external_original_reference=reference,
        journal_path=tmp_path/'owner-numeric-evidence.json')
    baseline=reference.original_bundle
    owner.install_baseline(baseline,plan=baseline.render_plan,authority='SYNTHETIC NUMERIC ROOT ONLY')
    token=owner.transaction_token()
    owner.promote(candidate,plan=candidate.render_plan,authority='SYNTHETIC CLOCK/CAS ONLY',_token=token)
    assert owner.incumbent is candidate
    validate_serialized(reference,candidate)
    parsed=frozen_shot_video_clock_context_from_dict(json.loads(json.dumps(result.shot_clock_context.to_dict())),
        source_domain=reference.source_domain)
    assert parsed==result.shot_clock_context
    assert len(candidate.render_plan.authorized_shot_video_clocks)==2
    assert not candidate.native_motion_production.to_dict()['quality_approved']


def test_zero_phase_can_retain_root_context_without_inventing_a_local_clock(tmp_path):
    reference,candidate,result=fixture(tmp_path,phases=(0,0))
    assert not candidate.render_plan.authorized_shot_video_clocks
    reference.validate_candidate(candidate,candidate.render_plan)
    validate_serialized(reference,candidate)


def test_naked_or_unbound_local_clock_cannot_cross_external_reference_gate(tmp_path):
    reference,candidate,result=fixture(tmp_path)
    with pytest.raises(ValueError,match='phase-zero context'):
        reference.validate_numeric_plan(candidate.render_plan)
    with pytest.raises(ValueError,match='typed complete selection'):
        reference.validate_candidate(replace(candidate,native_motion_production=None),candidate.render_plan)
    policy=reference.to_dict()
    with pytest.raises(ValueError,match='phase-zero context'):
        validate_external_original_reference_payload(policy,plan=candidate.render_plan,look=None,
            source_domain_payload=policy['original_source_domain'],source_identities=policy['source_identities'])


@pytest.mark.parametrize('damage',('native_time','crop','roll','look'))
def test_every_protected_native_crop_roll_and_look_change_is_rejected(tmp_path,damage):
    reference,candidate,result=fixture(tmp_path)
    segment=candidate.render_plan.segments[0]
    changes={'native_time':dict(source_start=segment.source_start+1/30),
        'crop':dict(crop=NormalizedCrop(.05,.05,.9,.9)),
        'roll':dict(editorial_transform=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1,.5,.5,.2),))),
        'look':dict(look_adjustment=SourceLookAdjustment(brightness=.02))}[damage]
    changed=replace(candidate.render_plan,segments=(replace(segment,**changes),*candidate.render_plan.segments[1:]))
    with pytest.raises(ValueError):
        reference.validate_numeric_plan(changed,shot_clock_context=result.shot_clock_context)
    with pytest.raises(ValueError):validate_serialized(reference,replace(candidate,render_plan=changed))


@pytest.mark.parametrize('damage',('audio_reference','master_start','output_fps','source_order'))
def test_global_audio_source_and_output_grid_changes_are_not_local_phase_permission(tmp_path,damage):
    reference,candidate,result=fixture(tmp_path)
    changes={'audio_reference':dict(reference_source=2),'master_start':dict(master_audio_start=.01),
        'output_fps':dict(fps=29.97),'source_order':dict(sources=tuple(reversed(candidate.render_plan.sources)))}[damage]
    with pytest.raises(ValueError):
        changed=replace(candidate.render_plan,**changes)
        reference.validate_numeric_plan(changed,shot_clock_context=result.shot_clock_context)


def test_phase_zero_context_cannot_borrow_another_root_or_widen_protection(tmp_path):
    reference,candidate,result=fixture(tmp_path)
    for context in (replace(result.shot_clock_context,original_reference_plan=
            replace(reference.original_bundle.render_plan,reference_source=2)),
            replace(result.shot_clock_context,protected_output_intervals=((0,29),(90,120)))):
        with pytest.raises(ValueError,match='original root'):
            reference.validate_numeric_plan(candidate.render_plan,shot_clock_context=context)
    with pytest.raises(ValueError,match='protected'):
        replace(result.shot_clock_context,protected_output_intervals=((30,40),))


def test_stale_phase_zero_clock_preimage_is_rejected_in_live_and_serialized_replay(tmp_path):
    reference,candidate,result=fixture(tmp_path)
    context=replace(result.shot_clock_context,incumbent_plan=replace(result.aligned_plan,
        output=tmp_path/'changed-phase-zero-root.mp4'))
    with pytest.raises(ValueError):
        reference.validate_numeric_plan(candidate.render_plan,shot_clock_context=context)
    with pytest.raises(ValueError):validate_serialized(reference,candidate,context.to_dict())


def test_removing_authorized_clocks_does_not_launder_retained_nonzero_source_times(tmp_path):
    reference,candidate,result=fixture(tmp_path)
    naked=replace(candidate.render_plan,authorized_shot_video_clocks=())
    with pytest.raises(ValueError):reference.validate_candidate(replace(candidate,render_plan=naked),naked)
    with pytest.raises(ValueError):
        reference.validate_numeric_plan(naked,shot_clock_context=result.shot_clock_context)
    with pytest.raises(ValueError):validate_serialized(reference,replace(candidate,render_plan=naked))


def test_interval_comparison_consumes_actual_clocks_not_a_stripped_copy(tmp_path):
    reference,candidate,result=fixture(tmp_path)
    args=dict(reference=reference.original_bundle.render_plan,candidate=candidate.render_plan,
        reference_source_clocks=reference.source_domain.clocks_by_request,
        candidate_source_clocks=reference.source_domain.clocks_by_request,output_intervals=((0,30),(90,120)))
    with pytest.raises(ValueError,match='frozen source context'):compare_group_reference_interval(**args)
    compared=compare_group_reference_interval(**args,candidate_shot_clock_context=result.shot_clock_context)
    assert compared.planned_parameters_preserved
    assert len(candidate.render_plan.authorized_shot_video_clocks)==2
    assert not compared.to_dict()['encoded_pixel_identity_verified']


def test_source_replacement_cannot_hide_behind_a_self_consistent_clock_transport(tmp_path):
    reference,candidate,result=fixture(tmp_path)
    reference.source_domain.eligible_sources[0].write_bytes(b'REPLACED SYNTHETIC ROOT FILE')
    with pytest.raises(ValueError,match='changed after'):
        reference.validate_candidate(candidate,candidate.render_plan)


def test_serialized_root_is_canonical_but_not_an_independent_admission_badge(tmp_path):
    reference,candidate,result=fixture(tmp_path)
    value=deepcopy(result.shot_clock_context.to_dict());value['quality_approved']=True
    with pytest.raises(ValueError,match='source/root contract'):
        validate_serialized(reference,candidate,value)
