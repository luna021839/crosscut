"""Complete numeric GROUP metadata after an atomic whole-shot proposal.

This is a producer/consumer construction step, not the source-camera owner or
raw-v2 admission. The static owner must supply freshly produced source/camera
dependent motion and face metadata; the parent still owes raw semantic replay.
No child `complete` flag and no descriptor hash certifies media observations.
"""
from dataclasses import dataclass, replace

from .group_joint_native_preparation import _plan_identity
from .group_joint_render_observations import GroupJointRenderedObservations
from .group_joint_state_factory import (assemble_joint_group_states,
    assemble_single_group_source_camera_state)
from .group_source_domain import GroupSourceDomain
from .group_transition import GroupTransitionPlan
from .group_boundary_evidence_preservation import preserve_boundary_evidence
from .shot_context_binding import rebind_unchanged_shot_context
from .coupled_shot_binding import rebind_unchanged_coupled_shot
from .timeline_review import review_timeline
from .editorial_motion import EditorialMotionPlan


@dataclass(frozen=True, slots=True)
class GroupJointMetadataProposal:
    incumbent_plan_identity: str
    candidate_bundle: object
    changed_segment_indices: tuple[int, ...]

    def to_dict(self):
        return dict(contract='joint_group_complete_numeric_metadata_proposal_v1',
            incumbent_plan_identity=self.incumbent_plan_identity,
            candidate_bundle=self.candidate_bundle.to_dict(),
            changed_segment_indices=list(self.changed_segment_indices),
            complete_numeric_bundle_created=True,
            descriptor_pixels_independently_replayed=False,
            raw_v2_semantics_independently_replayed=False,
            candidate_admitted=False, quality_approved=False)


def complete_joint_group_bundle_metadata(*, incumbent, observations, domain,
        fresh_motion, fresh_face_match, phrase_times, shot_clock_context=None):
    """Build one full bundle; never install a new plan alongside old metadata.

    Only construction and numeric freshness are checked here. These checks do
    not prove that supplied fresh rows came from pixels: the raw-v2 parent must
    replay them before compare-and-promote. Unsupported incident legacy proof
    causes rollback, not a global reset. Both real 3/4-state paths are supported.
    """
    from .pipeline import (_GroupRenderCandidateBundle,
        _validate_review_first_group_bundle, _deterministic_exact_shot_clock_from_timeline)
    if (not isinstance(incumbent, _GroupRenderCandidateBundle)
            or not isinstance(observations, GroupJointRenderedObservations)
            or not isinstance(domain, GroupSourceDomain)):
        raise ValueError('joint metadata requires the complete incumbent, observations and source domain')
    phrases = tuple(phrase_times)
    _validate_review_first_group_bundle(incumbent, eligible_indices=domain.eligible_indices,
        offsets=domain.audio_offsets, timewarps=domain.source_timewarps,
        common_start=incumbent.render_plan.master_audio_start, phrase_times=phrases)
    before = incumbent.render_plan
    if observations.incumbent_plan_identity != _plan_identity(before):
        raise ValueError('joint descriptor proposal belongs to another whole incumbent')
    path = observations.path
    if not isinstance(path, tuple) or len(path) not in (3, 4):
        raise ValueError('joint metadata needs one complete actual-shot state path')
    assembler = (assemble_single_group_source_camera_state if len(path) == 3
        else assemble_joint_group_states)
    expected, changed = assembler(plan=before, domain=domain, path=path)
    if observations.plan.authorized_shot_video_clocks:
        from .shot_video_clock import FrozenShotVideoClockContext, attach_authorized_shot_video_clocks
        if (not isinstance(shot_clock_context, FrozenShotVideoClockContext) or shot_clock_context.incumbent_plan != before
                or shot_clock_context.source_domain != domain):
            raise ValueError('joint numeric metadata lacks its external frozen shot-clock context')
        expected = attach_authorized_shot_video_clocks(plan=expected,
            context=shot_clock_context, clocks=observations.plan.authorized_shot_video_clocks)
    if expected != observations.plan or changed != observations.changed_segment_indices:
        raise ValueError('joint descriptor plan/changed inventory differs from the assembled whole states')
    if not changed:
        if observations.timeline != incumbent.visual_timeline:
            raise ValueError('no-op joint proposal cannot replace the incumbent descriptors')
        return GroupJointMetadataProposal(_plan_identity(before), incumbent, ())
    # These owners bind learned cameras, native clocks or final look ramps.
    # This constructor has no replay for their incident dependencies yet.
    if any(value is not None for value in (incumbent.dynamic_crop_application,
            incumbent.cut_local_sync, incumbent.splice_graph_authorization,
            incumbent.boundary_look_match, incumbent.output_geometry_deferral,
            incumbent.stage_edit_dynamic_crop_version, incumbent.stage_edit_dynamic_crop_v3_decodes,
            incumbent.dynamic_crop_motion_timeline_v3)):
        raise ValueError('joint metadata needs refreshed learned/clock/look authority before changing its plan')
    refreshed = observations.refreshed_segment_indices
    required = set(range(max(0, min(changed)-1), min(len(before.segments), max(changed)+2)))
    if (not isinstance(refreshed, tuple) or len(set(refreshed)) != len(refreshed)
            or any(type(i) is not int or not 0 <= i < len(before.segments) for i in refreshed)
            or not required.issubset(refreshed)):
        raise ValueError('joint descriptor refresh omitted a changed cell or incident descriptor edge')
    timeline = observations.timeline
    if (len(timeline.choices) != len(incumbent.visual_timeline.choices)
            or timeline.candidate_review_summaries
            or 'UNREFRESHED' in timeline.analysis_basis):
        raise ValueError('joint descriptor proposal retained a stale scaffold or candidate review summary')
    for index, (old, new) in enumerate(zip(incumbent.visual_timeline.choices, timeline.choices, strict=True)):
        if index not in refreshed and old != new:
            raise ValueError('joint descriptor proposal changed an unrefreshed unrelated cell')
        segment = expected.segments[index]
        if (abs(new.source_local_start-segment.source_start) > 1e-6
                or abs(new.source_local_end-segment.source_end) > 1e-6):
            raise ValueError('joint descriptor source-local endpoints differ from the current renderer')
    old_motion = incumbent.editorial_motion_plan
    # Every untouched segment retains its original authored/dense history.
    # Changed rows may be freshly constructed, not source-index-mutated old
    # claims. The parent's raw replay establishes that distinction.
    if not isinstance(fresh_motion, EditorialMotionPlan) or len(fresh_motion.segments) != len(before.segments):
        raise ValueError('joint metadata lacks complete fresh motion source/camera rows')
    for index, (old, new) in enumerate(zip(old_motion.segments, fresh_motion.segments, strict=True)):
        if index not in changed and old != new:
            raise ValueError('joint metadata lost unrelated motion segment history')
    # A global plan binding may change; preserve the actual support payload
    # when rebinding, never copy an old whole-plan digest to the new plan.
    contexts = tuple(rebind_unchanged_shot_context(row, before, expected)
        for row in old_motion.shot_context_cameras)
    coupled = tuple(rebind_unchanged_coupled_shot(row, before, expected)
        for row in old_motion.coupled_shot_cameras)
    if any(row not in fresh_motion.shot_context_cameras for row in contexts) or any(
            row not in fresh_motion.coupled_shot_cameras for row in coupled):
        raise ValueError('joint metadata lost a retained whole-shot camera receipt')
    old_alignment, new_alignment = old_motion.boundary_subject_alignment_plan, fresh_motion.boundary_subject_alignment_plan
    incident = {edge for cell in changed for edge in (cell-1, cell)
        if 0 <= edge < len(before.transitions)}
    for name in ('accepted_boundary_indices', 'natural_match_boundary_indices'):
        retained = set(getattr(old_motion, name))-incident
        if not retained.issubset(getattr(fresh_motion, name)):
            raise ValueError('joint metadata lost unrelated accepted boundary history')
    for name in ('boundary_residual_attention_mismatches', 'required_boundary_failures'):
        if any(row[0] not in incident and row not in getattr(fresh_motion, name)
                for row in getattr(old_motion, name)):
            raise ValueError('joint metadata lost unrelated boundary failure/residual history')
    if any(getattr(fresh_motion, name) != getattr(old_motion, name)
            for name in ('member_roll_audit', 'member_containment_trial_audit')):
        raise ValueError('GROUP metadata cannot rewrite a member-only evidence family')
    if old_alignment is not None:
        if new_alignment is None:
            raise ValueError('joint metadata cannot erase the incumbent subject-alignment family')
        new_decisions = {row.boundary_index: row for row in new_alignment.decisions}
        for row in old_alignment.decisions:
            if row.boundary_index not in incident and new_decisions.get(row.boundary_index) != row:
                raise ValueError('joint metadata lost unrelated subject-alignment decision history')
    old_face = incumbent.face_match_plan
    if old_face is not None:
        if fresh_face_match is None or len(fresh_face_match.segments) != len(before.segments):
            raise ValueError('joint metadata cannot erase its incumbent face/pan family')
        for index, (old, new) in enumerate(zip(old_face.segments, fresh_face_match.segments, strict=True)):
            if index not in changed and old != new:
                raise ValueError('joint metadata lost unrelated face/pan history')
    repairs, affine, preservation = preserve_boundary_evidence(old_plan=before, new_plan=expected,
        old_timeline=incumbent.visual_timeline, new_timeline=timeline,
        boundary_repairs=incumbent.boundary_repairs, boundary_affine_plan=incumbent.boundary_affine_plan)
    review = review_timeline(timeline, available_source_count=len(domain.eligible_indices))
    candidate = replace(incumbent, render_plan=expected, visual_timeline=timeline,
        timeline_review=review, pre_render_score=float(review.weighted_score),
        editorial_motion_plan=fresh_motion, face_match_plan=fresh_face_match,
        boundary_repairs=repairs, boundary_affine_plan=affine,
        group_transition_plan=GroupTransitionPlan(expected.transitions,
            analysis_basis='joint_source_camera_numeric_metadata_pending_raw_v2_parent_replay'),
        stage_edit_exact_shot_clock=_deterministic_exact_shot_clock_from_timeline(timeline,
            section_plan=incumbent.section_plan, phrase_times=phrases), phrase_times=phrases,
        candidate_lineage=(*incumbent.candidate_lineage, dict(
            stage='joint_source_camera_complete_numeric_metadata_construction',
            changed_segment_indices=changed, refreshed_segment_indices=refreshed,
            boundary_evidence_preservation=preservation,
            candidate_admitted=False, raw_v2_parent_replay_required=True, quality_approved=False)))
    _validate_review_first_group_bundle(candidate, eligible_indices=domain.eligible_indices,
        offsets=domain.audio_offsets, timewarps=domain.source_timewarps,
        common_start=before.master_audio_start, phrase_times=phrases, shot_clock_context=shot_clock_context)
    return GroupJointMetadataProposal(_plan_identity(before), candidate, changed)
