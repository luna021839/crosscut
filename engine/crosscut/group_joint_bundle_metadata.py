"""Complete numeric GROUP metadata after an atomic whole-shot proposal.

This is a producer/consumer construction step, not the source-camera owner or
raw-v2 admission. The static owner must supply freshly produced source/camera
dependent motion and face metadata; the parent still owes raw semantic replay.
No child `complete` flag and no descriptor hash certifies media observations.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .render import RenderPlan
    from .visual_score import VisualTimelinePlan
    from .shot_video_clock import FrozenShotVideoClockContext
    from .group_joint_source_camera import GroupOutputRole

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


@dataclass(frozen=True,slots=True)
class SelectedNativeActorFrame:
    output_tick: int
    request_source_index: int
    native_frame_index: int
    frame_local_head_id: int | None
    native_observation_sha256: str | None


@dataclass(frozen=True,slots=True)
class SelectedNativeMotionShot:
    first_output_frame: int
    end_output_frame: int
    request_source_index: int
    physical_native_run: tuple[int,int]
    role: GroupOutputRole | None
    phase_native_frames: int
    actor_path_identity: str | None
    actor_frames: tuple[SelectedNativeActorFrame,...]
    render_segment_indices: tuple[int,...]


@dataclass(frozen=True,slots=True)
class GroupNativeMotionProduction:
    """The selected program itself, carried intact from producer to encoder.

    This record does not create native observations, actor appearance vectors,
    cameras, phase permission or a quality decision. Its normal producer must
    already have selected those exact objects before constructing the record.
    """
    visual_timeline: VisualTimelinePlan
    render_plan: RenderPlan
    editorial_motion_plan: EditorialMotionPlan
    source_domain: GroupSourceDomain
    shot_clock_context: FrozenShotVideoClockContext | None
    selected_shots: tuple[SelectedNativeMotionShot,...]
    original_plan_identity: str
    protected_output_intervals: tuple[tuple[int,int],...]
    state: Literal['SELECTED_FOR_REVIEW','KEEP_ORIGINAL']
    complete_bundle: object | None = None
    selection_evidence_json: str | None = None
    selection_sha256: str | None = None

    def to_dict(self):
        from dataclasses import asdict
        return dict(contract='normal_native_motion_selected_program_v2',state=self.state,
            original_plan_identity=self.original_plan_identity,
            selected_plan_identity=_plan_identity(self.render_plan),
            protected_output_intervals=[list(x) for x in self.protected_output_intervals],
            selected_shots=[dict(asdict(s),role=None if s.role is None else s.role.value)
                for s in self.selected_shots],
            has_local_video_clock_context=self.shot_clock_context is not None,
            shot_video_clock_context=None if self.shot_clock_context is None else self.shot_clock_context.to_dict(),
            has_complete_bundle=self.complete_bundle is not None,
            selection_sha256=self.selection_sha256,
            quality_approved=False)


def _selected_production_payload(production):
    """Bind every retained field, not just the segment list or a child badge.

    The inner bundle has no production backlink, avoiding a serialization
    cycle. Only the final output filename is relocatable; the frozen phase-zero
    parent is serialized in full because its clocks bind that exact parent.
    This seal detects mutation. It does not certify observations or quality.
    """
    from dataclasses import asdict
    plan=production.render_plan.to_json()
    plan.pop('output')
    context=production.shot_clock_context
    return dict(contract='complete_native_motion_selection_seal_v1',
        render_plan=plan,complete_bundle=production.complete_bundle.to_dict(),
        request_sources=[str(path) for path in production.source_domain.request_sources],
        source_domain=production.source_domain.to_dict(),
        shot_clock_context=None if context is None else context.to_dict(),
        selected_shots=[dict(asdict(shot),role=None if shot.role is None else shot.role.value)
            for shot in production.selected_shots],
        original_plan_identity=production.original_plan_identity,
        protected_output_intervals=[list(pair) for pair in production.protected_output_intervals],
        state=production.state,selection_evidence=json.loads(production.selection_evidence_json))


def _selection_digest(production):
    return hashlib.sha256(json.dumps(_selected_production_payload(production),
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _check_native_selection(production):
    from .group_joint_source_camera import GroupOutputRole
    from .render import RenderPlan, segment_frame_counts
    from .render_boundary_clock import native_frame_clocks_for_output
    from .shot_video_clock import FrozenShotVideoClockContext
    from .pipeline import _GroupRenderCandidateBundle, _validate_review_first_group_bundle
    inner=production.complete_bundle
    if (not isinstance(inner,_GroupRenderCandidateBundle)
            or inner.native_motion_production is not None
            or not isinstance(production.render_plan,RenderPlan)
            or inner.render_plan!=production.render_plan
            or inner.visual_timeline!=production.visual_timeline
            or inner.editorial_motion_plan!=production.editorial_motion_plan
            or not isinstance(production.source_domain,GroupSourceDomain)
            or production.render_plan.sources!=production.source_domain.request_sources):
        raise ValueError('native production requires its exact complete selected bundle without a backlink')
    if (production.state not in ('SELECTED_FOR_REVIEW','KEEP_ORIGINAL')
            or not isinstance(production.selected_shots,tuple)
            or not isinstance(production.protected_output_intervals,tuple)
            or not isinstance(production.original_plan_identity,str)
            or len(production.original_plan_identity)!=64
            or any(c not in '0123456789abcdef' for c in production.original_plan_identity)
            or type(inner.candidate_index) is not int or inner.candidate_index<0
            or type(inner.review_rank) is not int or inner.review_rank<0
            or not isinstance(inner.cadence_profile,str) or not inner.cadence_profile):
        raise ValueError('native selection requires its frozen program and actual stream identity')
    context=production.shot_clock_context
    if context is not None and (not isinstance(context,FrozenShotVideoClockContext)
            or context.source_domain!=production.source_domain):
        raise ValueError('native selection borrowed another phase-zero source clock context')
    if context is not None and context.original_reference_plan is not None and (
            _plan_identity(context.original_reference_plan)!=production.original_plan_identity
            or context.protected_output_intervals!=production.protected_output_intervals
            or context.mutable_output_intervals!=tuple((shot.first_output_frame,shot.end_output_frame)
                for shot in production.selected_shots)):
        raise ValueError('selected native program changed its original root or declared protection/scope')
    try:
        evidence=json.loads(production.selection_evidence_json)
    except (TypeError,ValueError) as exc:
        raise ValueError('native production needs retained selection evidence') from exc
    if not isinstance(evidence,dict):
        raise ValueError('native production needs retained selection evidence')
    # JSON encoding also rejects non-finite metadata, without treating a
    # producer-supplied quality/admission flag as executable authority.
    json.dumps(evidence,allow_nan=False)
    domain=production.source_domain
    _validate_review_first_group_bundle(inner,eligible_indices=domain.eligible_indices,
        offsets=domain.audio_offsets,timewarps=domain.source_timewarps,
        common_start=production.render_plan.master_audio_start,
        phrase_times=inner.phrase_times,shot_clock_context=context)
    offsets=[0]
    for count in segment_frame_counts(production.render_plan):offsets.append(offsets[-1]+count)
    previous_end=0
    for pair in production.protected_output_intervals:
        if (not isinstance(pair,tuple) or len(pair)!=2
                or any(type(v) is not int for v in pair)
                or not previous_end<=pair[0]<pair[1]<=offsets[-1]):
            raise ValueError('native selection has malformed protected output intervals')
        previous_end=pair[1]
    if production.state=='KEEP_ORIGINAL':
        if production.selected_shots or _plan_identity(production.render_plan)!=production.original_plan_identity:
            raise ValueError('KEEP_ORIGINAL must retain the original plan without replacement shots')
        return
    if len(production.selected_shots)!=2:
        raise ValueError('selected native production needs both complete adjacent shot paths')
    previous_end=None
    for shot in production.selected_shots:
        if (not isinstance(shot,SelectedNativeMotionShot)
                or type(shot.first_output_frame) is not int or type(shot.end_output_frame) is not int
                or not 0<=shot.first_output_frame<shot.end_output_frame<=offsets[-1]
                or not 2<=shot.end_output_frame-shot.first_output_frame<=160
                or previous_end is not None and previous_end!=shot.first_output_frame
                or type(shot.request_source_index) is not int
                or shot.request_source_index not in domain.eligible_indices
                or shot.role is not None and not isinstance(shot.role,GroupOutputRole)
                or type(shot.phase_native_frames) is not int or abs(shot.phase_native_frames)>2
                or not isinstance(shot.physical_native_run,tuple) or len(shot.physical_native_run)!=2
                or any(type(v) is not int for v in shot.physical_native_run)
                or not 0<=shot.physical_native_run[0]<shot.physical_native_run[1]
                or not isinstance(shot.render_segment_indices,tuple)
                or any(type(v) is not int for v in shot.render_segment_indices)
                or shot.first_output_frame not in offsets or shot.end_output_frame not in offsets
                or shot.render_segment_indices!=tuple(range(offsets.index(shot.first_output_frame),
                    offsets.index(shot.end_output_frame)))
                or not isinstance(shot.actor_frames,tuple)
                or len(shot.actor_frames)!=shot.end_output_frame-shot.first_output_frame
                or any(a<shot.end_output_frame and b>shot.first_output_frame
                    for a,b in production.protected_output_intervals)):
            raise ValueError('selected native shot differs from its whole output/source/segment path')
        if shot.actor_path_identity is not None and (not isinstance(shot.actor_path_identity,str)
                or len(shot.actor_path_identity)!=64
                or any(c not in '0123456789abcdef' for c in shot.actor_path_identity)):
            raise ValueError('selected native shot has malformed actor path identity')
        authorities=tuple(clock for clock in production.render_plan.authorized_shot_video_clocks
            if set(clock.segment_indices).intersection(shot.render_segment_indices))
        if (not authorities and shot.phase_native_frames!=0
                or authorities and (len(authorities)!=1
                    or authorities[0].segment_indices!=shot.render_segment_indices
                    or authorities[0].first_delta_native_frames!=shot.phase_native_frames
                    or authorities[0].last_delta_native_frames!=shot.phase_native_frames)):
            raise ValueError('selected native phase differs from its authorized whole-shot clock')
        schedule=native_frame_clocks_for_output(production.render_plan,domain.clocks_by_request,
            range(shot.first_output_frame,shot.end_output_frame))
        for tick,row in zip(range(shot.first_output_frame,shot.end_output_frame),shot.actor_frames,strict=True):
            actual=schedule[tick]
            if (not isinstance(row,SelectedNativeActorFrame)
                    or type(row.output_tick) is not int or row.output_tick!=tick
                    or type(row.request_source_index) is not int
                    or row.request_source_index!=shot.request_source_index
                    or actual.source_index!=shot.request_source_index
                    or type(row.native_frame_index) is not int
                    or row.native_frame_index!=actual.native_frame_index or actual.terminal_clone
                    or not shot.physical_native_run[0]<=row.native_frame_index<shot.physical_native_run[1]
                    or row.frame_local_head_id is not None and (type(row.frame_local_head_id) is not int
                        or row.frame_local_head_id<0)
                    or row.native_observation_sha256 is not None and (
                        not isinstance(row.native_observation_sha256,str)
                        or len(row.native_observation_sha256)!=64
                        or any(c not in '0123456789abcdef' for c in row.native_observation_sha256))
                    or row.frame_local_head_id is not None and row.native_observation_sha256 is None):
                raise ValueError('selected actor/native frame path differs from the exact renderer sampler')
        previous_end=shot.end_output_frame


def seal_native_motion_production(*,complete_bundle,source_domain,shot_clock_context,
        selected_shots,original_plan_identity,protected_output_intervals,state,selection_evidence):
    """Seal an already complete producer result; never invent missing metadata.

    This function is not the full-topology observation producer or admission
    gate. The caller owes fresh actor/body/face/boundary metadata for changed
    incidents and preserved exterior history before passing its full bundle.
    """
    production=GroupNativeMotionProduction(complete_bundle.visual_timeline,complete_bundle.render_plan,
        complete_bundle.editorial_motion_plan,source_domain,shot_clock_context,selected_shots,
        original_plan_identity,protected_output_intervals,state,complete_bundle,
        json.dumps(selection_evidence,sort_keys=True,separators=(',',':'),allow_nan=False))
    _check_native_selection(production)
    return replace(production,selection_sha256=_selection_digest(production))


def validate_native_motion_production(production,*,attached_bundle=None):
    """Reject missing/full-field mutations while permitting output relocation."""
    if not isinstance(production,GroupNativeMotionProduction):
        raise ValueError('native production requires a typed complete selection')
    _check_native_selection(production)
    if (not isinstance(production.selection_sha256,str)
            or production.selection_sha256!=_selection_digest(production)):
        raise ValueError('complete selected native bundle changed after motion/camera evaluation')
    if attached_bundle is not None:
        from .pipeline import _GroupRenderCandidateBundle
        if (not isinstance(attached_bundle,_GroupRenderCandidateBundle)
                or attached_bundle.native_motion_production is not production):
            raise ValueError('selected native bundle lost its production binding')
        detached=replace(attached_bundle,native_motion_production=None,
            render_plan=replace(attached_bundle.render_plan,output=production.render_plan.output))
        if (detached.render_plan!=production.render_plan
                or detached.to_dict()!=production.complete_bundle.to_dict()):
            raise ValueError('complete selected native bundle changed after motion/camera evaluation')
