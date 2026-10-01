"""Coupled source/entry/exit proposal for one GROUP shot, before camera proof.

A source that matches the entering pose can contain a broadcaster cut just
before the old exit. Fixed-span selection cannot use it even when shifting the
exit a few frames would expose a good interval. This constructs that interval;
it does NOT grant geometry, identity, alignment or production-selection proof.
"""
from dataclasses import dataclass, fields, replace
import math

from .group_cut_camera_proposal import (
    assert_preserved_output_boundaries, retime_base_pan, splice_boundary_camera,
    UNREVIEWED_GROUP_SOURCE_TRANSITION_REASON,
)
from .render import RenderPlan, RenderTransition, aligned_render_segment, segment_frame_counts, _can_share_source_clock
from .render_boundary_clock import native_frame_clocks_for_output
from .render_frame_schedule import VerifiedRenderFrameClock
from .visual_score import VisualTimelinePlan


@dataclass(frozen=True)
class GroupSourceSpanProposal:
    plan: RenderPlan
    clock_timeline: VisualTimelinePlan
    segment_index: int
    source_index: int
    entry_offset_frames: int
    exit_offset_frames: int
    preserved_neighbor_boundaries: tuple[int, ...]
    maximum_shift_frames: int = 8
    last_segment_index: int | None = None

    @property
    def end_segment_index(self):
        return self.segment_index if self.last_segment_index is None else self.last_segment_index

    def to_dict(self):
        return dict(contract=('coupled_group_source_run_clock_proposal_v2'
            if self.last_segment_index is not None else 'coupled_group_source_span_clock_proposal_v1'),
            segment_index=self.segment_index,source_index=self.source_index,
            last_segment_index=self.end_segment_index,
            entry_offset_frames=self.entry_offset_frames,exit_offset_frames=self.exit_offset_frames,
            maximum_shift_frames=self.maximum_shift_frames,
            preserved_neighbor_boundaries=list(self.preserved_neighbor_boundaries),
            preserved_output_frames_per_neighbor_side=14,
            timeline_descriptors_refreshed=False,camera_geometry_validated=False,
            reset_transition_boundaries=list(range(self.segment_index-1,self.end_segment_index+1)),
            transition_evidence_refreshed=False,
            source_reference_alignment_authority='caller_supplied',
            duration_floor_basis='visible_source_run_not_internal_beat_cells',
            production_selected=False,quality_approved=False)


def build_group_source_span_proposal(*,plan,timeline,segment_index,source_index,
        entry_offset_frames,exit_offset_frames,audio_offsets,source_timewarps,
        scene_cut_times,source_clocks,maximum_shift_frames=8,last_segment_index=None):
    """Identity source domain; conserve outer clocks and every untouched pixel input.

    Do not copy the previous source's crop, zoom, grade or face observations as
    evidence for the replacement. Both new boundaries require fresh complete
    camera construction and visual review after this numeric proposal. An
    explicit last index replaces a complete coalescible render-source run,
    not one artificial beat cell. Render-clock continuity does NOT prove the
    original run contains no broadcaster cuts; the replacement interval must.
    """
    if not isinstance(plan,RenderPlan) or not isinstance(timeline,VisualTimelinePlan):
        raise TypeError('source span proposal needs a complete plan and timeline')
    if (type(maximum_shift_frames) is not int or not 1 <= maximum_shift_frames <= math.floor(plan.fps)):
        raise ValueError('source span search radius must be an explicit positive budget of at most one second')
    if (type(segment_index) is not int or not 1 <= segment_index < len(plan.segments)-1
            or type(source_index) is not int or not 0 <= source_index < len(plan.sources)
            or any(type(value) is not int or abs(value)>maximum_shift_frames
                   for value in (entry_offset_frames,exit_offset_frames))):
        raise ValueError('source span proposal needs one interior shot and bounded whole-frame shifts')
    last=segment_index if last_segment_index is None else last_segment_index
    if type(last) is not int or not segment_index <= last < len(plan.segments)-1:
        raise ValueError('source run proposal needs an explicit interior final cell')
    if (not len(audio_offsets)==len(source_timewarps)==len(scene_cut_times)==len(plan.sources)
            or len(timeline.choices)!=len(plan.segments)
            or plan.aspect_ratio.value!='16:9'
            or any(transition.kind.uses_overlap for transition in plan.transitions)
            or any(type(value) not in (int,float) or not math.isfinite(value) for value in audio_offsets)
            or any(type(value) not in (int,float) or not math.isfinite(value) or value<0
                   for cuts in scene_cut_times for value in cuts)):
        raise ValueError('source span proposal source/clock/transition domain differs')
    for index,(choice,segment) in enumerate(zip(timeline.choices,plan.segments,strict=True)):
        if (choice.segment_index!=index or choice.source_index!=segment.source_index or any(abs(a-b)>1e-6 for a,b in (
                (choice.canonical_master_start,plan.master_audio_start+segment.master_start),
                (choice.duration_seconds,segment.duration),(choice.source_local_start,segment.source_start),
                (choice.source_local_end,segment.source_end)))):
            raise ValueError('source span proposal timeline is not bound to the render clock')
    i=segment_index
    if last_segment_index is not None:
        run=plan.segments[i:last+1]
        if (not _can_share_source_clock(plan,first_index=i,candidate_last_index=last)
                or run[0].source_index in (plan.segments[i-1].source_index,plan.segments[last+1].source_index)
                or sum(segment_frame_counts(plan)[i:last+1])>300):
            raise ValueError('source run proposal requires one complete bounded render-source run')
    if source_index in {plan.segments[j].source_index for j in range(i-1,last+2)}:
        raise ValueError('source span proposal needs a distinct replacement and cross-source outer seams')
    if any(segment.crop is not None or segment.outgoing_overlap_transform is not None
            or segment.boundary_look_track is not None for segment in plan.segments[i-1:last+2]):
        raise ValueError('source span proposal must precede fixed crops, overlap and look finalization')
    old_left,old_middle,old_right=plan.segments[i-1],plan.segments[i],plan.segments[last+1]
    entry=old_middle.master_start+entry_offset_frames/plan.fps
    exit=old_right.master_start+exit_offset_frames/plan.fps
    times=(old_left.master_start,entry,exit,old_right.master_start+old_right.duration)
    if any(b-a < .55 for a,b in zip(times,times[1:])):
        raise ValueError('source span proposal shortened a shot below its constructor floor')
    segments=list(plan.segments)
    choices=list(timeline.choices)
    intervals=[(i-1,old_left.source_index,times[0],entry)]
    intervals.extend((j,source_index,entry if j==i else plan.segments[j].master_start,
        exit if j==last else plan.segments[j].master_start+plan.segments[j].duration)
        for j in range(i,last+1))
    intervals.append((last+1,old_right.source_index,exit,times[-1]))
    # The three visible shot intervals already meet the .55s floor above.
    # Internal beat cells inside one continuous replacement are NOT new cuts;
    # imposing that floor again on each cell incorrectly rejects usable runs.
    # Retiming must still leave each cell positive and the renderer's globally
    # quantized two-frame minimum below remains mandatory.
    if any(end <= start for _,_,start,end in intervals):
        raise ValueError('source span proposal collapsed an internal beat cell')
    for j,source,start,end in intervals:
        old=segments[j]
        new=aligned_render_segment(source_index=source,
            reference_start=plan.master_audio_start+start,common_master_start=plan.master_audio_start,
            source_offset=audio_offsets[source],duration=end-start,timewarp=source_timewarps[source])
        if j not in range(i,last+1):
            new=replace(new,pan_crop=retime_base_pan(old.pan_crop,
                relative_shift=start-old.master_start,duration=new.duration),look_adjustment=old.look_adjustment)
        segments[j]=new
        clock=source_clocks.get(source)
        if not isinstance(clock,VerifiedRenderFrameClock):
            raise ValueError('replacement source has no verified frame clock')
        native_tick=1./float(clock.frame_rate)
        handle=3/plan.fps*new.source_playback_rate
        choices[j]=replace(choices[j],source_index=source,
            canonical_master_start=plan.master_audio_start+start,
            canonical_master_end=plan.master_audio_start+end,
            source_local_start=new.source_start,source_local_end=new.source_end,
            internal_scene_cut_master_seconds=tuple(plan.master_audio_start+start+
                (cut-new.source_start)/new.source_playback_rate for cut in scene_cut_times[source]
                if new.source_start<cut<new.source_end),
            entry_scene_cut_near=any(new.source_start+native_tick < cut <= new.source_start+handle
                for cut in scene_cut_times[source]),
            exit_scene_cut_near=any(new.source_end-handle <= cut < new.source_end-native_tick
                for cut in scene_cut_times[source]))
    # The desired replacement must really be one continuous native shot.
    replacement_start,replacement_end=segments[i].source_start,segments[last].source_end
    # Inspect the UNION too: a native cut exactly at an artificial cell seam
    # must not disappear from every open per-cell internal-cut interval.
    if any(replacement_start < cut < replacement_end for cut in scene_cut_times[source_index]):
        raise ValueError('replacement interval still crosses a broadcaster camera cut')
    # A same-source tier-D hold cannot authorize a new cross-source cut. Even
    # an old tier-C/A/B score refers to different pixels and source identities.
    # Reset only the replaced interval's seams, leaving every outside recipe
    # intact. These neutral cuts still need fresh pose/identity/camera review.
    reset_boundaries=tuple(range(i-1,last+1))
    transitions=tuple(RenderTransition(tier='C',reason=UNREVIEWED_GROUP_SOURCE_TRANSITION_REASON)
        if boundary in reset_boundaries else transition
        for boundary,transition in enumerate(plan.transitions))
    proposed=replace(plan,segments=tuple(segments),transitions=transitions)
    counts=segment_frame_counts(proposed)
    changed_frames=range(sum(counts[:i-1]),sum(counts[:last+2]))
    changed_clocks=native_frame_clocks_for_output(proposed,source_clocks,changed_frames)
    if any(frame.terminal_clone for frame in changed_clocks.values()):
        raise ValueError('source span proposal cannot introduce terminal-cloned frames')
    for j,outgoing in ((i-1,True),(last+1,False)):
        old,new=plan.segments[j],segments[j]
        segments[j]=replace(new,editorial_transform=(splice_boundary_camera(old.editorial_transform,None,
            relative_shift=new.master_start-old.master_start,frame_count=counts[j],fps=plan.fps,outgoing=outgoing)
            if old.editorial_transform is not None else None))
    proposed=replace(proposed,segments=tuple(segments))
    preserved=assert_preserved_output_boundaries(plan,proposed,source_clocks,(i-2,last+1),
        allow_same_source_seams=True,complete_interval_bounds=(i-1,last+1),
        reset_transition_boundaries=reset_boundaries)
    return GroupSourceSpanProposal(proposed,replace(timeline,choices=tuple(choices),
        analysis_basis='UNREFRESHED_source_span_clock_proposal_not_a_selected_timeline'),i,source_index,
        entry_offset_frames,exit_offset_frames,preserved,maximum_shift_frames,last_segment_index)


def refresh_group_source_span_observations(*,proposal,audio_offsets,source_timewarps,sync_scores,
        scene_cut_times,feature_cache_directory=None,feature_derived_cache_directory=None,
        require_persistent_feature_cache=False):
    """Fresh local observations for camera audition, NOT source selection proof.

    Only the replaced run and its two retimed neighbors are sensed. The new
    source cannot retain the old source's face/cut/family/quality metadata.
    The fixed path deliberately bypasses legacy selection: cameras still need
    to be constructed before the complete path's ordinary visual review.
    No passing global review, named identity, phase or output QA is implied.
    """
    if not isinstance(proposal,GroupSourceSpanProposal):
        raise TypeError('local source observation refresh needs a typed clock proposal')
    from .visual_score import (build_visual_timeline_optimizer_seed,VisualTimelineCandidateSet,
        VisualTimelineCandidateRecord)
    from .timeline import TimelineChoice,TimelineResult,TimelineMode
    from .timeline_review import review_timeline
    plan=proposal.plan;first=proposal.segment_index-1;last=proposal.end_segment_index+1
    chosen=proposal.clock_timeline.choices[first:last+1]
    origin=plan.segments[first].master_start
    seed=build_visual_timeline_optimizer_seed(sources=plan.sources,
        canonical_spans=tuple((c.canonical_master_start,c.canonical_master_end) for c in chosen),
        audio_offsets=audio_offsets,source_timewarps=source_timewarps,sync_scores=sync_scores,
        scene_cut_times=scene_cut_times,candidate_source_indices_by_segment=tuple((c.source_index,) for c in chosen),
        group_incumbent_render_segments=tuple(replace(s,master_start=s.master_start-origin)
            for s in plan.segments[first:last+1]),feature_cache_directory=feature_cache_directory,
        feature_derived_cache_directory=feature_derived_cache_directory,
        require_persistent_feature_cache=require_persistent_feature_cache)
    cells=seed.optimizer_workspace.candidates_by_segment
    if any(len(cell)!=1 for cell in cells):
        raise ValueError('local source observation refresh expanded its fixed source domain')
    fixed=TimelineResult(TimelineMode.GENERAL,None,tuple(TimelineChoice(cell[0],0.,0.,0.,False)
        for cell in cells),0.)
    review=review_timeline(fixed,available_source_count=len(plan.sources))
    record=VisualTimelineCandidateRecord(0,'unapproved_fixed_source_clock_observations',fixed,review,0,True)
    local=VisualTimelineCandidateSet(candidates=(record,),selected_candidate_index=0,
        sources=seed.sources,audio_offsets=seed.audio_offsets,source_timewarps=seed.source_timewarps,
        source_clock_basis=seed.source_clock_basis,requested_maximum_results=seed.requested_maximum_results,
        scene_cut_times=seed.scene_cut_times,face_metadata_by_candidate=seed.face_metadata_by_candidate,
        subject_metadata_by_candidate=seed.subject_metadata_by_candidate,
        optimizer_workspace=seed.optimizer_workspace,optimizer_input_identity=seed.optimizer_input_identity,
        feature_cache_directory=seed.feature_cache_directory,feature_derived_cache_directory=seed.feature_derived_cache_directory,
        require_persistent_feature_cache=seed.require_persistent_feature_cache).materialize()
    choices=list(proposal.clock_timeline.choices)
    for offset,choice in enumerate(local.choices):
        index=first+offset
        if (choice.source_index!=plan.segments[index].source_index or any(abs(a-b)>1e-6 for a,b in (
                (choice.source_local_start,plan.segments[index].source_start),
                (choice.source_local_end,plan.segments[index].source_end)))):
            raise ValueError('fresh source observations changed the proposed source clock')
        unchanged_outer_edge=({field.name:getattr(choices[index],field.name)
            for field in fields(choice) if field.name.endswith('_from_previous')} if offset==0 else {})
        choices[index]=replace(choice,segment_index=index,**unchanged_outer_edge)
    return VisualTimelinePlan(tuple(choices),0.,
        analysis_basis='fresh_local_source_span_observations_not_a_selected_timeline_v1',
        candidate_generation_basis='fixed_local_clock_observations_no_global_selection_or_approval_v1',
        source_clock_basis=local.source_clock_basis,geometry_basis=local.geometry_basis)
