"""Coupled source/entry/exit proposal for one GROUP shot, before camera proof.

A source that matches the entering pose can contain a broadcaster cut just
before the old exit. Fixed-span selection cannot use it even when shifting the
exit a few frames would expose a good interval. This constructs that interval;
it does NOT grant geometry, identity, alignment or production-selection proof.
"""
from dataclasses import dataclass, replace
import math

from .group_cut_camera_proposal import (
    assert_preserved_output_boundaries, retime_base_pan, splice_boundary_camera,
)
from .render import RenderPlan, aligned_render_segment, segment_frame_counts
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

    def to_dict(self):
        return dict(contract='coupled_group_source_span_clock_proposal_v1',
            segment_index=self.segment_index,source_index=self.source_index,
            entry_offset_frames=self.entry_offset_frames,exit_offset_frames=self.exit_offset_frames,
            maximum_shift_frames=self.maximum_shift_frames,
            preserved_neighbor_boundaries=list(self.preserved_neighbor_boundaries),
            preserved_output_frames_per_neighbor_side=14,
            timeline_descriptors_refreshed=False,camera_geometry_validated=False,
            source_reference_alignment_authority='caller_supplied',
            production_selected=False,quality_approved=False)


def build_group_source_span_proposal(*,plan,timeline,segment_index,source_index,
        entry_offset_frames,exit_offset_frames,audio_offsets,source_timewarps,
        scene_cut_times,source_clocks,maximum_shift_frames=8):
    """Identity source domain; conserve outer clocks and every untouched pixel input.

    Do not copy the previous source's crop, zoom, grade or face observations as
    evidence for the replacement. Both new boundaries require fresh complete
    camera construction and visual review after this numeric proposal.
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
    if source_index in {plan.segments[j].source_index for j in (i-1,i,i+1)}:
        raise ValueError('source span proposal needs a distinct replacement and cross-source outer seams')
    if any(segment.crop is not None or segment.outgoing_overlap_transform is not None
            or segment.boundary_look_track is not None for segment in plan.segments[i-1:i+2]):
        raise ValueError('source span proposal must precede fixed crops, overlap and look finalization')
    old_left,old_middle,old_right=plan.segments[i-1:i+2]
    entry=old_middle.master_start+entry_offset_frames/plan.fps
    exit=old_right.master_start+exit_offset_frames/plan.fps
    times=(old_left.master_start,entry,exit,old_right.master_start+old_right.duration)
    if any(b-a < .55 for a,b in zip(times,times[1:])):
        raise ValueError('source span proposal shortened a shot below its constructor floor')
    segments=list(plan.segments)
    choices=list(timeline.choices)
    for j,source,start,end in zip((i-1,i,i+1),
            (old_left.source_index,source_index,old_right.source_index),times[:-1],times[1:],strict=True):
        old=segments[j]
        new=aligned_render_segment(source_index=source,
            reference_start=plan.master_audio_start+start,common_master_start=plan.master_audio_start,
            source_offset=audio_offsets[source],duration=end-start,timewarp=source_timewarps[source])
        if j!=i:
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
    if choices[i].internal_scene_cut_master_seconds:
        raise ValueError('replacement interval still crosses a broadcaster camera cut')
    proposed=replace(plan,segments=tuple(segments))
    counts=segment_frame_counts(proposed)
    changed_frames=range(sum(counts[:i-1]),sum(counts[:i+2]))
    changed_clocks=native_frame_clocks_for_output(proposed,source_clocks,changed_frames)
    if any(frame.terminal_clone for frame in changed_clocks.values()):
        raise ValueError('source span proposal cannot introduce terminal-cloned frames')
    for j,outgoing in ((i-1,True),(i+1,False)):
        old,new=plan.segments[j],segments[j]
        segments[j]=replace(new,editorial_transform=splice_boundary_camera(old.editorial_transform,None,
            relative_shift=new.master_start-old.master_start,frame_count=counts[j],fps=plan.fps,outgoing=outgoing))
    proposed=replace(proposed,segments=tuple(segments))
    preserved=assert_preserved_output_boundaries(plan,proposed,source_clocks,(i-2,i+1))
    return GroupSourceSpanProposal(proposed,replace(timeline,choices=tuple(choices),
        analysis_basis='UNREFRESHED_source_span_clock_proposal_not_a_selected_timeline'),i,source_index,
        entry_offset_frames,exit_offset_frames,preserved,maximum_shift_frames)
