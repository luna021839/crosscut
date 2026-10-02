"""Bounded numeric cut-time hypotheses on the existing whole renderer clocks.

Moving a cut is different from advancing one source's choreography phase.
The outgoing/incoming linear render units keep their absolute source/camera
curves; only ownership of at most eight output ticks changes. Nothing here
refreshes descriptors, roles, receipts, independent audio or quality evidence.
Do not publish the stale incumbent bundle with the returned preview plan.
"""
from dataclasses import dataclass, replace
import hashlib
import json

from .group_actual_shots import (VisibleGroupCut, assert_native_cut_events_preserved,
    visible_group_cut_inventory)
from .group_source_domain import GroupSourceDomain
from .group_cut_camera_proposal import retime_base_pan, UNREVIEWED_GROUP_SOURCE_TRANSITION_REASON
from .render import (RenderPlan, EditorialTransformTrack, RenderTransition,
    segment_frame_counts, _video_render_units)
from .render_boundary_clock import native_frame_clocks_for_output


def _plan_hash(plan):
    return hashlib.sha256(json.dumps(plan.to_json(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _retime_camera(track, *, shift, duration):
    if track is None:return None
    times=sorted({0.,duration,*(k.time_seconds-shift for k in track.keyframes
        if 0.<k.time_seconds-shift<duration)})
    keys=tuple(replace(track.evaluate(max(0.,time+shift)),time_seconds=time) for time in times)
    # Trimming a high-zoom sample must not change the entire sampling canvas.
    return EditorialTransformTrack(keys,reason='absolute_incumbent_camera_cut_time_preview_not_dense_authority',
        sampling_canvas_zoom=track.effective_sampling_canvas_zoom)


@dataclass(frozen=True,slots=True)
class NumericGroupCutTimingProposal:
    plan: RenderPlan
    incumbent_plan_sha256: str
    candidate_plan_sha256: str
    boundary_index: int
    timing_offset_frames: int
    old_cut_output_tick: int
    new_cut_output_tick: int
    changed_cell_indices: tuple[int,int]
    changed_source_owner_ticks: tuple[int,...]
    actual_native_camera_events_preserved: bool
    collapsed_native_camera_event: VisibleGroupCut | None = None

    def to_dict(self):
        return dict(contract='numeric_absolute_whole_source_clock_group_cut_timing_preview_v1',
            incumbent_plan_sha256=self.incumbent_plan_sha256,candidate_plan_sha256=self.candidate_plan_sha256,
            boundary_index=self.boundary_index,timing_offset_frames=self.timing_offset_frames,
            old_cut_output_tick=self.old_cut_output_tick,new_cut_output_tick=self.new_cut_output_tick,
            changed_cell_indices=self.changed_cell_indices,changed_source_owner_ticks=self.changed_source_owner_ticks,
            actual_native_camera_events_preserved=self.actual_native_camera_events_preserved,
            collapsed_native_camera_event=(None if self.collapsed_native_camera_event is None
                else self.collapsed_native_camera_event.to_dict()),
            unaffected_native_camera_events_preserved=True,
            single_incident_native_flash_replaced_with_source_switch=self.collapsed_native_camera_event is not None,
            original_source_clock_phase_retained=True,cadence_positions_changed_only_at_target=True,
            source_selection_and_switch_count_unchanged=True,master_audio_unchanged=True,
            descriptors_roles_motion_receipts_require_refresh=True,
            authorized_local_clock_receipts_transferred=False,
            projected_or_encoded_camera_pixel_identity_certified=False,
            identity_phase_lip_sync_or_pose_certified=False,
            normal_app_bundle_created=False,candidate_admitted=False,quality_approved=False)


@dataclass(frozen=True,slots=True)
class NativeGroupCutFlash:
    boundary_index: int
    source_switch: VisibleGroupCut
    native_event: VisibleGroupCut

    @property
    def timing_offset_frames(self):
        return self.native_event.first_incoming_output_frame-self.source_switch.first_incoming_output_frame

    def to_dict(self):
        return dict(boundary_index=self.boundary_index,
            source_switch=self.source_switch.to_dict(),native_event=self.native_event.to_dict(),
            timing_offset_frames=self.timing_offset_frames,
            flash_frame_count=abs(self.timing_offset_frames),
            observation_basis='caller_observed_native_cuts_on_exact_renderer_samples',
            decoded_pixels_reclassified=False,motion_or_quality_certified=False)


def native_flash_timing_frontier(*,plan,domain,maximum_flash_frames=8):
    """Exact adjacent native/source-cut collisions, not every short shot.

    The same output scheduler used for rendering maps the caller's native cut
    observations. Ordinary source-to-source quick cuts are not collapsed. This
    is a candidate domain; selecting a new cut still needs refreshed camera,
    descriptors, motion, look and decoded-output review.
    """
    if type(maximum_flash_frames) is not int or not 1<=maximum_flash_frames<=8:
        raise ValueError('native flash frontier requires a one-to-eight-frame bound')
    if not isinstance(plan,RenderPlan) or plan.output_frame_count>20_000:
        raise ValueError('native flash frontier exceeds the bounded full-event replay inventory')
    counts=segment_frame_counts(plan);cursor=0;boundaries={}
    for index,count in enumerate(counts[:-1]):
        cursor+=count
        if plan.segments[index].source_index!=plan.segments[index+1].source_index:
            boundaries[cursor]=index
    events=visible_group_cut_inventory(plan,domain);result=[]
    for index,event in enumerate(events):
        if event.kind!='source_switch' or event.first_incoming_output_frame not in boundaries:
            continue
        for neighbor_index in (index-1,index+1):
            if not 0<=neighbor_index<len(events):continue
            native=events[neighbor_index]
            delta=native.first_incoming_output_frame-event.first_incoming_output_frame
            expected_source=(event.outgoing_source_index if delta<0 else event.incoming_source_index)
            if (native.kind=='native_camera_cut' and 1<=abs(delta)<=maximum_flash_frames
                    and native.outgoing_source_index==native.incoming_source_index==expected_source):
                result.append(NativeGroupCutFlash(boundaries[event.first_incoming_output_frame],event,native))
    return tuple(result)


def propose_numeric_group_cut_timing(*,plan,domain,boundary_index,timing_offset_frames,
        collapse_incident_native_flash=False):
    """One changed hypothesis, not an unconstrained cut-time search or gate.

    The default cannot drop a broadcaster cut hidden inside the old cell. An
    explicit flash repair may merge exactly one adjacent native event into the
    source switch at its exact tick, never waive other native events or shorten
    a cell below two frames. It cannot manufacture an internal cell seam
    or copy a declared shot-clock authority across changed output intervals.
    Final whole-plan sampling is independently recomputed on both sides.
    """
    if type(collapse_incident_native_flash) is not bool:
        raise TypeError('native flash repair policy must be an explicit boolean')
    if (not isinstance(plan,RenderPlan) or not isinstance(domain,GroupSourceDomain)
            or plan.sources!=domain.request_sources or type(boundary_index) is not int
            or not 0<=boundary_index<len(plan.segments)-1
            or type(timing_offset_frames) is not int or not 1<=abs(timing_offset_frames)<=8):
        raise ValueError('GROUP cut-time preview needs a bound plan/domain and one nonzero eight-frame hypothesis')
    if plan.authorized_shot_video_clocks:
        raise ValueError('changed cut intervals require renewed external shot-clock authority, not transferred receipts')
    if plan.output_frame_count>20_000:
        raise ValueError('GROUP cut-time preview exceeds the bounded full-event replay inventory')
    if any(t.kind.uses_overlap for t in plan.transitions):
        raise ValueError('GROUP cut-time preview cannot retime blended pixels')
    left,right=plan.segments[boundary_index:boundary_index+2]
    if left.source_index==right.source_index:
        raise ValueError('GROUP cut-time preview needs an actual cross-source boundary')
    if any(s.boundary_look_track is not None or s.outgoing_overlap_transform is not None for s in (left,right)):
        raise ValueError('GROUP cut-time preview must precede finalized local look and overlap handles')
    before_counts=segment_frame_counts(plan);old_tick=sum(before_counts[:boundary_index+1])
    units=_video_render_units(plan,before_counts)
    old_units=tuple(next(u for u in units if u.first_segment_index<=index<=u.last_segment_index)
        for index in (boundary_index,boundary_index+1))
    if sum(u.frame_count for u in old_units)>600:
        raise ValueError('GROUP cut-time preview exceeds the two-whole-unit replay budget')
    rates=tuple((u.segments[-1].source_end-u.segments[0].source_start)/sum(s.duration for s in u.segments)
        for u in old_units)
    delta=timing_offset_frames/plan.fps
    proposed_left=replace(left,duration=left.duration+delta,
        source_duration=left.effective_source_duration+delta*rates[0],
        pan_crop=retime_base_pan(left.pan_crop,relative_shift=0.,duration=left.duration+delta),
        editorial_transform=_retime_camera(left.editorial_transform,shift=0.,duration=left.duration+delta))
    proposed_right=replace(right,source_start=right.source_start+delta*rates[1],
        master_start=right.master_start+delta,duration=right.duration-delta,
        source_duration=right.effective_source_duration-delta*rates[1],
        pan_crop=retime_base_pan(right.pan_crop,relative_shift=delta,duration=right.duration-delta),
        editorial_transform=_retime_camera(right.editorial_transform,shift=delta,duration=right.duration-delta))
    transitions=list(plan.transitions)
    # This is deliberately unreviewed. Old endpoint/dense receipts no longer
    # describe the moved cut, even if the numerical source name stays equal.
    transitions[boundary_index]=RenderTransition(tier='C',reason=UNREVIEWED_GROUP_SOURCE_TRANSITION_REASON)
    proposed=replace(plan,segments=(*plan.segments[:boundary_index],proposed_left,proposed_right,
        *plan.segments[boundary_index+2:]),transitions=tuple(transitions))
    counts=segment_frame_counts(proposed);new_tick=sum(counts[:boundary_index+1])
    if (proposed.output_frame_count!=plan.output_frame_count or new_tick-old_tick!=timing_offset_frames
            or any(a!=b for i,(a,b) in enumerate(zip(before_counts,counts))
                if i not in (boundary_index,boundary_index+1))):
        raise ValueError('GROUP cut-time preview changed outside output cadence/counts')
    new_units=_video_render_units(proposed,counts)
    for ordinal,unit in enumerate(old_units):
        new=next((u for u in new_units if u.first_segment_index==unit.first_segment_index
            and u.last_segment_index==unit.last_segment_index),None)
        if new is None:
            raise ValueError('GROUP cut-time preview changed whole renderer-unit topology')
        rate=(new.segments[-1].source_end-new.segments[0].source_start)/sum(s.duration for s in new.segments)
        if abs(rate-rates[ordinal])>1e-10:
            raise ValueError('GROUP cut-time preview changed a whole-unit playback rate')
    before=native_frame_clocks_for_output(plan,domain.clocks_by_request,range(plan.output_frame_count))
    after=native_frame_clocks_for_output(proposed,domain.clocks_by_request,range(proposed.output_frame_count))
    interval=tuple(range(min(old_tick,new_tick),max(old_tick,new_tick)))
    for tick in before:
        a,b=before[tick],after[tick]
        if a.terminal_clone or b.terminal_clone:
            raise ValueError('GROUP cut-time preview cannot retain or introduce terminal clones')
        if tick not in interval and (a.source_index,a.native_frame_index)!=(b.source_index,b.native_frame_index):
            raise ValueError('GROUP cut-time preview changed an outside actual native sample')
    events_before=visible_group_cut_inventory(plan,domain);events_after=visible_group_cut_inventory(proposed,domain)
    collapsed=None
    if collapse_incident_native_flash:
        matches=tuple(row for row in native_flash_timing_frontier(plan=plan,domain=domain)
            if row.boundary_index==boundary_index and row.timing_offset_frames==timing_offset_frames)
        if len(matches)!=1:
            raise ValueError('flash repair requires one exact adjacent native flash anchor')
        collapsed=matches[0].native_event
    else:
        assert_native_cut_events_preserved(plan,proposed,domain)
    def signature(event):
        tick=(new_tick if event.kind=='source_switch' and event.first_incoming_output_frame==old_tick
            else event.first_incoming_output_frame)
        return tick,event.kind,event.outgoing_source_index,event.incoming_source_index,event.native_cut_frame
    if tuple(signature(e) for e in events_before if e!=collapsed)!=tuple((e.first_incoming_output_frame,e.kind,
            e.outgoing_source_index,e.incoming_source_index,e.native_cut_frame) for e in events_after):
        raise ValueError('GROUP cut-time preview changed an unrelated actual cut event')
    return NumericGroupCutTimingProposal(proposed,_plan_hash(plan),_plan_hash(proposed),boundary_index,
        timing_offset_frames,old_tick,new_tick,(boundary_index,boundary_index+1),interval,collapsed is None,collapsed)
