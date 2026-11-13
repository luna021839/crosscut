"""Numeric full-source/camera states and exact two-shot assembly.

Unlike the old solo replacement constructor, a proposed B may temporarily
use C's old source because C can change in the SAME transaction. Only the
assembled A-B-C-D graph owns the final renderer clock. No raw-family gate,
prior single-shot success, geometry/motion approval or media work occurs here.
"""
from dataclasses import replace
import hashlib
import json

from .group_actual_shots import assert_native_cut_events_preserved,visible_group_cut_inventory
from .group_joint_source_camera import (GroupSourceCameraState,FixedGroupNeighborObservationState,
    joint_state_from_render_plan)
from .group_source_domain import GroupSourceDomain
from .group_cut_camera_proposal import UNREVIEWED_GROUP_SOURCE_TRANSITION_REASON
from .render import (RenderPlan,RenderTransition,aligned_render_segment,segment_frame_counts,
    _combined_identity_segment,_segments_share_source_clock,_can_share_source_clock)
from .render_boundary_clock import native_frame_clocks_for_output
from .render_frame_schedule import sampler_frame_schedule
from .shot_context_camera import slice_shot_camera_track


def _cell_interval(plan,first,end):
    counts=segment_frame_counts(plan);ticks=[0]
    for count in counts:ticks.append(ticks[-1]+count)
    if first not in ticks or end not in ticks or not 0<=first<end<=ticks[-1]:
        raise ValueError('joint replacement actual-shot bounds require owned cell splitting')
    return ticks.index(first),ticks.index(end),counts


def fixed_neighbor_state_from_render_plan(*,plan,domain,slot,first_output_frame,
        end_output_frame,role,raw_shot_family=None,editorial_contract_sha256=None):
    """Read an unchanged actual neighbor without publishing a split timeline.

    A broadcaster cut can end the neighbor inside a canonical cell while the
    mutable middle shot is already aligned. Read-only fragment views keep
    every original native sample, crop, camera/canvas and look phase unchanged
    without splitting the selected RenderPlan. Even a one-tick tail is merely
    an observation fragment, not a newly rendered cell. The assembler still
    reconstructs it from the original plan; mutable states retain cell ownership.
    This is not a complete mutable-epoch metadata/receipt owner.
    """
    if (not isinstance(plan,RenderPlan) or not isinstance(domain,GroupSourceDomain)
            or plan.sources!=domain.request_sources or any(t.kind.uses_overlap for t in plan.transitions)
            or type(slot) is not int or slot not in (0,2,3)):
        raise ValueError('partial-cell neighbor view is only for a fixed outer state')
    counts=segment_frame_counts(plan);ticks=[0]
    for count in counts:ticks.append(ticks[-1]+count)
    if (type(first_output_frame) is not int or type(end_output_frame) is not int
            or not 0<=first_output_frame<end_output_frame<=ticks[-1]):
        raise ValueError('fixed neighbor interval is outside the original renderer')
    if first_output_frame in ticks and end_output_frame in ticks:
        return joint_state_from_render_plan(plan=plan,domain=domain,slot=slot,
            first_output_frame=first_output_frame,end_output_frame=end_output_frame,
            role=role,raw_shot_family=raw_shot_family,
            editorial_contract_sha256=editorial_contract_sha256)
    if plan.authorized_shot_video_clocks or any(s.outgoing_overlap_transform is not None for s in plan.segments):
        raise ValueError('read-only fragments need supported original hard-cut clock/camera authority')
    from .group_interval_materialization import _slice_camera,_slice_look
    from .group_cut_camera_proposal import retime_base_pan
    indices=tuple(i for i in range(len(counts)) if ticks[i]<end_output_frame and ticks[i+1]>first_output_frame)
    if not _can_share_source_clock(plan,first_index=indices[0],candidate_last_index=indices[-1]):
        raise ValueError('fixed fragment view is not one actual source clock')
    fragments=[];fragment_counts=[]
    for index in indices:
        segment=plan.segments[index];a=max(first_output_frame,ticks[index]);b=min(end_output_frame,ticks[index+1])
        local=a-ticks[index];offset=local/plan.fps
        stop=segment.duration if b==ticks[index+1] else (b-ticks[index])/plan.fps
        duration=stop-offset
        fragments.append(segment if a==ticks[index] and b==ticks[index+1] else replace(segment,
            source_start=segment.source_start+offset*segment.source_playback_rate,
            master_start=segment.master_start+offset,duration=duration,
            source_duration=duration*segment.source_playback_rate if segment.source_duration is not None else None,
            pan_crop=retime_base_pan(segment.pan_crop,relative_shift=offset,duration=duration),
            editorial_transform=_slice_camera(segment.editorial_transform,offset_seconds=offset,duration=duration),
            boundary_look_track=_slice_look(segment.boundary_look_track,offset_seconds=offset,duration=duration)))
        fragment_counts.append(b-a)
    samples=native_frame_clocks_for_output(plan,domain.clocks_by_request,range(first_output_frame,end_output_frame))
    source=fragments[0].source_index;eligible=domain.eligible_index(source)
    if any(row.source_index!=source or row.terminal_clone for row in samples.values()):
        raise ValueError('fixed fragment view cannot borrow other-source or terminal-cloned pixels')
    identity=hashlib.sha256(json.dumps(domain.render_frame_clocks[eligible].to_dict(),
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return FixedGroupNeighborObservationState(slot,first_output_frame,end_output_frame,eligible,
        tuple(fragments),tuple(row.native_frame_index for row in samples.values()),role,identity,
        plan.fps,(plan.width,plan.height),tuple(fragment_counts),raw_shot_family,
        editorial_contract_sha256).bind_source(domain)


def candidate_joint_group_state(*,plan,domain,incumbent_state,eligible_source_index,camera_track=None):
    """Construct an entire alternative native source unit, not a solo edit.

    Caller declares the role on incumbent_state before camera search. The
    complete camera is sliced using GLOBAL cell frame counts. Source timing
    comes only from the matching eligible-local offset/timewarp. A later joint
    assembly MUST reproduce every exact native frame before it can be sensed
    as a production proposal. This numeric state itself grants no approval.
    """
    if (not isinstance(plan,RenderPlan) or not isinstance(domain,GroupSourceDomain)
            or not isinstance(incumbent_state,GroupSourceCameraState)
            or isinstance(incumbent_state,FixedGroupNeighborObservationState)
            or incumbent_state.slot not in (1,2) or plan.sources!=domain.request_sources):
        raise ValueError('joint state factory needs complete owned plan/domain and a mutable shot')
    first,end,counts=_cell_interval(plan,incumbent_state.first_output_frame,incumbent_state.end_output_frame)
    old=joint_state_from_render_plan(plan=plan,domain=domain,slot=incumbent_state.slot,
        first_output_frame=incumbent_state.first_output_frame,end_output_frame=incumbent_state.end_output_frame,
        role=incumbent_state.role,editorial_contract_sha256=incumbent_state.editorial_contract_sha256)
    if old.key!=incumbent_state.key:raise ValueError('joint factory incumbent is not bound to its complete renderer state')
    request_source=domain.request_index(eligible_source_index)
    if sum(counts[first:end])>300:
        raise ValueError('joint source factory exceeds its mutable whole-shot frame budget')
    selected=[];cursor=0
    for index in range(first,end):
        original=plan.segments[index]
        segment=aligned_render_segment(source_index=request_source,
            reference_start=plan.master_audio_start+original.master_start,
            common_master_start=plan.master_audio_start,source_offset=domain.audio_offsets[eligible_source_index],
            duration=original.duration,timewarp=domain.source_timewarps[eligible_source_index])
        if camera_track is not None:
            segment=replace(segment,editorial_transform=slice_shot_camera_track(camera_track,
                frame_offset=cursor,frame_count=counts[index],fps=plan.fps))
        cursor+=counts[index];selected.append(segment)
    # This is a numeric hypothesis, not a solo full-plan replacement. The old
    # neighboring tier-D/source receipts can be stale until BOTH mutable states
    # are installed by assemble_joint_group_states.
    if not _segments_share_source_clock(tuple(selected),plan.transitions[first:end-1],fps=plan.fps):
        raise ValueError('joint candidate source is not one complete linear renderer clock')
    # Exactly the renderer's identity coalescing arithmetic, including the
    # single-segment subtraction/half-native-frame rounding distinction.
    unit=selected[0] if len(selected)==1 else _combined_identity_segment(tuple(selected))
    clock=domain.render_frame_clocks[eligible_source_index]
    schedule=sampler_frame_schedule(source_start=unit.source_start,source_playback_rate=unit.source_playback_rate,
        native_fps=float(clock.frame_rate),output_fps=plan.fps,frame_count=cursor,
        source_frame_count=clock.frame_count,maximum_terminal_hold_frames=0)
    if schedule.terminal_hold_output_indices:raise ValueError('joint source state cannot introduce terminal clones')
    identity=hashlib.sha256(json.dumps(clock.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return GroupSourceCameraState(incumbent_state.slot,incumbent_state.first_output_frame,
        incumbent_state.end_output_frame,eligible_source_index,tuple(selected),tuple(schedule.source_frame_indices),
        incumbent_state.role,identity,plan.fps,(plan.width,plan.height),tuple(counts[first:end]),
        editorial_contract_sha256=incumbent_state.editorial_contract_sha256).bind_source(domain)


def assemble_joint_group_states(*,plan,domain,path):
    """Atomic numeric assembly; stale keys/clocks/cut counts reject as a whole.

    Only B/C cells are mutable; A/D and every outside pixel/clock are fixed.
    Full renderer sampling is recalculated after BOTH replacements. Adjacent
    source aliasing cannot erase a cut or change an outer sample. Descriptor,
    dense-subject, motion, review and binding refresh still belong to the owner.
    """
    return _assemble_group_state_path(plan=plan,domain=domain,path=path,state_count=4)


def assemble_single_group_source_camera_state(*,plan,domain,path,
        replace_incident_clock_jumps=False):
    """Fixed L -> one complete mutable X -> fixed R, not a fake B/C seam.

    X may span multiple canonical cells but owns one entire native/camera
    trajectory. This sibling numeric factory grants no semantic admission,
    and does not silently change the existing four-state editorial contract.
    """
    if type(replace_incident_clock_jumps) is not bool:
        raise ValueError('incident clock-jump replacement policy must be explicit')
    return _assemble_group_state_path(plan=plan,domain=domain,path=path,state_count=3,
        replace_incident_clock_jumps=replace_incident_clock_jumps)


def _assemble_group_state_path(*,plan,domain,path,state_count,replace_incident_clock_jumps=False):
    if (not isinstance(plan,RenderPlan) or not isinstance(domain,GroupSourceDomain)
            or not isinstance(path,tuple) or len(path)!=state_count or any(not isinstance(s,GroupSourceCameraState)
                or s.slot!=i for i,s in enumerate(path))
            or any(a.end_output_frame!=b.first_output_frame for a,b in zip(path,path[1:]))):
        raise ValueError('joint assembly needs one complete consistent declared state path')
    segments=list(plan.segments);changed=set()
    for state in path:
        state.bind_source(domain)
        if state.slot in (0,state_count-1):
            original=fixed_neighbor_state_from_render_plan(plan=plan,domain=domain,slot=state.slot,
                first_output_frame=state.first_output_frame,end_output_frame=state.end_output_frame,role=state.role,
                editorial_contract_sha256=state.editorial_contract_sha256)
            if original.key!=state.key:raise ValueError('joint replacement changed a fixed outer neighbor')
        else:
            if isinstance(state,FixedGroupNeighborObservationState):
                raise ValueError('read-only fixed fragments cannot become mutable renderer cells')
            first,end,counts=_cell_interval(plan,state.first_output_frame,state.end_output_frame)
            if state.segment_output_frame_counts!=tuple(counts[first:end]) or len(state.segments)!=end-first:
                raise ValueError('joint state changed canonical cell/frame authority')
            for index,segment in zip(range(first,end),state.segments,strict=True):
                if (segment.master_start,segment.duration)!=(plan.segments[index].master_start,plan.segments[index].duration):
                    raise ValueError('joint state changed fixed source/output interval bounds')
                if segment!=plan.segments[index]:changed.add(index)
                segments[index]=segment
    transitions=tuple(RenderTransition(tier='C',reason=UNREVIEWED_GROUP_SOURCE_TRANSITION_REASON)
        if i in changed or i+1 in changed else transition for i,transition in enumerate(plan.transitions))
    proposed=replace(plan,segments=tuple(segments),transitions=transitions) if changed else plan
    if segment_frame_counts(proposed)!=segment_frame_counts(plan):
        raise ValueError('joint replacement changed global output frame counts')
    # Retain native camera events and source-switch cadence, not cell counts.
    assert_native_cut_events_preserved(plan,proposed,domain)
    before,after=visible_group_cut_inventory(plan,domain),visible_group_cut_inventory(proposed,domain)
    incident = {path[1].first_output_frame,path[-2].end_output_frame}
    same_cadence = len(before)==len(after) and all(
        a.first_incoming_output_frame==b.first_incoming_output_frame and (
            a.kind==b.kind or replace_incident_clock_jumps
            and a.first_incoming_output_frame in incident
            and a.kind=='same_source_clock_jump' and b.kind=='source_switch')
        for a,b in zip(before,after))
    if not same_cadence:
        raise ValueError('joint source/camera replacement changed actual visible cut cadence')
    for state in path:
        builder=(fixed_neighbor_state_from_render_plan if state.slot in (0,state_count-1)
            else joint_state_from_render_plan)
        rebound=builder(plan=proposed,domain=domain,slot=state.slot,
            first_output_frame=state.first_output_frame,end_output_frame=state.end_output_frame,role=state.role,
            editorial_contract_sha256=state.editorial_contract_sha256)
        if rebound.key!=state.key:
            raise ValueError('assembled joint renderer clock/camera differs from the observed whole state')
    return proposed,tuple(sorted(changed))
