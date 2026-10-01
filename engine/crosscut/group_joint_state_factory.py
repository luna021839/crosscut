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
from .group_joint_source_camera import GroupSourceCameraState,joint_state_from_render_plan
from .group_source_domain import GroupSourceDomain
from .group_cut_camera_proposal import UNREVIEWED_GROUP_SOURCE_TRANSITION_REASON
from .render import (RenderPlan,RenderTransition,aligned_render_segment,segment_frame_counts,
    _combined_identity_segment,_segments_share_source_clock)
from .render_frame_schedule import sampler_frame_schedule
from .shot_context_camera import slice_shot_camera_track


def _cell_interval(plan,first,end):
    counts=segment_frame_counts(plan);ticks=[0]
    for count in counts:ticks.append(ticks[-1]+count)
    if first not in ticks or end not in ticks or not 0<=first<end<=ticks[-1]:
        raise ValueError('joint replacement actual-shot bounds require owned cell splitting')
    return ticks.index(first),ticks.index(end),counts


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
    if (not isinstance(plan,RenderPlan) or not isinstance(domain,GroupSourceDomain)
            or not isinstance(path,tuple) or len(path)!=4 or any(not isinstance(s,GroupSourceCameraState)
                or s.slot!=i for i,s in enumerate(path))
            or any(a.end_output_frame!=b.first_output_frame for a,b in zip(path,path[1:]))):
        raise ValueError('joint assembly needs one complete consistent four-state path')
    segments=list(plan.segments);changed=set()
    for state in path:
        state.bind_source(domain)
        first,end,counts=_cell_interval(plan,state.first_output_frame,state.end_output_frame)
        if state.segment_output_frame_counts!=tuple(counts[first:end]) or len(state.segments)!=end-first:
            raise ValueError('joint state changed canonical cell/frame authority')
        if state.slot in (0,3):
            original=joint_state_from_render_plan(plan=plan,domain=domain,slot=state.slot,
                first_output_frame=state.first_output_frame,end_output_frame=state.end_output_frame,role=state.role,
                editorial_contract_sha256=state.editorial_contract_sha256)
            if original.key!=state.key:raise ValueError('joint replacement changed a fixed outer neighbor')
        else:
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
    if tuple((e.first_incoming_output_frame,e.kind) for e in before)!=tuple((e.first_incoming_output_frame,e.kind) for e in after):
        raise ValueError('joint source/camera replacement changed actual visible cut cadence')
    for state in path:
        rebound=joint_state_from_render_plan(plan=proposed,domain=domain,slot=state.slot,
            first_output_frame=state.first_output_frame,end_output_frame=state.end_output_frame,role=state.role,
            editorial_contract_sha256=state.editorial_contract_sha256)
        if rebound.key!=state.key:
            raise ValueError('assembled joint renderer clock/camera differs from the observed whole state')
    return proposed,tuple(sorted(changed))
