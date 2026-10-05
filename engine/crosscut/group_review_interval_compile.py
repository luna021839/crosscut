"""Exact review interval materialization over the existing renderer sampler.

KEEP slicing preserves each old cell's source phase, camera and sampling
canvas. Constant donor phase changes source_start only, not playback speed.
No original role, native inventory or authority receipt is reinterpreted.
"""
from dataclasses import replace
import math

from .group_interval_materialization import _slice_camera,_slice_look
from .group_cut_camera_proposal import retime_base_pan
from .render import RenderTransition,segment_frame_counts,_video_render_units
from .render_boundary_clock import native_frame_clocks_for_output


def output_offsets(plan):
    values=[0]
    for count in segment_frame_counts(plan):values.append(values[-1]+count)
    return tuple(values)


def cover_review_sampling_interval(plan,first,end):
    """Cover counterfactual sample requests with existing whole render cells.

    This does NOT move an actual edit boundary or hide a native camera cut.
    Analysis queries need legal two-frame cells too; extending a counterfactual
    camera's hold outside its output shot is explicit in its caller's track.
    """
    if any(type(t) is not int for t in (first,end)) or not 0<=first<end<=plan.output_frame_count:
        raise ValueError('counterfactual sample interval lies outside the complete plan')
    offsets=output_offsets(plan)
    return max(t for t in offsets if t<=first),min(t for t in offsets if t>=end)


def shifted_review_source_plan(plan,*,phase_frames,maximum_phase_frames=2,output_interval=None):
    if (type(phase_frames) is not int or type(maximum_phase_frames) is not int
            or not 0<=maximum_phase_frames<=6 or abs(phase_frames)>maximum_phase_frames
            or plan.authorized_shot_video_clocks):
        raise ValueError('review source phase must be one bounded constant without borrowed clock authority')
    if phase_frames==0:return plan
    if output_interval is not None:
        first,end=output_interval
        inside=clip_review_segments(plan,first,end)
        delta=phase_frames/plan.fps
        if any(s.source_start+delta<0 for s in inside):
            raise ValueError('constant review source phase starts outside its source')
        rows=[]
        if first:rows.append((0,first,clip_review_segments(plan,0,first)))
        rows.append((first,end,tuple(replace(s,source_start=s.source_start+delta) for s in inside)))
        if end<plan.output_frame_count:
            rows.append((end,plan.output_frame_count,clip_review_segments(plan,end,plan.output_frame_count)))
        return assemble_review_intervals(plan,rows)
    delta=phase_frames/plan.fps
    if any(s.source_start+delta<0 for s in plan.segments):
        raise ValueError('constant review source phase starts outside its source')
    return replace(plan,segments=tuple(replace(s,source_start=s.source_start+delta) for s in plan.segments))


def clip_review_segments(plan,first,end,*,camera=None):
    """Return exact source-local pieces within the renderer's two-frame bound."""
    if (any(type(t) is not int for t in (first,end)) or not 0<=first<end<=plan.output_frame_count
            or plan.authorized_shot_video_clocks or any(t.kind.uses_overlap for t in plan.transitions)):
        raise ValueError('review clipping needs one exact hard-cut output interval')
    starts=output_offsets(plan);parts=[]
    # The renderer shares an actual clock across adjacent source segments.
    # Segment durations need not equal their quantized frame counts / fps.
    # Resetting a sliced segment from its authored source_start can therefore
    # select a different native frame. Derive the request from the exact
    # original render unit, while geometry still uses its own cell phase.
    clocks={};cursor=0
    for unit in _video_render_units(plan,segment_frame_counts(plan)):
        head,tail=unit.segments[0],unit.segments[-1]
        rate=(unit.coalesced_segment.source_playback_rate if unit.coalesced_segment is not None else
            (tail.source_end-head.source_start)/sum(s.duration for s in unit.segments))
        for j in range(unit.first_segment_index,unit.last_segment_index+1):
            clocks[j]=(cursor,head.source_start,rate)
        cursor+=unit.frame_count
    for i,s in enumerate(plan.segments):
        low,high=max(first,starts[i]),min(end,starts[i+1])
        if low>=high:continue
        if high-low<2:
            raise ValueError('review interval creates an unrealizable one-frame render piece')
        offset=(low-starts[i])/plan.fps
        stop=s.duration if high==starts[i+1] else (high-starts[i])/plan.fps
        duration=stop-offset
        unit_first,unit_source_start,rate=clocks[i]
        source_start=unit_source_start+(low-unit_first)/plan.fps*rate
        if low==starts[i] and high==starts[i+1] and camera is None:
            # Borrowing an ENTIRE authored cell can still detach it from the
            # shared decode clock which previously supplied its first raster.
            # Keep geometry/look objects exact, but materialize that physical
            # clock even when no local geometry slicing was necessary.
            part=(s if abs(s.source_start-source_start)<=1e-12 and abs(s.source_playback_rate-rate)<=1e-12
                else replace(s,source_start=source_start,source_duration=duration*rate))
        else:
            part=replace(s,source_start=source_start,
                master_start=s.master_start+offset,duration=duration,
                source_duration=duration*rate,
                pan_crop=retime_base_pan(s.pan_crop,relative_shift=offset,duration=duration),
                editorial_transform=(_slice_camera(s.editorial_transform,offset_seconds=offset,duration=duration)
                    if camera is None else _slice_camera(camera,
                        offset_seconds=(low-first)/plan.fps,duration=duration)),
                boundary_look_track=_slice_look(s.boundary_look_track,offset_seconds=offset,duration=duration))
        parts.append(part)
    return tuple(parts)


def assemble_review_intervals(original,intervals):
    """Only serialize already realized pieces; never refit a camera here."""
    if not isinstance(intervals,(tuple,list)) or not 1<=len(intervals)<=256:
        raise ValueError('review serialization needs a finite realized interval list')
    segments=[];previous=0
    for first,end,parts in intervals:
        if (first!=previous or not first<end or not isinstance(parts,tuple) or not parts):
            raise ValueError('review intervals omit or overlap output frames')
        segments.extend(parts);previous=end
    if previous!=original.output_frame_count or len(segments)>256:
        raise ValueError('review serialization does not cover its exact bounded whole timeline')
    plan=replace(original,segments=tuple(segments),transitions=tuple(
        RenderTransition(tier='C',reason='unapproved_realized_review_boundary') for _ in segments[1:]))
    if (plan.output_frame_count!=original.output_frame_count
            or abs(plan.duration-original.duration)>1e-8):
        raise ValueError('review serialization changed audio duration or the quantized output count')
    starts=output_offsets(plan)
    j=0
    for first,end,parts in intervals:
        if starts[j]!=first or starts[j+len(parts)]!=end:
            raise ValueError('realized interval changed its compiled output boundary')
        j+=len(parts)
    return plan


def realize_interval_in_plan(base,first,end,*,camera=None):
    rows=[]
    if first:rows.append((0,first,clip_review_segments(base,0,first)))
    rows.append((first,end,clip_review_segments(base,first,end,camera=camera)))
    if end<base.output_frame_count:
        rows.append((end,base.output_frame_count,clip_review_segments(base,end,base.output_frame_count)))
    return assemble_review_intervals(base,rows)


def verify_realized_sampler(base,candidate,domain,ticks,*,require_source_request_time=True):
    """Verify actual rasters, with a separately declared unquantized-time contract.

    Adjacent decoder units can have different fractional request origins while
    selecting exactly the same native rasters. That is not native drift, but is
    still reported and rejected by the default, stricter time contract. A
    raster-only caller must explicitly opt in and compare its entire interval.
    """
    if type(require_source_request_time) is not bool:
        raise ValueError('source request time contract must be explicit')
    ticks=tuple(ticks)
    if not ticks or len(ticks)>36000 or any(type(t) is not int or not 0<=t<base.output_frame_count for t in ticks):
        raise ValueError('realized sampler comparison needs exact bounded output requests')
    a=native_frame_clocks_for_output(base,domain.clocks_by_request,ticks)
    b=native_frame_clocks_for_output(candidate,domain.clocks_by_request,ticks)
    differences=[]
    for tick in ticks:
        left,right=a[tick],b[tick]
        difference=abs(left.source_request_seconds-right.source_request_seconds)
        if (left.source_index!=right.source_index or left.native_frame_index!=right.native_frame_index
                or left.terminal_clone!=right.terminal_clone
                or (require_source_request_time and difference>1e-9)):
            raise ValueError('realized interval serializer changed its selected source/native sampler: '
                f'tick={tick}, source={left.source_index}->{right.source_index}, '
                f'native={left.native_frame_index}->{right.native_frame_index}, '
                f'request={left.source_request_seconds:.12f}->{right.source_request_seconds:.12f}')
        if difference>1e-9:differences.append(difference)
    return dict(compared_output_ticks=len(ticks),selected_sampler_preserved=True,
        actual_native_rasters_preserved=True,
        source_request_time_contract='exact' if require_source_request_time else 'native_raster_identity',
        unquantized_source_request_times_preserved=not differences,
        source_request_time_difference_count=len(differences),
        maximum_source_request_time_difference_seconds=max(differences,default=0.),
        quality_approved=False)
