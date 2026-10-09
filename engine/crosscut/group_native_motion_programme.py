"""Realize selected source/cut/native-phase/cameras on the full GROUP renderer.

Topology is materialized BEFORE the local video clock context is frozen. The
selected cameras then attach without re-fitting. This is numeric construction,
not a motion selector or media-quality certificate; the normal producer must
supply and measure the actual anonymous actor paths and all three incidents.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

from .group_joint_source_camera import GroupOutputRole, joint_state_from_render_plan
from .group_joint_native_phase import NativePhaseHypothesis, NativePhaseSearchPolicy, phase_adjusted_group_state
from .group_review_interval_compile import (assemble_review_intervals, clip_review_segments,
    output_offsets, verify_realized_sampler)
from .render import (RenderPlan, EditorialTransformTrack, aligned_render_segment,
    segment_frame_counts, _video_render_units)
from .group_source_domain import GroupSourceDomain
from .shot_video_clock import (FrozenShotVideoClockContext, attach_authorized_shot_video_clocks,
    authorize_shot_video_clock)


@dataclass(frozen=True, slots=True)
class NativeMotionSide:
    _maximum_output_frames: ClassVar[int] = 160
    first_output_frame: int
    end_output_frame: int
    request_source_index: int
    phase_native_frames: int = 0
    camera: EditorialTransformTrack | None = None
    proposal_role: GroupOutputRole = GroupOutputRole.FOCAL_HEAD

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.first_output_frame, self.end_output_frame,
                self.request_source_index, self.phase_native_frames))
                or not 0 <= self.first_output_frame < self.end_output_frame
                or self.request_source_index < 0
                or not 2 <= self.end_output_frame-self.first_output_frame <= self._maximum_output_frames
                or abs(self.phase_native_frames) > 2
                or self.camera is not None and not isinstance(self.camera, EditorialTransformTrack)
                or not isinstance(self.proposal_role, GroupOutputRole)):
            raise ValueError("native motion side needs an exact bounded source/phase/camera proposal")


@dataclass(frozen=True, slots=True)
class ProgrammeNativeMotionSide(NativeMotionSide):
    """A whole-programme proposal, not permission to enlarge local repair."""
    _maximum_output_frames: ClassVar[int] = 36000


@dataclass(frozen=True, slots=True)
class RealizedNativeMotionProgramme:
    aligned_plan: RenderPlan
    render_plan: RenderPlan
    shot_clock_context: FrozenShotVideoClockContext
    sides: tuple[NativeMotionSide, ...]
    preservation: dict


def _aligned_parts(original, domain, side):
    local = domain.eligible_index(side.request_source_index)
    # A moved cut must not inherit a one-frame fragment at an OLD cut. New
    # source/camera shots own their canonical intervals, not the obsolete cells.
    # Immutable neighbours are still sliced with the exact original sampler.
    from bisect import bisect_right
    offsets = output_offsets(original)
    def reference_at(tick):
        if tick == original.output_frame_count:
            part=original.segments[-1]
            return part.master_start+part.duration
        index=bisect_right(offsets,tick)-1
        return original.segments[index].master_start+(tick-offsets[index])/original.fps
    first,end=reference_at(side.first_output_frame),reference_at(side.end_output_frame)
    return (aligned_render_segment(source_index=side.request_source_index,
        reference_start=original.master_audio_start+first,
        common_master_start=original.master_audio_start, source_offset=domain.audio_offsets[local],
        duration=end-first, timewarp=domain.source_timewarps[local]),)


def _restore_untouched_transitions(original, plan, first, end):
    old_starts = output_offsets(original)
    old = {old_starts[i+1]: value for i, value in enumerate(original.transitions)}
    new_starts = output_offsets(plan)
    transitions = tuple(old.get(tick, transition) if tick < first or tick > end else transition
        for tick, transition in zip(new_starts[1:-1], plan.transitions, strict=True))
    return replace(plan, transitions=transitions)


def _restore_untouched_programme_transitions(original, plan, spans):
    old_starts = output_offsets(original)
    old = {old_starts[i+1]: value for i, value in enumerate(original.transitions)}
    new_starts = output_offsets(plan)
    transitions = tuple(old.get(tick, transition)
        if not any(first <= tick <= end for first,end in spans) else transition
        for tick,transition in zip(new_starts[1:-1],plan.transitions,strict=True))
    return replace(plan,transitions=transitions)


def _immutable_parts(original, first, end):
    """Keep the ORIGINAL shared decode unit, not independently reset cells.

    Fractional authored durations differ from quantized cell frame counts. A
    source_start derived independently at every old cell breaks its existing
    Tier-D continuity. Only the first surviving piece of an original decode
    unit acquires the exact old output-sampler origin; following pieces share
    that one clock at its original rate. Geometry still uses each old cell's
    actual sliced camera phase. Final raster replay verifies the full exterior.
    """
    parts=list(clip_review_segments(original,first,end))
    starts=output_offsets(original)
    cells=[i for i in range(len(original.segments)) if first<starts[i+1] and end>starts[i]]
    by_cell=dict(zip(cells,range(len(parts)),strict=True))
    unit_first=0
    for unit in _video_render_units(original,segment_frame_counts(original)):
        kept=[i for i in range(unit.first_segment_index,unit.last_segment_index+1) if i in by_cell]
        if kept:
            head,tail=unit.segments[0],unit.segments[-1]
            rate=(unit.coalesced_segment.source_playback_rate if unit.coalesced_segment is not None else
                (tail.source_end-head.source_start)/sum(s.duration for s in unit.segments))
            low=max(first,starts[kept[0]])
            source_start=head.source_start+(low-unit_first)/original.fps*rate
            for cell in kept:
                index=by_cell[cell];part=parts[index]
                parts[index]=replace(part,source_start=source_start,source_duration=part.duration*rate)
                source_start+=part.duration*rate
        unit_first+=unit.frame_count
    return tuple(parts)


def realize_native_motion_programme(*, original, domain, sides,
        protected_output_intervals=(), phase_policy=None, selection_scope='adjacent_two_shots'):
    """Real selected shots, moved cuts, exact native phases and authored cameras.

    Existing original-role intent is deliberately not redefined here. The
    proposal_role is a computational framing hypothesis, not an inferred or
    weakened original anatomical duty. Common-clock queries retain the separate
    externally frozen original-role programme throughout sensing and selection.
    """
    variable = selection_scope=='variable_programme'
    if (selection_scope not in ('adjacent_two_shots','variable_programme')
            or not isinstance(original, RenderPlan) or not isinstance(domain, GroupSourceDomain)
            or not 1 <= original.output_frame_count <= 36000
            or not isinstance(sides, tuple)
            or (not 1<=len(sides)<=1024 if variable else len(sides)!=2)
            or any(type(side) not in (NativeMotionSide,ProgrammeNativeMotionSide) for side in sides)
            or any(side.end_output_frame-side.first_output_frame>(36000 if variable else 160) for side in sides)
            or any(a.end_output_frame>b.first_output_frame if variable
                else a.end_output_frame!=b.first_output_frame for a,b in zip(sides,sides[1:]))
            or sides[-1].end_output_frame > original.output_frame_count
            or original.authorized_shot_video_clocks
            or any(t.kind.uses_overlap for t in original.transitions)
            or original.sources != domain.request_sources):
        raise ValueError("native motion programme requires declared ordered full renderer intervals")
    first, end = sides[0].first_output_frame, sides[-1].end_output_frame
    if (not isinstance(protected_output_intervals, tuple)
            or any(not isinstance(span, tuple) or len(span) != 2
                or any(type(t) is not int for t in span)
                or not 0 <= span[0] < span[1] <= original.output_frame_count
                for span in protected_output_intervals)
            or any(side.first_output_frame < b and side.end_output_frame > a
                for side in sides for a,b in protected_output_intervals)):
        raise ValueError("native motion programme intersects protected original pixels")
    policy = NativePhaseSearchPolicy() if phase_policy is None else phase_policy
    if not isinstance(policy, NativePhaseSearchPolicy):
        raise ValueError("native phase search needs its externally declared policy")
    intervals = []
    cursor=0
    spans=tuple((s.first_output_frame,s.end_output_frame) for s in sides)
    for side in sides:
        if cursor<side.first_output_frame:
            intervals.append((cursor,side.first_output_frame,
                _immutable_parts(original,cursor,side.first_output_frame)))
        intervals.append((side.first_output_frame,side.end_output_frame,_aligned_parts(original,domain,side)))
        cursor=side.end_output_frame
    if cursor<original.output_frame_count:
        intervals.append((cursor,original.output_frame_count,_immutable_parts(original,cursor,original.output_frame_count)))
    aligned=assemble_review_intervals(original,intervals,maximum_segments=512 if variable else 256)
    aligned=(_restore_untouched_programme_transitions(original,aligned,spans) if variable
        else _restore_untouched_transitions(original,aligned,first,end))
    # A different cut has a different canonical scaffold. Freezing the OLD
    # incumbent and attaching its clocks would be a false authority binding.
    context = FrozenShotVideoClockContext(aligned, domain, policy,original,
        spans,protected_output_intervals,selection_scope=selection_scope)
    offsets = output_offsets(aligned)
    segments = list(aligned.segments); clocks = []
    selected_grids = []
    for slot, side in enumerate(sides, 1):
        state = joint_state_from_render_plan(plan=aligned, domain=domain, slot=1 if variable else slot,
            first_output_frame=side.first_output_frame, end_output_frame=side.end_output_frame,
            role=side.proposal_role)
        if side.phase_native_frames:
            hypothesis = NativePhaseHypothesis(state.key, policy.sha256,
                side.phase_native_frames, side.phase_native_frames)
            phased = phase_adjusted_group_state(state=state, domain=domain, hypothesis=hypothesis, policy=policy)
            clocks.append(authorize_shot_video_clock(context=context, aligned_state=state,
                phase_state=phased, hypothesis=hypothesis))
        else:
            phased = state
        selected_grids.append(phased.native_frame_indices)
        low, high = offsets.index(side.first_output_frame), offsets.index(side.end_output_frame)
        if len(phased.segments) != high-low:
            raise ValueError("native motion state changed its aligned cell inventory")
        for index, segment in zip(range(low, high), phased.segments, strict=True):
            if side.camera is not None:
                from .group_interval_materialization import _slice_camera
                segment = replace(segment, editorial_transform=_slice_camera(side.camera,
                    offset_seconds=(offsets[index]-side.first_output_frame)/original.fps,
                    duration=segment.duration))
            segments[index] = segment
    selected = attach_authorized_shot_video_clocks(plan=replace(aligned, segments=tuple(segments)),
        context=context, clocks=tuple(clocks))
    from .render_boundary_clock import native_frame_clocks_for_output
    from bisect import bisect_right
    runs = []
    for side, expected in zip(sides, selected_grids, strict=True):
        samples = native_frame_clocks_for_output(selected, domain.clocks_by_request,
            range(side.first_output_frame, side.end_output_frame))
        values = tuple(samples.values())
        if (tuple(sample.native_frame_index for sample in values) != expected
                or any(sample.source_index != side.request_source_index or sample.terminal_clone for sample in values)):
            raise ValueError("authored camera changed the exact selected native schedule")
        clock = domain.clocks_by_request[side.request_source_index]
        import math
        cuts = tuple(math.ceil(t*float(clock.frame_rate)-1e-9) for t in domain.cuts_for_request(side.request_source_index))
        ids = {bisect_right(cuts, sample.native_frame_index) for sample in values}
        if len(ids) != 1:
            raise ValueError("one motion side crosses an actual native broadcaster cut")
        run = next(iter(ids))
        runs.append((0 if run == 0 else cuts[run-1], clock.frame_count if run == len(cuts) else cuts[run]))
    outside = tuple(t for t in range(original.output_frame_count)
        if not any(a<=t<b for a,b in spans))
    preservation = (verify_realized_sampler(original, selected, domain, outside,
        require_source_request_time=False) if outside else dict(compared_output_ticks=0,
            actual_native_rasters_preserved=True, quality_approved=False))
    preservation = dict(preservation, physical_native_runs=[list(r) for r in runs],
        selection_scope=selection_scope,selected_shot_count=len(sides),
        topology_frozen_before_native_phase=True, constant_native_phase_not_output_frame_phase=True,
        camera_refit_after_selection=False, original_role_duties_redefined=False, quality_approved=False)
    return RealizedNativeMotionProgramme(aligned, selected, context, sides, preservation)
