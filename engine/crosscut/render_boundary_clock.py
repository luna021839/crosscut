"""Locate the actual native frames meeting at hard source-change cuts.

This consumes the renderer's own quantized units and native-frame scheduler.
It neither substitutes half-frame inspection times for output ticks nor grants
pose/crop/quality approval. Overlap transitions and unverified source clocks
are explicitly unsupported rather than silently described as hard-cut pixels.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from .render import RenderPlan, _video_render_units, segment_frame_counts
from .render_frame_schedule import VerifiedRenderFrameClock, sampler_frame_schedule


@dataclass(frozen=True, slots=True)
class RenderedBoundaryNativeFrame:
    output_frame_index: int
    source_index: int
    native_frame_index: int
    source_request_seconds: float
    native_presentation_seconds: float
    unit_output_tick: int
    unit_first_segment_index: int
    terminal_clone: bool


@dataclass(frozen=True, slots=True)
class RenderedHardCutClock:
    outgoing_segment_index: int
    incoming_segment_index: int
    first_incoming_output_frame: int
    outgoing: RenderedBoundaryNativeFrame
    incoming: RenderedBoundaryNativeFrame


def hard_source_cut_native_clocks(
    plan: RenderPlan,
    source_clocks: Mapping[int, VerifiedRenderFrameClock],
) -> tuple[RenderedHardCutClock, ...]:
    """Return exact output-neighbor native IDs, including same-source coalescing.

    Caller-supplied clocks must have already been verified against the current
    files. A clock result is numeric scheduling evidence, not evidence that a
    subsequent read of a replaced/disconnected file would return those pixels.
    """
    if not isinstance(plan, RenderPlan):
        raise TypeError('hard-cut clock inspection requires a RenderPlan')
    if any(transition.kind.uses_overlap for transition in plan.transitions):
        raise ValueError('overlap pixels require a composited boundary clock')
    counts = segment_frame_counts(plan)
    units = _video_render_units(plan, counts)
    schedules = []
    starts = []
    cursor = 0
    for unit in units:
        starts.append(cursor)
        cursor += unit.frame_count
        first, last = unit.segments[0], unit.segments[-1]
        clock = source_clocks.get(first.source_index)
        if not isinstance(clock, VerifiedRenderFrameClock):
            raise ValueError(f'source {first.source_index} has no verified native clock')
        # Preserve the renderer's arithmetic, including its single-segment
        # path: subtracting an added endpoint can change half-frame rounding.
        rate = (unit.coalesced_segment.source_playback_rate
                if unit.coalesced_segment is not None else
                (last.source_end - first.source_start) / sum(segment.duration for segment in unit.segments))
        schedule = sampler_frame_schedule(source_start=first.source_start,
            source_playback_rate=rate, native_fps=float(clock.frame_rate),
            output_fps=plan.fps, frame_count=unit.frame_count,
            source_frame_count=clock.frame_count,
            maximum_terminal_hold_frames=math.ceil(plan.fps))
        schedules.append((schedule, rate, clock))

    def sample(unit_index, tick):
        unit = units[unit_index]
        schedule, rate, clock = schedules[unit_index]
        first = unit.segments[0]
        native_index = schedule.source_frame_indices[tick]
        return RenderedBoundaryNativeFrame(
            output_frame_index=starts[unit_index] + tick,
            source_index=first.source_index,
            native_frame_index=native_index,
            source_request_seconds=first.source_start + tick / plan.fps * rate,
            native_presentation_seconds=native_index / float(clock.frame_rate),
            unit_output_tick=tick,
            unit_first_segment_index=unit.first_segment_index,
            terminal_clone=tick in schedule.terminal_hold_output_indices,
        )

    results = []
    for index, (left, right) in enumerate(zip(units, units[1:])):
        if left.segments[-1].source_index == right.segments[0].source_index:
            continue
        results.append(RenderedHardCutClock(
            outgoing_segment_index=left.last_segment_index,
            incoming_segment_index=right.first_segment_index,
            first_incoming_output_frame=starts[index + 1],
            outgoing=sample(index, left.frame_count - 1),
            incoming=sample(index + 1, 0),
        ))
    return tuple(results)


def native_frame_clocks_for_output(
    plan: RenderPlan,
    source_clocks: Mapping[int, VerifiedRenderFrameClock],
    output_frame_indices,
    *, allow_pure_frames_in_overlap_plan: bool = False,
) -> dict[int, RenderedBoundaryNativeFrame]:
    """Locate requested pure hard-cut output frames on the real sampler clock.

    This also covers interior frames needed by source-fidelity QA. No legacy
    ``ceil(source_start) + round(relative_time)`` fallback is silently used.
    Blended plans require a separate composited-frame contract.
    """
    if not isinstance(plan, RenderPlan):
        raise TypeError('native frame inspection requires a RenderPlan')
    if (not allow_pure_frames_in_overlap_plan
            and any(transition.kind.uses_overlap for transition in plan.transitions)):
        raise ValueError('overlap pixels require a composited boundary clock')
    indices = tuple(output_frame_indices)
    if any(type(i) is not int or not 0 <= i < plan.output_frame_count for i in indices):
        raise ValueError('requested output frame is outside the render plan')
    wanted = set(indices)
    counts = segment_frame_counts(plan)
    boundary = 0
    for index, transition in enumerate(plan.transitions):
        boundary += counts[index]
        if transition.kind.uses_overlap and any(boundary <= i < boundary + transition.frame_count for i in wanted):
            raise ValueError('requested overlap pixels have no single native reference')
    cursor = 0
    result = {}
    for unit in _video_render_units(plan, counts):
        first, last = unit.segments[0], unit.segments[-1]
        selected = sorted(i for i in wanted if cursor <= i < cursor + unit.frame_count)
        if selected:
            clock = source_clocks.get(first.source_index)
            if not isinstance(clock, VerifiedRenderFrameClock):
                raise ValueError(f'source {first.source_index} has no verified native clock')
            rate = (unit.coalesced_segment.source_playback_rate
                    if unit.coalesced_segment is not None else
                    (last.source_end - first.source_start) / sum(s.duration for s in unit.segments))
            schedule = sampler_frame_schedule(source_start=first.source_start,
                source_playback_rate=rate, native_fps=float(clock.frame_rate),
                output_fps=plan.fps, frame_count=unit.frame_count,
                source_frame_count=clock.frame_count,
                maximum_terminal_hold_frames=math.ceil(plan.fps))
            held = set(schedule.terminal_hold_output_indices)
            for index in selected:
                tick = index - cursor
                native = schedule.source_frame_indices[tick]
                result[index] = RenderedBoundaryNativeFrame(index, first.source_index,
                    native, first.source_start + tick / plan.fps * rate,
                    native / float(clock.frame_rate), tick, unit.first_segment_index,
                    tick in held)
        cursor += unit.frame_count
    if set(result) != wanted:
        raise RuntimeError('native sample coverage differs from requested output frames')
    return result


def hard_cut_boundary_sample_clocks(
    plan: RenderPlan, source_clocks: Mapping[int, VerifiedRenderFrameClock], *,
    boundary_index: int, flank_frames: int = 3,
) -> tuple[RenderedBoundaryNativeFrame, ...]:
    """Four actually visible audition frames, never synthetic half-frame times.

    Outgoing-before/last and incoming-first/after use the existing renderer
    scheduling authority. A short run cannot borrow frames from another source,
    and a terminal clone is not newly observed continuity evidence.
    """
    if not isinstance(plan, RenderPlan):
        raise TypeError('hard-cut audition requires a RenderPlan')
    if (isinstance(boundary_index, bool) or not isinstance(boundary_index, int)
            or not 0 <= boundary_index < len(plan.segments)-1):
        raise ValueError('audition boundary is outside the render plan')
    if isinstance(flank_frames, bool) or not isinstance(flank_frames, int) or flank_frames < 1:
        raise ValueError('audition flank frames must be a positive integer')
    left, right = plan.segments[boundary_index:boundary_index+2]
    if left.source_index == right.source_index:
        raise ValueError('audition requires a cross-source hard cut')
    counts = segment_frame_counts(plan)
    cut_frame = sum(counts[:boundary_index+1])
    requested = (cut_frame-1-flank_frames, cut_frame-1, cut_frame, cut_frame+flank_frames)
    samples = native_frame_clocks_for_output(plan, source_clocks, requested)
    result = tuple(samples[index] for index in requested)
    if any(sample.terminal_clone for sample in result):
        raise ValueError('terminal clone is not audition evidence')
    if (any(sample.source_index != left.source_index for sample in result[:2])
            or any(sample.source_index != right.source_index for sample in result[2:])):
        raise ValueError('visible source run is shorter than audition flank')
    return result
