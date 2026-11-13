"""Numeric inventory of visible GROUP cuts on the renderer's native clock.

Canonical cells and source names are not shots. One continuous source can
contain broadcaster camera cuts; splitting a cell cannot manufacture one.
Input cut times are caller-observed data, not detector or quality approval.
No media is decoded, and this inventory does not establish pose continuity.
"""
from bisect import bisect_right
from dataclasses import dataclass
import math

from .group_source_domain import GroupSourceDomain
from .render import RenderPlan, _can_share_source_clock, segment_frame_counts
from .render_boundary_clock import native_frame_clocks_for_output


@dataclass(frozen=True, slots=True)
class VisibleGroupCut:
    first_incoming_output_frame: int
    kind: str
    outgoing_source_index: int
    incoming_source_index: int
    outgoing_native_frame: int
    incoming_native_frame: int
    native_cut_frame: int | None = None

    def to_dict(self):
        return dict(first_incoming_output_frame=self.first_incoming_output_frame,
            kind=self.kind, outgoing_source_index=self.outgoing_source_index,
            incoming_source_index=self.incoming_source_index,
            outgoing_native_frame=self.outgoing_native_frame,
            incoming_native_frame=self.incoming_native_frame,
            native_cut_frame=self.native_cut_frame)


@dataclass(frozen=True, slots=True)
class VisibleGroupShot:
    first_output_frame: int
    end_output_frame: int
    source_index: int
    first_segment_index: int
    last_segment_index: int

    @property
    def frame_count(self):
        return self.end_output_frame-self.first_output_frame

    def to_dict(self):
        return dict(first_output_frame=self.first_output_frame,
            end_output_frame=self.end_output_frame, frame_count=self.frame_count,
            source_index=self.source_index, first_segment_index=self.first_segment_index,
            last_segment_index=self.last_segment_index)


def visible_group_cut_inventory(plan, source_domain):
    """Map source switches/native cuts to actual adjacent output samples.

    A native cut is the first scheduled incoming sample crossing its observed
    cut frame. It may fall inside a cell, on a cell seam or between skipped
    source frames. Cuts before the first visible sample are not visible events.
    Same-source seeks are separate events, never inferred camera cuts through
    an interval that was not rendered. Multiple native cuts skipped in one
    output tick remain distinct records, not a claimed extra visible frame.
    """
    if (not isinstance(plan, RenderPlan) or not isinstance(source_domain, GroupSourceDomain)
            or tuple(plan.sources)!=source_domain.request_sources
            or any(s.source_index not in source_domain.eligible_indices for s in plan.segments)):
        raise ValueError('visible GROUP cuts need the complete bound eligible source domain')
    if any(t.kind.uses_overlap for t in plan.transitions):
        raise ValueError('visible GROUP hard-cut inventory cannot certify blended pixels')
    counts=segment_frame_counts(plan);starts=[0]
    for count in counts:starts.append(starts[-1]+count)
    samples=native_frame_clocks_for_output(plan,source_domain.clocks_by_request,range(starts[-1]))
    if any(row.terminal_clone for row in samples.values()):
        raise ValueError('terminal-cloned output cannot be actual-shot evidence')
    cuts={source:tuple(sorted(set(math.ceil(t*float(clock.frame_rate)-1e-9)
            for t in source_domain.cuts_for_request(source))))
        for source,clock in source_domain.clocks_by_request.items()}
    seam_indices={starts[i]:i for i in range(1,len(plan.segments))}
    events=[]
    for tick in range(1,starts[-1]):
        left,right=samples[tick-1],samples[tick];kind=None
        if left.source_index!=right.source_index:
            kind='source_switch'
        elif tick in seam_indices:
            index=seam_indices[tick]
            # Decoder coalescing needs one nearly linear clock. Physical
            # continuity does not: a continuous piecewise rate map may need
            # separate decoder runs while still crossing a real native cut.
            # A shared run uses the renderer's linear sampler and has no seek;
            # otherwise test the authored source endpoint, not the rate change.
            if (not _can_share_source_clock(plan,first_index=index-1,candidate_last_index=index)
                    and abs(plan.segments[index-1].source_end-plan.segments[index].source_start)>1e-9):
                kind='same_source_clock_jump'
        if kind is not None:
            events.append(VisibleGroupCut(tick,kind,left.source_index,right.source_index,
                left.native_frame_index,right.native_frame_index))
            continue
        if right.native_frame_index<left.native_frame_index:
            raise ValueError('continuous GROUP shot has a reversed native frame clock')
        observed=cuts[right.source_index]
        for frame in observed[bisect_right(observed,left.native_frame_index):
                bisect_right(observed,right.native_frame_index)]:
            events.append(VisibleGroupCut(tick,'native_camera_cut',left.source_index,right.source_index,
                left.native_frame_index,right.native_frame_index,frame))
    return tuple(events)


def visible_group_shots(plan, source_domain, *, cut_inventory=None):
    """Actual visible intervals, independent of artificial canonical cell size."""
    events=visible_group_cut_inventory(plan,source_domain) if cut_inventory is None else cut_inventory
    if not isinstance(events,tuple) or any(not isinstance(e,VisibleGroupCut) for e in events):
        raise ValueError('visible GROUP shots need a typed cut inventory')
    counts=segment_frame_counts(plan);ends=[];cursor=0
    for count in counts:cursor+=count;ends.append(cursor)
    if any(not 0<e.first_incoming_output_frame<cursor for e in events):
        raise ValueError('visible GROUP cut is outside the output clock')
    ticks=sorted({0,cursor,*(e.first_incoming_output_frame for e in events)})
    result=[]
    for first,end in zip(ticks,ticks[1:]):
        a,b=bisect_right(ends,first),bisect_right(ends,end-1)
        source=plan.segments[a].source_index
        if any(s.source_index!=source for s in plan.segments[a:b+1]):
            raise ValueError('visible GROUP shot inventory omitted a source switch')
        result.append(VisibleGroupShot(first,end,source,a,b))
    return tuple(result)


def assert_native_cut_events_preserved(before, after, source_domain):
    """Native event identity AND its output tick, not just aggregate counts."""
    if (before.output_frame_count!=after.output_frame_count or before.fps!=after.fps
            or before.sources!=after.sources):
        raise ValueError('native cut preservation requires one unchanged output/source clock domain')
    def native(plan):
        return tuple((e.first_incoming_output_frame,e.incoming_source_index,e.native_cut_frame)
            for e in visible_group_cut_inventory(plan,source_domain) if e.kind=='native_camera_cut')
    if native(before)!=native(after):
        raise ValueError('proposal changed an actual native camera cut identity or output tick')


def bounded_actual_shot_windows(plan, source_domain, *, maximum_windows=2,
        maximum_mutable_shots=2, minimum_visible_frames=2, maximum_mutable_frames=300):
    """Small consecutive actual-shot windows between fixed visible neighbors.

    This is a search domain, not a candidate selection. A shot bounded by a
    broadcaster cut inside a cell is represented by exact output ticks, not
    expanded to erase that cut. The subsequent joint selector must support
    those fractional-cell bounds or explicitly decline that window.
    """
    if any(type(v) is not int or v<1 for v in (maximum_windows,maximum_mutable_shots,
            minimum_visible_frames,maximum_mutable_frames)) or maximum_mutable_shots>2:
        raise ValueError('actual-shot search requires explicit positive bounded work limits')
    shots=visible_group_shots(plan,source_domain);windows=[]
    for first in range(1,len(shots)-1):
        for size in range(1,maximum_mutable_shots+1):
            end=first+size
            if end>=len(shots):continue
            mutable=shots[first:end]
            if (any(s.frame_count<minimum_visible_frames for s in mutable)
                    or sum(s.frame_count for s in mutable)>maximum_mutable_frames):continue
            windows.append((shots[first-1],*mutable,shots[end]))
            if len(windows)>=maximum_windows:return tuple(windows)
    return tuple(windows)
