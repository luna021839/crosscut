"""Small exact-tick boundary choices and inserted native-cut collision checks.

This is a bounded review search domain, not an aesthetic minimum-shot rule.
Unchanged broadcaster edits are retained, including already short shots.
"""


def coalesce_review_mutable_intervals(intervals, *, output_frame_count):
    """Normalize overlapping lattice spans without enlarging their union."""
    if (type(output_frame_count) is not int or not 1 <= output_frame_count <= 36000
            or not isinstance(intervals, (list, tuple)) or not 1 <= len(intervals) <= 256
            or any(not isinstance(s, (list, tuple)) or len(s) != 2
                or any(type(t) is not int for t in s)
                or not 0 <= s[0] < s[1] <= output_frame_count for s in intervals)):
        raise ValueError('mutable lattice union needs bounded exact intervals')
    merged = []
    for first, end in sorted(intervals):
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((first, end))
    return tuple(merged)


def review_boundary_options(tick,*,output_frame_count,observed_event_ticks=(),
        protected_intervals=(),radius=2,maximum_options=3):
    if (type(tick) is not int or type(output_frame_count) is not int
            or not 0<=tick<=output_frame_count<=36000
            or type(radius) is not int or not 1<=radius<=6
            or type(maximum_options) is not int or not 1<=maximum_options<=7):
        raise ValueError('review boundaries need finite exact output ticks')
    if any(type(t) is not int or not 0<t<output_frame_count for t in observed_event_ticks):
        raise ValueError('boundary lattice borrowed an absent native event')
    if tick in (0,output_frame_count):return (tick,)
    intervals=tuple(protected_intervals)
    if any(type(a) is not int or type(b) is not int or not 0<=a<b<=output_frame_count
            for a,b in intervals):
        raise ValueError('boundary protection is not an exact output interval')
    if any(a<=tick<=b for a,b in intervals):return (tick,)
    events=sorted(set(t for t in observed_event_ticks if 0<abs(t-tick)<=radius),
        key=lambda t:(abs(t-tick),t))
    candidates=[tick,*events,tick-radius,tick+radius]
    selected=[]
    for t in candidates:
        if (0<t<output_frame_count and t not in selected
                and not any(min(t,tick)<b and a<max(t,tick) for a,b in intervals)):
            selected.append(t)
        if len(selected)==maximum_options:break
    return tuple(sorted(selected))


def inserted_flash_collisions(candidate_events,original_events,*,maximum_flash_frames=2):
    """Find new 1--2 frame shots adjoining an authored source switch/seek.

    The operands are actual sampler-mapped cut inventories. Two records at
    the same output tick are one visible event, never a zero-frame shot.
    """
    if type(maximum_flash_frames) is not int or not 1<=maximum_flash_frames<=2:
        raise ValueError('inserted flash check needs a bounded collision window')
    def group(events):
        grouped={}
        for e in events:
            row=e.to_dict() if hasattr(e,'to_dict') else e
            if (not isinstance(row,dict) or type(row.get('first_incoming_output_frame')) is not int
                    or row['first_incoming_output_frame']<=0 or row.get('kind') not in
                    ('source_switch','same_source_clock_jump','native_camera_cut')):
                raise ValueError('flash check needs actual typed native/source events')
            signature=(row['kind'],row.get('outgoing_source_index'),row.get('incoming_source_index'),
                row.get('outgoing_native_frame'),row.get('incoming_native_frame'),row.get('native_cut_frame'))
            grouped.setdefault(row['first_incoming_output_frame'],set()).add(signature)
        return grouped
    before,after=group(original_events),group(candidate_events)
    ticks=sorted(after);problems=[]
    for a,b in zip(ticks,ticks[1:]):
        if b-a>maximum_flash_frames:continue
        authored={'source_switch','same_source_clock_jump'}
        has_authored=any(signature[0] in authored for t in (a,b) for signature in after[t])
        # A retained source-switch tick can acquire a NEW nearby native cut;
        # the switch need not move for the resulting flash to be new.
        new_authored=has_authored and any(after[t]!=before.get(t,set()) for t in (a,b))
        if new_authored:
            problems.append(dict(first=a,end=b,frame_count=b-a,
                first_kinds=sorted({s[0] for s in after[a]}),end_kinds=sorted({s[0] for s in after[b]})))
    return tuple(problems)
