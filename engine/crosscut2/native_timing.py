"""Bounded exact-schedule native-fragment repair; no new sensor/grid analysis.

Selection's sparse pose ports remain pre-repair observations. Every changed
boundary is explicitly unvalidated until actual-pixel/audiovisual review.
"""
import numpy as np


def flash_fragments(schedules, native, *, maximum_frames=5):
    rows=[]
    for index,(start,segment,schedule) in enumerate(schedules):
        end=start+len(schedule.source_frame_indices)
        cuts=sorted(tick for tick in native if start<tick<end)
        for a,b in zip([start,*cuts],[*cuts,end]):
            if 0<b-a<=maximum_frames:
                rows.append(dict(segment=index,first_frame=int(a),final_frame=int(b),frames=int(b-a)))
    return rows


def repair_native_timing(plan, build, census, *, check, maximum_frames=5, maximum_passes=2):
    from crosscut.render_frame_schedule import probe_render_frame_clock
    from .programme_critic import programme_schedules
    clocks=[probe_render_frame_clock(source) for source in plan.sources]
    cuts=[record['frames'] for record in census]
    initial=np.array([round(s.master_start*plan.fps) for s in plan.segments]+[plan.output_frame_count],int)
    bounds=initial.copy();changes=[]
    def evaluate(boundaries):
        candidate=build(boundaries)
        schedules,_,native=programme_schedules(candidate,clocks,cuts)
        fragments=flash_fragments(schedules,native,maximum_frames=maximum_frames)
        # Count first, then favor removing the shortest visible shot fragments.
        risk=sum((maximum_frames+1-row['frames']) for row in fragments)
        return candidate,schedules,native,fragments,(len(fragments),risk)
    current,schedules,native,before,score=evaluate(bounds)
    for _ in range(maximum_passes):
        changed=False
        for index in range(1,len(bounds)-1):
            check()
            choices={int(bounds[index]),*[int(tick) for tick in native
                if abs(tick-initial[index])<=maximum_frames]}
            best=(score,0,int(bounds[index]),None)
            for tick in choices:
                if tick==bounds[index] or abs(tick-initial[index])>maximum_frames:
                    continue
                if min(tick-bounds[index-1],bounds[index+1]-tick)<int(np.ceil(.9*plan.fps)):
                    continue
                proposed=bounds.copy();proposed[index]=tick
                evaluated=evaluate(proposed)
                key=(evaluated[-1],abs(tick-initial[index]),tick)
                if key<best[:3]:best=(*key,evaluated)
            if best[3] is not None and best[0]<score:
                old=int(bounds[index]);bounds[index]=best[2]
                current,schedules,native,_,score=best[3]
                changes.append(dict(boundary=index,previous_frame=old,frame=int(bounds[index]),
                    initial_frame=int(initial[index]),reason='exact_schedule_short_native_fragment',
                    new_pose_port_measured=False))
                changed=True
        if not changed:break
    _,_,_,after,_=evaluate(bounds)
    return bounds,dict(maximum_adjustment_frames=maximum_frames,changes=changes,
        fragments_before=before,fragments_after=after,new_source_inferences=0,
        selected_dp_cost_is_pre_repair=True,changed_pose_ports_are_not_certified=True)
