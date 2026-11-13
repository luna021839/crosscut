"""Cheap dense camera-control burden, separate from performer/native motion.

This measures the authored crop/pan/zoom/roll at EVERY realized output tick.
It does not detect a head, penalize an actor's movement, or certify quality.
The values are normalized control-space rates, not decoded optical flow.
"""
from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

import numpy as np

from .render import _base_crop_values, _editorial_state_values, segment_frame_counts
from .render_boundary_clock import native_frame_clocks_for_output


@dataclass(frozen=True)
class AddedCameraMotion:
    ranking_cost: float
    maximum_control_velocity: float
    maximum_control_acceleration: float
    mean_control_velocity: float
    measured_output_ticks: int
    excluded_native_or_edit_cut_steps: int
    controls_sha256: str

    def to_dict(self):
        return dict(asdict(self), contract='dense_added_camera_control_motion_v1',
            coordinate_basis='authored_normalized_camera_controls_per_output_second',
            every_output_tick_measured=True, source_or_performer_motion_penalized=False,
            decoded_optical_flow_measured=False, quality_approved=False)


def measure_added_camera_motion(plan, *, segment_indices, source_clocks, scene_cut_times):
    """Measure complete selected cells on the renderer clock, without decoding.

    Split at real edit/native cuts, not at sparse caller-provided samples.
    The base-pan simplification is the SAME one used by the normal renderer.
    Costs stay a soft candidate ordering term; anatomy and containment gates
    are independent and must never be relaxed to compensate for this score.
    """
    indices=tuple(segment_indices)
    if (not indices or len(set(indices))!=len(indices)
            or any(type(i) is not int or not 0<=i<len(plan.segments) for i in indices)):
        raise ValueError('added camera motion requires exact unique renderer cells')
    counts=segment_frame_counts(plan)
    starts=np.cumsum((0,*counts)).tolist()
    ticks=tuple(t for i in sorted(indices) for t in range(starts[i],starts[i+1]))
    if not ticks or len(ticks)>20_000:
        raise ValueError('dense camera-control comparison exceeds its output-tick budget')
    native=native_frame_clocks_for_output(plan,source_clocks,ticks)
    routes=[];coordinates=[];payload=[]
    for i in sorted(indices):
        segment=plan.segments[i]
        if segment.pan_crop is not None:
            segment=replace(segment,pan_crop=segment.pan_crop.simplified())
        source=segment.source_index
        cuts=tuple(scene_cut_times[source])
        if (tuple(sorted(cuts))!=cuts
                or any(type(t) not in (int,float) or not math.isfinite(t) or t<0 for t in cuts)):
            raise ValueError('camera motion requires the actual ordered native scene cuts')
        rate=float(source_clocks[source].frame_rate)
        for tick in range(starts[i],starts[i+1]):
            t=min(segment.duration,(tick-starts[i])/plan.fps)
            x,y,w,h=_base_crop_values(segment,t)
            z,cx,cy,roll=_editorial_state_values(segment.editorial_transform,t)
            # Use relative pan distance in its own crop/camera canvas. Keep
            # base and editorial controls separate: an unknown source aspect
            # must not be silently invented as an exact pixel affine matrix.
            coordinates.append((math.log(z),z*cx,z*cy*plan.height/plan.width,
                math.radians(roll),math.log(w),math.log(h),(x+w/2)/w,(y+h/2)/h))
            sample=native[tick]
            routes.append((i,source,bisect_right(cuts,sample.native_frame_index/rate)))
            payload.append((tick,source,sample.native_frame_index,x,y,w,h,z,cx,cy,roll))
    values=np.asarray(coordinates,dtype=float)
    # A zoom about a fixed camera centre is already represented by log zoom;
    # it must not invent a pan just because z*centre changes with magnification.
    deltas=np.diff(values,axis=0)
    for j in range(len(deltas)):
        z0,z1=payload[j][-4],payload[j+1][-4]
        deltas[j,1]=(payload[j+1][-3]-payload[j][-3])*(z0+z1)/2
        deltas[j,2]=(payload[j+1][-2]-payload[j][-2])*(z0+z1)/2*plan.height/plan.width
    mask=np.asarray([a==b and y==x+1 for a,b,x,y in
        zip(routes,routes[1:],ticks,ticks[1:])],dtype=bool)
    velocities=deltas*plan.fps
    speeds=np.linalg.norm(velocities[mask],axis=1)
    acceleration_mask=mask[:-1]&mask[1:]
    accelerations=np.linalg.norm(np.diff(velocities,axis=0)[acceleration_mask],axis=1)*plan.fps
    peak=float(speeds.max()) if speeds.size else 0.
    mean=float(speeds.mean()) if speeds.size else 0.
    acceleration=float(accelerations.max()) if accelerations.size else 0.
    # Uncalibrated editorial preference, NOT an acceptability threshold.
    cost=mean+.25*peak+.03*acceleration
    digest=hashlib.sha256(json.dumps(dict(fps=plan.fps,size=[plan.width,plan.height],
        samples=payload,native_scene_routes=routes),separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return AddedCameraMotion(cost,peak,acceleration,mean,len(ticks),int((~mask).sum()),digest)
