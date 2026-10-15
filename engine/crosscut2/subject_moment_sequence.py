"""Joint edit-moment/camera refinement before the final programme is frozen.

The coarse source sequence is a proposal, not the already approved film.
Only common-music cut instants can move. No independent source phase, new
cut, camera clamp, post-selection fit, or quality approval is introduced.
"""
import hashlib
import math
import numpy as np
from .boundary_camera_arc import build
from .fixed_holds import _curve_at


def hold_curve(first, final, interior, entry, exit, *, cap):
    first,final=int(first),int(final)
    if final-first<12:return None
    ticks,values=interior
    base=_curve_at(np.asarray(ticks),np.asarray(values),np.arange(first,final))
    return build(np.asarray(entry),np.asarray(exit),base,cap=cap)


def curve_digest(curve):return hashlib.sha256(np.asarray(curve,dtype=np.float64).tobytes()).hexdigest()


def camera_motion(values, native_indices, native_cuts, *, fps):
    """Score digital movement, not actual broadcaster camera discontinuities."""
    values=np.asarray(values,float);native=np.asarray(native_indices)
    if (values.shape!=(len(native),3) or native.dtype.kind not in 'iu' or np.any(np.diff(native)<0)
            or not np.isfinite(values).all() or not math.isfinite(fps) or fps<=0 or np.any(values[:,0]<=0)):
        raise ValueError('finite scored cameras and actual integer native schedule required')
    delta=np.diff(values[:,1:],axis=0)*[16/9,1.];zoom=np.diff(np.log(values[:,0]))
    movement=fps*((delta**2).sum(-1)+zoom**2)
    for cut in native_cuts:movement[(native[:-1]<cut)&(native[1:]>=cut)]=0.
    return float(movement.sum())


def choose(levels, hold, *, check=lambda:None):
    """Exact chain DP: articulated configuration, then connection/camera cost.

    A level contains observed alternatives for ONE existing cut, or its sole
    unchanged anchor when observation was inconclusive. Those two kinds must
    not compete by assigning UNKNOWN a made-up perfect action residual.
    ``hold`` validates and scores the actual whole camera arc before selection.
    """
    if len(levels)<2 or any(not values for values in levels):
        raise ValueError('complete ordered edit-moment candidate levels required')
    for values in levels:
        kinds={bool(v['observed']) for v in values}
        if len(kinds)>1:raise ValueError('UNKNOWN anchor cannot compete as an observed action match')
        frames=[v['frame'] for v in values]
        if len(set(frames))!=len(frames):raise ValueError('one frozen camera proposal per cut moment required')
        for value in values:
            costs=(value['configuration_cost'],value['connection_cost'])
            if type(value['frame']) is not int or not all(math.isfinite(c) and c>=0 for c in costs):
                raise ValueError('finite nonnegative, actual output-clock candidates required')
    costs=[(0.,0.)]*len(levels[0]);parents=[];edges=[];evaluated=0
    for i,values in enumerate(levels[1:]):
        check();best=[(math.inf,math.inf)]*len(values);back=[None]*len(values);proofs=[None]*len(values)
        for q,b in enumerate(values):
            for p,a in enumerate(levels[i]):
                if not math.isfinite(costs[p][0]) or b['frame']-a['frame']<12:continue
                edge=hold(i,a,b);evaluated+=1
                if edge is None:continue
                motion=float(edge['camera_cost'])
                if not math.isfinite(motion) or motion<0:raise ValueError('finite scored camera motion required')
                score=(costs[p][0]+b['configuration_cost'],
                       costs[p][1]+b['connection_cost']+motion)
                if score<best[q]:best[q]=score;back[q]=p;proofs[q]=edge
        parents.append(back);edges.append(proofs);costs=best
    q=min(range(len(costs)),key=lambda k:costs[k]);objective=costs[q]
    if not math.isfinite(objective[0]):raise ValueError('no realizable common-clock refinement sequence')
    selected=[q];chosen=[]
    for i in range(len(parents)-1,-1,-1):
        chosen.append(edges[i][q]);q=parents[i][q];selected.append(q)
    return selected[::-1],chosen[::-1],dict(objective=list(objective),evaluated_arcs=evaluated,
        source_sequence_changed=False,independent_source_phase_search=False,
        authored_cut_count_changed=False,post_selection_camera_fit=False,
        professional_quality_approved=False)
