"""Joint edit-moment/camera refinement before the final programme is frozen.

The coarse source sequence is a proposal, not the already approved film.
Only common-music cut instants can move. No independent source phase, new
cut, camera clamp, post-selection fit, or quality approval is introduced.
"""
import hashlib
import math
import numpy as np
from .boundary_camera_arc import build,CameraArc,realizable,_hermite
from .fixed_holds import _curve_at


def hold_curve(first, final, interior, entry, exit, *, cap,family='connectors'):
    first,final=int(first),int(final)
    if final-first<12:return None
    ticks,values=interior
    base=_curve_at(np.asarray(ticks),np.asarray(values),np.arange(first,final))
    if family=='connectors':return build(np.asarray(entry),np.asarray(exit),base,cap=cap)
    if family=='convex_connectors':
        return convex_connectors(base,entry,exit,cap=cap)
    if family not in ('whole_viewport','whole_open'):raise ValueError('declared finite camera family required')
    entry=np.asarray(entry,float);exit=np.asarray(exit,float);length=len(base)
    if length<14:return build(entry,exit,base,cap=cap)
    if entry.shape!=(7,3) or exit.shape!=(7,3) or not realizable(entry,cap) or not realizable(exit,cap):return None
    # Inverse zoom is viewport width. Padding bounds are linear in this
    # coordinate, so zero-slope Hermite endpoints form a feasible convex path.
    # Nonzero strip velocities may be infeasible; reject, never clamp them.
    a=entry.copy();b=exit.copy();a[:,0]=1/a[:,0];b[:,0]=1/b[:,0]
    mid=(length-1)//2
    if family=='whole_open' and 7<mid<length-8:
        center=base[mid].copy()
        if not realizable(center[None],cap):return None
        center[0]=1/center[0];zero=np.zeros(3)
        before=_hermite(a[-1],a[-1]-a[-2],center,zero,mid-6)
        after=_hermite(center,zero,b[0],b[1]-b[0],length-7-mid)
        middle=np.concatenate((before,center[None],after))
    else:middle=_hermite(a[-1],a[-1]-a[-2],b[0],b[1]-b[0],length-13)
    if np.any(middle[:,0]<=0):return None
    middle[:,0]=1/middle[:,0]
    path=np.concatenate((entry,middle,exit))
    if len(path)!=length or not realizable(path,cap):return None
    changed=np.any(path!=base,axis=1)
    bends=np.flatnonzero(np.any(np.diff(path,2,axis=0)!=0.,axis=1))+1
    path.setflags(write=False);changed.setflags(write=False)
    return CameraArc(path,changed,tuple(sorted({0,length-1,*map(int,bends)})),family)


def convex_connectors(base,entry,exit,*,cap,connector_frames=12):
    """Local crop bridges in the convex inverse-zoom/centre domain.

    For constant observed seam strips, blend each endpoint with the actual
    baseline viewport, not an additive zoom/pan offset. Both endpoints and
    that baseline can be legal while an additive offset leaves the padding
    domain. This family preserves every seam sample and the long-hold middle;
    it is not a camera/actor/comfort certificate or a whole-shot zoom hold.
    The caller must not bridge an original native camera discontinuity.
    """
    base=np.asarray(base,float);entry=np.asarray(entry,float);exit=np.asarray(exit,float)
    length=len(base)
    if (isinstance(cap,(bool,np.bool_)) or not np.isfinite(cap) or cap<1
            or length<14 or entry.shape!=(7,3) or exit.shape!=(7,3)
            or type(connector_frames) is not int or connector_frames<1
            or not realizable(base,cap) or not realizable(entry,cap) or not realizable(exit,cap)):
        return None
    if not np.array_equal(entry,np.broadcast_to(entry[-1],entry.shape)) \
            or not np.array_equal(exit,np.broadcast_to(exit[0],exit.shape)):
        return None  # no fabricated extension of a moving seam camera
    inverse=base.copy();inverse[:,0]=1/inverse[:,0]
    a=entry[-1].copy();b=exit[0].copy();a[0]=1/a[0];b[0]=1/b[0]
    if length>=14+2*connector_frames:
        for side,anchor in ((0,a),(1,b)):
            for step in range(1,connector_frames):
                k=6+step if side==0 else length-7-step
                t=step/connector_frames;weight=1-3*t*t+2*t*t*t
                inverse[k]=weight*anchor+(1-weight)*inverse[k]
    else:
        span=length-13
        for k in range(7,length-7):
            t=(k-6)/span;weight=3*t*t-2*t*t*t
            inverse[k]=(1-weight)*a+weight*b
    inverse[:,0]=1/inverse[:,0]
    inverse[:7]=entry;inverse[-7:]=exit
    if length>=14+2*connector_frames:
        inverse[6+connector_frames:length-6-connector_frames]=base[6+connector_frames:length-6-connector_frames]
    if not realizable(inverse,cap):return None
    changed=np.any(inverse!=base,axis=1)
    bends=np.flatnonzero(np.any(np.diff(inverse,2,axis=0)!=0.,axis=1))+1
    inverse.setflags(write=False);changed.setflags(write=False)
    return CameraArc(inverse,changed,tuple(sorted({0,length-1,*map(int,bends)})),'convex_connectors')


def observed_view_cost(values,native_indices,windows,*,fps):
    """Integrate only actual played-window head/body/crop evidence.

    Unobserved interiors are not certified. No source/camera fit or inferred
    performer motion appears here; overlapping observations are averaged.
    """
    values=np.asarray(values,float);native=np.asarray(native_indices,int)
    if values.shape!=(len(native),3) or not np.isfinite(values).all() or fps<=0:
        raise ValueError('one actual whole-hold camera/native schedule required')
    loss=np.zeros(len(native));count=np.zeros(len(native))
    for window in windows:
        at=native-int(window.ordinals[0]);inside=(at>=0)&(at<len(window.ordinals))
        ids=np.flatnonzero(inside)
        ids=ids[window.associated[at[ids]]]
        if not len(ids):continue
        at=at[ids];heads=window.heads[at];camera=values[ids]
        maximum=max(len(window.competitors[k]) for k in at)+1
        rows=np.zeros((len(ids),maximum,4));rows[:,0]=heads
        for j,k in enumerate(at):rows[j,1:1+len(window.competitors[k])]=window.competitors[k]
        centers=.5+camera[:,None,:1]*(rows[...,:2]-camera[:,None,1:]);height=rows[...,2]*camera[:,None,0]
        half=height[...,None]*[.45*9/16,.5]
        extent=np.maximum(0.,np.minimum(1.,centers+half)-np.maximum(0.,centers-half))
        scores=extent.prod(-1)*np.exp(-((centers[...,0]-.5)/.4)**2)
        scores=np.where((rows[...,2]>.019)&(rows[...,3]>=.75),scores,0.)
        # Ratios are mathematically in[0,1]; rounding of extent subtraction
        # must not turn a fully visible target into a negative content cost.
        # Only the scalar measurement is bounded, never the replayed camera.
        fraction=np.clip(np.prod(extent[:,0]/np.maximum(1e-12,2*half[:,0]),axis=-1),0.,1.)
        attention=np.clip(scores[:,0]/np.maximum(1e-12,scores.max(-1)),0.,1.)
        body=.5+camera[:,None,:1]*(window.body[at,:,:2]-camera[:,None,1:])
        supported=window.support[at]
        clipped=(supported&((body<0).any(-1)|(body>1).any(-1))).sum(-1)/np.maximum(1,supported.sum(-1))
        # Same oversized-head effort as joint_cameras. Matching at a seam is
        # not a reason to keep enlarging a readable face throughout the shot.
        tight=6.*np.maximum(0.,height[:,0]-np.maximum(.24,heads[:,2]))
        loss[ids]+=1.-fraction+1.-attention+clipped+tight;count[ids]+=1
    known=count>0
    visibility=float((loss/np.maximum(1,count)).sum()/fps)
    # Same played digital-zoom regularizer as SourceCatalogue.rates; not a
    # newly tuned reward or a blanket1.15x limit. Actual match crops stay legal.
    zoom=float((.1*np.log(np.maximum(1.,values[:,0]))/np.log(2.)).sum()/fps)
    return dict(observed_crop_cost=visibility,digital_zoom_cost=zoom,
        observed_output_frames=int(known.sum()),unobserved_output_frames=int((~known).sum()),
        full_hold_subject_visibility_certified=False)


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


def camera_path_cost(values, native_indices, native_cuts, windows, *, fps):
    """Score an actual path independently of which families were proposed.

    Choosing whether to offer a whole-shot family must never turn off the
    content cost of an otherwise identical local connector. Missing subject
    observations remain explicit coverage, not a whole-hold visibility claim.
    """
    motion = camera_motion(values, native_indices, native_cuts, fps=fps)
    view = observed_view_cost(values, native_indices, windows, fps=fps)
    return dict(camera_cost=motion+view['observed_crop_cost']+view['digital_zoom_cost'],
                digital_motion_cost=motion, **view)


def choose(levels, hold, *, check=lambda:None):
    """Exact chain DP over one finite pose/connection/camera sum.

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
    costs=[0.]*len(levels[0]);parents=[];edges=[];evaluated=0
    for i,values in enumerate(levels[1:]):
        check();best=[math.inf]*len(values);back=[None]*len(values);proofs=[None]*len(values)
        for q,b in enumerate(values):
            for p,a in enumerate(levels[i]):
                if not math.isfinite(costs[p]) or b['frame']-a['frame']<12:continue
                edge=hold(i,a,b);evaluated+=1
                if edge is None:continue
                motion=float(edge['camera_cost'])
                if not math.isfinite(motion) or motion<0:raise ValueError('finite scored camera motion required')
                score=costs[p]+b['configuration_cost']+b['connection_cost']+motion
                if score<best[q]:best[q]=score;back[q]=p;proofs[q]=edge
        parents.append(back);edges.append(proofs);costs=best
    q=min(range(len(costs)),key=lambda k:costs[k]);objective=costs[q]
    if not math.isfinite(objective):raise ValueError('no realizable common-clock refinement sequence')
    selected=[q];chosen=[]
    for i in range(len(parents)-1,-1,-1):
        chosen.append(edges[i][q]);q=parents[i][q];selected.append(q)
    return selected[::-1],chosen[::-1],dict(objective=objective,evaluated_arcs=evaluated,
        objective_contract='finite_sum_articulated_configuration_connection_whole_camera_v2',
        lexicographic_infinite_pose_priority=False,weights_fitted=False,
        source_sequence_changed=False,independent_source_phase_search=False,
        authored_cut_count_changed=False,post_selection_camera_fit=False,
        professional_quality_approved=False)
