"""Experimental cross-source SUBJECT retrieval before camera/programme DP.

The target key survives projection. Common-clock dance observations are
context, not fictitious played-H costs. No cut reward, fit, named identity,
normal-app activation or professional-quality acceptance is provided here.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np
from crosscut.render_frame_schedule import sampler_frame_schedule
from .subject_window_bank import SubjectWindow
from .paired_geometry import joint_cameras


@dataclass(frozen=True)
class SubjectClock:
    source: int
    binding: str
    output_frames: np.ndarray
    native_indices: np.ndarray
    phase_frames: int
    fps: float
    schedule_sha256: str


def clock(*, source, binding, relative_frames, source_start, playback_rate,
          output_fps, native_fps, native_count, phase_frames=0):
    ticks=np.asarray(relative_frames)
    if (ticks.ndim!=1 or len(ticks)<5 or ticks.dtype.kind not in 'iu' or np.any(np.diff(ticks)<=0)
            or not np.any(ticks==-1) or not np.any(ticks==0)
            or type(phase_frames) is not int or phase_frames not in (-2,-1,0,1,2)
            or not binding or not all(math.isfinite(v) and v>0 for v in (output_fps,native_fps,playback_rate))):
        raise ValueError('explicit common output window and finite constant phase hypothesis required')
    start=source_start+phase_frames/output_fps+int(ticks[0])/output_fps*playback_rate
    schedule=sampler_frame_schedule(source_start=start,source_playback_rate=playback_rate,
        native_fps=native_fps,output_fps=output_fps,frame_count=int(ticks[-1]-ticks[0]+1),
        source_frame_count=native_count)
    native=np.asarray(schedule.source_frame_indices,dtype=np.int64)[ticks-ticks[0]].copy()
    metadata=dict(source=int(source),binding=binding,source_start=source_start,playback_rate=playback_rate,
        output_fps=output_fps,native_fps=native_fps,native_count=native_count,phase_frames=phase_frames)
    digest=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()+ticks.tobytes()+native.tobytes()).hexdigest()
    ticks=ticks.copy();ticks.setflags(write=False);native.setflags(write=False)
    return SubjectClock(source,binding,ticks,native,phase_frames,output_fps,digest)


def _sample(window,hypothesis):
    if window.source!=hypothesis.source or window.binding!=hypothesis.binding:
        raise ValueError('subject/window source binding mismatch')
    at=hypothesis.native_indices-window.ordinals[0]
    if np.any(at<0) or np.any(at>=len(window.ordinals)) or not np.array_equal(window.ordinals[at],hypothesis.native_indices):
        raise ValueError('clock requests unobserved native frames; no interpolation/cache relabel')
    return at


def incoming_camera(left_head,right_head,left_camera,*,cap):
    """One-sided legal crop. The already played outgoing camera is immutable."""
    if not 1<=cap<=2:raise ValueError('declared existing source zoom capacity required')
    a,b,c=np.asarray(left_head),np.asarray(right_head),np.asarray(left_camera)
    target=.5+c[0]*(a[:2]-c[1:]);height=a[2]*c[0]
    zoom=float(np.clip(height/max(.001,b[2]),1.,cap))
    center=b[:2]-(target-.5)/zoom
    center=np.clip(center,.5/zoom,1.-.5/zoom)
    offset=center-.5;radius=np.linalg.norm(offset)
    center=.5+offset*min(1.,.25/max(1e-12,radius))
    return np.r_[zoom,center]


def _visibility(head,camera,others):
    rows=np.vstack((head,others));z=camera[0]
    centers=.5+z*(rows[:,:2]-camera[1:]);sizes=rows[:,2]*z
    half=sizes[:,None]*[.45*9/16,.5]
    extent=np.maximum(0.,np.minimum(1.,centers+half)-np.maximum(0.,centers-half))
    score=extent.prod(-1)*np.exp(-((centers[:,0]-.5)/.4)**2)
    score=np.where((rows[:,2]>.019)&(rows[:,3]>=.75),score,0.)
    fraction=float(np.prod(extent[0]/np.maximum(1e-12,2*half[0])))
    # Return the PINNED actor even when someone else dominates the viewport.
    return fraction,float(score[0]/max(1e-12,float(score.max())))


def compare(left,right,left_clock,right_clock,*,left_cameras,incoming_cap,
            nonzero_audio_and_mouth_verified=False,following_four_seconds_verified=False,
            outgoing_locked=True,outgoing_cap=None):
    if not isinstance(left,SubjectWindow) or not isinstance(right,SubjectWindow):raise ValueError('subject windows required')
    if (left_clock.phase_frames!=0 or left_clock.fps!=right_clock.fps
            or not np.array_equal(left_clock.output_frames,right_clock.output_frames)):
        raise ValueError('fixed outgoing clock and common music window required')
    a,b=_sample(left,left_clock),_sample(right,right_clock);ticks=left_clock.output_frames
    cameras=np.asarray(left_cameras,float)
    if (cameras.shape!=(len(ticks),3) or not np.isfinite(cameras).all()
            or np.any(cameras[:,0]<1) or np.any(cameras[:,0]>2) or np.any(cameras[:,1:]<0)
            or np.any(cameras[:,1:]>1)):
        raise ValueError('actual outgoing camera samples required')
    la,rb=int(np.flatnonzero(ticks==-1)[0]),int(np.flatnonzero(ticks==0)[0])
    ca=cameras[la];cb=incoming_camera(left.heads[a[la]],right.heads[b[rb]],ca,cap=incoming_cap)
    if not outgoing_locked:
        # A NEW joint camera proposal, before programme selection. This is
        # not permission to patch the frozen old programme after DP or claim
        # that a changed outgoing view is a matched-prefix H/K comparison.
        if outgoing_cap is None:raise ValueError('explicit outgoing zoom capacity required')
        ha,hb=left.heads[a[la]],right.heads[b[rb]]
        if min(ha[2],hb[2])>.019:
            lc,rc,_=joint_cameras(ha[:2],hb[:2],ha[2],hb[2],caps=(outgoing_cap,incoming_cap))
            ca=np.array([lc.zoom,lc.x,lc.y]);cb=np.array([rc.zoom,rc.x,rc.y])
            cameras=np.broadcast_to(ca,cameras.shape)
    common=left.support[a]&right.support[b]&left.associated[a,None]&right.associated[b,None]
    same_shot=np.ones(len(ticks),bool)
    # A seed track is not propagated across an original camera cut. The
    # producer already splits it; this explicit check also guards imports.
    for window,at in ((left,a),(right,b)):
        low,high=window.ordinals[at].min(),window.ordinals[at].max()
        if any(low<cut<=high for cut in window.native_cuts):
            same_shot[:]=False
    common &= same_shot[:,None]
    xy_a=left.body[a,:,:2]*[16/9,1.];xy_b=right.body[b,:,:2]*[16/9,1.]
    def normalized(xy,heads,valid):
        shoulders=valid[:,0]&valid[:,1]
        root=np.where(shoulders[:,None],xy[:,:2].mean(1),heads[:,:2]*[16/9,1.])
        return (xy-root[:,None])/np.maximum(.02,heads[:,None,2,None])
    na=normalized(xy_a,left.heads[a],left.support[a]);nb=normalized(xy_b,right.heads[b],right.support[b])
    residual=np.linalg.norm(na-nb,axis=-1)
    phase=float(residual[common].mean()) if common.any() else None
    # Pose NN is intentionally sparse; compare successive ACTUAL common
    # observations, not adjacent array slots containing unobserved zeros.
    # This is finite-interval motion, never interpolated wrist positions.
    motion_terms=[];activity_terms=[];interval_counts={}
    for joint in range(8):
        samples=np.flatnonzero(common[:,joint])
        for first,last in zip(samples[:-1],samples[1:]):
            gap=int(ticks[last]-ticks[first])
            if gap>6:continue
            if (not left.associated[a[first]:a[last]+1].all()
                    or not right.associated[b[first]:b[last]+1].all()):continue
            dt=gap/left_clock.fps
            va=(na[last,joint]-na[first,joint])/dt
            vb=(nb[last,joint]-nb[first,joint])/dt
            motion_terms.append(float(np.linalg.norm(va-vb)*dt))
            activity_terms.append(float(max(np.linalg.norm(va),np.linalg.norm(vb))))
            edge=(int(first),int(last));interval_counts[edge]=interval_counts.get(edge,0)+1
    motion=float(np.mean(motion_terms)) if motion_terms else None
    activity=float(np.mean(activity_terms)) if activity_terms else None
    av,bv=left.vectors[a],right.vectors[b]
    appearances=(np.linalg.norm(av,axis=1)>.5)&(np.linalg.norm(bv,axis=1)>.5)&left.associated[a]&right.associated[b]
    cosine=float(np.median(np.sum(av[appearances]*bv[appearances],axis=1))) if appearances.any() else None
    enough=int((common.sum(-1)>=2).sum())>=5 and sum(v>=2 for v in interval_counts.values())>=3
    evidence=bool(enough and int(appearances.sum())>=3)
    pa=.5+cameras[:,None,:1]*(left.body[a,:,:2]-cameras[:,None,1:])
    pb=.5+cb[0]*(right.body[b,:,:2]-cb[1:])
    projected=np.linalg.norm((pa-pb)*[16/9,1.],axis=-1)
    scale=np.maximum(.02,(left.heads[a,2]*cameras[:,0]+right.heads[b,2]*cb[0])/2.)
    screen=float((projected/scale[:,None])[common].mean()) if common.any() else None
    ha=left.heads[a[la],2]*ca[0];hb=right.heads[b[rb],2]*cb[0]
    center_a=.5+ca[0]*(left.heads[a[la],:2]-ca[1:]);center_b=.5+cb[0]*(right.heads[b[rb],:2]-cb[1:])
    face=max(abs(math.log(max(.001,ha)/max(.001,hb)))/math.log(1.5),
        float(np.linalg.norm((center_a-center_b)*[16/9,1.]))/max(.02,(ha+hb)/2.))
    visible,attention=_visibility(right.heads[b[rb]],cb,right.competitors[b[rb]])
    status='SUPPORTED_CANDIDATE' if evidence else 'NO_SUPPORTED_MATCH'
    if visible<.6 or attention<.6:status='TARGET_OCCLUDED_OR_NOT_ATTENDED'
    score=None if not evidence else float(phase+motion+screen+face+(1.-cosine))
    return dict(status=status,source_A=left.source,track_A=left.track,source_B=right.source,track_B=right.track,
        clock_hypothesis_frames=right_clock.phase_frames,clock_A=left_clock.schedule_sha256,
        clock_B=right_clock.schedule_sha256,native_A=left_clock.native_indices.tolist(),native_B=right_clock.native_indices.tolist(),
        incoming_camera=cb.tolist(),outgoing_camera=ca.tolist(),common_joints_per_sample=common.sum(-1).tolist(),
        comparable_motion_pairs=len(motion_terms),measured_motion_intervals=len(interval_counts),
        phase_residual=phase,motion_residual=motion,
        screen_residual=screen,face_residual=face,appearance_cosine=cosine,source_activity=activity,
        target_visible_fraction=visible,target_attention_ratio=attention,rank_score=score,
        outgoing_camera_locked=bool(outgoing_locked),new_joint_camera_before_selection=not outgoing_locked,
        candidate_has_measured_action=evidence,matched_identity_certified=False,anatomical_ownership_certified=False,
        target_reselected_after_camera=False,following_four_seconds_verified=following_four_seconds_verified,
        diagnostic_phase_only=bool(right_clock.phase_frames and not nonzero_audio_and_mouth_verified),
        # Residuals are engineering measurements, not calibrated editorial
        # quality gates. In real clips a lower phase/motion score did NOT
        # establish a better action cut. No automatic user-facing approval.
        diagnostic_render_eligible=bool(status=='SUPPORTED_CANDIDATE' and following_four_seconds_verified
            and (right_clock.phase_frames==0 or nonzero_audio_and_mouth_verified)),
        preview_eligible=False,requires_actual_visual_review=True,
        professional_quality_approved=False,normal_app_activation_approved=False)


def rank(results,*,include_diagnostic_phases=False):
    """Unsupported is a separate outcome, never a cheap neutral action match."""
    available=[r for r in results if r['status']=='SUPPORTED_CANDIDATE' and r['rank_score'] is not None]
    diagnostics=[r for r in available if r['diagnostic_phase_only']]
    measured=available if include_diagnostic_phases else [r for r in available if not r['diagnostic_phase_only']]
    unsupported=[r for r in results if r not in available]
    measured.sort(key=lambda r:(r['rank_score'],r['source_B'],r['track_B'],abs(r['clock_hypothesis_frames'])))
    return dict(supported=measured,unsupported=unsupported,diagnostic_phase_candidates=diagnostics,
        status='SUPPORTED_CANDIDATES' if measured else 'NO_SUPPORTED_MATCH',
        H_hold_is_not_a_success=True,weights_fitted=False,cut_reward_added=False,
        diagnostic_phases_explicitly_included=include_diagnostic_phases)
