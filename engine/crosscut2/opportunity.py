"""Observed music/action opportunities, not identity or quality certificates.

The .8 event cap is the existing moving-match cap, frozen before a media trial.
Missing/contrary action removes only this credit, never the source or the cut.
"""
import math
import numpy as np
from .action_geometry import normalized_body, _arm_segments
from .paired_geometry import Camera

OPPORTUNITY_CAP = .8


def music_on_nodes(nodes, *, fps, master_start, rhythm):
    nodes = np.asarray(nodes)
    if (nodes.ndim!=1 or len(nodes)<2 or nodes[0]!=0 or np.any(np.diff(nodes)<=0)
            or not math.isfinite(fps) or fps <= 0 or not math.isfinite(master_start)):
        raise ValueError('finite canonical music clock required')
    values = np.zeros(len(nodes))
    locations = {int(t): i for i, t in enumerate(nodes)}
    # Use observed strengths, not index modulo four or imputed downbeats.
    for beat in rhythm.get('beat_grid', ()):
        t, strength = float(beat['time_seconds']), float(beat['strength'])
        if not math.isfinite(t) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('finite observed beat strength in [0,1] required')
        index = locations.get(round((t-master_start)*fps))
        if index is not None:
            values[index] = max(values[index], strength)
    values[[0,-1]] = 0.
    return values


def continuous_music_context(nodes, *, fps, master_start, rhythm):
    """Separate observed beat events from their continuous strength envelope.

    A source-observed action moment between beats is not an invented beat.
    Interpolation supplies only local musical strength, never pose, a new
    episode boundary, a downbeat label, or repeated event credit.
    """
    music_on_nodes(nodes, fps=fps, master_start=master_start, rhythm=rhythm)
    nodes = np.asarray(nodes)
    beats = {}
    for beat in rhythm.get('beat_grid', ()):
        tick = round((float(beat['time_seconds'])-master_start)*fps)
        beats[tick] = max(beats.get(tick, 0.), float(beat['strength']))
    ticks = np.asarray(sorted(beats), np.int64)
    if len(ticks):
        strength = np.interp(nodes, ticks, [beats[int(t)] for t in ticks], left=0., right=0.)
    else:
        strength = np.zeros(len(nodes))
    strength[[0,-1]] = 0.
    events = ticks[(ticks > 0) & (ticks < nodes[-1])]
    return strength, events


def phase_tail(left, right):
    """Worst observed articulated residual; no torso-only match credit."""
    a, av = normalized_body(left)
    b, bv = normalized_body(right)
    shared = av & bv
    shared[:2] = False; shared[6:] = False
    if a is not None and b is not None and shared.sum() >= 2:
        return float(np.linalg.norm(a[shared]-b[shared],axis=1).max()), 'shoulder_relative_articulation'
    a, av = _arm_segments(left); b, bv = _arm_segments(right)
    shared = av & bv
    if shared.sum() >= 2:
        return float(np.linalg.norm(a[shared]-b[shared],axis=1).max()/2.), 'directed_complete_arm_segments'
    return None, None


def joint_opportunity(n, left, right, measurement, *, music_strength):
    if not math.isfinite(music_strength) or not 0 <= music_strength <= 1:
        raise ValueError('bound observed musical strength required')
    result = dict(music_strength=float(music_strength),joint_fit=0.,credit=0.,
                  cap=OPPORTUNITY_CAP,reason='unobserved_joint_connection')
    windows = measurement.get('synchronous_motion_windows', ())
    if (measurement['role'] != 'focal' or measurement['appearance_cosine'] is None
            or len(windows) != 2 or any(w['loss'] is None for w in windows)
            or not any(w['moving_joints'] for w in windows)):
        return result
    # Require the actual distinct source-native samples, not duplicate ordinals.
    for row in (left,right):
        indices = row.get('pose_native_indices')
        if indices is None or indices[n,0] == indices[n,1] or indices[n,2] == indices[n,3]:
            return dict(result,reason='unmeasured_or_repeated_pose_clock')
    direction = measurement['maximum_direction_disagreement']
    if direction >= .5:
        return dict(result,reason='observed_contrary_or_one_sided_motion')
    phase,basis = phase_tail(left['body'][n,1],right['body'][n,2])
    screen = measurement['projected_screen_motion']
    if phase is None or screen is None or screen['prediction_loss'] is None:
        return result
    crop = max(left['utility_parts']['crop_clipping'][n,0],right['utility_parts']['crop_clipping'][n,1])
    identity_risk = 5.*max(0.,.65-measurement['appearance_cosine'])
    speed_tail = max((v for w in windows for v in w['speed_losses']),default=0.)
    # These are finite preference units from the existing objective, NOT
    # probabilities. A weak component cannot be canceled by another limb.
    residuals = [measurement['geometry_loss'],identity_risk,3.*phase,
                 1.5*measurement['synchronous_worst_window_loss'],.3*speed_tail,crop,
                 screen['position_loss']/.12,screen['prediction_loss']/.12]
    fit = min(1./(1.+max(0.,r)) for r in residuals) * max(0.,1.-2.*direction)
    return dict(result,joint_fit=float(fit),credit=float(OPPORTUNITY_CAP*music_strength*fit),
                reason='observed_partial_joint_connection',phase_tail=phase,phase_basis=basis,
                normalized_residuals=list(map(float,residuals)))


def paired_clock_connection(n,left,right,measurement,*,music_strength):
    """Calibrate against real same-clock continuation, not a stationary pose.

    Both cached sources contain all four [-7,-1,0,+6] samples. Comparing
    A[-1] with B[0] as if natural motion were an error penalizes even A->A.
    Compare A[t]/B[t] at each real sample instead; keep same-clock velocity,
    actual camera/owner-through costs and source uncertainty. No extrapolated
    stationary pose and no subtraction of a noisy/jerky camera's own risk.
    """
    base=joint_opportunity(n,left,right,measurement,music_strength=music_strength)
    result=dict(connection_opportunity=dict(base,joint_fit=0.,credit=0.,reason='unobserved_paired_clock_connection'),
                nonnegative_risk_cost=measurement['nonnegative_risk_cost'],
                paired_clock_supported=False)
    if base['joint_fit']<=0 or not left['continuous'][n] or not right['continuous'][n]:return result
    phases,bases,screens,faces,cosines=[],[],[],[],[]
    for sample in range(4):
        _,av=_arm_segments(left['body'][n,sample]);_,bv=_arm_segments(right['body'][n,sample])
        if (av&bv).sum()<2:return result
        phase,basis=phase_tail(left['body'][n,sample],right['body'][n,sample])
        if phase is None:return result
        a,b=left['projected'][n,sample],right['projected'][n,sample]
        valid=lambda p:(p[:,2]>=.3)&(p[:,:2]>=0).all(1)&(p[:,:2]<=1).all(1)
        shared=valid(left['body'][n,sample])&valid(right['body'][n,sample])
        shared[:2]=False;shared[6:]=False
        if shared.sum()<2:return result
        phases.append(phase);bases.append(basis)
        screens.append(float(np.linalg.norm((a[shared,:2]-b[shared,:2])*[16/9,1.],axis=1).max()))
    for side in (0,1):
        ha,hb=left['heads'][n,side],right['heads'][n,side]
        if min(ha[2],hb[2])<=.019:return result
        ca,cb=Camera(*left['cameras'][n,side]),Camera(*right['cameras'][n,side])
        pa,pb=ca.project(ha[:2]),cb.project(hb[:2])
        if np.any(pa<0) or np.any(pa>1) or np.any(pb<0) or np.any(pb>1):return result
        faces.append(float(np.linalg.norm((pa-pb)*[1.,9/16]))/.06
                     +abs(math.log(float(ha[2]*ca.zoom/(hb[2]*cb.zoom))))/.25)
        va,vb=left['vectors'][n,side],right['vectors'][n,side]
        if min(np.linalg.norm(va),np.linalg.norm(vb))<.5:return result
        cosines.append(float(np.clip(va@vb,-1.,1.)))
    direction=measurement['maximum_direction_disagreement']
    velocity=measurement['synchronous_worst_window_loss']
    identity=5.*max(0.,.65-min(cosines))
    crop=max(left['utility_parts']['crop_clipping'][n,0],right['utility_parts']['crop_clipping'][n,1])
    speed=max((v for w in measurement['synchronous_motion_windows'] for v in w['speed_losses']),default=0.)
    residuals=[max(faces),identity,3.*max(phases),1.5*velocity,.6*max(screens)/.12,.3*speed,crop]
    fit=min(1./(1.+r) for r in residuals)*max(0.,1.-2.*direction)
    risk=sum(residuals[:5])+measurement['boundary_native_readability_cost']
    opportunity=dict(base,joint_fit=float(fit),credit=float(OPPORTUNITY_CAP*music_strength*fit),
        reason='observed_same_clock_partial_joint_connection',
        phase_tail_by_sample=phases,phase_basis_by_sample=bases,screen_tail_by_sample=screens,
        face_geometry_by_side=faces,appearance_by_side=cosines,normalized_residuals=residuals)
    return dict(connection_opportunity=opportunity,nonnegative_risk_cost=float(risk),
                paired_clock_supported=True)
