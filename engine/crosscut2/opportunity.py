"""Observed music/action opportunities, not identity or quality certificates.

The .8 event cap is the existing moving-match cap, frozen before a media trial.
Missing/contrary action removes only this credit, never the source or the cut.
"""
import math
import numpy as np
from .action_geometry import normalized_body, _arm_segments

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
