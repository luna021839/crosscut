"""Evaluate REAL decoded camera curves on original observed label clocks.

No teacher-forced count, replacement interpolator, synthetic target points or
production-quality claim. The renderer track owns interpolation and endpoints.
"""
from __future__ import annotations

import numpy as np


DECODED_OBSERVATION_METRIC_CONTRACT='actual_decoded_source_count_time_renderer_on_original_observations_v1'
COORDINATE_NAMES=('zoom','center_x','center_y','roll_degrees')
RANGES=np.array([.18,.04,.04,1.60])
STATIC_PRIOR=np.array([1.02,.5,.5,0.])
RANGES.setflags(write=False);STATIC_PRIOR.setflags(write=False)


def decoded_observation_error(decoded,observed_keyframes,*,expected_source_id):
    if not decoded.eligible or decoded.track is None:
        raise ValueError('decoded renderer track is not eligible')
    if decoded.source_id!=expected_source_id:
        raise ValueError('decoded source identity mismatch')
    observed=tuple(observed_keyframes)
    if not observed:raise ValueError('original observed keyframes are required')
    times=np.array([point.time_seconds for point in observed],float)
    duration=decoded.shot_duration_seconds
    if (duration is None or not np.isfinite(duration) or duration<=0 or not np.isfinite(times).all()
            or times[0]!=0 or np.any(np.diff(times)<=0) or times[-1]>duration+1e-6):
        raise ValueError('observed clock is outside decoded shot duration')
    target=np.array([[getattr(point,name) for name in COORDINATE_NAMES] for point in observed],float)
    realized=[decoded.track.evaluate(float(t)) for t in times]
    predicted=np.array([[getattr(point,name) for name in COORDINATE_NAMES] for point in realized],float)
    if not np.isfinite(target).all() or not np.isfinite(predicted).all():
        raise ValueError('finite original and realized camera values required')
    return dict(predicted=predicted,target=target,
        normalized_absolute_error=np.abs((predicted-target)/RANGES),
        static_prior_normalized_absolute_error=np.abs((STATIC_PRIOR-target)/RANGES))


def summarize_decoded_observations(rows,*,expected_tracks,source_mismatches):
    """JSON-safe score and health from only realized, applied trajectories."""
    complete=len(rows)==expected_tracks and expected_tracks>0 and source_mismatches==0
    if not rows:
        return dict(contract=DECODED_OBSERVATION_METRIC_CONTRACT,complete=False,
            expected_tracks=expected_tracks,evaluated_tracks=0,source_identity_mismatch_count=source_mismatches,
            coordinates={},curve_delta_and_range={},normalized_mae=None,static_prior_normalized_mae=None,
            improvement_over_static_prior=None,observed_keyframe_count=0)
    predicted=np.concatenate([row['predicted'] for row in rows])
    target=np.concatenate([row['target'] for row in rows])
    errors=np.concatenate([row['normalized_absolute_error'] for row in rows])
    prior_errors=np.concatenate([row['static_prior_normalized_absolute_error'] for row in rows])
    physical=np.abs(predicted-target);prior_physical=np.abs(STATIC_PRIOR-target)
    coordinates={};curves={}
    for i,name in enumerate(COORDINATE_NAMES):
        mae=float(physical[:,i].mean());baseline=float(prior_physical[:,i].mean())
        normalized=float(errors[:,i].mean());prior_normalized=float(prior_errors[:,i].mean())
        coordinates[name]=dict(label_count=len(target),mean_absolute_error=mae,
            p95_absolute_error=float(np.quantile(physical[:,i],.95)),normalized_mean_absolute_error=normalized,
            normalized_p95_absolute_error=float(np.quantile(errors[:,i],.95)),
            static_prior_mean_absolute_error=baseline,static_prior_normalized_mean_absolute_error=prior_normalized,
            mean_absolute_error_improvement_over_static_prior=baseline-mae,
            normalized_improvement_over_static_prior=prior_normalized-normalized)
        pr=np.array([np.ptp(row['predicted'][:,i]) for row in rows])
        tr=np.array([np.ptp(row['target'][:,i]) for row in rows])
        ps=[np.diff(row['predicted'][:,i]) for row in rows if len(row['predicted'])>1]
        ts=[np.diff(row['target'][:,i]) for row in rows if len(row['target'])>1]
        psteps=np.concatenate(ps) if ps else np.empty(0)
        tsteps=np.concatenate(ts) if ts else np.empty(0)
        delta=float(np.abs(psteps-tsteps).mean()) if len(psteps) else None
        pstep=float(np.abs(psteps).mean()) if len(psteps) else None
        tstep=float(np.abs(tsteps).mean()) if len(tsteps) else None
        curves[name]=dict(delta_label_count=len(psteps),delta_mean_absolute_error=delta,
            normalized_delta_mean_absolute_error=None if delta is None else delta/RANGES[i],
            predicted_mean_absolute_step=pstep,target_mean_absolute_step=tstep,
            predicted_normalized_mean_absolute_step=None if pstep is None else pstep/RANGES[i],
            target_normalized_mean_absolute_step=None if tstep is None else tstep/RANGES[i],
            predicted_mean_range=float(pr.mean()),target_mean_range=float(tr.mean()),
            mean_absolute_range_error=float(np.abs(pr-tr).mean()),
            predicted_normalized_mean_range=float(pr.mean()/RANGES[i]),target_normalized_mean_range=float(tr.mean()/RANGES[i]))
    mae=float(errors.mean());baseline=float(prior_errors.mean())
    return dict(contract=DECODED_OBSERVATION_METRIC_CONTRACT,complete=complete,
        expected_tracks=expected_tracks,evaluated_tracks=len(rows),source_identity_mismatch_count=source_mismatches,
        observed_keyframe_count=len(target),coordinates=coordinates,curve_delta_and_range=curves,
        normalized_mae=mae,static_prior_normalized_mae=baseline,improvement_over_static_prior=baseline-mae,
        conditional_on_target_source_and_duration=True,predicted_count_and_time_consumed=True,
        interpolation_owner='EditorialTransformTrack.evaluate',synthetic_target_points_used=False)
