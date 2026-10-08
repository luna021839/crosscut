"""Directional, phase-aware seam costs over actual owned-body observations.

Local video-phase proposals have explicit intended output ticks; their raw
source/native addresses are never rewritten. This scorer does not retime a
renderer, grant clock authority, certify actors, or substitute face animation
for choreography. Missing evidence is INCONCLUSIVE, not a perfect match.
"""
import math
import numpy as np

from .group_review_measured_features import local_motion, pair_risk
from .group_review_native_window import cosine


def _shape(view):
    d=None if view is None else view.get('descriptor')
    return None if d is None or d.pose_shape is None else {p[0]:np.asarray(p[1:],float) for p in d.pose_shape}


def review_actor_association_valid(view):
    """Same observed/unresolved boundary for original and candidate actors."""
    return bool(view is not None and view.get('descriptor') is not None
        and not view.get('reacquisition_ambiguous',False)
        and view.get('selected_actor_association_state') not in ('AMBIGUOUS','UNMATCHED','UNKNOWN'))


def _window_links(sequence):
    """Check every sampled edge; equal endpoints do not establish a path.

    These are anonymous appearance hypotheses, not identity certificates.
    Missing observations remain unresolved. A changed clock/native address is
    a contract error; a camera/gap/appearance bridge is unmeasured evidence.
    """
    if any(not review_actor_association_valid(v) for v in sequence):
        return False
    for left,right in zip(sequence,sequence[1:]):
        a,b=left['row'],right['row']
        ra,rb=a['native_observation'],b['native_observation']
        if (ra['request_source_index']!=rb['request_source_index']
                or ra['source_clock_identity']!=rb['source_clock_identity']
                or ra['source_file_identity']!=rb['source_file_identity']
                or ra['native_frame_index']>rb['native_frame_index']):
            raise ValueError('motion seam window changed source/native/clock identity')
        if ra['native_frame_index']==rb['native_frame_index'] and ra!=rb:
            raise ValueError('motion seam native alias has conflicting recorded rows')
        if (ra['native_frame_index']==rb['native_frame_index']
                and left['descriptor'].frame_local_head_id!=right['descriptor'].frame_local_head_id):
            return False
        native_a,native_b=left.get('native_camera_unit_index'),right.get('native_camera_unit_index')
        same_camera=(native_a==native_b if type(native_a) is int and type(native_b) is int
            else native_a is None and native_b is None
            and a['projection']['camera_sha256']==b['projection']['camera_sha256'])
        if (not same_camera
                or rb['native_frame_index']-ra['native_frame_index']>12):
            return False
        appearance=cosine(left['vector'],right['vector'])
        if appearance is None or appearance<.25:
            return False
    return True


def review_window_connected(sequence, *, require_native_motion=False):
    """Explicit anonymous association check, not an aesthetic or identity badge.

    Native aliases can preserve an occurrence but cannot create a second
    temporal observation. Background inventory truncation alone is not a
    selected-actor ambiguity signal.
    """
    return bool(_window_links(sequence) and (not require_native_motion or len({
        v['row']['native_observation']['native_frame_index'] for v in sequence})>=2))


def _velocity(shapes,ticks,common,fps,*,outgoing):
    # A turn-and-return trajectory can have equal whole-window endpoints.
    # The cut tangent uses the nearest two actual observations, not a
    # whole-shot endpoint displacement. Separate aligned-window comparison
    # consumes every interior observation; tangent alone is not a certificate.
    selected=slice(-2,None) if outgoing else slice(0,2)
    y=np.asarray([[shape[k] for k in common] for shape in shapes[selected]])
    x=np.asarray(ticks[selected],float)/fps
    x=x-x.mean()
    return np.sum(x[:,None,None]*(y-y.mean(axis=0)),axis=0)/float(x@x)


def score_review_motion_seam(outgoing,incoming,*,fps,incoming_output_ticks=None,
        maximum_phase_frames=2):
    """Score one *selected* actor sequence on each side, not independent heads.

    For an unchanged clock, incoming_output_ticks equals the measured ticks.
    For a declared phase proposal, it names the proposed common-output grid;
    stored observations still retain their actual source/native/time addresses.
    One missing or reversed window cannot win by dropping adverse evidence.
    """
    if (not math.isfinite(fps) or fps <= 0 or not 2 <= len(outgoing) <= 12
            or not 2 <= len(incoming) <= 12 or type(maximum_phase_frames) is not int
            or not 0 <= maximum_phase_frames <= 6):
        raise ValueError('motion seam needs two explicitly bounded temporal windows')
    def ticks(sequence):
        values=tuple(v['row']['output_tick'] for v in sequence if v is not None)
        if len(values)!=len(sequence):return None
        if any(type(t) is not int for t in values) or any(a>=b for a,b in zip(values,values[1:])):
            raise ValueError('motion seam observation window changed output order')
        return values
    left_ticks,right_ticks=ticks(outgoing),ticks(incoming)
    proposed=right_ticks if incoming_output_ticks is None else tuple(incoming_output_ticks)
    if (proposed is not None and (len(proposed)!=len(incoming)
            or any(type(t) is not int for t in proposed)
            or any(a>=b for a,b in zip(proposed,proposed[1:])))):
        raise ValueError('phase proposal changed its exact output window')
    if left_ticks is not None and proposed is not None and left_ticks[-1]>=proposed[0]:
        raise ValueError('motion seam did not join outgoing to incoming output time')
    if proposed is not None and right_ticks is not None and len({a-b for a,b in zip(proposed,right_ticks)})!=1:
        raise ValueError('motion seam supports one explicit constant video phase, not an implicit tempo change')
    if proposed is not None and right_ticks is not None and abs(proposed[0]-right_ticks[0])>maximum_phase_frames:
        raise ValueError('motion seam constant phase exceeds its declared frame bound')
    connected_left,connected_right=_window_links(outgoing),_window_links(incoming)
    ml=None if left_ticks is None else local_motion(outgoing[0],outgoing[-1],fps)
    mr=None if right_ticks is None else local_motion(incoming[0],incoming[-1],fps)
    base=pair_risk(outgoing[-1],incoming[0],ml,mr)
    known=(ml is not None and mr is not None
        and ml['basis']==mr['basis']=='owned_upper_body_shape_velocity'
        and ml['indices']==mr['indices'] and proposed is not None
        and left_ticks is not None and right_ticks is not None
        and connected_left and connected_right)
    result=dict(contract='recorded_owned_body_motion_seam_review_v1',state='INCONCLUSIVE',
        pose_phase_error=None,direction_error=None,velocity_error=None,exploration_cost=base['risk']+.8,
        endpoint_metrics=base,requested_outgoing_ticks=left_ticks,measured_incoming_ticks=right_ticks,
        proposed_incoming_output_ticks=proposed,raw_observation_addresses_rewritten=False,
        all_sampled_window_edges_connected=connected_left and connected_right,
        anonymous_appearance_paths_are_actor_certificates=False,
        renderer_clock_authority_granted=False,named_actor_certified=False,quality_approved=False)
    shapes=[_shape(v) for v in (*outgoing,*incoming)]
    if any(s is None for s in shapes):return result
    common=tuple(sorted(set.intersection(*(set(s) for s in shapes))))
    if len(common)<4:return result
    observed=set.union(*(set(s) for s in shapes))
    if set(common)!=observed:
        # Dropping an observed, possibly adverse wrist from the comparison
        # cannot manufacture a low-error match over the remaining shoulders.
        # Consistently observed partial-body windows are still allowed; this
        # condition concerns *lost evidence*, not a mandatory full-body shot.
        return dict(result,observed_joint_indices=list(common),
            unpaired_observed_joint_indices=sorted(observed-set(common)),
            evidence_loss_prevents_body_phase_comparison=True)
    if not known:return result
    # Both windows cover the exact requested body observations. Only the
    # constant phase mapping is proposed; no interior observation is dropped.
    lb,ra=shapes[len(outgoing)-1],shapes[len(outgoing)]
    lv=_velocity(shapes[:len(outgoing)],left_ticks,common,fps,outgoing=True)
    rv=_velocity(shapes[len(outgoing):],proposed,common,fps,outgoing=False)
    dt=(proposed[0]-left_ticks[-1])/fps
    phase=float(np.linalg.norm(np.asarray([ra[k]-lb[k] for k in common])-lv*dt,axis=1).mean())
    nl,nr=float(np.linalg.norm(lv)),float(np.linalg.norm(rv))
    direction=(1.-float(np.sum(lv*rv)/(nl*nr)))/2 if nl>1e-8 and nr>1e-8 else None
    velocity=float(np.linalg.norm(lv-rv)/(1.+max(nl,nr)))
    known_direction=direction is not None
    return dict(result,state='OBSERVED_BODY_PHASE_HYPOTHESIS',pose_phase_error=phase,
        direction_error=direction,velocity_error=velocity,
        exploration_cost=base['risk']+2.*phase+velocity+(.5 if direction is None else direction),
        observed_joint_indices=list(common),both_moving_directions_observed=known_direction,
        phase_proposal_requires_actual_renderer_recompilation=proposed!=right_ticks,
        timewarped_lipsync_verified=False)


def compare_aligned_review_motion_windows(left,right,*,fps,requested_output_ticks,
        requested_joint_indices):
    """Compare two source-local paths at the SAME declared musical instants.

    Interior positions and every adjacent velocity are consumed. Acceleration
    or speed is not itself penalized: identical fast choreography has zero
    discrepancy. The caller freezes joints/ticks before comparing candidates;
    a lost wrist cannot shrink the comparison to the remaining shoulders.
    """
    ticks=tuple(requested_output_ticks);joints=tuple(requested_joint_indices)
    if (not 2<=len(ticks)<=16 or len(left)!=len(ticks) or len(right)!=len(ticks)
            or any(type(t) is not int for t in ticks)
            or any(a>=b for a,b in zip(ticks,ticks[1:]))
            or tuple(sorted(set(joints)))!=joints or not 4<=len(joints)<=6
            or any(type(k) is not int or not 0<=k<6 for k in joints)
            or not math.isfinite(fps) or fps<=0):
        raise ValueError('aligned motion needs fixed bounded output and joint requests')
    result=dict(contract='same_output_clock_actor_body_trajectory_review_v1',
        state='INCONCLUSIVE',requested_output_ticks=ticks,requested_joint_indices=joints,
        trajectory_error=None,velocity_error=None,direction_error=None,
        exploration_cost=None,quality_approved=False,motion_certified=False,
        fast_motion_is_intrinsically_bad=False,raw_observation_addresses_rewritten=False)
    for sequence in (left,right):
        if any(v is None or v.get('descriptor') is None for v in sequence):return result
        if tuple(v['row']['output_tick'] for v in sequence)!=ticks:
            raise ValueError('aligned motion window borrowed a different output request')
        if not _window_links(sequence):return result
    shapes=[[_shape(v) for v in sequence] for sequence in (left,right)]
    missing=tuple((side,i,k) for side,sequence in enumerate(shapes)
        for i,shape in enumerate(sequence) for k in joints if shape is None or k not in shape)
    if missing:
        return dict(result,missing_requested_joint_observations=missing,
            evidence_loss_prevents_body_phase_comparison=True)
    a,b=[np.asarray([[s[k] for k in joints] for s in sequence]) for sequence in shapes]
    dt=np.diff(np.asarray(ticks,float))/fps
    av,bv=np.diff(a,axis=0)/dt[:,None,None],np.diff(b,axis=0)/dt[:,None,None]
    trajectory=float(np.linalg.norm(a-b,axis=2).mean())
    an,bn=np.linalg.norm(av,axis=(1,2)),np.linalg.norm(bv,axis=(1,2))
    velocity=float(np.mean(np.linalg.norm(av-bv,axis=(1,2))/(1.+np.maximum(an,bn))))
    moving=(an>1e-8)&(bn>1e-8)
    direction=(float(np.mean((1.-np.clip(np.sum(av[moving]*bv[moving],axis=(1,2))
        /(an[moving]*bn[moving]),-1.,1.))/2)) if moving.any() else None)
    return dict(result,state='OBSERVED_ALIGNED_BODY_TRAJECTORY_HYPOTHESIS',
        trajectory_error=trajectory,velocity_error=velocity,direction_error=direction,
        exploration_cost=2.*trajectory+velocity+(0. if direction is None else direction),
        adjacent_velocity_observations=len(dt),interior_pose_observations=len(ticks),
        direction_unobserved=direction is None)


def _head_orientation(view):
    if view is None or view.get('descriptor') is None:return None
    head_id=view['descriptor'].frame_local_head_id
    head=next((h for h in view['row']['projection']['heads'] if h['observation_id']==head_id),None)
    if head is None or len(head['projected_landmarks_xy'])<3:return None
    # Eyes and nose express head orientation. Mouth deformation is NOT an
    # observed dance movement and must not purchase a choreography reward.
    points=np.asarray(head['projected_landmarks_xy'][:3],float)*np.asarray((1.,9/16))
    if points.shape!=(3,2) or not np.isfinite(points).all():return None
    eyes=points[1]-points[0];span=float(np.linalg.norm(eyes))
    if span<1e-6:return None
    return np.concatenate((eyes/span,(points[2]-points[:2].mean(axis=0))/span))


def compare_aligned_review_head_windows(left,right,*,fps,requested_output_ticks):
    """Independent measured head-pose connection when arms are unobserved.

    This is not a fallback body certificate. A caller with known body duties
    must still compare those duties; losing a bad wrist cannot select HEAD
    instead. Requested ticks and selected anonymous occurrences stay fixed.
    """
    ticks=tuple(requested_output_ticks)
    if (not 2<=len(ticks)<=16 or len(left)!=len(ticks) or len(right)!=len(ticks)
            or any(type(t) is not int for t in ticks) or any(a>=b for a,b in zip(ticks,ticks[1:]))
            or not math.isfinite(fps) or fps<=0):
        raise ValueError('head path needs an exact finite common output grid')
    result=dict(contract='same_output_clock_selected_head_orientation_review_v1',
        state='INCONCLUSIVE',requested_output_ticks=ticks,trajectory_error=None,velocity_error=None,
        exploration_cost=None,choreography_motion_observed=False,
        body_motion_certified=False,quality_approved=False,mouth_points_used=False)
    for sequence in (left,right):
        if any(v is None for v in sequence):return result
        if tuple(v['row']['output_tick'] for v in sequence)!=ticks:
            raise ValueError('head path changed its frozen output grid')
        if not _window_links(sequence):return result
    signatures=[[_head_orientation(v) for v in sequence] for sequence in (left,right)]
    if any(v is None for values in signatures for v in values):return result
    a,b=[np.asarray(values) for values in signatures]
    dt=np.diff(np.asarray(ticks,float))/fps
    av,bv=np.diff(a,axis=0)/dt[:,None],np.diff(b,axis=0)/dt[:,None]
    trajectory=float(np.linalg.norm(a-b,axis=1).mean())
    velocity=float(np.mean(np.linalg.norm(av-bv,axis=1)/(1.+np.maximum(
        np.linalg.norm(av,axis=1),np.linalg.norm(bv,axis=1)))))
    return dict(result,state='OBSERVED_HEAD_ORIENTATION_PATH_HYPOTHESIS',
        trajectory_error=trajectory,velocity_error=velocity,
        exploration_cost=2.*trajectory+velocity,interior_head_observations=len(ticks))


def score_review_head_seam(outgoing,incoming,*,fps):
    """Actual output head-pose tangents, distinct from body choreography.

    No phase is proposed here: both flanks already use the selected renderer
    mapping. A body-required consumer cannot use this to erase a lost wrist.
    """
    if (not 2<=len(outgoing)<=12 or not 2<=len(incoming)<=12
            or not math.isfinite(fps) or fps<=0):
        raise ValueError('head seam needs two bounded actual output windows')
    result=dict(contract='actual_output_selected_head_seam_review_v1',state='INCONCLUSIVE',
        exploration_cost=None,pose_phase_error=None,velocity_error=None,direction_error=None,
        body_motion_certified=False,choreography_motion_observed=False,mouth_points_used=False,
        quality_approved=False)
    if not _window_links(outgoing) or not _window_links(incoming):return result
    left_ticks=tuple(v['row']['output_tick'] for v in outgoing)
    right_ticks=tuple(v['row']['output_tick'] for v in incoming)
    if (any(type(t) is not int for t in (*left_ticks,*right_ticks))
            or any(a>=b for a,b in zip(left_ticks,left_ticks[1:]))
            or any(a>=b for a,b in zip(right_ticks,right_ticks[1:]))
            or left_ticks[-1]>=right_ticks[0]):
        raise ValueError('head seam changed its actual ordered output windows')
    signatures=[_head_orientation(v) for v in (*outgoing,*incoming)]
    if any(v is None for v in signatures):return result
    a,b=np.asarray(signatures[:len(outgoing)]),np.asarray(signatures[len(outgoing):])
    lv=(a[-1]-a[-2])*fps/(left_ticks[-1]-left_ticks[-2])
    rv=(b[1]-b[0])*fps/(right_ticks[1]-right_ticks[0])
    phase=float(np.linalg.norm(b[0]-a[-1]-lv*(right_ticks[0]-left_ticks[-1])/fps))
    nl,nr=float(np.linalg.norm(lv)),float(np.linalg.norm(rv))
    velocity=float(np.linalg.norm(lv-rv)/(1.+max(nl,nr)))
    direction=(float((1.-np.clip(lv@rv/(nl*nr),-1.,1.))/2) if nl>1e-8 and nr>1e-8 else None)
    return dict(result,state='OBSERVED_OUTPUT_HEAD_POSE_SEAM_HYPOTHESIS',
        pose_phase_error=phase,velocity_error=velocity,direction_error=direction,
        exploration_cost=2.*phase+velocity+(0. if direction is None else direction),
        requested_outgoing_ticks=left_ticks,requested_incoming_ticks=right_ticks)
