"""Native-event-frozen common-time motion routes for selected review actors.

The route is chosen from sampler/native-camera identities BEFORE pose errors
are inspected. Losing a requested wrist or an intermediate actor does not
turn a body comparison into a head comparison. This is a review hypothesis,
not named identity, normal app admission or expert-quality certification.
"""
from dataclasses import dataclass
import json
import math
import numpy as np

from .group_performer_family_owner import digest
from .group_review_motion_seam import (
    review_window_connected,compare_aligned_review_head_windows)


@dataclass(frozen=True,slots=True)
class FrozenReviewMotionIntent:
    """Original actor and time-specific part requests, independent of donors."""
    token_json: str

    def __post_init__(self):
        value=json.loads(self.token_json)
        ticks=value.get('ticks',[]);requests=value.get('joint_requests',[])
        if (value.get('contract')!='frozen_selected_original_motion_intent_v2'
                or value.get('kind') not in ('BODY','HEAD_ONLY','UNKNOWN')
                or not 2<=len(ticks)<=16 or len(requests)!=len(ticks)
                or any(type(t) is not int or t<0 for t in ticks)
                or any(a>=b for a,b in zip(ticks,ticks[1:]))
                or not isinstance(value.get('original_actor_identity'),str)
                or any(j is not None and (not isinstance(j,list)
                    or j!=sorted(set(j)) or any(type(k) is not int or not 0<=k<6 for k in j))
                    for j in requests)):
            raise ValueError('motion intent needs exact selected original actor/time/part requests')

    @property
    def payload(self):return json.loads(self.token_json)

    @property
    def identity(self):return digest(self.payload)


def freeze_selected_review_motion_intent(reference, *, requested_ticks,original_actor_identity,
        head_only_declared=False,reference_native_units=None):
    """A known body joint stays requested when another original pose is lost.

    Observed visibility is per tick, NOT a whole-arm lifetime requirement.
    Unknown original slots remain unknown rather than shrinking an intersection.
    HEAD_ONLY requires an independently declared original intent and no known
    body-action observation in this window.
    """
    ticks=tuple(requested_ticks)
    if len(reference)!=len(ticks) or type(head_only_declared) is not bool:
        raise ValueError('original motion intent changed its selected request inventory')
    units=tuple(('same-original-shot',) for _ in ticks) if reference_native_units is None else tuple(reference_native_units)
    if len(units)!=len(ticks):raise ValueError('original motion intent changed its native camera unit inventory')
    requests=[];known=set();missing=[]
    for tick,view in zip(ticks,reference):
        if view is not None and view['row']['output_tick']!=tick:
            raise ValueError('original actor motion request borrowed another tick')
        shape=None if view is None or view.get('descriptor') is None else view['descriptor'].pose_shape
        indices=None if shape is None else sorted({p[0] for p in shape})
        requests.append(indices)
        if indices is not None:known.update(indices)
        if view is None:missing.append(tick)
    body=len(known)>=4
    kind='BODY' if body else 'HEAD_ONLY' if head_only_declared and not missing else 'UNKNOWN'
    if body:
        # Preserve the explicit unknown slot. It cannot certify a head-only
        # match, nor falsely assert a wrist was visible for a whole epoch.
        by_unit={u:set() for u in units}
        for unit,indices in zip(units,requests):
            if indices is not None:by_unit[unit].update(indices)
        # A body seen AFTER a different original shot cannot manufacture a
        # visible wrist in an earlier genuine head-only shot. Detector loss
        # WITHIN a known body shot still retains that shot's known requests.
        requests=[sorted(by_unit[u]) if j is None else j for u,j in zip(units,requests)]
    token=dict(contract='frozen_selected_original_motion_intent_v2',ticks=list(ticks),
        original_actor_identity=original_actor_identity,kind=kind,joint_requests=requests,
        known_joint_union=sorted(known),original_unmatched_ticks=missing,
        original_native_unit_keys=[list(u) for u in units],
        inferred_whole_arm_visibility_asserted=False,quality_approved=False)
    return FrozenReviewMotionIntent(json.dumps(token,sort_keys=True,allow_nan=False))


def same_native_review_continuation(outgoing,incoming,*,expected_native_addresses):
    """Actual sampler continuity is not a new cross-performance phase match.

    Pose detection can fail in one frame without interrupting the source's
    native sequence. Selected head association and exact native addresses are
    still required. Camera/framing jumps and content preservation are separate
    consumers; this does not certify them or call continuity an improvement.
    """
    sequence=tuple(outgoing)+tuple(incoming)
    result=dict(contract='exact_source_native_review_continuation_v1',continuity_supported=False,
        new_cross_performance_motion_match=False,motion_improvement_claimed=False,quality_approved=False)
    if len(outgoing)<2 or len(incoming)<2 or any(v is None for v in sequence):return result
    if len(expected_native_addresses)!=len(sequence):raise ValueError('native continuation request inventory changed')
    # Editorial camera metadata may differ while the actual source sequence
    # is unchanged. Do not claim camera continuity from native continuity.
    physical=[];views=[]
    for v,expected in zip(sequence,expected_native_addresses):
        raw=v['row']['native_observation']
        physical.append((raw['request_source_index'],raw['source_clock_identity'],
            tuple(raw['source_file_identity']),raw['native_frame_index']))
        if tuple(expected)!=physical[-1]:return dict(result,actual_sampler_does_not_follow_frozen_native_sequence=True)
        copy=dict(v,row=dict(v['row'],projection=dict(v['row']['projection'],camera_sha256='source-sequence-only')))
        views.append(copy)
    if len({p[:3] for p in physical})!=1:return result
    if not review_window_connected(views,require_native_motion=True):return result
    return dict(result,continuity_supported=True,actual_native_addresses=[list(p) for p in physical],
        editorial_camera_continuity_certified=False)


def freeze_review_native_motion_route(*,ticks,boundary_tick,left_native,right_native):
    """Choose FULL/PAST/FUTURE from actual native units, never a low-error slice.

    Each signature is (source, clock, source-file token, native camera unit,
    native frame). PAST requires the incoming native shot to continue across
    the cut; FUTURE requires the outgoing shot to do so. Temporal aliases alone
    do not support either body/head action velocity.
    """
    ticks=tuple(ticks)
    if (not 4<=len(ticks)<=16 or len(left_native)!=len(ticks) or len(right_native)!=len(ticks)
            or any(type(t) is not int for t in ticks) or any(a>=b for a,b in zip(ticks,ticks[1:]))
            or type(boundary_tick) is not int or boundary_tick not in ticks):
        raise ValueError('native motion route needs two exact bounded sampler inventories')
    past=tuple(i for i,t in enumerate(ticks) if t<boundary_tick)
    future=tuple(i for i,t in enumerate(ticks) if t>=boundary_tick)
    def continuous(values,indices):
        rows=[values[i] for i in indices]
        if len(rows)<2 or any(r is None or len(r)!=5 for r in rows):return False
        return (len({tuple(r[:4]) for r in rows})==1
            and all(type(r[4]) is int for r in rows)
            and all(0<=b[4]-a[4]<=12 for a,b in zip(rows,rows[1:]))
            and len({r[4] for r in rows})>=2)
    all_indices=tuple(range(len(ticks)))
    if continuous(left_native,all_indices) and continuous(right_native,all_indices):
        route='FULL_ALIGNED';indices=all_indices;continuation=None
    elif continuous(left_native,past) and continuous(right_native,all_indices):
        route='PAST_COMMON_INCOMING_CONTINUES';indices=past;continuation='right'
    elif continuous(right_native,future) and continuous(left_native,all_indices):
        route='FUTURE_COMMON_OUTGOING_CONTINUES';indices=future;continuation='left'
    else:route='ACTUAL_ONLY_UNRESOLVED';indices=();continuation=None
    return dict(contract='native_event_frozen_motion_route_v1',route=route,
        boundary_tick=boundary_tick,requested_ticks=list(ticks),comparison_indices=list(indices),
        continuation_side=continuation,continuation_indices=list(all_indices) if continuation else [],
        left_native=[None if r is None else list(r) for r in left_native],
        right_native=[None if r is None else list(r) for r in right_native],
        route_selected_from_pose_errors=False,quality_approved=False)


def compare_frozen_review_motion_route(left,right, *, fps,intent,route):
    """Consume exactly the frozen route and joint requests; no head fallback.

    Every requested visible joint contributes position. Adjacent velocity
    uses only the joints requested at BOTH instants, not joints a candidate
    happened to retain. Identical natural fast turns have zero discrepancy.
    """
    if not isinstance(intent,FrozenReviewMotionIntent) or not math.isfinite(fps) or fps<=0:
        raise ValueError('motion route needs a frozen original intent and finite clock')
    token=intent.payload;ticks=tuple(token['ticks'])
    if (len(left)!=len(ticks) or len(right)!=len(ticks)
            or route.get('contract')!='native_event_frozen_motion_route_v1'
            or tuple(route.get('requested_ticks',()))!=ticks):
        raise ValueError('motion consumer changed its frozen route/request inventory')
    result=dict(contract='selected_native_common_time_motion_evidence_v2',state='INCONCLUSIVE',
        route=route['route'],intent_identity=intent.identity,intent_kind=token['kind'],
        exploration_cost=None,trajectory_error=None,velocity_error=None,direction_error=None,
        same_time_action_supported=False,body_motion_certified=False,quality_approved=False,
        actual_cut_tangent_penalizes_natural_turn=False)
    indices=tuple(route['comparison_indices'])
    if len(indices)<2 or token['kind']=='UNKNOWN':return result
    for side,values in (('left',left),('right',right)):
        sequence=[values[i] for i in indices]
        if any(v is not None and v['row']['output_tick']!=ticks[i] for i,v in zip(indices,sequence)):
            raise ValueError('motion route borrowed another output instant')
        if not review_window_connected(sequence,require_native_motion=True):
            return dict(result,unresolved_actor_or_native_window=side)
    continuation=route['continuation_side']
    if continuation:
        values=left if continuation=='left' else right
        if not review_window_connected([values[i] for i in route['continuation_indices']],
                require_native_motion=True):
            return dict(result,unresolved_crossing_continuation=continuation)
    if token['kind']=='HEAD_ONLY':
        measured=compare_aligned_review_head_windows([left[i] for i in indices],
            [right[i] for i in indices],fps=fps,requested_output_ticks=[ticks[i] for i in indices])
        if measured['state']!='OBSERVED_HEAD_ORIENTATION_PATH_HYPOTHESIS':return result
        return dict(result,**{k:measured[k] for k in ('exploration_cost','trajectory_error','velocity_error')},
            state='OBSERVED_NATIVE_COMMON_HEAD_ACTION_HYPOTHESIS',same_time_action_supported=True,
            choreography_motion_observed=False)
    # Common-time comparison and actual output coverage are separate duties.
    # A PAST route may ignore the non-output left future camera cut, but it
    # may not ignore disappearing incoming wrists in that actual future.
    actual_missing=[]
    for side,values in enumerate((left,right)):
        for i,tick in enumerate(ticks):
            if (side==0)!=(tick<route['boundary_tick']):continue
            view=values[i]
            if view is None:
                actual_missing.append((side,tick,'actual_output_actor_unobserved'));continue
            if view['row']['output_tick']!=tick:
                raise ValueError('actual output body coverage borrowed another instant')
            shape=view['descriptor'].pose_shape
            observed=set() if shape is None else {p[0] for p in shape}
            requested=token['joint_requests'][i]
            if requested is None:
                actual_missing.append((side,tick,'original_joint_request_unresolved'))
            else:actual_missing.extend((side,tick,k) for k in requested if k not in observed)
    if actual_missing:
        return dict(result,missing_actual_output_joint_requests=actual_missing,
            evidence_loss_prevents_head_fallback=True)
    shapes=[];missing=[]
    for side,values in enumerate((left,right)):
        side_shapes={}
        for i in indices:
            shape=values[i]['descriptor'].pose_shape
            points={} if shape is None else {p[0]:np.asarray(p[1:],float) for p in shape}
            requested=token['joint_requests'][i]
            if requested is None:missing.append((side,ticks[i],'original_joint_request_unresolved'));continue
            missing.extend((side,ticks[i],k) for k in requested if k not in points)
            side_shapes[i]=points
        shapes.append(side_shapes)
    if missing:return dict(result,missing_frozen_joint_requests=missing,evidence_loss_prevents_head_fallback=True)
    positions=[];velocities=[];directions=[]
    for i in indices:
        positions.extend(float(np.linalg.norm(shapes[0][i][k]-shapes[1][i][k]))
            for k in token['joint_requests'][i])
    for a,b in zip(indices,indices[1:]):
        joints=tuple(sorted(set(token['joint_requests'][a])&set(token['joint_requests'][b])))
        if len(joints)<4:continue
        dt=(ticks[b]-ticks[a])/fps
        av,bv=[np.asarray([(s[b][k]-s[a][k])/dt for k in joints]) for s in shapes]
        an,bn=float(np.linalg.norm(av)),float(np.linalg.norm(bv))
        velocities.append(float(np.linalg.norm(av-bv)/(1.+max(an,bn))))
        if an>1e-8 and bn>1e-8:directions.append(float((1.-np.clip(np.sum(av*bv)/(an*bn),-1.,1.))/2))
    if not positions or not velocities:return dict(result,body_velocity_requests_unresolved=True)
    trajectory=float(np.mean(positions));velocity=float(np.mean(velocities))
    direction=None if not directions else float(np.mean(directions))
    return dict(result,state='OBSERVED_NATIVE_COMMON_BODY_ACTION_HYPOTHESIS',
        trajectory_error=trajectory,velocity_error=velocity,direction_error=direction,
        exploration_cost=2.*trajectory+velocity+(0. if direction is None else direction),
        same_time_action_supported=True,adjacent_body_velocity_observations=len(velocities),
        requested_joint_positions=len(positions),choreography_motion_observed=True)
