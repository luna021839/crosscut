"""Independent observed head tilt/tangents for local motion exploration.

Face-position matching cannot hide an opposite head inclination. COCO eyes
are used as orientation observations, not fake detector/mouth landmarks or
named identity. No detector, source search, interpolation or QA approval.
"""
import math
import numpy as np

from .group_review_camera_repair import _rotated_point


def bind_source_local_head_candidate(candidates,expected_face):
    """Bind an existing source-local occurrence, never rerank its identity.

    The caller owns the native frame/run provenance of the measured template
    and candidates. A slightly larger appearance score on a nearby person
    cannot replace that occurrence. Ambiguous/absent geometry stays unknown;
    this is not cross-camera or named-identity admission.
    """
    def box(face):
        values=np.asarray(face.get('box_xywh'),float)
        if (values.shape!=(4,) or not np.isfinite(values).all() or np.any(values[:2]<0)
                or np.any(values[2:]<=0) or np.any(values[:2]+values[2:]>1)):
            raise ValueError('source-local head binding needs contained measured boxes')
        return values
    target=box(expected_face);eligible=[]
    for candidate in candidates:
        observed=box(candidate['face'])
        intersection=float(np.prod(np.maximum(0.,np.minimum(target[:2]+target[2:],observed[:2]+observed[2:])
            -np.maximum(target[:2],observed[:2]))))
        overlap=intersection/(float(np.prod(target[2:])+np.prod(observed[2:]))-intersection)
        distance=float(np.linalg.norm((observed[:2]+observed[2:]/2-target[:2]-target[2:]/2)*[1.,9/16]))
        if overlap>=.35 and distance<=.065:eligible.append((overlap,candidate))
    eligible.sort(key=lambda value:value[0],reverse=True)
    if not eligible or len(eligible)>1 and eligible[0][0]-eligible[1][0]<.12:return None
    return eligible[0][1]


def observed_head_eye_angle(row,camera):
    body=row.get('body')
    if (not body or body.get('reason')!='observed' or body.get('face_anchor_consistent') is not True):
        return None
    points=np.asarray(body.get('raw_keypoints_xyc'),float)
    values=np.asarray(camera,float)
    if (points.shape!=(17,3) or not np.isfinite(points).all()
            or values.shape!=(4,) or not np.isfinite(values).all()):
        raise ValueError('head action requires actual finite raw17 and native camera geometry')
    # Only the actually measured eyes are needed for inclination. Missing ears
    # do not remove good eyes, nor get filled to manufacture a full head basis.
    eyes=points[[2,1]]
    if np.any(eyes[:,2]<.3) or np.any(eyes[:,:2]<0) or np.any(eyes[:,:2]>1):return None
    z,x,y,roll=values
    projected=np.array([_rotated_point(p,roll) for p in eyes[:,:2]])
    projected=(projected-[x,y])*z+.5
    if np.any(projected<0) or np.any(projected>1):return None
    delta=(projected[1]-projected[0])*[1,9/16]
    if delta[0]<=0 or np.linalg.norm(delta)<1e-6:return None
    return float(math.degrees(math.atan2(delta[1],delta[0])))


def measure_observed_head_incident(*,left_rows,right_rows,left_cameras,right_cameras,
        left_ticks,right_ticks,fps):
    """Fixed tilt+angular-tangent request; unknown never equals perfect match."""
    if (not isinstance(left_ticks,tuple) or not isinstance(right_ticks,tuple)
            or len(left_ticks)!=2 or len(right_ticks)!=2
            or any(type(t) is not int or t<0 for t in (*left_ticks,*right_ticks))
            or not left_ticks[0]<left_ticks[1]<right_ticks[0]<right_ticks[1]
            or any(len(v)!=2 for v in (left_rows,right_rows,left_cameras,right_cameras))
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0):
        raise ValueError('head incident requires its exact finite two-sided tangent inventory')
    for rows in (left_rows,right_rows):
        if any(type(r.get('native_ordinal')) is not int or r['native_ordinal']<0 for r in rows):
            raise ValueError('head incident lost its physical native ordinal')
    physical=all(rows[0].get('source_index')==rows[1].get('source_index')
        and 0<rows[1]['native_ordinal']-rows[0]['native_ordinal']<=12 for rows in (left_rows,right_rows))
    a=[observed_head_eye_angle(r,c) for r,c in zip(left_rows,left_cameras,strict=True)]
    b=[observed_head_eye_angle(r,c) for r,c in zip(right_rows,right_cameras,strict=True)]
    tilt=None if a[1] is None or b[0] is None else abs(math.remainder(b[0]-a[1],360.))
    tilt_cost=(0.,1.) if tilt is None else (min(1.,tilt/90.),)*2
    av=bv=None
    if physical and all(v is not None for v in (*a,*b)):
        av=math.remainder(a[1]-a[0],360.)*fps/(left_ticks[1]-left_ticks[0])
        bv=math.remainder(b[1]-b[0],360.)*fps/(right_ticks[1]-right_ticks[0])
        loss=abs(av-bv)/(30.+abs(av)+abs(bv));velocity_cost=(loss,loss)
    else:velocity_cost=(0.,1.)
    bounds=[.75*tilt_cost[i]+.25*velocity_cost[i] for i in (0,1)]
    return dict(contract='actual_observed_head_tilt_tangent_exploration_v1',
        left_ticks=list(left_ticks),right_ticks=list(right_ticks),actual_projected_eye_angles=[a,b],
        head_tilt_difference_degrees=tilt,angular_velocities_degrees_per_second=[av,bv],
        tilt_cost_interval=list(tilt_cost),angular_velocity_cost_interval=list(velocity_cost),
        exploration_cost_interval=bounds,exploration_score=bounds[1],physical_tangents_observed=physical,
        missing_imputed_as_zero=False,ears_required_or_invented=False,actor_identity_certified=False,
        full_action_certified=False,quality_approved=False)
