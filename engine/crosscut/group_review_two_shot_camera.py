"""Realize actual donor neighbours jointly, before selecting a two-shot block.

Original entry/exit and donor/donor boundary have distinct targets. The camera
is a whole selected transform, not a post-selection cosmetic refit. Numeric
geometry and native choreography are separate evidence; no quality approval.
"""
from dataclasses import asdict,dataclass,replace
import json
import math
import numpy as np
from scipy.optimize import minimize
from .group_performer_family_owner import digest
from .group_review_camera_repair import fit_whole_camera_track,transformed_descriptor,geometry_cost
from .group_review_camera_repair import project_review_editorial_point
from .group_review_projected_action import fit_review_anatomical_keys
from .render import EditorialTransformKeyframe,EditorialTransformTrack


def measure_review_partial_joint_trajectory(left,right,*,ticks,fps,requested_joints=(0,1,2,3,4,5)):
    """Exploration evidence for visible anatomy; NOT complete action admission.

    Missing joints keep explicit request IDs and bounded cost intervals. Rank
    by the fixed-inventory upper cost, not a donor's easier observed subset:
    removing an inconvenient limb cannot improve its exploratory score.
    Velocity uses consecutive common measured samples only, never interpolation
    through occlusion. Normalizing shoulders removes framing, not body action.
    """
    if (not 3<=len(ticks)<=15 or len(left)!=len(ticks) or len(right)!=len(ticks)
            or tuple(sorted(set(ticks)))!=ticks or any(type(t) is not int for t in ticks)
            or requested_joints!=tuple(sorted(set(requested_joints))) or not {0,1}<=set(requested_joints)
            or any(type(j) is not int or not 0<=j<6 for j in requested_joints)
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0):
        raise ValueError('partial trajectory needs its exact predeclared physical time/joint inventory')
    normalized=[];missing=[]
    for side,frames in (('left',left),('right',right)):
        values=[]
        for t,points in zip(ticks,frames):
            if not isinstance(points,dict) or any(j not in range(6) or len(p)!=2 or
                    not all(math.isfinite(v) and 0<=v<=1 for v in p) for j,p in points.items()):
                raise ValueError('partial trajectory points must be directly observed finite screen anatomy')
            missing.extend((side,t,j) for j in requested_joints if j not in points)
            if not {0,1}<=set(points):values.append({});continue
            center=((points[0][0]+points[1][0])/2,(points[0][1]+points[1][1])*9/32)
            width=math.hypot(points[0][0]-points[1][0],(points[0][1]-points[1][1])*9/16)
            if width<=1e-6:values.append({});continue
            values.append({j:((points[j][0]-center[0])/width,(points[j][1]*9/16-center[1])/width)
                for j in requested_joints if j in points})
        normalized.append(values)
    a,b=normalized;positions=[];velocities=[];directions=[]
    weights={j:.1 if j<2 else 1.5 if j in (4,5) else 1. for j in requested_joints}
    observed_articulation=0;known_weight=0.;error=0.
    bounded_position=0.;bounded_velocity=0.;velocity_weight=0.
    bounded_direction=0.;direction_weight=0.;articulated_velocity_requests=0
    nonstationary_windows=0
    for t,pa,pb in zip(ticks,a,b):
        for j in requested_joints:
            if j not in pa or j not in pb:continue
            distance=math.dist(pa[j],pb[j]);weight=weights[j]
            error+=distance*weight;known_weight+=weight;observed_articulation+=j>=2
            bounded_position+=weight*distance/(1.+distance)
            positions.append(dict(tick=t,joint=j,error=distance))
    for i in range(1,len(ticks)):
        dt=(ticks[i]-ticks[i-1])/fps
        for j in requested_joints:
            if not all(j in p for p in (a[i-1],a[i],b[i-1],b[i])):continue
            av=tuple((a[i][j][k]-a[i-1][j][k])/dt for k in (0,1))
            bv=tuple((b[i][j][k]-b[i-1][j][k])/dt for k in (0,1))
            an,bn=math.hypot(*av),math.hypot(*bv)
            velocity=math.dist(av,bv)/(1+max(an,bn));velocities.append(velocity)
            weight=weights[j];velocity_weight+=weight
            bounded_velocity+=weight*math.dist(av,bv)/(1.+an+bn)
            if j>=2:
                articulated_velocity_requests+=1;direction_weight+=weight
                nonstationary_windows+=max(an,bn)>.05
                if min(an,bn)>.05:
                    direction=(1-max(-1.,min(1.,sum(x*y for x,y in zip(av,bv))/(an*bn))))/2.
                    directions.append(direction);bounded_direction+=weight*direction
                # A stationary side has no defined direction. The measured
                # speed discrepancy remains in velocity; do not invent one.
    expected=len(ticks)*sum(weights.values());coverage=known_weight/expected
    expected_velocity=(len(ticks)-1)*sum(weights.values())
    expected_direction=(len(ticks)-1)*sum(w for j,w in weights.items() if j>=2)
    def interval(loss,known,total):
        if total<=0:return (0.,0.)
        return (loss/total,(loss+max(0.,total-known))/total)
    position_interval=interval(bounded_position,known_weight,expected)
    velocity_interval=interval(bounded_velocity,velocity_weight,expected_velocity)
    direction_interval=interval(bounded_direction,direction_weight,expected_direction)
    score_interval=tuple(position_interval[i]+.25*velocity_interval[i]+.35*direction_interval[i]
        for i in (0,1))
    score=None if observed_articulation<3 or articulated_velocity_requests<2 else score_interval[1]
    return dict(contract='partial_actual_joint_trajectory_exploration_v1',
        state='UNMEASURED' if score is None else 'PARTIAL_OBSERVED_HYPOTHESIS',exploration_score=score,
        requested_ticks=list(ticks),requested_joint_indices=list(requested_joints),
        missing_joint_requests=missing,measured_position_requests=positions,coverage=coverage,
        ranking_basis='fixed_requested_joint_time_upper_cost_not_observed_subset_mean',
        exploratory_cost_interval=list(score_interval),
        bounded_position_cost_interval=list(position_interval),
        bounded_velocity_cost_interval=list(velocity_interval),
        bounded_direction_cost_interval=list(direction_interval),
        observed_subset_position_error=None if not known_weight else error/known_weight,
        measured_articulated_velocity_requests=articulated_velocity_requests,
        nonstationary_common_action_observed=bool(nonstationary_windows),
        missing_cost_bounds_are_observations=False,
        measured_velocity_requests=len(velocities),expected_velocity_requests=(len(ticks)-1)*len(requested_joints),
        missing_imputed_as_zero=False,missing_joint_trajectory_interpolated=False,
        full_action_certified=False,actor_identity_certified=False,quality_approved=False)


def _fixed_zoom_support_pan_path(desired_keys,points_by_frame,locked_indices,fps):
    """Bounded continuous pan proposal with the AUTHORED zoom left unchanged.

    Support imposes constraints only at its actually observed time. Smooth
    camera anticipation is not an interpolated/unobserved anatomical point.
    The exact renderer validates every proposed key and between-key corner.
    """
    receipt=dict(contract='bounded_continuous_actual_support_pan_v1',maximum_iterations=96,
        maximum_pan_variables=320,anatomical_observations_interpolated=False,
        unknown_support_filled=False,authored_zoom_modified=False,quality_approved=False)
    if any(abs(k.roll_degrees)>1e-9 for k in desired_keys):
        return None,dict(receipt,state='UNSUPPORTED_NONZERO_ROLL')
    desired=np.asarray([(k.center_x,k.center_y) for k in desired_keys]);lower=[];upper=[]
    for i,(k,points) in enumerate(zip(desired_keys,points_by_frame)):
        half=.5/k.zoom;lo=np.full(2,half);hi=np.full(2,1-half)
        if points:
            xy=np.asarray(points);lo=np.maximum(lo,xy.max(axis=0)-half);hi=np.minimum(hi,xy.min(axis=0)+half)
        if i in locked_indices:lo=hi=desired[i].copy()
        if np.any(lo>hi):return None,dict(receipt,state='FIXED_ZOOM_SUPPORT_INFEASIBLE',frame_index=i)
        closest=np.clip(np.full(2,.5),lo,hi)
        if float(np.sum((closest-.5)**2))>.25**2+1e-10:
            return None,dict(receipt,state='FIXED_ZOOM_PAN_CAP_INFEASIBLE',frame_index=i)
        lower.append(lo);upper.append(hi)
    lo,hi=np.asarray(lower),np.asarray(upper);start=np.clip(desired,lo,hi)
    # A feasible seed lies on the segment from the closest-to-center legal
    # point to the desired pan. Both endpoints obey this frame's rectangle;
    # intersect that segment with the unchanged renderer radial pan cap.
    for i in range(len(start)):
        if np.sum((start[i]-.5)**2)<=.25**2:continue
        origin=np.clip(np.full(2,.5),lo[i],hi[i]);delta=start[i]-origin;offset=origin-.5
        a=float(delta@delta);b=2*float(offset@delta);c=float(offset@offset)-.25**2
        alpha=0. if a<=1e-15 else max(0.,min(1.,(-b+math.sqrt(max(0.,b*b-4*a*c)))/(2*a)))
        start[i]=origin+alpha*delta
    weights=np.asarray((1.,(9/16)**2));velocity_weight=.02*fps**2;acceleration_weight=.0002*fps**4
    def objective(flat):
        x=flat.reshape((-1,2));delta=x-desired;velocity=np.diff(x,axis=0);acceleration=np.diff(x,n=2,axis=0)
        cost=float(4*np.sum(delta**2*weights)+velocity_weight*np.sum(velocity**2*weights)+
            acceleration_weight*np.sum(acceleration**2*weights))
        gradient=8*delta*weights
        v=2*velocity_weight*velocity*weights;gradient[:-1]-=v;gradient[1:]+=v
        a=2*acceleration_weight*acceleration*weights
        gradient[:-2]+=a;gradient[1:-1]-=2*a;gradient[2:]+=a
        return cost,gradient.ravel()
    def radial(flat):return .25**2-np.sum((flat.reshape((-1,2))-.5)**2,axis=1)
    def radial_jac(flat):
        x=flat.reshape((-1,2));jac=np.zeros((len(x),len(flat)));indices=np.arange(len(x))
        jac[indices,2*indices]=-2*(x[:,0]-.5);jac[indices,2*indices+1]=-2*(x[:,1]-.5)
        return jac
    result=minimize(objective,start.ravel(),jac=True,method='SLSQP',bounds=list(zip(lo.ravel(),hi.ravel())),
        constraints=({'type':'ineq','fun':radial,'jac':radial_jac},),options={'maxiter':96,'ftol':1e-10})
    receipt.update(optimizer_converged=bool(result.success),iterations=int(getattr(result,'nit',0)))
    if not np.isfinite(result.x).all():return None,dict(receipt,state='NONFINITE_OPTIMIZER_RESULT')
    try:
        keys=tuple(replace(k,center_x=float(x),center_y=float(y)) for k,(x,y) in zip(desired_keys,result.x.reshape((-1,2))))
        if any(not all(-1e-9<=v<=1+1e-9 for v in project_review_editorial_point(p,k))
                for k,points in zip(keys,points_by_frame) for p in points):
            return None,dict(receipt,state='ACTUAL_SUPPORT_VALIDATION_FAILED')
        track=EditorialTransformTrack(keys,reason='continuous_actual_support_pan_with_authored_zoom')
    except ValueError:return None,dict(receipt,state='EXACT_RENDERER_INTEGRITY_FAILED')
    if any(keys[i]!=desired_keys[i] for i in locked_indices):
        return None,dict(receipt,state='LOCKED_ANCHOR_CHANGED')
    return track,dict(receipt,state='REALIZED_INSPECTION_HYPOTHESIS_NOT_APPROVAL')


def fit_review_temporal_camera_corridor(desired_keys,*,observed_points_by_frame,locked_indices=(),fps):
    """Preserve observed support at its own time, not a cross-time union.

    A moving head/hand need not remain inside every OTHER instant's camera.
    Finite whole-path optimization trades intervention against camera speed
    and acceleration. Panning to retain a moving hand is a separate option
    from shrinking the crop back to neutral; one scalar alpha cannot express
    both. Missing observations remain unmeasured, never support PASS.
    """
    if (not isinstance(desired_keys,(list,tuple)) or not 2<=len(desired_keys)<=160
            or len(observed_points_by_frame)!=len(desired_keys)
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0
            or any(not isinstance(k,EditorialTransformKeyframe) or abs(k.time_seconds-i/fps)>1e-9
                for i,k in enumerate(desired_keys))
            or tuple(sorted(set(locked_indices)))!=locked_indices
            or any(type(i) is not int or not 0<=i<len(desired_keys) for i in locked_indices)):
        raise ValueError('temporal camera corridor needs its exact bounded per-frame request')
    def camera_vector(k):return (k.center_x,k.center_y*9/16,math.log(k.zoom)/math.sqrt(2),k.roll_degrees*.01)
    def distance(a,b):return sum((x-y)**2 for x,y in zip(a,b))
    choices=[];unknown=[];actual_support=[]
    for i,(key,points) in enumerate(zip(desired_keys,observed_points_by_frame)):
        if points is None:points=();unknown.append(i)
        if (not isinstance(points,(tuple,list)) or len(points)>32
                or any(len(p)!=2 or not all(math.isfinite(v) and 0<=v<=1 for v in p) for p in points)):
            raise ValueError('temporal support must be directly observed screen points at each instant')
        if not points and i not in unknown:unknown.append(i)
        actual_support.append(points)
        candidates=[];seen=set()
        for alpha in ((1.,) if i in locked_indices else (1.,.85,.7,.55,.4,.25,0.)):
            try:k=EditorialTransformKeyframe(key.time_seconds,1+(key.zoom-1)*alpha,
                .5+(key.center_x-.5)*alpha,.5+(key.center_y-.5)*alpha,key.roll_degrees*alpha)
            except ValueError:continue
            proposals=[k]
            if points and i not in locked_indices:
                # Rotation is about the native frame BEFORE pan/zoom. Solve
                # the actual support interval in that rotated coordinate
                # system, then let the renderer integrity constructor check
                # corners, the radial pan cap and the exact roll overscan.
                centered=replace(k,center_x=.5,center_y=.5)
                rotated=[tuple(.5+(v-.5)/k.zoom for v in project_review_editorial_point(p,centered)) for p in points]
                half=.5/k.zoom
                bounds=[(max(half,max(p[axis] for p in rotated)-half),
                    min(1-half,min(p[axis] for p in rotated)+half)) for axis in (0,1)]
                if all(lo<=hi+1e-9 for lo,hi in bounds):
                    center=tuple(max(lo,min(hi,v)) for v,(lo,hi) in zip((k.center_x,k.center_y),bounds))
                    try:proposals.append(replace(k,center_x=center[0],center_y=center[1]))
                    except ValueError:pass
            for proposal in proposals:
                token=tuple(round(v,10) for v in (proposal.zoom,proposal.center_x,proposal.center_y,proposal.roll_degrees))
                if token in seen:continue
                seen.add(token)
                if all(all(-1e-9<=v<=1+1e-9 for v in project_review_editorial_point(p,proposal)) for p in points):
                    candidates.append((proposal,4.*distance(camera_vector(proposal),camera_vector(key))))
        if not candidates:return None,dict(reason='no_legal_actual_time_support_camera',frame_index=i,quality_approved=False)
        choices.append(candidates)
    # Second-order DP keeps one best history for each ordered pair of states.
    # Greedily keeping only the last camera loses acceleration information.
    previous={(None,j):(local,(k,)) for j,(k,local) in enumerate(choices[0])}
    for frame,candidates in enumerate(choices[1:],1):
        legal_edges={}
        for before_index,(before,_) in enumerate(choices[frame-1]):
            for index,(k,_) in enumerate(candidates):
                try:EditorialTransformTrack((replace(before,time_seconds=0.),replace(k,time_seconds=1/fps)),
                    reason='actual_between_frame_camera_integrity')
                except ValueError:continue
                legal_edges[(before_index,index)]=fps**2*distance(camera_vector(k),camera_vector(before))
        current={}
        for (_,before_index),(cost,path) in previous.items():
            before=path[-1];bv=camera_vector(before)
            for index,(k,local) in enumerate(candidates):
                velocity=legal_edges.get((before_index,index))
                if velocity is None:continue
                acceleration=0.
                if len(path)>1:
                    av=camera_vector(path[-2]);kv=camera_vector(k)
                    acceleration=fps**4*sum((z-2*y+x)**2 for x,y,z in zip(av,bv,kv))
                option=(cost+local+.02*velocity+.0002*acceleration,path+(k,))
                token=(before_index,index)
                if token not in current or option[0]<current[token][0]:current[token]=option
        if not current:return None,dict(reason='no_integrity_preserving_temporal_path',quality_approved=False)
        previous=current
    cost,keys=min(previous.values(),key=lambda p:p[0])
    track=EditorialTransformTrack(keys,reason='time_specific_known_support_coupled_focal_camera')
    pan_track,pan_receipt=_fixed_zoom_support_pan_path(desired_keys,actual_support,locked_indices,fps)
    selected_family='discrete_joint_pan_zoom'
    if pan_track is not None:
        vectors=[camera_vector(k) for k in pan_track.keyframes]
        pan_cost=4*sum(distance(v,camera_vector(k)) for v,k in zip(vectors,desired_keys))+.02*fps**2*sum(
            distance(a,b) for a,b in zip(vectors,vectors[1:]))+.0002*fps**4*sum(
            sum((z-2*y+x)**2 for x,y,z in zip(a,b,c)) for a,b,c in zip(vectors,vectors[1:],vectors[2:]))
        pan_receipt['whole_path_cost']=pan_cost
        if pan_cost<cost:
            track=pan_track;keys=track.keyframes;cost=pan_cost;selected_family='continuous_fixed_authored_zoom_pan'
    vectors=[camera_vector(k) for k in keys]
    steps=[math.sqrt(distance(a,b)) for a,b in zip(vectors,vectors[1:])]
    accelerations=[math.sqrt(sum((z-2*y+x)**2 for x,y,z in zip(a,b,c))) for a,b,c in zip(vectors,vectors[1:],vectors[2:])]
    return track,dict(contract='time_specific_actual_support_camera_corridor_v1',frames=len(keys),
        support_time_slots_kept_separate=True,cross_time_coordinate_union_used=False,
        unmeasured_support_frame_indices=unknown,locked_frame_indices=list(locked_indices),
        maximum_states_per_frame=14,maximum_ordered_state_pairs=196,whole_path_cost=cost,
        continuous_pan_proposal=pan_receipt,selected_camera_path_family=selected_family,
        support_constrained_pan_is_distinct_from_zoom_release=True,second_order_camera_path=True,
        maximum_camera_vector_step=max(steps,default=0.),maximum_camera_vector_second_difference=max(accelerations,default=0.),
        maximum_log_zoom_step=max((abs(math.log(b.zoom/a.zoom)) for a,b in zip(keys,keys[1:])),default=0.),
        known_support_preserved_at_observed_times=True,
        complete_support_certified=not unknown,actor_identity_certified=False,quality_approved=False)


def fit_review_coupled_boundary_keys(*,left_head,right_head,left_joint_points,right_joint_points,
        requested_joints=(0,1,2,3,4,5),left_required_visible_points=(),right_required_visible_points=(),
        maximum_candidates=8,maximum_zoom=2.,target_centers=((.5,.5),(.42,.45),(.5,.45),(.58,.45),
            (.42,.525),(.5,.525),(.58,.525),(.42,.6),(.5,.6),(.58,.6)),
        camera_fit_bases=('face_and_body','face_anchor')):
    """Fit BOTH selected cameras to common actual focal anatomy, before render.

    Independently observed missing joints are retained as missing requests.
    Common measured joints may help a geometric hypothesis, but cannot turn
    incomplete action, appearance identity, or artistic attention into PASS.
    No body bounding box or other performer's joints stand in for this actor.
    """
    if (type(maximum_candidates) is not int or not 1<=maximum_candidates<=8
            or requested_joints!=tuple(sorted(set(requested_joints)))
            or not requested_joints or any(type(j) is not int or not 0<=j<6 for j in requested_joints)
            or not 1<=maximum_zoom<=2 or not isinstance(target_centers,tuple) or not 1<=len(target_centers)<=10
            or any(len(c)!=2 or not all(math.isfinite(v) and .25<=v<=.75 for v in c) for c in target_centers)
            or not isinstance(camera_fit_bases,tuple) or not 1<=len(camera_fit_bases)<=2
            or len(set(camera_fit_bases))!=len(camera_fit_bases)
            or any(b not in ('face_and_body','face_anchor') for b in camera_fit_bases)):
        raise ValueError('coupled boundary fit needs its independently bounded actual joint request')
    def validate(head,points):
        if not isinstance(head,dict) or not isinstance(points,dict):
            raise ValueError('coupled camera needs actual projected head and joint inventories')
        center=head.get('center',());w=head.get('width');h=head.get('height')
        if (len(center)!=2 or type(w) not in (int,float) or type(h) not in (int,float)
                or not all(math.isfinite(x) and 0<=x<=1 for x in (*center,w,h)) or min(w,h)<=0
                or any(j not in range(6) or len(p)!=2 or not all(math.isfinite(x) and 0<=x<=1 for x in p)
                    for j,p in points.items())):
            raise ValueError('coupled boundary input is not actual visible screen geometry')
    validate(left_head,left_joint_points);validate(right_head,right_joint_points)
    common=tuple(j for j in requested_joints if j in left_joint_points and j in right_joint_points)
    missing=dict(left=[j for j in requested_joints if j not in left_joint_points],
        right=[j for j in requested_joints if j not in right_joint_points])
    def face(head):
        x,y=head['center'];w,h=head['width'],head['height']
        return {16:(x,y),17:(x-w/2,y-h/2),18:(x+w/2,y-h/2),19:(x-w/2,y+h/2),20:(x+w/2,y+h/2)}
    a={**{j:left_joint_points[j] for j in common},**face(left_head)}
    b={**{j:right_joint_points[j] for j in common},**face(right_head)}
    # Target extent is derived from BOTH actual views, never a single fixed
    # context whose scale/position may be impossible for the other camera.
    height=min(maximum_zoom*min(left_head['height'],right_head['height']),
        1.35*max(left_head['height'],right_head['height']))
    width=height*math.sqrt(left_head['width']/left_head['height']*right_head['width']/right_head['height'])
    options=[];seen=set()
    for destination in target_centers:
        target=face(dict(center=destination,width=width,height=height))
        for j in common:
            target[j]=tuple(destination[axis]+height*((a[j][axis]-left_head['center'][axis])/left_head['height']+
                (b[j][axis]-right_head['center'][axis])/right_head['height'])/2 for axis in (0,1))
        basis=tuple(sorted(target))
        for fit_basis in camera_fit_bases:
            # Perspective differences cannot all be repaired by one 2-D
            # camera. Keep a face-anchor proposal as well as the anatomical
            # compromise; ALL common joints still contribute to scoring and
            # ALL independently required points still constrain visibility.
            fitted_basis=basis if fit_basis=='face_and_body' else (16,17,18,19,20)
            source_a={j:a[j] for j in fitted_basis};source_b={j:b[j] for j in fitted_basis}
            fitted_target={j:target[j] for j in fitted_basis}
            left=fit_review_anatomical_keys(source_a,fitted_target,requested_joints=fitted_basis,time_seconds=0.,
                required_visible_points=left_required_visible_points,maximum_candidates=maximum_candidates,maximum_zoom=maximum_zoom)
            right=fit_review_anatomical_keys(source_b,fitted_target,requested_joints=fitted_basis,time_seconds=0.,
                required_visible_points=right_required_visible_points,maximum_candidates=maximum_candidates,maximum_zoom=maximum_zoom)
            for ka,kb in ((x,y) for x in left for y in right):
                token=tuple(round(v,10) for v in (ka.zoom,ka.center_x,ka.center_y,ka.roll_degrees,
                    kb.zoom,kb.center_x,kb.center_y,kb.roll_degrees))
                if token in seen:continue
                seen.add(token)
                pa={j:project_review_editorial_point(p,ka) for j,p in a.items()}
                pb={j:project_review_editorial_point(p,kb) for j,p in b.items()}
                point_error=sum(math.hypot(pa[j][0]-pb[j][0],(pa[j][1]-pb[j][1])*9/16) for j in basis)/len(basis)
                size_error=abs(math.log(ka.zoom*left_head['height']/(kb.zoom*right_head['height'])))
                head_error=math.hypot(pa[16][0]-pb[16][0],(pa[16][1]-pb[16][1])*9/16)
                score=3*head_error+point_error+.6*size_error+.005*math.log(ka.zoom*kb.zoom)
                options.append((score,ka,kb,point_error,size_error,fit_basis,head_error))
    options.sort(key=lambda v:(v[0],v[1].zoom,v[2].zoom,v[1].center_x,v[2].center_x))
    receipt=dict(contract='coupled_actual_focal_boundary_camera_hypotheses_v1',requested_joint_indices=list(requested_joints),
        measured_common_joint_indices=list(common),missing_joint_requests=missing,
        complete_requested_action_basis=not any(missing.values()),camera_pair_count=len(options),
        independently_declared_common_target_centers=[list(c) for c in target_centers],
        shared_target_must_be_dead_center=False,independently_declared_camera_fit_bases=list(camera_fit_bases),
        maximum_geometric_camera_pairs=len(target_centers)*len(camera_fit_bases)*maximum_candidates**2,
        body_requests_removed_to_fit_face=False,
        partial_geometry_is_complete_motion_match=False,actor_identity_certified=False,
        artistic_focal_intent_certified=False,quality_approved=False)
    return tuple((ka,kb,dict(receipt,geometry_score=score,projected_point_error=point_error,
        log_face_scale_error=size_error,camera_fit_basis=fit_basis,projected_head_center_error=head_error))
        for score,ka,kb,point_error,size_error,fit_basis,head_error in options[:maximum_candidates])


def _joint_target(left,right,maximum_zoom):
    if left is None or right is None or left.head_center is None or right.head_center is None:return None
    if any(d.face_height is None or d.face_width is None for d in (left,right)):return None
    destination=(.5,.45)
    # Both donors may need magnification to pan legally. Merely choosing the
    # larger native face leaves that donor at zoom 1 and unable to translate.
    required=[]
    for d in (left,right):
        x,y=d.head_center
        if not 0<x<1 or not 0<y<1:return None
        minimum_zoom=max(1.,destination[0]/x,(1-destination[0])/(1-x),
            destination[1]/y,(1-destination[1])/(1-y))
        required.append(d.face_height*minimum_zoom)
    height=min(max(required),maximum_zoom*min(left.face_height,right.face_height))
    aspect=math.sqrt(left.face_width/left.face_height*right.face_width/right.face_height)
    roll=(None if left.head_roll_radians is None or right.head_roll_radians is None else
        left.head_roll_radians+math.remainder(right.head_roll_radians-left.head_roll_radians,2*math.pi)/2)
    return replace(left,head_center=destination,face_height=height,face_width=height*aspect,
        body_box=None,pose_shape=None,head_roll_radians=roll)


def realize_review_two_shot_camera_pair(left_views,right_views,*,first,boundary,end,fps,
        original_entry,original_exit,required_support_check,maximum_zoom=2.):
    if (any(type(t) is not int for t in (first,boundary,end)) or not first<boundary<end
            or not 2<=boundary-first<=64 or not 2<=end-boundary<=64
            or len(left_views)!=boundary-first or len(right_views)!=end-boundary
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0
            or not callable(required_support_check)):
        raise ValueError('two-shot camera needs bounded actual donor intervals and original entry/exit')
    for expected,views in ((range(first,boundary),left_views),(range(boundary,end),right_views)):
        if any(v is not None and v['row']['output_tick']!=t for t,v in zip(expected,views)):
            raise ValueError('joint camera borrowed another selected donor instant')
    result=dict(contract='actual_neighbour_two_shot_camera_realization_v1',state='UNKNOWN',
        interval=[first,end],boundary=boundary,camera_fitted_before_selection=True,
        post_selection_refit_permitted=False,native_choreography_match_certified=False,quality_approved=False)
    if any(v is None for v in (*left_views,*right_views)):return None,None,result
    joint=_joint_target(left_views[-1]['descriptor'],right_views[0]['descriptor'],maximum_zoom)
    if joint is None:return None,None,result
    tracks=[];support=[]
    for side,views,start,target_a,target_b in (
            ('left',left_views,first,original_entry,joint),
            ('right',right_views,boundary,joint,original_exit)):
        by_tick={v['row']['output_tick']:v for v in views}
        def allowed(row,key):
            verdict=required_support_check(side,by_tick[row['output_tick']],key)
            if verdict not in ('PASS','FAIL','UNKNOWN'):raise ValueError('joint camera requires actual three-state support')
            return verdict!='FAIL'  # UNKNOWN remains explicit, never certification.
        duration=(len(views)-1)/fps
        track=fit_whole_camera_track([v['row'] for v in views],target_a,target_b,duration=duration,
            sample_times=tuple(i/fps for i in range(len(views))),
            head_ids=tuple(v['descriptor'].frame_local_head_id for v in views),
            descriptors=tuple(v['descriptor'] for v in views),maximum_zoom=maximum_zoom,
            required_support_check=allowed)
        if track is None:return None,None,dict(result,reason='joint_whole_camera_support_failed')
        tracks.append(track)
        support.extend(dict(side=side,tick=v['row']['output_tick'],
            state=required_support_check(side,v,track.evaluate((v['row']['output_tick']-start)/fps))) for v in views)
    left_track,right_track=tracks
    a=transformed_descriptor(left_views[-1]['descriptor'],left_track.evaluate((len(left_views)-1)/fps))
    b=transformed_descriptor(right_views[0]['descriptor'],right_track.evaluate(0.))
    entry=transformed_descriptor(left_views[0]['descriptor'],left_track.evaluate(0.))
    exit=transformed_descriptor(right_views[-1]['descriptor'],right_track.evaluate((len(right_views)-1)/fps))
    costs=dict(entry=geometry_cost(original_entry,entry),internal=geometry_cost(a,b),
        exit=geometry_cost(exit,original_exit))
    bindings=[dict(tick=v['row']['output_tick'],raw_parent=digest(v['row']['native_observation']),
        head=v['descriptor'].frame_local_head_id,joint_evidence=v.get('review_joint_evidence_identity'))
        for v in (*left_views,*right_views)]
    result.update(state='REALIZED',joint_target=joint.to_dict(),left_track=asdict(left_track),
        right_track=asdict(right_track),geometry_costs=costs,support=support,
        actual_donor_binding_identity=digest(bindings),
        actual_support_state='FAIL' if any(s['state']=='FAIL' for s in support) else
            'UNKNOWN' if any(s['state']=='UNKNOWN' for s in support) else 'PASS')
    result['realization_identity']=digest(result)
    return left_track,right_track,result


@dataclass(frozen=True,slots=True)
class FrozenRealizedTwoShotCandidate:
    token_json:str

    def __post_init__(self):
        p=self.payload
        if (p.get('contract')!='sealed_realized_two_shot_candidate_v1'
                or p.get('technical_state') not in ('PASS','FAIL','UNKNOWN')
                or type(p.get('changed_frames')) is not int or p['changed_frames']<0
                or p.get('camera_fitted_before_selection') is not True
                or p.get('post_selection_camera_refit') is not False
                or any(type(p.get(k)) not in (int,float) or not math.isfinite(p[k]) or p[k]<0
                    for k in ('native_motion_error','projected_geometry_error','camera_instability'))):
            raise ValueError('two-shot candidate needs sealed actual realization and separate finite scores')

    @property
    def payload(self):return json.loads(self.token_json)

    @property
    def identity(self):return digest(self.payload)

    def assert_unchanged(self,payload):
        if payload!=self.payload:raise ValueError('selected two-shot camera/sampler was changed after sealing')


def select_review_two_shot_diagnostic(candidates):
    if not isinstance(candidates,(list,tuple)) or len(candidates)>336:
        raise ValueError('two-shot diagnostic requires a finite predeclared candidate inventory')
    if any(not isinstance(c,FrozenRealizedTwoShotCandidate) for c in candidates):
        raise ValueError('cannot select an unrealized two-shot camera')
    complete=[c for c in candidates if c.payload['technical_state']=='PASS' and c.payload['changed_frames']>0]
    # No KEEP/old UNKNOWN penalty; one complete changed diagnostic hypothesis.
    return min(complete,key=lambda c:(c.payload['native_motion_error'],c.payload['projected_geometry_error'],
        c.payload['camera_instability'],c.identity)) if complete else None
