"""Declared anatomical motion in the actual registered output camera.

Native shoulder-normalized motion and projected screen connection are different
measurements. This consumer uses owned native joint provenance and the complete
registered crop/pan/zoom/roll, never a normalized pose pretending to be pixels.
"""
import math
import numpy as np
from .group_performer_family_owner import digest
from .group_scene_camera_queries import GroupSceneRoleQuery
from .group_source_changing_owner import project_point,project_box,_base_box
from .render import EditorialTransformKeyframe,minimum_roll_safe_zoom
from .group_review_camera_repair import project_review_editorial_point
from .group_review_motion_contract import FrozenReviewNativeMotionRoute
from .group_review_declared_action import FrozenReviewActionDeclaration
from .group_review_motion_seam import review_actor_association_valid


def project_observed_review_face(view,query,tick):
    """The selected native occurrence under its ACTUAL registered camera.

    A descriptor alone is not evidence: recompute its box from the bound raw
    head and compare it with the registered projection. Missing/ambiguous or
    cropped heads remain unknown. No body point can stand in for a face.
    """
    if not isinstance(query,GroupSceneRoleQuery):
        raise ValueError('projected face needs the registered complete camera')
    if not review_actor_association_valid(view):return None
    row=view['row'];raw=row['native_observation'];_,_,native=query.sample_at(tick)
    if (row['output_tick']!=tick
            or row['query'].get('registered_query_sha256',digest(row['query']))!=query.key
            or raw['native_frame_index']!=native or raw['source_clock_identity']!=query.source_clock_identity
            or row['projection']['camera_sha256']!=query.camera.key):
        raise ValueError('projected face borrowed another actual camera/native sample')
    head_id=view['descriptor'].frame_local_head_id
    heads=[h for h in raw.get('heads',()) if h['observation_id']==head_id]
    projected=[h for h in row['projection'].get('heads',()) if h['observation_id']==head_id]
    if len(heads)!=1 or len(projected)!=1:return None
    box,fraction=project_box(heads[0]['box_xywh'],query,tick)
    recorded=projected[0]
    if (len(recorded['projected_box_xywh'])!=4
            or any(not math.isfinite(v) for v in (*box,*recorded['projected_box_xywh']))
            or any(not math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-9)
                for a,b in zip(box,recorded['projected_box_xywh']))
            or not math.isclose(fraction,recorded['visible_fraction'],rel_tol=1e-8,abs_tol=1e-9)):
        raise ValueError('projected face box differs from its bound actual raw occurrence/camera')
    x,y,w,h=box
    if fraction<.995 or w<=0 or h<=0:return None
    landmarks=heads[0].get('landmarks_xy');projected_landmarks=None;landmark_scale=None;eye_span=None
    if (isinstance(landmarks,(tuple,list)) and len(landmarks)==5
            and all(isinstance(p,(tuple,list)) and len(p)==2
                and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in p) for p in landmarks)):
        points=tuple(tuple(float(v) for v in project_point(p,query,tick)) for p in landmarks)
        if all(all(math.isfinite(v) and 0<=v<=1 for v in p) for p in points):
            aspect=query.camera.output_size[1]/query.camera.output_size[0]
            physical=np.asarray(points)*[1.,aspect]
            extent=float(np.sqrt(np.mean(np.sum((physical-physical.mean(0))**2,axis=1))))
            eyes=float(np.linalg.norm(physical[0]-physical[1]))
            if extent>1e-6 and eyes>1e-6:
                projected_landmarks=points;landmark_scale=extent;eye_span=eyes
    return dict(center=(x+w/2,y+h/2),width=w,height=h,
        native_parent=digest(raw),selected_head=head_id,registered_query_identity=query.key,
        native_head_identity=digest(heads[0]),camera_identity=query.camera.key,
        landmarks_xy=projected_landmarks,five_point_scale=landmark_scale,eye_span=eye_span,
        facial_landmark_geometry_state='OBSERVED' if projected_landmarks is not None else 'UNAVAILABLE',
        detector_box_extent_is_facial_scale=False)


def measure_review_projected_face(left,right,*,left_queries,right_queries,fps,route):
    """Separate actual screen/face-scale costs, never an actor/quality badge.

    Consume the same immutable physical route and EVERY requested sample.
    This can expose a bad visual match even when normalized arm poses agree.
    """
    if (not isinstance(route,FrozenReviewNativeMotionRoute)
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0):
        raise ValueError('projected face needs its sealed native route and output clock')
    ticks=route['requested_ticks']
    if (len(left)!=len(ticks) or len(right)!=len(ticks)
            or set(left_queries)!=set(ticks) or set(right_queries)!=set(ticks)):
        raise ValueError('projected face omitted or changed its frozen request inventory')
    indices=route['comparison_indices'];values=[];bindings=[];missing=[];sizes=set()
    for side,views,queries in (('left',left,left_queries),('right',right,right_queries)):
        faces=[]
        for i in indices:
            t=ticks[i];v=views[i];q=queries[t]
            if not isinstance(q,GroupSceneRoleQuery) or not math.isclose(q.camera.output_fps,fps,abs_tol=1e-9):
                raise ValueError('projected face changed its actual output clock')
            sizes.add(tuple(q.camera.output_size))
            if v is not None:
                raw=v['row']['native_observation']
                actual=[raw['request_source_index'],raw['source_clock_identity'],digest(raw['source_file_identity']),
                    v.get('native_camera_unit_index'),raw['native_frame_index']]
                if actual!=route[side+'_native'][i]:
                    raise ValueError('projected face borrowed another frozen native route')
            face=project_observed_review_face(v,q,t);faces.append(face)
            if face is None:missing.append((side,t))
            bindings.append(dict(side=side,tick=t,registered_query_identity=q.key,face=face))
        values.append(faces)
    if indices and len(sizes)!=1:raise ValueError('projected face changed its common output canvas')
    result=dict(contract='actual_registered_camera_face_connection_evidence_v1',state='UNKNOWN',
        geometry_kind='observed_detector_head_box_not_eye_landmarks',
        route_identity=route.identity,actual_input_identity=digest(bindings),actual_inputs=bindings,
        face_center_position_error=None,face_scale_log_error=None,face_aspect_log_error=None,
        face_center_velocity_error=None,complete_on_declared_basis=False,
        actor_correspondence_certified=False,motion_suitability_certified=False,quality_approved=False)
    if not indices or missing:return dict(result,missing_face_requests=missing)
    width,height=next(iter(sizes));aspect=np.asarray((1.,height/width))
    a,b=[np.asarray([f['center'] for f in faces])*aspect for faces in values]
    dt=np.diff([ticks[i]/fps for i in indices]);av,bv=np.diff(a,axis=0)/dt[:,None],np.diff(b,axis=0)/dt[:,None]
    velocity=float((np.linalg.norm(av-bv,axis=1)/(1.+np.maximum(np.linalg.norm(av,axis=1),
        np.linalg.norm(bv,axis=1)))).mean()) if len(dt) else None
    return dict(result,state='MEASURED',face_center_position_error=float(np.linalg.norm(a-b,axis=1).mean()),
        face_scale_log_error=float(np.mean([abs(math.log(y['height']/x['height'])) for x,y in zip(*values)])),
        face_aspect_log_error=float(np.mean([abs(math.log((y['width']/y['height'])/(x['width']/x['height'])))
            for x,y in zip(*values)])),face_center_velocity_error=velocity,
        measured_position_samples=len(indices),measured_velocity_windows=len(dt),complete_on_declared_basis=True)


def project_observed_review_joints(view,query,tick):
    if not isinstance(query,GroupSceneRoleQuery):raise ValueError('projected action needs the registered complete camera')
    if not review_actor_association_valid(view):return {}
    row=view['row'];raw=row['native_observation'];_,_,native=query.sample_at(tick)
    query_id=row['query'].get('registered_query_sha256',digest(row['query']))
    if (row['output_tick']!=tick or query_id!=query.key
            or raw['native_frame_index']!=native or raw['source_clock_identity']!=query.source_clock_identity
            or row['projection']['camera_sha256']!=query.camera.key):
        raise ValueError('projected joint evidence borrowed another actual camera/native sample')
    points={}
    for index,p in view.get('review_joint_provenance',{}).items():
        index=int(index)
        if (not 0<=index<6 or p.get('raw_parent_sha256')!=digest(raw)
                or p.get('basis') not in ('PRIMARY_NATIVE_MODEL','FIXED_ROI_NATIVE_MODEL')
                or type(p.get('model_confidence')) not in (int,float)
                or not math.isfinite(p['model_confidence']) or p['model_confidence']<.42):
            raise ValueError('projected anatomical evidence lost its actual native parent/provenance')
        if p['basis']=='FIXED_ROI_NATIVE_MODEL' and p.get('receipt_sha256')!=view.get('review_body_receipt_sha256'):
            raise ValueError('projected anatomical evidence borrowed another actual fixed-ROI receipt')
        xy=p.get('xy_native')
        if (not isinstance(xy,(list,tuple)) or len(xy)!=2
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in xy)):
            raise ValueError('projected action cannot reconstruct pixels from normalized shape alone')
        segment,phase,_=query.sample_at(tick)
        base_xy=_base_box((*xy,0.,0.),segment,phase)[:2]
        # A pan after the base crop cannot recover a removed native pixel.
        if not all(0<=v<=1 for v in base_xy):continue
        points[index]=tuple(float(v) for v in project_point(xy,query,tick))
    return points


def measure_review_partial_screen_trajectory(left,right,*,ticks,fps,aspect=9/16):
    """Keep actual screen translation in a bounded partial motion comparison.

    Inputs must already use the cameras being compared. Unlike shoulder-relative
    articulation this retains root travel and absolute hand direction. Missing
    terms occupy their original denominator and never improve the upper cost.
    It is exploration evidence, not a complete-action or perceptual certificate.
    """
    if (not isinstance(ticks,tuple) or not 3<=len(ticks)<=15 or ticks!=tuple(sorted(set(ticks)))
            or any(type(t) is not int for t in ticks) or len(left)!=len(ticks) or len(right)!=len(ticks)
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0
            or type(aspect) not in (int,float) or not math.isfinite(aspect) or aspect<=0):
        raise ValueError('screen motion needs exact bounded output times and aspect')
    values=[]
    for sequence in (left,right):
        frames=[]
        for points in sequence:
            if (not isinstance(points,dict) or any(type(j) is not int or not 0<=j<6
                    or len(p)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in p)
                    for j,p in points.items())):
                raise ValueError('screen motion needs actual finite joint positions')
            frame={j:np.asarray((p[0],p[1]*aspect)) for j,p in points.items()}
            if 0 in frame and 1 in frame:frame['root']=(frame[0]+frame[1])/2
            frames.append(frame)
        values.append(frames)
    weights={0:.1,1:.1,2:1.,3:1.,4:1.5,5:1.5,'root':2.}
    losses=[0.,0.,0.];known=[0.,0.,0.];roots=[];measured_velocity=0
    for i,(a,b) in enumerate(zip(*values)):
        for joint,weight in weights.items():
            if joint in a and joint in b:
                distance=float(np.linalg.norm(a[joint]-b[joint]))
                losses[0]+=weight*distance/(.05+distance);known[0]+=weight
            if i==0 or not all(joint in f for f in (values[0][i-1],a,values[1][i-1],b)):continue
            dt=(ticks[i]-ticks[i-1])/fps
            av=(a[joint]-values[0][i-1][joint])/dt
            bv=(b[joint]-values[1][i-1][joint])/dt
            an,bn=float(np.linalg.norm(av)),float(np.linalg.norm(bv))
            difference=float(np.linalg.norm(av-bv))
            velocity=difference/(.15+an+bn)
            direction=0. if min(an,bn)<.02 else (1-float(np.clip(av@bv/(an*bn),-1.,1.)))/2
            losses[1]+=weight*velocity;losses[2]+=weight*direction
            known[1]+=weight;known[2]+=weight;measured_velocity+=1
            if joint=='root':roots.append(dict(tick=ticks[i],left_velocity=av.tolist(),
                right_velocity=bv.tolist(),velocity_difference=difference,direction_error=direction,
                both_nonstationary=min(an,bn)>=.02))
    total=sum(weights.values());denominators=[len(ticks)*total,(len(ticks)-1)*total,(len(ticks)-1)*total]
    intervals=[(loss/d,(loss+max(0.,d-k))/d) for loss,k,d in zip(losses,known,denominators)]
    cost=tuple(2*intervals[0][i]+intervals[1][i]+.7*intervals[2][i] for i in (0,1))
    return dict(contract='partial_actual_screen_root_and_joint_motion_v1',
        state='UNMEASURED' if not measured_velocity else 'PARTIAL_OBSERVED_HYPOTHESIS',
        exploratory_cost_interval=list(cost),exploration_score=None if not measured_velocity else cost[1],
        position_cost_interval=list(intervals[0]),velocity_cost_interval=list(intervals[1]),
        direction_cost_interval=list(intervals[2]),root_velocity_samples=roots,
        observed_root_direction_error=None if not roots else sum(r['direction_error'] for r in roots)/len(roots),
        measured_velocity_requests=measured_velocity,coverage=known[0]/denominators[0],
        actual_screen_root_translation_removed=False,missing_observations_interpolated=False,
        full_action_certified=False,quality_approved=False)


def measure_review_projected_action(left,right,*,left_queries,right_queries,fps,declaration,route):
    if (not isinstance(declaration,FrozenReviewActionDeclaration)
            or not isinstance(route,FrozenReviewNativeMotionRoute)
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0):
        raise ValueError('projected action needs independently sealed requests and a finite output clock')
    ticks=declaration.payload['ticks'];joints=declaration.payload['requested_joint_indices']
    if (route['requested_ticks']!=ticks or len(left)!=len(ticks) or len(right)!=len(ticks)
            or set(left_queries)!=set(ticks) or set(right_queries)!=set(ticks)):
        raise ValueError('projected action omitted or changed its frozen request inventory')
    result=dict(contract='actual_registered_camera_anatomical_action_evidence_v1',state='UNKNOWN',
        declaration_identity=declaration.identity,route_identity=route.identity,
        units='normalized_output_width_aspect_corrected_position_relative_velocity_v1',
        position_request_ids=[r['request_id'] for r in route['position_requests']],
        velocity_request_ids=[r['request_id'] for r in route['velocity_requests']],
        projected_position_error=None,projected_velocity_error=None,projected_action_error=None,
        complete_on_declared_basis=False,quality_approved=False)
    if not route['comparison_indices']:return result
    points=[];bindings=[];missing=[];outside=[]
    for side,views,queries in (('left',left,left_queries),('right',right,right_queries)):
        values={}
        for i in route['comparison_indices']:
            t=ticks[i];v=views[i];q=queries[t];expected=route[side+'_native'][i]
            if v is not None:
                raw=v['row']['native_observation']
                actual=[raw['request_source_index'],raw['source_clock_identity'],digest(raw['source_file_identity']),
                    v.get('native_camera_unit_index'),raw['native_frame_index']]
                if actual!=expected:raise ValueError('projected action borrowed another frozen native route')
            xy=project_observed_review_joints(v,q,t);values[i]=xy
            for joint in joints:
                if joint not in xy:missing.append((side,t,joint))
                elif any(not math.isfinite(n) or not 0<=n<=1 for n in xy[joint]):outside.append((side,t,joint))
            bindings.append(dict(side=side,tick=t,registered_query_identity=q.key,
                native_parent=None if v is None else digest(v['row']['native_observation']),
                selected_head=None if v is None else v['descriptor'].frame_local_head_id,
                native_head_identity=None if v is None else next((digest(h) for h in v['row']['native_observation'].get('heads',())
                    if h['observation_id']==v['descriptor'].frame_local_head_id),None),
                anatomical_provenance_identity=None if v is None else v.get('review_joint_evidence_identity'),
                projected_points={str(k):list(p) for k,p in xy.items()}))
        points.append(values)
    result['actual_projected_input_identity']=digest(bindings)
    result['actual_projected_inputs']=bindings
    if missing or outside:return dict(result,state='FAIL' if outside else 'UNKNOWN',
        missing_declared_joint_requests=missing,outside_output_joint_requests=outside)
    aspect=np.asarray((1.,9/16));indices=route['comparison_indices']
    a,b=[np.asarray([[values[i][j] for j in joints] for i in indices])*aspect for values in points]
    dt=np.diff([ticks[i]/fps for i in indices]);av,bv=np.diff(a,axis=0)/dt[:,None,None],np.diff(b,axis=0)/dt[:,None,None]
    position=float(np.linalg.norm(a-b,axis=2).mean())
    an,bn=np.linalg.norm(av,axis=(1,2)),np.linalg.norm(bv,axis=(1,2))
    velocity=float((np.linalg.norm(av-bv,axis=(1,2))/(1.+np.maximum(an,bn))).mean())
    return dict(result,state='MEASURED',projected_position_error=position,projected_velocity_error=velocity,
        projected_action_error=2.*position+velocity,complete_on_declared_basis=True,
        requested_joint_positions=2*len(indices)*len(joints),
        requested_velocity_windows=len(indices)-1,measured_velocity_windows=len(dt))


def fit_review_anatomical_keys(source_points,target_points,*,requested_joints,time_seconds,
        maximum_zoom=2.,maximum_roll_degrees=1.6,required_visible_points=(),maximum_candidates=8):
    """Bounded screen-space similarity fit, not a warp or missing-joint fill.

    The caller chooses the full joint basis BEFORE fitting. Targets are actual
    original entry/exit or actual donor/donor same-time anatomy, not face-only
    centers. Legal pan and scale bounds are renderer bounds, not pose scores.
    """
    if type(maximum_candidates) is not int or not 1<=maximum_candidates<=64:
        raise ValueError('camera candidate budget must be predeclared within 1..64')
    visible=np.asarray(tuple(required_visible_points),dtype=float)
    if visible.size:
        if visible.ndim!=2 or visible.shape[1]!=2 or not np.isfinite(visible).all():
            raise ValueError('known preservation points must be finite observed 2D points')
        if np.any(visible<0.) or np.any(visible>1.):
            raise ValueError('known preservation support is outside the neutral source raster')
    joints=tuple(requested_joints)
    if (joints!=tuple(sorted(set(joints))) or len(joints)<3
            or not 1<=maximum_zoom<=2 or not 0<=maximum_roll_degrees<=1.6):
        raise ValueError('anatomical camera fit needs its predeclared bounded joint basis')
    if any(j not in source_points or j not in target_points for j in joints):return ()
    aspect=np.asarray((1.,9/16));a=np.asarray([source_points[j] for j in joints])*aspect
    b=np.asarray([target_points[j] for j in joints])*aspect
    if not np.isfinite(a).all() or not np.isfinite(b).all():return ()
    ac,bc=a.mean(axis=0),b.mean(axis=0);aa,bb=a-ac,b-bc
    energy=float(np.sum(aa*aa))
    if energy<=1e-12:return ()
    dot=float(np.sum(aa*bb));cross=float(np.sum(aa[:,0]*bb[:,1]-aa[:,1]*bb[:,0]))
    angle=max(-maximum_roll_degrees,min(maximum_roll_degrees,math.degrees(math.atan2(cross,dot))))
    center=np.asarray((.5,.5))*aspect
    # The unconstrained similarity optimum may show a black corner or exceed
    # the pan bound. That does not mean the native pose is unmeasured: solve a
    # finite, independently fixed legal camera family against the SAME joints.
    # Every option still uses the renderer's integrity constructor.
    options=[]
    for roll in sorted({0.,angle}):
        c,s=math.cos(math.radians(roll)),math.sin(math.radians(roll))
        rotation=np.asarray(((c,-s),(s,c)));rotated=aa@rotation.T
        optimum=max(1.,min(maximum_zoom,float(np.sum(rotated*bb))/energy))
        zooms=sorted({1.,optimum,minimum_roll_safe_zoom(roll),1.08,1.16,1.28,1.4,1.75,maximum_zoom})
        for zoom in zooms:
            if not 1.<=zoom<=maximum_zoom:continue
            rotated_center=(ac-center)@rotation.T+center
            desired=(rotated_center-(bc-center)/zoom)/aspect
            inset=.5/zoom;desired=np.clip(desired,max(inset,.25),min(1-inset,.75))
            for alpha in (0.,.25,.5,.75,1.):
                cx,cy=.5+alpha*(desired-.5)
                try:key=EditorialTransformKeyframe(time_seconds,zoom,float(cx),float(cy),roll)
                except ValueError:continue
                if any(not all(0.<=v<=1. for v in project_review_editorial_point(p,key)) for p in visible):
                    continue
                projected=np.asarray([project_review_editorial_point(source_points[j],key) for j in joints])*aspect
                error=float(np.sum((projected-b)**2))
                options.append((error,key))
    result=[];seen=set()
    for _,key in sorted(options,key=lambda p:(p[0],p[1].zoom,abs(p[1].roll_degrees))):
        signature=(key.time_seconds,key.zoom,key.center_x,key.center_y,key.roll_degrees)
        if signature in seen:continue
        seen.add(signature);result.append(key)
        if len(result)==maximum_candidates:break
    return tuple(result)


def fit_review_anatomical_key(source_points,target_points,*,requested_joints,time_seconds,
        maximum_zoom=2.,maximum_roll_degrees=1.6,required_visible_points=()):
    """Compatibility wrapper; supplied actual known points constrain the fit.

    No missing joint is inferred and this does not certify actor or quality.
    The bounded producer should consume the plural feasible-camera API.
    """
    keys=fit_review_anatomical_keys(source_points,target_points,requested_joints=requested_joints,
        time_seconds=time_seconds,maximum_zoom=maximum_zoom,maximum_roll_degrees=maximum_roll_degrees,
        required_visible_points=required_visible_points,maximum_candidates=1)
    return keys[0] if keys else None
