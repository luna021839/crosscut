"""Realize actual donor neighbours jointly, before selecting a two-shot block.

Original entry/exit and donor/donor boundary have distinct targets. The camera
is a whole selected transform, not a post-selection cosmetic refit. Numeric
geometry and native choreography are separate evidence; no quality approval.
"""
from dataclasses import asdict,dataclass,replace
import json
import math
from .group_performer_family_owner import digest
from .group_review_camera_repair import fit_whole_camera_track,transformed_descriptor,geometry_cost
from .group_review_camera_repair import project_review_editorial_point
from .group_review_projected_action import fit_review_anatomical_keys
from .render import EditorialTransformKeyframe,EditorialTransformTrack


def measure_review_partial_joint_trajectory(left,right,*,ticks,fps,requested_joints=(0,1,2,3,4,5)):
    """Exploration evidence for visible anatomy; NOT complete action admission.

    Missing joints keep explicit request IDs and a nonzero uncertainty cost.
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
    for t,pa,pb in zip(ticks,a,b):
        for j in requested_joints:
            if j not in pa or j not in pb:continue
            distance=math.dist(pa[j],pb[j]);weight=weights[j]
            error+=distance*weight;known_weight+=weight;observed_articulation+=j>=2
            positions.append(dict(tick=t,joint=j,error=distance))
    for i in range(1,len(ticks)):
        dt=(ticks[i]-ticks[i-1])/fps
        for j in requested_joints:
            if not all(j in p for p in (a[i-1],a[i],b[i-1],b[i])):continue
            av=tuple((a[i][j][k]-a[i-1][j][k])/dt for k in (0,1))
            bv=tuple((b[i][j][k]-b[i-1][j][k])/dt for k in (0,1))
            an,bn=math.hypot(*av),math.hypot(*bv)
            velocity=math.dist(av,bv)/(1+max(an,bn));velocities.append(velocity)
            if j>=2 and min(an,bn)>.05:
                directions.append(1-max(-1.,min(1.,sum(x*y for x,y in zip(av,bv))/(an*bn))))
    expected=len(ticks)*sum(weights.values());coverage=known_weight/expected
    score=None if observed_articulation<3 else error/max(known_weight,1e-9)+.25*(
        sum(velocities)/len(velocities) if velocities else 1.)+.35*(
        sum(directions)/len(directions) if directions else 0.)+.8*(1-coverage)
    return dict(contract='partial_actual_joint_trajectory_exploration_v1',
        state='UNMEASURED' if score is None else 'PARTIAL_OBSERVED_HYPOTHESIS',exploration_score=score,
        requested_ticks=list(ticks),requested_joint_indices=list(requested_joints),
        missing_joint_requests=missing,measured_position_requests=positions,coverage=coverage,
        measured_velocity_requests=len(velocities),expected_velocity_requests=(len(ticks)-1)*len(requested_joints),
        missing_imputed_as_zero=False,missing_joint_trajectory_interpolated=False,
        full_action_certified=False,actor_identity_certified=False,quality_approved=False)


def fit_review_temporal_camera_corridor(desired_keys,*,observed_points_by_frame,locked_indices=(),fps):
    """Preserve observed support at its own time, not a cross-time union.

    A moving head/hand need not remain inside every OTHER instant's camera.
    Finite whole-path optimization trades intervention against camera speed;
    missing observations remain explicitly unmeasured, never support PASS.
    """
    if (not isinstance(desired_keys,(list,tuple)) or not 2<=len(desired_keys)<=160
            or len(observed_points_by_frame)!=len(desired_keys)
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0
            or any(not isinstance(k,EditorialTransformKeyframe) or abs(k.time_seconds-i/fps)>1e-9
                for i,k in enumerate(desired_keys))
            or tuple(sorted(set(locked_indices)))!=locked_indices
            or any(type(i) is not int or not 0<=i<len(desired_keys) for i in locked_indices)):
        raise ValueError('temporal camera corridor needs its exact bounded per-frame request')
    choices=[];unknown=[]
    for i,(key,points) in enumerate(zip(desired_keys,observed_points_by_frame)):
        if points is None:points=();unknown.append(i)
        if (not isinstance(points,(tuple,list)) or len(points)>32
                or any(len(p)!=2 or not all(math.isfinite(v) and 0<=v<=1 for v in p) for p in points)):
            raise ValueError('temporal support must be directly observed screen points at each instant')
        if not points and i not in unknown:unknown.append(i)
        candidates=[];seen=set()
        for alpha in ((1.,) if i in locked_indices else (1.,.85,.7,.55,.4,.25,0.)):
            try:k=EditorialTransformKeyframe(key.time_seconds,1+(key.zoom-1)*alpha,
                .5+(key.center_x-.5)*alpha,.5+(key.center_y-.5)*alpha,key.roll_degrees*alpha)
            except ValueError:continue
            token=(k.zoom,k.center_x,k.center_y,k.roll_degrees)
            if token in seen:continue
            seen.add(token)
            if all(all(-1e-9<=v<=1+1e-9 for v in project_review_editorial_point(p,k)) for p in points):
                candidates.append((k,.15*(1-alpha)**2))
        if not candidates:return None,dict(reason='no_legal_actual_time_support_camera',frame_index=i,quality_approved=False)
        choices.append(candidates)
    previous=[(local,(k,)) for k,local in choices[0]]
    for candidates in choices[1:]:
        current=[]
        for k,local in candidates:
            options=[]
            for cost,path in previous:
                before=path[-1]
                try:EditorialTransformTrack((replace(before,time_seconds=0.),replace(k,time_seconds=1/fps)),
                    reason='actual_between_frame_camera_integrity')
                except ValueError:continue
                velocity=fps**2*((k.center_x-before.center_x)**2+(k.center_y-before.center_y)**2+
                    .5*math.log(k.zoom/before.zoom)**2+.0001*(k.roll_degrees-before.roll_degrees)**2)
                options.append((cost+local+.02*velocity,path+(k,)))
            if options:current.append(min(options,key=lambda p:p[0]))
        if not current:return None,dict(reason='no_integrity_preserving_temporal_path',quality_approved=False)
        previous=current
    cost,keys=min(previous,key=lambda p:p[0])
    track=EditorialTransformTrack(keys,reason='time_specific_known_support_coupled_focal_camera')
    return track,dict(contract='time_specific_actual_support_camera_corridor_v1',frames=len(keys),
        support_time_slots_kept_separate=True,cross_time_coordinate_union_used=False,
        unmeasured_support_frame_indices=unknown,locked_frame_indices=list(locked_indices),
        maximum_states_per_frame=7,whole_path_cost=cost,known_support_preserved_at_observed_times=True,
        complete_support_certified=not unknown,actor_identity_certified=False,quality_approved=False)


def fit_review_coupled_boundary_keys(*,left_head,right_head,left_joint_points,right_joint_points,
        requested_joints=(0,1,2,3,4,5),left_required_visible_points=(),right_required_visible_points=(),
        maximum_candidates=8,maximum_zoom=2.):
    """Fit BOTH selected cameras to common actual focal anatomy, before render.

    Independently observed missing joints are retained as missing requests.
    Common measured joints may help a geometric hypothesis, but cannot turn
    incomplete action, appearance identity, or artistic attention into PASS.
    No body bounding box or other performer's joints stand in for this actor.
    """
    if (type(maximum_candidates) is not int or not 1<=maximum_candidates<=8
            or requested_joints!=tuple(sorted(set(requested_joints)))
            or not requested_joints or any(type(j) is not int or not 0<=j<6 for j in requested_joints)
            or not 1<=maximum_zoom<=2):
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
    target_head=dict(center=(.5,.5),width=width,height=height);target=face(target_head)
    for j in common:
        target[j]=tuple(.5+height*((a[j][axis]-left_head['center'][axis])/left_head['height']+
            (b[j][axis]-right_head['center'][axis])/right_head['height'])/2 for axis in (0,1))
    basis=tuple(sorted(target))
    left=fit_review_anatomical_keys(a,target,requested_joints=basis,time_seconds=0.,
        required_visible_points=left_required_visible_points,maximum_candidates=maximum_candidates,maximum_zoom=maximum_zoom)
    right=fit_review_anatomical_keys(b,target,requested_joints=basis,time_seconds=0.,
        required_visible_points=right_required_visible_points,maximum_candidates=maximum_candidates,maximum_zoom=maximum_zoom)
    options=[]
    for ka in left:
        for kb in right:
            pa={j:project_review_editorial_point(p,ka) for j,p in a.items()}
            pb={j:project_review_editorial_point(p,kb) for j,p in b.items()}
            point_error=sum(math.hypot(pa[j][0]-pb[j][0],(pa[j][1]-pb[j][1])*9/16) for j in basis)/len(basis)
            size_error=abs(math.log(ka.zoom*left_head['height']/(kb.zoom*right_head['height'])))
            score=2*point_error+.3*size_error+.005*math.log(ka.zoom*kb.zoom)
            options.append((score,ka,kb,point_error,size_error))
    options.sort(key=lambda v:(v[0],v[1].zoom,v[2].zoom,v[1].center_x,v[2].center_x))
    receipt=dict(contract='coupled_actual_focal_boundary_camera_hypotheses_v1',requested_joint_indices=list(requested_joints),
        measured_common_joint_indices=list(common),missing_joint_requests=missing,
        complete_requested_action_basis=not any(missing.values()),camera_pair_count=len(options),
        partial_geometry_is_complete_motion_match=False,actor_identity_certified=False,
        artistic_focal_intent_certified=False,quality_approved=False)
    return tuple((ka,kb,dict(receipt,geometry_score=score,projected_point_error=point_error,
        log_face_scale_error=size_error)) for score,ka,kb,point_error,size_error in options[:maximum_candidates])


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
