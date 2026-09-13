"""Joint time-varying registration of two source-bound overlap trajectories.

This solver never detects or identifies a face and never fills missing faces.
Inputs are observed faces, separately pixel-verified faces, or extents of the
same observed focal joints. These evidence bases are never mixed. It solves
both viewport transforms together, retaining the actual choreography.
"""
from dataclasses import dataclass, replace
import math
import numpy as np
from scipy.optimize import minimize

from .direct_member_transform import attention_mismatch, project_box
from .render import EditorialTransformKeyframe, EditorialTransformTrack
from .joint_dissolve_pose import paired_overlap_pose_arrays, pose_registration_metrics


@dataclass(frozen=True)
class OverlapFaceObservation:
    source_index: int
    master_seconds: float
    source_seconds: float
    box: tuple[float, float, float, float]
    basis: str = 'observed_focal_face'

    def __post_init__(self):
        if isinstance(self.source_index,bool) or not isinstance(self.source_index,int) or self.source_index < 0:
            raise ValueError('overlap source index is invalid')
        if self.basis not in ('observed_focal_face','bidirectionally_tracked_focal_face'):
            raise ValueError('overlap face evidence basis is unsupported')
        if len(self.box)!=4 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
                for v in (self.master_seconds,self.source_seconds,*self.box)):
            raise ValueError('overlap geometry and clocks must be finite')
        x,y,w,h=self.box
        if min(self.master_seconds,self.source_seconds,x,y)<0 or min(w,h)<=0 or x+w>1.000001 or y+h>1.000001:
            raise ValueError('overlap observation is outside its source frame')


@dataclass(frozen=True)
class OverlapPoseAnchorObservation:
    """Extent of the same measured joint IDs, explicitly not a face box."""
    source_index: int
    master_seconds: float
    source_seconds: float
    box: tuple[float, float, float, float]
    common_joint_indices: tuple[int, ...]
    basis: str = 'observed_focal_common_joint_extent'

    def __post_init__(self):
        # Shared scalar/viewport validation does not turn this into face evidence.
        OverlapFaceObservation(self.source_index, self.master_seconds, self.source_seconds, self.box)
        if (self.basis != 'observed_focal_common_joint_extent'
                or len(self.common_joint_indices) < 4
                or tuple(sorted(set(self.common_joint_indices))) != self.common_joint_indices
                or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < 8
                       for i in self.common_joint_indices)):
            raise ValueError('pose anchor needs the same measured joint IDs')


def joint_pose_anchor_observations(pose_observations):
    """Build anatomical extents only with exact focal poses on every frame.

    The source-focal provider, not an anonymous detector or this solver, owns
    target identity. No face is inferred, and missing joints are not filled.
    """
    if pose_observations is None or len(pose_observations) != 2:
        return None, dict(status='exact_pose_missing', common_joint_indices=[])
    if (not pose_observations[0] or len(pose_observations[0]) != len(pose_observations[1])
            or any(p is None for seq in pose_observations for p in seq)):
        return None, dict(status='exact_pose_missing', common_joint_indices=[])
    points, audit = paired_overlap_pose_arrays(pose_observations, pose_observations)
    if points is None:
        return None, audit
    indices = tuple(audit['common_joint_indices'])
    sequences = []
    for side, seq in enumerate(pose_observations):
        anchors = []
        for frame, pose in enumerate(seq):
            low = points[side, frame].min(axis=0)
            size = points[side, frame].max(axis=0) - low
            if min(size) <= 1e-6:
                return None, dict(audit, status='degenerate_common_joint_extent')
            anchors.append(OverlapPoseAnchorObservation(pose.source_index,
                pose.master_seconds, pose.source_seconds,
                tuple(float(v) for v in (*low, *size)), indices))
        sequences.append(tuple(anchors))
    return tuple(sequences), dict(audit, face_detection_claimed=False)


def fit_joint_dissolve_registration(left, right, *, fps, maximum_zooms,
        maximum_center_shift=.16, maximum_position_mismatch=.08,
        maximum_scale_mismatch=.06, pose_observations=None):
    """One bounded solve in affine coordinates z*x+t, with viewport constraints.

Registration error is balanced with modest magnification and camera temporal
smoothness. There is no search over source identities, cuts or time shifts.
Both returned tracks use overlap-relative output frame times.
    """
    left,right=tuple(left),tuple(right)
    n=len(left)
    if (not 2<=n<=64 or len(right)!=n or isinstance(fps,bool) or not math.isfinite(fps) or fps<=0
            or len(maximum_zooms)!=2 or any(not math.isfinite(z) or not 1<=z<=2 for z in maximum_zooms)
            or not 0<maximum_center_shift<=.25
            or not 0<maximum_position_mismatch<=1 or not 0<maximum_scale_mismatch<=1):
        raise ValueError('joint overlap dimensions or limits are invalid')
    if any(not isinstance(s,(OverlapFaceObservation,OverlapPoseAnchorObservation)) for s in (*left,*right)):
        raise TypeError('joint overlap needs typed source-bound observations')
    pose_anchored = isinstance(left[0], OverlapPoseAnchorObservation)
    if any(isinstance(s, OverlapPoseAnchorObservation) != pose_anchored for s in (*left,*right)):
        raise ValueError('joint overlap cannot mix face and anatomical extent bases')
    if pose_anchored:
        expected, _ = joint_pose_anchor_observations(pose_observations)
        if expected != (left, right):
            raise ValueError('pose anchors must derive from these exact common joints')
    if left[0].source_index==right[0].source_index:
        raise ValueError('joint overlap must connect different sources')
    start=left[0].master_seconds
    for seq in (left,right):
        if any(s.source_index!=seq[0].source_index or abs(s.master_seconds-(start+i/fps))>1e-7 for i,s in enumerate(seq)):
            raise ValueError('joint overlap observations are not output-frame bound')
        if any(b.source_seconds<=a.source_seconds for a,b in zip(seq,seq[1:])):
            raise ValueError('joint overlap source clock must increase')
    boxes=np.array([[s.box for s in seq] for seq in (left,right)],dtype=float)
    centers=boxes[:,:,:2]+boxes[:,:,2:]/2
    sizes=np.sqrt(np.prod(boxes[:,:,2:],axis=-1))
    normalizer=np.maximum(sizes.mean(axis=0),1e-6)
    seed=np.zeros((2,n,3));seed[:,:,0]=1.
    pose_points,pose_audit=paired_overlap_pose_arrays(pose_observations,(left,right))
    raw_pose_error=(np.mean(np.sum((pose_points[0]-pose_points[1])**2,axis=-1))
                    if pose_points is not None else None)
    def unpack(v):return v.reshape(2,n,3)
    def objective(v):
        a=unpack(v);z=a[:,:,0];t=a[:,:,1:]
        output=centers*z[:,:,None]+t
        scale=sizes*z
        pan=t-(1-z[:,:,None])/2
        registration=500.*np.sum((output[0]-output[1])**2)+10.*np.sum(((scale[0]-scale[1])/normalizer)**2)
        if pose_points is not None:
            projected=pose_points*z[:,:,None,None]+t[:,:,None,:]
            registration+=500.*np.sum(np.mean(np.sum((projected[0]-projected[1])**2,axis=-1),axis=-1))
        preference=.2*np.sum((z-1)**2)+.3*np.sum(pan**2)
        smooth=3.*np.sum(np.diff(a,axis=1)**2)
        if n>2:smooth+=8.*np.sum(np.diff(a,n=2,axis=1)**2)
        return float(registration+preference+smooth)
    def inside(v):
        a=unpack(v);z=a[:,:,0];t=a[:,:,1:]
        # t<=0 and t+z>=1 keep the original frame covering the viewport.
        center_numerator=.5-t-.5*z[:,:,None]
        return np.concatenate((-t.ravel(),(t+z[:,:,None]-1).ravel(),
            (maximum_center_shift**2*z**2-np.sum(center_numerator**2,axis=-1)).ravel()))
    def registration_limits(v):
        a=unpack(v);z=a[:,:,0]
        output=centers*z[:,:,None]+a[:,:,1:]
        scale=sizes*z
        ratio=math.exp(maximum_scale_mismatch*math.log(3.2)/2)
        return np.concatenate(((maximum_position_mismatch*.42)**2-np.sum((output[0]-output[1])**2,axis=-1),
            ratio*scale[0]-scale[1],ratio*scale[1]-scale[0]))
    bounds=[bound for cap in maximum_zooms for _ in range(n) for bound in ((1.,cap),(-1.,0.),(-1.,0.))]
    constraints=[{'type':'ineq','fun':inside},{'type':'ineq','fun':registration_limits}]
    def pose_nonregression(v):
        a=unpack(v)
        projected=pose_points*a[:,:,None,:1]+a[:,:,None,1:]
        return raw_pose_error-np.mean(np.sum((projected[0]-projected[1])**2,axis=-1))
    if pose_points is not None:
        constraints.append({'type':'ineq','fun':pose_nonregression})
    result=minimize(objective,seed.ravel(),method='SLSQP',bounds=bounds,
        constraints=tuple(constraints),
        options={'maxiter':150,'ftol':1e-10})
    audit=dict(contract='joint_temporal_dissolve_registration_v1',solver_success=bool(result.success),
        solver_message=str(result.message),iterations=int(result.nit),output_frame_count=n,
        source_indices=[left[0].source_index,right[0].source_index],quality_approval=False,
        time_shift_applied=False,infeasibility_proven=False,raw_observation_count=sum(s.basis=='observed_focal_face' for s in (*left,*right)),
        pixel_tracked_observation_count=sum(s.basis=='bidirectionally_tracked_focal_face' for s in (*left,*right)),
        pose_anchor_observation_count=sum(isinstance(s,OverlapPoseAnchorObservation) for s in (*left,*right)),
        registration_basis='observed_focal_common_joint_extent' if pose_anchored else 'focal_face')
    audit['focal_pose_registration']=dict(pose_audit,quality_approval=False)
    if not result.success or not np.isfinite(result.x).all() or inside(result.x).min() < -1e-7:
        return None,dict(audit,reason='joint_solver_no_verified_solution')
    values=unpack(result.x)
    if pose_points is not None:
        audit['focal_pose_registration'].update(
            before=pose_registration_metrics(pose_points,seed),
            after=pose_registration_metrics(pose_points,values),
            nonregression_required=True,
        )
        if pose_nonregression(result.x) < -1e-9:
            return None,dict(audit,reason='joint_camera_worsens_focal_pose')
    tracks=[]
    try:
        for side in range(2):
            points=[]
            for i,(zoom,tx,ty) in enumerate(values[side]):
                # Remove only solver floating-point edge noise, not a real pan.
                cx=float(np.clip((.5-tx)/zoom,.5/zoom,1.-.5/zoom))
                cy=float(np.clip((.5-ty)/zoom,.5/zoom,1.-.5/zoom))
                points.append(EditorialTransformKeyframe(i/fps,float(zoom),cx,cy))
            tracks.append(EditorialTransformTrack(tuple(points),reason='joint_temporal_dissolve_registration_v1'))
    except ValueError:
        return None,dict(audit,reason='joint_transform_renderer_invalid')
    if pose_points is not None:
        # Verify the actual renderer tracks after removing floating-point edge
        # noise, not just the optimizer's pre-materialization affine values.
        rendered_affine=np.array([[[p.zoom,.5-p.center_x*p.zoom,.5-p.center_y*p.zoom]
            for p in track.keyframes] for track in tracks])
        audit['focal_pose_registration']['after']=pose_registration_metrics(pose_points,rendered_affine)
        if pose_nonregression(rendered_affine.ravel()) < -1e-9:
            return None,dict(audit,reason='joint_camera_worsens_focal_pose')
    residuals=[]
    for i in range(n):
        projected=[]
        for side in range(2):
            p=tracks[side].evaluate(i/fps)
            projected.append(project_box(boxes[side,i],zoom=p.zoom,center_x=p.center_x,center_y=p.center_y))
        residuals.append(attention_mismatch(*projected))
    audit.update(maximum_position_mismatch=max(v[0] for v in residuals),maximum_scale_mismatch=max(v[1] for v in residuals),
        maximum_zooms=[max(p.zoom for p in track.keyframes) for track in tracks])
    if audit['maximum_position_mismatch']>maximum_position_mismatch+1e-9 or audit['maximum_scale_mismatch']>maximum_scale_mismatch+1e-9:
        return None,dict(audit,reason='joint_registration_residual_exceeded')
    return tuple(tracks),dict(audit,reason='measured_joint_registration')


def compose_joint_dissolve_cameras(incoming, outgoing, *, incoming_frame_count, fps):
    """Join owned entry/exit cameras without changing either rendered overlap.

    Keep the ordinary identity plateau when the ramps fit. If they intersect,
    connect the pinned entry and exit states directly instead of imposing an
    unnecessary zoom-out/zoom-in. The resulting nominal path still needs dense
    containment and decoded motion QA; composition is not quality approval.
    """
    if (incoming.reason != 'joint_dissolve_incoming_ramp_pending_containment_review'
            or outgoing.reason != 'joint_dissolve_outgoing_ramp_pending_containment_review'
            or isinstance(incoming_frame_count, bool) or not isinstance(incoming_frame_count, int)
            or not 2 <= incoming_frame_count <= 64 or isinstance(fps,bool) or not math.isfinite(fps) or fps <= 0):
        raise ValueError('joint composition requires owned incoming/outgoing cameras')
    if abs(incoming.final_keyframe.time_seconds-outgoing.final_keyframe.time_seconds)>1e-9:
        raise ValueError('joint cameras do not share a segment clock')
    # Include the first pure incoming frame, not just the last blended frame.
    entry_end=incoming_frame_count/fps
    exit_start=outgoing.keyframes[-2].time_seconds
    if entry_end >= exit_start-1e-9:
        raise ValueError('joint camera pinned overlap intervals collide')
    if (len(incoming.keyframes)<incoming_frame_count+2 or
            abs(incoming.keyframes[incoming_frame_count].time_seconds-entry_end)>1e-9):
        raise ValueError('joint incoming camera does not bind the previous overlap')
    incoming_ramp_end=incoming.keyframes[incoming_frame_count+1].time_seconds
    outgoing_ramp_start=outgoing.keyframes[-3].time_seconds
    if incoming_ramp_end <= outgoing_ramp_start+1e-9:
        prefix=incoming.keyframes[:incoming_frame_count+2]
        suffix=outgoing.keyframes[-3:]
        if abs(incoming_ramp_end-outgoing_ramp_start)<=1e-9:
            suffix=suffix[1:]
    else:
        prefix=incoming.keyframes[:incoming_frame_count+1]
        suffix=outgoing.keyframes[-2:]
    return EditorialTransformTrack((*prefix,*suffix),
        reason='joint_dissolve_composed_ramps_pending_containment_review')


def materialize_joint_dissolve_registration(*, left_segment, right_segment,
        left_observations, right_observations, tracks, transition,
        common_master_start, fps, ramp_seconds=.75, incoming_transition=None):
    """Bind measured overlap tracks to the exact source clocks used by render.

The caller must still verify containment over the added nominal ramps and
review rendered pixels. This is explicitly source-measured, not output-QA
approval. Only the immediately previous joint-owned incoming camera may be
composed, with its exact transition supplied by the automatic stage.
    """
    from .render import TransitionKind, MEMBER_SOURCE_MEASURED_JOINT_OVERLAP_AUTHORITY
    if transition.kind is not TransitionKind.FACE_DISSOLVE:
        raise ValueError('joint camera materialization requires a face dissolve')
    previous_camera=left_segment.editorial_transform
    if incoming_transition is not None:
        if (incoming_transition.kind is not TransitionKind.FACE_DISSOLVE or
                incoming_transition.transform_overlap_authority != MEMBER_SOURCE_MEASURED_JOINT_OVERLAP_AUTHORITY or
                previous_camera is None or previous_camera.reason != 'joint_dissolve_incoming_ramp_pending_containment_review'):
            raise ValueError('joint composition lacks previous registered transition')
        left_segment=replace(left_segment,editorial_transform=None)
    if any(s.editorial_transform is not None or s.outgoing_overlap_transform is not None
           or s.crop is not None or s.pan_crop is not None for s in (left_segment,right_segment)):
        raise ValueError('joint camera materialization cannot overwrite an existing camera')
    if len(tracks)!=2 or any(not isinstance(t,EditorialTransformTrack) for t in tracks):
        raise TypeError('joint camera materialization needs two renderer tracks')
    count=transition.frame_count
    if any(len(seq)!=count for seq in (left_observations,right_observations)):
        raise ValueError('joint observations do not cover the rendered overlap')
    cut=common_master_start+right_segment.master_start
    for seq,segment,local_start in ((left_observations,left_segment,left_segment.source_end),
                                    (right_observations,right_segment,right_segment.source_start)):
        for i,sample in enumerate(seq):
            if (sample.source_index!=segment.source_index or abs(sample.master_seconds-(cut+i/fps))>1e-7
                or abs(sample.source_seconds-(local_start+i/fps*segment.source_playback_rate))>.25/fps):
                raise ValueError('joint registration does not bind the rendered source clocks')
    duration=count/fps
    if (not math.isfinite(ramp_seconds) or ramp_seconds<=0 or
        left_segment.duration <= 2/fps or right_segment.duration<=duration):
        raise ValueError('joint registration lacks nominal camera ramp space')
    left_hold=left_segment.duration-2/fps
    first=tracks[0].keyframes[0]
    ramp_start=max(0.,left_hold-ramp_seconds)
    plateau=(EditorialTransformKeyframe(ramp_start,1.),) if ramp_start>0. else ()
    left_track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.),*plateau,
        replace(first,time_seconds=left_hold),replace(first,time_seconds=left_segment.duration)),
        reason='joint_dissolve_outgoing_ramp_pending_containment_review')
    return_time=min(duration+ramp_seconds,right_segment.duration)
    tail=(EditorialTransformKeyframe(right_segment.duration,1.),) if return_time<right_segment.duration else ()
    right_track=EditorialTransformTrack((*tracks[1].keyframes,
        replace(tracks[1].final_keyframe,time_seconds=duration),
        EditorialTransformKeyframe(return_time,1.),*tail),
        reason='joint_dissolve_incoming_ramp_pending_containment_review')
    if incoming_transition is not None:
        left_track=compose_joint_dissolve_cameras(previous_camera,left_track,
            incoming_frame_count=incoming_transition.frame_count,fps=fps)
    return (replace(left_segment,editorial_transform=left_track,outgoing_overlap_transform=tracks[0]),
            replace(right_segment,editorial_transform=right_track),
            replace(transition,transform_overlap_authority=MEMBER_SOURCE_MEASURED_JOINT_OVERLAP_AUTHORITY))


def shorten_joint_incoming_return(camera, transition, containment, *, fps,
        minimum_return_frames=4, maximum_zoom_rate=1.0, maximum_center_speed=.50):
    """End a return ramp before unsafe footage, preserving the entire overlap.

    A fixed .75-second return is not an editing requirement. This constructs
    one bounded alternate camera from already-measured failure times; it does
    not waive containment, retry registration or change the source/cut. The
    caller must review every transformed frame of the alternate camera again.
    """
    audit=dict(contract='joint_incoming_visibility_timed_return_v1',
        adjusted=False,quality_approval=False,overlap_states_preserved=False)
    from .render import TransitionKind
    if (not isinstance(camera,EditorialTransformTrack)
            or camera.reason!='joint_dissolve_incoming_ramp_pending_containment_review'
            or transition.kind is not TransitionKind.FACE_DISSOLVE
            or isinstance(fps,bool) or not math.isfinite(fps) or fps<=0
            or isinstance(minimum_return_frames,bool) or not isinstance(minimum_return_frames,int)
            or minimum_return_frames<4
            or not all(math.isfinite(v) and v>0 for v in (maximum_zoom_rate,maximum_center_speed))
            or (camera.final_keyframe.zoom,camera.final_keyframe.center_x,
                camera.final_keyframe.center_y,camera.final_keyframe.roll_degrees)!=(1.,.5,.5,0.)):
        return None,dict(audit,reason='unsupported_camera_return_domain')
    count=transition.frame_count
    failures=[row for row in containment.get('frames',()) if not row.get('passed',False)]
    if (not failures or any(row.get('segment_index')!=1 or row.get('scope')!='nominal'
                           for row in failures)
            or len(camera.keyframes)<count+2
            or abs(camera.keyframes[count].time_seconds-count/fps)>1e-9):
        return None,dict(audit,reason='failure_not_confined_to_owned_incoming_return')
    first_unsafe=min(row['output_local_seconds'] for row in failures)
    return_time=(math.ceil(first_unsafe*fps-1e-9)-1)/fps
    pinned=camera.keyframes[count]
    duration=return_time-pinned.time_seconds
    if (duration<minimum_return_frames/fps-1e-9
            or return_time>=camera.keyframes[count+1].time_seconds-1e-9
            or return_time>=camera.final_keyframe.time_seconds):
        return None,dict(audit,reason='insufficient_verified_return_frames')
    zoom_rate=abs(pinned.zoom-1.)/duration
    center_speed=math.hypot(pinned.center_x-.5,pinned.center_y-.5)/duration
    if (zoom_rate>maximum_zoom_rate+1e-9 or center_speed>maximum_center_speed+1e-9
            or abs(pinned.roll_degrees)>1e-9):
        return None,dict(audit,reason='short_return_would_exceed_camera_speed')
    try:
        track=EditorialTransformTrack((*camera.keyframes[:count+1],
            EditorialTransformKeyframe(return_time,1.),camera.final_keyframe),reason=camera.reason)
    except ValueError:
        return None,dict(audit,reason='short_return_renderer_invalid')
    return track,dict(audit,adjusted=True,overlap_states_preserved=True,
        reason='visibility_timed_return_pending_containment',
        first_unsafe_output_seconds=first_unsafe,return_end_seconds=return_time,
        previous_return_end_seconds=camera.keyframes[count+1].time_seconds,
        return_frame_count=int(round(duration*fps)),zoom_rate=zoom_rate,center_speed=center_speed)


def review_joint_dissolve_containment(*, segments, transitions, source_geometries,
        geometry_provider, fps, minimum_face_visible=.98, minimum_body_visible=.72,
        pixel_containment_provider=None, legacy_face_only_maximum_zoom=1.12,
        legacy_face_only_maximum_center_shift=.025, frame_counts=None):
    """Review every transformed output frame on its actual render source clock.

The provider accepts (request source index, source-local seconds), NOT master
time. A missing face may use an independently source-bound focal body, without
pretending a face was detected. No face/body evidence is an explicit failure.
This is containment evidence, not a final artistic/pixel quality judgement.
    """
    from .direct_member_editorial_motion import DirectMemberFaceSample
    from .direct_member_target_face import FOCAL_FACE_BASIS
    from .focal_selector_geometry import FOCAL_SELECTOR_BODY_BASIS
    from .direct_member_transform import map_box_to_portrait_fill,box_visible_fraction
    from .render import quantized_segment_frame_counts
    from .focal_pixel_containment_support import FocalPixelContainmentSupport
    segments=tuple(segments);transitions=tuple(transitions)
    counts=quantized_segment_frame_counts(segments,fps) if frame_counts is None else tuple(frame_counts)
    if len(counts)!=len(segments) or any(isinstance(n,bool) or not isinstance(n,int) or n<2 for n in counts):
        raise ValueError('containment needs exact full-plan frame counts')
    rows=[];failures=[]
    for index,segment in enumerate(segments):
        schedules=[]
        if segment.editorial_transform is not None:
            schedules.extend(('nominal',n/fps,segment.editorial_transform.evaluate(n/fps)) for n in range(counts[index]))
        if segment.outgoing_overlap_transform is not None:
            schedules.extend(('overlap',n/fps,segment.outgoing_overlap_transform.evaluate(n/fps)) for n in range(transitions[index].frame_count))
        for scope,seconds,state in schedules:
            if state.zoom==1. and state.center_x==.5 and state.center_y==.5 and state.roll_degrees==0.:
                continue
            local=(segment.source_start if scope=='nominal' else segment.source_end)+seconds*segment.source_playback_rate
            sample=geometry_provider(segment.source_index,local)
            row=dict(segment_index=index,scope=scope,output_local_seconds=seconds,source_local_seconds=local)
            widened=(state.zoom>legacy_face_only_maximum_zoom+1e-9 or
                math.hypot(state.center_x-.5,state.center_y-.5)>legacy_face_only_maximum_center_shift+1e-9)
            valid=isinstance(sample,DirectMemberFaceSample) and sample.alignment_geometry_basis in (FOCAL_FACE_BASIS,FOCAL_SELECTOR_BODY_BASIS)
            if abs(state.roll_degrees)>1e-9:
                row.update(passed=False,reason='rotated_containment_not_measured')
            elif not valid:
                support=pixel_containment_provider(segment.source_index,local) if pixel_containment_provider else None
                if (isinstance(support,FocalPixelContainmentSupport) and support.source_index==segment.source_index
                    and abs(support.source_seconds-local)<=1e-9):
                    fraction=box_visible_fraction(map_box_to_portrait_fill(support.bbox_xywh,source_geometries[segment.source_index]),zoom=state.zoom,center_x=state.center_x,center_y=state.center_y)
                    row.update(passed=fraction>=minimum_face_visible and not widened,
                        reason='widened_subject_evidence_missing' if widened else 'bidirectional_focal_pixel_containment',
                        face_visible_fraction=fraction,bidirectional_iou=support.bidirectional_iou,
                        anchor_source_seconds=list(support.anchor_source_seconds),face_detection_claimed=False)
                else:
                    row.update(passed=False,reason='focal_geometry_missing')
            else:
                face_fraction=None;body_fraction=None
                geometry=source_geometries[segment.source_index]
                if sample.face_count==1:
                    face_fraction=box_visible_fraction(map_box_to_portrait_fill(sample.bbox_xywh,geometry),zoom=state.zoom,center_x=state.center_x,center_y=state.center_y)
                if sample.subject_bbox_xywh is not None and sample.subject_confidence>=.30:
                    body_fraction=box_visible_fraction(map_box_to_portrait_fill(sample.subject_bbox_xywh,geometry),zoom=state.zoom,center_x=state.center_x,center_y=state.center_y)
                passed=(face_fraction is not None or body_fraction is not None) and (face_fraction is None or face_fraction>=minimum_face_visible) and (body_fraction is None or body_fraction>=minimum_body_visible) and (not widened or body_fraction is not None)
                row.update(passed=passed,face_visible_fraction=face_fraction,body_visible_fraction=body_fraction,
                    basis=sample.alignment_geometry_basis,reason='measured_containment' if passed else 'focal_geometry_cropped')
            rows.append(row)
            if not row['passed']:failures.append(row)
    return dict(contract='joint_dissolve_render_clock_containment_v1',passed=bool(rows) and not failures,
        examined_frame_count=len(rows),failure_count=len(failures),frames=rows,quality_approval=False)
