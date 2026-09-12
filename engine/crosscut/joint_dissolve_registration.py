"""Joint time-varying registration of two source-bound overlap trajectories.

This solver never detects or identifies a face and never fills missing boxes.
Inputs must be observed or separately verified by short pixel tracking. It
solves both viewport transforms together, retaining the actual choreography.
"""
from dataclasses import dataclass, replace
import math
import numpy as np
from scipy.optimize import minimize

from .direct_member_transform import attention_mismatch, project_box
from .render import EditorialTransformKeyframe, EditorialTransformTrack


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


def fit_joint_dissolve_registration(left, right, *, fps, maximum_zooms,
        maximum_center_shift=.16, maximum_position_mismatch=.08,
        maximum_scale_mismatch=.06):
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
    if any(not isinstance(s,OverlapFaceObservation) for s in (*left,*right)):
        raise TypeError('joint overlap needs typed source-bound observations')
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
    def unpack(v):return v.reshape(2,n,3)
    def objective(v):
        a=unpack(v);z=a[:,:,0];t=a[:,:,1:]
        output=centers*z[:,:,None]+t
        scale=sizes*z
        pan=t-(1-z[:,:,None])/2
        registration=500.*np.sum((output[0]-output[1])**2)+10.*np.sum(((scale[0]-scale[1])/normalizer)**2)
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
    result=minimize(objective,seed.ravel(),method='SLSQP',bounds=bounds,
        constraints=({'type':'ineq','fun':inside},{'type':'ineq','fun':registration_limits}),
        options={'maxiter':150,'ftol':1e-10})
    audit=dict(contract='joint_temporal_dissolve_registration_v1',solver_success=bool(result.success),
        solver_message=str(result.message),iterations=int(result.nit),output_frame_count=n,
        source_indices=[left[0].source_index,right[0].source_index],quality_approval=False,
        time_shift_applied=False,infeasibility_proven=False,raw_observation_count=sum(s.basis=='observed_focal_face' for s in (*left,*right)),
        pixel_tracked_observation_count=sum(s.basis!='observed_focal_face' for s in (*left,*right)))
    if not result.success or not np.isfinite(result.x).all() or inside(result.x).min() < -1e-7:
        return None,dict(audit,reason='joint_solver_no_verified_solution')
    values=unpack(result.x)
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


def materialize_joint_dissolve_registration(*, left_segment, right_segment,
        left_observations, right_observations, tracks, transition,
        common_master_start, fps, ramp_seconds=.75):
    """Bind measured overlap tracks to the exact source clocks used by render.

The caller must still verify containment over the added nominal ramps and
review rendered pixels. This is explicitly source-measured, not output-QA
approval. Existing cameras are preserved by rejecting conflicting ownership.
    """
    from .render import TransitionKind, MEMBER_SOURCE_MEASURED_JOINT_OVERLAP_AUTHORITY
    if transition.kind is not TransitionKind.FACE_DISSOLVE:
        raise ValueError('joint camera materialization requires a face dissolve')
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
        left_segment.duration <= ramp_seconds+2/fps or right_segment.duration<=duration+ramp_seconds):
        raise ValueError('joint registration lacks nominal camera ramp space')
    left_hold=left_segment.duration-2/fps
    first=tracks[0].keyframes[0]
    left_track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.),
        EditorialTransformKeyframe(left_hold-ramp_seconds,1.),
        replace(first,time_seconds=left_hold),replace(first,time_seconds=left_segment.duration)),
        reason='joint_dissolve_outgoing_ramp_pending_containment_review')
    right_track=EditorialTransformTrack((*tracks[1].keyframes,
        replace(tracks[1].final_keyframe,time_seconds=duration),
        EditorialTransformKeyframe(duration+ramp_seconds,1.),
        EditorialTransformKeyframe(right_segment.duration,1.)),
        reason='joint_dissolve_incoming_ramp_pending_containment_review')
    return (replace(left_segment,editorial_transform=left_track,outgoing_overlap_transform=tracks[0]),
            replace(right_segment,editorial_transform=right_track),
            replace(transition,transform_overlap_authority=MEMBER_SOURCE_MEASURED_JOINT_OVERLAP_AUTHORITY))


def review_joint_dissolve_containment(*, segments, transitions, source_geometries,
        geometry_provider, fps, minimum_face_visible=.98, minimum_body_visible=.72,
        pixel_containment_provider=None, legacy_face_only_maximum_zoom=1.12,
        legacy_face_only_maximum_center_shift=.025):
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
    counts=quantized_segment_frame_counts(segments,fps)
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
