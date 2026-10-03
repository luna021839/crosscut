"""Registered raw17 focal motion, distinct from per-joint texture paths.

The source-local face pixel path resolves a conditional owner. Fresh registered
pose measurements describe its arms; an LK contradiction check is a DIFFERENT
sensor and missing LK is not a fabricated zero or successful part path. This
module never manufactures the legacy pixel-verified formation motion type.
Native model observations do not certify physical identity, masks or quality.
"""
from dataclasses import dataclass,fields
from hashlib import sha256
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .group_joint_native_observer import NativeGroupObservation,NativeGroupHead
from .group_joint_person_ownership import OwnedPersonObservation,ownership_policy_fingerprint
from .group_joint_temporal_pose import (FrozenPoseMotionDutiesV1,NativePoseTrackEvidenceV1,
    PoseJointObservationState,native_pose_windows_from_rows,native_pose_association_graph)

CONTRACT='source_native_focal_registered_raw17_pose_sequence_v1'
VIEW_CONTRACT='actual_output_focal_registered_pose_sequence_view_v1'
ARM_JOINTS={'left_arm':(5,7,9),'right_arm':(6,8,10)}
POLICY=dict(contract='registered_pose_and_separate_image_contradiction_sensor_v1',
    minimum_raw_joint_confidence=.3,maximum_native_samples=14,
    image_maximum_width=1280,minimum_roi_contrast_std=2.,
    maximum_working_forward_backward_error=1.25,
    maximum_pose_image_residual_shoulder_fraction=.18,
    minimum_pose_image_residual_working_pixels=2.,
    part_roi_maximum_features=16,part_roi_feature_bins=4,
    part_roi_feature_sampling='bounded_equal_longitudinal_bins_v1',
    part_roi_minimum_observed_features=3,
    part_roi_minimum_longitudinal_coverage=.35,
    part_roi_shoulder_span_radius_fraction=.08,part_roi_minimum_radius=6.,part_roi_maximum_radius=15.,
    mandatory_lk_corners_per_joint=False,missing_lk_is_not_zero_or_pixel_verification=True,
    calibrated_anatomy_or_professional_quality=False)


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def policy_sha256():
    return digest(dict(policy=POLICY,implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        opencv_version=cv2.__version__))


def native_measurement_domain(domain):
    """Unused alignment fields cannot own source-native sensor measurements.

    Actual renderer joins and frozen complete cameras are separately bound by
    the output view/parent. This native-only basis makes final source replay
    identical without importing audio phase authority into pose kinematics.
    """
    from .group_source_domain import GroupSourceDomain
    return GroupSourceDomain(domain.request_sources,domain.eligible_indices,
        (0.,)*len(domain.eligible_indices),(None,)*len(domain.eligible_indices),
        domain.scene_cut_times,domain.render_frame_clocks)


def registered_raw17_basis(row):
    """A real bound model input contract; legacy charts are not upgraded."""
    from .group_native_head_orientation import native_head_analysis_spec,SUPPORTED_ANALYSIS_ROLLS
    names=set(dict(row.model_content_sha256)) if isinstance(row,NativeGroupObservation) else set()
    head_names={native_head_analysis_spec(roll)[0] for roll in (None,*SUPPORTED_ANALYSIS_ROLLS)}
    return (isinstance(row,NativeGroupObservation)
        and names==set(dict(row.model_adapter_fingerprints)) and len(names)==3
        and names-{'nanodet_people','rtmpose_body7'}<=head_names
        and {'nanodet_people','rtmpose_body7'}<=names
        and row.person_ownership_policy_sha256==ownership_policy_fingerprint())


def restore_native_row(value):
    """Exact serialized preimage restoration, never a permissive flag importer."""
    def unpack(cls,item):
        return {field.name:item[field.name] for field in fields(cls) if field.init and field.name in item}
    people=[]
    nested=('upper_body_xyc','source_pixel_to_pose_input_affine','raw_pose_xyc','part_states')
    flat=('box_xywh','candidate_head_observation_ids','competing_person_observation_ids')
    for item in value['people']:
        data=unpack(OwnedPersonObservation,item)
        for key in nested:data[key]=tuple(tuple(p) for p in data.get(key,()))
        for key in flat:data[key]=tuple(data.get(key,()))
        if data.get('pose_crop_xywh') is not None:data['pose_crop_xywh']=tuple(data['pose_crop_xywh'])
        if data.get('temporal_motion') is not None:
            raise ValueError('raw model pose cannot import pixel-verified projected formation motion')
        people.append(OwnedPersonObservation(**data))
    heads=[]
    for item in value['heads']:
        data=unpack(NativeGroupHead,item)
        data['box_xywh']=tuple(data['box_xywh'])
        data['landmarks_xy']=tuple(tuple(p) for p in data['landmarks_xy'])
        data['candidate_body_observation_ids']=tuple(data.get('candidate_body_observation_ids',()))
        heads.append(NativeGroupHead(**data))
    data=unpack(NativeGroupObservation,value)
    data.update(people=tuple(people),heads=tuple(heads))
    for key in ('frame_size_wh','source_file_identity'):data[key]=tuple(data[key])
    for key in ('model_content_sha256','model_adapter_fingerprints'):
        data[key]=tuple(tuple(p) for p in data.get(key,()))
    if 'raw_person_detector_result' in value:
        data['raw_person_detector_json']=json.dumps(value['raw_person_detector_result'],sort_keys=True,allow_nan=False)
    row=NativeGroupObservation(**data)
    if digest(row.to_dict())!=digest(value):
        raise ValueError('raw pose restoration changed its exact observation/uncertainty preimage')
    return row


def _working(pixels):
    h,w=pixels.shape[:2]
    if w>POLICY['image_maximum_width']:
        factor=POLICY['image_maximum_width']/w
        pixels=cv2.resize(pixels,(POLICY['image_maximum_width'],round(h*factor)),interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)


def _required_bones(required_joints):
    return tuple((name,(a,b)) for name,joints in ARM_JOINTS.items() if set(joints)<=set(required_joints)
        for a,b in zip(joints,joints[1:]))


def _bone_geometry(previous,current,joints,size):
    before=np.asarray([previous.raw_pose_xyc[n][:2] for n in joints],float)*size
    after=np.asarray([current.raw_pose_xyc[n][:2] for n in joints],float)*size
    span=float(np.linalg.norm((np.asarray(previous.raw_pose_xyc[6][:2])-
        previous.raw_pose_xyc[5][:2])*size))
    radius=min(POLICY['part_roi_maximum_radius'],max(POLICY['part_roi_minimum_radius'],
        span*POLICY['part_roi_shoulder_span_radius_fraction']))
    return before,after,radius


def _bone_projection(points,bone):
    vector=bone[1]-bone[0];length=float(np.linalg.norm(vector))
    if length<1.:
        return None,None,None
    unit=vector/length;normal=np.asarray((-unit[1],unit[0]))
    offsets=np.asarray(points,float)-bone[0]
    return (offsets*unit).sum(axis=-1)/length,(offsets*normal).sum(axis=-1),normal


def _bone_constraint_summary(value,previous,current,size,limit):
    """Rederive conditional regional-flow algebra, not a physical arm mask."""
    name=value['part_name'];joints=tuple(value['raw_joint_indices'])
    before,after,radius=_bone_geometry(previous,current,joints,size)
    attempts=value['attempts']
    if (not isinstance(attempts,list) or len(attempts)>POLICY['part_roi_maximum_features']
            or value['corridor_half_width_working_pixels']!=radius):
        raise ValueError('required part image constraint changed its bounded native ROI')
    usable=[];residuals=[]
    for attempt in attempts:
        if (not isinstance(attempt,dict) or not isinstance(attempt.get('source_feature_xy'),list)
                or len(attempt['source_feature_xy'])!=2
                or any(type(v) not in (int,float) for v in attempt['source_feature_xy'])):
            raise ValueError('part image feature requires exact finite numeric coordinates')
        point=np.asarray(attempt['source_feature_xy'],float)
        if point.shape!=(2,) or not np.isfinite(point).all() or np.any(point<0) or np.any(point>=size):
            raise ValueError('required part feature must be on its actual native working raster')
        alpha,beta,_=_bone_projection((point,),before)
        if alpha is None:
            raise ValueError('degenerate model bone cannot own an image motion constraint')
        nearest=before[0]+np.clip(alpha[0],0,1)*(before[1]-before[0])
        if float(np.linalg.norm(point-nearest))>radius+1e-8:
            raise ValueError('required part feature escaped the actual model-bone image corridor')
        target=attempt['tracked_target_xy'];back=attempt['backward_feature_xy']
        statuses=(attempt['forward_lk_succeeded'],attempt['backward_lk_succeeded'])
        if any(type(flag) is not bool for flag in statuses):
            raise ValueError('part image flow must keep its actual forward/backward status')
        vectors=[]
        for vector in (target,back):
            if vector is None:vectors.append(None)
            else:
                if (not isinstance(vector,list) or len(vector)!=2
                        or any(type(v) not in (int,float) for v in vector)):
                    raise ValueError('part image flow requires exact numeric coordinates')
                vector=np.asarray(vector,float)
                if vector.shape!=(2,) or not np.isfinite(vector).all():
                    raise ValueError('part image flow coordinates are malformed')
                vectors.append(vector)
        target,back=vectors
        observed=all(statuses) and target is not None and back is not None
        fb=float(np.linalg.norm(back-point)) if observed else None
        observed=bool(observed and fb<=POLICY['maximum_working_forward_backward_error']
            and np.all(target>=0) and np.all(target<size))
        _,_,normal=_bone_projection((after[0],),after)
        if observed and normal is not None:
            predicted=after[0]+alpha[0]*(after[1]-after[0])+beta[0]*normal
            residual=float(np.linalg.norm(target-predicted))
            usable.append(float(alpha[0]));residuals.append(residual)
    coverage=max(usable)-min(usable) if usable else 0.
    enough=(len(usable)>=POLICY['part_roi_minimum_observed_features']
        and coverage>=POLICY['part_roi_minimum_longitudinal_coverage'])
    median=float(np.median(residuals)) if enough else None
    contradiction=bool(enough and median>limit)
    return dict(part_name=name,raw_joint_indices=list(joints),corridor_half_width_working_pixels=radius,
        observed_feature_count=len(usable),longitudinal_coverage=coverage,
        median_model_image_residual_working_pixels=median,
        status=('observed_contradiction' if contradiction else 'observed_compatible' if enough else 'unknown'),
        contradiction=contradiction)


def _spatial_bone_features(image,mask,before,*,budget):
    """Observe the bone's length, not just its strongest texture cluster.

    The total feature cap and flow/admission limits are unchanged. Per-bin
    relative corner detection keeps weak but real local structure observable;
    empty bins stay empty. Sampling is fixed before any tracking outcome.
    """
    ys,xs=np.nonzero(mask)
    if not len(xs):return None
    alpha,_,_=_bone_projection(np.column_stack((xs,ys)),before)
    if alpha is None:return None
    count=POLICY['part_roi_feature_bins']
    bins=np.minimum(count-1,np.maximum(0,np.floor(alpha*count).astype(int)))
    points=[]
    for index in range(count):
        selected=bins==index
        if not np.any(selected):continue
        region=np.zeros_like(mask);region[ys[selected],xs[selected]]=255
        if budget is not None:budget.spend('pose_comparisons')
        value=cv2.goodFeaturesToTrack(image,maxCorners=POLICY['part_roi_maximum_features']//count,
            qualityLevel=.01,minDistance=3.,mask=region,blockSize=5)
        if value is not None:points.extend(value)
    return np.asarray(points,np.float32).reshape(-1,1,2) if points else None


def _part_image_constraints(a,b,previous,current,required_joints,opts,limit,budget):
    """Local bone corridors, not a whole-body rigid affine or mandatory joint corners.

    Several actual two-dimensional image features must span each required
    bone. Remote texture cannot fill an unobserved arm. This remains a
    model-conditioned consistency sensor, never an anatomical support mask.
    """
    h,w=a.shape;size=np.asarray((w,h),float);result=[]
    for name,joints in _required_bones(required_joints):
        before,after,radius=_bone_geometry(previous,current,joints,size)
        mask=np.zeros_like(a)
        low=np.maximum(0,np.floor(before.min(0)-radius).astype(int))
        high=np.minimum((w,h),np.ceil(before.max(0)+radius+1).astype(int))
        if np.all(high>low) and np.linalg.norm(before[1]-before[0])>=1.:
            xx,yy=np.meshgrid(np.arange(low[0],high[0]),np.arange(low[1],high[1]))
            points=np.column_stack((xx.ravel(),yy.ravel()))
            alpha,_,_=_bone_projection(points,before)
            nearest=before[0]+np.clip(alpha,0,1)[:,None]*(before[1]-before[0])
            selected=np.linalg.norm(points-nearest,axis=1)<=radius
            mask[low[1]:high[1],low[0]:high[0]]=selected.reshape(xx.shape).astype(np.uint8)*255
        points=_spatial_bone_features(a,mask,before,budget=budget)
        attempts=[]
        if points is not None:
            if budget is not None:budget.spend('pose_comparisons',len(points))
            forward,fs,_=cv2.calcOpticalFlowPyrLK(a,b,points,None,**opts)
            backward,bs,_=(cv2.calcOpticalFlowPyrLK(b,a,forward,None,**opts)
                if forward is not None else (None,None,None))
            for i,point in enumerate(points[:,0]):
                finite=lambda data:data is not None and np.isfinite(data[i,0]).all()
                attempts.append(dict(source_feature_xy=list(map(float,point)),
                    tracked_target_xy=list(map(float,forward[i,0])) if finite(forward) else None,
                    backward_feature_xy=list(map(float,backward[i,0])) if finite(backward) else None,
                    forward_lk_succeeded=bool(fs is not None and fs[i,0] and finite(forward)),
                    backward_lk_succeeded=bool(bs is not None and bs[i,0] and finite(backward))))
        value=dict(part_name=name,raw_joint_indices=list(joints),
            corridor_half_width_working_pixels=radius,attempts=attempts)
        value.update(_bone_constraint_summary(value,previous,current,size,limit))
        result.append(value)
    return result


def image_pose_contradiction(previous_pixels,current_pixels,previous_person,current_person,required_joints,*,budget=None):
    """Observe disagreement, not an anatomical part mask or mandatory corners.

    These are uncalibrated sensor-consistency diagnostics. An observable point
    that tracks elsewhere contradicts a jumping model point. An untrackable
    point remains explicitly UNKNOWN; it does not become motion=0 or an arm
    pixel-path certificate. A flat/missing ROI is unsupported image evidence.
    """
    if (any(not isinstance(p,np.ndarray) or p.dtype!=np.uint8 or p.ndim!=3 or p.shape[2]!=3
            for p in (previous_pixels,current_pixels)) or previous_pixels.shape!=current_pixels.shape):
        raise ValueError('pose image consistency requires identical actual native BGR8 rasters')
    a,b=_working(previous_pixels),_working(current_pixels);h,w=a.shape
    size=np.asarray((w,h),float)
    contrasts=[]
    for image,person in ((a,previous_person),(b,current_person)):
        x,y,ww,hh=person.box_xywh
        left,top=max(0,math.floor(x*w)),max(0,math.floor(y*h))
        right,bottom=min(w,math.ceil((x+ww)*w)),min(h,math.ceil((y+hh)*h))
        roi=image[top:bottom,left:right]
        contrasts.append(None if not roi.size else float(roi.std()))
    image_supported=all(v is not None and v>=POLICY['minimum_roi_contrast_std'] for v in contrasts)
    joint_ids=tuple(sorted(set(required_joints)))
    start=np.asarray([previous_person.raw_pose_xyc[n][:2] for n in joint_ids],np.float32)*size
    target=np.asarray([current_person.raw_pose_xyc[n][:2] for n in joint_ids],float)*size
    opts=dict(winSize=(31,31),maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    forward,fs,_=cv2.calcOpticalFlowPyrLK(a,b,start.astype(np.float32).reshape(-1,1,2),None,**opts)
    backward,bs,_=(cv2.calcOpticalFlowPyrLK(b,a,forward,None,**opts)
        if forward is not None else (None,None,None))
    span=float(np.linalg.norm((np.asarray(previous_person.raw_pose_xyc[6][:2])-
        previous_person.raw_pose_xyc[5][:2])*size))
    limit=max(POLICY['minimum_pose_image_residual_working_pixels'],
        POLICY['maximum_pose_image_residual_shoulder_fraction']*span)
    attempts=[]
    for i,joint in enumerate(joint_ids):
        valid=(forward is not None and backward is not None and fs is not None and bs is not None
            and bool(fs[i,0]) and bool(bs[i,0])
            and np.isfinite(forward[i,0]).all() and np.isfinite(backward[i,0]).all())
        fb=float(np.linalg.norm(backward[i,0]-start[i])) if valid else None
        observed=bool(valid and fb<=POLICY['maximum_working_forward_backward_error'])
        residual=float(np.linalg.norm(forward[i,0]-target[i])) if observed else None
        attempts.append(dict(raw_joint_index=joint,lk_point_status='observed' if observed else 'unknown',
            forward_backward_error_working_pixels=fb,model_image_residual_working_pixels=residual,
            contradiction=bool(observed and residual>limit)))
    parts=_part_image_constraints(a,b,previous_person,current_person,required_joints,opts,limit,budget)
    return dict(contract=POLICY['contract'],working_size=[w,h],roi_contrast_std=contrasts,
        image_supported=image_supported,model_image_residual_limit_working_pixels=limit,
        attempts=attempts,required_part_image_constraints=parts,
        required_part_motion_constrained=bool(parts and all(p['status']=='observed_compatible' for p in parts)),
        contradiction=any(p['contradiction'] for p in (*attempts,*parts)),
        anatomical_part_support_certified=False,arm_pixel_path_verified=False)


@dataclass(frozen=True,slots=True)
class SourceNativeFocalPoseSequenceV1:
    window: NativePoseTrackEvidenceV1
    head_ids: tuple[int|None,...]
    head_links_json: str
    image_checks_json: str
    association_graph_json: str
    policy_sha256: str

    def __post_init__(self):
        if (not isinstance(self.window,NativePoseTrackEvidenceV1)
                or self.window.state is not None or self.window.output_ticks is not None
                or not isinstance(self.head_ids,tuple) or len(self.head_ids)!=len(self.window.anchors)
                or any(n is not None and (type(n) is not int or n<0) for n in self.head_ids)
                or self.policy_sha256!=policy_sha256()):
            raise ValueError('focal pose sequence needs source-native typed samples and current registered policy')
        rows=tuple(anchor.observation for anchor in self.window.anchors)
        if (any(not registered_raw17_basis(row) for row in rows)
                or any(b.native_frame_index!=a.native_frame_index+1 for a,b in zip(rows,rows[1:]))):
            raise ValueError('focal pose sequence must observe each native raster on one bound model basis')
        links=json.loads(self.head_links_json);checks=json.loads(self.image_checks_json)
        graph=json.loads(self.association_graph_json)
        digest([links,checks,graph])
        if len(links)!=len(rows)-1 or len(checks) not in (0,len(rows)-1):
            raise ValueError('focal model sequence omitted a native source step')

    @property
    def coordinate_reason(self):
        anchors=self.window.anchors
        for anchor,head in zip(anchors,self.head_ids,strict=True):
            owner=next((p for p in anchor.observation.people if p.observation_id==anchor.person_observation_id),None)
            if owner is None or head is None or owner.unique_head_observation_id!=head:
                return 'OWNER_UNRESOLVED'
        links=json.loads(self.head_links_json)
        if any([a,b] not in link['unique_links'] for a,b,link in zip(self.head_ids,self.head_ids[1:],links)):
            return 'OWNER_PIXEL_CONTINUATION_UNRESOLVED'
        for anchor in anchors:
            if any(anchor.joint_states[n]!=PoseJointObservationState.VISIBLE_SUPPORTED
                    or anchor.torso_local_xy[n] is None for n in self.window.duties.required_joint_indices):
                return 'REQUIRED_MODEL_JOINT_UNKNOWN'
        return None

    @property
    def reason(self):
        if self.coordinate_reason is not None:return self.coordinate_reason
        checks=json.loads(self.image_checks_json)
        if not checks or any(c['image_supported'] is not True for c in checks):return 'NATIVE_ROI_IMAGE_UNSUPPORTED'
        if any(c['contradiction'] for c in checks):return 'MODEL_POSE_IMAGE_DISAGREEMENT'
        if any(c['required_part_motion_constrained'] is not True for c in checks):
            return 'REQUIRED_PART_IMAGE_MOTION_UNCONSTRAINED'
        return None

    def to_dict(self):
        return dict(contract=CONTRACT,policy_sha256=self.policy_sha256,
            measurement_available=self.reason is None,reason=self.reason,
            model_coordinate_measurement_available=self.coordinate_reason is None,
            required_motion_available=self.reason is None,
            native_pose_window=self.window.to_dict(),head_observation_ids=list(self.head_ids),
            native_frame_indices=[a.observation.native_frame_index for a in self.window.anchors],
            raw_observation_sha256=[a.observation.sha256 for a in self.window.anchors],
            native_pixel_sha256=[a.observation.native_pixel_sha256 for a in self.window.anchors],
            head_pixel_links=json.loads(self.head_links_json),image_consistency=json.loads(self.image_checks_json),
            association_alternatives=json.loads(self.association_graph_json),
            native_measurement_domain_excludes_audio_phase_authority=True,
            source_local_owner_basis='unique_native_face_pixel_path_and_frame_local_body_anchor',
            raw_confidence_is_calibrated_probability=False,required_duties_removed=False,
            pose_or_velocity_imputed=False,same_source_arm_pixel_path_verified=False,
            physical_part_masks_or_visibility_certified=False,source_local_identity_certified=False,
            cross_camera_identity_certified=False,body_phase_certified=False,quality_approved=False)


def measure_native_focal_sequence(*,sensors,rows,selected_head_ids,required_part_names,duty_reference):
    """All alternatives survive; one externally selected face owns the view.

    selected_head_ids is (native row index, frame-local head) seed plus final
    endpoint constraint. No model-pose distance chooses a replacement owner.
    """
    if (not isinstance(rows,tuple) or not 3<=len(rows)<=14
            or not isinstance(required_part_names,tuple) or not required_part_names
            or len(set(required_part_names))!=len(required_part_names)
            or not set(required_part_names)<=set(ARM_JOINTS)):
        raise ValueError('focal pose measurement requires three distinct native poses and immutable arm duties')
    domain=native_measurement_domain(sensors.session.domain)
    required=tuple(sorted({5,6,*(n for name in required_part_names for n in ARM_JOINTS[name])}))
    duties=FrozenPoseMotionDutiesV1(duty_reference,required,POLICY['minimum_raw_joint_confidence'])
    graph=native_pose_association_graph(domain=domain,rows=rows,budget=sensors.session.budget)
    links=tuple(sensors._head_links(a,b) for a,b in zip(rows,rows[1:]))
    (seed,head),(last,last_head)=selected_head_ids
    if (any(type(n) is not int for n in (seed,head,last,last_head))
            or not 0<=seed<last<len(rows)):
        raise ValueError('focal pose seed/constraint must refer to actual chronological native anchors')
    heads=[None]*len(rows);heads[seed]=head
    for index in range(seed,len(rows)-1):
        choices=[b for a,b in links[index]['unique_links'] if a==heads[index]]
        heads[index+1]=choices[0] if len(choices)==1 else None
    for index in range(seed-1,-1,-1):
        choices=[a for a,b in links[index]['unique_links'] if b==heads[index+1]]
        heads[index]=choices[0] if len(choices)==1 else None
    if heads[last]!=last_head:heads[last]=None
    owners=[]
    for row,head in zip(rows,heads,strict=True):
        values=[p for p in row.people if head is not None and p.unique_head_observation_id==head]
        owners.append(values[0].observation_id if len(values)==1 else None)
    window=native_pose_windows_from_rows(domain=domain,rows=rows,choices=tuple((owner,) for owner in owners),
        duties=duties,maximum_hypotheses=1)[0]
    checks=[]
    if all(owner is not None for owner in owners) and all(
            anchor.joint_states[n]==PoseJointObservationState.VISIBLE_SUPPORTED
            for anchor in window.anchors for n in required):
        for i,(a,b) in enumerate(zip(rows,rows[1:])):
            aa=sensors.session.native_pixels(a.request_source_index,a.native_frame_index)
            bb=sensors.session.native_pixels(b.request_source_index,b.native_frame_index)
            if [sha256(memoryview(p)).hexdigest() for p in (aa,bb)]!=[a.native_pixel_sha256,b.native_pixel_sha256]:
                raise ValueError('actual model-pose source pixels changed before image consistency')
            before=next(p for p in a.people if p.observation_id==owners[i])
            after=next(p for p in b.people if p.observation_id==owners[i+1])
            sensors.session.budget.spend('pose_comparisons',len(required))
            checks.append(dict(image_pose_contradiction(aa,bb,before,after,required,budget=sensors.session.budget),
                raw_pair=[a.sha256,b.sha256]))
            del aa,bb
    frozen=lambda value:json.dumps(value,sort_keys=True,allow_nan=False)
    return SourceNativeFocalPoseSequenceV1(window,tuple(heads),frozen(links),frozen(checks),
        frozen(graph.to_dict()),policy_sha256())


def focal_pose_view(sensors,state,previous,current,*,output_ticks,selected_head_ids,required_part_names):
    """One registered sensor, separate exact output/camera/duty view.

    A dense one-native step borrows one adjacent sample ONLY inside this
    declared homogeneous scope, not across its real source cut or old wide.
    The third sample is actual inference, not endpoint interpolation.
    """
    from .group_source_changing_owner import project_point
    from .group_joint_source_camera import GroupOutputRole
    if state.role not in (GroupOutputRole.FOCAL_HEAD,GroupOutputRole.FOCAL_HEAD_ARMS):
        raise ValueError('registered focal pose type cannot certify a formation/composition duty')
    state.bind_source(sensors.session.domain)
    if (not isinstance(output_ticks,tuple) or len(output_ticks)!=2
            or any(type(t) is not int for t in output_ticks) or output_ticks[0]>=output_ticks[1]
            or any(not state.first_output_frame<=t<state.end_output_frame for t in output_ticks)
            or tuple(state.native_frame_indices[t-state.first_output_frame] for t in output_ticks)!=
                (previous.native_frame_index,current.native_frame_index)):
        raise ValueError('registered pose view differs from the actual renderer output/native occurrence')
    if not all(registered_raw17_basis(row) for row in (previous,current)):
        raise ValueError('legacy unbound row cannot request the registered native pose backend')
    h=sensors.hypothesis(state);scope=h['shot_intent']['required_output_interval']
    if (not state.first_output_frame<=scope[0]<scope[1]<=state.end_output_frame
            or any(not scope[0]<=t<scope[1] for t in output_ticks)):
        raise ValueError('registered focal pose output occurrence escaped its frozen required scope')
    program=getattr(sensors,'original_duty_program',None)
    # A temporal sensor can need a part before its editorial activation in
    # order to measure its velocity. Its observation mask is the UNION over
    # this output interval, not a retroactive duty for each earlier frame.
    if program is not None:
        from .group_scene_original_duties import duty_at
        expected=tuple(sorted({name for tick in range(output_ticks[0],output_ticks[1]+1)
            for name in duty_at(program,state,tick)['required_part_names']}))
        if required_part_names!=expected:
            raise ValueError('registered pose measurement mask differs from original tick duties')
    first,last=previous.native_frame_index,current.native_frame_index
    if not 0<last-first<=12:
        raise ValueError('registered focal pose needs distinct bounded native endpoint samples')
    lower=state.native_frame_indices[scope[0]-state.first_output_frame]
    upper=state.native_frame_indices[scope[1]-state.first_output_frame-1]
    start,end=first,last
    if end-start+1<3:
        if start>lower:start-=1
        elif end<upper:end+=1
        else:raise ValueError('declared focal scope has no third distinct native pose sample')
    rows=tuple(sensors.raw_native(state,n,h['analysis_roll']) for n in range(start,end+1))
    if rows[first-start]!=previous or rows[last-start]!=current:
        raise ValueError('registered pose current endpoint raw preimages changed')
    # Output camera values NEVER own this cache. Sharing is allowed only for
    # identical source/model/variant/native/seed/required-part/policy inputs.
    reference='original_homogeneous_focal_scope:'+digest(dict(span=h['span_key'],
        scope=scope,role=h['role'],intent=h['shot_intent'],analysis_roll=h['analysis_roll'],
        parts=required_part_names))
    cache=getattr(sensors,'focal_pose_native_cache',None)
    if cache is None:cache=sensors.focal_pose_native_cache={}
    # IDs are frame-local. The same pair of numbers in the same borrowed
    # three-row window can select different people when its primary rows move.
    primary_anchors=((first,selected_head_ids[0]),(last,selected_head_ids[1]))
    cache_key=(tuple(row.sha256 for row in rows),primary_anchors,required_part_names,reference,policy_sha256())
    if cache_key not in cache:
        measured=measure_native_focal_sequence(sensors=sensors,rows=rows,
            selected_head_ids=((first-start,selected_head_ids[0]),(last-start,selected_head_ids[1])),
            required_part_names=required_part_names,duty_reference=reference)
        cache[cache_key]=measured
        sensors.session.budget.spend('pose_evidence_bytes',len(json.dumps(measured.to_dict(),sort_keys=True).encode()))
    measured=cache[cache_key];body_ids=[a.person_observation_id for a in measured.window.anchors]
    for index,selected in ((first-start,selected_head_ids[0]),(last-start,selected_head_ids[1])):
        anchor=measured.window.anchors[index]
        owner=next((p for p in rows[index].people if p.observation_id==anchor.person_observation_id),None)
        if (anchor.observation.sha256!=rows[index].sha256
                or measured.head_ids[index] not in (selected,None)
                or measured.reason is None and (owner is None
                    or owner.unique_head_observation_id!=selected or measured.head_ids[index]!=selected)):
            raise ValueError('cached registered pose primary owner differs from requested native anchor')
    output=[];output_required=set()
    if measured.coordinate_reason is None:
        for tick in range(output_ticks[0],output_ticks[1]+1):
            native=state.native_frame_indices[tick-state.first_output_frame];index=native-start
            if not 0<=index<len(rows):
                raise ValueError('registered pose output join escaped the actually observed native sequence')
            head=measured.head_ids[index]
            # Preserve all existing headroom and arm/hand-envelope clipping
            # checks. More generous model sensing cannot waive camera duties.
            sensors.frame(state,tick,head_id=head)
            expected=tuple(sorted({name for _,name in
                sensors.obligations(rows[index],state,tick,head).required_parts}))
            if not set(expected)<=set(required_part_names) or program is None and expected!=required_part_names:
                raise ValueError('registered focal pose required arm lifecycle differs from original raw scope')
            output_required.update(expected)
            person=next(p for p in rows[index].people if p.observation_id==body_ids[index])
            output.append(dict(output_tick=tick,output_seconds=tick/state.output_fps,
                native_frame_index=native,raw_observation_sha256=rows[index].sha256,
                head_observation_id=head,person_observation_id=body_ids[index],
                original_duty_reference=sensors.duty_binding(state,tick)[0],
                projected_required_joint_xy=[[joint,*project_point(person.raw_pose_xyc[joint][:2],state,tick)]
                    for joint in measured.window.duties.required_joint_indices]))
        if tuple(sorted(output_required))!=required_part_names:
            raise ValueError('registered pose required mask was not owned by its actual output interval')
    result=dict(contract=VIEW_CONTRACT,state_key=state.key,complete_state_preimage=state.to_dict(),
        original_duty_program_sha256=None if program is None else program['program_sha256'],
        output_ticks=list(output_ticks),output_fps=state.output_fps,
        primary_native_frames=[first,last],primary_raw_observation_sha256=[previous.sha256,current.sha256],
        selected_head_ids=list(selected_head_ids),required_body_scope='selected_focal_owner',
        required_part_names=list(required_part_names),
        required_body_observation_ids=[[body_ids[first-start]],[body_ids[last-start]]],
        native_source_measurement=measured.to_dict(),actual_output_samples=output,
        measurement_available=measured.reason is None,reason=measured.reason,
        model_coordinate_measurement_available=measured.coordinate_reason is None,
        required_motion_available=measured.reason is None,
        source_native_velocity_basis='actual_raw17_native_seconds_not_output_phase',
        output_camera_velocity_basis='actual_projected_samples_at_renderer_output_seconds',
        repeated_native_samples_are_not_independent_observations=True,
        same_source_arm_pixel_path_verified=False,formation_pixel_motion_authority=False,
        cross_camera_identity_certified=False,body_phase_certified=False,quality_approved=False)
    sensors.session.budget.spend('pose_evidence_bytes',len(json.dumps(result,sort_keys=True).encode()))
    return result


def required_motion_available(value):
    """Explicit dispatch; never set old pixel-flow flags on new model evidence."""
    if not isinstance(value,dict):return False
    if value.get('contract')==VIEW_CONTRACT:
        return value.get('measurement_available') is True and value.get('required_motion_available') is True
    from .group_scene_observations import JOINT_LINK_CONTRACT
    return value.get('contract')==JOINT_LINK_CONTRACT and value.get('complete_required_joint_paths') is True


def validate_focal_pose_view(value,previous,current,*,raw_inventory,source_domain,hypothesis,state_key,
        original_duty_program=None):
    """Received measurement replay, NOT fresh image/model or phase authority.

    The consumer rederives raw ownership, joints, native seconds, all alternatives
    and declared image-check algebra. Independent parent/final MUST reobserve
    pixels and actual model results; these serialized image values cannot prove
    their own history or camera/physical identity by having an internal hash.
    """
    from fractions import Fraction
    from .group_source_domain import GroupSourceDomain
    from .render_frame_schedule import VerifiedRenderFrameClock
    if original_duty_program is not None:
        from .group_scene_original_duties import validate_original_duty_program,duty_at
        validate_original_duty_program(original_duty_program,raw_inventory=raw_inventory)
        if (original_duty_program['eligible_request_indices']!=source_domain['eligible_request_indices']
                or value.get('original_duty_program_sha256')!=original_duty_program['program_sha256']):
            raise ValueError('registered pose view lost its external original duty programme')
    elif value.get('original_duty_program_sha256') is not None:
        raise ValueError('registered pose consumer cannot borrow duties from its received view')
    if (not isinstance(value,dict) or value.get('contract')!=VIEW_CONTRACT
            or value.get('measurement_available') is not True or value.get('reason') is not None
            or value.get('required_motion_available') is not True
            or value.get('model_coordinate_measurement_available') is not True
            or value.get('state_key')!=state_key or value.get('required_body_scope')!='selected_focal_owner'
            or hypothesis['role'] not in ('focal_head','focal_head_and_observed_arms')
            or any(value.get(k) is not False for k in ('same_source_arm_pixel_path_verified',
                'formation_pixel_motion_authority','cross_camera_identity_certified','body_phase_certified','quality_approved'))):
        raise ValueError('registered focal pose view changed its type, duty or evidence authority')
    digest(value)
    first,last=previous['native_frame_index'],current['native_frame_index']
    if (value.get('primary_native_frames')!=[first,last]
            or value.get('primary_raw_observation_sha256')!=[digest(previous),digest(current)]
            or not 0<last-first<=12):
        raise ValueError('registered focal pose primary source-native occurrence changed')
    parts=value.get('required_part_names')
    if (not isinstance(parts,list) or not parts or len(set(parts))!=len(parts) or not set(parts)<=set(ARM_JOINTS)):
        raise ValueError('registered focal pose required arm mask changed')
    proof=value.get('native_source_measurement') or {}
    refs=proof.get('raw_observation_sha256')
    if (proof.get('contract')!=CONTRACT or not isinstance(refs,list) or not 3<=len(refs)<=14
            or any(ref not in raw_inventory for ref in refs)):
        raise ValueError('registered focal pose sequence omitted actual intermediate raw model observations')
    rows=tuple(restore_native_row(raw_inventory[ref]) for ref in refs)
    eligible=source_domain['eligible_request_indices'];clocks=source_domain['render_frame_clocks']
    clock_values=tuple(VerifiedRenderFrameClock(Fraction(c['frame_rate']),c['frame_count'],
        c['packet_grid_sha256'],c['grid_basis']) for c in clocks)
    files=[Path('EXPLICIT_EXCLUDED_SOURCE_NOT_OPENED') for _ in range(max(eligible)+1)]
    for i,path in zip(eligible,source_domain['eligible_sources'],strict=True):files[i]=Path(path)
    # Native sensing uses ONLY these verified native clocks/cuts. Audio warp
    # history/camera projection is independently replayed by the normal parent.
    domain=GroupSourceDomain(tuple(files),tuple(eligible),(0.,)*len(eligible),
        (None,)*len(eligible),tuple(tuple(c) for c in source_domain['scene_cut_times']),clock_values)
    state=restore_view_state(value.get('complete_state_preimage'),domain,hypothesis,state_key)
    natives=[row.native_frame_index for row in rows]
    if (natives!=list(range(natives[0],natives[-1]+1)) or first not in natives or last not in natives
            or natives[first-natives[0]]!=first or natives[last-natives[0]]!=last
            or rows[first-natives[0]].sha256!=digest(previous) or rows[last-natives[0]].sha256!=digest(current)):
        raise ValueError('registered focal pose all-native sequence differs from its primary endpoints')
    window=proof.get('native_pose_window') or {}
    supplied=window.get('duties') or {}
    reference='original_homogeneous_focal_scope:'+digest(dict(span=hypothesis['span_key'],
        scope=hypothesis['shot_intent']['required_output_interval'],role=hypothesis['role'],
        intent=hypothesis['shot_intent'],analysis_roll=hypothesis['analysis_roll'],parts=tuple(parts)))
    duties=FrozenPoseMotionDutiesV1(reference,tuple(sorted({5,6,*(n for name in parts for n in ARM_JOINTS[name])})),
        POLICY['minimum_raw_joint_confidence'])
    if (digest(supplied)!=digest(dict(reference=duties.reference,required_joint_indices=duties.required_joint_indices,
            minimum_joint_confidence=duties.minimum_joint_confidence))):
        raise ValueError('registered focal pose weakened its frozen required raw joints')
    anchors=window.get('anchors')
    if not isinstance(anchors,list) or len(anchors)!=len(rows):
        raise ValueError('registered focal pose omitted its actual model/ownership samples')
    owners=tuple(a.get('person_observation_id') for a in anchors)
    rebuilt=native_pose_windows_from_rows(domain=domain,rows=rows,choices=tuple((pid,) for pid in owners),
        duties=duties,maximum_hypotheses=1)[0]
    heads=proof.get('head_observation_ids');links=proof.get('head_pixel_links');checks=proof.get('image_consistency')
    if (not isinstance(heads,list) or len(heads)!=len(rows) or not isinstance(links,list)
            or len(links)!=len(rows)-1 or not isinstance(checks,list) or len(checks)!=len(rows)-1):
        raise ValueError('registered focal pose omitted source-local owner/image checks')
    for index,(a,b,link,check) in enumerate(zip(rows[:-1],rows[1:],links,checks,strict=True)):
        pair=[a.sha256,b.sha256]
        measured=link.get('measured_links');unique=link.get('unique_links')
        if (link.get('raw_pair')!=pair or not isinstance(measured,list) or not isinstance(unique,list)
                or any(not isinstance(p,list) or len(p)!=2
                    or p[0] not in {h.observation_id for h in a.heads}
                    or p[1] not in {h.observation_id for h in b.heads} for p in measured)
                or len({tuple(p) for p in measured})!=len(measured)
                or unique!=[p for p in measured if sum(q[0]==p[0] for q in measured)==1
                    and sum(q[1]==p[1] for q in measured)==1]):
            raise ValueError('registered focal pose head continuation/alternatives changed')
        width,height=a.frame_size_wh
        working_width=min(width,POLICY['image_maximum_width'])
        working_height=round(height*working_width/width)
        person=next((p for p in a.people if p.observation_id==owners[index]),None)
        if person is None:raise ValueError('registered focal pose required owner is unobserved')
        span=float(np.linalg.norm((np.asarray(person.raw_pose_xyc[6][:2])-
            person.raw_pose_xyc[5][:2])*np.asarray((working_width,working_height))))
        limit=max(POLICY['minimum_pose_image_residual_working_pixels'],
            POLICY['maximum_pose_image_residual_shoulder_fraction']*span)
        contrasts=check.get('roi_contrast_std');attempts=check.get('attempts')
        if (check.get('contract')!=POLICY['contract'] or check.get('raw_pair')!=pair
                or check.get('working_size')!=[working_width,working_height]
                or check.get('model_image_residual_limit_working_pixels')!=limit
                or not isinstance(contrasts,list) or len(contrasts)!=2
                or any(type(v) not in (int,float) or not math.isfinite(v)
                    or v<POLICY['minimum_roi_contrast_std'] for v in contrasts)
                or check.get('image_supported') is not True
                or any(check.get(k) is not False for k in ('anatomical_part_support_certified','arm_pixel_path_verified'))
                or not isinstance(attempts,list) or [p.get('raw_joint_index') for p in attempts]!=list(duties.required_joint_indices)):
            raise ValueError('registered focal model image consistency input/ROI policy changed')
        for attempt in attempts:
            fb=attempt.get('forward_backward_error_working_pixels');residual=attempt.get('model_image_residual_working_pixels')
            if attempt.get('lk_point_status')=='observed':
                if (type(fb) not in (int,float) or not 0<=fb<=POLICY['maximum_working_forward_backward_error']
                        or type(residual) not in (int,float) or not 0<=residual<=limit):
                    raise ValueError('registered model pose contradicts measured original image motion')
            elif attempt.get('lk_point_status')!='unknown' or residual is not None:
                raise ValueError('unknown image support cannot become a zero/known motion cost')
            if attempt.get('contradiction') is not False:
                raise ValueError('registered pose image contradiction was promoted to measured support')
        part_constraints=check.get('required_part_image_constraints')
        expected_bones=[(name,list(joints)) for name,joints in _required_bones(duties.required_joint_indices)]
        if (not isinstance(part_constraints,list) or not part_constraints
                or [(p.get('part_name'),p.get('raw_joint_indices')) for p in part_constraints]!=expected_bones
                or check.get('required_part_motion_constrained') is not True):
            raise ValueError('required part image motion is unconstrained or its original duty changed')
        after_person=next((p for p in b.people if p.observation_id==owners[index+1]),None)
        if after_person is None:raise ValueError('required part image motion lost its actual current owner')
        for part in part_constraints:
            summary=_bone_constraint_summary(part,person,after_person,
                np.asarray((working_width,working_height),float),limit)
            if (set(part)!=set(summary)|{'attempts'}
                    or any(part.get(key)!=expected for key,expected in summary.items())
                    or summary['status']!='observed_compatible'):
                raise ValueError('required part image motion differs from its actual regional constraints')
        if check.get('contradiction') is not False:
            raise ValueError('registered pose image contradiction changed')
    graph=native_pose_association_graph(domain=domain,rows=rows)
    frozen=lambda v:json.dumps(v,sort_keys=True,allow_nan=False)
    reconstructed=SourceNativeFocalPoseSequenceV1(rebuilt,tuple(heads),frozen(links),frozen(checks),
        frozen(graph.to_dict()),policy_sha256()).to_dict()
    if digest(reconstructed)!=digest(proof) or reconstructed['measurement_available'] is not True:
        raise ValueError('registered focal pose measurement does not replay from original raw samples')
    selected=value.get('selected_head_ids')
    expected_bodies=[[owners[first-natives[0]]],[owners[last-natives[0]]]]
    if (selected!=[heads[first-natives[0]],heads[last-natives[0]]]
            or value.get('required_body_observation_ids')!=expected_bodies):
        raise ValueError('registered focal pose view borrowed another selected head/body owner')
    ticks=value.get('output_ticks');fps=value.get('output_fps');samples=value.get('actual_output_samples')
    if (not isinstance(ticks,list) or len(ticks)!=2 or any(type(t) is not int for t in ticks)
            or ticks[0]>=ticks[1] or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0
            or not isinstance(samples,list) or [s.get('output_tick') for s in samples]!=list(range(ticks[0],ticks[1]+1))):
        raise ValueError('registered focal pose actual output sampling interval changed')
    scope=hypothesis['shot_intent']['required_output_interval']
    if fps!=state.output_fps or any(not scope[0]<=t<scope[1] for t in ticks):
        raise ValueError('registered focal pose output clock/required scope differs from its complete state')
    from .group_source_changing_owner import project_point
    from .group_scene_role_obligations import scene_role_obligations
    original_required=set()
    for sample in samples:
        n=sample.get('native_frame_index')
        if type(n) is not int or n not in natives:
            raise ValueError('registered focal pose output borrowed an unobserved native raster')
        index=n-natives[0];points=sample.get('projected_required_joint_xy')
        if (sample.get('output_seconds')!=sample['output_tick']/fps
                or sample.get('raw_observation_sha256')!=rows[index].sha256
                or sample.get('head_observation_id')!=heads[index] or sample.get('person_observation_id')!=owners[index]
                or not isinstance(points,list) or [p[0] for p in points]!=list(duties.required_joint_indices)
                or any(len(p)!=3 or any(type(v) not in (int,float) or not math.isfinite(v)
                    or not 0<=v<=1 for v in p[1:]) for p in points)):
            raise ValueError('registered focal pose output lost required measured joints/clock')
        ref=None if original_duty_program is None else duty_at(original_duty_program,state,sample['output_tick'])
        if sample.get('original_duty_reference')!=ref:
            raise ValueError('registered pose output changed its externally owned original tick duty')
        names=None if ref is None else tuple(ref['required_part_names'])
        expected_parts={name for _,name in scene_role_obligations(rows[index],state.role,heads[index],
            required_part_names=names).required_parts}
        original_required.update(expected_parts)
        if (state.native_frame_indices[sample['output_tick']-state.first_output_frame]!=n
                or not expected_parts<=set(parts) or ref is None and sorted(expected_parts)!=parts):
            raise ValueError('registered focal pose native join/arm lifecycle differs from its frozen state')
        person=next(p for p in rows[index].people if p.observation_id==owners[index])
        expected=[[joint,*project_point(person.raw_pose_xyc[joint][:2],state,sample['output_tick'])]
            for joint in duties.required_joint_indices]
        if points!=expected:
            raise ValueError('registered focal pose received output points differ from actual complete camera projection')
    if sorted(original_required)!=parts:
        raise ValueError('registered pose measurement mask expanded/removed original interval duties')
    if ([samples[0]['native_frame_index'],samples[-1]['native_frame_index']]!=[first,last]
            or any(a['native_frame_index']>b['native_frame_index'] for a,b in zip(samples,samples[1:]))):
        raise ValueError('registered focal pose chronological output/native join changed')


def restore_view_state(value,domain,hypothesis,state_key):
    """Camera structure replay against an externally owned state hash.

    This does not replace the parent's fresh model/pixel/render-plan replay.
    It makes output-space motion algebra depend on the exact bound camera.
    """
    from .group_joint_source_camera import GroupSourceCameraState,GroupOutputRole
    from .render_plan_serialization import render_segment_from_json
    from .group_scene_intent import validate_intent
    from .group_source_changing_owner import NativeFamilySensors
    if not isinstance(value,dict) or not isinstance(value.get('key_preimage'),dict):
        raise ValueError('registered focal pose lost its complete camera preimage')
    p=value['key_preimage']
    try:
        state=GroupSourceCameraState(p['slot'],p['first'],p['end'],p['source'],
            tuple(render_segment_from_json(s) for s in p['segments']),tuple(p['native']),
            GroupOutputRole(p['role']),p['clock'],p['fps'],tuple(p['output_size']),
            tuple(p['cell_frame_counts']),value.get('raw_shot_family'),p.get('editorial_contract_sha256'))
        state.bind_source(domain)
        if (state.key!=state_key or digest(state.to_dict())!=digest(value)
                or state.editorial_contract_sha256!=digest(hypothesis)
                or list(NativeFamilySensors.span_key(state))!=hypothesis['span_key']):
            raise ValueError('registered focal pose changed its externally owned complete camera state')
        validate_intent(hypothesis['shot_intent'],role=state.role.value,
            first=state.first_output_frame,end=state.end_output_frame,
            endpoint_scope=hypothesis.get('endpoint_scope','whole'))
        return state
    except (KeyError,TypeError,OverflowError) as exc:
        raise ValueError('registered focal pose complete camera/intent preimage missing') from exc
