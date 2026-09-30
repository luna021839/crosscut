"""Experimental whole-shot composition against BOTH fixed neighbor pictures.

This geometric proposal does not change source time, recognize identity,
certify pose continuity or grant a production QA binding. Its caller must
obtain every head envelope from actual source observations/tracking, prove
the shot has no native camera cut, and validate the resulting encoded pixels.
"""
from dataclasses import dataclass, replace
import math
from time import monotonic
import cv2

from .boundary_subject_alignment import (
    BoundarySubjectAlignmentConfig, BoundarySubjectObservation,
    SHOT_CAMERA_EXPERIMENT_BASIS, _TransformState, _target_state,
    _track_for_segment, _follow_face_camera_path, _project_box,
    _CaptureProvider, _canonical_frame, _fixed_crop_at,
    _default_observation_provider, _close_face_head_envelope, _optical_flow_box,
)
from .render import EditorialTransformTrack, RenderPlan, segment_frame_counts
from .render_boundary_clock import native_frame_clocks_for_output
from .render_frame_schedule import probe_render_frame_clock,VerifiedRenderFrameClock
from .render_geometry import reproject_editorial_frame
from .media import probe_media
from .roll_aware_face_geometry import detect_roll_aware_faces
from .visual_score import _dominant_face, _normalized_face_metadata


@dataclass(frozen=True)
class ShotContextCamera:
    track: EditorialTransformTrack
    entry: _TransformState
    exit: _TransformState
    minimum_head_visible_fraction: float
    frame_count: int
    face_scale_basis: str
    endpoint_scale_policy: str = 'exact'

    def to_dict(self):
        return dict(analysis_basis=SHOT_CAMERA_EXPERIMENT_BASIS,
            policy='one_native_shot_against_both_fixed_composed_neighbors_v1',
            face_scale_basis=self.face_scale_basis,endpoint_scale_policy=self.endpoint_scale_policy,
            entry=self.entry.to_dict(),exit=self.exit.to_dict(),
            frame_count=self.frame_count,minimum_head_visible_fraction=self.minimum_head_visible_fraction,
            head_input_provenance_validated_by_this_geometry_function=False,
            source_clock_validated_by_this_geometry_function=False,
            hand_retention_or_gesture_continuity_certified=False,
            production_binding_supported=False,quality_approved=False)


def fixed_neighbor_face_state(source, target, *, maximum_zoom, scale_basis='face_box_height',
        endpoint_scale_policy='exact'):
    """Solve a feasible native-pixel crop, not an identity/phase assertion.

    Eye span changes with yaw and detector regression. It is not interchangeable
    with output face height. Keep the two objectives explicit for ablation;
    never call better position a demonstrated improvement in both dimensions.
    """
    if scale_basis not in {'face_box_height', 'eye_span'}:
        raise ValueError('unknown fixed-neighbor face scale basis')
    if endpoint_scale_policy not in {'exact','bounded_residual'}:
        raise ValueError('unknown fixed-neighbor endpoint scale policy')
    if type(maximum_zoom) not in (float,int) or not math.isfinite(maximum_zoom) or not 1.<=maximum_zoom<=2.:
        raise ValueError('fixed-neighbor camera needs a verified native zoom budget')
    for observed in (source,target):
        if (not isinstance(observed,BoundarySubjectObservation)
                or not observed.face_is_unambiguous_dominant
                or observed.face_box_xywh is None
                or observed.face_landmarks_xy is None or len(observed.face_landmarks_xy)!=5
                or observed.face_eye_scale is None or observed.face_head_roll_radians is None):
            raise ValueError('fixed-neighbor endpoint needs an observed unambiguous five-point face')
    zoom=(target.face_box_xywh[3]/source.face_box_xywh[3] if scale_basis=='face_box_height'
        else target.face_eye_scale/source.face_eye_scale)
    if endpoint_scale_policy=='exact' and not 1.<=zoom<=maximum_zoom:
        raise ValueError('fixed-neighbor scale would exceed native pixel budget or invent outside pixels')
    # Exact equality is not a requirement for a useful camera: an already
    # almost-matched endpoint may need 0.985x while the other benefits from
    # 1.65x. Clamp only the proposal to real pixels. BOTH residuals still face
    # analytic and independently redetected output-pixel admission afterward.
    # This does not enlarge a native zoom budget or invent outside pixels.
    zoom=min(maximum_zoom,max(1.,zoom))
    roll=max(-1.6,min(1.6,math.degrees(target.face_head_roll_radians-source.face_head_roll_radians)))
    state,_=_target_state(box=source.face_box_xywh,target_box=target.face_box_xywh,
        zoom=zoom,roll_degrees=roll,maximum_center_shift=.10,settings=BoundarySubjectAlignmentConfig())
    if state is None:
        raise ValueError('fixed-neighbor endpoint has no renderer-safe crop')
    return state


def build_shot_context_camera(*, source_endpoints, neighbor_endpoints, head_envelopes,
        fps, maximum_zoom, scale_basis='face_box_height',endpoint_scale_policy='exact'):
    """Join both cut states through the source's measured, nonlinear head path.

    Head-envelope input provenance is the caller's separate responsibility.
    A tracked head region is NOT relabelled a face detection here. This layer
    only computes a curve and checks every provided envelope against it.
    """
    if len(source_endpoints)!=2 or len(neighbor_endpoints)!=2:
        raise ValueError('shot camera needs both source and fixed-neighbor endpoints')
    if type(fps) not in (float,int) or not math.isfinite(fps) or fps<=0:
        raise ValueError('shot camera needs a finite positive output clock')
    heads=tuple(head_envelopes)
    if not 2<=len(heads)<=300:
        raise ValueError('shot camera permits 2..300 provided head envelopes')
    entry,exit_state=(fixed_neighbor_face_state(source,target,maximum_zoom=maximum_zoom,scale_basis=scale_basis,
        endpoint_scale_policy=endpoint_scale_policy)
        for source,target in zip(source_endpoints,neighbor_endpoints))
    straight=_track_for_segment(last_frame_time=(len(heads)-1)/fps,ramp_frames=14,fps=fps,
        entry=(0,entry),exit=(1,exit_state),settle_policy='shot_continuous')
    if straight is None:
        raise ValueError('fixed-neighbor endpoints are already neutral')
    track=_follow_face_camera_path(straight,heads,fps=fps,maximum_center_shift=.10)
    visible=[]
    for index,head in enumerate(heads):
        key=track.evaluate(index/fps)
        state=_TransformState(key.zoom,key.center_x,key.center_y,key.roll_degrees)
        amount=_project_box(head,state)[1]
        if amount<.995:
            raise ValueError(f'fixed-neighbor camera clips provided head envelope at frame {index}')
        visible.append(amount)
    return ShotContextCamera(track,entry,exit_state,min(visible),len(heads),scale_basis,endpoint_scale_policy)


def observe_shot_head(frame):
    """Experimental native-resolution pass, then bounded roll reacquisition."""
    original=_default_observation_provider(frame)
    faces=detect_roll_aware_faces(frame,angles=(0.,))
    primary=_dominant_face(faces)
    if primary is None:
        faces=detect_roll_aware_faces(frame)
        primary=_dominant_face(faces)
    box,points,roll,eye,_=_normalized_face_metadata(primary,
        frame_width=frame.shape[1],frame_height=frame.shape[0])
    return replace(original,face_count=len(faces),face_box_xywh=box,
        face_landmarks_xy=points,face_head_roll_radians=roll,face_eye_scale=eye,
        face_is_unambiguous_dominant=box is not None)


def _observe_composed_head(frame, segment, time, size):
    """Redetect in actual output-coordinate pixels, not a transformed box.

    Face inference is not scale equivariant. Projecting one native bounding
    box can promise an improvement that disappears after the crop is decoded.
    This is a pre-encode pixel check; it still does not replace encoded QA.
    """
    from .prescribed_render_geometry import PrescribedGeometrySampler
    from .post_render_perceptual_quality import _reference_view_for_span
    from .visual_score import _working_frame
    geometry=PrescribedGeometrySampler(None,segment,master_start=0.)
    pixels=_reference_view_for_span(frame,geometry.span,relative_time=time,size=size)
    working=_working_frame(pixels,maximum_width=960)
    return observe_shot_head(cv2.resize(working,(854,480)))


@dataclass(frozen=True)
class ShotContextProposal:
    plan: RenderPlan
    segment_index: int
    camera: ShotContextCamera
    head_evidence: tuple[dict, ...]
    native_frame_ids: tuple[int, ...]
    elapsed_seconds: float
    source_clock: VerifiedRenderFrameClock | None = None
    boundary_framing: tuple[dict, ...] = ()
    pixel_boundary_framing: tuple[dict, ...] = ()

    def to_dict(self):
        return dict(analysis_basis=SHOT_CAMERA_EXPERIMENT_BASIS,segment_index=self.segment_index,
            camera=self.camera.to_dict(),head_evidence=list(self.head_evidence),
            native_frame_ids=list(self.native_frame_ids),elapsed_seconds=self.elapsed_seconds,
            source_clock=self.source_clock.to_dict() if self.source_clock else None,
            boundary_framing=list(self.boundary_framing),
            pixel_boundary_framing=list(self.pixel_boundary_framing),
            all_sampled_native_clocks_unchanged=True,audio_and_source_phase_unchanged=True,
            other_segment_cameras_unchanged=True,native_cuts_in_changed_shot=0,
            source_local_performer_identity_certified=False,production_binding_supported=False,
            quality_approved=False)


def propose_shot_context_camera(*, plan, segment_index, scene_cut_times,
        experimental=False, maximum_runtime_seconds=30., scale_basis='face_box_height',
        endpoint_scale_policy='bounded_residual'):
    """Bounded reusable native-clock proposal; does not render or auto-publish.

    Only one cross-source-bounded native shot is touched. Callers still need
    independent decoded framing/motion QA and an explicit experimental consumer;
    boundary-local production bindings must not accept this geometry report.
    App callers use a separate complete-shot binding plus output-pixel checks.
    """
    if experimental is not True:
        raise ValueError('shot-context proposal requires an explicit experimental consumer')
    if (not isinstance(plan,RenderPlan) or type(segment_index) is not int
            or not 0<segment_index<len(plan.segments)-1
            or len(scene_cut_times)!=len(plan.sources)
            or type(maximum_runtime_seconds) not in (float,int)
            or not math.isfinite(maximum_runtime_seconds) or not 0<maximum_runtime_seconds<=60):
        raise ValueError('invalid bounded shot-context request')
    index=segment_index
    middle=plan.segments[index]
    adjacent=plan.segments[index-1:index+2]
    if (plan.aspect_ratio.value!='16:9' or any(t.kind.uses_overlap for t in plan.transitions)
            or any(s.crop is not None for s in adjacent)
            or any(middle.source_index==s.source_index for s in (adjacent[0],adjacent[2]))):
        raise ValueError('shot-context proposal requires an isolated landscape hard-cut source run')
    cuts=tuple(scene_cut_times[middle.source_index])
    if (any(type(t) not in (float,int) or not math.isfinite(t) or t<0 for t in cuts)
            or tuple(sorted(set(cuts)))!=cuts):
        raise ValueError('shot-context native scene-cut evidence must be ordered and unique')
    counts=segment_frame_counts(plan)
    if not 2<=counts[index]<=300:
        raise ValueError('shot-context source run exceeds the 300-frame work budget')
    started=monotonic()
    def deadline():
        if monotonic()-started>maximum_runtime_seconds:
            raise ValueError('shot-context proposal exhausted its work budget')
    clocks={}
    for i in sorted({segment.source_index for segment in adjacent}):
        deadline()
        clocks[i]=probe_render_frame_clock(plan.sources[i],
            timeout_seconds=max(.01,min(10.,maximum_runtime_seconds-(monotonic()-started))))
    if any(clocks[s.source_index] is None for s in adjacent):
        raise ValueError('shot-context source frame clock is unverified')
    begin,end=sum(counts[:index]),sum(counts[:index+1])
    samples=native_frame_clocks_for_output(plan,clocks,range(begin-1,end+1))
    if any(tick.terminal_clone for tick in samples.values()):
        raise ValueError('shot-context camera requires actually observed source frames, not terminal clones')
    first,last=samples[begin],samples[end-1]
    native_fps=float(clocks[middle.source_index].frame_rate)
    if any(first.native_frame_index/native_fps < cut <= last.native_frame_index/native_fps for cut in cuts):
        raise ValueError('shot-context camera cannot animate across a native camera cut')
    video=probe_media(plan.sources[middle.source_index]).video_streams
    if not video or not video[0].width or not video[0].height:
        raise ValueError('shot-context native resolution is unknown')
    maximum_zoom=min(2.,video[0].width/plan.width,video[0].height/plan.height)
    capture=_CaptureProvider(plan.sources)
    frames=[];observations=[];neighbors=[];neighbor_pixels=[];source_end_pixels=[]
    try:
        for i,n in ((index-1,begin-1),(index+1,end)):
            deadline();tick=samples[n];old=plan.segments[i]
            time=(counts[i]-1)/plan.fps if i<index else 0.
            raw=capture.frame_at_native_index(tick.source_index,tick.native_frame_index,clocks[tick.source_index])
            if raw is None:raise ValueError('shot-context neighbor frame is unavailable')
            neighbor_pixels.append(raw)
            frame=_canonical_frame(raw,width=854,height=480,
                base_crop=_fixed_crop_at(old.pan_crop,time) if old.pan_crop else None)
            if old.editorial_transform:
                key=old.editorial_transform.evaluate(time)
                frame=reproject_editorial_frame(frame,size=(854,480),zoom=key.zoom,center_x=key.center_x,
                    center_y=key.center_y,roll_degrees=key.roll_degrees,
                    maximum_zoom=max(k.zoom for k in old.editorial_transform.keyframes))
            neighbors.append(observe_shot_head(frame))
        for n in range(begin,end):
            deadline();tick=samples[n]
            raw=capture.frame_at_native_index(tick.source_index,tick.native_frame_index,clocks[tick.source_index])
            if raw is None:raise ValueError('shot-context source frame is unavailable')
            if n in (begin,end-1):source_end_pixels.append(raw)
            frame=_canonical_frame(raw,width=854,height=480,base_crop=None)
            frames.append(frame);observations.append(observe_shot_head(frame))
    finally:capture.close()
    heads=[];evidence=[]
    for n,observed in enumerate(observations):
        deadline()
        if observed.face_is_unambiguous_dominant:
            heads.append(_close_face_head_envelope(observed.face_box_xywh))
            evidence.append(dict(frame=n,basis='observed_face_head_envelope',head_envelope_xywh=heads[-1]))
            continue
        support=[]
        for ref in range(max(0,n-2),min(len(frames),n+3)):
            other=observations[ref]
            if ref==n or not other.face_is_unambiguous_dominant:continue
            tracked=_optical_flow_box(frames[ref],frames[n],_close_face_head_envelope(other.face_box_xywh))
            if tracked is not None:support.append((ref,tracked))
        if len(support)<2:
            raise ValueError(f'shot-context head containment is unobserved at frame {n}')
        x=min(b[0] for _,b in support);y=min(b[1] for _,b in support)
        right=max(b[0]+b[2] for _,b in support);bottom=max(b[1]+b[3] for _,b in support)
        heads.append((x,y,right-x,bottom-y))
        evidence.append(dict(frame=n,basis='two_neighbor_pixel_tracked_head_union',
            reference_frames=[i for i,_ in support],face_landmarks_observed=False,head_envelope_xywh=heads[-1]))
    camera=build_shot_context_camera(source_endpoints=(observations[0],observations[-1]),
        neighbor_endpoints=neighbors,head_envelopes=heads,fps=plan.fps,maximum_zoom=maximum_zoom,
        scale_basis=scale_basis,endpoint_scale_policy=endpoint_scale_policy)
    from .rendered_proxy_review import _box_discontinuity
    framing=[]
    for n,source,target,state in ((0,observations[0],neighbors[0],camera.entry),
            (counts[index]-1,observations[-1],neighbors[-1],camera.exit)):
        before=source.face_box_xywh
        time=n/plan.fps
        if middle.pan_crop:
            x,y,w,h=_fixed_crop_at(middle.pan_crop,time)
            if abs(w-h)>1e-6:
                raise ValueError('shot-context comparison requires an isotropic existing base crop')
            before=((before[0]-x)/w,(before[1]-y)/h,before[2]/w,before[3]/h)
        if middle.editorial_transform:
            old=middle.editorial_transform.evaluate(time)
            before=_project_box(before,_TransformState(old.zoom,old.center_x,old.center_y,old.roll_degrees))[0]
        after=_project_box(source.face_box_xywh,state)[0]
        framing.append(dict(boundary_index=index-1 if n==0 else index,
            before=list(_box_discontinuity(before,target.face_box_xywh)),
            after=list(_box_discontinuity(after,target.face_box_xywh)),
            basis='same_native_face_projected_through_both_camera_plans_not_encoded_qa'))
    changed=list(plan.segments)
    changed[index]=replace(middle,pan_crop=None,editorial_transform=camera.track)
    candidate=replace(plan,segments=tuple(changed))
    pixel_framing=[]
    for side,(neighbor_index,native,neighbor) in enumerate(zip((index-1,index+1),source_end_pixels,neighbor_pixels)):
        deadline()
        middle_time=0. if side==0 else (counts[index]-1)/plan.fps
        neighbor_time=(counts[neighbor_index]-1)/plan.fps if side==0 else 0.
        target=_observe_composed_head(neighbor,plan.segments[neighbor_index],neighbor_time,(plan.width,plan.height))
        observed=[_observe_composed_head(native,version.segments[index],middle_time,(plan.width,plan.height))
            for version in (plan,candidate)]
        if any(not item.face_is_unambiguous_dominant or item.face_box_xywh is None for item in (target,*observed)):
            raise ValueError('shot-context composed endpoint face is unobserved')
        pixel_framing.append(dict(boundary_index=index-1 if side==0 else index,
            before=list(_box_discontinuity(observed[0].face_box_xywh,target.face_box_xywh)),
            after=list(_box_discontinuity(observed[1].face_box_xywh,target.face_box_xywh)),
            basis='redetected_full_output_coordinate_pixels_not_encoded_qa'))
    after=native_frame_clocks_for_output(candidate,clocks,range(begin-1,end+1))
    if any((a.source_index,a.native_frame_index)!=(after[n].source_index,after[n].native_frame_index)
            for n,a in samples.items()):
        raise ValueError('shot-context composition changed the scheduled native frame clock')
    deadline()
    return ShotContextProposal(candidate,index,camera,tuple(evidence),
        tuple(samples[n].native_frame_index for n in range(begin,end)),monotonic()-started,
        clocks[middle.source_index],tuple(framing),tuple(pixel_framing))
