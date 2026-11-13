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
    _local_focal_observation_provider,
)
from .render import (EditorialTransformKeyframe, EditorialTransformTrack, NormalizedCrop,
    PanCropKeyframe, PanCropTrack, RenderPlan, segment_frame_counts, _can_share_source_clock)
from .render_boundary_clock import native_frame_clocks_for_output
from .render_frame_schedule import probe_render_frame_clock,VerifiedRenderFrameClock
from .render_geometry import reproject_editorial_frame
from .media import probe_media
from .roll_aware_face_geometry import detect_roll_aware_faces
from .visual_score import _dominant_face, _normalized_face_metadata
from .group_focal_attention import focal_landmarks_track


def _usable_shot_face(observed, *, local_focal_attention=False):
    return (isinstance(observed,BoundarySubjectObservation)
        and (observed.face_is_unambiguous_dominant
            or local_focal_attention and observed.selected_focal_subject is not None)
        and observed.face_box_xywh is not None
        and observed.face_landmarks_xy is not None and len(observed.face_landmarks_xy)==5
        and observed.face_eye_scale is not None and observed.face_head_roll_radians is not None)


def _focal_pixel_path_step(reference,target,before,after):
    """Measured local continuity, never centrality or member recognition."""
    from .cut_anchored_containment import _iou
    if focal_landmarks_track(reference,target,before.face_landmarks_xy,after.face_landmarks_xy):
        return 'measured_landmarks_forward_backward_pixel_flow'
    tracked=_optical_flow_box(reference,target,before.face_box_xywh)
    if tracked is not None and _iou(tracked,after.face_box_xywh)>=.60:
        return 'head_box_forward_backward_pixel_flow'
    # A lighting flash/nonrigid face motion may defeat LK even though both
    # neighboring faces are freshly observed. Reuse the existing bounded
    # local geometry + appearance check; this cannot synthesize missing faces
    # or search for a similar person elsewhere in the image.
    from .observed_face_containment import recover_observed_face_step
    if recover_observed_face_step(reference,target,before.face_box_xywh,(after.face_box_xywh,),
            maximum_detection_width=960) is not None:
        return 'observed_local_face_geometry_and_appearance'
    return None


def _source_local_focal_path(frames,observations,deadline):
    """Do not let a visible background face replace an occluded foreground.

    Every accepted face owns a measured body and connects to the last accepted
    face through real pixels. At most two misses can be bridged; missing heads
    still need the separate two-neighbor containment evidence below.
    """
    accepted=[];provenance=[];previous=None
    for n,observed in enumerate(observations):
        deadline();basis=None
        if observed.selected_focal_subject is not None and _usable_shot_face(observed,local_focal_attention=True):
            if n==0:basis='observed_source_local_focal_anchor'
            elif previous is not None and n-previous<=3:
                basis=_focal_pixel_path_step(frames[previous],frames[n],accepted[previous],observed)
        if basis is None:
            accepted.append(BoundarySubjectObservation());provenance.append(None)
        else:
            provenance.append(dict(tracked_from_frame=previous,temporal_track_basis=basis))
            accepted.append(observed);previous=n
        if previous is None or n-previous>2:
            raise ValueError(f'shot-context source-local focal path is unobserved at frame {n}')
    return accepted,provenance


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
        endpoint_scale_policy='exact',local_focal_attention=False):
    """Solve a feasible native-pixel crop, not an identity/phase assertion.

    Eye span changes with yaw and detector regression. It is not interchangeable
    with output face height. Keep the two objectives explicit for ablation;
    never call better position a demonstrated improvement in both dimensions.
    """
    if scale_basis not in {'face_box_height', 'eye_span'}:
        raise ValueError('unknown fixed-neighbor face scale basis')
    if type(local_focal_attention) is not bool:
        raise ValueError('fixed-neighbor local focal attention must be explicit boolean')
    if endpoint_scale_policy not in {'exact','bounded_residual'}:
        raise ValueError('unknown fixed-neighbor endpoint scale policy')
    if type(maximum_zoom) not in (float,int) or not math.isfinite(maximum_zoom) or not 1.<=maximum_zoom<=2.:
        raise ValueError('fixed-neighbor camera needs a verified native zoom budget')
    for observed in (source,target):
        if not _usable_shot_face(observed,local_focal_attention=local_focal_attention):
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
        fps, maximum_zoom, scale_basis='face_box_height',endpoint_scale_policy='exact',local_focal_attention=False):
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
        endpoint_scale_policy=endpoint_scale_policy,local_focal_attention=local_focal_attention)
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


def retain_opposite_cut_camera(segment, native_track, *, head_envelopes, fps,
        outgoing=False, preserved_boundary_frames=4, ramp_frames=14):
    """Keep one cut's existing nonneutral camera, not just its final keypoint.

    A complete native-coordinate candidate can use a different base pan from
    the incumbent. Reparameterize its unrolled viewport into the incumbent's
    fixed-size pan; move that pan only enough to contain the desired viewport.
    Smoothly return to the incumbent BEFORE the opposite comparison window,
    then retain every original pan/editorial state in that window exactly.
    This is a bounded geometry proposal: every native head envelope is checked,
    but neither pixel identity after encoding nor editorial quality is claimed.
    """
    heads=tuple(head_envelopes);count=len(heads)
    if (not isinstance(native_track,EditorialTransformTrack)
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0
            or type(outgoing) is not bool or type(preserved_boundary_frames) is not int
            or not 4<=preserved_boundary_frames<=14 or type(ramp_frames) is not int
            or not 4<=ramp_frames<=30 or not 2*preserved_boundary_frames+ramp_frames<=count<=80
            or abs(segment.duration-count/fps)>1/fps+1e-8
            or segment.crop is not None or segment.outgoing_overlap_transform is not None
            or segment.editorial_transform is None
            or any(abs(k.roll_degrees)>1e-9 for k in native_track.keyframes)
            or segment.editorial_transform is not None
                and any(abs(k.roll_degrees)>1e-9 for k in segment.editorial_transform.keyframes)):
        raise ValueError('opposite-cut retention needs a bounded complete unrolled native shot')
    if segment.pan_crop is None:width=height=1.
    else:width,height=segment.pan_crop.fixed_crop_size
    if abs(width-height)>1e-9:
        raise ValueError('native viewport retention cannot change the base crop aspect')
    original=segment.editorial_transform
    keys=[];pan_keys=[]
    def retained(n):
        return n<preserved_boundary_frames if outgoing else n>=count-preserved_boundary_frames
    for n,head in enumerate(heads):
        if (head is None or len(head)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in head)
                or head[0]<0 or head[1]<0 or head[2]<=0 or head[3]<=0
                or head[0]+head[2]>1.000001 or head[1]+head[3]>1.000001):
            raise ValueError('opposite-cut retention needs every measured native head envelope')
        time=n/fps
        x,y,_,_=_fixed_crop_at(segment.pan_crop,time) if segment.pan_crop else (0.,0.,1.,1.)
        old=original.evaluate(time);new=native_track.evaluate(time)
        if retained(n):
            key=replace(old,time_seconds=time)
        else:
            # Quintic easing has zero endpoint velocity/acceleration. There is
            # no one-frame reset from a new camera to the protected old one.
            distance=n-preserved_boundary_frames+1 if outgoing else count-preserved_boundary_frames-n
            progress=max(0.,min(1.,distance/ramp_frames))
            amount=progress**3*(10.+progress*(-15.+6.*progress))
            old_native=(old.zoom/width,x+width*old.center_x,y+height*old.center_y)
            zoom,cx,cy=(a*(1.-amount)+b*amount for a,b in
                zip(old_native,(new.zoom,new.center_x,new.center_y)))
            local_zoom=zoom*width
            if local_zoom<1.-1e-9:
                raise ValueError('native candidate needs pixels outside the fixed-size base pan')
            local_zoom=max(1.,local_zoom)
            radius=.5/zoom
            # Preserve the old pan when it already contains the viewport.
            # Otherwise choose the closest containing pan, never outside pixels.
            left_low,left_high=max(0.,cx+radius-width),min(1.-width,cx-radius)
            top_low,top_high=max(0.,cy+radius-height),min(1.-height,cy-radius)
            if left_low>left_high+1e-9 or top_low>top_high+1e-9:
                raise ValueError('native candidate viewport cannot fit inside real source pixels')
            x=min(left_high,max(left_low,x));y=min(top_high,max(top_low,y))
            key=EditorialTransformKeyframe(time,local_zoom,(cx-x)/width,(cy-y)/height,0.)
        transformed=((head[0]-x)/width,(head[1]-y)/height,head[2]/width,head[3]/height)
        state=_TransformState(key.zoom,key.center_x,key.center_y,key.roll_degrees)
        if _project_box(transformed,state)[1]<.995:
            raise ValueError(f'opposite-cut retention clips measured head at frame {n}')
        keys.append(key)
        if segment.pan_crop is not None:
            pan_keys.append(PanCropKeyframe(time,NormalizedCrop(x,y,width,height)))
    track=EditorialTransformTrack(tuple(keys),reason='complete_native_camera_with_preserved_opposite_cut_window_v1',
        sampling_canvas_zoom=max(original.effective_sampling_canvas_zoom,max(k.zoom for k in keys)))
    pan=PanCropTrack(tuple(pan_keys)) if pan_keys else None
    result=replace(segment,pan_crop=pan,editorial_transform=track)
    for n in range(count):
        if not retained(n):continue
        time=n/fps
        actual=result.editorial_transform.evaluate(time);expected=original.evaluate(time)
        if (any(abs(getattr(actual,name)-getattr(expected,name))>1e-9 for name in
                    ('zoom','center_x','center_y','roll_degrees'))
                or segment.pan_crop is not None and any(abs(a-b)>1e-9 for a,b in
                    zip(_fixed_crop_at(result.pan_crop,time),_fixed_crop_at(segment.pan_crop,time)))):
            raise ValueError('opposite-cut retention changed a protected output camera state')
    return result


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


def _observe_composed_head(frame, segment, time, size, *, local_focal_attention=False):
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
    frame=cv2.resize(working,(854,480));observed=observe_shot_head(frame)
    return (_local_focal_observation_provider(frame) if local_focal_attention
        and not _usable_shot_face(observed) else observed)


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
    last_segment_index: int | None = None
    local_focal_attention: bool = False

    @property
    def end_segment_index(self):
        return self.segment_index if self.last_segment_index is None else self.last_segment_index

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
            quality_approved=False,
            local_focal_attention=self.local_focal_attention,
            **({'last_segment_index':self.last_segment_index} if self.last_segment_index is not None else {}))


def slice_shot_camera_track(track, *, frame_offset, frame_count, fps):
    """Preserve each actual whole-run camera state on a cell's output clock.

    A logical beat-cell boundary must not reset the camera to neutral or
    restart its easing. No source clock or cut boundary is merged or moved.
    """
    if (type(frame_offset) is not int or frame_offset<0 or type(frame_count) is not int
            or frame_count<1 or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0):
        raise ValueError('invalid source-run camera slice')
    keys=tuple(replace(track.evaluate((frame_offset+n)/fps),time_seconds=n/fps) for n in range(frame_count))
    canvas=track.sampling_canvas_zoom
    if canvas is None and max(k.zoom for k in keys)<track.effective_sampling_canvas_zoom:
        canvas=track.effective_sampling_canvas_zoom
    return EditorialTransformTrack(keys,reason=track.reason,sampling_canvas_zoom=canvas)


def propose_shot_context_camera(*, plan, segment_index, scene_cut_times,
        experimental=False, maximum_runtime_seconds=30., scale_basis='face_box_height',
        endpoint_scale_policy='bounded_residual', last_segment_index=None,local_focal_attention=False):
    """Bounded reusable native-clock proposal; does not render or auto-publish.

    Only one cross-source-bounded native shot is touched. Callers still need
    independent decoded framing/motion QA and an explicit experimental consumer;
    boundary-local production bindings must not accept this geometry report.
    App callers use a separate complete-shot binding plus output-pixel checks.
    """
    if experimental is not True:
        raise ValueError('shot-context proposal requires an explicit experimental consumer')
    if type(local_focal_attention) is not bool:
        raise ValueError('shot-context local focal attention must be explicit boolean')
    if (not isinstance(plan,RenderPlan) or type(segment_index) is not int
            or not 0<segment_index<len(plan.segments)-1
            or len(scene_cut_times)!=len(plan.sources)
            or type(maximum_runtime_seconds) not in (float,int)
            or not math.isfinite(maximum_runtime_seconds) or not 0<maximum_runtime_seconds<=60):
        raise ValueError('invalid bounded shot-context request')
    index=segment_index
    last_index=index if last_segment_index is None else last_segment_index
    if type(last_index) is not int or not index<=last_index<len(plan.segments)-1:
        raise ValueError('shot-context source-run end is invalid')
    middle=plan.segments[index]
    adjacent=plan.segments[index-1:last_index+2]
    if (plan.aspect_ratio.value!='16:9' or any(t.kind.uses_overlap for t in plan.transitions)
            or any(s.crop is not None for s in adjacent)
            or any(s.outgoing_overlap_transform is not None for s in plan.segments[index:last_index+1])
            or any(middle.source_index==s.source_index for s in (adjacent[0],adjacent[-1]))
            or not _can_share_source_clock(plan,first_index=index,candidate_last_index=last_index)):
        raise ValueError('shot-context proposal requires an isolated landscape hard-cut source run')
    cuts=tuple(scene_cut_times[middle.source_index])
    if (any(type(t) not in (float,int) or not math.isfinite(t) or t<0 for t in cuts)
            or tuple(sorted(set(cuts)))!=cuts):
        raise ValueError('shot-context native scene-cut evidence must be ordered and unique')
    counts=segment_frame_counts(plan)
    count=sum(counts[index:last_index+1])
    if not 2<=count<=300:
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
    begin,end=sum(counts[:index]),sum(counts[:last_index+1])
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
    observer=_local_focal_observation_provider if local_focal_attention else observe_shot_head
    try:
        for i,n in ((index-1,begin-1),(last_index+1,end)):
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
                    maximum_zoom=old.editorial_transform.effective_sampling_canvas_zoom)
            neighbor=observe_shot_head(frame)
            if local_focal_attention and not _usable_shot_face(neighbor):neighbor=observer(frame)
            neighbors.append(neighbor)
            if local_focal_attention and not _usable_shot_face(neighbors[-1],local_focal_attention=True):
                raise ValueError('shot-context fixed neighbor has no unambiguous focal face')
        # Endpoint feasibility precedes dense sensing. Unknown targets and
        # two neutral endpoints cannot gain a camera by examining 300 frames.
        # Keep the very same endpoint observations for dense construction.
        endpoint_frames={};endpoint_observations={}
        for n,target in zip((begin,end-1),neighbors):
            deadline();tick=samples[n]
            raw=capture.frame_at_native_index(tick.source_index,tick.native_frame_index,clocks[tick.source_index])
            if raw is None:raise ValueError('shot-context source endpoint is unavailable')
            frame=_canonical_frame(raw,width=854,height=480,base_crop=None)
            observed=observer(frame)
            fixed_neighbor_face_state(observed,target,maximum_zoom=maximum_zoom,scale_basis=scale_basis,
                endpoint_scale_policy=endpoint_scale_policy,local_focal_attention=local_focal_attention)
            if local_focal_attention and observed.selected_focal_subject is None:
                raise ValueError('shot-context source-local focal endpoint has no measured body owner')
            endpoint_frames[n]=frame;endpoint_observations[n]=observed;source_end_pixels.append(raw)
        early_states=tuple(fixed_neighbor_face_state(endpoint_observations[n],target,
            maximum_zoom=maximum_zoom,scale_basis=scale_basis,endpoint_scale_policy=endpoint_scale_policy,
            local_focal_attention=local_focal_attention) for n,target in zip((begin,end-1),neighbors))
        if _track_for_segment(last_frame_time=(count-1)/plan.fps,ramp_frames=14,fps=plan.fps,
                entry=(0,early_states[0]),exit=(1,early_states[1]),settle_policy='shot_continuous') is None:
            raise ValueError('fixed-neighbor endpoints are already neutral')
        for n in range(begin,end):
            deadline()
            if n in endpoint_frames:
                frame,observed=endpoint_frames[n],endpoint_observations[n]
            else:
                tick=samples[n]
                raw=capture.frame_at_native_index(tick.source_index,tick.native_frame_index,clocks[tick.source_index])
                if raw is None:raise ValueError('shot-context source frame is unavailable')
                frame=_canonical_frame(raw,width=854,height=480,base_crop=None);observed=observer(frame)
            frames.append(frame);observations.append(observed)
    finally:capture.close()
    focal_provenance=None
    if local_focal_attention:
        observations,focal_provenance=_source_local_focal_path(frames,observations,deadline)
    heads=[];evidence=[]
    for n,observed in enumerate(observations):
        deadline()
        if _usable_shot_face(observed,local_focal_attention=local_focal_attention):
            heads.append(_close_face_head_envelope(observed.face_box_xywh))
            row=dict(frame=n,basis='observed_face_head_envelope',head_envelope_xywh=heads[-1])
            if local_focal_attention:
                focal=observed.selected_focal_subject
                row.update(basis='observed_focal_face_head_envelope',face_selection_basis=focal.basis,
                    face_box_xywh=focal.bbox_xywh,owned_body_box_xywh=focal.person.bbox_xywh,
                    body_detection_confidence=focal.person.detection_confidence,**focal_provenance[n])
            evidence.append(row)
            continue
        support=[]
        for ref in range(max(0,n-2),min(len(frames),n+3)):
            other=observations[ref]
            if ref==n or not _usable_shot_face(other,local_focal_attention=local_focal_attention):continue
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
        scale_basis=scale_basis,endpoint_scale_policy=endpoint_scale_policy,local_focal_attention=local_focal_attention)
    from .rendered_proxy_review import _box_discontinuity
    framing=[]
    for side,source,target,state in ((0,observations[0],neighbors[0],camera.entry),
            (1,observations[-1],neighbors[-1],camera.exit)):
        endpoint_index=index if side==0 else last_index
        middle=plan.segments[endpoint_index]
        before=source.face_box_xywh
        time=0. if side==0 else (counts[last_index]-1)/plan.fps
        if middle.pan_crop:
            x,y,w,h=_fixed_crop_at(middle.pan_crop,time)
            if abs(w-h)>1e-6:
                raise ValueError('shot-context comparison requires an isotropic existing base crop')
            before=((before[0]-x)/w,(before[1]-y)/h,before[2]/w,before[3]/h)
        if middle.editorial_transform:
            old=middle.editorial_transform.evaluate(time)
            before=_project_box(before,_TransformState(old.zoom,old.center_x,old.center_y,old.roll_degrees))[0]
        after=_project_box(source.face_box_xywh,state)[0]
        framing.append(dict(boundary_index=index-1 if side==0 else last_index,
            before=list(_box_discontinuity(before,target.face_box_xywh)),
            after=list(_box_discontinuity(after,target.face_box_xywh)),
            basis='same_native_face_projected_through_both_camera_plans_not_encoded_qa'))
    changed=list(plan.segments)
    frame_offset=0
    for i in range(index,last_index+1):
        track=(camera.track if index==last_index else slice_shot_camera_track(camera.track,
            frame_offset=frame_offset,frame_count=counts[i],fps=plan.fps))
        changed[i]=replace(plan.segments[i],pan_crop=None,editorial_transform=track)
        frame_offset+=counts[i]
    candidate=replace(plan,segments=tuple(changed))
    pixel_framing=[]
    observe_options={'local_focal_attention':True} if local_focal_attention else {}
    for side,(neighbor_index,native,neighbor) in enumerate(zip((index-1,last_index+1),source_end_pixels,neighbor_pixels)):
        deadline()
        endpoint_index=index if side==0 else last_index
        middle_time=0. if side==0 else (counts[last_index]-1)/plan.fps
        neighbor_time=(counts[neighbor_index]-1)/plan.fps if side==0 else 0.
        target=_observe_composed_head(neighbor,plan.segments[neighbor_index],neighbor_time,(plan.width,plan.height),**observe_options)
        observed=[_observe_composed_head(native,version.segments[endpoint_index],middle_time,(plan.width,plan.height),**observe_options)
            for version in (plan,candidate)]
        if any(not _usable_shot_face(item,local_focal_attention=local_focal_attention) for item in (target,*observed)):
            raise ValueError('shot-context composed endpoint face is unobserved')
        if local_focal_attention:
            from .cut_anchored_containment import _iou
            source=observations[0 if side==0 else -1].face_box_xywh
            expected=[]
            for version in (plan,candidate):
                segment=version.segments[endpoint_index];box=source
                if segment.pan_crop:
                    x,y,w,h=_fixed_crop_at(segment.pan_crop,middle_time)
                    box=((box[0]-x)/w,(box[1]-y)/h,box[2]/w,box[3]/h)
                if segment.editorial_transform:
                    key=segment.editorial_transform.evaluate(middle_time)
                    box=_project_box(box,_TransformState(key.zoom,key.center_x,key.center_y,key.roll_degrees))[0]
                expected.append(box)
            if (_iou(target.face_box_xywh,neighbors[side].face_box_xywh)<.60
                    or any(_iou(item.face_box_xywh,box)<.60 for item,box in zip(observed,expected))):
                raise ValueError('shot-context composed focal face changed its geometric owner')
        pixel_framing.append(dict(boundary_index=index-1 if side==0 else last_index,
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
        clocks[middle.source_index],tuple(framing),tuple(pixel_framing),
        last_index if last_index!=index else None,local_focal_attention)
