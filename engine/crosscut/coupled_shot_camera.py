"""A complete-shot camera and the next cut's tracked entry handle, atomically.

This refines geometry only. Native clocks, audio and cut times do not change;
source-local pixel tracking is not cross-camera identity or choreography QA.
"""
from dataclasses import dataclass, replace
import math
from time import monotonic

from .boundary_subject_alignment import (
    _CaptureProvider, _canonical_frame, _TransformState, _SubjectGeometry,
    _joint_face_alignment_states, _track_for_segment, _follow_face_camera_path,
    _close_face_head_envelope, _project_box, _clamp_center,
    BoundarySubjectAlignmentConfig,
)
from .cut_anchored_containment import track_cut_handle
from .media import probe_media
from .render import segment_frame_counts
from .render_boundary_clock import native_frame_clocks_for_output
from .render_frame_schedule import probe_render_frame_clock
from .rendered_proxy_review import _box_discontinuity
from .roll_aware_face_geometry import detect_roll_aware_faces
from .shot_context_camera import observe_shot_head, _observe_composed_head
from .shot_context_binding import build_shot_context_binding
from .visual_score import _normalized_face_metadata


@dataclass(frozen=True)
class CoupledShotProposal:
    outgoing: object
    incoming_head_evidence: tuple
    incoming_native_frame_ids: tuple
    incoming_source_clock: object
    attempts: tuple
    elapsed_seconds: float

    @property
    def plan(self): return self.outgoing.plan

    @property
    def segment_index(self): return self.outgoing.segment_index

    def to_dict(self):
        return dict(contract='whole_shot_and_tracked_entry_camera_proposal_v1',
            changed_segments=[self.segment_index,self.segment_index+1],
            attempts=list(self.attempts),elapsed_seconds=self.elapsed_seconds,
            incoming_head_evidence=list(self.incoming_head_evidence),
            audio_or_source_phase_changed=False,cross_camera_identity_certified=False,
            gesture_continuity_certified=False,professional_quality_approved=False)


def _state(key):
    return _TransformState(key.zoom,key.center_x,key.center_y,key.roll_degrees)


def _visible(track,heads,fps):
    return min(_project_box(head,_state(track.evaluate(frame/fps)))[1]
        for frame,head in enumerate(heads))


def propose_coupled_shot_camera(*, previous, scene_cut_times,
        maximum_runtime_seconds=20.):
    """At most five joint geometries and two output-pixel corrections each.

    The accepted complete-shot evidence is reused. Only its cut endpoints and
    the next native shot's <=14-frame entry handle are decoded again. All
    source clocks are independently probed and compared across the final plan.
    """
    from .shot_context_refinement import both_boundary_framings_improve
    build_shot_context_binding(previous)
    plan=previous.plan;index=previous.segment_index;counts=segment_frame_counts(plan)
    if (type(maximum_runtime_seconds) not in (float,int)
            or not math.isfinite(maximum_runtime_seconds) or not 0<maximum_runtime_seconds<=60
            or not 0<index<len(plan.segments)-2
            or len(scene_cut_times)!=len(plan.sources)):
        raise ValueError('invalid bounded coupled-shot request')
    incoming=plan.segments[index+1]
    if (incoming.crop is not None or incoming.pan_crop is not None
            or incoming.editorial_transform is not None
            or incoming.outgoing_overlap_transform is not None
            or any(t.kind.value!='hard_cut' for t in plan.transitions[index-1:index+2])):
        raise ValueError('coupled entry requires an unchanged native hard-cut camera')
    count=min(14,counts[index+1])
    if count<2 or (count-1)/plan.fps>.65:
        raise ValueError('coupled entry is outside the short containment handle')
    started=monotonic()
    def deadline():
        if monotonic()-started>maximum_runtime_seconds:
            raise ValueError('coupled camera exhausted its work budget')
    clocks={}
    for source in sorted({s.source_index for s in plan.segments}):
        deadline()
        clocks[source]=probe_render_frame_clock(plan.sources[source],
            timeout_seconds=max(.01,min(10.,maximum_runtime_seconds-(monotonic()-started))))
        if clocks[source] is None:raise ValueError('coupled source frame clock is unknown')
    first=sum(counts[:index]);cut=first+counts[index]
    indices=tuple(sorted({first-1,first,cut-1,*range(cut,cut+count)}))
    samples=native_frame_clocks_for_output(plan,clocks,indices)
    if any(tick.terminal_clone for tick in samples.values()):
        raise ValueError('coupled camera cannot use terminal clones')
    if clocks[plan.segments[index].source_index]!=previous.source_clock:
        raise ValueError('complete-shot source clock changed')
    native_fps=float(clocks[incoming.source_index].frame_rate)
    cuts=tuple(scene_cut_times[incoming.source_index])
    if (any(type(t) not in (int,float) or not math.isfinite(t) or t<0 for t in cuts)
            or tuple(sorted(set(cuts)))!=cuts):
        raise ValueError('coupled native scene cuts must be ordered unique observations')
    if any(samples[cut].native_frame_index/native_fps<t<=samples[cut+count-1].native_frame_index/native_fps for t in cuts):
        raise ValueError('coupled entry crosses a native source cut')
    pixels={};frames=[];faces=[];observations={}
    capture=_CaptureProvider(plan.sources)
    try:
        for n in indices:
            deadline();tick=samples[n]
            raw=capture.frame_at_native_index(tick.source_index,tick.native_frame_index,clocks[tick.source_index])
            if raw is None:raise ValueError('coupled exact native frame unavailable')
            if n in (first-1,first,cut-1,cut):pixels[n]=raw
            if n in (first,cut-1,cut):
                observations[n]=observe_shot_head(_canonical_frame(raw,width=854,height=480,base_crop=None))
            if n>=cut:
                frame=_canonical_frame(raw,width=854,height=480,base_crop=None);frames.append(frame)
                measured=[_normalized_face_metadata(face,frame_width=854,frame_height=480)[0]
                    for face in detect_roll_aware_faces(frame,angles=(0.,))]
                faces.append(tuple(box for box in measured if box is not None))
    finally:capture.close()
    if any(not o.face_is_unambiguous_dominant for o in observations.values()):
        raise ValueError('coupled cut endpoint has no observed focal face')
    deadline()
    tracked=track_cut_handle(frames=frames,times=tuple(i/plan.fps for i in range(count)),
        face_boxes=faces,anchor_box=observations[cut].face_box_xywh,anchor_at_end=False,output_fps=plan.fps)
    if not tracked.complete:
        raise ValueError(f'coupled cut-anchor tracking {tracked.failure} at {tracked.failed_index}')
    heads=tuple(tuple(row['head_envelope_xywh']) for row in previous.head_evidence)
    next_heads=tuple(_close_face_head_envelope(sample.box) for sample in tracked.samples)
    evidence=tuple(dict(frame=sample.index,face_box_xywh=list(sample.box),
        head_envelope_xywh=list(head),basis=sample.basis,
        consecutive_detector_misses=sample.consecutive_detector_misses)
        for sample,head in zip(tracked.samples,next_heads))
    maximums=[]
    for segment in (plan.segments[index],incoming):
        deadline();streams=probe_media(plan.sources[segment.source_index]).video_streams
        if (not streams or type(streams[0].width) is not int or type(streams[0].height) is not int
                or min(streams[0].width,streams[0].height)<=0):
            raise ValueError('coupled native pixel budget unavailable')
        maximums.append(min(2.,streams[0].width/plan.width,streams[0].height/plan.height))
    def subject(observed):
        box=observed.face_box_xywh
        return _SubjectGeometry('face',box,box[3],containment_boxes=(_close_face_head_envelope(box),))
    states=_joint_face_alignment_states(subject(observations[cut-1]),subject(observations[cut]),
        maximum_zooms=tuple(maximums),settings=BoundarySubjectAlignmentConfig())
    settings_size=(plan.width,plan.height);last=(counts[index]-1)/plan.fps
    left_time=(counts[index-1]-1)/plan.fps
    fixed_neighbor=_observe_composed_head(pixels[first-1],plan.segments[index-1],left_time,settings_size)
    def pixel_rows(version):
        observed=[_observe_composed_head(pixels[n],version.segments[i],time,settings_size)
            for n,i,time in ((first,index,0.),(cut-1,index,last),(cut,index+1,0.))]
        if any(not o.face_is_unambiguous_dominant for o in (*observed,fixed_neighbor)):
            raise ValueError('coupled output-coordinate face is unobserved')
        return (_box_discontinuity(observed[0].face_box_xywh,fixed_neighbor.face_box_xywh),
            _box_discontinuity(observed[1].face_box_xywh,observed[2].face_box_xywh))
    def analytic_rows(version):
        boxes=[]
        for n,i,time in ((first,index,0.),(cut-1,index,last),(cut,index+1,0.)):
            track=version.segments[i].editorial_transform
            boxes.append(_project_box(observations[n].face_box_xywh,_state(track.evaluate(time)) if track else _TransformState())[0])
        return (_box_discontinuity(boxes[0],fixed_neighbor.face_box_xywh),_box_discontinuity(boxes[1],boxes[2]))
    baseline=pixel_rows(plan);analytic_baseline=analytic_rows(plan)
    entry=_state(plan.segments[index].editorial_transform.evaluate(0.))
    attempts=[];accepted=[]
    for left,right,_,_ in states:
        deadline();row=dict(outgoing=left.to_dict(),incoming=right.to_dict(),pixel_corrections=[])
        try:
            straight=_track_for_segment(last_frame_time=last,ramp_frames=count,fps=plan.fps,
                entry=(index-1,entry),exit=(index,left),settle_policy='shot_continuous')
            outgoing_track=_follow_face_camera_path(straight,heads,fps=plan.fps,maximum_center_shift=.1)
            if _visible(outgoing_track,heads,plan.fps)<.995:raise ValueError('coupled outgoing head clipped')
            def compose(right):
                track=_track_for_segment(last_frame_time=(counts[index+1]-1)/plan.fps,
                    ramp_frames=count,fps=plan.fps,entry=(index,right),exit=None)
                if track is None or _visible(track,next_heads,plan.fps)<.995:
                    raise ValueError('coupled incoming head clipped')
                segments=list(plan.segments)
                segments[index]=replace(segments[index],editorial_transform=outgoing_track)
                segments[index+1]=replace(segments[index+1],editorial_transform=track)
                return replace(plan,segments=tuple(segments))
            candidate=compose(right);measured=pixel_rows(candidate)
            for iteration in range(2):
                deadline()
                if measured[1][0]<=baseline[1][0]+1e-6 and measured[1][1]<=baseline[1][1]+2./max(settings_size):break
                out=_observe_composed_head(pixels[cut-1],candidate.segments[index],last,settings_size).face_box_xywh
                inc=_observe_composed_head(pixels[cut],candidate.segments[index+1],0.,settings_size).face_box_xywh
                if out is None or inc is None:raise ValueError('coupled calibration face unavailable')
                ratio=out[3]/inc[3];zoom=right.zoom*ratio
                if not 1.<=zoom<=maximums[1]:raise ValueError('coupled calibration exceeds native pixels')
                a=tuple(out[i]+out[i+2]/2 for i in (0,1));b=tuple(inc[i]+inc[i+2]/2 for i in (0,1))
                cx,cy=_clamp_center(right.center_x+(.5+ratio*(b[0]-.5)-a[0])/zoom,
                    right.center_y+(.5+ratio*(b[1]-.5)-a[1])/zoom,zoom=zoom,maximum_shift=.1)
                right=_TransformState(zoom,cx,cy,0.);candidate=compose(right);measured=pixel_rows(candidate)
                row['pixel_corrections'].append(dict(incoming=right.to_dict(),measured=measured))
            analytic=analytic_rows(candidate)
            rows=lambda old,new,basis:tuple(dict(boundary_index=index-1+i,before=list(a),after=list(b),basis=basis)
                for i,(a,b) in enumerate(zip(old,new)))
            result=replace(previous,plan=candidate,camera=replace(previous.camera,track=outgoing_track,
                entry=entry,exit=left,minimum_head_visible_fraction=_visible(outgoing_track,heads,plan.fps)),
                boundary_framing=rows(analytic_baseline,analytic,'same_native_face_projected_through_both_camera_plans_not_encoded_qa'),
                pixel_boundary_framing=rows(baseline,measured,'redetected_full_output_coordinate_pixels_not_encoded_qa'))
            row.update(pixel_framing=list(result.pixel_boundary_framing),analytic_framing=list(result.boundary_framing))
            if not both_boundary_framings_improve(result):raise ValueError('coupled framing does not improve both cuts')
            accepted.append((sum(sum(values) for values in measured),result));row['status']='geometry_admitted_pending_encoded_qa'
        except ValueError as exc:row.update(status='unavailable',reason=str(exc))
        attempts.append(row)
    if not accepted:raise ValueError('no coupled camera improved both measured cut framings')
    _,result=min(accepted,key=lambda item:item[0]);deadline()
    before=native_frame_clocks_for_output(plan,clocks,range(plan.output_frame_count))
    after=native_frame_clocks_for_output(result.plan,clocks,range(result.plan.output_frame_count))
    if any((tick.source_index,tick.native_frame_index)!=(after[n].source_index,after[n].native_frame_index)
            for n,tick in before.items()):raise ValueError('coupled cameras changed the native frame schedule')
    deadline()
    return CoupledShotProposal(result,evidence,tuple(samples[n].native_frame_index for n in range(cut,cut+count)),
        clocks[incoming.source_index],tuple(attempts),monotonic()-started)
