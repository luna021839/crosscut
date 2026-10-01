"""Shared camera-only replacement guard for previously bound subject windows.

A head-only receipt cannot supersede a face/performer/formation/head+arm
camera merely by keeping its index. Without a newly measured full-subject
replacement contract, retain the previous dense camera window and its four
motion ticks exactly. This is a numeric geometry/source-clock guarantee, not
new head/arm measurements, cross-camera identity or encoded quality approval.
"""
import json
import math

from .boundary_subject_alignment import GLOBAL_ACTION_BASIS
from .post_render_qa import _bound_local_alignment_tracks
from .render import RenderPlan,segment_frame_counts


def _camera_state(segment,time):
    crop=segment.crop
    if segment.pan_crop is not None:crop=segment.pan_crop.evaluate(time)
    base=(crop.x,crop.y,crop.width,crop.height) if crop is not None else None
    track=segment.editorial_transform
    if track is None:return (base,None,None)
    key=track.evaluate(time)
    return (base,(key.zoom,key.center_x,key.center_y,key.roll_degrees),track.effective_sampling_canvas_zoom)


def _same_state(before,after):
    for a,b in zip(before,after,strict=True):
        if a is None or b is None:
            if a!=b:return False
        elif isinstance(a,tuple):
            if any(not math.isclose(x,y,abs_tol=1e-9,rel_tol=0.) for x,y in zip(a,b,strict=True)):return False
        elif not math.isclose(a,b,abs_tol=1e-9,rel_tol=0.):return False
    return True


def validate_camera_replacement_preservation(before,after,motion):
    """Check real prior policy, clock and dense windows before attaching.

    This guard intentionally does not claim a head-only proposer has sensed
    arms. Exact unchanged prior windows retain ANY known previous subject and
    independent motion there. Camera changes outside those observed windows
    still need their own complete-shot containment and final encoded QA.
    A broader reframe of a required window must supply a separate fresh dense
    subject/motion preservation contract, not switch this guard off.
    """
    if not isinstance(before,RenderPlan) or not isinstance(after,RenderPlan):
        raise TypeError('camera replacement preservation needs complete typed render plans')
    counts=segment_frame_counts(before)
    if (_source_output_domain(before)!=_source_output_domain(after)
            or len(before.segments)!=len(after.segments)
            or counts!=segment_frame_counts(after)
            or before.transitions!=after.transitions
            or any((a.source_index,a.source_start,a.source_end,a.master_start,a.duration)!=
                (b.source_index,b.source_start,b.source_end,b.master_start,b.duration)
                for a,b in zip(before.segments,after.segments,strict=True))):
        raise ValueError('prior camera obligation requires unchanged exact source/output clocks')
    changed={i for i,(a,b) in enumerate(zip(before.segments,after.segments,strict=True))
        if any(getattr(a,k)!=getattr(b,k) for k in ('crop','pan_crop','editorial_transform','outgoing_overlap_transform'))}
    alignment=motion.boundary_subject_alignment_plan
    if alignment is None or not changed:
        return dict(contract='prior_bound_camera_windows_exactly_retained_v1',
            preserved_obligations=[],new_motion_observations=False,quality_approved=False)
    raw_motion=motion.to_dict()
    bound_segments,boundaries=_bound_local_alignment_tracks(raw_motion,json.loads(json.dumps(before.to_json())))
    decisions={d.boundary_index:d for d in alignment.decisions if d.eligible}
    rows=[]
    for item in alignment.segments:
        index=item.segment_index
        if index not in changed or item.track is None:continue
        if (not 0<=index<len(counts) or index not in bound_segments
                or not item.boundary_indices or any(b not in boundaries for b in item.boundary_indices)):
            raise ValueError('prior camera subject obligation has no complete bound native authority')
        frames={n for n in range(counts[index]) if (key:=item.track.evaluate(n/before.fps))
            and (abs(key.zoom-1.)>1e-8 or abs(key.center_x-.5)>1e-8
                or abs(key.center_y-.5)>1e-8 or abs(key.roll_degrees)>1e-8)}
        policies=[]
        for boundary in item.boundary_indices:
            decision=decisions.get(boundary)
            if decision is None or decision.subject_kind is None:
                raise ValueError('prior camera subject obligation lost its measured subject kind')
            side='incoming' if index==boundary+1 else 'outgoing' if index==boundary else None
            if side is None:raise ValueError('prior camera obligation names an unrelated boundary')
            frames.update(range(min(4,counts[index])) if side=='incoming' else
                range(max(0,counts[index]-4),counts[index]))
            policies.append(dict(boundary_index=boundary,subject_kind=decision.subject_kind,
                evidence_basis=alignment.analysis_basis,
                protected_subject='head_and_observed_arms' if alignment.analysis_basis==GLOBAL_ACTION_BASIS
                    and decision.subject_kind=='face' else decision.subject_kind))
        if any(not _same_state(_camera_state(before.segments[index],n/before.fps),
                _camera_state(after.segments[index],n/after.fps)) for n in frames):
            raise ValueError('head-only replacement changed a prior camera subject or motion obligation; fresh full-subject proof required')
        rows.append(dict(segment_index=index,preserved_local_output_frames=sorted(frames),policies=policies))
    return dict(contract='prior_bound_camera_windows_exactly_retained_v1',
        preserved_obligations=rows,source_output_clock_unchanged=True,
        missing_observations_imputed=False,new_motion_observations=False,quality_approved=False)


def _source_output_domain(plan):
    return (plan.sources,plan.reference_source,plan.master_audio_start,plan.fps,plan.width,plan.height,
        plan.aspect_ratio,plan.member_pose_dissolve_requested,plan.member_pose_dissolve_effective,
        plan.member_pose_dissolve_count)
