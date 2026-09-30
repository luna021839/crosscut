"""One bounded GROUP cut-time/camera proposal, never a quality certificate.

The source/reference mapping is supplied by the caller's alignment authority.
Changing the cut does not stretch an old pan over a different source interval:
base pans keep their absolute output-time curve. A new boundary camera can
replace only its own settle/ramp, with a neutral seam to the retained opposite
boundary. Actual neighboring native frames and camera states must be unchanged.
"""
from dataclasses import dataclass, replace
import math

from .boundary_subject_alignment import (
    BoundarySubjectAlignmentConfig, BoundarySubjectAlignmentPlan,
    _fixed_crop_at, build_boundary_subject_alignment_plan,
)
from .face_match import FaceMatchPlan, FaceMatchSegment
from .render import (
    EditorialTransformKeyframe, EditorialTransformTrack, NormalizedCrop,
    PanCropKeyframe, PanCropTrack, RenderPlan, RenderSegment, segment_frame_counts,
)
from .render_boundary_clock import native_frame_clocks_for_output
from .visual_score import VisualTimelinePlan


def _neutral(keyframe):
    return all(math.isclose(actual, expected, rel_tol=0., abs_tol=1e-9)
        for actual, expected in zip((keyframe.zoom,keyframe.center_x,keyframe.center_y,keyframe.roll_degrees),
                                   (1.,.5,.5,0.)))


def retime_base_pan(track, *, relative_shift, duration):
    """Retain fixed crop size even when its widest original sample is trimmed."""
    if not math.isfinite(relative_shift) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('pan retiming requires finite shift and positive duration')
    if track is None:
        return None
    times = sorted({0., duration, *(key.time_seconds-relative_shift for key in track.keyframes
        if 0. < key.time_seconds-relative_shift < duration)})
    return PanCropTrack(tuple(PanCropKeyframe(time, NormalizedCrop(*_fixed_crop_at(
        track, max(0., time+relative_shift)))) for time in times))


def splice_boundary_camera(old_track, new_track, *, relative_shift, frame_count, fps, outgoing):
    """Keep the other cut's camera exactly; never bridge nonneutral states."""
    if (type(frame_count) is not int or frame_count < 2 or type(outgoing) is not bool
            or not math.isfinite(fps) or fps <= 0 or not math.isfinite(relative_shift)):
        raise ValueError('camera splice requires a valid output clock and side')
    last = (frame_count-1)/fps
    new_track = new_track or EditorialTransformTrack((EditorialTransformKeyframe(0.,1.),))
    if old_track is None:
        return new_track
    old_times = {0., last, *(k.time_seconds-relative_shift for k in old_track.keyframes
        if 0. <= k.time_seconds-relative_shift <= last)}
    new_times = {0., last, *(k.time_seconds for k in new_track.keyframes if k.time_seconds <= last)}
    def old_at(time):
        return replace(old_track.evaluate(max(0.,time+relative_shift)),time_seconds=time)
    # A neutral seam exists only where both piecewise-linear curves really
    # reach neutral, not where their scalar zooms merely cross one another.
    seams = sorted(time for time in old_times|new_times
        if _neutral(old_at(time)) and _neutral(new_track.evaluate(time)))
    if not seams:
        raise ValueError('boundary camera cannot preserve the neighboring nonneutral track')
    seam = max(seams) if outgoing else min(seams)
    if outgoing:
        # The new tail may start after this seam. Keep only old head samples.
        keys = [old_at(time) for time in sorted(old_times) if time < seam]
        keys += [new_track.evaluate(time) for time in sorted(new_times|{seam}) if time >= seam]
    else:
        keys = [new_track.evaluate(time) for time in sorted(new_times|{seam}) if time <= seam]
        keys += [old_at(time) for time in sorted(old_times) if time > seam]
    result = EditorialTransformTrack(tuple(keys), reason='camera_conditioned_cut_with_neutral_neighbor_seam_v1')
    # Do not permit a seam that quietly discards the other cut's active ramp.
    protected = range(min(14,frame_count)) if outgoing else range(max(0,frame_count-14),frame_count)
    for frame in protected:
        time = frame/fps
        if any(not math.isclose(a,b,rel_tol=0.,abs_tol=1e-9)
               for a,b in zip(vars_for_state(result.evaluate(time)),vars_for_state(old_at(time)))):
            raise ValueError('boundary camera changes the neighboring cut window')
    return result


@dataclass(frozen=True)
class GroupCutCameraProposal:
    plan: RenderPlan
    clock_timeline: VisualTimelinePlan
    alignment: BoundarySubjectAlignmentPlan
    boundary_index: int
    timing_offset_frames: int
    preserved_neighbor_boundaries: tuple[int, ...]

    def to_dict(self):
        return dict(contract='bounded_clock_preserving_group_cut_camera_proposal_v1',
            boundary_index=self.boundary_index,
            timing_offset_frames=self.timing_offset_frames,
            preserved_neighbor_boundaries=list(self.preserved_neighbor_boundaries),
            preserved_output_frames_per_neighbor_side=14,
            neighbor_preservation_basis='native_frame_ids_and_numeric_camera_states_not_encoded_pixel_identity',
            subject_alignment=self.alignment.to_dict(),
            source_reference_alignment_authority='caller_supplied_not_proven_by_camera_geometry',
            timeline_descriptors_refreshed=False, production_selected=False, quality_approved=False)


def build_group_cut_camera_proposal(*, plan, timeline, boundary_index, shifted_segments,
                                   source_clocks, scene_cut_times, source_index_map=None):
    """Bind one proposed aligned pair to dense local geometry and its neighbors.

    ``clock_timeline`` deliberately is NOT a refreshed visual selection or
    review: its descriptors belong to the former endpoint times. Consumers
    must not publish it as a completed repaired timeline.
    """
    if not isinstance(plan,RenderPlan) or not isinstance(timeline,VisualTimelinePlan):
        raise TypeError('camera proposal requires a full plan and visual timeline')
    if (type(boundary_index) is not int or not 0 <= boundary_index < len(plan.segments)-1
            or len(timeline.choices) != len(plan.segments) or len(shifted_segments) != 2):
        raise ValueError('camera proposal must replace one adjacent pair')
    if plan.aspect_ratio.value != '16:9' or any(t.kind.uses_overlap for t in plan.transitions):
        raise ValueError('camera proposal currently requires landscape hard cuts')
    source_index_map = tuple(range(len(plan.sources))) if source_index_map is None else tuple(source_index_map)
    if (len(set(source_index_map)) != len(source_index_map) or any(type(index) is not int
            or not 0 <= index < len(plan.sources) for index in source_index_map)
            or len(scene_cut_times) != len(source_index_map)
            or any(type(choice.source_index) is not int or not 0 <= choice.source_index < len(source_index_map)
                   for choice in timeline.choices)):
        raise ValueError('camera proposal has an invalid eligible-source mapping')
    if any(type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds < 0
           for values in scene_cut_times for seconds in values):
        raise ValueError('camera proposal requires finite nonnegative native cut times')
    for index,(choice,segment) in enumerate(zip(timeline.choices,plan.segments,strict=True)):
        if (choice.segment_index != index or source_index_map[choice.source_index] != segment.source_index
                or any(abs(a-b) > 1e-6 for a,b in (
                    (choice.canonical_master_start,segment.master_start+plan.master_audio_start),
                    (choice.duration_seconds,segment.duration),
                    (choice.source_local_start,segment.source_start),
                    (choice.source_local_end,segment.source_end)))):
            raise ValueError('camera proposal timeline and rendered source clocks disagree')
    old_pair = plan.segments[boundary_index:boundary_index+2]
    if old_pair[0].source_index == old_pair[1].source_index:
        raise ValueError('camera proposal needs a cross-source cut')
    if any(not isinstance(new,RenderSegment) or new.source_index != old.source_index
           for old,new in zip(old_pair,shifted_segments)):
        raise ValueError('camera proposal cannot change source selection')
    left,right = shifted_segments
    movement_frames=(right.master_start-old_pair[1].master_start)*plan.fps
    if abs(movement_frames) > 8+1e-7 or abs(movement_frames-round(movement_frames)) > 1e-7:
        raise ValueError('camera proposal is limited to eight whole output frames')
    if (abs(left.master_start-old_pair[0].master_start) > 1e-9
            or abs(right.master_start+right.duration-old_pair[1].master_start-old_pair[1].duration) > 1e-9
            or abs(left.master_start+left.duration-right.master_start) > 1e-9):
        raise ValueError('camera proposal must conserve its outer output interval')
    choices,segments = list(timeline.choices),list(plan.segments)
    for index,new in zip((boundary_index,boundary_index+1),shifted_segments):
        old = segments[index]
        if source_index_map[choices[index].source_index] != new.source_index or old.crop is not None:
            raise ValueError('camera proposal source domain or base crop is unsupported')
        if old.outgoing_overlap_transform is not None or old.boundary_look_track is not None:
            raise ValueError('camera proposal must precede overlap and look finalization')
        relative = new.master_start-old.master_start
        segments[index] = replace(new,
            pan_crop=retime_base_pan(old.pan_crop,relative_shift=relative,duration=new.duration),
            crop=None,editorial_transform=None,look_adjustment=old.look_adjustment,
            outgoing_overlap_transform=None,boundary_look_track=None)
        choices[index] = replace(choices[index],
            canonical_master_start=new.master_start+plan.master_audio_start,
            canonical_master_end=new.master_start+new.duration+plan.master_audio_start,
            source_local_start=new.source_start,source_local_end=new.source_end,
            internal_scene_cut_master_seconds=tuple(new.master_start+plan.master_audio_start
                +(seconds-new.source_start)/new.source_playback_rate
                for seconds in scene_cut_times[choices[index].source_index]
                if new.source_start < seconds < new.source_end),
            entry_scene_cut_near=any(abs(seconds-new.source_start) <= 3/plan.fps
                for seconds in scene_cut_times[choices[index].source_index]),
            exit_scene_cut_near=any(abs(seconds-new.source_end) <= 3/plan.fps
                for seconds in scene_cut_times[choices[index].source_index]))
    clock_timeline = replace(timeline,choices=tuple(choices))
    local_sources = tuple(plan.sources[index] for index in source_index_map)
    face_plan = FaceMatchPlan(tuple(FaceMatchSegment(i,choice.source_index,
        segments[i].pan_crop is not None,None,None,None,None,None,segments[i].pan_crop)
        for i,choice in enumerate(choices)),None,2.,(.5,.5))
    numeric_plan = replace(plan,segments=tuple(segments))
    counts = segment_frame_counts(numeric_plan)
    # Reserve the opposite boundary's complete 14-frame window before
    # authoring either ramp. The neutral join frame can be shared. This is a
    # deterministic window choice, not repeated weaker-containment retries.
    window_frames = min(14,*(counts[index]-14+1
        for index in (boundary_index,boundary_index+1)))
    if window_frames < 8:
        raise ValueError('camera proposal has no disjoint validated ramp window')
    alignment = build_boundary_subject_alignment_plan(visual_timeline=clock_timeline,
        sources=local_sources,transitions=plan.transitions,face_match_plan=face_plan,
        fps=plan.fps,output_width=plan.width,output_height=plan.height,
        render_frame_clocks=tuple(source_clocks.get(index) for index in source_index_map),
        boundary_indices=(boundary_index,),
        preserved_editorial_tracks=tuple(segment.editorial_transform for segment in plan.segments),
        preserved_track_time_shifts=tuple(new.master_start-old.master_start
            for old,new in zip(plan.segments,segments,strict=True)),
        config=BoundarySubjectAlignmentConfig(window_frames=window_frames))
    decision = alignment.decision_for(boundary_index)
    if decision is None or not decision.eligible:
        raise ValueError('camera proposal lacks dense local subject alignment')
    for index in (boundary_index,boundary_index+1):
        old,new = plan.segments[index],segments[index]
        track=alignment.track_for(index)
        if track is None:
            retained=splice_boundary_camera(old.editorial_transform,None,
                relative_shift=new.master_start-old.master_start,frame_count=counts[index],
                fps=plan.fps,outgoing=index==boundary_index)
            if any(not _neutral(retained.evaluate(frame/plan.fps)) for frame in range(counts[index])):
                raise ValueError('untransformed side retains a camera outside the new dense receipt')
        segments[index] = replace(new,editorial_transform=track)
    proposal = replace(plan,segments=tuple(segments))
    preserved = assert_preserved_output_boundaries(plan,proposal,source_clocks,
        (boundary_index-1,boundary_index+1))
    return GroupCutCameraProposal(proposal,clock_timeline,alignment,boundary_index,round(movement_frames),preserved)


def assert_preserved_output_boundaries(plan, proposal, source_clocks, neighbors):
    """Prove the unchanged outer windows of a local proposal, never its new cuts."""
    if any(getattr(plan,name)!=getattr(proposal,name) for name in (
            'sources','fps','width','height','transitions','master_audio_start','reference_source')):
        raise ValueError('camera proposal changed the output or source domain')
    before_counts,after_counts = segment_frame_counts(plan),segment_frame_counts(proposal)
    if plan.output_frame_count != proposal.output_frame_count:
        raise ValueError('camera proposal changed total output frames')
    preserved = []
    for neighbor in neighbors:
        if not 0 <= neighbor < len(plan.segments)-1:
            continue
        if plan.segments[neighbor].source_index == plan.segments[neighbor+1].source_index:
            # Coalesced source runs need their own interior-clock validation.
            raise ValueError('camera proposal touches an unvalidated same-source neighboring seam')
        cut=sum(before_counts[:neighbor+1])
        if (cut != sum(after_counts[:neighbor+1]) or min(
                *before_counts[neighbor:neighbor+2],*after_counts[neighbor:neighbor+2]) < 14):
            raise ValueError('camera proposal lacks complete neighboring fourteen-frame windows')
        wanted=tuple(range(cut-14,cut+14))
        before=native_frame_clocks_for_output(plan,source_clocks,wanted)
        after=native_frame_clocks_for_output(proposal,source_clocks,wanted)
        for frame_index in wanted:
            a,b=before[frame_index],after[frame_index]
            if (a.native_frame_index,a.source_index,a.output_frame_index) != (
                    b.native_frame_index,b.source_index,b.output_frame_index) or a.terminal_clone or b.terminal_clone:
                raise ValueError('camera proposal changed neighboring native frames')
            index = neighbor if frame_index < cut else neighbor+1
            ta = (a.output_frame_index-sum(before_counts[:index]))/plan.fps
            tb = (b.output_frame_index-sum(after_counts[:index]))/plan.fps
            old,new = plan.segments[index],proposal.segments[index]
            if any(getattr(old,name)!=getattr(new,name) for name in (
                    'crop','look_adjustment','boundary_look_track','outgoing_overlap_transform')):
                raise ValueError('camera proposal changed neighboring crop, grade or overlap')
            ca = _fixed_crop_at(old.pan_crop,ta) if old.pan_crop else (0.,0.,1.,1.)
            cb = _fixed_crop_at(new.pan_crop,tb) if new.pan_crop else (0.,0.,1.,1.)
            ka = old.editorial_transform.evaluate(ta) if old.editorial_transform else EditorialTransformKeyframe(ta,1.)
            kb = new.editorial_transform.evaluate(tb) if new.editorial_transform else EditorialTransformKeyframe(tb,1.)
            va,vb = (ca,tuple(vars_for_state(ka))),(cb,tuple(vars_for_state(kb)))
            if any(not math.isclose(x,y,rel_tol=0.,abs_tol=1e-7)
                   for ar,br in zip(va,vb) for x,y in zip(ar,br)):
                raise ValueError('camera proposal changed neighboring crop or camera')
        preserved.append(neighbor)
    return tuple(preserved)


def vars_for_state(keyframe):
    return keyframe.zoom,keyframe.center_x,keyframe.center_y,keyframe.roll_degrees
