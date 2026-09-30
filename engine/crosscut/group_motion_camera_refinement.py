"""Bounded common-clock motion/camera refinement of an owned GROUP plan.

Actual projected output motion and faces propose and rank time offsets and
complete dense camera candidates. Source motion remains diagnostic evidence:
source-only bones cannot become required observations on a different crop.
Raw timeline failures stay raw failures.
This grants only a better local candidate for final decoded QA, never a
professional quality, temporal-phase or cross-camera identity certificate.
"""
from dataclasses import dataclass,replace
import math
from pathlib import Path
from time import monotonic

from .optional_native_worker import isolated_optional
from .render import aligned_render_segment,segment_frame_counts,EditorialTransformTrack
from .render_boundary_clock import hard_cut_boundary_sample_clocks
from .render_frame_schedule import probe_render_frame_clock
from .closeup_gesture import (CloseupGestureProvider,CloseupGestureEvidence,
    compare_arm_articulation,measure_arm_motion,MOTION_CONTRACT)
from .observed_head_motion import HeadMotionEvidence,HEAD_MOTION_BASIS,HEAD_DIMENSIONS
from .group_cut_camera_proposal import build_group_cut_camera_proposal,retime_base_pan
from .prescribed_render_geometry import PrescribedGeometrySampler
from .post_render_perceptual_quality import _reference_view_for_span
from .rendered_proxy_review import _default_geometry_provider,_box_discontinuity,ProxyReviewConfig
from .timeline import ShotType
from .timeline_review import review_timeline
from .visual_boundary_repair import _visual_evidence,_updated_entry,_updated_exit,_new_hard_failures
from .visual_score import _VideoSampler


def measure_projected_motion(frames,samples,provider,*,output_fps,required_basis=None):
    """Four actual output-coordinate pixel views, BEFORE a full encode.

    This reuses the loaded pose model; no output film/feature cache is needed.
    Missing bodies/arms stay unknown. A camera cannot claim source-arm gains
    while losing previously observable output bones. Head is a separate basis.
    """
    import cv2
    from .native_global_face import observe_native_global_face
    from .visual_score import describe_frame,_working_frame
    from .focal_pose import face_associated_person
    from .observed_head_motion import measure_head_motion,head_motion_signature
    from .group_focal_attention import focal_landmarks_track
    from .boundary_subject_alignment import _optical_flow_box
    from .cut_anchored_containment import _iou
    if len(frames)!=4 or len(samples)!=4:
        raise ValueError('projected motion requires four exact output views')
    if required_basis not in (None,HEAD_MOTION_BASIS,MOTION_CONTRACT):
        raise ValueError('projected motion must retain an explicit known evidence basis')
    if (type(output_fps) not in (int,float) or not math.isfinite(output_fps) or output_fps<=0
            or any(frame is None for frame in frames)):
        return None
    ticks=tuple(sample.output_frame_index for sample in samples)
    if (ticks[1]-ticks[0],ticks[2]-ticks[1],ticks[3]-ticks[2])!=(3,1,3):
        raise ValueError('projected motion views must own the exact four output ticks')
    if (samples[0].source_index!=samples[1].source_index
            or samples[2].source_index!=samples[3].source_index
            or samples[0].source_index==samples[2].source_index
            or any(sample.terminal_clone for sample in samples)):
        return None
    ids=tuple(sample.native_frame_index for sample in samples)
    intervals=(3/output_fps,1/output_fps,3/output_fps)
    working=[];heads=[];arms=[]
    for pixels in frames:
        frame=cv2.resize(_working_frame(pixels,maximum_width=960),(854,480))
        head=observe_native_global_face(frame)
        descriptor=describe_frame(frame,include_subject_geometry=True,include_formation_pose=False)
        owner=face_associated_person(head.bbox_xywh,descriptor.formation_people)
        arm=(provider.pose.describe(frame,bbox_xywh=owner.bbox_xywh,face_bbox_xywh=head.bbox_xywh)
            if head.bbox_xywh is not None and owner is not None else None)
        working.append(frame);heads.append(head);arms.append(arm)
    endpoint=compare_arm_articulation(arms[1],arms[2],outgoing_native_frame=ids[1],
        incoming_native_frame=ids[2],allow_partial=True)
    if required_basis!=HEAD_MOTION_BASIS:
        motion=measure_arm_motion(tuple(arms),native_frames=ids,output_intervals_seconds=intervals)
        if motion is not None and endpoint is not None:return replace(endpoint,motion=motion)
        if required_basis==MOTION_CONTRACT:return None
    for a,b in ((0,1),(2,3)):
        ha,hb=heads[a],heads[b]
        if ha.landmarks_xy and hb.landmarks_xy and focal_landmarks_track(
                working[a],working[b],ha.landmarks_xy,hb.landmarks_xy):continue
        box=_optical_flow_box(working[a],working[b],ha.bbox_xywh) if ha.bbox_xywh else None
        if box is None or hb.bbox_xywh is None or _iou(box,hb.bbox_xywh)<.60:return None
    motion=measure_head_motion(tuple(head_motion_signature(h.geometry,h.landmarks_xy) for h in heads),
        native_frames=ids,output_intervals_seconds=intervals,same_source_face_paths_pixel_verified=True)
    return HeadMotionEvidence(motion,endpoint) if motion is not None else None


@dataclass(frozen=True)
class GroupMotionCameraRefinement:
    plan: object
    timeline: object
    proposals: tuple
    attempts: tuple
    elapsed_seconds: float

    def to_dict(self):
        return dict(contract='bounded_common_clock_group_motion_camera_refinement_v1',
            changed_boundary_count=len(self.proposals),attempts=list(self.attempts),
            elapsed_seconds=self.elapsed_seconds,raw_geometry_gates_relabelled=False,
            quality_approved=False,temporal_phase_proven=False,
            cross_camera_identity_proven=False,ranking_calibrated=False)


def camera_motion_rank(gesture, face_metrics, *, enforce_face_limits=True):
    """Missing motion/face evidence has no numeric rank; legacy QA limits stay."""
    limits=ProxyReviewConfig()
    if gesture is None or gesture.motion is None or face_metrics is None:
        return None
    scale,position=face_metrics
    if (any(type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in (scale,position))
            or enforce_face_limits and (scale > limits.maximum_face_scale_discontinuity
                or position > limits.maximum_face_position_discontinuity)):
        return None
    return gesture.ranking_cost+scale+position


def measured_motion_improves(candidate, before):
    """Compare the incumbent's *four-frame* domain without losing any bone.

    An arm visible in only the two cut frames has no incumbent motion value.
    It stays unknown here rather than acting as a newly required four-frame
    measurement. Static-only callers keep their separate, stricter comparator.
    New measured bones still contribute to the candidate's worst-bone cost.
    """
    if isinstance(before,HeadMotionEvidence):
        return isinstance(candidate,HeadMotionEvidence) and candidate.improves_over(before)
    return bool(isinstance(candidate,CloseupGestureEvidence) and isinstance(before,CloseupGestureEvidence)
        and candidate.motion is not None and before.motion is not None
        and before.motion.observed_bones <= candidate.motion.observed_bones
        and candidate.motion.output_intervals_seconds == before.motion.output_intervals_seconds
        and candidate.preserves_pose_continuity_of(before)
        and candidate.ranking_cost < before.ranking_cost-1e-6)


def motion_evidence_basis(evidence):
    if isinstance(evidence,HeadMotionEvidence):return HEAD_MOTION_BASIS
    return MOTION_CONTRACT if isinstance(evidence,CloseupGestureEvidence) else None


def projected_motion_preserved(candidate,before):
    """A framing gain cannot buy lost bones or worse measured output motion."""
    if isinstance(before,HeadMotionEvidence):
        return isinstance(candidate,HeadMotionEvidence) and candidate.preserves_motion_of(before)
    return bool(isinstance(before,CloseupGestureEvidence) and isinstance(candidate,CloseupGestureEvidence)
        and before.motion is not None and candidate.motion is not None
        and before.motion.observed_bones<=candidate.motion.observed_bones
        and before.motion.output_intervals_seconds==candidate.motion.output_intervals_seconds
        and candidate.preserves_pose_continuity_of(before)
        and candidate.ranking_cost<=before.ranking_cost+1e-9)


def measured_motion_frontier(triage,maximum_boundaries):
    """Do not compare uncalibrated head and arm costs on one numeric axis.

    Alternate independently ranked bases inside the same four-camera budget;
    a missing basis donates its slots to the other, never creates extra work.
    """
    groups=[sorted((row for row in triage if isinstance(row[2],kind)),
        key=lambda row:row[:2],reverse=True) for kind in (CloseupGestureEvidence,HeadMotionEvidence)]
    selected=[]
    while len(selected)<maximum_boundaries and any(groups):
        for group in groups:
            if group and len(selected)<maximum_boundaries:selected.append(group.pop(0))
    return tuple(selected)


def retimed_incumbent_camera_preview(plan,numeric,boundary):
    """Cheap candidate ordering only; this curve owns no dense camera receipt.

    Preview each new native endpoint through the incumbent's absolute camera
    rather than selecting the two lowest motion costs with no framing evidence.
    The ordinary proposer still constructs and densely validates its own curve.
    """
    if (type(boundary) is not int or not 0<=boundary<len(plan.segments)-1
            or len(plan.segments)!=len(numeric.segments)):
        raise ValueError('incumbent camera preview requires one exact adjacent pair')
    segments=list(numeric.segments)
    for index in (boundary,boundary+1):
        old,new=plan.segments[index],numeric.segments[index]
        if old.source_index!=new.source_index:
            raise ValueError('incumbent camera preview cannot change sources')
        shift=new.master_start-old.master_start
        track=old.editorial_transform
        if track is not None:
            times=sorted({0.,new.duration,*(key.time_seconds-shift for key in track.keyframes
                if 0.<key.time_seconds-shift<new.duration)})
            keys=tuple(replace(track.evaluate(max(0.,time+shift)),time_seconds=time) for time in times)
            canvas=track.effective_sampling_canvas_zoom
            track=EditorialTransformTrack(keys,reason='retimed_incumbent_camera_ranking_preview_not_dense_authority',
                sampling_canvas_zoom=canvas if track.sampling_canvas_zoom is not None
                    or max(key.zoom for key in keys)<canvas else None)
        segments[index]=replace(new,editorial_transform=track,
            pan_crop=retime_base_pan(old.pan_crop,relative_shift=shift,duration=new.duration),
            crop=old.crop,look_adjustment=old.look_adjustment)
    return replace(numeric,segments=tuple(segments))


def measured_output_face_box(frame):
    """A bounded fresh GLOBAL face observation, never a central-person guess.

    The ordinary proxy detector may miss a small/rolled visible face. Use the
    same 854px global-dominance reacquisition as the complete-camera proposer;
    do not relabel a local body owner or a missing face as globally dominant.
    These pixels rank a candidate; they do not grant a dense camera binding.
    """
    from .native_global_face import observe_native_global_face
    return observe_native_global_face(frame).bbox_xywh


def focal_boundary_shortlist(timeline, *, maximum_candidates=12):
    """Rank the actual cut endpoints, not the unrelated middle of each shot.

    A medium -> close cut can be just as face-dominant as two close-up labels.
    Discrete label priority must not exclude it before motion is measured.
    """
    if type(maximum_candidates) is not int or not 1<=maximum_candidates<=48:
        raise ValueError('focal shortlist permits 1..48 bounded cached candidates')
    focal={ShotType.CLOSE_UP,ShotType.MEDIUM}
    targets=[]
    for index,(left,right) in enumerate(zip(timeline.choices,timeline.choices[1:])):
        if (left.source_index==right.source_index
                or (left.exit_shot_type or left.shot_type) not in focal
                or (right.entry_shot_type or right.shot_type) not in focal):
            continue
        scales=(left.exit_shot_scale if left.exit_shot_scale is not None else left.shot_scale,
                right.entry_shot_scale if right.entry_shot_scale is not None else right.shot_scale)
        if any(type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1
               for value in scales):
            continue
        targets.append((-min(scales),index))
    return tuple(index for _,index in sorted(targets)[:maximum_candidates])


@isolated_optional()
def refine_group_motion_camera(*,plan,timeline,audio_offsets,source_timewarps,scene_cut_times,
        model_path,source_index_map=None,feature_cache_directory=None,
        feature_derived_cache_directory=None,require_persistent_feature_cache=False,
        maximum_boundaries=4,maximum_camera_candidates=2,protected_segment_indices=(),
        maximum_candidate_boundaries=32,maximum_measured_boundaries=12):
    if type(maximum_boundaries) is not int or not 1 <= maximum_boundaries <= 4:
        raise ValueError('motion/camera refinement permits at most four boundaries')
    if type(maximum_camera_candidates) is not int or not 1 <= maximum_camera_candidates <= 2:
        raise ValueError('motion/camera refinement permits at most two camera candidates per boundary')
    if (type(maximum_candidate_boundaries) is not int or not 1<=maximum_candidate_boundaries<=48
            or type(maximum_measured_boundaries) is not int
            or not 1<=maximum_measured_boundaries<=maximum_candidate_boundaries):
        raise ValueError('motion/camera refinement requires bounded cached and measured candidate budgets')
    protected=tuple(protected_segment_indices)
    if (len(set(protected))!=len(protected)
            or any(type(index) is not int or not 0<=index<len(plan.segments) for index in protected)):
        raise ValueError('motion/camera protected segment domain is invalid')
    mapping=tuple(range(len(plan.sources))) if source_index_map is None else tuple(source_index_map)
    if (not len(mapping)==len(audio_offsets)==len(source_timewarps)==len(scene_cut_times)
            or len(set(mapping))!=len(mapping)
            or any(type(index) is not int or not 0<=index<len(plan.sources) for index in mapping)
            or any(type(value) not in (int,float) or not math.isfinite(value) for value in audio_offsets)
            or any(type(value) not in (int,float) or not math.isfinite(value) or value<0
                   for cuts in scene_cut_times for value in cuts)):
        raise ValueError('motion/camera source domains disagree')
    if (len(plan.segments)!=len(timeline.choices)
            or any(type(choice.source_index) is not int or not 0<=choice.source_index<len(mapping)
                   or mapping[choice.source_index]!=segment.source_index
                   for choice,segment in zip(timeline.choices,plan.segments))):
        raise ValueError('motion/camera timeline source domain disagrees with render plan')
    if plan.aspect_ratio.value != '16:9' or any(t.kind.uses_overlap for t in plan.transitions):
        raise ValueError('motion/camera refinement requires landscape hard cuts')
    started=monotonic()
    shortlist=focal_boundary_shortlist(timeline,maximum_candidates=maximum_candidate_boundaries)
    targets=tuple(index for index in shortlist if not {index,index+1}.intersection(protected))
    attempts=[dict(stage='protected_context',boundary_index=index,admitted=False,
        reason='independently_bound_whole_shot_or_fixed_neighbor_not_a_two_cell_edit')
        for index in shortlist if index not in targets]
    if not targets:
        return GroupMotionCameraRefinement(plan,timeline,(),tuple(attempts),monotonic()-started)
    clocks={i:probe_render_frame_clock(source) for i,source in enumerate(plan.sources)}
    provider=CloseupGestureProvider(plan.sources,Path(model_path),verify_native_global_focal=True)
    samplers={}
    local_index={source:index for index,source in enumerate(mapping)}
    def sampler(source):
        if source not in samplers:
            samplers[source]=_VideoSampler(plan.sources[source],feature_cache_directory=feature_cache_directory,
                feature_derived_cache_directory=feature_derived_cache_directory,
                require_persistent_feature_cache=require_persistent_feature_cache)
        return samplers[source]
    def measured(candidate,boundary,required_basis=None):
        samples=hard_cut_boundary_sample_clocks(candidate,clocks,boundary_index=boundary)
        for first,last in ((samples[0],samples[1]),(samples[2],samples[3])):
            rate=float(clocks[first.source_index].frame_rate)
            if any(first.native_frame_index/rate < cut <= last.native_frame_index/rate
                   for cut in scene_cut_times[local_index[first.source_index]]):
                return None,None,None
        descriptors=[]
        values=[]
        for sample in samples:
            seconds=sample.native_frame_index/float(clocks[sample.source_index].frame_rate)
            descriptor=sampler(sample.source_index).feature_descriptor_at(seconds,
                include_subject_geometry=True,include_formation_pose=True)
            if descriptor is None:
                return None,None,None
            descriptors.append(descriptor)
            values.append((sample.source_index,seconds,clocks[sample.source_index],descriptor))
        gesture=(provider.compare_motion(*values,output_fps=plan.fps)
            if required_basis!=HEAD_MOTION_BASIS else None)
        if gesture is None and hasattr(provider,'compare_head_motion'):
            gesture=provider.compare_head_motion(*values,output_fps=plan.fps)
        return gesture,tuple(descriptors),samples
    def faces(candidate,boundary,samples):
        counts=segment_frame_counts(candidate)
        boxes=[]
        for index,sample in zip((boundary,boundary+1),samples[1:3]):
            frame=provider.capture.frame_at_native_index(sample.source_index,sample.native_frame_index,
                clocks[sample.source_index])
            if frame is None:
                return None
            geometry=PrescribedGeometrySampler(sampler(sample.source_index),candidate.segments[index],
                master_start=candidate.master_audio_start+candidate.segments[index].master_start)
            relative=(sample.output_frame_index-sum(counts[:index]))/candidate.fps
            output=_reference_view_for_span(frame,geometry.span,relative_time=relative,
                size=(candidate.width,candidate.height))
            boxes.append(measured_output_face_box(output))
        return _box_discontinuity(*boxes) if all(box is not None for box in boxes) else None
    def projected(candidate,boundary,samples,required_basis=None):
        counts=segment_frame_counts(candidate);frames=[]
        for index,sample in zip((boundary,boundary,boundary+1,boundary+1),samples):
            frame=provider.capture.frame_at_native_index(sample.source_index,sample.native_frame_index,
                clocks[sample.source_index])
            if frame is None:return None
            geometry=PrescribedGeometrySampler(sampler(sample.source_index),candidate.segments[index],
                master_start=candidate.master_audio_start+candidate.segments[index].master_start)
            frames.append(_reference_view_for_span(frame,geometry.span,
                relative_time=(sample.output_frame_index-sum(counts[:index]))/candidate.fps,
                size=(candidate.width,candidate.height)))
        return measure_projected_motion(frames,samples,provider,output_fps=candidate.fps,required_basis=required_basis)
    # These cameras belong to an independent complete context transaction.
    # Editing one of its fixed neighbors is not a proposal this stage can
    # compose. Avoid decoding/ranking a candidate known to lose its receipt.
    triage=[]
    selected=[]
    try:
        for boundary in targets:
            if len(triage)>=maximum_measured_boundaries:
                break
            try:
                source_gesture,source_descriptors,samples=measured(plan,boundary)
                gesture=projected(plan,boundary,samples) if samples is not None else None
                baseline_faces=(faces(plan,boundary,samples) if gesture is not None
                    and gesture.motion is not None and samples is not None else None)
            except ValueError as error:
                attempts.append(dict(stage='baseline_motion',boundary_index=boundary,
                    admitted=False,reason=str(error)))
                continue
            baseline_rank=camera_motion_rank(gesture,baseline_faces,enforce_face_limits=False)
            if baseline_rank is not None:
                # Cached face scale only orders sensing; missing four-frame
                # motion/output faces do not consume a measured repair slot.
                # Keep these exact measurements rather than decode them again.
                triage.append((baseline_rank,boundary,gesture,samples,baseline_faces,baseline_rank,
                    source_gesture,source_descriptors))
            attempts.append(dict(stage='baseline_motion',boundary_index=boundary,admitted=False,
                motion_cost=gesture.ranking_cost if gesture is not None and gesture.motion is not None else None,
                motion_evidence_basis=motion_evidence_basis(gesture),
                source_motion_cost=source_gesture.ranking_cost if source_gesture is not None else None,
                source_motion_evidence_basis=motion_evidence_basis(source_gesture),
                motion_ranking_coordinate_basis='actual_projected_output_pixels',
                output_face_metrics=baseline_faces,complete_measured_candidate=baseline_rank is not None))
        attempts.append(dict(stage='bounded_measured_focal_frontier',admitted=False,
            cached_candidate_budget=maximum_candidate_boundaries,measured_candidate_budget=maximum_measured_boundaries,
            cached_eligible_boundaries=len(shortlist),measured_candidate_count=len(triage),
            measured_boundary_indices=[row[1] for row in triage],
            cached_geometry_used_only_for_sensing_order=True,quality_approved=False))
        for _,boundary,before,samples,before_faces,before_rank,source_before,before_descriptors in measured_motion_frontier(triage,maximum_boundaries):
            if any(abs(boundary-proposal.boundary_index)<=1 for proposal in selected):
                continue
            offsets=[]
            for offset in (-6,-4,-2,2,4,6):
                delta=offset/plan.fps
                left,right=timeline.choices[boundary:boundary+2]
                cut=right.canonical_master_start+delta
                if min(cut-left.canonical_master_start,right.canonical_master_end-cut)<.55:
                    continue
                pair=tuple(aligned_render_segment(source_index=mapping[choice.source_index],
                    reference_start=start,common_master_start=plan.master_audio_start,
                    source_offset=audio_offsets[choice.source_index],duration=end-start,
                    timewarp=source_timewarps[choice.source_index])
                    for choice,start,end in ((left,left.canonical_master_start,cut),(right,cut,right.canonical_master_end)))
                numeric=replace(plan,segments=(*plan.segments[:boundary],*pair,*plan.segments[boundary+2:]))
                try:
                    source_gesture,descriptors,native=measured(numeric,boundary,required_basis=motion_evidence_basis(before))
                except ValueError:
                    continue
                if native is None:continue
                preview=retimed_incumbent_camera_preview(plan,numeric,boundary)
                gesture=projected(preview,boundary,native,required_basis=motion_evidence_basis(before))
                improves=measured_motion_improves(gesture,before)
                preview_faces=preview_rank=None
                if projected_motion_preserved(gesture,before):
                    preview_faces=faces(preview,boundary,native)
                    preview_rank=camera_motion_rank(gesture,preview_faces,enforce_face_limits=False)
                attempts.append(dict(stage='directional_candidate',boundary_index=boundary,
                    offset_frames=offset,admitted=False,motion_rank_improved=improves,
                    motion_evidence_basis=motion_evidence_basis(before),
                    before_motion_bones=(sorted(before.motion.observed_bones) if isinstance(before,CloseupGestureEvidence) else []),
                    proposed_motion_bones=(sorted(gesture.motion.observed_bones)
                        if isinstance(gesture,CloseupGestureEvidence) and gesture.motion is not None else []),
                    head_motion_dimensions=list(HEAD_DIMENSIONS) if isinstance(before,HeadMotionEvidence) else [],
                    endpoint_only_bones_no_longer_observed=(sorted(before.observed_bones-gesture.observed_bones)
                        if gesture is not None else []),
                    proposed_motion_cost=gesture.ranking_cost if gesture is not None and gesture.motion is not None else None,
                    source_motion_cost=source_gesture.ranking_cost if source_gesture is not None else None,
                    source_motion_evidence_basis=motion_evidence_basis(source_gesture),
                    motion_ranking_coordinate_basis='actual_projected_output_pixels',
                    incumbent_camera_preview_face_metrics=preview_faces,
                    incumbent_camera_preview_joint_rank=preview_rank,
                    preview_grants_dense_camera_authority=False,
                    unknown_motion_bones_imputed=False))
                # A crop can hide a bone genuinely seen in the raw source.
                # Compare the incumbent output domain, not that unrelated raw
                # domain. Equally good output motion can accompany better
                # framing; no source-only motion cost authorizes this candidate.
                # The incumbent-camera preview orders work; it cannot veto a
                # timing whose worse FACE framing is exactly what the complete
                # dense camera is supposed to repair. Motion must still survive.
                if preview_rank is not None and projected_motion_preserved(gesture,before):
                    offsets.append((gesture.ranking_cost,offset,pair,gesture,descriptors,native,preview_rank,source_gesture))
            eligible=[];before_projected=before
            ranked=sorted(offsets,key=lambda row:(row[6],row[0],abs(row[1]),row[1]))
            # A bad cut can need camera alignment without any timing change.
            # Reserve one of the SAME two slots for this unchanged-clock recipe,
            # rather than requiring a source-motion gain before trying a camera.
            unchanged=(before.ranking_cost,0,plan.segments[boundary:boundary+2],before,
                before_descriptors,samples,before_rank,source_before)
            frontier=((*ranked[:maximum_camera_candidates-1],unchanged)
                if maximum_camera_candidates>1 else tuple(ranked[:1] or [unchanged]))
            for _,offset,pair,gesture,descriptors,native,preview_rank,source_gesture in frontier:
                row=dict(stage='complete_camera_candidate',boundary_index=boundary,offset_frames=offset,
                    before_motion_cost=before.ranking_cost,proposed_motion_cost=gesture.ranking_cost,
                    motion_evidence_basis=motion_evidence_basis(before),
                    before_motion_evidence=before.to_dict(),proposed_motion_evidence=gesture.to_dict(),
                    before_source_motion_evidence=source_before.to_dict() if source_before else None,
                    proposed_source_motion_evidence=source_gesture.to_dict() if source_gesture else None,
                    motion_ranking_coordinate_basis='actual_projected_output_pixels',
                    incumbent_camera_preview_joint_rank=preview_rank,
                    preview_grants_dense_camera_authority=False,
                    before_output_face_metrics=before_faces,admitted=False)
                try:
                    camera_arguments=dict(plan=plan,timeline=timeline,boundary_index=boundary,
                        shifted_segments=pair,source_clocks=clocks,scene_cut_times=scene_cut_times,
                        source_index_map=mapping,preserved_boundary_frames=4)
                    try:
                        proposal=build_group_cut_camera_proposal(**camera_arguments)
                    except ValueError as error:
                        reason=str(error).split(': ',1)[-1]
                        if reason not in {'shot_policy_family_mismatch','exact_subject_would_be_clipped',
                                'per_frame_layout_mismatch_gate'}:
                            raise
                        # One explicit, globally dominant action-camera retry
                        # inside this candidate's existing bounded slot. Do
                        # not relabel central attention or lower containment.
                        # Aggregate background layout can fail an otherwise
                        # valid dominant-performer action shot; this separate
                        # recipe must freshly prove its own head AND arm owner.
                        row['ordinary_camera_rejection']=reason
                        proposal=build_group_cut_camera_proposal(**camera_arguments,
                            global_action_framing=True,action_framing_model_path=model_path)
                        row['camera_geometry_contract']='global_head_arms_and_retained_opposite_camera'
                    final_native=hard_cut_boundary_sample_clocks(proposal.plan,clocks,boundary_index=boundary)
                    if any((a.source_index,a.native_frame_index,a.output_frame_index)!=(b.source_index,b.native_frame_index,b.output_frame_index)
                           for a,b in zip(native,final_native,strict=True)):
                        raise ValueError('camera construction changed the measured native motion clock')
                    projected_faces=faces(proposal.plan,boundary,native)
                    after_projected=(projected(proposal.plan,boundary,native,
                        required_basis=motion_evidence_basis(before_projected))
                        if before_projected is not None else None)
                    output_before_rank=camera_motion_rank(before_projected,before_faces,enforce_face_limits=False)
                    rank=camera_motion_rank(after_projected,projected_faces)
                    row.update(proposed_output_face_metrics=projected_faces,
                        before_source_motion_face_rank=camera_motion_rank(source_before,before_faces,enforce_face_limits=False),
                        before_joint_rank=output_before_rank,proposed_joint_rank=rank,
                        before_projected_motion=before_projected.to_dict() if before_projected else None,
                        proposed_projected_motion=after_projected.to_dict() if after_projected else None,
                        projected_motion_required_for_promotion=True)
                    if not projected_motion_preserved(after_projected,before_projected):
                        row['reason']='projected_output_motion_unobserved_or_worsened'
                        attempts.append(row);continue
                    if rank is None or output_before_rank is None or rank>=output_before_rank-1e-9:
                        row['reason']='output_motion_face_rank_not_improved'
                        attempts.append(row)
                        continue
                    raw=_visual_evidence(descriptors[1],descriptors[2],outgoing_preceding=descriptors[0],
                        incoming_following=descriptors[3],output_fps=plan.fps,side_sample_span_seconds=3/plan.fps)
                    choices=list(proposal.clock_timeline.choices)
                    cut=choices[boundary+1].canonical_master_start
                    for index,update,descriptor in ((boundary,_updated_exit,descriptors[1]),
                                                   (boundary+1,_updated_entry,descriptors[2])):
                        choice=choices[index]
                        choices[index]=update(choice,boundary=cut,
                            source_local_boundary=choice.source_local_end if index==boundary else choice.source_local_start,
                            descriptor=descriptor,evidence=raw,
                            internal_scene_cut_master_seconds=choice.internal_scene_cut_master_seconds,
                            scene_risk=choice.exit_scene_cut_near if index==boundary else choice.entry_scene_cut_near,
                            boundary_changed=True)
                    refreshed=replace(proposal.clock_timeline,choices=tuple(choices))
                    failures=_new_hard_failures(review_timeline(timeline,available_source_count=len(mapping)),
                        review_timeline(refreshed,available_source_count=len(mapping)))
                    row['raw_new_failures']=failures
                    # These remain reported RAW failures; actual output face
                    # bounds and complete dense containment govern this proposal.
                    if any(failure['code'] not in {'scale_mismatch_exceeded','focus_mismatch_exceeded','layout_mismatch_exceeded'}
                           or any(index!=boundary for index in failure.get('occurrences',())) for failure in failures):
                        row['reason']='new_non_camera_or_unrelated_timeline_failure'
                        attempts.append(row)
                        continue
                    row.update(eligible=True,reason='complete_local_candidate_for_final_decoded_qa',
                        raw_new_failures=failures,proposal=proposal.to_dict())
                    eligible.append((rank,abs(offset),offset,proposal,refreshed,row))
                    attempts.append(row)
                except ValueError as error:
                    row['reason']=str(error)
                    attempts.append(row)
            if eligible:
                _,_,_,proposal,refreshed,row=min(eligible,key=lambda value:value[:3])
                row['admitted']=True
                selected.append(proposal)
                plan,timeline=proposal.plan,refreshed
    finally:
        provider.close()
        for value in samplers.values():
            value.close()
    return GroupMotionCameraRefinement(plan,timeline,tuple(selected),tuple(attempts),monotonic()-started)
