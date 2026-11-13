"""Bounded common-clock motion/camera refinement of an owned GROUP plan.

Actual projected output motion and faces propose and rank time offsets and
complete dense camera candidates. Source motion remains diagnostic evidence:
source-only bones cannot become required observations on a different crop.
Raw timeline failures stay raw failures.
This grants only a better local candidate for final decoded QA, never a
professional quality, temporal-phase or cross-camera identity certificate.
"""
from dataclasses import dataclass,replace
from collections.abc import Mapping
import math
from pathlib import Path
from time import monotonic

from .optional_native_worker import isolated_optional
from .render import segment_frame_counts,EditorialTransformTrack
from .render_boundary_clock import hard_cut_boundary_sample_clocks
from .render_frame_schedule import probe_render_frame_clock
from .closeup_gesture import (CloseupGestureProvider,CloseupGestureEvidence,
    compare_arm_articulation,measure_arm_motion,MOTION_CONTRACT)
from .observed_head_motion import HeadMotionEvidence,HEAD_MOTION_BASIS,HEAD_DIMENSIONS
from .group_cut_camera_proposal import build_group_cut_camera_proposal,retime_base_pan
from .group_cut_timing import propose_numeric_group_cut_timing,native_flash_timing_frontier
from .group_source_domain import GroupSourceDomain
from .prescribed_render_geometry import PrescribedGeometrySampler
from .post_render_perceptual_quality import _reference_view_for_span
from .rendered_proxy_review import _default_geometry_provider,_box_discontinuity,ProxyReviewConfig
from .timeline import ShotType
from .timeline_review import review_timeline
from .visual_boundary_repair import _visual_evidence,_updated_entry,_updated_exit,_new_hard_failures
from .visual_score import _VideoSampler
from .group_added_camera_motion import measure_added_camera_motion


PROJECTED_ARM_HEAD_CONTRACT = 'exact_output_arm_motion_with_independently_measured_head_guard_v1'


@dataclass(frozen=True, slots=True)
class ProjectedArmHeadMotionEvidence(CloseupGestureEvidence):
    """Arm ranking with a separate observed head-motion non-regression guard.

    Head and arm units are not averaged, and an absent head is not zero cost.
    This remains an optional camera-candidate comparison, not cut approval.
    """
    head_motion_guard: HeadMotionEvidence | None = None

    def __post_init__(self):
        CloseupGestureEvidence.__post_init__(self)
        if self.motion is None:
            raise ValueError('projected arm/head evidence requires real four-frame arm motion')
        guard = self.head_motion_guard
        if guard is not None and (not isinstance(guard, HeadMotionEvidence)
                or guard.motion.native_frames != self.motion.native_frames
                or guard.motion.output_intervals_seconds != self.motion.output_intervals_seconds):
            raise ValueError('projected head guard must own the same exact four-frame output clock')
        if guard is not None and guard.endpoint_arm_guard is not None:
            raise ValueError('secondary head evidence must not replace moving arms with a static endpoint gate')

    def to_dict(self):
        payload = CloseupGestureEvidence.to_dict(self)
        payload.update(contract=PROJECTED_ARM_HEAD_CONTRACT,
            primary_motion_evidence_basis=MOTION_CONTRACT,
            head_motion_guard=self.head_motion_guard.to_dict() if self.head_motion_guard else None,
            head_motion_unknown_not_imputed=self.head_motion_guard is None,
            head_and_arm_costs_averaged=False,
            jointly_observed_motion_required_for_camera_promotion=True)
        return payload

    @classmethod
    def from_dict(cls, payload):
        """Do not inherit an arm-only decoder that silently drops the head."""
        from .observed_head_motion import ObservedHeadMotion
        from .closeup_gesture import ObservedArmMotion
        if not isinstance(payload, Mapping) or payload.get('contract') != PROJECTED_ARM_HEAD_CONTRACT:
            raise ValueError('unknown projected arm/head evidence contract')
        guard = None
        raw = payload.get('head_motion_guard')
        if raw is not None:
            if not isinstance(raw, Mapping) or not isinstance(raw.get('motion'), Mapping):
                raise ValueError('projected head guard must contain measured scalar evidence')
            measured = raw['motion']
            guard = HeadMotionEvidence(ObservedHeadMotion(tuple(measured['native_frames']),
                tuple(measured['output_intervals_seconds']),measured['predicted_pose_cost'],
                measured['velocity_jump_per_output_tick']))
            if guard.to_dict() != dict(raw):
                raise ValueError('projected head payload disagrees with its measurements or authority')
        value = cls(payload['outgoing_native_frame'],payload['incoming_native_frame'],
            tuple(payload['bone_mismatches']),payload['outgoing_focal_basis'],payload['incoming_focal_basis'],
            ObservedArmMotion.from_dict(payload['motion']),guard)
        if value.to_dict() != dict(payload):
            raise ValueError('projected arm/head payload lost observations or changed its authority')
        return value


def _projected_head_guard_preserved(candidate, before):
    """A better arm rank cannot hide or discard an observed head regression."""
    guard = before.head_motion_guard if isinstance(before, ProjectedArmHeadMotionEvidence) else None
    if guard is None:
        return True
    return (isinstance(candidate, ProjectedArmHeadMotionEvidence)
        and candidate.head_motion_guard is not None
        and candidate.head_motion_guard.preserves_motion_of(guard))


def _observed_head_evidence(evidence):
    if isinstance(evidence, HeadMotionEvidence):
        return evidence
    if isinstance(evidence, ProjectedArmHeadMotionEvidence):
        return evidence.head_motion_guard
    return None


@dataclass(frozen=True, slots=True)
class ProjectedBoundaryObservation:
    """Motion and framing from the SAME four actual output pixel views.

    This is run-local sensing, not stored geometry or quality authority. The
    two endpoint heads have already been measured for motion; decoding,
    projecting and detecting them again adds no independent evidence.
    """
    motion: CloseupGestureEvidence | HeadMotionEvidence | None = None
    face_metrics: tuple[float, float] | None = None


def observe_projected_boundary(frames,samples,provider,*,output_fps,required_basis=None):
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
        return ProjectedBoundaryObservation()
    ticks=tuple(sample.output_frame_index for sample in samples)
    if (ticks[1]-ticks[0],ticks[2]-ticks[1],ticks[3]-ticks[2])!=(3,1,3):
        raise ValueError('projected motion views must own the exact four output ticks')
    if (samples[0].source_index!=samples[1].source_index
            or samples[2].source_index!=samples[3].source_index
            or samples[0].source_index==samples[2].source_index
            or any(sample.terminal_clone for sample in samples)):
        return ProjectedBoundaryObservation()
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
    face_metrics = (_box_discontinuity(heads[1].bbox_xywh,heads[2].bbox_xywh)
        if heads[1].bbox_xywh is not None and heads[2].bbox_xywh is not None else None)
    endpoint=compare_arm_articulation(arms[1],arms[2],outgoing_native_frame=ids[1],
        incoming_native_frame=ids[2],allow_partial=True)
    arm_motion = (measure_arm_motion(tuple(arms),native_frames=ids,output_intervals_seconds=intervals)
        if required_basis != HEAD_MOTION_BASIS else None)
    if required_basis == MOTION_CONTRACT and (arm_motion is None or endpoint is None):
        return ProjectedBoundaryObservation(face_metrics=face_metrics)
    paths_verified = True
    for a,b in ((0,1),(2,3)):
        ha,hb=heads[a],heads[b]
        if ha.landmarks_xy and hb.landmarks_xy and focal_landmarks_track(
                working[a],working[b],ha.landmarks_xy,hb.landmarks_xy):continue
        box=_optical_flow_box(working[a],working[b],ha.bbox_xywh) if ha.bbox_xywh else None
        if box is None or hb.bbox_xywh is None or _iou(box,hb.bbox_xywh)<.60:
            paths_verified = False
            break
    motion=measure_head_motion(tuple(head_motion_signature(h.geometry,h.landmarks_xy) for h in heads),
        native_frames=ids,output_intervals_seconds=intervals,same_source_face_paths_pixel_verified=paths_verified)
    if arm_motion is not None and endpoint is not None:
        # Arms already have a real four-frame motion comparator. Do not add a
        # second frozen-pose requirement through the head-only fallback guard.
        head = HeadMotionEvidence(motion) if motion is not None else None
        measured = ProjectedArmHeadMotionEvidence(endpoint.outgoing_native_frame,endpoint.incoming_native_frame,
            endpoint.bone_mismatches,endpoint.outgoing_focal_basis,endpoint.incoming_focal_basis,
            arm_motion,head)
    else:
        measured = HeadMotionEvidence(motion,endpoint) if motion is not None else None
    return ProjectedBoundaryObservation(measured,face_metrics)


def measure_projected_motion(frames,samples,provider,*,output_fps,required_basis=None):
    """Compatibility entry point for callers needing only the measured motion."""
    return observe_projected_boundary(frames,samples,provider,
        output_fps=output_fps,required_basis=required_basis).motion


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
            cross_camera_identity_proven=False,ranking_calibrated=False,
            secondary_output_head_motion_not_discarded_when_arms_observed=True,
            head_and_arm_costs_averaged=False)


def camera_motion_rank(gesture, face_metrics, *, enforce_face_limits=True, added_camera_cost=0.):
    """Missing motion/face evidence has no numeric rank; legacy QA limits stay."""
    limits=ProxyReviewConfig()
    if (gesture is None or gesture.motion is None or face_metrics is None
            or type(added_camera_cost) not in (int,float)
            or not math.isfinite(added_camera_cost) or added_camera_cost<0):
        return None
    scale,position=face_metrics
    if (any(type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in (scale,position))
            or enforce_face_limits and (scale > limits.maximum_face_scale_discontinuity
                or position > limits.maximum_face_position_discontinuity)):
        return None
    # Soft, uncalibrated editorial preference. A four-frame seam gain must
    # pay for the whole authored camera path, not only its endpoint crop.
    # This never buys a waiver of measured head/arm or physical canvas gates.
    return gesture.ranking_cost+scale+position+.2*added_camera_cost


def measured_motion_improves(candidate, before):
    """Compare the incumbent's *four-frame* domain without losing any bone.

    An arm visible in only the two cut frames has no incumbent motion value.
    It stays unknown here rather than acting as a newly required four-frame
    measurement. Static-only callers keep their separate, stricter comparator.
    New measured bones still contribute to the candidate's worst-bone cost.
    """
    if not _projected_head_guard_preserved(candidate, before):
        return False
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
    if not _projected_head_guard_preserved(candidate, before):
        return False
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


def boundary_timing_hypotheses(*,boundary_index,native_flashes,allow_retiming):
    """Try exact native collisions before the ordinary even-frame offsets.

    These are source-ownership proposals, not motion/geometry approval. The
    explicit collapse policy is never lent to a different nearby offset.
    Fixed-clock callers cannot gain a flash-removal exception through this list.
    """
    if type(allow_retiming) is not bool:
        raise TypeError('cut-time hypothesis policy must be boolean')
    if not allow_retiming:return ()
    exact=tuple((flash.timing_offset_frames,True) for flash in native_flashes
        if flash.boundary_index==boundary_index)
    used={offset for offset,_ in exact}
    return (*exact,*tuple((offset,False) for offset in (-6,-4,-2,2,4,6) if offset not in used))


@isolated_optional()
def refine_group_motion_camera(*,plan,timeline,audio_offsets,source_timewarps,scene_cut_times,
        model_path,source_index_map=None,feature_cache_directory=None,
        feature_derived_cache_directory=None,require_persistent_feature_cache=False,
        maximum_boundaries=4,maximum_camera_candidates=2,protected_segment_indices=(),
        maximum_candidate_boundaries=32,maximum_measured_boundaries=12,
        allow_retiming=True,require_incident_edge_replay=False,maximum_accepted_boundaries=None):
    if type(allow_retiming) is not bool or type(require_incident_edge_replay) is not bool:
        raise TypeError("motion/camera clock and incident replay policies must be boolean")
    if type(maximum_boundaries) is not int or not 1 <= maximum_boundaries <= 4:
        raise ValueError('motion/camera refinement permits at most four boundaries')
    if maximum_accepted_boundaries is not None and (type(maximum_accepted_boundaries) is not int
            or not 1 <= maximum_accepted_boundaries <= maximum_boundaries):
        raise ValueError('accepted boundary budget must fit the measured frontier')
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
    if not targets and (not allow_retiming or not any(scene_cut_times)):
        return GroupMotionCameraRefinement(plan,timeline,(),tuple(attempts),monotonic()-started)
    clocks={i:probe_render_frame_clock(plan.sources[i]) for i in mapping}
    domain=GroupSourceDomain(tuple(plan.sources),mapping,tuple(audio_offsets),tuple(source_timewarps),
        tuple(tuple(cuts) for cuts in scene_cut_times),tuple(clocks[index] for index in mapping))
    native_flashes=(native_flash_timing_frontier(plan=plan,domain=domain)
        if allow_retiming and plan.output_frame_count<=20_000 else ())
    # The old focal label belongs to the short wide that is being removed.
    # Let a real native/source collision reach motion sensing without treating
    # that label as a permanent veto. The same measured/camera work limits and
    # known-motion preservation checks still apply; missing motion is not zero.
    flash_targets=tuple(dict.fromkeys(f.boundary_index for f in native_flashes
        if not {f.boundary_index,f.boundary_index+1}.intersection(protected)))
    targets=(*flash_targets,*tuple(i for i in targets if i not in flash_targets))[:maximum_candidate_boundaries]
    attempts.append(dict(stage='exact_native_flash_frontier',admitted=False,
        collisions=[f.to_dict() for f in native_flashes],candidate_generation_only=True,
        motion_geometry_or_quality_gates_relaxed=False))
    if not targets:
        return GroupMotionCameraRefinement(plan,timeline,(),tuple(attempts),monotonic()-started)
    provider=CloseupGestureProvider(plan.sources,Path(model_path),verify_native_global_focal=True)
    samplers={}
    local_index={source:index for index,source in enumerate(mapping)}
    camera_cuts={source:scene_cut_times[index] for index,source in enumerate(mapping)}
    camera_burdens={}
    def camera_burden(candidate,boundary):
        key=(id(candidate),boundary)
        if key not in camera_burdens:
            evidence=measure_added_camera_motion(candidate,segment_indices=(boundary,boundary+1),
                source_clocks=clocks,scene_cut_times=camera_cuts)
            # Retain the plan too: Python cannot reuse an id in this bounded
            # cache and accidentally transfer one camera's cost to another.
            camera_burdens[key]=(candidate,evidence)
        return camera_burdens[key][1]
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
    def projected(candidate,boundary,samples,required_basis=None):
        counts=segment_frame_counts(candidate);frames=[]
        for index,sample in zip((boundary,boundary,boundary+1,boundary+1),samples):
            frame=provider.capture.frame_at_native_index(sample.source_index,sample.native_frame_index,
                clocks[sample.source_index])
            if frame is None:return ProjectedBoundaryObservation()
            geometry=PrescribedGeometrySampler(sampler(sample.source_index),candidate.segments[index],
                master_start=candidate.master_audio_start+candidate.segments[index].master_start)
            frames.append(_reference_view_for_span(frame,geometry.span,
                relative_time=(sample.output_frame_index-sum(counts[:index]))/candidate.fps,
                size=(candidate.width,candidate.height)))
        return observe_projected_boundary(frames,samples,provider,
            output_fps=candidate.fps,required_basis=required_basis)
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
                observation=(projected(plan,boundary,samples) if samples is not None
                    else ProjectedBoundaryObservation())
                gesture,baseline_faces=observation.motion,observation.face_metrics
                baseline_camera=camera_burden(plan,boundary)
            except ValueError as error:
                attempts.append(dict(stage='baseline_motion',boundary_index=boundary,
                    admitted=False,reason=str(error)))
                continue
            baseline_rank=camera_motion_rank(gesture,baseline_faces,enforce_face_limits=False,
                added_camera_cost=baseline_camera.ranking_cost)
            if baseline_rank is not None:
                # Cached face scale only orders sensing; missing four-frame
                # motion/output faces do not consume a measured repair slot.
                # Keep these exact measurements rather than decode them again.
                triage.append((baseline_rank,boundary,gesture,samples,baseline_faces,baseline_rank,
                    source_gesture,source_descriptors))
            attempts.append(dict(stage='baseline_motion',boundary_index=boundary,admitted=False,
                motion_cost=gesture.ranking_cost if gesture is not None and gesture.motion is not None else None,
                motion_evidence_basis=motion_evidence_basis(gesture),
                observed_output_head_motion=(_observed_head_evidence(gesture).to_dict()
                    if _observed_head_evidence(gesture) is not None else None),
                source_motion_cost=source_gesture.ranking_cost if source_gesture is not None else None,
                source_motion_evidence_basis=motion_evidence_basis(source_gesture),
                motion_ranking_coordinate_basis='actual_projected_output_pixels',
                whole_added_camera_motion=baseline_camera.to_dict(),
                output_face_metrics=baseline_faces,complete_measured_candidate=baseline_rank is not None))
        attempts.append(dict(stage='bounded_measured_focal_frontier',admitted=False,
            cached_candidate_budget=maximum_candidate_boundaries,measured_candidate_budget=maximum_measured_boundaries,
            cached_eligible_boundaries=len(shortlist),measured_candidate_count=len(triage),
            measured_boundary_indices=[row[1] for row in triage],
            cached_geometry_used_only_for_sensing_order=True,quality_approved=False))
        for _,boundary,before,samples,before_faces,before_rank,source_before,before_descriptors in measured_motion_frontier(triage,maximum_boundaries):
            if maximum_accepted_boundaries is not None and len(selected)>=maximum_accepted_boundaries:
                break
            if any(abs(boundary-proposal.boundary_index)<=1 for proposal in selected):
                continue
            offsets=[]
            for offset,collapse_flash in boundary_timing_hypotheses(boundary_index=boundary,
                    native_flashes=native_flashes,allow_retiming=allow_retiming):
                delta=offset/plan.fps
                left,right=timeline.choices[boundary:boundary+2]
                cut=right.canonical_master_start+delta
                if min(cut-left.canonical_master_start,right.canonical_master_end-cut)<.55:
                    continue
                # A boundary-time change is not a new source phase. Rebuilding
                # two cells from global audio offsets discarded incumbent local
                # video timing and broke adjacent same-native renderer units.
                # Preserve the complete absolute native/camera curves instead;
                # typed local-clock receipts must be renewed, never transferred.
                try:
                    timing=propose_numeric_group_cut_timing(plan=plan,domain=domain,
                        boundary_index=boundary,timing_offset_frames=offset,
                        collapse_incident_native_flash=collapse_flash)
                except ValueError as error:
                    attempts.append(dict(stage='directional_clock_preview',boundary_index=boundary,
                        offset_frames=offset,admitted=False,reason=str(error),
                        failed_before_candidate_motion_sensing=True))
                    continue
                preview=timing.plan
                pair=preview.segments[boundary:boundary+2]
                try:
                    source_gesture,descriptors,native=measured(preview,boundary,required_basis=motion_evidence_basis(before))
                except ValueError:
                    continue
                if native is None:continue
                observation=projected(preview,boundary,native,required_basis=motion_evidence_basis(before))
                gesture=observation.motion
                improves=measured_motion_improves(gesture,before)
                preview_faces=preview_rank=None
                if projected_motion_preserved(gesture,before):
                    preview_faces=observation.face_metrics
                    preview_rank=camera_motion_rank(gesture,preview_faces,enforce_face_limits=False,
                        added_camera_cost=camera_burden(preview,boundary).ranking_cost)
                attempts.append(dict(stage='directional_candidate',boundary_index=boundary,
                    offset_frames=offset,admitted=False,motion_rank_improved=improves,
                    motion_evidence_basis=motion_evidence_basis(before),
                    before_motion_bones=(sorted(before.motion.observed_bones) if isinstance(before,CloseupGestureEvidence) else []),
                    proposed_motion_bones=(sorted(gesture.motion.observed_bones)
                        if isinstance(gesture,CloseupGestureEvidence) and gesture.motion is not None else []),
                    head_motion_dimensions=list(HEAD_DIMENSIONS) if _observed_head_evidence(before) is not None else [],
                    secondary_head_motion_required=isinstance(before,ProjectedArmHeadMotionEvidence)
                        and before.head_motion_guard is not None,
                    secondary_head_motion_preserved=_projected_head_guard_preserved(gesture,before),
                    endpoint_only_bones_no_longer_observed=(sorted(before.observed_bones-gesture.observed_bones)
                        if gesture is not None else []),
                    proposed_motion_cost=gesture.ranking_cost if gesture is not None and gesture.motion is not None else None,
                    source_motion_cost=source_gesture.ranking_cost if source_gesture is not None else None,
                    source_motion_evidence_basis=motion_evidence_basis(source_gesture),
                    motion_ranking_coordinate_basis='actual_projected_output_pixels',
                    incumbent_camera_preview_face_metrics=preview_faces,
                    incumbent_camera_preview_joint_rank=preview_rank,
                    numeric_cut_time_preview=timing.to_dict(),
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
                    offsets.append((gesture.ranking_cost,offset,pair,gesture,descriptors,native,
                        preview_rank,source_gesture,timing if collapse_flash else None))
            eligible=[];before_projected=before
            ranked=sorted(offsets,key=lambda row:(row[6],row[0],abs(row[1]),row[1]))
            # A bad cut can need camera alignment without any timing change.
            # Reserve one of the SAME two slots for this unchanged-clock recipe,
            # rather than requiring a source-motion gain before trying a camera.
            unchanged=(before.ranking_cost,0,plan.segments[boundary:boundary+2],before,
                before_descriptors,samples,before_rank,source_before,None)
            frontier=((*ranked[:maximum_camera_candidates-1],unchanged)
                if maximum_camera_candidates>1 else tuple(ranked[:1] or [unchanged]))
            for _,offset,pair,gesture,descriptors,native,preview_rank,source_gesture,flash_timing in frontier:
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
                    if flash_timing is not None:
                        camera_arguments.update(native_flash_cut_timing=flash_timing,source_domain=domain)
                    try:
                        proposal=build_group_cut_camera_proposal(**camera_arguments)
                    except ValueError as error:
                        if hasattr(error,'alignment'):
                            row['ordinary_dense_camera_failure']=error.alignment.to_dict()
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
                    observation=(projected(proposal.plan,boundary,native,
                        required_basis=motion_evidence_basis(before_projected))
                        if before_projected is not None else ProjectedBoundaryObservation())
                    projected_faces,after_projected=observation.face_metrics,observation.motion
                    old_camera=camera_burden(plan,boundary)
                    new_camera=camera_burden(proposal.plan,boundary)
                    output_before_rank=camera_motion_rank(before_projected,before_faces,enforce_face_limits=False,
                        added_camera_cost=old_camera.ranking_cost)
                    rank=camera_motion_rank(after_projected,projected_faces,
                        added_camera_cost=new_camera.ranking_cost)
                    row.update(proposed_output_face_metrics=projected_faces,
                        before_source_motion_face_rank=camera_motion_rank(source_before,before_faces,enforce_face_limits=False),
                        before_joint_rank=output_before_rank,proposed_joint_rank=rank,
                        before_whole_added_camera_motion=old_camera.to_dict(),
                        proposed_whole_added_camera_motion=new_camera.to_dict(),
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
                    if require_incident_edge_replay:
                        # A paired recipe changes two cells. The opposite real
                        # edges constrain selection too, not just a post-hoc QA.
                        # Internal broadcast cuts need a whole-epoch owner; do
                        # not stretch this two-cell receipt over such a cut.
                        if any(timeline.choices[i].internal_scene_cut_master_seconds
                               for i in (boundary,boundary+1)):
                            row['reason']='performer_family_internal_epoch_refresh_required'
                            attempts.append(row);continue
                        incident=[];neighbor_safe=True
                        for edge in (boundary-1,boundary+1):
                            if not 0<=edge<len(plan.transitions):
                                continue
                            if plan.segments[edge].source_index==plan.segments[edge+1].source_index:
                                # Same-native cell seams are not real cuts. The
                                # complete owner checks identical exterior pixels.
                                continue
                            _,_,old_samples=measured(plan,edge)
                            _,_,new_samples=measured(proposal.plan,edge)
                            if old_samples is None or new_samples is None or old_samples!=new_samples:
                                neighbor_safe=False
                                incident.append(dict(boundary_index=edge,reason='incident_clock_unavailable_or_changed'))
                                continue
                            old_observation=projected(plan,edge,old_samples)
                            new_observation=projected(proposal.plan,edge,new_samples,
                                required_basis=motion_evidence_basis(old_observation.motion))
                            old_motion,new_motion=old_observation.motion,new_observation.motion
                            old_faces,new_faces=old_observation.face_metrics,new_observation.face_metrics
                            safe=(old_motion is None or projected_motion_preserved(new_motion,old_motion))
                            if old_faces is not None:
                                safe=safe and new_faces is not None and all(
                                    a<=b+1e-9 for a,b in zip(new_faces,old_faces,strict=True))
                            neighbor_safe=neighbor_safe and safe
                            incident.append(dict(boundary_index=edge,preserved=safe,
                                before_motion=old_motion.to_dict() if old_motion else None,
                                after_motion=new_motion.to_dict() if new_motion else None,
                                before_faces=old_faces,after_faces=new_faces,
                                unknown_motion_not_imputed=old_motion is None))
                        row['incident_edge_replay']=incident
                        if not neighbor_safe:
                            row['reason']='performer_incident_edge_observation_lost_or_regressed'
                            attempts.append(row);continue
                    row.update(eligible=True,reason='complete_local_candidate_for_final_decoded_qa',
                        raw_new_failures=failures,proposal=proposal.to_dict())
                    eligible.append((rank,abs(offset),offset,proposal,refreshed,row))
                    attempts.append(row)
                except ValueError as error:
                    if hasattr(error,'alignment'):
                        row['dense_camera_failure']=error.alignment.to_dict()
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
