"""Exact-frame arm articulation for optional close-cut timing refinement.

This is a separate RTMPose measurement, not the legacy MediaPipe 60D basis,
an identity decision, a temporal phase proof, or a quality approval. Four
directed arm bones remain usable when a tight shot crops the hips. Every arm
joint used in a cost must actually be visible and associated with the selected
focal face. Occluded bones stay unknown. Timing refinement can compare at least
two common measured bones, but cannot win by dropping a previously seen bone.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from pathlib import Path

import numpy as np

from .body_pose_2d import BodyPose2DAnalyzer, BodyPose2DEvidence
from .focal_pose import face_associated_person
from .group_focal_attention import CENTRAL_FOCAL_BASIS, GLOBAL_FOCAL_BASIS


CONTRACT = 'exact_face_owned_four_arm_bone_articulation_v1'
MASKED_CONTRACT = 'exact_face_owned_observed_arm_bones_v2'
MOTION_CONTRACT = 'exact_four_frame_observed_arm_motion_cost_v1'
_BONES = ((0, 2), (2, 4), (1, 3), (3, 5))


@dataclass(frozen=True, slots=True)
class ObservedArmMotion:
    """Continuous ranking measurements, NOT a calibrated phase certificate.

    All four frames must measure the same bones. Angles use real pixel aspect;
    velocities use output time, so source frame rate and playback rate cannot
    silently change the units. Raw joints/angles are not serialized.
    """
    native_frames: tuple[int, int, int, int]
    output_intervals_seconds: tuple[float, float, float]
    predicted_jump_radians: tuple[float | None, ...]
    velocity_jump_per_output_tick_radians: tuple[float | None, ...]

    def __post_init__(self):
        if (len(self.native_frames) != 4
                or any(type(n) is not int or n < 0 for n in self.native_frames)
                or not self.native_frames[0] < self.native_frames[1]
                or not self.native_frames[2] < self.native_frames[3]):
            raise ValueError('arm motion requires chronological non-cloned native frames')
        if (len(self.output_intervals_seconds) != 3
                or any(isinstance(t, bool) or not isinstance(t, (float, int))
                       or not math.isfinite(t) or t <= 0 for t in self.output_intervals_seconds)):
            raise ValueError('arm motion requires positive finite output intervals')
        if (len(self.predicted_jump_radians) != 4
                or len(self.velocity_jump_per_output_tick_radians) != 4
                or any((a is None) != (b is None) for a,b in zip(
                    self.predicted_jump_radians, self.velocity_jump_per_output_tick_radians))
                or sum(v is not None for v in self.predicted_jump_radians) < 2
                or any(v is not None and (isinstance(v, bool)
                       or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0)
                       for v in (*self.predicted_jump_radians,
                                 *self.velocity_jump_per_output_tick_radians))
                or any(v is not None and v > math.pi+1e-12 for v in self.predicted_jump_radians)):
            raise ValueError('arm motion requires at least two common finite bone measurements')

    @property
    def observed_bones(self):
        return frozenset(i for i,v in enumerate(self.predicted_jump_radians) if v is not None)

    @property
    def cost(self):
        # Penalize a visible pose jump AND a discontinuous movement, in the
        # same angular units. A matching still pose moving backwards cannot
        # receive a perfect score. This is not a learned quality threshold.
        return max(a+b for a,b in zip(self.predicted_jump_radians,
            self.velocity_jump_per_output_tick_radians) if a is not None)/math.pi

    def to_dict(self):
        return dict(contract=MOTION_CONTRACT, native_frames=list(self.native_frames),
            output_intervals_seconds=list(self.output_intervals_seconds),
            predicted_jump_radians=list(self.predicted_jump_radians),
            velocity_jump_per_output_tick_radians=list(self.velocity_jump_per_output_tick_radians),
            observed_bone_indices=sorted(self.observed_bones), cost=self.cost,
            ranking_only=True, calibrated=False, temporal_phase_proven=False,
            cross_camera_identity_proven=False, quality_approved=False)

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload,Mapping) or payload.get('contract') != MOTION_CONTRACT:
            raise ValueError('unknown observed arm motion contract')
        if (payload.get('ranking_only') is not True
                or any(payload.get(key) is not False for key in (
                    'calibrated','temporal_phase_proven','cross_camera_identity_proven','quality_approved'))
                or type(payload.get('cost')) not in (int,float)):
            raise ValueError('arm motion payload cannot claim calibrated quality or phase authority')
        value=cls(tuple(payload['native_frames']),tuple(payload['output_intervals_seconds']),
            tuple(payload['predicted_jump_radians']),
            tuple(payload['velocity_jump_per_output_tick_radians']))
        if value.to_dict() != dict(payload):
            raise ValueError('arm motion payload disagrees with its measurements')
        return value


@dataclass(frozen=True, slots=True)
class CloseupGestureEvidence:
    outgoing_native_frame: int
    incoming_native_frame: int
    bone_mismatches: tuple[float | None, ...]
    outgoing_focal_basis: str = GLOBAL_FOCAL_BASIS
    incoming_focal_basis: str = GLOBAL_FOCAL_BASIS
    motion: ObservedArmMotion | None = None

    def __post_init__(self):
        if any(basis not in (GLOBAL_FOCAL_BASIS, CENTRAL_FOCAL_BASIS)
               for basis in (self.outgoing_focal_basis, self.incoming_focal_basis)):
            raise ValueError('gesture focal basis must describe the measured selection')
        if (any(type(index) is not int or index < 0 for index in
                (self.outgoing_native_frame, self.incoming_native_frame))
                or len(self.bone_mismatches) != 4
                or sum(v is not None for v in self.bone_mismatches) < 2
                or any(v is not None and (isinstance(v, bool) or not isinstance(v, (float, int))
                       or not math.isfinite(v) or not 0 <= v <= 1) for v in self.bone_mismatches)):
            raise ValueError('gesture evidence requires exact frame IDs and at least two measured arm costs')
        if self.motion is not None and (
                not isinstance(self.motion, ObservedArmMotion)
                or self.motion.native_frames[1:3] != (self.outgoing_native_frame,self.incoming_native_frame)
                or not self.motion.observed_bones <= self.observed_bones):
            raise ValueError('arm motion must belong to the exact observed endpoint bones')

    @property
    def observed_bones(self) -> frozenset[int]:
        return frozenset(i for i, value in enumerate(self.bone_mismatches) if value is not None)

    def preserves_observations_of(self, before: CloseupGestureEvidence) -> bool:
        return (before.observed_bones <= self.observed_bones
                and ((self.motion is None and before.motion is None)
                     or (self.motion is not None and before.motion is not None
                         and before.motion.observed_bones <= self.motion.observed_bones
                         and self.motion.output_intervals_seconds == before.motion.output_intervals_seconds)))

    def improves_over(self, before: CloseupGestureEvidence) -> bool:
        # Include newly observed bones in the candidate's maximum as well.
        # Filling a gap with a badly mismatched hand is not an improvement.
        return (self.preserves_observations_of(before) and self.preserves_pose_continuity_of(before)
                and self.ranking_cost < before.ranking_cost-1e-6)

    def preserves_pose_continuity_of(self, before: CloseupGestureEvidence) -> bool:
        if self.motion is None and before.motion is None:
            return True
        if self.motion is None or before.motion is None:
            return False
        # A lower velocity penalty must not buy a larger visible pose jump.
        # Compare motion-predicted poses (not identical frozen endpoints), so
        # natural one-frame progress can still beat a backwards movement.
        return (max(v for v in self.motion.predicted_jump_radians if v is not None)
                <= max(v for v in before.motion.predicted_jump_radians if v is not None)+1e-9)

    @property
    def ranking_cost(self) -> float:
        return self.motion.cost if self.motion is not None else self.mismatch

    @property
    def mismatch(self) -> float:
        # The worst visible arm matters: averaging it with three unchanged
        # bones can disguise exactly the one hand that breaks across the cut.
        return max(value for value in self.bone_mismatches if value is not None)

    def to_dict(self):
        result = dict(contract=CONTRACT if len(self.observed_bones) == 4 else MASKED_CONTRACT,
            feature_basis='four_directed_screen_arm_bones',
            outgoing_native_frame=self.outgoing_native_frame,
            incoming_native_frame=self.incoming_native_frame,
            bone_count=len(self.observed_bones), bone_mismatches=list(self.bone_mismatches),
            observed_bone_indices=sorted(self.observed_bones),
            unobserved_bone_count=4-len(self.observed_bones),
            observation_policy='at_least_two_exact_bones_no_loss_against_incumbent',
            mismatch=self.mismatch, aggregation='maximum_bone_angular_mismatch',
            outgoing_focal_basis=self.outgoing_focal_basis,
            incoming_focal_basis=self.incoming_focal_basis,
            cross_camera_identity_proven=False,
            temporal_phase_proven=False, quality_approved=False)
        if self.motion is not None:
            result['motion'] = self.motion.to_dict()
        return result

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload,Mapping):
            raise ValueError('gesture payload must contain scalar evidence')
        return cls(payload['outgoing_native_frame'],payload['incoming_native_frame'],
            tuple(payload['bone_mismatches']),payload['outgoing_focal_basis'],
            payload['incoming_focal_basis'],
            ObservedArmMotion.from_dict(payload['motion']) if 'motion' in payload else None)


def arm_articulation(observation: BodyPose2DEvidence | None):
    """Four unit vectors in real pixel geometry; no offscreen joint guessing."""
    if (observation is None or observation.reason != 'observed'
            or observation.face_anchor_consistent is not True
            or len(observation.upper_body_xyc) != 8):
        return None
    points = np.asarray(observation.upper_body_xyc[:6], dtype=np.float64)
    dimensions = np.asarray(observation.frame_size_wh, dtype=np.float64)
    if (points.shape != (6, 3) or dimensions.shape != (2,)
            or not np.isfinite(points).all() or not np.isfinite(dimensions).all()
            or np.any(dimensions <= 0) or np.any(points[:, 2] < .3)
            or np.any(points[:, 2] > 1) or np.any(points[:, :2] < 0)
            or np.any(points[:, :2] >= 1)):
        return None
    pixels = points[:, :2] * dimensions
    shoulder = float(np.linalg.norm(pixels[0]-pixels[1]))
    if shoulder < 2.:
        return None
    result = []
    for start, end in _BONES:
        direction = pixels[end]-pixels[start]
        length = float(np.linalg.norm(direction))
        if length < max(1., shoulder*.03):
            return None
        result.append(tuple(float(v) for v in direction/length))
    return tuple(result)


def observed_arm_articulation(observation: BodyPose2DEvidence | None):
    """Keep per-bone visibility, not an all-or-nothing whole-arm declaration.

    Detection/ownership/face-anchor requirements are unchanged. No low-score,
    offscreen, nonfinite or collapsed joint is imputed or relabelled observed.
    """
    full = arm_articulation(observation)
    if full is not None:
        return full
    if (observation is None or observation.reason != 'observed'
            or observation.face_anchor_consistent is not True
            or len(observation.upper_body_xyc) != 8):
        return None
    points = np.asarray(observation.upper_body_xyc[:6], dtype=np.float64)
    dimensions = np.asarray(observation.frame_size_wh, dtype=np.float64)
    if (points.shape != (6, 3) or dimensions.shape != (2,)
            or not np.isfinite(dimensions).all() or np.any(dimensions <= 0)):
        return None
    valid = (np.isfinite(points).all(axis=1) & (points[:, 2] >= .3) & (points[:, 2] <= 1)
             & (points[:, :2] >= 0).all(axis=1) & (points[:, :2] < 1).all(axis=1))
    pixels = points[:, :2] * dimensions
    result = []
    for start, end in _BONES:
        direction = pixels[end]-pixels[start]
        length = float(np.linalg.norm(direction)) if valid[start] and valid[end] else 0.
        result.append(tuple(float(v) for v in direction/length) if length >= 2. else None)
    return tuple(result) if sum(value is not None for value in result) >= 2 else None


def compare_arm_articulation(left, right, *, outgoing_native_frame, incoming_native_frame,
                            outgoing_focal_basis=GLOBAL_FOCAL_BASIS,
                            incoming_focal_basis=GLOBAL_FOCAL_BASIS, allow_partial=False):
    if type(allow_partial) is not bool:
        raise ValueError('partial arm observation policy must be boolean')
    observe = observed_arm_articulation if allow_partial else arm_articulation
    a, b = observe(left), observe(right)
    if a is None or b is None:
        return None
    if any(type(index) is not int or index < 0
           for index in (outgoing_native_frame, incoming_native_frame)):
        raise ValueError('gesture comparison requires exact native frame IDs')
    values = tuple(min(1., max(0., (1.-sum(x*y for x,y in zip(v,w)))/2.))
                   if v is not None and w is not None else None
                   for v,w in zip(a,b))
    if sum(value is not None for value in values) < 2:
        return None
    return CloseupGestureEvidence(outgoing_native_frame, incoming_native_frame, values,
                                  outgoing_focal_basis, incoming_focal_basis)


def measure_arm_motion(observations, *, native_frames, output_intervals_seconds):
    """Measure signed short-window motion; missing joints stay unknown."""
    if len(observations) != 4:
        raise ValueError('arm motion requires four chronological observations')
    # Validate clocks even when a pose is missing. Never treat a clone or a
    # reversed source window as an observation of static movement.
    ObservedArmMotion(tuple(native_frames),tuple(output_intervals_seconds),
                      (0.,0.,None,None),(0.,0.,None,None))
    bones=tuple(observed_arm_articulation(observation) for observation in observations)
    if any(value is None for value in bones):
        return None
    mask=tuple(i for i in range(4) if all(row[i] is not None for row in bones))
    if len(mask) < 2:
        return None
    before_dt,cut_dt,after_dt=output_intervals_seconds
    jumps,velocities=[None]*4,[None]*4
    def wrap(angle):
        return math.atan2(math.sin(angle),math.cos(angle))
    for i in mask:
        a,b,c,d=(math.atan2(row[i][1],row[i][0]) for row in bones)
        outgoing=wrap(b-a)/before_dt
        incoming=wrap(d-c)/after_dt
        jumps[i]=abs(wrap(c-b-(outgoing+incoming)*.5*cut_dt))
        velocities[i]=abs(incoming-outgoing)*cut_dt
    return ObservedArmMotion(tuple(native_frames),tuple(output_intervals_seconds),
                             tuple(jumps),tuple(velocities))


class CloseupGestureProvider:
    """Lazy, source-checked native reads and a bounded run-local pose cache."""
    def __init__(self, sources, model_path: Path):
        # Keep this optional dependency outside the persistent descriptor path.
        from .boundary_subject_alignment import _CaptureProvider
        self.sources = tuple(Path(p) for p in sources)
        self.identities = tuple(self._identity(p) for p in self.sources)
        self.capture = _CaptureProvider(self.sources)
        self.pose = BodyPose2DAnalyzer(model_path)
        self.cache = OrderedDict()
        self.head_cache = OrderedDict()

    @staticmethod
    def _identity(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def observe(self, source_index, seconds, clock, descriptor):
        if self._identity(self.sources[source_index]) != self.identities[source_index]:
            raise OSError('gesture source changed during analysis')
        if clock is None or not math.isfinite(seconds) or seconds < 0:
            return None
        index = int(seconds*float(clock.frame_rate)+.5)
        if not 0 <= index < clock.frame_count:
            return None
        face = descriptor.primary_face_bbox
        body = face_associated_person(face, descriptor.formation_people)
        if face is not None and body is None:
            return None
        # Cache by exact input evidence, including the unknown-primary case.
        # A central focal supplement is local to this optional arm measurement;
        # it never re-labels a persistent descriptor or proves member identity.
        key = source_index, index, face, tuple(descriptor.formation_people)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        frame = self.capture.frame_at_native_index(source_index, index, clock)
        if frame is None:
            return None
        from .visual_score import _working_frame
        frame = _working_frame(frame, maximum_width=960)
        from .group_focal_attention import resolve_group_focal_subject
        focal = resolve_group_focal_subject(frame, descriptor)
        if focal is None:
            self.cache[key] = None
            if len(self.cache) > 256:
                self.cache.popitem(last=False)
            return None
        face, body, basis = focal.bbox_xywh, focal.person, focal.basis
        measured = self.pose.describe(frame, bbox_xywh=body.bbox_xywh, face_bbox_xywh=face)
        result = index, measured, basis
        self.cache[key] = result
        if len(self.cache) > 256:
            self.cache.popitem(last=False)
        return result

    def compare(self, outgoing, incoming):
        # Each tuple is (source, exact request time, verified clock, descriptor).
        left, right = self.observe(*outgoing), self.observe(*incoming)
        if left is None or right is None:
            return None
        return compare_arm_articulation(left[1], right[1],
            outgoing_native_frame=left[0], incoming_native_frame=right[0],
            outgoing_focal_basis=left[2], incoming_focal_basis=right[2], allow_partial=True)

    def compare_motion(self, outgoing_preceding, outgoing, incoming, incoming_following,
                       *, output_fps, flank_frames=3):
        """Exact four-frame ranking, never a substitute for identity/phase QA."""
        if (isinstance(output_fps,bool) or not isinstance(output_fps,(float,int))
                or not math.isfinite(output_fps) or output_fps <= 0
                or type(flank_frames) is not int or flank_frames < 1):
            raise ValueError('arm motion requires a valid output clock')
        if (outgoing_preceding[0] != outgoing[0] or incoming[0] != incoming_following[0]
                or outgoing[0] == incoming[0]):
            raise ValueError('arm motion windows must stay within their two source runs')
        frames=tuple(self.observe(*sample) for sample in
            (outgoing_preceding,outgoing,incoming,incoming_following))
        if any(frame is None for frame in frames):
            return None
        ids=tuple(frame[0] for frame in frames)
        if ids[0] >= ids[1] or ids[2] >= ids[3]:
            return None
        evidence=compare_arm_articulation(frames[1][1],frames[2][1],
            outgoing_native_frame=ids[1],incoming_native_frame=ids[2],
            outgoing_focal_basis=frames[1][2],incoming_focal_basis=frames[2][2],allow_partial=True)
        if evidence is None:
            return None
        motion=measure_arm_motion(tuple(frame[1] for frame in frames),native_frames=ids,
            output_intervals_seconds=(flank_frames/output_fps,1./output_fps,flank_frames/output_fps))
        return replace(evidence,motion=motion) if motion is not None else None

    def compare_head_motion(self, outgoing_preceding, outgoing, incoming, incoming_following,
                            *, output_fps, flank_frames=3):
        """A separate pixel-linked head basis, only for arm-unmeasured callers.

        Same-source paths must link the actual four native frames. Known
        primary faces are preserved; missing five-point geometry gets one
        bounded global reacquisition, never a central-face substitute.
        """
        from .observed_head_motion import head_motion_signature,measure_head_motion,HeadMotionEvidence
        from .roll_aware_face_geometry import detect_roll_aware_faces
        from .visual_score import _dominant_face,_normalized_face_metadata,_working_frame
        from .group_focal_attention import focal_landmarks_track
        from .boundary_subject_alignment import _optical_flow_box
        from .cut_anchored_containment import _iou
        import cv2
        if (type(output_fps) not in (float,int) or not math.isfinite(output_fps) or output_fps<=0
                or type(flank_frames) is not int or flank_frames<1):
            raise ValueError('head motion requires a valid output clock')
        requests=(outgoing_preceding,outgoing,incoming,incoming_following)
        if (outgoing_preceding[0]!=outgoing[0] or incoming[0]!=incoming_following[0]
                or outgoing[0]==incoming[0]):
            raise ValueError('head paths must stay within two different source runs')
        observations=[]
        for source,seconds,clock,descriptor in requests:
            if self._identity(self.sources[source])!=self.identities[source]:
                raise OSError('head source changed during analysis')
            if clock is None or type(seconds) not in (float,int) or not math.isfinite(seconds) or seconds<0:
                return None
            index=int(seconds*float(clock.frame_rate)+.5)
            if not 0<=index<clock.frame_count:return None
            frame=self.capture.frame_at_native_index(source,index,clock)
            if frame is None:return None
            frame=cv2.resize(_working_frame(frame,maximum_width=960),(854,480))
            known=descriptor.primary_face_bbox
            geometry=getattr(descriptor,'primary_face_geometry',None)
            points=getattr(descriptor,'primary_face_landmarks_xy',None)
            key=(source,index,known,geometry,points)
            if key in self.head_cache:
                signature,box,points=self.head_cache[key]
                self.head_cache.move_to_end(key)
            else:
                signature=head_motion_signature(geometry,points)
                box=known
                if signature is None:
                    faces=detect_roll_aware_faces(frame,angles=(0.,))
                    primary=_dominant_face(faces)
                    if primary is None:
                        primary=_dominant_face(detect_roll_aware_faces(frame))
                    measured_box,measured_points,_,_,measured_geometry=_normalized_face_metadata(primary,
                        frame_width=854,frame_height=480)
                    if measured_box is not None and (known is None or _iou(known,measured_box)>=.60):
                        box,points=measured_box,measured_points
                        signature=head_motion_signature(measured_geometry,points)
                self.head_cache[key]=(signature,box,points)
                if len(self.head_cache)>256:self.head_cache.popitem(last=False)
            if signature is None or box is None or points is None:return None
            observations.append((index,signature,box,points,frame))
        ids=tuple(row[0] for row in observations)
        if ids[0]>=ids[1] or ids[2]>=ids[3]:return None
        for a,b in (observations[:2],observations[2:]):
            if focal_landmarks_track(a[4],b[4],a[3],b[3]):continue
            tracked=_optical_flow_box(a[4],b[4],a[2])
            if tracked is None or _iou(tracked,b[2])<.60:return None
        motion=measure_head_motion(tuple(row[1] for row in observations),native_frames=ids,
            output_intervals_seconds=(flank_frames/output_fps,1/output_fps,flank_frames/output_fps),
            same_source_face_paths_pixel_verified=True)
        if motion is None:return None
        guard=self.compare(outgoing,incoming)
        # The head path follows the GLOBAL dominant face. A legacy local arm
        # fallback may have selected a different central person, even on the
        # same frame IDs; it cannot become this head's arm guard.
        if guard is not None and (guard.outgoing_focal_basis!=GLOBAL_FOCAL_BASIS
                or guard.incoming_focal_basis!=GLOBAL_FOCAL_BASIS):
            guard=None
        return HeadMotionEvidence(motion,guard)

    def close(self):
        self.capture.close()


def boundary_motion_priorities(plan, boundary_indices, *, model_path,
                               scene_cut_times, feature_cache_directory=None,
                               feature_derived_cache_directory=None,
                               require_persistent_feature_cache=False):
    """Triage current cuts once; retain the existing bounded audition count.

    Previously the largest faces always received the limited exact auditions,
    even when their motion already matched and another medium shot jumped.
    These scalar costs only order that work. They do not authorize any edit,
    exclude a source, or classify unobserved motion as a successful match.
    """
    from .render_boundary_clock import hard_cut_boundary_sample_clocks
    from .render_frame_schedule import probe_render_frame_clock
    from .visual_score import _VideoSampler
    indices=tuple(boundary_indices)
    if (len(indices) != len(set(indices))
            or any(type(i) is not int or not 0 <= i < len(plan.segments)-1 for i in indices)
            or len(scene_cut_times) != len(plan.sources)):
        raise ValueError('motion priority source/boundary domain is invalid')
    used={plan.segments[i+side].source_index for i in indices for side in (0,1)}
    clocks={source:probe_render_frame_clock(plan.sources[source]) for source in used}
    samplers={}
    provider=CloseupGestureProvider(plan.sources,Path(model_path))
    priorities={}
    try:
        for index in indices:
            if any(clocks[plan.segments[index+side].source_index] is None for side in (0,1)):
                continue
            try:
                samples=hard_cut_boundary_sample_clocks(plan,clocks,boundary_index=index)
            except ValueError:
                continue
            if any(any(a.native_presentation_seconds < float(t) <= b.native_presentation_seconds
                       for t in scene_cut_times[a.source_index])
                   for a,b in (samples[:2],samples[2:])):
                continue
            requests=[]
            for sample in samples:
                source=sample.source_index
                if source not in samplers:
                    samplers[source]=_VideoSampler(plan.sources[source],
                        feature_cache_directory=feature_cache_directory,
                        feature_derived_cache_directory=feature_derived_cache_directory,
                        require_persistent_feature_cache=require_persistent_feature_cache)
                descriptor=samplers[source].feature_descriptor_at(sample.source_request_seconds,
                    include_subject_geometry=True,include_formation_pose=True)
                if descriptor is None:
                    break
                requests.append((source,sample.source_request_seconds,clocks[source],descriptor))
            if len(requests) != 4:
                continue
            evidence=provider.compare_motion(*requests,output_fps=plan.fps)
            if evidence is not None:
                priorities[index]=evidence.ranking_cost
    finally:
        provider.close()
        for sampler in samplers.values():sampler.close()
    return priorities
