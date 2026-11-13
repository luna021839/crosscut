"""Four-frame, source-local five-point head motion for arm-cropped close cuts.

This is a distinct, uncalibrated ranking basis, not an arm observation, 3D head
pose, temporal-phase certificate or cross-camera identity decision. The caller
must prove each same-source two-frame face path from actual pixels. Missing
landmarks or paths remain unknown. No face vector or image is serialized.
"""
from dataclasses import dataclass
import math

import numpy as np

from .closeup_gesture import CloseupGestureEvidence
from .group_focal_attention import GLOBAL_FOCAL_BASIS


HEAD_MOTION_BASIS = 'exact_four_frame_source_local_five_point_head_motion_v1'
HEAD_DIMENSIONS = ('roll_pi', 'eye_aligned_nose_x', 'eye_aligned_nose_y',
    'eye_aligned_mouth_y', 'eye_aligned_mouth_asymmetry')


def head_motion_signature(geometry, landmarks):
    """Reuse measured YuNet geometry, with no body/central-face substitution."""
    if geometry is None or landmarks is None:
        return None
    if any(isinstance(v,bool) for v in geometry) or any(isinstance(v,bool) for p in landmarks for v in p):
        return None
    values = np.asarray(geometry, dtype=float)
    points = np.asarray(landmarks, dtype=float)
    if (values.shape != (10,) or points.shape != (5,2)
            or not np.isfinite(values).all() or not np.isfinite(points).all()
            or np.any(points < 0) or np.any(points >= 1)
            or values[4] < 4/480 or abs(values[5]) > math.radians(75)):
        return None
    # Near-vertical sorted-eye direction is not continuous. Extreme/offscreen
    # profiles stay unmeasured, rather than inventing a front-facing head.
    signature = (values[5]/math.pi, *values[6:10])
    if any(abs(v)>3 for v in signature):
        return None
    return tuple(float(v) for v in signature)


@dataclass(frozen=True, slots=True)
class ObservedHeadMotion:
    native_frames: tuple[int,int,int,int]
    output_intervals_seconds: tuple[float,float,float]
    predicted_pose_cost: float
    velocity_jump_per_output_tick: float

    def __post_init__(self):
        if (len(self.native_frames)!=4 or any(type(n) is not int or n<0 for n in self.native_frames)
                or self.native_frames[0]>=self.native_frames[1]
                or self.native_frames[2]>=self.native_frames[3]):
            raise ValueError('head motion needs chronological non-cloned native frames')
        if (len(self.output_intervals_seconds)!=3 or any(type(t) not in (int,float)
                or not math.isfinite(t) or t<=0 for t in self.output_intervals_seconds)):
            raise ValueError('head motion needs positive finite output intervals')
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0
                for v in (self.predicted_pose_cost,self.velocity_jump_per_output_tick)):
            raise ValueError('head motion costs must be observed finite nonnegative scalars')

    @property
    def cost(self):
        return self.predicted_pose_cost+self.velocity_jump_per_output_tick

    def to_dict(self):
        return dict(contract=HEAD_MOTION_BASIS,native_frames=list(self.native_frames),
            output_intervals_seconds=list(self.output_intervals_seconds),
            measured_dimensions=list(HEAD_DIMENSIONS),predicted_pose_cost=self.predicted_pose_cost,
            velocity_jump_per_output_tick=self.velocity_jump_per_output_tick,cost=self.cost,
            same_source_face_paths_pixel_verified=True,arm_motion_observed=False,
            ranking_only=True,calibrated=False,temporal_phase_proven=False,
            cross_camera_identity_proven=False,quality_approved=False)


@dataclass(frozen=True, slots=True)
class HeadMotionEvidence:
    motion: ObservedHeadMotion
    endpoint_arm_guard: CloseupGestureEvidence | None = None

    def __post_init__(self):
        if not isinstance(self.motion,ObservedHeadMotion):
            raise TypeError('head evidence needs its separate measured basis')
        guard=self.endpoint_arm_guard
        if guard is not None and (not isinstance(guard,CloseupGestureEvidence)
                or (guard.outgoing_native_frame,guard.incoming_native_frame)!=self.motion.native_frames[1:3]
                or guard.outgoing_focal_basis!=GLOBAL_FOCAL_BASIS
                or guard.incoming_focal_basis!=GLOBAL_FOCAL_BASIS):
            raise ValueError('head arm guard must own the exact globally selected endpoint clock')

    @property
    def ranking_cost(self):
        return self.motion.cost

    @property
    def observed_bones(self):
        # Static endpoint bones are not relabelled four-frame arm motion.
        return self.endpoint_arm_guard.observed_bones if self.endpoint_arm_guard else frozenset()

    def preserves_motion_of(self,before):
        if (not isinstance(before,HeadMotionEvidence)
                or self.motion.output_intervals_seconds!=before.motion.output_intervals_seconds
                or self.motion.predicted_pose_cost>before.motion.predicted_pose_cost+1e-9
                or self.ranking_cost>before.ranking_cost+1e-9):
            return False
        if before.endpoint_arm_guard is not None:
            guard=self.endpoint_arm_guard
            if guard is None or not before.observed_bones<=self.observed_bones:
                return False
            # A head improvement may not lose or worsen any measured endpoint
            # arm bone. Missing arm motion still remains explicitly unknown.
            if any(guard.bone_mismatches[i]>before.endpoint_arm_guard.bone_mismatches[i]+1e-9
                    for i in before.observed_bones):
                return False
        return True

    def improves_over(self,before):
        return self.preserves_motion_of(before) and self.ranking_cost<before.ranking_cost-1e-6

    def to_dict(self):
        return dict(contract=HEAD_MOTION_BASIS,motion=self.motion.to_dict(),
            endpoint_arm_guard=self.endpoint_arm_guard.to_dict() if self.endpoint_arm_guard else None,
            arm_motion_unknown_not_imputed=True,quality_approved=False)


def measure_head_motion(signatures,*,native_frames,output_intervals_seconds,
        same_source_face_paths_pixel_verified):
    """Predict natural one-tick progress instead of matching frozen faces."""
    # Clock mistakes remain errors even when geometry/path evidence is missing.
    ObservedHeadMotion(tuple(native_frames),tuple(output_intervals_seconds),0.,0.)
    if type(same_source_face_paths_pixel_verified) is not bool:
        raise ValueError('head paths require an explicit pixel-evidence boolean')
    if len(signatures)!=4:
        raise ValueError('head motion requires four observed signatures')
    if not same_source_face_paths_pixel_verified or any(s is None for s in signatures):
        return None
    values=np.asarray(signatures,dtype=float)
    if values.shape!=(4,5) or not np.isfinite(values).all() or np.any(np.abs(values)>3):
        return None
    before_dt,cut_dt,after_dt=output_intervals_seconds
    outgoing=(values[1]-values[0])/before_dt
    incoming=(values[3]-values[2])/after_dt
    predicted=values[2]-values[1]-(outgoing+incoming)*.5*cut_dt
    velocity=(incoming-outgoing)*cut_dt
    # All five dimensions must be observed. A max cost cannot hide the one
    # visible head component that reverses across the cut by averaging it away.
    return ObservedHeadMotion(tuple(native_frames),tuple(output_intervals_seconds),
        float(np.max(np.abs(predicted))),float(np.max(np.abs(velocity))))
