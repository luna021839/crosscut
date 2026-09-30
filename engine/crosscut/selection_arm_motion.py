"""Transient, directional arm evidence for source selection, before repair.

These are the selector's sampled source frames, NOT final-render exact-frame,
phase, identity or quality certificates. No landmarks/angles enter audit JSON.
The same RTMPose measurement is shared by all outgoing/incoming DP edges; it
must not run once for every source pair or once for every cadence profile.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import math


CONTRACT = 'sampled_focal_arm_direction_selection_v1'


@dataclass(frozen=True, slots=True)
class SelectionArmWindow:
    source_id: str
    master_times: tuple[float, float]
    native_frames: tuple[int, int]
    angles: tuple[tuple[float | None, ...], tuple[float | None, ...]] = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self,'master_times',tuple(self.master_times))
        object.__setattr__(self,'native_frames',tuple(self.native_frames))
        object.__setattr__(self,'angles',tuple(tuple(row) for row in self.angles))
        if (not isinstance(self.source_id, str) or not self.source_id
                or len(self.master_times) != 2
                or any(type(t) not in (int, float) or not math.isfinite(t)
                       for t in self.master_times)
                or not self.master_times[0] < self.master_times[1]
                or len(self.native_frames) != 2
                or any(type(n) is not int or n < 0 for n in self.native_frames)
                or not self.native_frames[0] < self.native_frames[1]):
            raise ValueError('selection arm window needs a chronological source/master clock')
        if (len(self.angles) != 2 or any(len(row) != 4 for row in self.angles)
                or any(a is not None and (type(a) not in (int, float)
                    or not math.isfinite(a) or abs(a) > math.pi)
                    for row in self.angles for a in row)
                or len(self.observed_bones) < 2):
            raise ValueError('selection arm window needs two commonly measured bones')

    @property
    def observed_bones(self):
        return frozenset(i for i in range(4) if all(row[i] is not None for row in self.angles))


@dataclass(frozen=True, slots=True)
class SelectionArmCost:
    observed_bones: tuple[int, ...]
    predicted_jump_radians: float
    velocity_jump_radians: float

    @property
    def cost(self):
        return min(1., (self.predicted_jump_radians + self.velocity_jump_radians) / math.pi)

    def to_dict(self):
        return dict(contract=CONTRACT, observed_bone_indices=list(self.observed_bones),
            predicted_jump_radians=self.predicted_jump_radians,
            velocity_jump_radians=self.velocity_jump_radians, cost=self.cost,
            sampling='selector_inset_source_frames', calibrated=False,
            exact_render_frame_evidence=False, temporal_phase_proven=False,
            cross_camera_identity_proven=False, quality_approved=False)


def selection_arm_cost(outgoing, incoming):
    """A matching still pose moving backwards is not a perfect cut.

    Missing evidence is None, never a zero observation. Native cuts and holds
    do not use this cross-source cost. Disjoint bone masks cannot be averaged
    as though both sources showed the same limbs.
    """
    if outgoing is None or incoming is None or outgoing.source_id == incoming.source_id:
        return None
    bones = tuple(sorted(outgoing.observed_bones & incoming.observed_bones))
    gap = incoming.master_times[0] - outgoing.master_times[1]
    if len(bones) < 2 or not 0 < gap <= .25:
        return None
    def wrap(value):
        return math.atan2(math.sin(value), math.cos(value))
    jumps, velocities = [], []
    for i in bones:
        a, b = (row[i] for row in outgoing.angles)
        c, d = (row[i] for row in incoming.angles)
        left = wrap(b-a) / (outgoing.master_times[1]-outgoing.master_times[0])
        right = wrap(d-c) / (incoming.master_times[1]-incoming.master_times[0])
        jumps.append(abs(wrap(c-b-.5*(left+right)*gap)))
        velocities.append(abs(right-left)*gap)
    # Keep worst pose and velocity rather than diluting one bad visible arm.
    return SelectionArmCost(bones, max(jumps), max(velocities))


class SelectionArmSampler:
    """One small pose session; source observations, not O(sources squared) pairs."""
    def __init__(self, model_path):
        from .body_pose_2d import BodyPose2DAnalyzer
        self.pose = BodyPose2DAnalyzer(model_path)
        self.cache = OrderedDict()
        self.inference_count = self.cache_hits = self.window_count = 0

    def _observe(self, sampler, seconds, descriptor):
        from .closeup_gesture import observed_arm_articulation
        from .focal_pose import face_associated_person
        if descriptor is None:
            return None
        face = descriptor.primary_face_bbox
        person = face_associated_person(face, descriptor.formation_people)
        if person is None:
            return None
        identity = sampler.frame_identity(seconds)
        key = (str(sampler.path), identity, face, person.bbox_xywh)
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        frame = sampler.frame_at(seconds)
        if frame is None:
            return None
        self.inference_count += 1
        arms = observed_arm_articulation(self.pose.describe(frame,
            bbox_xywh=person.bbox_xywh, face_bbox_xywh=face))
        angles = tuple(math.atan2(v[1], v[0]) if v is not None else None
                       for v in arms) if arms is not None else None
        result = (identity.frame_index, angles) if angles is not None else None
        self.cache[key] = result
        if len(self.cache) > 256:
            self.cache.popitem(last=False)
        return result

    def window(self, sampler, *, source_id, master_times, source_times, descriptors,
               scene_cut_times):
        if (len(master_times) != 2 or len(source_times) != 2 or len(descriptors) != 2
                or not master_times[0] < master_times[1]
                or not source_times[0] < source_times[1]
                or any(source_times[0] < cut <= source_times[1] for cut in scene_cut_times)):
            return None
        # Read both chronological frames once before any DP/cadence comparison.
        observations = tuple(self._observe(sampler, time, descriptor)
            for time, descriptor in zip(source_times, descriptors, strict=True))
        if any(row is None for row in observations):
            return None
        frames, angles = zip(*observations, strict=True)
        if frames[0] >= frames[1] or sum(all(row[i] is not None for row in angles)
                                       for i in range(4)) < 2:
            return None
        self.window_count += 1
        return SelectionArmWindow(str(source_id), tuple(master_times), tuple(frames), tuple(angles))
