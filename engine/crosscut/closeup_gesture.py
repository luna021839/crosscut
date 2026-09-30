"""Exact-frame arm articulation for optional close-cut timing refinement.

This is a separate RTMPose measurement, not the legacy MediaPipe 60D basis,
an identity decision, a temporal phase proof, or a quality approval. Four
directed arm bones remain usable when a tight shot crops the hips. Every arm
joint must actually be visible and associated with the selected focal face.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .body_pose_2d import BodyPose2DAnalyzer, BodyPose2DEvidence
from .focal_pose import face_associated_person
from .group_focal_attention import CENTRAL_FOCAL_BASIS, GLOBAL_FOCAL_BASIS


CONTRACT = 'exact_face_owned_four_arm_bone_articulation_v1'
_BONES = ((0, 2), (2, 4), (1, 3), (3, 5))


@dataclass(frozen=True, slots=True)
class CloseupGestureEvidence:
    outgoing_native_frame: int
    incoming_native_frame: int
    bone_mismatches: tuple[float, ...]
    outgoing_focal_basis: str = GLOBAL_FOCAL_BASIS
    incoming_focal_basis: str = GLOBAL_FOCAL_BASIS

    def __post_init__(self):
        if any(basis not in (GLOBAL_FOCAL_BASIS, CENTRAL_FOCAL_BASIS)
               for basis in (self.outgoing_focal_basis, self.incoming_focal_basis)):
            raise ValueError('gesture focal basis must describe the measured selection')
        if (any(type(index) is not int or index < 0 for index in
                (self.outgoing_native_frame, self.incoming_native_frame))
                or len(self.bone_mismatches) != 4
                or any(isinstance(v, bool) or not isinstance(v, (float, int))
                       or not math.isfinite(v) or not 0 <= v <= 1 for v in self.bone_mismatches)):
            raise ValueError('gesture evidence requires exact frame IDs and four finite arm costs')

    @property
    def mismatch(self) -> float:
        # The worst visible arm matters: averaging it with three unchanged
        # bones can disguise exactly the one hand that breaks across the cut.
        return max(self.bone_mismatches)

    def to_dict(self):
        return dict(contract=CONTRACT, feature_basis='four_directed_screen_arm_bones',
            outgoing_native_frame=self.outgoing_native_frame,
            incoming_native_frame=self.incoming_native_frame,
            bone_count=4, bone_mismatches=list(self.bone_mismatches),
            mismatch=self.mismatch, aggregation='maximum_bone_angular_mismatch',
            outgoing_focal_basis=self.outgoing_focal_basis,
            incoming_focal_basis=self.incoming_focal_basis,
            cross_camera_identity_proven=False,
            temporal_phase_proven=False, quality_approved=False)


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


def compare_arm_articulation(left, right, *, outgoing_native_frame, incoming_native_frame,
                            outgoing_focal_basis=GLOBAL_FOCAL_BASIS,
                            incoming_focal_basis=GLOBAL_FOCAL_BASIS):
    a, b = arm_articulation(left), arm_articulation(right)
    if a is None or b is None:
        return None
    if any(type(index) is not int or index < 0
           for index in (outgoing_native_frame, incoming_native_frame)):
        raise ValueError('gesture comparison requires exact native frame IDs')
    values = tuple(min(1., max(0., (1.-sum(x*y for x,y in zip(v,w)))/2.))
                   for v,w in zip(a,b))
    return CloseupGestureEvidence(outgoing_native_frame, incoming_native_frame, values,
                                  outgoing_focal_basis, incoming_focal_basis)


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
        basis = GLOBAL_FOCAL_BASIS
        if face is None:
            from .group_focal_attention import observe_central_focal_face
            focal = observe_central_focal_face(frame, descriptor.formation_people)
            if focal is not None:
                face, body, basis = focal.bbox_xywh, focal.person, CENTRAL_FOCAL_BASIS
        if body is None:
            self.cache[key] = None
            if len(self.cache) > 256:
                self.cache.popitem(last=False)
            return None
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
            outgoing_focal_basis=left[2], incoming_focal_basis=right[2])

    def close(self):
        self.capture.close()
