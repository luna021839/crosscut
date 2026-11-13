"""Exact source-focal screen joints for joint camera registration.

This is observed pose geometry, not a learned camera or a quality approval.
No imputation, anonymous formation pose, or midpoint substitution is allowed.
"""
from dataclasses import dataclass
import math

import numpy as np

from .focal_selector_geometry import FocalSelectorGeometry


@dataclass(frozen=True)
class OverlapPoseObservation:
    source_index: int
    master_seconds: float
    source_seconds: float
    points_xyc: tuple[tuple[float, float, float], ...]

    def __post_init__(self):
        if (isinstance(self.source_index, bool) or not isinstance(self.source_index, int)
                or self.source_index < 0 or len(self.points_xyc) != 8):
            raise ValueError('overlap pose source or joint count is invalid')
        values = (self.master_seconds, self.source_seconds,
                  *(v for point in self.points_xyc for v in point))
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in values):
            raise ValueError('overlap pose must have finite measured values')
        if min(self.master_seconds, self.source_seconds) < 0 or any(
            len(point) != 3 or not 0 <= point[2] <= 1 for point in self.points_xyc
        ):
            raise ValueError('overlap pose clock or confidence is invalid')


def source_focal_overlap_pose(value, *, source_index, expected_geometry_source_index,
                              master_seconds, source_seconds, frame_geometry):
    """Map verified source-screen joints into the renderer's aspect-fill domain."""
    if value is None:
        return None
    if not isinstance(value, FocalSelectorGeometry):
        raise TypeError('overlap pose needs source-focal geometry')
    if (value.source_index != expected_geometry_source_index
            or abs(value.source_seconds-source_seconds) > 1e-9):
        raise ValueError('overlap pose does not bind this source frame')
    if value.person is None or value.pose_backend != 'rtmpose_body7' or not value.pose_screen_xyc:
        return None
    x, y, width, height = frame_geometry.fill_crop_xywh
    points = tuple(((px-x)/width, (py-y)/height, confidence)
                   for px, py, confidence in value.pose_screen_xyc)
    return OverlapPoseObservation(source_index, master_seconds, source_seconds, points)


def paired_overlap_pose_arrays(observations, face_sequences):
    """Keep the same >=4 visible joint IDs across both sides and every frame."""
    if observations is None:
        return None, {'status': 'not_supplied', 'common_joint_indices': []}
    if len(observations) != 2 or any(len(seq) != len(face_sequences[0]) for seq in observations):
        raise ValueError('overlap pose windows do not cover the face clocks')
    for poses, faces in zip(observations, face_sequences):
        for pose, face in zip(poses, faces):
            if pose is None:
                continue
            if not isinstance(pose, OverlapPoseObservation):
                raise TypeError('overlap pose must be typed observations')
            if (pose.source_index != face.source_index
                    or abs(pose.source_seconds-face.source_seconds) > 1e-9
                    or abs(pose.master_seconds-face.master_seconds) > 1e-9):
                raise ValueError('overlap pose and face refer to different frames')
    if any(pose is None for seq in observations for pose in seq):
        return None, {'status': 'exact_pose_missing', 'common_joint_indices': []}
    values = np.array([[pose.points_xyc for pose in seq] for seq in observations])
    visible = ((values[:, :, :, 2] >= .20)
               & (values[:, :, :, :2] >= 0).all(axis=-1)
               & (values[:, :, :, :2] <= 1).all(axis=-1)).all(axis=(0, 1))
    indices = np.flatnonzero(visible).tolist()
    if len(indices) < 4:
        return None, {'status': 'insufficient_common_joints', 'common_joint_indices': indices}
    return values[:, :, visible, :2], {
        'status': 'exact_focal_screen_joints', 'common_joint_indices': indices,
        'source_frame_count': int(values.shape[0]*values.shape[1]),
        'missing_joints_interpolated': False,
    }


def pose_registration_metrics(points, affine):
    projected = points*affine[:, :, None, :1]+affine[:, :, None, 1:]
    distances = np.linalg.norm(projected[0]-projected[1], axis=-1)
    return {
        'joint_mean_distance': float(distances.mean()),
        'joint_rms_distance': float(np.sqrt(np.mean(distances**2))),
        'joint_maximum_distance': float(distances.max()),
    }
