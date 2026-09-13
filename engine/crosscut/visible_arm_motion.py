"""Visible, face-owned arm motion without requiring off-screen hips.

This is a separate six-joint, shoulder-width-normalized measurement basis.
It is not a replacement for whole-body descriptors, identity authorization,
or a calibrated accept/reject policy. Missing joints are never interpolated;
confidence values select observations but are not pose coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

from .body_pose_2d import BodyPose2DEvidence


ARM_MOTION_BASIS = 'face_owned_visible_arms_shoulder_width_v1'


@dataclass(frozen=True, slots=True)
class VisibleArmObservation:
    # Process-local only; public evidence is emitted by compare_arm_motion.
    coordinates: tuple[tuple[float, float], ...]
    visible: tuple[bool, ...]


def visible_arm_observation(
    evidence: BodyPose2DEvidence,
    *,
    minimum_joint_confidence: float,
) -> VisibleArmObservation | None:
    """Normalize observed shoulders/arms; never use or require the hip joints."""
    if (isinstance(minimum_joint_confidence, bool)
            or not isinstance(minimum_joint_confidence, Real)
            or not math.isfinite(minimum_joint_confidence)
            or not 0 < minimum_joint_confidence <= 1):
        raise ValueError('joint confidence must be finite and in (0, 1]')
    if not isinstance(evidence, BodyPose2DEvidence):
        raise TypeError('visible arms require measured BodyPose2DEvidence')
    if (evidence.reason != 'observed' or evidence.face_anchor_consistent is not True
            or len(evidence.upper_body_xyc) != 8):
        return None
    size = evidence.frame_size_wh
    if (len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int)
                             or v <= 0 for v in size)):
        return None
    points = evidence.upper_body_xyc[:6]
    if any(len(p) != 3 or any(isinstance(v, bool) or not isinstance(v, Real)
           or not math.isfinite(v) for v in p) or not 0 <= p[2] <= 1 for p in points):
        return None
    visible = tuple(bool(c >= minimum_joint_confidence and 0 <= x < 1 and 0 <= y < 1)
                    for x, y, c in points)
    if not all(visible[:2]):
        return None
    pixels = tuple((p[0] * size[0], p[1] * size[1]) for p in points)
    center = tuple((pixels[0][i] + pixels[1][i]) / 2 for i in (0, 1))
    scale = math.dist(pixels[0], pixels[1])
    if scale < 4:
        return None
    # Missing coordinates remain unused: the mask must intersect across every
    # sample before comparison. No confidence multiplication or clipping.
    coordinates = tuple(tuple((p[i] - center[i]) / scale for i in (0, 1)) for p in pixels)
    return VisibleArmObservation(coordinates, visible)


def compare_arm_motion(
    outgoing: Sequence[tuple[float, VisibleArmObservation | None]],
    incoming: Sequence[tuple[float, VisibleArmObservation | None]],
) -> dict[str, object]:
    """Return scalar endpoint distance and signed velocity on common arm joints.

    Times are source-local seconds. Both windows are chronological; unequal
    frame rates are supported. Ordinary least squares estimates each joint's
    velocity over the observed window. No speed, similarity or quality cutoff
    is asserted by this measurement API.
    """
    result: dict[str, object] = dict(
        basis=ARM_MOTION_BASIS, available=False, reason='missing_observation',
        outgoing_samples=len(outgoing), incoming_samples=len(incoming),
        common_arm_joint_count=0, endpoint_rms=None, velocity_cosine=None,
        outgoing_speed=None, incoming_speed=None,
        quality_calibrated=False, final_media_authority=False,
    )
    for window in (outgoing, incoming):
        if len(window) < 2:
            result['reason'] = 'too_few_temporal_samples'
            return result
        times = [sample[0] for sample in window]
        if (any(isinstance(t, bool) or not isinstance(t, Real) or not math.isfinite(t)
                for t in times) or any(b <= a for a, b in zip(times, times[1:]))):
            raise ValueError('arm-motion times must be finite and strictly increasing')
        for _, observation in window:
            if observation is None:
                return result
            if (not isinstance(observation, VisibleArmObservation)
                    or len(observation.coordinates) != 6 or len(observation.visible) != 6
                    or any(type(v) is not bool for v in observation.visible)
                    or any(len(p) != 2 or any(isinstance(v, bool) or not isinstance(v, Real)
                        or not math.isfinite(v) for v in p) for p in observation.coordinates)):
                raise ValueError('invalid visible-arm observation')
    joints = [j for j in range(2, 6)
              if all(sample[1].visible[j] for sample in (*outgoing, *incoming))]
    result['common_arm_joint_count'] = len(joints)
    if len(joints) < 2:
        result['reason'] = 'insufficient_common_arm_joints'
        return result

    def velocity(window):
        # Center local times before arithmetic to avoid large-clock cancellation.
        times = [sample[0] - window[0][0] for sample in window]
        mean = sum(times) / len(times)
        centered = [t - mean for t in times]
        denominator = sum(t * t for t in centered)
        return tuple(sum(t * sample[1].coordinates[j][axis]
                         for t, sample in zip(centered, window)) / denominator
                     for j in joints for axis in (0, 1))

    va, vb = velocity(outgoing), velocity(incoming)
    magnitude_a, magnitude_b = (math.sqrt(sum(v * v for v in values)) for values in (va, vb))
    cosine = (sum(a * b for a, b in zip(va, vb)) / (magnitude_a * magnitude_b)
              if magnitude_a > 1e-8 and magnitude_b > 1e-8 else None)
    distance = math.sqrt(sum(math.dist(outgoing[-1][1].coordinates[j],
                                      incoming[0][1].coordinates[j]) ** 2
                             for j in joints) / len(joints))
    result.update(available=True, reason='observed', endpoint_rms=distance,
                  velocity_cosine=max(-1., min(1., cosine)) if cosine is not None else None,
                  outgoing_speed=magnitude_a, incoming_speed=magnitude_b)
    return result


def compare_arm_endpoints(
    outgoing: VisibleArmObservation | None,
    incoming: VisibleArmObservation | None,
) -> dict[str, object]:
    """Measure exact-cut arm shape independently of temporal-flow availability.

    An absent preceding/following observation cannot erase a genuinely measured
    endpoint pair. Conversely, this endpoint measurement does not claim a
    movement direction, temporal phase match, or acceptance of the edit.
    """
    result: dict[str, object] = dict(
        basis=ARM_MOTION_BASIS, available=False, reason='missing_observation',
        common_arm_joint_count=0, endpoint_rms=None,
        temporal_motion_proven=False, quality_calibrated=False,
        final_media_authority=False,
    )
    if outgoing is None or incoming is None:
        return result
    for observation in (outgoing, incoming):
        if (not isinstance(observation, VisibleArmObservation)
                or len(observation.coordinates) != 6 or len(observation.visible) != 6
                or any(type(v) is not bool for v in observation.visible)
                or not all(observation.visible[:2])
                or any(len(p) != 2 or any(isinstance(v, bool) or not isinstance(v, Real)
                    or not math.isfinite(v) for v in p) for p in observation.coordinates)):
            raise ValueError('invalid shoulder-normalized visible-arm observation')
    joints = [j for j in range(2, 6) if outgoing.visible[j] and incoming.visible[j]]
    result['common_arm_joint_count'] = len(joints)
    if len(joints) < 2:
        result['reason'] = 'insufficient_common_arm_joints'
        return result
    distance = math.sqrt(sum(math.dist(outgoing.coordinates[j], incoming.coordinates[j])**2
                             for j in joints) / len(joints))
    result.update(available=True, reason='observed', endpoint_rms=distance)
    return result
