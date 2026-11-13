"""Finite upper-body phase/motion and visible-action crop costs.

Eight nonfacial screen joints are conditional pose observations, not a member
identity, mask or professional-quality certificate. Missing joints have a
finite cost; they are never replaced with a stationary zero trajectory.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ActionMatch:
    phase_loss: float | None
    velocity_loss: float | None
    common_joints: int
    moving_joints: int
    cost: float
    measurement_basis: str | None = None


def synchronous_motion_windows(left_motion, left_valid, right_motion, right_valid):
    """Compare two sources over the SAME pre/post-cut clock windows.

    A synchronized choreographic hit can reverse at the beat. Comparing
    outgoing pre-hit velocity with incoming post-hit velocity confounds that
    hit with an out-of-phase source. Missing windows remain missing.
    """
    a, b = np.asarray(left_motion, float), np.asarray(right_motion, float)
    av, bv = np.asarray(left_valid, bool), np.asarray(right_valid, bool)
    if (a.shape != (2, 8, 2) or b.shape != a.shape or av.shape != (2, 8)
            or bv.shape != av.shape or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('two finite same-clock articulation windows required')
    rows = []
    for window in (0, 1):
        shared = av[window] & bv[window]
        if shared.sum() < 2:
            rows.append(dict(loss=None, common_joints=int(shared.sum()), moving_joints=0,
                             maximum_direction_disagreement=None,joints=[],direction_losses=[],speed_losses=[]))
            continue
        va, vb = a[window, shared], b[window, shared]
        na, nb = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)
        moving = np.maximum(na, nb) >= .3
        if not moving.any():
            rows.append(dict(loss=0., common_joints=int(shared.sum()), moving_joints=0,
                             maximum_direction_disagreement=0.,joints=[],direction_losses=[],speed_losses=[]))
            continue
        va, vb, na, nb = va[moving], vb[moving], na[moving], nb[moving]
        direction = np.ones(len(va))
        both = np.minimum(na, nb) >= .15
        direction[both] = (1. - np.clip((va[both] * vb[both]).sum(1)
            / (na[both] * nb[both]), -1., 1.)) / 2.
        speed = np.minimum(2., np.abs(np.log((na + .15) / (nb + .15))))
        rows.append(dict(loss=float(np.mean(direction + .3 * speed)),
                         common_joints=int(shared.sum()), moving_joints=int(moving.sum()),
                         maximum_direction_disagreement=float(np.max(direction)),
                         joints=np.flatnonzero(shared)[moving].tolist(),
                         direction_losses=direction.tolist(),speed_losses=speed.tolist()))
    return tuple(rows)


def synchronous_velocity(left_motion, left_valid, right_motion, right_valid):
    """Backward-compatible mean; cut decisions also inspect the window tails."""
    rows = synchronous_motion_windows(left_motion, left_valid, right_motion, right_valid)
    losses = [row['loss'] for row in rows if row['loss'] is not None]
    return (float(np.mean(losses)), sum(row['moving_joints'] for row in rows)) if losses else (None, 0)


def _arm_segments(points):
    """Observed directed bones; one complete arm does not need both shoulders.

    Translation and isotropic scale disappear, but rotation, handedness and
    articulation remain. A partially visible arm is not completed/invented.
    """
    valid = (points[:, 2] >= .3) & (points[:, :2] >= 0).all(1) & (points[:, :2] <= 1).all(1)
    pairs = np.array([[0, 2], [2, 4], [1, 3], [3, 5]])
    pixels = points[:, :2] * [16 / 9, 1.]
    delta = pixels[pairs[:, 1]] - pixels[pairs[:, 0]]
    length = np.linalg.norm(delta, axis=1)
    supported = valid[pairs].all(1) & (length >= .025)
    # Both directed segments of an arm must be real observations.
    supported = np.repeat(supported.reshape(2, 2).all(1), 2)
    return delta / np.maximum(length[:, None], 1e-12), supported


def complete_arm_motion(early, late, *, seconds):
    """Same-clock angular motion for actually observed complete arms.

    Padded slots 2:6 carry directed bones, NOT body-relative joint positions.
    Consumers must select a common basis before comparing the two sources.
    """
    if not math.isfinite(seconds) or seconds<=0:
        raise ValueError('positive complete-arm motion window required')
    a,av=_arm_segments(np.asarray(early,float));b,bv=_arm_segments(np.asarray(late,float))
    velocity,valid=np.zeros((8,2)),np.zeros(8,bool)
    velocity[2:6]=(b-a)/seconds;valid[2:6]=av&bv
    return velocity,valid


def _compare_partial_arm(left, right, seconds):
    a, av = _arm_segments(left[1])
    b, bv = _arm_segments(right[0])
    common = av & bv
    if not common.any():
        return ActionMatch(None, None, 0, 0, 2.)
    phase = float(np.linalg.norm(a[common] - b[common], axis=1).mean() / 2.)
    prior, pv = _arm_segments(left[0])
    later, lv = _arm_segments(right[1])
    shared = common & pv & lv
    velocity, moving_count = None, 0
    if shared.any():
        va, vb = (a[shared] - prior[shared]) / seconds, (later[shared] - b[shared]) / seconds
        na, nb = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)
        moving = np.maximum(na, nb) >= .3
        moving_count = int(moving.sum())
        if moving_count:
            va, vb, na, nb = va[moving], vb[moving], na[moving], nb[moving]
            direction = np.ones(len(va))
            both = np.minimum(na, nb) >= .15
            direction[both] = (1. - np.clip((va[both] * vb[both]).sum(1)
                / (na[both] * nb[both]), -1., 1.)) / 2.
            velocity = float(np.mean(direction + .3 * np.minimum(2., np.abs(np.log((na + .15) / (nb + .15))))))
        else:
            velocity = 0.
    return ActionMatch(phase, velocity, 3 * int(common.reshape(2, 2).all(1).sum()), moving_count,
        3. * phase + (1.5 * velocity if velocity is not None else 1.), 'directed_complete_arm_segments')


def normalized_body(points):
    points = np.asarray(points, float)
    if points.shape != (8, 3) or not np.isfinite(points).all():
        raise ValueError('eight finite screen-space pose joints required')
    valid = (points[:, 2] >= .3) & (points[:, :2] >= 0).all(1) & (points[:, :2] <= 1).all(1)
    # Canonicalize translation and isotropic scale, not rotation/mirroring:
    # head/body tilt and handed action are precisely what must match.
    if not valid[:2].all():
        return None, valid
    pixels = points[:, :2] * [16 / 9, 1.]
    scale = float(np.linalg.norm(pixels[0] - pixels[1]))
    if scale < .025:
        return None, valid
    return (pixels - pixels[:2].mean(0)) / scale, valid


def compare_action(left, right, *, seconds):
    """Compare outgoing [earlier,last] and incoming [first,later] articulation.

    Body-relative velocities remove source-camera translation/scale. Camera
    motion in final pixels is a separate duty, not certified by this score.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    if (left.shape != (2, 8, 3) or right.shape != (2, 8, 3)
            or not np.isfinite(left).all() or not np.isfinite(right).all()
            or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('two ordered finite pose windows and positive time required')
    a, av = normalized_body(left[1])
    b, bv = normalized_body(right[0])
    common = av & bv
    count = int(common.sum()) if a is not None and b is not None else 0
    # Two shoulders/hips alone can agree perfectly while BOTH arms differ.
    # Do not advertise torso-anchor agreement as an articulated action match.
    if count < 4 or int(common[2:6].sum()) < 2:
        partial = _compare_partial_arm(left, right, seconds)
        return partial if partial.phase_loss is not None else ActionMatch(None, None, count, 0, 2.)
    weights = np.asarray((.5, .5, 1.5, 1.5, 2., 2., .5, .5))
    phase = float(np.average(np.linalg.norm(a[common] - b[common], axis=1),
        weights=weights[common]))
    prior, pv = normalized_body(left[0])
    later, lv = normalized_body(right[1])
    velocity, activity = None, 0
    if prior is not None and later is not None:
        shared = common & pv & lv
        # Shoulders define the normalization basis, not choreography activity.
        shared[:2] = False
        shared[6:] = False
        if shared.sum() >= 2:
            va, vb = (a[shared] - prior[shared]) / seconds, (later[shared] - b[shared]) / seconds
            na, nb = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)
            moving = np.maximum(na, nb) >= .3
            activity = int(moving.sum())
            if activity:
                va, vb, na, nb = va[moving], vb[moving], na[moving], nb[moving]
                both = np.minimum(na, nb) >= .15
                directions = np.ones(len(va))
                directions[both] = (1. - np.clip((va[both] * vb[both]).sum(1)
                    / (na[both] * nb[both]), -1., 1.)) / 2
                speeds = np.minimum(2., np.abs(np.log((na + .15) / (nb + .15))))
                velocity = float(np.mean(directions + .3 * speeds))
            else:
                velocity = 0.  # both actually observed stationary, not missing
    return ActionMatch(phase, velocity, count, activity,
        3. * phase + (1.5 * velocity if velocity is not None else 1.), 'shoulder_relative_articulation')


def action_crop_cost(zoom, center, support):
    """Charge crop-added clipping, not hands already absent from the source.

    Supports can contain the two measured temporal flanks. Soft edge clearance
    leaves room for an observed moving wrist instead of fitting only the face.
    """
    zoom = np.asarray(zoom, float).reshape(-1)
    center = np.asarray(center, float)
    if (center.shape != (len(zoom), 2) or not np.isfinite(zoom).all()
            or not np.isfinite(center).all() or np.any(zoom < 1)):
        raise ValueError('finite no-padding crop candidates required')
    if support is None:
        return np.zeros(len(zoom))
    points = np.asarray(support, float).reshape(-1, 8, 3)
    if not np.isfinite(points).all() or len(points) > 2:
        raise ValueError('one or two finite measured body supports required')
    visible = (points[:, :, 2] >= .3) & (points[:, :, :2] >= 0).all(2) & (points[:, :, :2] <= 1).all(2)
    projected = .5 + zoom[:, None, None, None] * (points[None, :, :, :2] - center[:, None, None])
    outside = np.maximum(0., -projected) + np.maximum(0., projected - 1.)
    # Original visible action near a border can stay there without being
    # punished merely for the native broadcaster's framing.
    native_clearance = np.minimum(points[:, :, :2], 1. - points[:, :, :2]).min(2)
    clearance = np.minimum(projected, 1. - projected).min(3)
    extra_edge_loss = np.maximum(0., np.minimum(.04, native_clearance)[None] - clearance)
    severity = 12. * outside.sum(3) + 2. * extra_edge_loss
    weights = np.asarray((.5, .5, 1., 1., 2., 2., .5, .5))
    return (severity * visible[None] * weights).sum((1, 2)) / max(1., float((visible * weights).sum()))


def framing_cost(left, right, ha, hb, support=(None, None)):
    """Finite aesthetic prior: use zoom for a match, not a larger face reward."""
    total = .65 * (math.log(left.zoom) + math.log(right.zoom))
    for camera, height, points in zip((left, right), (ha, hb), support, strict=True):
        # Penalize only extra tightness introduced by our crop. A native
        # close-up is not resized down/padded to manufacture this preference.
        total += 6. * max(0., height * camera.zoom - max(.24, height))
        total += float(action_crop_cost([camera.zoom], [[camera.x, camera.y]], points)[0])
    # Keep this producer's public numeric contract independent of NumPy's
    # weak scalar promotion (float32 heights must not leak into plan JSON).
    return float(total)


def compare_screen_motion(left, right, *, seconds, frame_seconds):
    """One-sided motion of ACTUAL projected screen joints, including camera.

    Input is already in rendered screen coordinates (or explicitly projected
    by a fixed candidate camera). This does not identify or track the actor.
    """
    a, b = np.asarray(left, float), np.asarray(right, float)
    if (a.shape != (2, 8, 3) or b.shape != (2, 8, 3)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not 0 < frame_seconds <= seconds or not math.isfinite(seconds)):
        raise ValueError('ordered projected joint windows and finite output clock required')
    valid = lambda p: (p[:, 2] >= .3) & (p[:, :2] >= 0).all(1) & (p[:, :2] <= 1).all(1)
    common = valid(a[1]) & valid(b[0])
    shared = common & valid(a[0]) & valid(b[1])
    count = int(common.sum())
    result = dict(position_loss=None, prediction_loss=None, velocity_loss=None,
        common_joints=count, velocity_joints=int(shared.sum()), cost=2.)
    if count < 4:
        return result
    weights = np.array([.5, .5, 1., 1., 2., 2., .5, .5])
    points_a, points_b = a[:, :, :2] * [16 / 9, 1.], b[:, :, :2] * [16 / 9, 1.]
    position = float(np.average(np.linalg.norm(points_a[1, common] - points_b[0, common], axis=1),
        weights=weights[common]))
    result.update(position_loss=position, cost=position / .12 + 1.)
    if shared.sum() < 4:
        return result
    va, vb = (points_a[1, shared] - points_a[0, shared]) / seconds, (points_b[1, shared] - points_b[0, shared]) / seconds
    na, nb = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)
    moving = np.maximum(na, nb) > .05
    direction = np.zeros(len(va))
    direction[moving] = 1.
    both = moving & (np.minimum(na, nb) > .025)
    direction[both] = (1. - np.clip((va[both] * vb[both]).sum(1) / (na[both] * nb[both]), -1., 1.)) / 2
    speed = np.minimum(2., np.abs(np.log((na + .025) / (nb + .025))))
    velocity = float(np.average(direction + .3 * speed, weights=weights[shared]))
    prediction = float(np.average(np.linalg.norm(points_a[1, shared] + frame_seconds * va
        - points_b[0, shared], axis=1), weights=weights[shared]))
    return dict(result, prediction_loss=prediction, velocity_loss=velocity,
        cost=position / .12 + prediction / .12 + .75 * velocity)
