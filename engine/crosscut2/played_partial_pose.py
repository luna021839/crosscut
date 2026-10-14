"""Actual played A[-7,-1] to B[0,+6] partial pose, finite tentative ranking.

Evidence-kind is NOT a probability. Unknown/contradicted = .5; model-only
shrinks raw residual halfway toward .5. Pixel support cannot remove ambiguous
ownership. Missing semantic channels are not dropped from the denominator.
"""
import math
import numpy as np
from .physical_display_observations import UNMEASURED, MODEL_ONLY, PIXEL_SUPPORTED, CONTRADICTED

PAIRS = np.array(((0, 2), (2, 4), (0, 4), (1, 3), (3, 5), (1, 5), (0, 1), (0, 6), (1, 7), (6, 7)))
CHANNELS = ((0, 1, 2), (3, 4, 5), (6, 7, 8, 9))


def rho(x):
    x = np.maximum(0., x)
    return x / (1. + x)


def tentative(raw, kind, ambiguous):
    return np.where((kind == UNMEASURED) | (kind == CONTRADICTED), .5,
        np.where((kind == PIXEL_SUPPORTED) & ~ambiguous, raw, .5 + .5 * (raw - .5)))


def features(observations, *, fps):
    """Each observation belongs to one physical face and one played side."""
    if not math.isfinite(fps) or fps <= 0:raise ValueError('positive played clock required')
    heads = np.asarray([o['head'] for o in observations], float)
    body = np.asarray([o['body'] for o in observations], float)
    evidence = np.asarray([o['evidence'] for o in observations], np.uint8)
    ambiguity = np.asarray([o['ownership_ambiguity'] for o in observations], bool)
    camera = np.asarray([o['camera_strip'][[0, -1]] for o in observations], float)
    side = np.asarray([o['side'] for o in observations], int)
    count = len(observations)
    if (heads.shape != (count, 4) or body.shape != (count, 2, 8, 3)
            or evidence.shape != (count, 2, 8) or ambiguity.shape != evidence.shape
            or camera.shape != (count, 2, 3) or np.any((evidence < 0) | (evidence > 3))
            or not all(np.isfinite(x).all() for x in (heads, body, camera))):
        raise ValueError('bound physical face/body/strip observation required')
    at = np.where(side == 0, 1, 0)
    endpoint_camera = camera[np.arange(count), at]
    center = (.5 + endpoint_camera[:, :1] * (heads[:, :2] - endpoint_camera[:, 1:])) * [16 / 9, 1.]
    height = heads[:, 2] * endpoint_camera[:, 0]
    seen = (heads[:, 2] > .019) & (heads[:, 3] >= .75)
    xy = (.5 + camera[:, :, None, :1] * (body[..., :2] - camera[:, :, None, 1:])) * [16 / 9, 1.]
    valid = (body[..., 2] >= .3) & (body[..., :2] >= 0).all(-1) & (body[..., :2] <= 1).all(-1)
    bone = xy[:, :, PAIRS[:, 1]] - xy[:, :, PAIRS[:, 0]]
    length = np.linalg.norm(bone, axis=-1)
    good = valid[..., PAIRS].all(-1) & (length > .005)
    unit = bone / np.maximum(1e-9, length[..., None])
    kinds = evidence[..., PAIRS]
    kind = np.where((kinds == UNMEASURED).any(-1) | ~good, UNMEASURED,
        np.where((kinds == CONTRADICTED).any(-1), CONTRADICTED,
        np.where((kinds == PIXEL_SUPPORTED).all(-1), PIXEL_SUPPORTED, MODEL_ONLY)))
    ambiguous = ambiguity[..., PAIRS].any(-1)
    physical = np.asarray([o['physical_window_valid'] for o in observations], bool)
    clocks = np.asarray([o['native_indices'] for o in observations], int)
    physical &= clocks[:, 1] > clocks[:, 0]
    return dict(heads=heads, center=center, height=height, seen=seen, xy=xy, unit=unit,
        endpoint=at, kind=kind, ambiguous=ambiguous, physical=physical, fps=fps)


def matrix(left, right, *, details=False):
    """Both feature sets contain ONLY exposed windows; no counterfactual H."""
    if left['fps'] != right['fps']:raise ValueError('same played fps required')
    la, rb = left['endpoint'], right['endpoint']; ai = np.arange(len(la)); bi = np.arange(len(rb))
    h = np.maximum(.005, (left['height'][:, None] + right['height'][None, :]) / 2.)
    face_seen = left['seen'][:, None] & right['seen'][None, :]
    log_scale = np.abs(np.log(np.maximum(.001, left['height'][:, None]) / np.maximum(.001, right['height'][None, :])))
    displacement = np.linalg.norm(left['center'][:, None] - right['center'][None, :], axis=-1) / h
    face_error = np.maximum(log_scale / math.log(1.5), displacement)
    allowed = ~face_seen | (face_error <= 1.)
    face = np.where(face_seen, np.minimum(1., face_error), 1.)
    # [source point coordinates] -> [actual per-side rendered coordinates].
    pa = left['xy'][ai, la][:, PAIRS[:, 1]]; pb = right['xy'][bi, rb][:, PAIRS[:, 1]]
    ua = left['unit'][ai, la]; ub = right['unit'][bi, rb]
    relative_a = (pa - left['center'][:, None]) / np.maximum(.005, left['height'][:, None, None])
    relative_b = (pb - right['center'][:, None]) / np.maximum(.005, right['height'][:, None, None])
    phase_raw = .5 * (np.linalg.norm(ua[:, None] - ub[None, :], axis=-1) / 2.
        + rho(np.linalg.norm(relative_a[:, None] - relative_b[None, :], axis=-1)))
    tau = 6. / left['fps']; dt = 1. / left['fps']
    va = (left['xy'][:, 1, PAIRS[:, 1]] - left['xy'][:, 0, PAIRS[:, 1]]) / tau
    vb = (right['xy'][:, 1, PAIRS[:, 1]] - right['xy'][:, 0, PAIRS[:, 1]]) / tau
    gap = pb[None, :] - pa[:, None]
    bridge_raw = .5 * (rho(np.linalg.norm(gap - va[:, None] * dt, axis=-1) / h[..., None])
        + rho(np.linalg.norm(gap - vb[None, :] * dt, axis=-1) / h[..., None]))
    velocity_raw = rho(tau * np.linalg.norm(va[:, None] - vb[None, :], axis=-1) / h[..., None])
    ka = left['kind'][ai, la]; kb = right['kind'][bi, rb]
    def combine(a, b):
        return np.where((a[:, None] == UNMEASURED) | (b[None, :] == UNMEASURED), UNMEASURED,
            np.where((a[:, None] == CONTRADICTED) | (b[None, :] == CONTRADICTED), CONTRADICTED,
            np.where((a[:, None] == PIXEL_SUPPORTED) & (b[None, :] == PIXEL_SUPPORTED), PIXEL_SUPPORTED, MODEL_ONLY)))
    kind = combine(ka, kb)
    kind = np.where(face_seen[..., None], kind, UNMEASURED)
    ambiguous = left['ambiguous'][ai, la][:, None] | right['ambiguous'][bi, rb][None, :]
    phase = tentative(phase_raw, kind, ambiguous)
    ma = np.where((left['kind'] == UNMEASURED).any(1), UNMEASURED,
        np.where((left['kind'] == CONTRADICTED).any(1), CONTRADICTED,
        np.where((left['kind'] == PIXEL_SUPPORTED).all(1), PIXEL_SUPPORTED, MODEL_ONLY)))
    mb = np.where((right['kind'] == UNMEASURED).any(1), UNMEASURED,
        np.where((right['kind'] == CONTRADICTED).any(1), CONTRADICTED,
        np.where((right['kind'] == PIXEL_SUPPORTED).all(1), PIXEL_SUPPORTED, MODEL_ONLY)))
    motion_kind = combine(ma, mb)
    motion_kind = np.where(face_seen[..., None] & left['physical'][:, None, None] & right['physical'][None, :, None], motion_kind, UNMEASURED)
    motion_ambiguous = left['ambiguous'].any(1)[:, None] | right['ambiguous'].any(1)[None, :]
    bridge = tentative(bridge_raw, motion_kind, motion_ambiguous)
    velocity = tentative(velocity_raw, motion_kind, motion_ambiguous)
    primitives = (phase + bridge + velocity) / 3.
    channels = np.stack([primitives[..., list(c)].mean(-1) for c in CHANNELS], axis=-1)
    pose = channels.mean(-1)
    result = dict(face_rate=face, pose_rate=pose, allowed=allowed, measured_face=face_seen,
        face_log_scale=log_scale, face_height_displacement=displacement,
        fixed_semantic_channel_rates=channels,
        phase_raw=phase_raw, bridge_raw=bridge_raw, velocity_raw=velocity_raw,
        phase_evidence_kind=kind, motion_evidence_kind=motion_kind,
        ownership_ambiguity=ambiguous, shrinkage=.5, anatomical_ownership_certified=False)
    return result if details else (face + pose, allowed)


def source_activity(observation, *, fps):
    """No digital camera appears in this function: stationary source stays0."""
    body = np.asarray(observation['body'], float)
    if not observation['physical_window_valid']:return 0.
    clocks = observation['native_indices']
    if clocks[1] <= clocks[0]:return 0.
    valid = (body[..., 2] >= .3) & (body[..., :2] >= 0).all(-1) & (body[..., :2] <= 1).all(-1)
    xy = body[..., :2] * [16 / 9, 1.]
    bones = xy[:, PAIRS[:, 1]] - xy[:, PAIRS[:, 0]]
    length = np.linalg.norm(bones, axis=-1)
    good = valid[..., PAIRS].all(-1).all(0) & (length > .005).all(0)
    unit = bones / np.maximum(1e-9, length[..., None])
    velocity = np.linalg.norm(unit[1] - unit[0], axis=-1) / (6. / fps)
    scale = max(.005, float(observation['head'][2]))
    # Relative joint-to-shoulder motion removes common source translation.
    relative = np.linalg.norm((bones[1] - bones[0]) / scale, axis=-1) / (6. / fps)
    kind = np.asarray(observation['evidence'])[..., PAIRS]
    bad = ((kind == UNMEASURED) | (kind == CONTRADICTED)).any((0, 2))
    ambiguity = np.asarray(observation['ownership_ambiguity'])[..., PAIRS].any((0, 2))
    cap = np.where((kind == PIXEL_SUPPORTED).all((0, 2)) & ~ambiguity, 1., .5)
    return float(np.max(np.where(good & ~bad, np.maximum(rho(velocity), rho(relative)) * cap, 0.), initial=0.))
