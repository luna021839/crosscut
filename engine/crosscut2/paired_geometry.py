"""Anonymous focal/formation geometry and jointly feasible two-sided crops.

These are editing costs, not named-person probabilities or action certificates.
Both cameras are scored AFTER clamping, using the same crop the renderer sees.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .action_geometry import action_crop_cost, compare_action, framing_cost


@dataclass(frozen=True)
class Camera:
    zoom: float = 1.
    x: float = .5
    y: float = .5

    def project(self, xy):
        return .5 + self.zoom * (np.asarray(xy) - [self.x, self.y])


@dataclass(frozen=True)
class Match:
    left: Camera
    right: Camera
    role: str
    cost: float
    geometry_loss: float
    appearance_cosine: float | None = None
    left_head_index: int | None = None
    right_head_index: int | None = None
    pose_loss: float | None = None
    velocity_loss: float | None = None
    action_joints: int = 0
    framing_loss: float = 0.


def joint_cameras(a, b, ha, hb, caps=(1.35, 1.35), *, support=(None, None)):
    """Minimize realized scale/position error with no padding or zoom-out.

    Search a small *shared* target-size grid; extra crop can make a common
    focal position reachable. Choosing each camera separately misses this.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (a.shape != (2,) or b.shape != (2,) or not np.isfinite([*a, *b, ha, hb, *caps]).all()
            or min(ha, hb) <= 0 or any(not 1 <= c <= 2 for c in caps)):
        raise ValueError('finite geometry and supported zoom caps required')
    lower = max(ha, hb)
    upper = max(ha * caps[0], hb * caps[1])
    target = np.linspace(lower, upper, 9)
    za, zb = np.clip(target / ha, 1., caps[0]), np.clip(target / hb, 1., caps[1])
    # Evaluate the shared target grid in one numeric pass, not 9 Python loops
    # for every head pairing at every boundary.
    lo_a, hi_a = 1 + za[:, None] * (a - 1), za[:, None] * a
    lo_b, hi_b = 1 + zb[:, None] * (b - 1), zb[:, None] * b
    lo, hi = np.maximum(lo_a, lo_b), np.minimum(hi_a, hi_b)
    target_xy = np.where(lo <= hi, np.clip((a + b) / 2, lo, hi), (lo + hi) / 2)
    xa, xb = np.clip(target_xy, lo_a, hi_a), np.clip(target_xy, lo_b, hi_b)
    ca, cb = a + (.5 - xa) / za[:, None], b + (.5 - xb) / zb[:, None]
    # The existing renderer also has a radial quarter-frame pan cap.
    for centers in (ca, cb):
        offsets = centers - .5
        radii = np.linalg.norm(offsets, axis=1)
        centers[:] = .5 + offsets * np.minimum(1., .25 / np.maximum(1e-12, radii))[:, None]
    xa, xb = .5 + za[:, None] * (a - ca), .5 + zb[:, None] * (b - cb)
    position = np.linalg.norm((xa - xb) * [1., 9 / 16], axis=1)
    scale = np.abs(np.log(ha * za / (hb * zb)))
    geometry = position / .06 + scale / .25
    effort = .65 * (np.log(za) + np.log(zb))
    effort += 6. * (np.maximum(0., ha * za - max(.24, ha))
        + np.maximum(0., hb * zb - max(.24, hb)))
    effort += action_crop_cost(za, ca, support[0]) + action_crop_cost(zb, cb, support[1])
    chosen = int(np.argmin(geometry + effort))
    return (Camera(float(za[chosen]), float(ca[chosen, 0]), float(ca[chosen, 1])),
        Camera(float(zb[chosen]), float(cb[chosen, 0]), float(cb[chosen, 1])), float(geometry[chosen]))


def _cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if min(na, nb) < .5:
        return None
    if a.shape != b.shape:
        raise ValueError('appearance families must not be mixed')
    return float(np.clip(a @ b / (na * nb), -1., 1.))


def match_heads(left, right, lv, rv, *, caps=(1.35, 1.35), left_salience=None, right_salience=None,
                left_pose=None, right_pose=None, motion_seconds=.2):
    """Compare focal and anonymous formation hypotheses, including occluders.

    Head rows are (center_x, center_y, height, confidence). Missing appearance
    has a finite cost, not a veto or a false perfect member match. Foreground
    occluders do not force the largest detected face to be the edited subject.
    """
    left, right = np.asarray(left), np.asarray(right)
    for window in (left_pose, right_pose):
        if window is not None and (np.asarray(window).shape != (2, 3, 8, 3)
                or not np.isfinite(window).all()):
            raise ValueError('conditional focal pose windows require two frames and three heads')
    if (left_pose is None) != (right_pose is None):
        raise ValueError('both actual sides are required for a conditional action comparison')
    a = left[(left[:, 2] > .019) & (left[:, 3] >= .75)]
    b = right[(right[:, 2] > .019) & (right[:, 3] >= .75)]
    if not len(a) or not len(b):
        return Match(Camera(), Camera(), 'unobserved', 4., 4.)
    def salience(rows, values):
        if values is None:
            values = rows[:3, 2] / rows[:, 2].max()
        values = np.asarray(values, float)
        if values.shape != (3,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('finite per-head salience required')
        return values / max(.001, values.max())
    sa, sb = salience(left, left_salience), salience(right, right_salience)
    formation_intent = sum(bool(len(rows) >= 3 and np.max(rows[:, 2]) < 1.6 * np.sort(rows[:, 2])[-2])
        for rows in (a, b))
    hypotheses = []
    for i, fa in enumerate(left[:3]):
        if fa[2] < .055 or fa[3] < .75:
            continue
        for j, fb in enumerate(right[:3]):
            if fb[2] < .055 or fb[3] < .75:
                continue
            support = (None, None) if left_pose is None else (left_pose[:, i], right_pose[:, j])
            ca, cb, geometry = joint_cameras(fa[:2], fb[:2], fa[2], fb[2], caps, support=support)
            framing = framing_cost(ca, cb, fa[2], fb[2], support)
            similarity = _cosine(lv[i], rv[j])
            # SFace cosine is evidence of appearance, NOT a calibrated label.
            identity = 1.4 if similarity is None else 5. * max(0., .65 - similarity)
            # A background head match is not the same as keeping the viewer's
            # subject. Temporal salience can discount a brief foreground occluder.
            focal_priority = -2. * (math.log(max(.02, sa[i])) + math.log(max(.02, sb[j])))
            action = None if left_pose is None else compare_action(support[0], support[1], seconds=motion_seconds)
            # Compare to the SAME finite missing-action prior (2.) whether
            # enabled or disabled. An all-UNKNOWN bank leaves the base graph,
            # focal/formation ranking and rhythm exactly unchanged; zero
            # refinement is not a claim that missing motion is a match.
            action_cost = 0. if action is None else action.cost - 2.
            if (action is not None and action.moving_joints >= 2
                    and action.velocity_loss is not None and action.velocity_loss < .4):
                action_cost -= .3
            hypotheses.append(Match(ca, cb, 'focal', geometry + identity + focal_priority
                + formation_intent + framing + action_cost, geometry, similarity, i, j,
                None if action is None else action.phase_loss,
                None if action is None else action.velocity_loss,
                0 if action is None else action.common_joints, framing))
    if min(len(a), len(b)) >= 3:
        # Keep the central foreground scale band, rather than tiny back heads.
        a = a[a[:, 2] >= .55 * np.median(a[:, 2])]
        b = b[b[:, 2] >= .55 * np.median(b[:, 2])]
        if min(len(a), len(b)) >= 3:
            a, b = a[np.argsort(a[:, 0])], b[np.argsort(b[:, 0])]
            ma, mb = a[:, :2].mean(0), b[:, :2].mean(0)
            ha, hb = float(np.median(a[:, 2])), float(np.median(b[:, 2]))
            ca, cb, geometry = joint_cameras(ma, mb, ha, hb, caps)
            framing = framing_cost(ca, cb, ha, hb)
            pa, pb = ca.project(a[:, :2]), cb.project(b[:, :2])
            # Sorted position is anonymous choreography geometry, not identity.
            if len(a) == len(b):
                layout = float(np.linalg.norm((pa - pb) * [1., 9 / 16], axis=1).mean()) / .08
            else:
                centered_a, centered_b = pa - pa.mean(0), pb - pb.mean(0)
                distances = np.linalg.norm((centered_a[:, None] - centered_b[None]) * [1., 9 / 16], axis=2)
                layout = float((distances.min(0).mean() + distances.min(1).mean()) / 2) / .08
            loss = geometry + layout + .45 * abs(len(a) - len(b))
            # Matching background layout is not a formation match when one
            # side has a dominant foreground subject. Keep this a finite
            # editorial cost rather than banning an otherwise useful source.
            role_change = 3. * (2 - formation_intent)
            hypotheses.append(Match(ca, cb, 'formation', loss + .35 + role_change + framing,
                loss, framing_loss=framing))
    return min(hypotheses, key=lambda r: (r.cost, r.role)) if hypotheses else Match(
        Camera(), Camera(), 'distant', 3., 3.)
