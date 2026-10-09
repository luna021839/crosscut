"""Anonymous focal/formation geometry and jointly feasible two-sided crops.

These are editing costs, not named-person probabilities or action certificates.
Both cameras are scored AFTER clamping, using the same crop the renderer sees.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


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


def joint_cameras(a, b, ha, hb, caps=(1.35, 1.35)):
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
    effort = .12 * (np.log(za) + np.log(zb))
    chosen = int(np.argmin(geometry + effort))
    return Camera(float(za[chosen]), *ca[chosen]), Camera(float(zb[chosen]), *cb[chosen]), float(geometry[chosen])


def _cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if min(na, nb) < .5:
        return None
    if a.shape != b.shape:
        raise ValueError('appearance families must not be mixed')
    return float(np.clip(a @ b / (na * nb), -1., 1.))


def match_heads(left, right, lv, rv, *, caps=(1.35, 1.35)):
    """Compare focal and anonymous formation hypotheses, including occluders.

    Head rows are (center_x, center_y, height, confidence). Missing appearance
    has a finite cost, not a veto or a false perfect member match. Foreground
    occluders do not force the largest detected face to be the edited subject.
    """
    left, right = np.asarray(left), np.asarray(right)
    a = left[(left[:, 2] > .019) & (left[:, 3] >= .75)]
    b = right[(right[:, 2] > .019) & (right[:, 3] >= .75)]
    if not len(a) or not len(b):
        return Match(Camera(), Camera(), 'unobserved', 4., 4.)
    hypotheses = []
    for i, fa in enumerate(left[:3]):
        if fa[2] < .055 or fa[3] < .75:
            continue
        for j, fb in enumerate(right[:3]):
            if fb[2] < .055 or fb[3] < .75:
                continue
            ca, cb, geometry = joint_cameras(fa[:2], fb[:2], fa[2], fb[2], caps)
            similarity = _cosine(lv[i], rv[j])
            # SFace cosine is evidence of appearance, NOT a calibrated label.
            identity = 1.4 if similarity is None else 5. * max(0., .65 - similarity)
            salience = .45 * (math.log(a[:, 2].max() / fa[2]) + math.log(b[:, 2].max() / fb[2]))
            hypotheses.append(Match(ca, cb, 'focal', geometry + identity + salience,
                geometry, similarity))
    if min(len(a), len(b)) >= 3:
        # Keep the central foreground scale band, rather than tiny back heads.
        a = a[a[:, 2] >= .55 * np.median(a[:, 2])]
        b = b[b[:, 2] >= .55 * np.median(b[:, 2])]
        if min(len(a), len(b)) >= 3:
            a, b = a[np.argsort(a[:, 0])], b[np.argsort(b[:, 0])]
            ma, mb = a[:, :2].mean(0), b[:, :2].mean(0)
            ha, hb = float(np.median(a[:, 2])), float(np.median(b[:, 2]))
            ca, cb, geometry = joint_cameras(ma, mb, ha, hb, caps)
            pa, pb = ca.project(a[:, :2]), cb.project(b[:, :2])
            # Sorted position is anonymous choreography geometry, not identity.
            if len(a) == len(b):
                layout = float(np.linalg.norm((pa - pb) * [1., 9 / 16], axis=1).mean()) / .08
            else:
                centered_a, centered_b = pa - pa.mean(0), pb - pb.mean(0)
                distances = np.linalg.norm((centered_a[:, None] - centered_b[None]) * [1., 9 / 16], axis=2)
                layout = float((distances.min(0).mean() + distances.min(1).mean()) / 2) / .08
            loss = geometry + layout + .45 * abs(len(a) - len(b))
            hypotheses.append(Match(ca, cb, 'formation', loss + .35, loss))
    return min(hypotheses, key=lambda r: (r.cost, r.role)) if hypotheses else Match(
        Camera(), Camera(), 'distant', 3., 3.)
