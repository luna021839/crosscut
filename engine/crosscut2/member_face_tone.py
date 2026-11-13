"""Window-stable, local skin-tone bridge for real paired face pictures.

Five-point guides define photometric sampling/support, not a semantic face
mask or an identity observation. No garment/stage colour is matched. Both
source faces move towards one slowly changing Lab reference, independently
of the later person/background dissolve; original endpoints remain intact.
"""
import cv2
import numpy as np


def _coordinates(shape, landmarks):
    points = np.asarray(landmarks, np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError('finite actual five-point face guide required')
    eyes = (points[0] + points[1]) / 2
    horizontal = points[1] - points[0]
    spacing = float(np.linalg.norm(horizontal))
    if spacing < 3:
        raise ValueError('face guide too small for skin-tone sampling')
    horizontal /= spacing
    vertical = np.array([-horizontal[1], horizontal[0]], np.float32)
    mouth = (points[3] + points[4]) / 2
    length = float((mouth - eyes) @ vertical)
    if length < 2:
        raise ValueError('degenerate facial vertical guide')
    # All later work is confined to this small photometric ROI.
    radius = spacing * 1.15 + length * 1.9
    h, w = shape[:2]
    x0, y0 = np.maximum(0, np.floor(eyes - radius).astype(int))
    x1, y1 = np.minimum([w, h], np.ceil(eyes + radius).astype(int) + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('facial guide outside the actual picture')
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - eyes[0], yy - eyes[1]
    x = (dx * horizontal[0] + dy * horizontal[1]) / spacing
    y = (dx * vertical[0] + dy * vertical[1]) / length
    return (slice(y0, y1), slice(x0, x1)), x, y


def _skin_reference(lab, x, y):
    # Bilateral cheeks avoid the eyes, mouth and bangs. Disagreement between
    # individual pictures is smoothed across the short real source window.
    cheeks = ((np.abs(np.abs(x) - .38) < .18) & (y > .3) & (y < .9)
        & (lab[..., 0] > 12) & (lab[..., 0] < 98))
    samples = lab[cheeks]
    if len(samples) < 6:
        raise ValueError('insufficient real cheek pixels for colour matching')
    return np.median(samples, axis=0)


def fit(outgoing, incoming, outgoing_landmarks, incoming_landmarks):
    """Return smoothed incoming-minus-outgoing Lab differences per picture.

    Accepts low-resolution copies of the SAME physical window. Missing/invalid
    observations are not invented; callers keep the ordinary effect instead.
    """
    lengths = {len(v) for v in (outgoing, incoming, outgoing_landmarks, incoming_landmarks)}
    if len(lengths) != 1 or not 3 <= len(outgoing) <= 60:
        raise ValueError('one bounded paired face-tone window required')
    references = []
    for pictures, guides in ((outgoing, outgoing_landmarks), (incoming, incoming_landmarks)):
        values = []
        for picture, guide in zip(pictures, guides, strict=True):
            if picture.dtype != np.uint8 or picture.ndim != 3 or picture.shape[2] != 3:
                raise ValueError('actual BGR8 source picture required')
            region, x, y = _coordinates(picture.shape, guide)
            lab = cv2.cvtColor(picture[region].astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
            values.append(_skin_reference(lab, x, y))
        references.append(np.asarray(values))
    delta = references[1] - references[0]
    # Robust to individual eyelid/hand/segmentation samples; no per-frame
    # auto-white-balance feedback that could create a new visible flicker.
    padded = np.pad(delta, ((2, 2), (0, 0)), mode='edge')
    delta = np.stack([np.median(padded[j:j+5], axis=0) for j in range(len(delta))])
    padded = np.pad(delta, ((2, 2), (0, 0)), mode='edge')
    delta = sum(weight * padded[j:j+len(delta)] for j, weight in enumerate((1, 2, 3, 2, 1))) / 9
    return np.clip(delta, [-18, -14, -14], [18, 14, 14]).astype(np.float32)


def correct(picture, landmarks, person, delta, strength):
    """Locally shift real skin colour; preserve clothes/background/shading."""
    delta = np.asarray(delta, np.float32)
    if delta.shape != (3,) or not np.isfinite(delta).all() or not np.isfinite(strength) or not -1 <= strength <= 1:
        raise ValueError('bounded finite Lab correction required')
    if np.any(np.abs(delta) > [18, 14, 14]):
        raise ValueError('face-tone correction exceeds its photometric limits')
    if strength == 0 or not np.any(delta):
        return picture
    region, x, y = _coordinates(picture.shape, landmarks)
    original = picture[region].astype(np.float32) / 255
    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    reference = _skin_reference(lab, x, y)
    radius = np.sqrt((x / .96)**2 + ((y - .38) / 1.36)**2)
    support = np.clip((1.12 - radius) / .22, 0, 1)
    # This colour weight excludes hair, eye makeup and lips, and is feathered
    # rather than a binary skin classifier. Person support excludes the stage.
    chroma = np.linalg.norm(lab[..., 1:] - reference[1:], axis=2)
    support *= np.clip((25 - chroma) / 12, 0, 1)
    support *= np.clip((lab[..., 0] - 10) / 18, 0, 1)
    support *= np.asarray(person[region], np.float32)
    lab += delta * strength
    changed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    result = picture.copy()
    result[region] = np.rint(np.clip(original + (changed - original) * support[..., None], 0, 1) * 255).astype(np.uint8)
    return result
