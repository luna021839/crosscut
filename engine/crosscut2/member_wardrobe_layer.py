"""Optional measured-garment transition; no clothing generation or selection.

Garments arrive before the background. Face registration is restricted to
the measured foreground, so the stage itself does not zoom or drift. Masks
are supplied observations, not rectangles invented from the face box.
"""
import math

import cv2
import numpy as np

GARMENT_LABELS = (5, 6, 7, 9, 10, 12)


def phases(progress):
    if not math.isfinite(progress) or not 0 <= progress <= 1:
        raise ValueError('transition progress must be in [0,1]')
    def smooth(a, b):
        v = float(np.clip((progress-a)/(b-a), 0., 1.))
        return v*v*(3.-2.*v)
    return smooth(.02, .72), smooth(.78, 1.)


def similarity(source_points, target_points):
    """Deterministic, orientation-preserving fit of actual matching landmarks."""
    a, b = np.asarray(source_points, float), np.asarray(target_points, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or len(a) < 3 or not np.isfinite([a, b]).all():
        raise ValueError('three or more finite corresponding landmarks required')
    ac, bc = a.mean(0), b.mean(0)
    aa, bb = a-ac, b-bc
    denominator = float(np.sum(aa*aa))
    if denominator < 1e-8:
        raise ValueError('degenerate measured landmarks')
    u, values, vt = np.linalg.svd(aa.T@bb)
    orientation = np.ones(2)
    orientation[-1] = np.linalg.det(u@vt)
    rotation = u@np.diag(orientation)@vt
    scale = float(np.sum(values*orientation)/denominator)
    matrix = scale*rotation.T
    return np.column_stack((matrix, bc-matrix@ac)).astype(np.float32)


def target_parts(labels, *, instance_mask):
    """Intersect measured clothes with a separately selected person instance.

    No person/garment silhouette is inferred from a face rectangle. Selection
    of the actual target instance belongs to the caller, not this compositor.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.dtype != np.uint8 or np.any(labels > 19):
        raise ValueError('actual SCHP LIP label raster required')
    instance = np.asarray(instance_mask, np.float32)
    if instance.shape != labels.shape or not np.isfinite(instance).all() or instance.min() < 0 or instance.max() > 1:
        raise ValueError('actual target instance mask must match the semantic raster')
    garment = instance*np.isin(labels, GARMENT_LABELS)
    return garment.astype(np.float32), instance


def _warp(value, matrix, size):
    return cv2.warpAffine(value, matrix, size, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101)


def composite(outgoing, incoming, *, incoming_garment,
        outgoing_person, incoming_person, incoming_to_outgoing, progress):
    """One real paired instant; duration/audio and neighbouring shots untouched."""
    if (outgoing.dtype != np.uint8 or incoming.dtype != np.uint8
            or outgoing.shape != incoming.shape or outgoing.ndim != 3 or outgoing.shape[2] != 3):
        raise ValueError('matching BGR8 pictures required')
    h, w = outgoing.shape[:2]
    masks = (incoming_garment, outgoing_person, incoming_person)
    if any(v.shape != (h, w) or not np.isfinite(v).all() or v.min() < 0 or v.max() > 1 for v in masks):
        raise ValueError('finite source-bound garment and person masks required')
    matrix = np.asarray(incoming_to_outgoing, np.float32)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all() or np.linalg.det(matrix[:, :2]) <= 0:
        raise ValueError('orientation-preserving registration required')
    garment_phase, background_phase = phases(progress)
    if progress == 0:
        return outgoing.copy()
    if progress == 1:
        return incoming.copy()
    identity = np.eye(2, 3, dtype=np.float32)
    a_matrix = identity+(cv2.invertAffineTransform(matrix)-identity)*background_phase
    b_matrix = identity+(matrix-identity)*(1-background_phase)
    prepared = []
    sigma = max(.6, min(w, h)*.003)
    for picture, person, warp in zip((outgoing, incoming), masks[1:],
            (a_matrix, b_matrix), strict=True):
        moved = _warp(person.astype(np.float32), warp, (w, h))
        replacement = cv2.GaussianBlur(np.maximum(person, moved), (0, 0), sigma)[..., None]
        registered = _warp(picture, warp, (w, h))
        prepared.append(picture.astype(np.float32)*(1-replacement)+registered.astype(np.float32)*replacement)
    # Only incoming GARMENT pixels may arrive early. The union with outgoing
    # clothes would import the new stage into a vanished skirt/coat silhouette.
    clothes = cv2.GaussianBlur(_warp(incoming_garment.astype(np.float32),b_matrix,(w,h)), (0, 0), sigma)
    alpha = (background_phase+(garment_phase-background_phase)*clothes)[..., None]
    return np.rint(np.clip(prepared[0]*(1-alpha)+prepared[1]*alpha, 0, 255)).astype(np.uint8)
