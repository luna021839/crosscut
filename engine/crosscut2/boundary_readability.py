"""Original face-ROI detail at a cut, separate from interval exposure.

No motion threshold, source exclusion, new pixels, or identity inference.
UNKNOWN retains .5; an unreadable exact endpoint cannot disappear in the
average quality of a much longer otherwise readable shot.
"""
import numpy as np
from .scene_readout import roi_readability


def pair(first, second, known_first, known_second):
    a, b, ka, kb = np.broadcast_arrays(first, second, known_first, known_second)
    if (ka.dtype != bool or kb.dtype != bool or not np.isfinite(a).all()
            or not np.isfinite(b).all() or np.any((a < 0) | (a > 1))
            or np.any((b < 0) | (b > 1))):
        raise ValueError('bounded original ROI readability and explicit measurement masks required')
    return 1. - np.minimum(np.where(ka, a, .5), np.where(kb, b, .5)), ka & kb


def native_fields(bank, port):
    """Transfer only identical physical ordinal AND identical original box.

    Extra native-cut faces frequently have no original ROI observation. They
    stay unknown rather than borrowing the neighbouring grid's detail score.
    Native ordinals, when present, are already source-clock-bound by the
    caller; historical producers are neither rewritten nor relabelled here.
    """
    heads = np.asarray(port['heads'])
    shape = heads.shape[:-1]
    values = np.full(shape, .5)
    known = np.zeros(shape, bool)
    if 'native_indices' not in port:
        return values, known
    native = np.asarray(port['native_indices'])
    observed = np.asarray(port.get('observed', np.ones(len(heads), bool)))
    if (heads.ndim != 4 or heads.shape[1:] != (2, 8, 4)
            or native.shape != heads.shape[:2] or native.dtype.kind not in 'iu'
            or observed.shape != (len(heads),) or observed.dtype != bool
            or np.any(native[observed, 1] - native[observed, 0] != 1)):
        raise ValueError('source-bound exact-adjacent native face ordinals required')
    quality, valid = roi_readability(bank)
    records = {}
    ambiguous = set()
    key = lambda ordinal, head: (int(ordinal), np.asarray(head, np.float64).tobytes())
    for n, side, h in zip(*np.nonzero(valid)):
        query = key(bank['native_indices'][n, side], bank['heads'][n, side, h])
        q = float(quality[n, side, h])
        if query in records and records[query] != q:
            ambiguous.add(query)
        records[query] = q
    for n, side, h in np.ndindex(shape):
        if not observed[n]:
            continue
        query = key(native[n, side], heads[n, side, h])
        if query in records and query not in ambiguous:
            values[n, side, h] = records[query]
            known[n, side, h] = True
    return values, known
