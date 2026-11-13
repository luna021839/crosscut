"""Pair-closed person-rectangle REQUEST reservation, not observations.

A reserved key is a planned forward request. An accepted pair has both
endpoints either already cached or planned; that is NOT successful pixel
inference, frame validity, a known extent or a completed bank. Cached empty
detection attempts count as attempt coverage only. No source or cut quotas.
"""
from __future__ import annotations

from numbers import Integral


def _ordinal(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError('physical keys need nonnegative integers, not bool')
    value = int(value)
    if value < 0:
        raise ValueError('physical keys need nonnegative integers')
    return value


def _key(key):
    if not isinstance(key, (tuple, list)) or len(key) != 2:
        raise ValueError('physical key (source, native ordinal) required')
    return _ordinal(key[0]), _ordinal(key[1])


def reserve_complete_pairs(pairs, available, maximum_new_frames=256):
    """Reserve missing endpoints pair-by-pair in the given priority order.

    A pair reserves only if ALL its currently missing endpoints fit the
    remaining global allowance; shared frames count once and no orphan half
    is reserved. Iteration never stops early: a later shared or one-miss pair
    can still fit. A deferred pair can never become closed later, because
    closing it would cost more than the allowance left at its turn.
    """
    if (isinstance(maximum_new_frames, bool) or not isinstance(maximum_new_frames, Integral)
            or maximum_new_frames < 0):
        raise ValueError('nonnegative integer global forward allowance required')
    budget = int(maximum_new_frames)
    try:
        ordered = list(pairs)
        cached_items = list(available)
    except TypeError as error:
        raise ValueError('iterable pairs and available keys required') from error
    normalized = []
    for pair in ordered:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError('each pair needs exactly two physical keys')
        normalized.append((_key(pair[0]), _key(pair[1])))
    cached = {_key(key) for key in cached_items}
    reserved, accepted, deferred, complete = set(), [], [], 0
    for index, (left, right) in enumerate(normalized):
        endpoints = {left, right}
        if endpoints <= cached:
            complete += 1
            accepted.append(index)
            continue
        missing = endpoints - cached - reserved
        if len(missing) <= budget - len(reserved):
            reserved |= missing
            accepted.append(index)
        else:
            deferred.append(index)
    frames = sorted(reserved)
    return dict(new_frames=frames, accepted_pair_indices=accepted, deferred_pair_indices=deferred,
        new_frame_count=len(frames), already_cached_pair_count=complete,
        pixel_inference_or_bank_completeness_certified=False)
