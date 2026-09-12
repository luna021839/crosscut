"""Shared required-effect handle admission and structured authoring failures.

Neither selection nor rendering may call an unrenderable required effect a
hard cut. This module has no decoder or selector dependencies, so both sides
can consume the same measurements without circular imports or extra analysis.
"""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping, Sequence


MINIMUM_MEMBER_BRIDGE_HANDLE_QUALITY = 0.68


def member_bridge_handle_quality_failures(
    qualities: Sequence[float | None], *, label: str,
    minimum_quality: float = MINIMUM_MEMBER_BRIDGE_HANDLE_QUALITY,
) -> tuple[str, ...]:
    if label not in {'outgoing', 'incoming'}:
        raise ValueError('bridge handle label is invalid')
    if (isinstance(minimum_quality, bool) or not isinstance(minimum_quality, (float, int))
            or not math.isfinite(minimum_quality) or not 0 <= minimum_quality <= 1):
        raise ValueError('bridge handle quality floor is invalid')
    if len(qualities) != 2:
        return (f'cinematic_bridge_{label}_future_samples_incomplete',)
    if any(isinstance(q, bool) or not isinstance(q, (float, int))
           or not math.isfinite(q) or not minimum_quality <= q <= 1.
           for q in qualities):
        return (f'cinematic_bridge_{label}_future_quality_too_low',)
    return ()


class RequiredMemberBridgeAuthoringError(RuntimeError):
    """An exact selected effect edge failed; never silently downgrade it."""

    def __init__(self, failures: Mapping[int, Sequence[str]], *,
                 edges_by_boundary: Mapping[int, tuple[float, int, int]] | None = None,
                 partition_unit_counts: Sequence[int] = ()) -> None:
        normalized = {}
        for index, reasons in failures.items():
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError('failed effect boundary index is invalid')
            values = tuple(dict.fromkeys(reasons))
            if not values or any(not isinstance(r, str) or not r for r in values):
                raise ValueError('failed effect reasons must be explicit')
            normalized[index] = values
        if not normalized:
            raise ValueError('required bridge failure cannot be empty')
        edges = dict(edges_by_boundary or {})
        if edges:
            if set(edges) != set(normalized):
                raise ValueError('effect failures must bind exactly to rejected edges')
            for seconds, left, right in edges.values():
                if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                        or not math.isfinite(seconds) or seconds < 0
                        or any(isinstance(i, bool) or not isinstance(i, int) or i < 0
                               for i in (left, right)) or left == right):
                    raise ValueError('rejected effect edge is invalid')
        counts = tuple(partition_unit_counts)
        if counts and any(isinstance(c, bool) or not isinstance(c, int) or c <= 0 for c in counts):
            raise ValueError('effect feedback partition units are invalid')
        if bool(edges) != bool(counts):
            raise ValueError('effect feedback needs both exact edges and partition')
        self.failures = MappingProxyType(normalized)
        self.edges_by_boundary = MappingProxyType(edges)
        self.partition_unit_counts = counts
        super().__init__('required direct member cinematic bridge could not be authored: '
            + '; '.join(f"boundary {i}: {','.join(reasons)}" for i,reasons in sorted(normalized.items())))

    def with_path(self, *, edges_by_boundary: Mapping[int, tuple[float, int, int]],
                  partition_unit_counts: Sequence[int]) -> RequiredMemberBridgeAuthoringError:
        return type(self)(self.failures, edges_by_boundary=edges_by_boundary,
                          partition_unit_counts=partition_unit_counts)

    def to_dict(self) -> dict:
        return {'contract':'required_member_effect_authoring_failure_v1',
                'failures':[{'boundary_index':i,'reason_codes':list(reasons),
                             'edge':list(self.edges_by_boundary[i]) if i in self.edges_by_boundary else None}
                            for i,reasons in sorted(self.failures.items())],
                'partition_unit_counts':list(self.partition_unit_counts),
                'hard_cut_fallback_authorized':False}
