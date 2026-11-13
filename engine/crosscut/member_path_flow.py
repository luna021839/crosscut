"""Exact source-to-edge flow for the member path MILPs.

One binary source is selected per span. At each boundary, the sum of outgoing
edges for a source equals its left-span selection, and the sum of incoming
edges equals its right-span selection. There is at most one edge per ordered
source pair, so these equalities force every edge to 0 or 1 without making the
edges integer variables. Missing edges remain forbidden, not filled in.

The old per-edge upper bounds have the same integer paths but a much weaker LP
relaxation: fractional source mass can be reused by multiple outgoing edges.
These flow equalities prevent that duplication without changing path policy,
edge costs, exact clocks, exposure/cadence bounds or visual eligibility.
"""
from collections.abc import Iterable, Sequence


def member_path_flow_rows(
    edge_specs: Sequence[Sequence],
    *,
    source_count: int,
    boundary_count: int,
    edge_base: int,
) -> Iterable[tuple[tuple[int, float], ...]]:
    """Yield rows with lower=upper=0; edge specs begin (boundary, left, right).

    Source variable (span, source) is span*source_count+source. Edge variables
    follow edge_specs order starting at edge_base. The callers own one-hot
    source rows and nonnegative edge bounds. Reject parallel edges: otherwise
    an integer source path could split its mass between incompatible effects.
    """
    outgoing = [[[] for _ in range(source_count)] for _ in range(boundary_count)]
    incoming = [[[] for _ in range(source_count)] for _ in range(boundary_count)]
    seen = set()
    for index, spec in enumerate(edge_specs):
        boundary, left, right = spec[:3]
        if (not 0 <= boundary < boundary_count or not 0 <= left < source_count
                or not 0 <= right < source_count):
            raise ValueError('member path edge is outside the source/span domain')
        key = (boundary, left, right)
        if key in seen:
            raise ValueError('member path flow requires one edge per ordered source pair')
        seen.add(key)
        outgoing[boundary][left].append((edge_base + index, 1.0))
        incoming[boundary][right].append((edge_base + index, 1.0))
    for boundary in range(boundary_count):
        for source in range(source_count):
            yield tuple(outgoing[boundary][source]) + ((boundary*source_count + source, -1.0),)
            yield tuple(incoming[boundary][source]) + (((boundary+1)*source_count + source, -1.0),)
