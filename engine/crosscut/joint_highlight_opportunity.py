"""Soft highlight preference, distinct from boundary/decoded quality proof."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from .timeline import ShotType, TimelineCandidate


def exact_nonwide_pair(outgoing: TimelineCandidate, incoming: TimelineCandidate) -> bool:
    """Use explicit transformed boundary classes, never midpoint fallback."""
    nonwide = {ShotType.MEDIUM, ShotType.CLOSE_UP}
    return (
        outgoing.source_id != incoming.source_id
        and outgoing.exit_shot_type in nonwide
        and incoming.entry_shot_type in nonwide
    )


def redistribute_highlight_switch_budget(
    candidates_by_segment: Sequence[Sequence[TimelineCandidate]],
    *,
    existing_bonuses: Mapping[int, float],
    section_targets: Sequence[Mapping[str, object]],
    section_index_at,
    maximum_nominal_shot_seconds: float,
) -> tuple[dict[int, float], dict[int, float], dict[str, object]]:
    """Move each highlight's existing budget to short adjacent opportunities.

    Budget is not multiplied by the number of source pairs/options. The DP
    applies the conditional part only after exact transformed boundary
    measurements. Nominal duration is a search preference, not evidence of
    eventual visible-run length or a qualified burst. Final QA still decides.
    """
    if not math.isfinite(maximum_nominal_shot_seconds) or maximum_nominal_shot_seconds <= 0:
        raise ValueError("highlight nominal shot duration must be positive")
    ordinary = dict(existing_bonuses)
    conditional: dict[int, float] = {}
    sections = []
    lattice = tuple(tuple(row) for row in candidates_by_segment)
    for target in section_targets:
        section = target['section_index']
        references = [row[0] for row in lattice if section_index_at(row[0].start_seconds) == section]
        if len(references) < 2:
            continue
        destinations = [right.segment_index for left, right in zip(references, references[1:])
                        if abs(left.end_seconds - right.start_seconds) <= 1e-7
                        and left.end_seconds - left.start_seconds <= maximum_nominal_shot_seconds + 1e-9
                        and right.end_seconds - right.start_seconds <= maximum_nominal_shot_seconds + 1e-9]
        indices = [c.segment_index for c in references[1:]]
        budget = sum(ordinary.get(i, 0.0) for i in indices)
        if not destinations or budget <= 0:
            sections.append(dict(section_index=section, applied=False, reason='no_short_adjacent_opportunity'))
            continue
        # Explicit zero suppresses the generic source-switch override/default
        # inside this section. Ordinary non-highlight scoring is unchanged.
        for index in indices:
            ordinary[index] = 0.0
        for index in destinations:
            conditional[index] = budget / len(destinations)
        sections.append(dict(section_index=section, applied=True,
                             original_bonus_budget=budget,
                             conditional_bonus_budget=sum(conditional[i] for i in destinations),
                             opportunity_segment_indices=destinations))
    return ordinary, conditional, dict(
        contract='exact_transformed_nonwide_highlight_opportunity_v1',
        sections=sections, bonus_requires_explicit_transformed_boundary_classes=True,
        midpoint_class_fallback=False, qualified_burst_proof=False,
        maximum_nominal_shot_seconds=maximum_nominal_shot_seconds,
    )
