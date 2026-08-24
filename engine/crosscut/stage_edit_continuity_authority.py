"""Shared authority names for StageEdit boundary-continuity scoring."""

from __future__ import annotations


CONTINUITY_AUTHORITY_CONTRACT = (
    "explicit_single_boundary_continuity_score_authority_v1"
)
HANDCRAFTED_CONTINUITY_AUTHORITY = "handcrafted_boundary_continuity_v1"
PRODUCTION_LEARNED_CONTINUITY_AUTHORITY = (
    "production_learned_source_pair_checkpoint_v1"
)
STAGE_EDIT_CONTINUITY_AUTHORITIES = frozenset(
    {
        HANDCRAFTED_CONTINUITY_AUTHORITY,
        PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
    }
)


__all__ = [
    "CONTINUITY_AUTHORITY_CONTRACT",
    "HANDCRAFTED_CONTINUITY_AUTHORITY",
    "PRODUCTION_LEARNED_CONTINUITY_AUTHORITY",
    "STAGE_EDIT_CONTINUITY_AUTHORITIES",
]
