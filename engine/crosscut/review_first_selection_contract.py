"""Fail-closed contract for one bounded review-first GROUP candidate.

The ordinary and learned production paths may compare several decoded proxies.
Review-first intentionally renders one complete full-song candidate instead.
That is a different release cohort, not permission to serialize an arbitrary
candidate count.  The independent post-render evaluator only lowers its
required cohort size when both artifacts agree on every invariant below.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


REVIEW_FIRST_RELEASE_CANDIDATE_COUNT = 1
REVIEW_FIRST_CANDIDATE_POOL = "bounded_review_first_single_full_song_v1"
REVIEW_FIRST_ALLOWED_AUTHORITIES = frozenset(
    {
        "stage_edit_generation",
        "stage_edit_candidate_reranker",
        "deterministic_full_song_visual_selector",
    }
)


def review_first_release_policy_payload() -> dict[str, object]:
    """Return the immutable one-candidate review release policy."""

    return {
        "contract": "bounded_review_first_single_delivery_v1",
        "required_complete_candidate_count": REVIEW_FIRST_RELEASE_CANDIDATE_COUNT,
        "required_decoded_review_count": REVIEW_FIRST_RELEASE_CANDIDATE_COUNT,
        "candidate_pool": REVIEW_FIRST_CANDIDATE_POOL,
        "selection_mode": "one_complete_full_song_candidate_then_final_qa",
        "source_coverage_requirements_changed": False,
        "individual_candidate_quality_requirements_changed": False,
        "silent_unreviewed_fallback_allowed": False,
    }


def _sequence(value: object) -> Sequence[object] | None:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return value
    return None


def review_first_release_binding_valid(
    selection: Mapping[str, object],
    render: Mapping[str, object],
) -> bool:
    """Verify the independently loaded selection and render declarations.

    This deliberately checks structural facts in addition to the policy token.
    A caller cannot opt into the smaller cohort by copying only the policy
    object or by changing a serialized count.
    """

    policy = review_first_release_policy_payload()
    authority = selection.get("selected_path_authority")
    candidates = _sequence(selection.get("candidates"))
    complete_candidates = _sequence(selection.get("complete_candidates"))
    candidate = (
        candidates[0]
        if candidates is not None
        and len(candidates) == REVIEW_FIRST_RELEASE_CANDIDATE_COUNT
        and isinstance(candidates[0], Mapping)
        else None
    )
    complete = (
        complete_candidates[0]
        if complete_candidates is not None
        and len(complete_candidates) == REVIEW_FIRST_RELEASE_CANDIDATE_COUNT
        and isinstance(complete_candidates[0], Mapping)
        else None
    )
    return bool(
        selection.get("release_policy") == policy
        and render.get("release_policy") == policy
        and selection.get("candidate_pool") == REVIEW_FIRST_CANDIDATE_POOL
        and render.get("candidate_pool") == REVIEW_FIRST_CANDIDATE_POOL
        and authority in REVIEW_FIRST_ALLOWED_AUTHORITIES
        and render.get("candidate_origin") == authority
        and selection.get("requested_review_count")
        == REVIEW_FIRST_RELEASE_CANDIDATE_COUNT
        and selection.get("reviewed_candidate_count")
        == REVIEW_FIRST_RELEASE_CANDIDATE_COUNT
        and selection.get("minimum_required_candidate_count")
        == REVIEW_FIRST_RELEASE_CANDIDATE_COUNT
        and selection.get("minimum_candidate_count_satisfied") is True
        and selection.get("complete_candidate_count")
        == REVIEW_FIRST_RELEASE_CANDIDATE_COUNT
        and selection.get("successful_review_count")
        == REVIEW_FIRST_RELEASE_CANDIDATE_COUNT
        and selection.get("exact_complete_candidate_count_satisfied") is True
        and selection.get("exact_successful_decoded_review_count_satisfied") is True
        and candidate is not None
        and candidate.get("candidate_index") == 0
        and candidate.get("selected") is True
        and complete is not None
        and complete.get("candidate_index") == 0
        and selection.get("selected_candidate_index") == 0
        and selection.get("final_selected_version") == "original"
        and isinstance(selection.get("final_selected_review"), Mapping)
    )


__all__ = [
    "REVIEW_FIRST_ALLOWED_AUTHORITIES",
    "REVIEW_FIRST_CANDIDATE_POOL",
    "REVIEW_FIRST_RELEASE_CANDIDATE_COUNT",
    "review_first_release_binding_valid",
    "review_first_release_policy_payload",
]
