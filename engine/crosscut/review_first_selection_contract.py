"""Fail-closed contract for one bounded review-first GROUP candidate.

The ordinary and learned production paths may compare several decoded proxies.
Review-first intentionally renders one complete full-song candidate instead.
That is a different release cohort, not permission to serialize an arbitrary
candidate count.  The independent post-render evaluator only lowers its
required cohort size when both artifacts agree on every invariant below.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy


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


def build_review_first_selection_payload(
    decoded_selection: Mapping[str, object],
    *,
    complete_candidate: Mapping[str, object],
    selected_path_authority: str,
) -> dict[str, object]:
    """The actual singleton producer, shared by pipeline and contract tests."""
    payload = deepcopy(dict(decoded_selection))
    candidates = _sequence(payload.get("candidates"))
    candidate = candidates[0] if candidates and len(candidates) == 1 else None
    review = candidate.get("review") if isinstance(candidate, Mapping) else None
    if (selected_path_authority not in REVIEW_FIRST_ALLOWED_AUTHORITIES
            or not isinstance(candidate, Mapping) or not isinstance(review, Mapping)
            or type(candidate.get("candidate_index")) is not int or candidate.get("candidate_index") != 0
            or candidate.get("selected") is not True or candidate.get("error") is not None
            or type(complete_candidate.get("candidate_index")) is not int or complete_candidate.get("candidate_index") != 0
            or not isinstance(review.get("passed"), bool)
            or payload.get("passed") is not review.get("passed")
            or any(type(payload.get(key)) is not int or payload.get(key) != 1
                   for key in ("requested_review_count", "reviewed_candidate_count", "minimum_required_candidate_count", "successful_review_count"))):
        raise ValueError("review-first producer needs exactly one complete decoded candidate")
    payload.update({
        "analysis_basis": REVIEW_FIRST_CANDIDATE_POOL,
        "release_policy": review_first_release_policy_payload(),
        "candidate_pool": REVIEW_FIRST_CANDIDATE_POOL,
        "selected_path_authority": selected_path_authority,
        "complete_candidates": [deepcopy(dict(complete_candidate))],
        "complete_candidate_count": 1,
        "exact_successful_decoded_review_count_satisfied": True,
        "exact_complete_candidate_count_satisfied": True,
        "final_selected_review": deepcopy(dict(review)),
        "final_selected_version": "original",
        "proxy_triggered_partial_reedit": {
            "attempted": False, "improved": False, "selected_version": "original",
            "changed_boundary_indices": [], "reason": "bounded_review_first_singleton",
        },
        "approval_deferred_to_final_post_render_qa": True,
        "delivery_completion_message_allowed": False,
    })
    return payload


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
        all(type(selection.get(key)) is int for key in (
            "requested_review_count", "reviewed_candidate_count", "minimum_required_candidate_count",
            "complete_candidate_count", "successful_review_count", "selected_candidate_index"))
        and
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
        and type(candidate.get("candidate_index")) is int
        and candidate.get("candidate_index") == 0
        and candidate.get("selected") is True
        and complete is not None
        and type(complete.get("candidate_index")) is int
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
    "build_review_first_selection_payload",
]
