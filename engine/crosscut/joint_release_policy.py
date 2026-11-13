"""Separate edit alternatives from the quality requirements of one deliverable.

Neither value is an input-video/source-coverage threshold. Every admitted edit
still needs its exact decoded boundary proofs and its full post-render review.
"""

JOINT_RELEASE_CANDIDATE_COUNT = 1
JOINT_COMPARISON_CANDIDATE_BUDGET = 8


def joint_release_policy_payload() -> dict[str, object]:
    return {
        "contract": "joint_quality_gated_single_delivery_v1",
        "minimum_passing_candidate_count": JOINT_RELEASE_CANDIDATE_COUNT,
        "comparison_candidate_budget": JOINT_COMPARISON_CANDIDATE_BUDGET,
        "source_coverage_requirements_changed": False,
        "individual_candidate_quality_requirements_changed": False,
    }
