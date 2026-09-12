"""Learned edit search breadth is not the number of deliverables required.

This policy never supplies a missing learned path or authorizes media itself.
Every selected edit still needs its own decoded, montage and final output QA.
"""
from collections.abc import Mapping

STAGE_EDIT_RELEASE_CANDIDATE_COUNT = 1
STAGE_EDIT_COMPARISON_CANDIDATE_BUDGET = 8
STAGE_EDIT_SINGLE_DELIVERY_VISUAL_CONTRACT = (
    "quality_gated_stage_edit_visual_dynamic_crop_v2_v3_single_delivery_v1"
)


def stage_edit_release_policy_payload() -> dict[str, object]:
    return {
        "contract": "stage_edit_quality_gated_single_delivery_v1",
        "minimum_passing_candidate_count": STAGE_EDIT_RELEASE_CANDIDATE_COUNT,
        "comparison_candidate_budget": STAGE_EDIT_COMPARISON_CANDIDATE_BUDGET,
        "minimum_visual_passing_learned_candidate_count": 1,
        "comparison_selection": "rank_all_available_decoded_passing_candidates",
        "source_coverage_requirements_changed": False,
        "individual_candidate_quality_requirements_changed": False,
        "silent_baseline_fallback_allowed": False,
    }


def select_stage_edit_delivery_cohort(bundles, reviews, **ranking_arguments):
    """Keep best-of-available comparison, including the predeclared hybrid floor.

    One required deliverable does not mean returning the first passing edit.
    No additional render is performed here; these are completed proxy reviews.
    """
    from .rendered_candidate_selection import select_rendered_candidate_reviews
    bundles, reviews = tuple(bundles), tuple(reviews)
    if not 1 <= len(bundles) == len(reviews) <= STAGE_EDIT_COMPARISON_CANDIDATE_BUDGET:
        raise ValueError("StageEdit delivery comparison requires 1...8 bound reviews")
    by_index = {item.candidate_index: item for item in bundles}
    review_indices = [item.candidate.candidate_index for item in reviews]
    if (len(by_index) != len(bundles) or set(review_indices) != set(by_index)
            or not all(item.passed for item in reviews)):
        raise ValueError("StageEdit delivery comparison requires bound passing candidates")
    result = select_rendered_candidate_reviews(reviews,
        requested_review_count=len(reviews), minimum_required_candidate_count=1,
        **ranking_arguments)
    selected = result.selected
    if not selected.passed:
        raise ValueError("StageEdit delivery comparison selected a failed review")
    return ((by_index[result.selected_candidate_index], selected),), {
        "contract": "stage_edit_best_available_decoded_delivery_v1",
        "comparison_candidate_count": len(reviews),
        "required_delivery_count": 1,
        "selected_candidate_index": result.selected_candidate_index,
        "comparison": result.to_dict(),
    }


def stage_edit_release_binding_valid(selection: Mapping, render: Mapping) -> bool:
    """Validate independent artifact agreement, not arbitrary serialized floors."""
    policy = stage_edit_release_policy_payload()
    batch = selection.get("stage_edit_visual_candidate_batch")
    if not isinstance(batch, Mapping):
        return False
    count, learned, floor = (
        batch.get("candidate_count"), batch.get("learned_candidate_count"),
        batch.get("hybrid_visual_floor_candidate_count"),
    )
    if any(type(value) is not int for value in (count, learned, floor)):
        return False
    candidates = batch.get("candidates")
    pool = (
        "stage_edit_generation_plus_typed_visual_floor_v1"
        if floor else "stage_edit_generation_only"
    )
    return bool(
        selection.get("release_policy") == policy
        and render.get("release_policy") == policy
        and batch.get("release_policy") == policy
        and batch.get("contract") == STAGE_EDIT_SINGLE_DELIVERY_VISUAL_CONTRACT
        and render.get("stage_edit_promotion_contract")
        == STAGE_EDIT_SINGLE_DELIVERY_VISUAL_CONTRACT
        and render.get("candidate_origin") == "explicit_stage_edit_generation"
        and selection.get("candidate_pool") == batch.get("candidate_pool") == pool
        and 1 <= learned <= count <= STAGE_EDIT_COMPARISON_CANDIDATE_BUDGET
        and floor in (0, 1) and learned + floor == count
        and batch.get("required_candidate_count") == 1
        and batch.get("all_prescribed_paths_passed_visual_hard_gates") is True
        and batch.get("silent_baseline_fallback_allowed") is False
        and isinstance(candidates, list) and len(candidates) == count
        and all(isinstance(item, Mapping) for item in candidates)
        and [item.get("candidate_index") for item in candidates] == list(range(count))
        and sum(item.get("candidate_origin") == "stage_edit_semimarkov"
                for item in candidates) == learned
        and sum(item.get("candidate_origin") == "hybrid_visual_floor_dynamic_crop_v3"
                for item in candidates) == floor
        and batch.get("dynamic_crop_version") in ((3,) if floor else (2, 3))
    )
