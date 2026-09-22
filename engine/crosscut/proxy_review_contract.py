"""Scalar proxy-integrity contract shared by QA producers and disposition.

Editorial scores cannot authorize technical failures. Unknown nested reasons
are retained as blocking issues rather than folded into a generic low score.
"""
from collections.abc import Mapping
from copy import deepcopy

PROXY_INTEGRITY_CONTRACT = "proxy_nested_technical_integrity_v1"
EDITORIAL_BOUNDARY_REASONS = frozenset({
    "sharpness_discontinuity", "exposure_discontinuity", "color_shock",
    "structure_shock", "face_scale_discontinuity", "face_position_discontinuity",
    "silhouette_scale_discontinuity", "silhouette_position_discontinuity",
    "motion_direction_discontinuity", "subject_continuity_unobserved",
    "boundary_measurements_unavailable", "weighted_boundary_score",
})


def proxy_integrity_evidence(review: Mapping) -> dict:
    snapshot = {key: deepcopy(review.get(key)) for key in (
        "schema_version", "geometry_provider_failure_count", "reviewed_boundary_count",
    )}
    # Preserve key absence for legacy-v2; modern producers must report all
    # technical channels explicitly even when there are no failures.
    if "introduced_cross_source_technical_issue_reasons" in review:
        snapshot["introduced_cross_source_technical_issue_reasons"] = deepcopy(
            review["introduced_cross_source_technical_issue_reasons"])
    raw_boundaries = review.get("boundaries")
    snapshot["boundaries"] = [
        {key: deepcopy(item[key]) for key in (
            "boundary_index", "passed", "issue_reasons", "technical_integrity_issue_reasons",
        ) if key in item} if isinstance(item, Mapping) else None
        for item in raw_boundaries
    ] if isinstance(raw_boundaries, list) else None
    issues = _integrity_issues(snapshot)
    return {"contract": PROXY_INTEGRITY_CONTRACT, "review": snapshot,
            "passed": not issues, "technical_issue_codes": issues}


def _integrity_issues(review: Mapping) -> list[str]:
    issues = []
    schema = review.get("schema_version")
    if type(schema) is not int or schema not in (2, 3):
        issues.append("proxy_integrity_schema_invalid")
    failures = review.get("geometry_provider_failure_count")
    if type(failures) is not int or failures < 0:
        issues.append("proxy_integrity_geometry_failure_count_invalid")
    elif failures:
        issues.append("proxy_integrity_geometry_provider_failed")

    def reasons(value, *, path, technical):
        if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
            issues.append(f"proxy_integrity_{path}_malformed")
            return []
        for reason in value:
            if technical or reason not in EDITORIAL_BOUNDARY_REASONS:
                issues.append(f"proxy_integrity_{reason}")
        return value

    key = "introduced_cross_source_technical_issue_reasons"
    if key in review or schema == 3:
        reasons(review.get(key), path="summary_reasons", technical=True)
    boundaries = review.get("boundaries")
    if not isinstance(boundaries, list):
        return list(dict.fromkeys([*issues, "proxy_integrity_boundaries_missing"]))
    count = review.get("reviewed_boundary_count")
    if type(count) is not int or count != len(boundaries):
        issues.append("proxy_integrity_boundary_count_invalid")
    for index, boundary in enumerate(boundaries):
        if not isinstance(boundary, Mapping):
            issues.append("proxy_integrity_boundary_malformed")
            continue
        if type(boundary.get("boundary_index")) is not int or boundary["boundary_index"] != index:
            issues.append("proxy_integrity_boundary_index_invalid")
        if type(boundary.get("passed")) is not bool:
            issues.append("proxy_integrity_boundary_passed_invalid")
        observed = reasons(boundary.get("issue_reasons"), path="boundary_reasons", technical=False)
        if boundary.get("passed") is True and observed:
            issues.append("proxy_integrity_passed_with_reasons")
        if boundary.get("passed") is False and not observed:
            issues.append("proxy_integrity_failed_without_reason")
        if "technical_integrity_issue_reasons" in boundary or schema == 3:
            reasons(boundary.get("technical_integrity_issue_reasons"), path="technical_reasons", technical=True)
    return list(dict.fromkeys(issues))


def proxy_integrity_evidence_valid(value: object) -> bool:
    if not isinstance(value, Mapping) or value.get("contract") != PROXY_INTEGRITY_CONTRACT:
        return False
    review = value.get("review")
    if not isinstance(review, Mapping):
        return False
    issues = _integrity_issues(review)
    return (value.get("passed") is True and not issues
            and value.get("technical_issue_codes") == issues)
