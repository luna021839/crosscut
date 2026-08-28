"""Decoded-output preflight for automatic direct-member edits.

The source-domain selector and dense camera solver cannot observe pixels after
FFmpeg has applied aspect fill, editorial transforms, local look ramps, and
overlap effects.  This module reuses the production renderer at the final QA's
480-pixel analysis width, combines the generic perceptual proxy review with
the exact member pose/motion continuity gate, and exposes boundary indices for
bounded automatic re-selection before an expensive delivery render starts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .direct_member_rendered_continuity import (
    DirectMemberRenderedContinuityResult,
    evaluate_direct_member_rendered_continuity,
)
from .render import RenderPlan
from .rendered_proxy_review import (
    RenderedProxyReview,
    render_and_review_proxy,
)


ANALYSIS_BASIS = "automatic_direct_member_decoded_output_proxy_preflight_v1"


@dataclass(frozen=True, slots=True)
class DirectMemberRenderedProxyPreflight:
    payload: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.payload.get("passed"))

    @property
    def repair_target_boundary_indices(self) -> tuple[int, ...]:
        values = self.payload.get("repair_target_boundary_indices", [])
        if not isinstance(values, list):
            raise ValueError("member proxy repair targets are malformed")
        return tuple(int(value) for value in values)

    def refinement_rank(self) -> tuple[int, int, int, int, float, float]:
        """Rank proxy look attempts without accepting a visual regression.

        Look refinement is allowed to improve encoded color consistency, but
        its changed pixels can also make the decoded geometry detectors less
        stable.  Keep the best fully materialized attempt instead of blindly
        retaining the final iteration.  Lower tuples are better.
        """

        quality_passed = bool(
            self.payload.get("member_output_quality_passed")
        )
        continuity_passed = bool(
            self.payload.get("direct_member_rendered_continuity_passed")
        )
        continuity = self.payload.get("rendered_continuity")
        continuity_payload = (
            continuity if isinstance(continuity, Mapping) else {}
        )
        metrics = continuity_payload.get("metrics")
        metric_payload = metrics if isinstance(metrics, Mapping) else {}
        thresholds = continuity_payload.get("thresholds")
        threshold_payload = (
            thresholds if isinstance(thresholds, Mapping) else {}
        )

        normalized_excess = 0.0
        for metric_name, threshold_name in (
            (
                "attention_position_mismatch_p95",
                "maximum_attention_position_mismatch_p95",
            ),
            (
                "attention_scale_mismatch_p95",
                "maximum_attention_scale_mismatch_p95",
            ),
            ("pose_mismatch_p95", "maximum_pose_mismatch_p95"),
            (
                "silhouette_mismatch_p95",
                "maximum_silhouette_mismatch_p95",
            ),
        ):
            value = metric_payload.get(metric_name)
            maximum = threshold_payload.get(threshold_name)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isinstance(maximum, (int, float))
                and not isinstance(maximum, bool)
                and float(maximum) > 0.0
            ):
                normalized_excess += max(
                    0.0,
                    float(value) / float(maximum) - 1.0,
                )

        generic = self.payload.get("rendered_proxy_review")
        generic_payload = generic if isinstance(generic, Mapping) else {}
        weighted_score = generic_payload.get("weighted_score")
        resolved_weighted_score = (
            float(weighted_score)
            if isinstance(weighted_score, (int, float))
            and not isinstance(weighted_score, bool)
            else float("-inf")
        )
        quality_failures = self.payload.get(
            "member_output_quality_failure_reasons"
        )
        quality_failure_count = (
            len(quality_failures) if isinstance(quality_failures, list) else 1
        )
        component_failure_count = int(not quality_passed) + int(
            not continuity_passed
        )
        return (
            int(not self.passed),
            component_failure_count,
            len(self.repair_target_boundary_indices),
            quality_failure_count,
            normalized_excess,
            -resolved_weighted_score,
        )

    def write(self, path: Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(dict(self.payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination

    def ranked_reselection_boundary_indices(
        self,
        *,
        maximum_count: int = 4,
    ) -> tuple[int, ...]:
        if (
            isinstance(maximum_count, bool)
            or not isinstance(maximum_count, int)
            or maximum_count < 1
        ):
            raise ValueError("member proxy reselection count must be positive")
        continuity = self.payload.get("rendered_continuity")
        if not isinstance(continuity, Mapping):
            raise ValueError("member proxy continuity payload is missing")
        targets = set(
            _integer_indices(
                self.payload.get("repair_target_boundary_indices")
            )
        )
        continuity_targets = set(
            _integer_indices(
                self.payload.get(
                    "continuity_repair_target_boundary_indices"
                )
            )
        )
        generic_targets = set(
            _integer_indices(
                self.payload.get("generic_repair_target_boundary_indices")
            )
        )
        raw_failures = continuity.get("failure_boundary_indices")
        failures_by_boundary: dict[int, set[str]] = {}
        if isinstance(raw_failures, Mapping):
            for reason, indices in raw_failures.items():
                if not isinstance(reason, str):
                    continue
                for boundary_index in _integer_indices(indices):
                    failures_by_boundary.setdefault(boundary_index, set()).add(
                        reason
                    )
        thresholds = continuity.get("thresholds")
        limits = thresholds if isinstance(thresholds, Mapping) else {}
        records = continuity.get("boundaries")
        records_by_index = {
            int(record["boundary_index"]): record
            for record in records
            if isinstance(record, Mapping)
            and isinstance(record.get("boundary_index"), int)
            and not isinstance(record.get("boundary_index"), bool)
        } if isinstance(records, list) else {}

        generic = self.payload.get("rendered_proxy_review")
        generic_payload = generic if isinstance(generic, Mapping) else {}
        admission = generic_payload.get("perceptual_source_admission")
        admission_payload = admission if isinstance(admission, Mapping) else {}
        rapid = admission_payload.get("rapid_quality_alternation")
        rapid_payload = rapid if isinstance(rapid, Mapping) else {}
        rapid_records = rapid_payload.get("boundaries")
        rapid_shock_by_index = {
            int(record["boundary_index"]): float(
                record["normalized_quality_shock"]
            )
            for record in rapid_records
            if isinstance(record, Mapping)
            and isinstance(record.get("boundary_index"), int)
            and not isinstance(record.get("boundary_index"), bool)
            and isinstance(
                record.get("normalized_quality_shock"),
                (int, float),
            )
            and not isinstance(
                record.get("normalized_quality_shock"),
                bool,
            )
        } if isinstance(rapid_records, list) else {}

        generic_records = generic_payload.get("boundaries")
        generic_records_by_index = {
            int(record["boundary_index"]): record
            for record in generic_records
            if isinstance(record, Mapping)
            and isinstance(record.get("boundary_index"), int)
            and not isinstance(record.get("boundary_index"), bool)
        } if isinstance(generic_records, list) else {}

        source_medians = admission_payload.get("source_median_consistency")
        source_median_payload = (
            source_medians if isinstance(source_medians, Mapping) else {}
        )
        failing_quality_source_indices = set(
            _integer_indices(
                source_median_payload.get("insufficient_source_indices")
            )
        )
        worst_pairs = source_median_payload.get("worst_source_pairs")
        if isinstance(worst_pairs, Mapping):
            for raw_pair in worst_pairs.values():
                if not isinstance(raw_pair, Mapping):
                    continue
                value = raw_pair.get("value")
                maximum = raw_pair.get("maximum_allowed")
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and isinstance(maximum, (int, float))
                    and not isinstance(maximum, bool)
                    and float(value) > float(maximum) + 1e-9
                ):
                    failing_quality_source_indices.update(
                        _integer_indices(raw_pair.get("source_indices"))
                    )

        def ratio(record: Mapping[str, Any], field: str, limit: str) -> float:
            value = record.get(field)
            maximum = limits.get(limit)
            if not isinstance(value, (int, float)) or not isinstance(
                maximum, (int, float)
            ) or float(maximum) <= 0.0:
                return 0.0
            return float(value) / float(maximum)

        ranked: list[tuple[float, int]] = []
        for boundary_index in targets:
            record = records_by_index.get(boundary_index, {})
            reasons = failures_by_boundary.get(boundary_index, set())
            normalized = max(
                ratio(
                    record,
                    "attention_position_mismatch",
                    "maximum_attention_position_mismatch_p95",
                ),
                ratio(
                    record,
                    "attention_scale_mismatch",
                    "maximum_attention_scale_mismatch_p95",
                ),
                ratio(
                    record,
                    "pose_mismatch",
                    "maximum_pose_mismatch_p95",
                ),
                ratio(
                    record,
                    "silhouette_mismatch",
                    "maximum_silhouette_mismatch_p95",
                ),
            )
            geometry_failures = sum(
                any(token in reason for token in ("position", "scale", "pose", "silhouette"))
                for reason in reasons
            )
            motion_failures = sum("motion" in reason for reason in reasons)
            effect_failures = sum("effect_endpoint" in reason for reason in reasons)
            generic_record = generic_records_by_index.get(
                boundary_index,
                {},
            )
            quality_sources = {
                int(value)
                for value in (
                    generic_record.get("outgoing_source_index"),
                    generic_record.get("incoming_source_index"),
                )
                if isinstance(value, int) and not isinstance(value, bool)
            }
            failing_quality_source_count = len(
                quality_sources & failing_quality_source_indices
            )
            technical_reasons = generic_record.get(
                "technical_integrity_issue_reasons"
            )
            technical_failure_count = (
                len(technical_reasons)
                if isinstance(technical_reasons, list)
                else 0
            )
            score = (
                normalized
                + 1.25 * geometry_failures
                + 0.85 * motion_failures
                + 1.0 * effect_failures
                + 0.35 * int(boundary_index in continuity_targets)
                + 0.35 * int(boundary_index in generic_targets)
                + 0.80 * min(
                    3.0,
                    rapid_shock_by_index.get(boundary_index, 0.0),
                )
                + 0.55 * failing_quality_source_count
                + 0.75 * technical_failure_count
            )
            ranked.append((score, boundary_index))
        return tuple(
            boundary_index
            for _score, boundary_index in sorted(
                ranked,
                key=lambda item: (-item[0], item[1]),
            )[:maximum_count]
        )

    def ranked_continuity_reselection_boundary_indices(
        self,
        *,
        maximum_count: int = 4,
    ) -> tuple[int, ...]:
        """Compatibility alias for the former continuity-only ranker."""

        return self.ranked_reselection_boundary_indices(
            maximum_count=maximum_count
        )


def _integer_indices(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        int(item)
        for item in value
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0
    )


def _generic_repair_targets(payload: Mapping[str, Any]) -> tuple[int, ...]:
    admission = payload.get("perceptual_source_admission")
    perceptual: tuple[int, ...] = ()
    if isinstance(admission, Mapping):
        decision = admission.get("admission")
        if isinstance(decision, Mapping):
            perceptual = _integer_indices(
                decision.get("repair_target_boundary_indices")
            )
    technical = tuple(
        int(boundary["boundary_index"])
        for boundary in payload.get("boundaries", [])
        if isinstance(boundary, Mapping)
        and isinstance(boundary.get("boundary_index"), int)
        and not isinstance(boundary.get("boundary_index"), bool)
        and bool(boundary.get("technical_integrity_issue_reasons"))
    )
    return tuple(sorted(set(perceptual) | set(technical)))


def _member_output_quality_failures(
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    policy = payload.get("review_policy")
    overall = payload.get("overall_quality")
    source_quality = payload.get("source_quality")
    admission = payload.get("perceptual_source_admission")
    domains = payload.get("boundary_domains")
    if not all(
        isinstance(value, Mapping)
        for value in (policy, overall, source_quality, admission, domains)
    ):
        return ("member_proxy_quality_contract_incomplete",)
    assert isinstance(policy, Mapping)
    assert isinstance(overall, Mapping)
    assert isinstance(source_quality, Mapping)
    assert isinstance(admission, Mapping)
    assert isinstance(domains, Mapping)
    failures: list[str] = []
    quality_p10 = overall.get("p10")
    minimum_quality = policy.get("minimum_overall_quality_p10")
    if not isinstance(quality_p10, (int, float)) or not isinstance(
        minimum_quality, (int, float)
    ) or float(quality_p10) + 1e-9 < float(minimum_quality):
        failures.append("member_proxy_overall_quality_p10")
    median_span = source_quality.get("median_span")
    maximum_span = policy.get("maximum_source_quality_median_span")
    if not isinstance(median_span, (int, float)) or not isinstance(
        maximum_span, (int, float)
    ) or float(median_span) > float(maximum_span) + 1e-9:
        failures.append("member_proxy_source_quality_median_span")
    admission_decision = admission.get("admission")
    if not isinstance(admission_decision, Mapping) or not bool(
        admission_decision.get("passed")
    ):
        failures.append("member_proxy_perceptual_source_admission")
    for domain in ("tier_d_continuity", "native_source_cut"):
        domain_payload = domains.get(domain)
        if not isinstance(domain_payload, Mapping) or not bool(
            domain_payload.get("passed")
        ):
            failures.append(f"member_proxy_{domain}_technical_integrity")
    if payload.get("introduced_cross_source_technical_issue_reasons"):
        failures.append("member_proxy_introduced_technical_integrity")
    failure_count = payload.get("geometry_provider_failure_count")
    if not isinstance(failure_count, int) or isinstance(
        failure_count, bool
    ) or failure_count != 0:
        failures.append("member_proxy_geometry_provider_failure")
    return tuple(failures)


def _continuity_repair_targets(
    result: DirectMemberRenderedContinuityResult,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(boundary_index)
                for indices in result.failure_boundary_indices.values()
                for boundary_index in indices
            }
        )
    )


def render_and_review_direct_member_proxy(
    plan: RenderPlan,
    proxy_output: Path,
    *,
    render_review: Callable[..., tuple[RenderPlan, RenderedProxyReview]]
    | None = None,
    continuity_evaluator: Callable[..., DirectMemberRenderedContinuityResult]
    | None = None,
) -> tuple[RenderPlan, DirectMemberRenderedProxyPreflight]:
    """Materialize and gate one exact low-resolution direct-member plan."""

    member_fields = (
        plan.member_pose_dissolve_requested,
        plan.member_pose_dissolve_effective,
        plan.member_pose_dissolve_count,
    )
    if not all(value is not None for value in member_fields):
        raise ValueError("direct-member proxy requires complete member metadata")
    review_function = render_review or render_and_review_proxy
    continuity_function = (
        continuity_evaluator or evaluate_direct_member_rendered_continuity
    )
    proxy_plan, generic_review = review_function(
        plan,
        Path(proxy_output),
        video_codec="h264_videotoolbox",
    )
    continuity = continuity_function(
        proxy_plan.output,
        proxy_plan.to_json(),
    )
    generic_payload = deepcopy(dict(generic_review.payload))
    generic_targets = _generic_repair_targets(generic_payload)
    continuity_targets = _continuity_repair_targets(continuity)
    repair_targets = tuple(
        sorted(set(generic_targets) | set(continuity_targets))
    )
    generic_passed = bool(generic_payload.get("passed"))
    member_quality_failures = _member_output_quality_failures(generic_payload)
    member_quality_passed = not member_quality_failures
    payload = {
        "schema_version": 1,
        "analysis": ANALYSIS_BASIS,
        "passed": bool(member_quality_passed and continuity.passed),
        "generic_rendered_proxy_passed": generic_passed,
        "member_output_quality_passed": member_quality_passed,
        "member_output_quality_failure_reasons": list(
            member_quality_failures
        ),
        "direct_member_rendered_continuity_passed": continuity.passed,
        "repair_target_boundary_indices": list(repair_targets),
        "generic_repair_target_boundary_indices": list(generic_targets),
        "continuity_repair_target_boundary_indices": list(continuity_targets),
        "rendered_proxy_review": generic_payload,
        "rendered_continuity": continuity.to_dict(),
        "authority": {
            "pixels": "production_filter_graph_low_resolution_encoded_output",
            "analysis_width_matches_final_member_gate": True,
            "source_domain_scores_are_sufficient": False,
            "generic_group_editorial_score_is_member_gate": False,
            "member_editorial_gate": "exact_pose_silhouette_attention_motion",
            "full_resolution_render_allowed_only_after_pass": True,
            "failure_effect": (
                "bounded_source_pair_or_cut_time_reselection_then_proxy_rerender"
            ),
        },
    }
    return (proxy_plan, DirectMemberRenderedProxyPreflight(payload))


__all__ = [
    "ANALYSIS_BASIS",
    "DirectMemberRenderedProxyPreflight",
    "render_and_review_direct_member_proxy",
]
