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
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .direct_member_rendered_continuity import (
    DirectMemberRenderedContinuityResult,
    evaluate_direct_member_rendered_continuity,
)
from .direct_member_target_face import FOCAL_FACE_BASIS, FocalFaceCandidate
from .direct_member_visual import (
    _ATTENTION_FACE_BASIS,
    _ATTENTION_SILHOUETTE_BASIS,
    _scale_log_ratio_evidence,
)
from .render import (
    DIRECT_MEMBER_POSE_DISSOLVE_REASON,
    EDITORIAL_TRANSFORM_MAXIMUM_CENTER_SHIFT,
    EDITORIAL_TRANSFORM_MAXIMUM_ZOOM,
    MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY,
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    TransitionKind,
    segment_frame_counts,
)
from .rendered_proxy_review import (
    RenderedProxyReview,
    render_and_review_proxy,
)


ANALYSIS_BASIS = "automatic_direct_member_decoded_output_proxy_preflight_v1"
CAMERA_REFINEMENT_ANALYSIS_BASIS = (
    "decoded_output_focal_endpoint_camera_inverse_refinement_v1"
)
OUTPUT_VERIFIED_DISSOLVE_ANALYSIS_BASIS = (
    "decoded_output_focal_endpoint_pose_dissolve_promotion_v1"
)
OUTPUT_VERIFIED_DISSOLVE_DEMOTION_ANALYSIS_BASIS = (
    "decoded_output_failed_pose_dissolve_selective_demotion_v1"
)
_OUTPUT_VERIFIED_DISSOLVE_DEMOTION_REASON = (
    "decoded_output_failed_pose_dissolve_hard_cut_v1"
)
_REPAIRABLE_EFFECT_ENDPOINT_FAILURE_REASONS = frozenset(
    {
        "rendered_effect_endpoint_descriptor_missing",
        "rendered_effect_endpoint_attention_missing",
        "rendered_effect_endpoint_position_discontinuity",
        "rendered_effect_endpoint_scale_discontinuity",
        "rendered_effect_endpoint_pose_discontinuity",
        "rendered_effect_endpoint_silhouette_discontinuity",
        "rendered_effect_endpoint_shape_evidence_missing",
    }
)


@dataclass(frozen=True, slots=True)
class _CameraState:
    zoom: float
    center_x: float
    center_y: float
    roll_degrees: float = 0.0

    def keyframe(self, time_seconds: float) -> EditorialTransformKeyframe:
        return EditorialTransformKeyframe(
            time_seconds=time_seconds,
            zoom=self.zoom,
            center_x=self.center_x,
            center_y=self.center_y,
            roll_degrees=self.roll_degrees,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "zoom": self.zoom,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "roll_degrees": self.roll_degrees,
        }


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

    @property
    def continuity_repair_target_boundary_indices(self) -> tuple[int, ...]:
        values = self.payload.get(
            "continuity_repair_target_boundary_indices",
            [],
        )
        if not isinstance(values, list):
            raise ValueError(
                "member proxy continuity repair targets are malformed"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in values
        ):
            raise ValueError(
                "member proxy continuity repair targets are malformed"
            )
        return tuple(values)

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

    def ranked_reselection_boundary_penalties(
        self,
        *,
        maximum_count: int = 4,
        minimum_penalty: float = 0.75,
        maximum_penalty: float = 4.0,
    ) -> tuple[tuple[int, float], ...]:
        """Return bounded severity costs for decoded-output path feedback.

        A failed decoded proxy proves that the selected edge looked poor after
        the complete render transform.  It does *not* prove that the edge is
        impossible to render.  These finite penalties let the selector prefer
        a different source pair without disconnecting the path graph when the
        source set has no perfect alternative.
        """

        if (
            isinstance(minimum_penalty, bool)
            or not isinstance(minimum_penalty, (int, float))
            or isinstance(maximum_penalty, bool)
            or not isinstance(maximum_penalty, (int, float))
            or not 0.0 < float(minimum_penalty) <= float(maximum_penalty)
        ):
            raise ValueError("member proxy penalty bounds are invalid")
        boundary_indices = self.ranked_reselection_boundary_indices(
            maximum_count=maximum_count
        )
        continuity = self.payload.get("rendered_continuity")
        if not isinstance(continuity, Mapping):
            raise ValueError("member proxy continuity payload is missing")
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
        raw_failures = continuity.get("failure_boundary_indices")
        failures_by_boundary: dict[int, set[str]] = {}
        if isinstance(raw_failures, Mapping):
            for reason, indices in raw_failures.items():
                if not isinstance(reason, str):
                    continue
                for boundary_index in _integer_indices(indices):
                    failures_by_boundary.setdefault(
                        boundary_index,
                        set(),
                    ).add(reason)

        def ratio(record: Mapping[str, Any], field: str, limit: str) -> float:
            value = record.get(field)
            maximum = limits.get(limit)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isinstance(maximum, (int, float))
                or isinstance(maximum, bool)
                or float(maximum) <= 0.0
            ):
                return 0.0
            return max(0.0, float(value) / float(maximum))

        penalties: list[tuple[int, float]] = []
        for boundary_index in boundary_indices:
            record = records_by_index.get(boundary_index, {})
            reasons = failures_by_boundary.get(boundary_index, set())
            normalized_geometry = max(
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
            geometry_failure_count = sum(
                any(
                    token in reason
                    for token in (
                        "position",
                        "scale",
                        "pose",
                        "silhouette",
                    )
                )
                for reason in reasons
            )
            motion_failure_count = sum(
                "motion" in reason for reason in reasons
            )
            severity = (
                0.55 * normalized_geometry
                + 0.45 * geometry_failure_count
                + 0.35 * motion_failure_count
            )
            penalty = min(
                float(maximum_penalty),
                max(float(minimum_penalty), severity),
            )
            penalties.append((boundary_index, penalty))
        return tuple(penalties)


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


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def promote_direct_member_pose_dissolves_from_rendered_proxy(
    plan: RenderPlan,
    preflight: DirectMemberRenderedProxyPreflight,
    *,
    source_durations_by_index: Mapping[int, float],
    highlight_intervals: Sequence[tuple[float, float]] = (),
    excluded_boundary_indices: Sequence[int] = (),
    preferred_duration_seconds: float = 0.23,
    target_dissolves_per_minute: float = 5.2,
    maximum_boundary_fraction: float = 0.36,
    minimum_boundary_gap: int = 1,
    minimum_event_spacing_seconds: float = 1.60,
) -> tuple[RenderPlan, dict[str, Any]]:
    """Promote output-matched hard cuts, then require an effect rerender.

    Source-domain geometry cannot judge a boundary after the final portrait
    fill and editorial camera have moved both pictures.  This refinement uses
    only a completely passing decoded hard-cut proxy with source-bound focal
    face evidence.  The returned overlap authority is intentionally pending:
    the caller must rerender the effect plan and accept it only when the normal
    decoded-output continuity gate passes its entry and exit endpoint checks.
    """

    if not isinstance(plan, RenderPlan):
        raise TypeError("output-verified dissolve promotion needs a render plan")
    if not isinstance(preflight, DirectMemberRenderedProxyPreflight):
        raise TypeError("output-verified dissolve promotion needs a preflight")
    if (
        not math.isfinite(preferred_duration_seconds)
        or not 0.18 <= preferred_duration_seconds <= 0.30
        or not math.isfinite(target_dissolves_per_minute)
        or target_dissolves_per_minute <= 0.0
        or not math.isfinite(maximum_boundary_fraction)
        or not 0.0 < maximum_boundary_fraction <= 1.0
        or isinstance(minimum_boundary_gap, bool)
        or not isinstance(minimum_boundary_gap, int)
        or minimum_boundary_gap < 1
        or not math.isfinite(minimum_event_spacing_seconds)
        or minimum_event_spacing_seconds <= 0.0
    ):
        raise ValueError("output-verified dissolve policy is invalid")
    normalized_durations: dict[int, float] = {}
    for source_index, value in source_durations_by_index.items():
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
        ):
            raise ValueError("output-verified dissolve source index is invalid")
        duration = _finite_float(value)
        if duration is None or duration <= 0.0:
            raise ValueError("output-verified dissolve source duration is invalid")
        normalized_durations[source_index] = duration
    intervals = tuple((float(start), float(end)) for start, end in highlight_intervals)
    if any(
        not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0.0
        or end <= start
        for start, end in intervals
    ):
        raise ValueError("output-verified dissolve highlight interval is invalid")
    excluded = frozenset(excluded_boundary_indices)
    if any(
        isinstance(boundary_index, bool)
        or not isinstance(boundary_index, int)
        or not 0 <= boundary_index < len(plan.transitions)
        for boundary_index in excluded
    ):
        raise ValueError("output-verified dissolve exclusion is invalid")

    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "analysis": OUTPUT_VERIFIED_DISSOLVE_ANALYSIS_BASIS,
        "applied": False,
        "eligible_candidate_count": 0,
        "selected_boundary_indices": [],
        "excluded_boundary_indices": sorted(excluded),
        "effect_proxy_revalidation_required": False,
        "authority": {
            "input": "completely_passing_decoded_hardcut_output_proxy",
            "target_member": "source_bound_focal_face_reference",
            "source_path_changed": False,
            "cut_timing_changed": False,
            "camera_changed": False,
            "look_changed": False,
            "audio_changed": False,
        },
        "privacy": {
            "normalized_scalar_geometry_only": True,
            "face_embeddings_persisted": False,
            "raw_frames_persisted": False,
        },
    }
    if plan.member_pose_dissolve_requested is not True:
        diagnostics["status"] = "effect_disabled_by_request"
        return (plan, diagnostics)
    if any(
        preflight.payload.get(name) is not True
        for name in (
            "passed",
            "member_output_quality_passed",
            "direct_member_rendered_continuity_passed",
        )
    ):
        diagnostics["status"] = "decoded_hardcut_proxy_not_pass"
        return (plan, diagnostics)
    continuity = preflight.payload.get("rendered_continuity")
    if not isinstance(continuity, Mapping) or continuity.get("passed") is not True:
        diagnostics["status"] = "rendered_continuity_not_pass"
        return (plan, diagnostics)
    raw_records = continuity.get("boundaries")
    if not isinstance(raw_records, list):
        diagnostics["status"] = "rendered_continuity_boundaries_missing"
        return (plan, diagnostics)
    records_by_index = {
        int(record["boundary_index"]): record
        for record in raw_records
        if isinstance(record, Mapping)
        and isinstance(record.get("boundary_index"), int)
        and not isinstance(record.get("boundary_index"), bool)
    }
    expected_indices = set(range(len(plan.transitions)))
    if set(records_by_index) != expected_indices:
        diagnostics["status"] = "rendered_continuity_boundary_binding_invalid"
        return (plan, diagnostics)

    thresholds = continuity.get("thresholds")
    limits = thresholds if isinstance(thresholds, Mapping) else {}
    position_limit = _finite_float(
        limits.get("maximum_attention_position_mismatch_p95")
    ) or 0.12
    scale_limit = _finite_float(
        limits.get("maximum_attention_scale_mismatch_p95")
    ) or 0.10
    pose_limit = _finite_float(limits.get("maximum_pose_mismatch_p95")) or 0.24
    effect_frames = max(3, int(round(preferred_duration_seconds * plan.fps)))
    frame_counts = segment_frame_counts(plan)
    rejection_counts: dict[str, int] = {}
    candidate_records: list[dict[str, Any]] = []

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    for boundary_index in range(len(plan.transitions)):
        transition = plan.transitions[boundary_index]
        outgoing = plan.segments[boundary_index]
        incoming = plan.segments[boundary_index + 1]
        record = records_by_index[boundary_index]
        motion = record.get("motion")
        motion_payload = motion if isinstance(motion, Mapping) else {}
        position = _finite_float(record.get("attention_position_mismatch"))
        scale = _finite_float(record.get("attention_scale_mismatch"))
        pose = _finite_float(record.get("pose_mismatch"))
        endpoint_shape = _finite_float(
            motion_payload.get("endpoint_pose_shape_mismatch")
        )
        subject_direction = _finite_float(
            motion_payload.get("subject_direction_cosine")
        )
        speed = _finite_float(motion_payload.get("motion_speed_ratio_mismatch"))
        phase = _finite_float(
            motion_payload.get("pose_phase_continuation_mismatch")
        )
        reason: str | None = None
        if boundary_index in excluded:
            reason = "excluded_after_failed_effect_proxy"
        elif (
            transition.kind is not TransitionKind.HARD_CUT
            or record.get("transition_kind") != TransitionKind.HARD_CUT.value
            or outgoing.source_index == incoming.source_index
        ):
            reason = "not_a_hard_source_switch"
        elif record.get("descriptors_available") is not True:
            reason = "decoded_boundary_descriptor_missing"
        elif record.get("focal_face_reference_guided") is not True:
            reason = "source_bound_focal_face_not_verified"
        elif position is None or position > position_limit + 1e-9:
            reason = "decoded_attention_position_not_effect_safe"
        elif scale is None or scale > scale_limit + 1e-9:
            reason = "decoded_attention_scale_not_effect_safe"
        elif not (
            (pose is not None and pose <= pose_limit + 1e-9)
            or (
                endpoint_shape is not None
                and endpoint_shape <= 0.22 + 1e-9
            )
        ):
            reason = "decoded_pose_shape_not_effect_safe"
        elif motion_payload.get("both_moving") is not True:
            reason = "decoded_boundary_not_motion_connected"
        elif subject_direction is None or subject_direction < -1e-9:
            reason = "decoded_subject_motion_direction_opposed"
        elif speed is None or speed > 0.75 + 1e-9:
            reason = "decoded_motion_speed_not_effect_safe"
        elif phase is not None and phase > 0.30 + 1e-9:
            reason = "decoded_pose_phase_not_effect_safe"
        elif effect_frames > frame_counts[boundary_index + 1]:
            reason = "incoming_segment_too_short_for_effect"
        else:
            source_duration = normalized_durations.get(outgoing.source_index)
            required_source_end = outgoing.source_end + (
                effect_frames / plan.fps
            ) * outgoing.source_playback_rate
            if (
                source_duration is None
                or required_source_end > source_duration + 1e-9
            ):
                reason = "outgoing_source_handle_unavailable"
        if reason is not None:
            reject(reason)
            continue
        output_seconds = _finite_float(record.get("output_seconds"))
        if output_seconds is None:
            output_seconds = sum(
                segment.duration
                for segment in plan.segments[: boundary_index + 1]
            )
        canonical_seconds = plan.master_audio_start + output_seconds
        highlight_priority = any(
            start - 1e-9 <= canonical_seconds <= end + 1e-9
            for start, end in intervals
        )
        candidate_records.append(
            {
                "boundary_index": boundary_index,
                "canonical_master_seconds": canonical_seconds,
                "highlight_priority": highlight_priority,
                "attention_position_mismatch": position,
                "attention_scale_mismatch": scale,
                "pose_mismatch": pose,
                "endpoint_pose_shape_mismatch": endpoint_shape,
                "subject_direction_cosine": subject_direction,
                "motion_speed_ratio_mismatch": speed,
                "pose_phase_continuation_mismatch": phase,
            }
        )

    diagnostics["eligible_candidate_count"] = len(candidate_records)
    diagnostics["rejection_reason_counts"] = dict(sorted(rejection_counts.items()))
    diagnostics["candidates"] = candidate_records
    if not candidate_records:
        diagnostics["status"] = "no_output_verified_effect_candidate"
        return (plan, diagnostics)
    target_count = max(
        1,
        int(round(plan.duration * target_dissolves_per_minute / 60.0)),
    )
    fraction_count = max(
        1,
        int(math.floor(len(plan.transitions) * maximum_boundary_fraction + 1e-12)),
    )
    maximum_count = min(len(candidate_records), target_count, fraction_count)
    ranked = sorted(
        candidate_records,
        key=lambda item: (
            not bool(item["highlight_priority"]),
            -float(item["subject_direction_cosine"]),
            float(
                item["pose_mismatch"]
                if item["pose_mismatch"] is not None
                else item["endpoint_pose_shape_mismatch"]
            ),
            float(item["attention_position_mismatch"])
            + float(item["attention_scale_mismatch"]),
            int(item["boundary_index"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    for candidate in ranked:
        if len(selected) >= maximum_count:
            break
        if any(
            abs(
                int(candidate["boundary_index"])
                - int(existing["boundary_index"])
            )
            <= minimum_boundary_gap
            or abs(
                float(candidate["canonical_master_seconds"])
                - float(existing["canonical_master_seconds"])
            )
            < minimum_event_spacing_seconds - 1e-9
            for existing in selected
        ):
            continue
        selected.append(candidate)
    if not selected:
        diagnostics["status"] = "effect_density_or_spacing_budget_exhausted"
        return (plan, diagnostics)

    transitions = list(plan.transitions)
    for candidate in selected:
        boundary_index = int(candidate["boundary_index"])
        transitions[boundary_index] = RenderTransition(
            kind=TransitionKind.FACE_DISSOLVE,
            frame_count=effect_frames,
            reason=DIRECT_MEMBER_POSE_DISSOLVE_REASON,
            face_mismatch=(
                float(candidate["attention_position_mismatch"])
                + float(candidate["attention_scale_mismatch"])
            )
            / 2.0,
            tier="A" if candidate["highlight_priority"] else "B",
            future_pose_max_mismatch=(
                float(candidate["pose_mismatch"])
                if candidate["pose_mismatch"] is not None
                else float(candidate["endpoint_pose_shape_mismatch"])
            ),
            transform_overlap_authority=(
                MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY
            ),
        )
    dissolve_count = sum(
        transition.kind is TransitionKind.FACE_DISSOLVE
        and transition.reason == DIRECT_MEMBER_POSE_DISSOLVE_REASON
        for transition in transitions
    )
    try:
        promoted = replace(
            plan,
            transitions=tuple(transitions),
            member_pose_dissolve_effective=bool(dissolve_count),
            member_pose_dissolve_count=dissolve_count,
        )
    except ValueError as exc:
        diagnostics["status"] = "promoted_effect_plan_contract_invalid"
        diagnostics["contract_error"] = str(exc)
        return (plan, diagnostics)
    diagnostics.update(
        {
            "applied": True,
            "status": "effect_proxy_revalidation_required",
            "selected_boundary_indices": [
                int(candidate["boundary_index"]) for candidate in selected
            ],
            "selected_highlight_count": sum(
                bool(candidate["highlight_priority"])
                for candidate in selected
            ),
            "selected_both_moving_count": len(selected),
            "transition_frame_count": effect_frames,
            "duration_seconds": effect_frames / plan.fps,
            "effect_proxy_revalidation_required": True,
        }
    )
    return (promoted, diagnostics)


def demote_failed_direct_member_pose_dissolves_from_rendered_proxy(
    plan: RenderPlan,
    preflight: DirectMemberRenderedProxyPreflight,
) -> tuple[RenderPlan, dict[str, Any]]:
    """Remove only decoded-output-invalid dissolves and retain proven effects.

    Source-domain pose matching can nominate several effects before the final
    portrait crop, camera track, and encoding are visible.  If the decoded
    effect proxy proves that only a subset has discontinuous entry/exit
    frames, replacing the entire source path discards already-good motion
    matches.  This repair changes those failed effects to duration-neutral
    hard cuts, keeps every surviving dissolve, and requires another decoded
    proxy before the plan can be accepted.
    """

    if not isinstance(plan, RenderPlan):
        raise TypeError("effect demotion needs a render plan")
    if not isinstance(preflight, DirectMemberRenderedProxyPreflight):
        raise TypeError("effect demotion needs a member proxy preflight")
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "analysis": OUTPUT_VERIFIED_DISSOLVE_DEMOTION_ANALYSIS_BASIS,
        "applied": False,
        "demoted_boundary_indices": [],
        "retained_boundary_indices": [],
        "effect_proxy_revalidation_required": False,
        "authority": {
            "input": "decoded_pose_dissolve_effect_endpoint_failures",
            "source_path_changed": False,
            "cut_timing_changed": False,
            "camera_changed": False,
            "look_changed": False,
            "audio_changed": False,
            "passing_effects_retained": True,
        },
    }
    continuity = preflight.payload.get("rendered_continuity")
    continuity_payload = (
        continuity if isinstance(continuity, Mapping) else {}
    )
    raw_failures = continuity_payload.get("failure_boundary_indices")
    if not isinstance(raw_failures, Mapping):
        diagnostics["status"] = "rendered_effect_failures_missing"
        return (plan, diagnostics)
    failure_reasons_by_boundary: dict[int, list[str]] = {}
    for reason in sorted(_REPAIRABLE_EFFECT_ENDPOINT_FAILURE_REASONS):
        raw_indices = raw_failures.get(reason)
        if not isinstance(raw_indices, list):
            continue
        for raw_index in raw_indices:
            if (
                isinstance(raw_index, bool)
                or not isinstance(raw_index, int)
                or not 0 <= raw_index < len(plan.transitions)
            ):
                continue
            failure_reasons_by_boundary.setdefault(raw_index, []).append(
                reason
            )
    current_effect_indices = tuple(
        boundary_index
        for boundary_index, transition in enumerate(plan.transitions)
        if transition.kind is TransitionKind.FACE_DISSOLVE
        and transition.reason == DIRECT_MEMBER_POSE_DISSOLVE_REASON
    )
    demoted_indices = tuple(
        boundary_index
        for boundary_index in current_effect_indices
        if boundary_index in failure_reasons_by_boundary
    )
    retained_indices = tuple(
        boundary_index
        for boundary_index in current_effect_indices
        if boundary_index not in failure_reasons_by_boundary
    )
    diagnostics["candidate_effect_boundary_indices"] = list(
        current_effect_indices
    )
    diagnostics["demoted_boundary_indices"] = list(demoted_indices)
    diagnostics["retained_boundary_indices"] = list(retained_indices)
    diagnostics["failure_reasons_by_boundary"] = {
        str(boundary_index): failure_reasons_by_boundary[boundary_index]
        for boundary_index in demoted_indices
    }
    if not demoted_indices:
        diagnostics["status"] = "no_failed_pose_dissolve_endpoint"
        return (plan, diagnostics)
    if not retained_indices:
        diagnostics["status"] = "all_pose_dissolves_failed_reselection_required"
        return (plan, diagnostics)

    raw_records = continuity_payload.get("boundaries")
    if not isinstance(raw_records, list):
        diagnostics["status"] = "demoted_effect_hardcut_evidence_missing"
        return (plan, diagnostics)
    records_by_index = {
        int(record["boundary_index"]): record
        for record in raw_records
        if isinstance(record, Mapping)
        and isinstance(record.get("boundary_index"), int)
        and not isinstance(record.get("boundary_index"), bool)
    }
    raw_thresholds = continuity_payload.get("thresholds")
    thresholds = raw_thresholds if isinstance(raw_thresholds, Mapping) else {}
    maximum_position = (
        _finite_float(thresholds.get("maximum_attention_position_mismatch"))
        or 0.20
    )
    maximum_scale = (
        _finite_float(thresholds.get("maximum_attention_scale_mismatch"))
        or 0.18
    )
    maximum_pose = (
        _finite_float(thresholds.get("maximum_pose_mismatch")) or 0.30
    )
    hardcut_viability_by_boundary: dict[str, dict[str, Any]] = {}
    unsafe_hardcut_indices: list[int] = []
    for boundary_index in demoted_indices:
        record = records_by_index.get(boundary_index)
        position = (
            _finite_float(record.get("attention_position_mismatch"))
            if isinstance(record, Mapping)
            else None
        )
        scale = (
            _finite_float(record.get("attention_scale_mismatch"))
            if isinstance(record, Mapping)
            else None
        )
        pose = (
            _finite_float(record.get("pose_mismatch"))
            if isinstance(record, Mapping)
            else None
        )
        failed_requirements: list[str] = []
        if not isinstance(record, Mapping):
            failed_requirements.append("decoded_boundary_record_missing")
        elif record.get("transition_kind") != TransitionKind.FACE_DISSOLVE.value:
            failed_requirements.append("decoded_transition_binding_invalid")
        if position is None or position > maximum_position + 1e-9:
            failed_requirements.append("hardcut_position_not_viable")
        if scale is None or scale > maximum_scale + 1e-9:
            failed_requirements.append("hardcut_scale_not_viable")
        if pose is None or pose > maximum_pose + 1e-9:
            failed_requirements.append("hardcut_pose_not_viable")
        if failed_requirements:
            unsafe_hardcut_indices.append(boundary_index)
        hardcut_viability_by_boundary[str(boundary_index)] = {
            "attention_position_mismatch": position,
            "maximum_attention_position_mismatch": maximum_position,
            "attention_scale_mismatch": scale,
            "maximum_attention_scale_mismatch": maximum_scale,
            "pose_mismatch": pose,
            "maximum_pose_mismatch": maximum_pose,
            "viable": not failed_requirements,
            "failed_requirements": failed_requirements,
        }
    diagnostics["hardcut_viability_by_boundary"] = (
        hardcut_viability_by_boundary
    )
    diagnostics["unsafe_hardcut_boundary_indices"] = unsafe_hardcut_indices
    if unsafe_hardcut_indices:
        diagnostics["status"] = (
            "demoted_effect_hardcut_continuity_not_viable"
        )
        return (plan, diagnostics)

    transitions = list(plan.transitions)
    for boundary_index in demoted_indices:
        transitions[boundary_index] = RenderTransition(
            kind=TransitionKind.HARD_CUT,
            frame_count=0,
            reason=_OUTPUT_VERIFIED_DISSOLVE_DEMOTION_REASON,
        )
    try:
        repaired = replace(
            plan,
            transitions=tuple(transitions),
            member_pose_dissolve_effective=True,
            member_pose_dissolve_count=len(retained_indices),
        )
    except (TypeError, ValueError) as exc:
        diagnostics["status"] = "demoted_effect_plan_contract_invalid"
        diagnostics["contract_error"] = str(exc)
        return (plan, diagnostics)
    diagnostics.update(
        {
            "applied": True,
            "status": "effect_proxy_revalidation_required",
            "effect_proxy_revalidation_required": True,
            "retained_effect_count": len(retained_indices),
            "demoted_effect_count": len(demoted_indices),
        }
    )
    return (repaired, diagnostics)


def _camera_state_at(
    segment: RenderSegment,
    time_seconds: float,
) -> _CameraState:
    track = segment.editorial_transform
    if track is None:
        return _CameraState(1.0, 0.5, 0.5, 0.0)
    keyframe = track.evaluate(time_seconds)
    return _CameraState(
        keyframe.zoom,
        keyframe.center_x,
        keyframe.center_y,
        keyframe.roll_degrees,
    )


def _interpolate_camera_state(
    left: _CameraState,
    right: _CameraState,
    alpha: float,
) -> _CameraState:
    amount = min(1.0, max(0.0, float(alpha)))
    return _CameraState(
        zoom=left.zoom + (right.zoom - left.zoom) * amount,
        center_x=left.center_x + (right.center_x - left.center_x) * amount,
        center_y=left.center_y + (right.center_y - left.center_y) * amount,
        roll_degrees=(
            left.roll_degrees
            + (right.roll_degrees - left.roll_degrees) * amount
        ),
    )


def _refined_endpoint_track(
    segment: RenderSegment,
    *,
    last_frame_time: float,
    entry_state: _CameraState | None,
    exit_state: _CameraState | None,
    ramp_seconds: float,
    entry_hold_seconds: float = 0.0,
) -> EditorialTransformTrack:
    """Blend output-measured endpoint fixes into the existing camera track."""

    if entry_state is None and exit_state is None:
        if segment.editorial_transform is None:
            raise ValueError("endpoint refinement needs at least one camera state")
        return segment.editorial_transform
    if not math.isfinite(last_frame_time) or last_frame_time <= 0.0:
        raise ValueError("endpoint refinement segment is too short")
    if (
        not math.isfinite(entry_hold_seconds)
        or entry_hold_seconds < 0.0
        or entry_hold_seconds > last_frame_time + 1e-9
    ):
        raise ValueError("endpoint refinement entry hold is invalid")

    original = lambda seconds: _camera_state_at(segment, seconds)
    entry_hold = min(entry_hold_seconds, last_frame_time)
    entry_guard = min(entry_hold + ramp_seconds, last_frame_time)
    exit_guard = max(0.0, last_frame_time - ramp_seconds)
    ramps_overlap = bool(
        entry_state is not None
        and exit_state is not None
        and entry_guard >= exit_guard - 1e-9
    )

    if ramps_overlap:
        assert entry_state is not None and exit_state is not None

        def state_at(seconds: float) -> _CameraState:
            return _interpolate_camera_state(
                entry_state,
                exit_state,
                seconds / last_frame_time,
            )

        times = (0.0, last_frame_time)
    else:
        original_entry_guard = original(entry_guard)
        original_exit_guard = original(exit_guard)

        def state_at(seconds: float) -> _CameraState:
            if entry_state is not None and seconds <= entry_hold + 1e-9:
                return entry_state
            if entry_state is not None and seconds < entry_guard - 1e-9:
                return _interpolate_camera_state(
                    entry_state,
                    original_entry_guard,
                    (seconds - entry_hold)
                    / max(entry_guard - entry_hold, 1e-9),
                )
            if exit_state is not None and seconds > exit_guard + 1e-9:
                return _interpolate_camera_state(
                    original_exit_guard,
                    exit_state,
                    (seconds - exit_guard)
                    / max(last_frame_time - exit_guard, 1e-9),
                )
            if entry_state is not None and seconds <= 1e-9:
                return entry_state
            if exit_state is not None and (
                seconds >= last_frame_time - 1e-9
            ):
                return exit_state
            return original(seconds)

        time_values = {0.0, last_frame_time}
        if entry_state is not None:
            if entry_hold > 1e-9:
                time_values.add(entry_hold)
            time_values.add(entry_guard)
        if exit_state is not None:
            time_values.add(exit_guard)
        if segment.editorial_transform is not None:
            time_values.update(
                min(last_frame_time, max(0.0, item.time_seconds))
                for item in segment.editorial_transform.keyframes
            )
        times = tuple(sorted(time_values))

    return EditorialTransformTrack(
        keyframes=tuple(
            state_at(time_seconds).keyframe(time_seconds)
            for time_seconds in times
        ),
        reason="decoded_output_focal_endpoint_inverse_refinement_v1",
    )


def _parsed_attention_geometry(
    record: Mapping[str, Any],
    side: str,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float],
    float | None,
] | None:
    attention = record.get("attention_geometry")
    if not isinstance(attention, Mapping):
        return None
    endpoint = attention.get(side)
    if not isinstance(endpoint, Mapping):
        return None
    raw_box = endpoint.get("bbox_xywh")
    raw_centroid = endpoint.get("centroid_xy")
    if not isinstance(raw_box, Sequence) or isinstance(raw_box, (str, bytes)):
        return None
    if not isinstance(raw_centroid, Sequence) or isinstance(
        raw_centroid, (str, bytes)
    ):
        return None
    if len(raw_box) != 4 or len(raw_centroid) != 2:
        return None
    box_values = tuple(_finite_float(value) for value in raw_box)
    centroid_values = tuple(_finite_float(value) for value in raw_centroid)
    if any(value is None for value in (*box_values, *centroid_values)):
        return None
    box = tuple(float(value) for value in box_values)
    centroid = tuple(float(value) for value in centroid_values)
    if box[2] <= 0.0 or box[3] <= 0.0:
        return None
    scale_area_proxy = _finite_float(endpoint.get("scale_area_proxy"))
    if scale_area_proxy is not None and scale_area_proxy <= 0.0:
        scale_area_proxy = None
    return (
        (box[0], box[1], box[2], box[3]),
        (centroid[0], centroid[1]),
        scale_area_proxy,
    )


def _zoom_candidates(maximum_zoom: float, current_zoom: float) -> tuple[float, ...]:
    values = {1.0, float(maximum_zoom), float(current_zoom)}
    step_count = int(math.floor((maximum_zoom - 1.0) / 0.01 + 1e-9))
    values.update(1.0 + step * 0.01 for step in range(step_count + 1))
    return tuple(
        sorted(
            value
            for value in values
            if 1.0 - 1e-9 <= value <= maximum_zoom + 1e-9
        )
    )


def _camera_state_is_safe(state: _CameraState) -> bool:
    if abs(state.roll_degrees) > 1e-9:
        return False
    half = 0.5 / state.zoom
    if not (
        half - 1e-9 <= state.center_x <= 1.0 - half + 1e-9
        and half - 1e-9 <= state.center_y <= 1.0 - half + 1e-9
    ):
        return False
    if math.hypot(state.center_x - 0.5, state.center_y - 0.5) > (
        EDITORIAL_TRANSFORM_MAXIMUM_CENTER_SHIFT + 1e-9
    ):
        return False
    try:
        state.keyframe(0.0)
    except ValueError:
        return False
    return True


def _bbox_is_inside_output(
    centroid: tuple[float, float],
    bbox: tuple[float, float, float, float],
    zoom_ratio: float,
) -> bool:
    half_width = bbox[2] * zoom_ratio / 2.0
    half_height = bbox[3] * zoom_ratio / 2.0
    return bool(
        centroid[0] - half_width >= -1e-9
        and centroid[0] + half_width <= 1.0 + 1e-9
        and centroid[1] - half_height >= -1e-9
        and centroid[1] + half_height <= 1.0 + 1e-9
    )


def _boundary_inverse_camera_states(
    *,
    outgoing_state: _CameraState,
    incoming_state: _CameraState,
    outgoing_geometry: tuple[
        tuple[float, float, float, float],
        tuple[float, float],
        float | None,
    ],
    incoming_geometry: tuple[
        tuple[float, float, float, float],
        tuple[float, float],
        float | None,
    ],
    outgoing_maximum_zoom: float,
    incoming_maximum_zoom: float,
    target_position_mismatch: float,
    target_scale_mismatch: float,
) -> tuple[_CameraState, _CameraState, dict[str, float]] | None:
    if (
        abs(outgoing_state.roll_degrees) > 1e-9
        or abs(incoming_state.roll_degrees) > 1e-9
        or outgoing_state.zoom > outgoing_maximum_zoom + 1e-9
        or incoming_state.zoom > incoming_maximum_zoom + 1e-9
    ):
        return None
    outgoing_box, outgoing_centroid, outgoing_scale_area = outgoing_geometry
    incoming_box, incoming_centroid, incoming_scale_area = incoming_geometry
    outgoing_source_point = (
        outgoing_state.center_x
        + (outgoing_centroid[0] - 0.5) / outgoing_state.zoom,
        outgoing_state.center_y
        + (outgoing_centroid[1] - 0.5) / outgoing_state.zoom,
    )
    incoming_source_point = (
        incoming_state.center_x
        + (incoming_centroid[0] - 0.5) / incoming_state.zoom,
        incoming_state.center_y
        + (incoming_centroid[1] - 0.5) / incoming_state.zoom,
    )
    delta_x = outgoing_centroid[0] - incoming_centroid[0]
    delta_y = outgoing_centroid[1] - incoming_centroid[1]
    distance = math.hypot(delta_x, delta_y)
    allowed_distance = target_position_mismatch * 0.42
    required_distance_reduction = max(0.0, distance - allowed_distance)
    unit_x = delta_x / distance if distance > 1e-12 else 0.0
    unit_y = delta_y / distance if distance > 1e-12 else 0.0
    position_splits = (
        tuple(step / 20.0 for step in range(21))
        if required_distance_reduction > 1e-12
        else (0.5,)
    )
    use_landmark_scale = bool(
        outgoing_scale_area is not None and incoming_scale_area is not None
    )
    outgoing_bbox_area = max(1e-9, outgoing_box[2] * outgoing_box[3])
    incoming_bbox_area = max(1e-9, incoming_box[2] * incoming_box[3])
    best: tuple[
        tuple[float, float, float],
        _CameraState,
        _CameraState,
        dict[str, float],
    ] | None = None

    for outgoing_zoom in _zoom_candidates(
        outgoing_maximum_zoom,
        outgoing_state.zoom,
    ):
        outgoing_zoom_ratio = outgoing_zoom / outgoing_state.zoom
        projected_outgoing_bbox_area = (
            outgoing_bbox_area * outgoing_zoom_ratio**2
        )
        projected_outgoing_landmark_area = (
            float(outgoing_scale_area) * outgoing_zoom_ratio**2
            if use_landmark_scale
            else None
        )
        for incoming_zoom in _zoom_candidates(
            incoming_maximum_zoom,
            incoming_state.zoom,
        ):
            incoming_zoom_ratio = incoming_zoom / incoming_state.zoom
            projected_incoming_bbox_area = (
                incoming_bbox_area * incoming_zoom_ratio**2
            )
            projected_incoming_landmark_area = (
                float(incoming_scale_area) * incoming_zoom_ratio**2
                if use_landmark_scale
                else None
            )
            (
                bbox_scale_log,
                landmark_scale_log,
                scale_direction_agreement,
                consensus_scale_log,
            ) = _scale_log_ratio_evidence(
                projected_outgoing_bbox_area,
                projected_incoming_bbox_area,
                projected_outgoing_landmark_area,
                projected_incoming_landmark_area,
            )
            scale_mismatch = min(
                1.0,
                abs(consensus_scale_log) / math.log(3.2),
            )
            if scale_mismatch > target_scale_mismatch + 1e-9:
                continue
            for outgoing_share in position_splits:
                outgoing_target = (
                    outgoing_centroid[0]
                    - unit_x
                    * required_distance_reduction
                    * outgoing_share,
                    outgoing_centroid[1]
                    - unit_y
                    * required_distance_reduction
                    * outgoing_share,
                )
                incoming_target = (
                    incoming_centroid[0]
                    + unit_x
                    * required_distance_reduction
                    * (1.0 - outgoing_share),
                    incoming_centroid[1]
                    + unit_y
                    * required_distance_reduction
                    * (1.0 - outgoing_share),
                )
                outgoing_candidate = _CameraState(
                    zoom=outgoing_zoom,
                    center_x=(
                        outgoing_source_point[0]
                        - (outgoing_target[0] - 0.5) / outgoing_zoom
                    ),
                    center_y=(
                        outgoing_source_point[1]
                        - (outgoing_target[1] - 0.5) / outgoing_zoom
                    ),
                )
                incoming_candidate = _CameraState(
                    zoom=incoming_zoom,
                    center_x=(
                        incoming_source_point[0]
                        - (incoming_target[0] - 0.5) / incoming_zoom
                    ),
                    center_y=(
                        incoming_source_point[1]
                        - (incoming_target[1] - 0.5) / incoming_zoom
                    ),
                )
                if not (
                    _camera_state_is_safe(outgoing_candidate)
                    and _camera_state_is_safe(incoming_candidate)
                    and _bbox_is_inside_output(
                        outgoing_target,
                        outgoing_box,
                        outgoing_zoom_ratio,
                    )
                    and _bbox_is_inside_output(
                        incoming_target,
                        incoming_box,
                        incoming_zoom_ratio,
                    )
                ):
                    continue
                position_mismatch = min(
                    1.0,
                    math.hypot(
                        outgoing_target[0] - incoming_target[0],
                        outgoing_target[1] - incoming_target[1],
                    )
                    / 0.42,
                )
                if position_mismatch > target_position_mismatch + 1e-9:
                    continue
                zoom_change = abs(
                    math.log(outgoing_zoom / outgoing_state.zoom)
                ) + abs(math.log(incoming_zoom / incoming_state.zoom))
                pan_change = math.hypot(
                    outgoing_candidate.center_x - outgoing_state.center_x,
                    outgoing_candidate.center_y - outgoing_state.center_y,
                ) + math.hypot(
                    incoming_candidate.center_x - incoming_state.center_x,
                    incoming_candidate.center_y - incoming_state.center_y,
                )
                objective = (
                    zoom_change
                    + 1.5 * pan_change
                    + 0.03
                    * (
                        outgoing_candidate.zoom
                        + incoming_candidate.zoom
                        - 2.0
                    )
                )
                rank = (
                    objective,
                    max(outgoing_zoom, incoming_zoom),
                    outgoing_zoom + incoming_zoom,
                )
                diagnostics = {
                    "predicted_attention_position_mismatch": (
                        position_mismatch
                    ),
                    "predicted_attention_scale_mismatch": scale_mismatch,
                    "outgoing_target_centroid_x": outgoing_target[0],
                    "outgoing_target_centroid_y": outgoing_target[1],
                    "incoming_target_centroid_x": incoming_target[0],
                    "incoming_target_centroid_y": incoming_target[1],
                    "stable_landmark_scale_used": float(
                        use_landmark_scale
                    ),
                    "predicted_bbox_scale_mismatch_diagnostic": min(
                        1.0,
                        abs(bbox_scale_log) / math.log(3.2),
                    ),
                    "predicted_landmark_scale_mismatch_diagnostic": (
                        min(
                            1.0,
                            abs(landmark_scale_log) / math.log(3.2),
                        )
                        if landmark_scale_log is not None
                        else 0.0
                    ),
                    "predicted_scale_direction_agreement": float(
                        scale_direction_agreement
                        if scale_direction_agreement is not None
                        else False
                    ),
                }
                if best is None or rank < best[0]:
                    best = (
                        rank,
                        outgoing_candidate,
                        incoming_candidate,
                        diagnostics,
                    )
    if best is None:
        return None
    return (best[1], best[2], best[3])


def _decoded_camera_geometry_authority(
    record: Mapping[str, Any],
    *,
    allow_official_fancam_subject_geometry: bool,
) -> str | None:
    """Return the narrow evidence class allowed to drive a camera repair."""

    if record.get("focal_face_reference_guided") is True:
        return "source_bound_focal_face_reference"
    raw_basis = record.get("attention_basis")
    basis = raw_basis if isinstance(raw_basis, Mapping) else {}
    outgoing_basis = basis.get("outgoing")
    incoming_basis = basis.get("incoming")
    if outgoing_basis == incoming_basis == FOCAL_FACE_BASIS:
        similarity = _finite_float(
            record.get("focal_face_correspondence_similarity")
        )
        if similarity is not None and similarity >= 0.50 - 1e-9:
            return "verified_official_fancam_cross_output_face_correspondence"
    if not allow_official_fancam_subject_geometry:
        return None
    pose_mismatch = _finite_float(record.get("pose_mismatch"))
    if pose_mismatch is None or pose_mismatch > 0.30 + 1e-9:
        return None
    if outgoing_basis == incoming_basis == _ATTENTION_FACE_BASIS:
        return "official_fancam_pose_bound_unambiguous_dominant_face"
    if (
        outgoing_basis == incoming_basis == _ATTENTION_SILHOUETTE_BASIS
        and record.get("descriptors_available") is True
    ):
        return "official_fancam_pose_bound_full_subject_silhouette"
    return None


def refine_direct_member_camera_plan_from_rendered_proxy(
    plan: RenderPlan,
    preflight: DirectMemberRenderedProxyPreflight,
    *,
    maximum_zoom_by_source_index: Mapping[int, float],
    repair_boundary_indices: Sequence[int] | None = None,
    target_margin_fraction: float = 0.80,
    ramp_seconds: float = 0.75,
    allow_official_fancam_subject_geometry: bool = False,
) -> tuple[RenderPlan, dict[str, Any]]:
    """Use exact decoded subject endpoints to correct one selected path.

    The default operation remains limited to source-bound or verified
    cross-output focal faces.  An official-target-fancam caller may opt into
    an unambiguous dominant face or a decoded full-subject silhouette only
    when actual pose evidence is present on both sides.  Unbound aggregate
    group silhouettes still never drive a member-camera pan.
    For a dissolve, the pure outgoing frame and pure incoming frame surrounding
    the overlap are aligned and the incoming correction is held through the
    whole effect. The returned plan keeps every source, cut time, crop, look,
    and audio decision intact; only boundary-local editorial transform
    keyframes and the reviewed overlap-transform authority may change.
    """

    if not isinstance(plan, RenderPlan):
        raise TypeError("camera refinement needs a render plan")
    if not isinstance(preflight, DirectMemberRenderedProxyPreflight):
        raise TypeError("camera refinement needs a member proxy preflight")
    if not isinstance(allow_official_fancam_subject_geometry, bool):
        raise TypeError("official fancam subject geometry flag must be boolean")
    repair_targets: frozenset[int] | None = None
    if repair_boundary_indices is not None:
        if isinstance(repair_boundary_indices, (str, bytes)):
            raise TypeError("camera refinement repair targets are invalid")
        parsed_repair_targets: set[int] = set()
        for boundary_index in repair_boundary_indices:
            if (
                isinstance(boundary_index, bool)
                or not isinstance(boundary_index, int)
                or not 0 <= boundary_index < len(plan.segments) - 1
            ):
                raise ValueError(
                    "camera refinement repair target is invalid"
                )
            parsed_repair_targets.add(boundary_index)
        repair_targets = frozenset(parsed_repair_targets)
    if (
        not math.isfinite(target_margin_fraction)
        or not 0.50 <= target_margin_fraction <= 1.0
    ):
        raise ValueError("camera refinement target margin is invalid")
    if not math.isfinite(ramp_seconds) or ramp_seconds <= 0.0:
        raise ValueError("camera refinement ramp must be positive")
    maximum_zooms: dict[int, float] = {}
    for source_index, value in maximum_zoom_by_source_index.items():
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
        ):
            raise ValueError("camera refinement source index is invalid")
        maximum = _finite_float(value)
        if (
            maximum is None
            or not 1.0 <= maximum <= EDITORIAL_TRANSFORM_MAXIMUM_ZOOM
        ):
            raise ValueError("camera refinement maximum zoom is invalid")
        maximum_zooms[source_index] = maximum

    continuity = preflight.payload.get("rendered_continuity")
    continuity_payload = (
        continuity if isinstance(continuity, Mapping) else {}
    )
    thresholds = continuity_payload.get("thresholds")
    threshold_payload = thresholds if isinstance(thresholds, Mapping) else {}
    position_p95 = _finite_float(
        threshold_payload.get("maximum_attention_position_mismatch_p95")
    )
    scale_p95 = _finite_float(
        threshold_payload.get("maximum_attention_scale_mismatch_p95")
    )
    target_position = (
        position_p95 * target_margin_fraction
        if position_p95 is not None and position_p95 > 0.0
        else 0.096
    )
    target_scale = (
        scale_p95 * target_margin_fraction
        if scale_p95 is not None and scale_p95 > 0.0
        else 0.080
    )
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "analysis": CAMERA_REFINEMENT_ANALYSIS_BASIS,
        "applied": False,
        "target_attention_position_mismatch": target_position,
        "target_attention_scale_mismatch": target_scale,
        "target_margin_fraction": target_margin_fraction,
        "ramp_seconds": ramp_seconds,
        "repair_target_boundary_indices": (
            sorted(repair_targets) if repair_targets is not None else None
        ),
        "refined_boundary_indices": [],
        "refined_overlap_boundary_indices": [],
        "unresolved_boundary_indices": [],
        "boundaries": [],
        "authority": {
            "geometry": (
                "exact_low_resolution_encoded_output_face_or_"
                "pose_bound_official_fancam_subject_endpoints"
            ),
            "focal_reference_required": (
                not allow_official_fancam_subject_geometry
            ),
            "verified_cross_output_face_minimum_similarity": 0.50,
            "official_fancam_subject_geometry_allowed": (
                allow_official_fancam_subject_geometry
            ),
            "aggregate_silhouette_camera_moves_allowed": (
                "official_target_fancam_and_actual_pose_bound_only"
                if allow_official_fancam_subject_geometry
                else False
            ),
            "source_path_changed": False,
            "cut_timing_changed": False,
            "look_changed": False,
            "audio_changed": False,
        },
        "privacy": {
            "normalized_geometry_only": True,
            "face_embeddings_persisted": False,
            "raw_frames_persisted": False,
        },
    }
    raw_records = continuity_payload.get("boundaries")
    if not isinstance(raw_records, list):
        diagnostics["status"] = "rendered_continuity_boundaries_missing"
        return (plan, diagnostics)

    frame_counts = segment_frame_counts(plan)
    last_frame_times = tuple(
        (frame_count - 1) / plan.fps for frame_count in frame_counts
    )
    corrections: dict[
        int,
        tuple[_CameraState, _CameraState, dict[str, Any]],
    ] = {}
    boundary_diagnostics: dict[int, dict[str, Any]] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            continue
        boundary_index = raw_record.get("boundary_index")
        if (
            isinstance(boundary_index, bool)
            or not isinstance(boundary_index, int)
            or not 0 <= boundary_index < len(plan.segments) - 1
        ):
            continue
        record_diagnostics: dict[str, Any] = {
            "boundary_index": boundary_index,
            "status": "not_eligible",
            "before_attention_position_mismatch": _finite_float(
                raw_record.get("attention_position_mismatch")
            ),
            "before_attention_scale_mismatch": _finite_float(
                raw_record.get("attention_scale_mismatch")
            ),
        }
        boundary_diagnostics[boundary_index] = record_diagnostics
        if (
            repair_targets is not None
            and boundary_index not in repair_targets
        ):
            record_diagnostics["status"] = "outside_repair_target"
            continue
        transition = plan.transitions[boundary_index]
        outgoing_segment = plan.segments[boundary_index]
        incoming_segment = plan.segments[boundary_index + 1]
        is_hard_cut = bool(
            transition.kind is TransitionKind.HARD_CUT
            and raw_record.get("transition_kind") == "hard_cut"
        )
        is_pose_dissolve = bool(
            transition.kind is TransitionKind.FACE_DISSOLVE
            and transition.reason == DIRECT_MEMBER_POSE_DISSOLVE_REASON
            and raw_record.get("transition_kind") == "face_dissolve"
        )
        if not (is_hard_cut or is_pose_dissolve):
            record_diagnostics["status"] = "unsupported_transition_kind"
            continue
        if outgoing_segment.source_index == incoming_segment.source_index:
            record_diagnostics["status"] = "same_source_boundary"
            continue
        geometry_authority = _decoded_camera_geometry_authority(
            raw_record,
            allow_official_fancam_subject_geometry=(
                allow_official_fancam_subject_geometry
            ),
        )
        record_diagnostics["geometry_authority"] = geometry_authority
        if geometry_authority is None:
            record_diagnostics["status"] = (
                "unverified_or_unbound_subject_geometry_reselection_required"
            )
            continue
        if (
            raw_record.get("outgoing_source_index")
            != outgoing_segment.source_index
            or raw_record.get("incoming_source_index")
            != incoming_segment.source_index
        ):
            record_diagnostics["status"] = "proxy_plan_source_mismatch"
            continue
        outgoing_geometry = _parsed_attention_geometry(
            raw_record,
            "outgoing",
        )
        incoming_geometry = _parsed_attention_geometry(
            raw_record,
            "incoming",
        )
        if outgoing_geometry is None or incoming_geometry is None:
            record_diagnostics["status"] = "focal_endpoint_geometry_missing"
            continue
        before_position = record_diagnostics[
            "before_attention_position_mismatch"
        ]
        before_scale = record_diagnostics["before_attention_scale_mismatch"]
        if (
            before_position is not None
            and before_scale is not None
            and before_position <= target_position + 1e-9
            and before_scale <= target_scale + 1e-9
        ):
            record_diagnostics["status"] = "already_inside_target_margin"
            continue
        outgoing_maximum = maximum_zooms.get(outgoing_segment.source_index)
        incoming_maximum = maximum_zooms.get(incoming_segment.source_index)
        if outgoing_maximum is None or incoming_maximum is None:
            record_diagnostics["status"] = "source_zoom_capacity_missing"
            continue
        outgoing_state = _camera_state_at(
            outgoing_segment,
            last_frame_times[boundary_index],
        )
        incoming_endpoint_time = (
            min(
                last_frame_times[boundary_index + 1],
                transition.frame_count / plan.fps,
            )
            if is_pose_dissolve
            else 0.0
        )
        incoming_state = _camera_state_at(
            incoming_segment,
            incoming_endpoint_time,
        )
        correction = _boundary_inverse_camera_states(
            outgoing_state=outgoing_state,
            incoming_state=incoming_state,
            outgoing_geometry=outgoing_geometry,
            incoming_geometry=incoming_geometry,
            outgoing_maximum_zoom=outgoing_maximum,
            incoming_maximum_zoom=incoming_maximum,
            target_position_mismatch=target_position,
            target_scale_mismatch=target_scale,
        )
        if correction is None:
            record_diagnostics["status"] = (
                "bounded_inverse_camera_solution_unavailable"
            )
            continue
        outgoing_refined, incoming_refined, prediction = correction
        record_diagnostics.update(
            {
                "status": "planned",
                "outgoing_before": outgoing_state.to_dict(),
                "incoming_before": incoming_state.to_dict(),
                "outgoing_after": outgoing_refined.to_dict(),
                "incoming_after": incoming_refined.to_dict(),
                "incoming_endpoint_time_seconds": incoming_endpoint_time,
                "overlap_transform_authority_required": is_pose_dissolve,
                **prediction,
            }
        )
        corrections[boundary_index] = (
            outgoing_refined,
            incoming_refined,
            record_diagnostics,
        )

    active_boundaries = set(corrections)
    refined_segments: tuple[RenderSegment, ...] = plan.segments
    while active_boundaries:
        candidate_segments: list[RenderSegment] = []
        failed_segment_indices: set[int] = set()
        for segment_index, segment in enumerate(plan.segments):
            entry_correction = corrections.get(segment_index - 1)
            exit_correction = corrections.get(segment_index)
            entry_state = (
                entry_correction[1]
                if segment_index - 1 in active_boundaries
                and entry_correction is not None
                else None
            )
            exit_state = (
                exit_correction[0]
                if segment_index in active_boundaries
                and exit_correction is not None
                else None
            )
            if entry_state is None and exit_state is None:
                candidate_segments.append(segment)
                continue
            try:
                entry_hold_seconds = (
                    plan.transitions[segment_index - 1].frame_count / plan.fps
                    if segment_index > 0
                    and segment_index - 1 in active_boundaries
                    and plan.transitions[segment_index - 1].kind.uses_overlap
                    else 0.0
                )
                track = _refined_endpoint_track(
                    segment,
                    last_frame_time=last_frame_times[segment_index],
                    entry_state=entry_state,
                    exit_state=exit_state,
                    ramp_seconds=ramp_seconds,
                    entry_hold_seconds=entry_hold_seconds,
                )
                candidate_segments.append(
                    replace(segment, editorial_transform=track)
                )
            except (TypeError, ValueError):
                failed_segment_indices.add(segment_index)
                candidate_segments.append(segment)
        if not failed_segment_indices:
            refined_segments = tuple(candidate_segments)
            break
        failed_boundaries = {
            boundary_index
            for boundary_index in active_boundaries
            if boundary_index in failed_segment_indices
            or boundary_index + 1 in failed_segment_indices
        }
        if not failed_boundaries:
            active_boundaries.clear()
            break
        for boundary_index in failed_boundaries:
            boundary_diagnostics[boundary_index]["status"] = (
                "track_interpolation_validation_failed"
            )
        active_boundaries.difference_update(failed_boundaries)

    if active_boundaries:
        refined_transitions = tuple(
            (
                replace(
                    transition,
                    transform_overlap_authority=(
                        MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY
                    ),
                )
                if boundary_index in active_boundaries
                and transition.kind is TransitionKind.FACE_DISSOLVE
                else transition
            )
            for boundary_index, transition in enumerate(plan.transitions)
        )
        try:
            refined_plan = replace(
                plan,
                segments=refined_segments,
                transitions=refined_transitions,
            )
        except (TypeError, ValueError) as exc:
            diagnostics["status"] = "refined_render_plan_validation_failed"
            diagnostics["validation_error"] = str(exc)
            active_boundaries.clear()
            refined_plan = plan
        else:
            diagnostics["status"] = "applied"
            diagnostics["applied"] = True
            for boundary_index in active_boundaries:
                boundary_diagnostics[boundary_index]["status"] = "applied"
    else:
        refined_plan = plan
        diagnostics["status"] = "no_bounded_focal_endpoint_solution"

    diagnostics["refined_boundary_indices"] = sorted(active_boundaries)
    diagnostics["refined_overlap_boundary_indices"] = sorted(
        boundary_index
        for boundary_index in active_boundaries
        if plan.transitions[boundary_index].kind is TransitionKind.FACE_DISSOLVE
    )
    diagnostics["unresolved_boundary_indices"] = sorted(
        boundary_index
        for boundary_index, item in boundary_diagnostics.items()
        if item["status"]
        in {
            "unverified_or_unbound_subject_geometry_reselection_required",
            "bounded_inverse_camera_solution_unavailable",
            "focal_endpoint_geometry_missing",
            "source_zoom_capacity_missing",
            "track_interpolation_validation_failed",
        }
    )
    diagnostics["boundaries"] = [
        boundary_diagnostics[index] for index in sorted(boundary_diagnostics)
    ]
    return (refined_plan, diagnostics)


def render_and_review_direct_member_proxy(
    plan: RenderPlan,
    proxy_output: Path,
    *,
    render_review: Callable[..., tuple[RenderPlan, RenderedProxyReview]]
    | None = None,
    continuity_evaluator: Callable[..., DirectMemberRenderedContinuityResult]
    | None = None,
    focal_reference_candidates_by_boundary_index: Mapping[
        int, tuple[FocalFaceCandidate, FocalFaceCandidate]
    ] | None = None,
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
        focal_reference_candidates_by_boundary_index=(
            focal_reference_candidates_by_boundary_index
        ),
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
            "focal_reference_policy": (
                "reuse_exact_transient_dense_planner_boundary_face_pair"
            ),
            "focal_reference_embeddings_persisted": False,
        },
    }
    return (proxy_plan, DirectMemberRenderedProxyPreflight(payload))


__all__ = [
    "ANALYSIS_BASIS",
    "CAMERA_REFINEMENT_ANALYSIS_BASIS",
    "OUTPUT_VERIFIED_DISSOLVE_ANALYSIS_BASIS",
    "OUTPUT_VERIFIED_DISSOLVE_DEMOTION_ANALYSIS_BASIS",
    "DirectMemberRenderedProxyPreflight",
    "demote_failed_direct_member_pose_dissolves_from_rendered_proxy",
    "promote_direct_member_pose_dissolves_from_rendered_proxy",
    "refine_direct_member_camera_plan_from_rendered_proxy",
    "render_and_review_direct_member_proxy",
]
