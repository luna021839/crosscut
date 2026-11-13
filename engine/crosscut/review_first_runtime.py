"""Bounded review-first runtime and final artifact disposition.

The professional selector and QA stack are still allowed to reject a file as a
release.  They must not, however, turn every editorial quality miss into an
unbounded re-search that produces no inspectable result.  This module keeps the
two decisions explicit:

* technical integrity failures are never exposed;
* a content-bound, decodable render may be exposed as ``review_ready``;
* only a complete QA pass is ``approved`` for delivery.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
import math
import os
import signal
import threading
from time import monotonic
from typing import Any, Callable, Iterator, Mapping, Sequence


REVIEW_FIRST_RUNTIME_CONTRACT = "ive_review_first_runtime_v1"
REVIEW_FIRST_TOTAL_BUDGET_SECONDS = 600.0
_owned_native_deadline_deferral = ContextVar("owned_native_deadline_deferral", default=None)


@contextmanager
def defer_deadline_during_owned_native_wait():
    """Our alarm may not interrupt the independently bounded worker's cleanup.

    This only defers this module's deadline exception, not user cancellation
    or foreign signal handlers. Native wait/guardian still enforce their own
    absolute cutoff, and the CLI supervisor still enforces the total ceiling.
    The deferred exception is re-raised after owned pipes/processes are closed.
    """
    pending = []
    token = _owned_native_deadline_deferral.set(pending)
    try:
        yield
    finally:
        _owned_native_deadline_deferral.reset(token)
        if pending:
            raise ReviewFirstDeadlineExceeded(pending[0])


class ReviewFirstStatus(str, Enum):
    RUNNING = "RUNNING"
    REVIEW_READY = "REVIEW_READY"
    APPROVED = "APPROVED"
    INPUT_UNUSABLE = "INPUT_UNUSABLE"


class ReviewFirstDeadlineExceeded(BaseException):
    """Non-recoverable workflow deadline; only the application boundary handles it.

    This intentionally bypasses broad ``Exception``/``ValueError`` recovery
    inside analysis and cache code.  Those paths may downgrade optional model
    failures, but must never swallow the user-facing total runtime ceiling.
    """


class ReviewFirstSearchCutoff(BaseException):
    """Control flow, not a per-candidate model error.

    Broad adapter/cache ``except Exception`` blocks must not hide the cutoff
    and start the next candidate. Only the first-result controller handles it.
    """


def _deadline_stage(stage: str) -> str:
    return str(stage).strip() or "unknown_stage"


@contextmanager
def enforce_review_first_deadline(
    absolute_deadline_monotonic_seconds: float | None,
    *,
    stage: str,
) -> Iterator[None]:
    """Interrupt a synchronous review-first workflow at one absolute deadline.

    POSIX main-thread alarms interrupt Python execution and interruptible
    waits. Native extensions may defer a signal indefinitely; the CLI's
    independent workflow supervisor supplies the actual process-level ceiling.
    Other environments here have before/after checks, not a hard native limit.
    """

    if absolute_deadline_monotonic_seconds is None:
        yield
        return
    if isinstance(absolute_deadline_monotonic_seconds, bool) or not isinstance(
        absolute_deadline_monotonic_seconds,
        (int, float),
    ):
        raise TypeError("review-first absolute deadline must be numeric or None")
    deadline = float(absolute_deadline_monotonic_seconds)
    if not math.isfinite(deadline):
        raise ValueError("review-first absolute deadline must be finite")
    stage_name = _deadline_stage(stage)
    now = monotonic()
    if deadline <= now:
        raise ReviewFirstDeadlineExceeded(
            f"review-first total workflow deadline exceeded during {stage_name}"
        )

    alarm_supported = bool(
        os.name == "posix"
        and threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "ITIMER_REAL")
        and hasattr(signal, "setitimer")
        and hasattr(signal, "getitimer")
    )
    if not alarm_supported:
        yield
        if monotonic() >= deadline:
            raise ReviewFirstDeadlineExceeded(
                f"review-first total workflow deadline exceeded during {stage_name}"
            )
        return

    entered_at = monotonic()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    previous_due = (
        entered_at + float(previous_delay) if previous_delay > 0.0 else None
    )
    previous_alarm_fired = False

    def _deadline_alarm(signum: int, frame: object) -> None:
        nonlocal previous_alarm_fired, previous_due
        fired_at = monotonic()
        deadline_due = fired_at >= deadline - 1e-6
        previous_is_due = bool(
            previous_due is not None and fired_at >= previous_due - 1e-6
        )
        if deadline_due and (
            not previous_is_due
            or previous_due is None
            or deadline <= previous_due + 1e-6
        ):
            pending = _owned_native_deadline_deferral.get()
            if pending is not None:
                pending.append("review-first workflow deadline exceeded during " + stage_name)
                return
            raise ReviewFirstDeadlineExceeded(
                "review-first total workflow deadline exceeded during "
                f"{stage_name}"
            )
        if previous_is_due:
            previous_alarm_fired = True
            if previous_interval > 0.0 and previous_due is not None:
                while previous_due <= fired_at:
                    previous_due += previous_interval
            else:
                previous_due = None
            if callable(previous_handler):
                previous_handler(signum, frame)
            elif previous_handler == signal.SIG_DFL:
                raise KeyboardInterrupt
        next_due = deadline
        if previous_due is not None:
            next_due = min(next_due, previous_due)
        signal.setitimer(
            signal.ITIMER_REAL,
            max(1e-6, next_due - monotonic()),
        )

    signal.signal(signal.SIGALRM, _deadline_alarm)
    first_due = deadline if previous_due is None else min(deadline, previous_due)
    signal.setitimer(
        signal.ITIMER_REAL,
        max(1e-6, first_due - monotonic()),
    )
    try:
        yield
        if monotonic() >= deadline:
            raise ReviewFirstDeadlineExceeded(
                f"review-first total workflow deadline exceeded during {stage_name}"
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_delay > 0.0:
            elapsed = max(0.0, monotonic() - entered_at)
            if (
                previous_interval > 0.0
                and previous_alarm_fired
                and previous_due is not None
            ):
                restored_delay = max(1e-6, previous_due - monotonic())
            elif previous_alarm_fired:
                restored_delay = 0.0
            else:
                restored_delay = max(1e-6, previous_delay - elapsed)
            if restored_delay > 0.0:
                signal.setitimer(
                    signal.ITIMER_REAL,
                    restored_delay,
                    previous_interval,
                )


# These producer codes cover both a measured quality miss and a missing value.
# Only the former is editorial; the evidence check below keeps missing/invalid
# measurements blocking instead of treating this as an unconditional allowlist.
_GROUP_PROXY_MEASUREMENT_CODES = {
    "proxy_quality_overall_quality_p10": ("overall_quality_p10", 1.0, "minimum"),
    "proxy_quality_source_quality_median_span": ("source_quality_median_span", 1.0, "maximum"),
    "proxy_quality_boundary_score_p10": ("boundary_score_p10", 100.0, "minimum"),
    "proxy_quality_proxy_weighted_score": ("weighted_score", 100.0, "minimum"),
    "proxy_subject_geometry_coverage_incomplete": ("subject_geometry_coverage", 1.0, "coverage"),
    "proxy_closeup_face_coverage_incomplete": ("closeup_face_coverage", 1.0, "coverage"),
}

_GROUP_EDITORIAL_CODES = frozenset(
    {
        "normal_interval_review_quality_unapproved",
        "performer_family_quality_unapproved",
        "boundary_professional_combined_evidence_gate_failed",
        "decoded_proxy_review_failed",
        "dominant_source_share_above_preferred",
        "dominant_source_share_hard",
        "editorial_event_spacing_below_minimum",
        "editorial_motion_unavailable",
        "editorial_zoom_target_exceeded",
        "editorial_zoom_target_underfilled",
        "insufficient_post_transition_review_handle",
        "insufficient_source_diversity",
        "ive_product_quality_floor_failed",
        "perceptual_chroma_high_step_pumping",
        "perceptual_detail_retention_median",
        "perceptual_detail_retention_p10",
        "perceptual_exact_structure_similarity_median",
        "perceptual_exact_structure_similarity_p10",
        "perceptual_luma_alternating_pumping",
        "perceptual_luma_high_step_pumping",
        "perceptual_rapid_quality_alternation",
        "perceptual_sharpness_alternating_pumping",
        "perceptual_sharpness_high_step_pumping",
        "perceptual_source_chroma_median_span",
        "perceptual_source_luma_median_span",
        "perceptual_source_sharpness_median_span",
        "production_rendered_montage_grammar_required_gate_failed",
        "protected_section_last_beat_cut",
        "proxy_boundary_issues_require_manual_review",
        "risky_source_ping_pong_present",
        "short_source_flash_present",
        "source_local_content_reused",
        "source_look_match_unavailable",
        "source_run_exceeds_hard_maximum",
        "source_run_median_outside_professional_band",
        "source_switch_density_below_hard_minimum",
        "source_switch_density_outside_professional_band",
        "source_switch_ratio_outside_professional_band",
        "timeline_hard_failures_present",
        "timeline_review_failed",
        "timeline_warnings_present",
        "used_source_look_evidence_unstable",
        "used_source_look_samples_insufficient",
        "visible_boundary_repair_evidence_missing",
        "visible_cut_far_from_beat_grid",
        "visual_shot_median_too_slow",
    } | _GROUP_PROXY_MEASUREMENT_CODES.keys()
)
_MEMBER_EDITORIAL_CODES = frozenset(
    {
        "member_look_calibration_predicted_quality_not_pass",
        "member_perceptual_quality_not_pass",
        "member_rendered_continuity_not_pass",
    }
)
_MEMBER_CONTINUITY_EDITORIAL_REASONS = frozenset({
    "no_rendered_source_switch_boundaries",
    "rendered_actual_pose_coverage_below_floor", "rendered_attention_coverage_below_floor",
    "rendered_focal_reference_coverage_below_floor",
    "rendered_attention_position_p95_above_floor", "rendered_attention_scale_p95_above_floor",
    "rendered_pose_p95_above_floor", "rendered_silhouette_p95_above_floor",
    "rendered_boundary_attention_missing", "rendered_boundary_shape_evidence_missing",
    "rendered_boundary_position_discontinuity", "rendered_boundary_scale_discontinuity",
    "rendered_boundary_pose_discontinuity", "rendered_boundary_silhouette_discontinuity",
    "rendered_boundary_motion_direction_opposed", "rendered_boundary_motion_speed_discontinuity",
    "rendered_boundary_pose_phase_discontinuity", "rendered_effect_endpoint_attention_missing",
    "rendered_effect_endpoint_shape_evidence_missing", "rendered_effect_endpoint_position_discontinuity",
    "rendered_effect_endpoint_scale_discontinuity", "rendered_effect_endpoint_pose_discontinuity",
    "rendered_effect_endpoint_silhouette_discontinuity",
})
MEMBER_PRE_RENDER_EDITORIAL_REASONS = frozenset({
    "visible_switch_density_outside_professional_range", "missing_actual_pose", "pose_over_hard_limit",
    "missing_silhouette", "silhouette_over_hard_limit", "missing_attention_geometry",
    "incompatible_attention_geometry", "unrepaired_attention_geometry", "opposed_or_missing_motion_direction",
    "incompatible_or_missing_motion_shape", "incompatible_motion_speed", "incompatible_pose_phase",
    "uncontracted_severe_boundary", "pose_mismatch_p95_above_professional_floor",
    "silhouette_mismatch_p95_above_professional_floor", "attention_position_p95_above_professional_floor",
    "attention_scale_p95_above_professional_floor", "motion_speed_p95_above_professional_floor",
    "pose_phase_p95_above_professional_floor", "editorial_camera_density_below_professional_floor",
    "native_output_portrait_duration_fraction_above_quality_floor",
    "highlight_cut_density_gate_missing_or_failed", "requested_safe_transition_effect_not_materialized",
})


def _issue_codes(
    report_payload: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw_issues = report_payload.get("issues")
    if not isinstance(raw_issues, Sequence) or isinstance(
        raw_issues, (str, bytes, bytearray)
    ):
        return (), ("qa_issue_list_invalid",)
    result: list[str] = []
    malformed: list[str] = []
    for index, issue in enumerate(raw_issues):
        if not isinstance(issue, Mapping):
            malformed.append(f"qa_issue_{index}_invalid")
            continue
        code = issue.get("code")
        if isinstance(code, str) and code.strip():
            result.append(code.strip())
        else:
            malformed.append(f"qa_issue_{index}_code_invalid")
    return tuple(dict.fromkeys(result)), tuple(malformed)


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _group_proxy_measurement_evidence_valid(
    code: str, report_payload: Mapping[str, Any],
) -> bool:
    """A valid decoded measurement may miss a quality target without corruption."""
    from .proxy_review_contract import proxy_integrity_evidence_valid

    metrics = _mapping((_mapping(report_payload.get("metrics")) or {}).get("proxy_quality_and_geometry"))
    if metrics is None or not proxy_integrity_evidence_valid(metrics.get("technical_integrity")):
        return False
    field_name, upper, gate = _GROUP_PROXY_MEASUREMENT_CODES[code]
    value = metrics.get(field_name)

    def finite_in_range(number: object) -> bool:
        return type(number) in (int, float) and math.isfinite(number) and 0 <= number <= upper

    if not finite_in_range(value):
        return False
    issues = report_payload.get("issues")
    matching = [issue for issue in issues if isinstance(issue, Mapping) and issue.get("code") == code]
    if not matching:
        return False
    # Validate every occurrence, not just the first duplicate with this code.
    for issue in matching:
        threshold = issue.get("threshold")
        if (not finite_in_range(issue.get("value")) or issue["value"] != value
                or not finite_in_range(threshold)):
            return False
        if gate == "minimum" and not value < threshold:
            return False
        if gate == "maximum" and not value > threshold:
            return False
        if gate == "coverage" and not (threshold == 1.0 and value < 1.0):
            return False
    if gate == "coverage":
        integrity_review = metrics["technical_integrity"]["review"]
        total = integrity_review["reviewed_boundary_count"]
        denominator = (metrics.get("closeup_relevant_boundary_count")
                       if field_name == "closeup_face_coverage" else total)
        if (metrics.get("boundary_sample_contract_valid") is not True
                or type(denominator) is not int or not 0 < denominator <= total):
            return False
        covered = value * denominator
        if not math.isclose(covered, round(covered), abs_tol=1e-8):
            return False
    return True


def _member_continuity_measurements_valid(continuity: Mapping[str, Any]) -> bool:
    from .member_continuity_contract import member_continuity_measurements_valid
    return member_continuity_measurements_valid(continuity)


def _member_editorial_evidence_valid(
    code: str,
    report_payload: Mapping[str, Any],
) -> bool:
    metrics = _mapping(report_payload.get("metrics"))
    if metrics is None:
        return False
    if code == "member_look_calibration_predicted_quality_not_pass":
        look = _mapping(metrics.get("member_look_calibration"))
        provenance = tuple(
            look[name]
            for name in ("provenance_valid", "cache_only_provenance_valid")
            if look is not None and name in look
        )
        return bool(
            look is not None
            and look.get("locations_bound") is True
            and provenance
            and all(value is True for value in provenance)
            and look.get("segment_binding_valid") is True
            and look.get("predicted_evidence_valid") is True
            and look.get("predicted_quality_valid") is False
        )
    if code == "member_rendered_continuity_not_pass":
        continuity = _mapping(metrics.get("rendered_continuity"))
        reasons = continuity.get("failure_reasons") if continuity is not None else None
        return bool(
            continuity is not None
            and _member_continuity_measurements_valid(continuity)
            and continuity.get("passed") is False
            and isinstance(reasons, list) and reasons
            and all(isinstance(reason, str) and reason in _MEMBER_CONTINUITY_EDITORIAL_REASONS for reason in reasons)
        )
    if code == "member_perceptual_quality_not_pass":
        perceptual = _mapping(metrics.get("perceptual_quality"))
        measurements = _mapping(perceptual.get("metrics")) if perceptual is not None else None
        video = _mapping(measurements.get("video")) if measurements is not None else None
        nested_codes, malformed = _issue_codes(perceptual) if perceptual is not None else ((), ("missing",))
        return bool(
            perceptual is not None
            and perceptual.get("schema_version") == 2
            and perceptual.get("analysis") == "full_final_decode_temporal_pumping_exact_source_retention_and_structure_binding_v2"
            and perceptual.get("verdict") in {"CONDITIONAL", "FAIL"}
            and perceptual.get("passed") is False
            and isinstance(perceptual.get("issues"), list)
            and bool(perceptual.get("issues"))
            and nested_codes and not malformed
            and all(nested_code in _GROUP_EDITORIAL_CODES and nested_code.startswith("perceptual_") for nested_code in nested_codes)
            and video is not None and video.get("full_frame_coverage") is True
            and video.get("dimensions_match_plan") is True and video.get("fps_matches_plan") is True
            and type(video.get("decoded_frame_count")) is int and video["decoded_frame_count"] > 0
            and video["decoded_frame_count"] == video.get("expected_frame_count")
        )
    return False


def member_proxy_technical_issues(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Admit a proxy to final QA without turning quality failure into approval."""
    from .proxy_review_contract import proxy_integrity_evidence
    generic = payload.get("rendered_proxy_review")
    if not isinstance(generic, Mapping):
        return ("member_proxy_review_missing",)
    issues = list(proxy_integrity_evidence(generic)["technical_issue_codes"])
    quality = payload.get("member_output_quality_failure_reasons")
    editorial_quality = {"member_proxy_overall_quality_p10", "member_proxy_source_quality_median_span",
                         "member_proxy_perceptual_source_admission"}
    if not isinstance(quality, list) or any(not isinstance(code, str) for code in quality):
        issues.append("member_proxy_quality_reasons_invalid")
    else:
        issues.extend(code for code in quality if code not in editorial_quality)
    continuity = payload.get("rendered_continuity")
    if not isinstance(continuity, Mapping) or type(continuity.get("passed")) is not bool:
        issues.append("member_proxy_continuity_invalid")
    else:
        # The same measurement contract is required on both the PASS and FAIL
        # paths. A bare passed=True must not bypass missing/contradictory data.
        reasons = continuity.get("failure_reasons")
        if continuity["passed"] is True:
            if reasons != [] or continuity.get("failure_boundary_indices") != {}:
                issues.append("member_proxy_continuity_passed_with_failures")
            if not _member_continuity_measurements_valid(continuity):
                issues.append("member_proxy_continuity_measurements_invalid")
        elif not _member_editorial_evidence_valid(
            "member_rendered_continuity_not_pass", {"metrics": {"rendered_continuity": continuity}}
        ):
            issues.append("member_proxy_continuity_technical_or_unknown_failure")
    return tuple(dict.fromkeys(issues))


@dataclass(frozen=True, slots=True)
class ReviewFirstQADisposition:
    status: ReviewFirstStatus
    artifact_role: str
    video_output_exposed: bool
    delivery_completion_message_allowed: bool
    technical_issue_codes: tuple[str, ...] = ()
    editorial_issue_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": REVIEW_FIRST_RUNTIME_CONTRACT,
            "status": self.status.value,
            "artifact_role": self.artifact_role,
            "video_output_exposed": self.video_output_exposed,
            "delivery_completion_message_allowed": (
                self.delivery_completion_message_allowed
            ),
            "technical_issue_codes": list(self.technical_issue_codes),
            "editorial_issue_codes": list(self.editorial_issue_codes),
        }


def artifact_disposition_payload_valid(payload: object, *, role: str) -> bool:
    """Shared Python/native wire contract; a status label alone grants nothing."""
    if not isinstance(payload, Mapping):
        return False
    technical, editorial = payload.get("technical_issue_codes"), payload.get("editorial_issue_codes")
    if (payload.get("contract") != REVIEW_FIRST_RUNTIME_CONTRACT
            or payload.get("artifact_role") != role
            or payload.get("video_output_exposed") is not True
            or not isinstance(technical, list) or technical
            or not isinstance(editorial, list)
            or any(not isinstance(code, str) or not code.strip() for code in editorial)):
        return False
    if role == "delivery":
        return (payload.get("status") == "APPROVED"
                and payload.get("delivery_completion_message_allowed") is True and not editorial)
    if role == "review_ready":
        return (payload.get("status") == "REVIEW_READY"
                and payload.get("delivery_completion_message_allowed") is False and bool(editorial))
    return False


def finalize_review_first_runtime_payload(payload: object, status: ReviewFirstStatus) -> None:
    """Record end-to-end elapsed time, including mandatory final QA."""
    if not isinstance(payload, dict):
        return
    payload["status"] = status.value
    deadline = payload.get("absolute_deadline_monotonic_seconds")
    budgets = payload.get("editorial_search_budget_seconds")
    total = budgets.get("absolute_search_ceiling") if isinstance(budgets, Mapping) else None
    if isinstance(deadline, (int, float)) and isinstance(total, (int, float)):
        now = monotonic()
        payload["elapsed_seconds"] = max(0.0, now - (deadline - total))
        payload["remaining_total_seconds"] = max(0.0, deadline - now)
        payload["elapsed_includes_final_qa"] = True


def _normal_interval_editorial_evidence_valid(report):
    from .group_normal_interval_owner import qa_metrics_valid
    metrics = (_mapping(report.get('metrics')) or {}).get('normal_interval_review')
    return qa_metrics_valid(metrics)


def _performer_family_editorial_evidence_valid(report):
    from .group_performer_family_owner import qa_metrics_valid
    return qa_metrics_valid((_mapping(report.get('metrics')) or {}).get('performer_family_review'))


def classify_review_first_qa(
    report_payload: Mapping[str, Any],
    *,
    content_binding_passed: bool,
    review_first_enabled: bool,
    member: bool,
) -> ReviewFirstQADisposition:
    """Separate an inspectable draft from an approved product.

    ``content_binding_passed`` is deliberately external: callers already own
    the strict immutable-file checks.  A report that failed without emitting a
    typed issue is treated as technical/unknown rather than silently exposed.
    """

    if not all(
        isinstance(value, bool)
        for value in (content_binding_passed, review_first_enabled, member)
    ):
        raise TypeError("review-first QA flags must be booleans")
    if not isinstance(report_payload, Mapping):
        report_payload = {}
    codes, malformed_codes = _issue_codes(report_payload)
    passed_value = report_payload.get("passed")
    passed_contract_valid = isinstance(passed_value, bool)
    report_passed = passed_value is True
    editorial_catalog = _MEMBER_EDITORIAL_CODES if member else _GROUP_EDITORIAL_CODES
    from .proxy_review_contract import proxy_integrity_evidence_valid
    proxy_metrics = _mapping((_mapping(report_payload.get("metrics")) or {}).get("proxy_quality_and_geometry"))
    proxy_codes = {"decoded_proxy_review_failed", "proxy_boundary_issues_require_manual_review"}
    proxy_integrity_valid = bool(proxy_metrics is not None and
        proxy_integrity_evidence_valid(proxy_metrics.get("technical_integrity")))
    member_continuity = (_mapping(report_payload.get("metrics")) or {}).get("rendered_continuity")
    member_continuity_valid = bool(isinstance(member_continuity, Mapping)
        and _member_continuity_measurements_valid(member_continuity))
    editorial = tuple(
        code
        for code in codes
        if code in editorial_catalog
        and (code != 'normal_interval_review_quality_unapproved'
             or _normal_interval_editorial_evidence_valid(report_payload))
        and (code != 'performer_family_quality_unapproved'
             or _performer_family_editorial_evidence_valid(report_payload))
        and (member or code not in proxy_codes or proxy_integrity_valid)
        and (member or code not in _GROUP_PROXY_MEASUREMENT_CODES
             or _group_proxy_measurement_evidence_valid(code, report_payload))
        and (
            not member
            or _member_editorial_evidence_valid(code, report_payload)
        )
    )
    technical = tuple(
        dict.fromkeys(
            (
                *malformed_codes,
                *(code for code in codes if code not in editorial),
                *(() if passed_contract_valid else ("qa_passed_flag_invalid",)),
                *(() if content_binding_passed else ("content_binding_failed",)),
                *(("normal_interval_unapproved_quality_issue_missing",)
                  if (_mapping(report_payload.get("metrics")) or {}).get("normal_interval_review") is not None
                  and "normal_interval_review_quality_unapproved" not in codes else ()),
                *(("performer_family_unapproved_quality_issue_missing",)
                  if (_mapping(report_payload.get("metrics")) or {}).get("performer_family_review") is not None
                  and "performer_family_quality_unapproved" not in codes else ()),
                *(("proxy_nested_technical_integrity_invalid",)
                  if not member and not proxy_integrity_valid else ()),
                *(("member_continuity_measurements_invalid",)
                  if member and not member_continuity_valid else ()),
                *(("member_continuity_failure_not_reported",)
                  if member and member_continuity_valid and member_continuity.get("passed") is False
                  and "member_rendered_continuity_not_pass" not in codes else ()),
                *(
                    ("qa_passed_with_issues",)
                    if report_passed and codes
                    else ()
                ),
            )
        )
    )

    if report_passed and content_binding_passed and not technical and not codes:
        return ReviewFirstQADisposition(
            status=ReviewFirstStatus.APPROVED,
            artifact_role="delivery",
            video_output_exposed=True,
            delivery_completion_message_allowed=True,
            technical_issue_codes=technical,
            editorial_issue_codes=editorial,
        )
    reviewable = bool(
        review_first_enabled
        and content_binding_passed
        and codes
        and not technical
        and not report_passed
    )
    if reviewable:
        return ReviewFirstQADisposition(
            status=ReviewFirstStatus.REVIEW_READY,
            artifact_role="review_ready",
            video_output_exposed=True,
            delivery_completion_message_allowed=False,
            editorial_issue_codes=editorial,
        )
    return ReviewFirstQADisposition(
        status=ReviewFirstStatus.INPUT_UNUSABLE,
        artifact_role="rejected",
        video_output_exposed=False,
        delivery_completion_message_allowed=False,
        technical_issue_codes=technical
        or (("unknown_qa_failure",) if not codes or report_passed else ()),
        editorial_issue_codes=editorial,
    )


@dataclass(slots=True)
class ReviewFirstRuntime:
    """Monotonic hard ceiling and audit clock for the user-facing workflow."""

    enabled: bool
    total_budget_seconds: float = REVIEW_FIRST_TOTAL_BUDGET_SECONDS
    first_render_budget_seconds: float = 360.0
    refinement_stop_seconds: float = 540.0
    absolute_deadline_monotonic_seconds: float | None = None
    _clock: Callable[[], float] = field(default=monotonic, repr=False)
    _started_at: float = field(init=False, repr=False)
    _checkpoints: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        budgets = (
            self.first_render_budget_seconds,
            self.refinement_stop_seconds,
            self.total_budget_seconds,
        )
        if not all(
            not isinstance(value, bool) and isinstance(value, (int, float))
            for value in budgets
        ):
            raise TypeError("review-first budgets must be numeric")
        if not all(math.isfinite(float(value)) for value in budgets):
            raise ValueError("review-first budgets must be finite")
        if not (
            0 < self.first_render_budget_seconds
            <= self.refinement_stop_seconds
            <= self.total_budget_seconds
        ):
            raise ValueError("review-first budgets must be positive and ordered")
        now = float(self._clock())
        if self.absolute_deadline_monotonic_seconds is not None:
            if isinstance(
                self.absolute_deadline_monotonic_seconds,
                bool,
            ) or not isinstance(
                self.absolute_deadline_monotonic_seconds,
                (int, float),
            ):
                raise TypeError(
                    "review-first absolute deadline must be numeric or None"
                )
            deadline = float(self.absolute_deadline_monotonic_seconds)
            if not math.isfinite(deadline):
                raise ValueError("review-first absolute deadline must be finite")
            self.absolute_deadline_monotonic_seconds = deadline
            self._started_at = deadline - float(self.total_budget_seconds)
        else:
            self._started_at = now

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, float(self._clock()) - self._started_at)

    @property
    def deadline_monotonic_seconds(self) -> float:
        if self.absolute_deadline_monotonic_seconds is not None:
            return self.absolute_deadline_monotonic_seconds
        return self._started_at + float(self.total_budget_seconds)

    @property
    def remaining_total_seconds(self) -> float:
        return max(
            0.0,
            self.deadline_monotonic_seconds - float(self._clock()),
        )

    @property
    def optional_refinement_allowed(self) -> bool:
        return bool(
            self.enabled
            and self.elapsed_seconds < min(self.first_render_budget_seconds, self.refinement_stop_seconds)
            and self.remaining_total_seconds > 0.0
        )

    def checkpoint(self, stage: str) -> None:
        name = str(stage).strip()
        if not name:
            raise ValueError("review-first checkpoint stage cannot be blank")
        self._checkpoints.append(
            {
                "stage": name,
                "elapsed_seconds": self.elapsed_seconds,
                "remaining_total_seconds": self.remaining_total_seconds,
            }
        )
        self.require_total_budget(name)

    def require_total_budget(self, stage: str) -> None:
        if self.enabled and self.remaining_total_seconds <= 0.0:
            raise ReviewFirstDeadlineExceeded(
                "review-first total workflow deadline exceeded during "
                f"{_deadline_stage(stage)}"
            )

    def bounded_timeout(self, maximum_seconds: float, stage: str) -> float:
        if not isinstance(maximum_seconds, (int, float)):
            raise TypeError("maximum timeout must be numeric")
        maximum = float(maximum_seconds)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum timeout must be positive and finite")
        if not self.enabled:
            return maximum
        self.require_total_budget(stage)
        return min(maximum, self.remaining_total_seconds)

    def require_first_render_budget(self, stage: str) -> None:
        """Stop optional search, while leaving the baseline render available."""

        self.require_total_budget(stage)
        if self.enabled and self.elapsed_seconds >= self.first_render_budget_seconds:
            raise ReviewFirstSearchCutoff(
                "review-first optional search exceeded the first-render budget "
                f"during {str(stage).strip() or 'unknown_stage'}"
            )

    def to_dict(
        self,
        *,
        status: ReviewFirstStatus = ReviewFirstStatus.RUNNING,
    ) -> dict[str, Any]:
        if not isinstance(status, ReviewFirstStatus):
            raise TypeError("status must be ReviewFirstStatus")
        return {
            "contract": REVIEW_FIRST_RUNTIME_CONTRACT,
            "enabled": self.enabled,
            "status": status.value,
            "elapsed_seconds": self.elapsed_seconds,
            "absolute_deadline_monotonic_seconds": (
                self.deadline_monotonic_seconds if self.enabled else None
            ),
            "remaining_total_seconds": self.remaining_total_seconds,
            "editorial_search_budget_seconds": {
                "stop_candidate_search_for_first_render": (
                    self.first_render_budget_seconds
                ),
                "stop_optional_refinement": self.refinement_stop_seconds,
                "absolute_search_ceiling": self.total_budget_seconds,
            },
            "encoder_runtime_limits_seconds": {
                "proxy": 420.0,
                "final": 900.0,
            },
            "optional_refinement_allowed": self.optional_refinement_allowed,
            "checkpoints": [dict(item) for item in self._checkpoints],
        }
