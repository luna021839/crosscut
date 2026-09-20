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


def _deadline_stage(stage: str) -> str:
    return str(stage).strip() or "unknown_stage"


@contextmanager
def enforce_review_first_deadline(
    absolute_deadline_monotonic_seconds: float | None,
    *,
    stage: str,
) -> Iterator[None]:
    """Interrupt a synchronous review-first workflow at one absolute deadline.

    On the POSIX main thread this uses a process alarm, so a stalled acquisition,
    model call, callback, or subprocess wait cannot silently turn the nominal
    budget into an advisory value.  Other environments retain fail-closed
    before/after checks.
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


_GROUP_EDITORIAL_CODES = frozenset(
    {
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
    }
)
_MEMBER_EDITORIAL_CODES = frozenset(
    {
        "member_look_calibration_predicted_quality_not_pass",
        "member_perceptual_quality_not_pass",
        "member_rendered_continuity_not_pass",
    }
)


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


def _member_editorial_evidence_valid(
    code: str,
    report_payload: Mapping[str, Any],
) -> bool:
    metrics = _mapping(report_payload.get("metrics"))
    if metrics is None:
        return False
    if code == "member_look_calibration_predicted_quality_not_pass":
        look = _mapping(metrics.get("member_look_calibration"))
        return bool(
            look is not None
            and look.get("locations_bound") is True
            and look.get("provenance_valid") is True
            and look.get("segment_binding_valid") is True
            and look.get("predicted_quality_valid") is False
        )
    if code == "member_rendered_continuity_not_pass":
        continuity = _mapping(metrics.get("rendered_continuity"))
        return bool(
            continuity is not None
            and continuity.get("schema_version") == 1
            and isinstance(continuity.get("analysis"), str)
            and continuity.get("passed") is False
            and isinstance(continuity.get("failure_reasons"), list)
            and isinstance(continuity.get("failure_boundary_indices"), Mapping)
            and isinstance(continuity.get("metrics"), Mapping)
            and isinstance(continuity.get("boundaries"), list)
        )
    if code == "member_perceptual_quality_not_pass":
        perceptual = _mapping(metrics.get("perceptual_quality"))
        return bool(
            perceptual is not None
            and perceptual.get("schema_version") == 2
            and isinstance(perceptual.get("analysis"), str)
            and perceptual.get("verdict") in {"CONDITIONAL", "FAIL"}
            and perceptual.get("passed") is False
            and isinstance(perceptual.get("issues"), list)
            and bool(perceptual.get("issues"))
            and isinstance(perceptual.get("metrics"), Mapping)
        )
    return False


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
    codes, malformed_codes = _issue_codes(report_payload)
    passed_value = report_payload.get("passed")
    passed_contract_valid = isinstance(passed_value, bool)
    report_passed = passed_value is True
    editorial_catalog = _MEMBER_EDITORIAL_CODES if member else _GROUP_EDITORIAL_CODES
    editorial = tuple(
        code
        for code in codes
        if code in editorial_catalog
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
            and self.elapsed_seconds < self.refinement_stop_seconds
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
            raise ReviewFirstDeadlineExceeded(
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
