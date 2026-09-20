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

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any, Callable, Mapping, Sequence


REVIEW_FIRST_RUNTIME_CONTRACT = "ive_review_first_runtime_v1"


class ReviewFirstStatus(str, Enum):
    RUNNING = "RUNNING"
    REVIEW_READY = "REVIEW_READY"
    APPROVED = "APPROVED"
    INPUT_UNUSABLE = "INPUT_UNUSABLE"


class ReviewFirstDeadlineExceeded(ValueError):
    """Optional selection/refinement exceeded the first-render deadline."""


_GROUP_TECHNICAL_CODES = frozenset(
    {
        "artifact_content_changed_during_qa",
        "artifact_content_fingerprint_unavailable",
        "artifact_content_stability_unavailable",
        "artifact_json_unavailable_or_invalid",
        "ffprobe_failed",
        "ffprobe_unavailable_or_invalid",
        "full_decode_failed",
        "full_decode_unavailable",
        "dynamic_crop_v3_activation_version_mismatch",
        "dynamic_crop_v3_evidence_without_version",
        "dynamic_crop_v3_post_render_evidence_replay_failed",
        "dynamic_crop_version_invalid",
        "editorial_motion_track_payload_invalid",
        "editorial_transform_safety_bound_exceeded",
        "perceptual_quality_contract_invalid",
        "perceptual_quality_evaluation_failed",
        "perceptual_quality_issue_contract_invalid",
        "perceptual_final_video_unavailable",
        "perceptual_no_decoded_frames",
        "plan_contract_missing",
        "production_cut_local_sync_activation_contract_invalid",
        "production_cut_local_sync_binding_invalid",
        "production_cut_local_sync_evidence_missing",
        "production_cut_local_sync_required_gate_failed",
        "production_cut_local_sync_residual_exceeds_half_frame",
        "production_cut_local_sync_unknown_boundaries",
        "production_sync_duplicate_source_evidence",
        "production_sync_evidence_missing_or_unmapped",
        "production_sync_residual_exceeds_half_frame",
        "production_sync_used_source_evidence_missing",
        "production_sync_used_source_not_piecewise_verified",
        "selection_unknown_final_version",
        "source_local_interval_invalid",
    }
)
_MEMBER_TECHNICAL_CODES = frozenset(
    {
        "member_artifact_final_fingerprint_unavailable",
        "member_artifact_fingerprint_unavailable",
        "member_artifact_json_invalid",
        "member_artifacts_changed_during_qa",
        "member_ffprobe_failed",
        "member_ffprobe_plan_contract_mismatch",
        "member_full_decode_failed",
        "member_cut_local_sync_binding_invalid",
        "member_cut_local_sync_evidence_missing",
        "member_cut_local_sync_required_gate_failed",
        "member_cut_local_sync_residual_exceeds_half_frame",
        "member_cut_local_sync_timewarp_evidence_missing",
        "member_cut_local_sync_unknown_boundaries",
        "member_direct_segments_not_bound_to_render_plan",
        "member_perceptual_evaluation_failed",
        "member_pixel_provenance_incomplete",
        "member_rendered_continuity_evaluation_failed",
        "member_source_files_changed_during_pixel_review",
        "member_source_index_mapping_not_canonical",
    }
)


def _issue_codes(report_payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw_issues = report_payload.get("issues")
    if not isinstance(raw_issues, Sequence) or isinstance(
        raw_issues, (str, bytes, bytearray)
    ):
        return ()
    result: list[str] = []
    for issue in raw_issues:
        if not isinstance(issue, Mapping):
            continue
        code = issue.get("code")
        if isinstance(code, str) and code.strip():
            result.append(code.strip())
    return tuple(dict.fromkeys(result))


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
    codes = _issue_codes(report_payload)
    technical_catalog = (
        _MEMBER_TECHNICAL_CODES if member else _GROUP_TECHNICAL_CODES
    )
    technical = tuple(code for code in codes if code in technical_catalog)
    editorial = tuple(code for code in codes if code not in technical_catalog)
    report_passed = report_payload.get("passed") is True

    if report_passed and content_binding_passed:
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
        technical_issue_codes=technical or (("unknown_qa_failure",) if not codes else ()),
        editorial_issue_codes=editorial,
    )


@dataclass(slots=True)
class ReviewFirstRuntime:
    """Monotonic audit clock for the bounded user-facing workflow.

    The clock is an editorial-work budget, not a kill switch for an ffmpeg
    write already in flight.  Callers use ``optional_refinement_allowed`` to
    skip additional searches once the first complete render budget is spent.
    """

    enabled: bool
    total_budget_seconds: float = 600.0
    first_render_budget_seconds: float = 360.0
    refinement_stop_seconds: float = 540.0
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
        if not all(isinstance(value, (int, float)) for value in budgets):
            raise TypeError("review-first budgets must be numeric")
        if not (
            0 < self.first_render_budget_seconds
            <= self.refinement_stop_seconds
            <= self.total_budget_seconds
        ):
            raise ValueError("review-first budgets must be positive and ordered")
        self._started_at = float(self._clock())

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, float(self._clock()) - self._started_at)

    @property
    def optional_refinement_allowed(self) -> bool:
        return bool(
            self.enabled and self.elapsed_seconds < self.refinement_stop_seconds
        )

    def checkpoint(self, stage: str) -> None:
        name = str(stage).strip()
        if not name:
            raise ValueError("review-first checkpoint stage cannot be blank")
        self._checkpoints.append(
            {"stage": name, "elapsed_seconds": self.elapsed_seconds}
        )

    def require_first_render_budget(self, stage: str) -> None:
        """Stop optional search, while leaving the baseline render available."""

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
