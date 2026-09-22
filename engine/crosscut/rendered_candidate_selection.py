"""Sequentially render and rank multiple complete edit candidates.

Only one working 480p proxy exists at a time. An optional diagnostic sink may
retain one completed failed proxy separately. This is a storage guarantee and a
correctness boundary: the ranking consumes decoded production-render output,
not timeline metadata.  Candidate payloads stay in memory so the selected
timeline and all of its derived plans can be committed atomically by the
pipeline after review.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import cv2

from .render import RenderPlan
from .professional_reference_cadence import (
    PRIOR_ANALYSIS_BASIS as PROFESSIONAL_CADENCE_PRIOR_ANALYSIS_BASIS,
    ProfessionalReferenceCadenceProfile,
)
from .professional_reference_gate import ProfessionalReferenceProfile
from .rendered_proxy_review import RenderedProxyReview, render_and_review_proxy
from .stage_edit_learned_reviewer_adapter import (
    LEARNED_REVIEWER_RERANK_ANALYSIS_BASIS,
    LEARNED_REVIEWER_RERANK_SELECTION_BASIS,
    LearnedReviewerProductionSession,
)


ANALYSIS_BASIS = "sequential_decoded_proxy_candidate_rerank_v1"
SELECTION_BASIS = (
    "pass_then_optional_bounded_editorial_ranking_score_then_issue_fraction_"
    "then_boundary_p10_then_pre_render_score"
)
PROFESSIONAL_PLATEAU_ANALYSIS_BASIS = (
    "professional_reference_quality_plateau_dynamics_rerank_v1"
)
PROFESSIONAL_PLATEAU_SELECTION_BASIS = (
    "professional_gate_and_technical_zero_then_decoded_quality_plateau_"
    "then_human_visible_cut_count_envelope_then_preferred_source_switch_"
    "rate_else_default_decoded_quality_rank"
)
PROFESSIONAL_CADENCE_ANALYSIS_BASIS = (
    "professional_reference_quality_plateau_dynamics_cadence_rerank_v1"
)
PROFESSIONAL_CADENCE_SELECTION_BASIS = (
    "professional_gate_and_technical_zero_then_decoded_quality_plateau_"
    "then_human_visible_cut_count_envelope_then_preferred_source_switch_"
    "rate_plus_bounded_positive_only_professional_cadence_prior"
)
PROFESSIONAL_PLATEAU_RANGE_FRACTION = 0.25
PROFESSIONAL_PLATEAU_MINIMUM_POINTS = 1.0
PROFESSIONAL_PLATEAU_MAXIMUM_POINTS = 2.0
PREFERRED_SOURCE_SWITCHES_PER_MINUTE = (13.5, 17.5)


@dataclass(frozen=True, slots=True)
class RenderedCandidateRankMetadata:
    """Typed pre-render evidence used only by professional plateau reranking."""

    source_switches_per_minute: float

    def __post_init__(self) -> None:
        value = float(self.source_switches_per_minute)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "source_switches_per_minute must be finite and non-negative"
            )
        object.__setattr__(self, "source_switches_per_minute", value)


@dataclass(frozen=True, slots=True)
class RenderedCandidate:
    candidate_index: int
    render_plan: RenderPlan = field(repr=False)
    pre_render_score: float
    payload: Any = field(default=None, repr=False)
    rank_metadata: RenderedCandidateRankMetadata | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
        ):
            raise ValueError("candidate_index must be a non-negative integer")
        if not isinstance(self.render_plan, RenderPlan):
            raise TypeError("render_plan must be a RenderPlan")
        if not math.isfinite(float(self.pre_render_score)):
            raise ValueError("pre_render_score must be finite")
        if self.rank_metadata is not None and not isinstance(
            self.rank_metadata,
            RenderedCandidateRankMetadata,
        ):
            raise TypeError(
                "rank_metadata must be RenderedCandidateRankMetadata"
            )


@dataclass(frozen=True, slots=True)
class RenderedCandidateReview:
    candidate: RenderedCandidate = field(repr=False)
    review_payload: dict[str, Any] | None
    error: str | None = None

    def __post_init__(self) -> None:
        if (self.review_payload is None) == (self.error is None):
            raise ValueError("candidate review needs exactly one payload or error")

    @property
    def passed(self) -> bool:
        return bool(self.review_payload and self.review_payload.get("passed"))

    @property
    def weighted_score(self) -> float:
        if self.review_payload is None:
            return -math.inf
        return float(self.review_payload.get("weighted_score", -math.inf))

    @property
    def ranking_weighted_score(self) -> float:
        """Return an optional editorial rank without mutating technical QA."""

        if self.review_payload is None:
            return -math.inf
        value = self.review_payload.get(
            "ranking_weighted_score",
            self.review_payload.get("weighted_score", -math.inf),
        )
        try:
            number = float(value)
        except (TypeError, ValueError):
            return -math.inf
        return number if math.isfinite(number) else -math.inf

    @property
    def issue_fraction(self) -> float:
        if self.review_payload is None:
            return 1.0
        summary = self.review_payload.get("visible_boundary_score")
        if not isinstance(summary, dict):
            summary = self.review_payload.get("boundary_score")
        if not isinstance(summary, dict):
            return 1.0
        return float(summary.get("issue_fraction", 1.0))

    @property
    def boundary_p10(self) -> float:
        if self.review_payload is None:
            return -math.inf
        summary = self.review_payload.get("visible_boundary_score")
        if not isinstance(summary, dict):
            summary = self.review_payload.get("boundary_score")
        if not isinstance(summary, dict):
            return -math.inf
        value = summary.get("p10")
        return (
            float(value)
            if type(value) in (int, float) and math.isfinite(value)
            else -math.inf
        )

    def to_dict(self, *, selected: bool) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate.candidate_index,
            "selected": bool(selected),
            "pre_render_score": self.candidate.pre_render_score,
            "passed": self.passed,
            "weighted_score": (
                self.weighted_score if math.isfinite(self.weighted_score) else None
            ),
            "ranking_weighted_score": (
                self.ranking_weighted_score
                if math.isfinite(self.ranking_weighted_score)
                else None
            ),
            "issue_fraction": self.issue_fraction,
            "boundary_score_p10": (
                self.boundary_p10 if math.isfinite(self.boundary_p10) else None
            ),
            "error": self.error,
            "review": deepcopy(self.review_payload),
        }


@dataclass(frozen=True, slots=True)
class RenderedCandidateSelection:
    reviews: tuple[RenderedCandidateReview, ...]
    selected_candidate_index: int
    requested_review_count: int
    minimum_required_candidate_count: int
    analysis_basis: str = ANALYSIS_BASIS
    professional_plateau_rerank: dict[str, Any] | None = field(
        default=None,
        repr=False,
    )
    professional_cadence_prior: dict[str, Any] | None = field(
        default=None,
        repr=False,
    )
    learned_reviewer_rerank: dict[str, Any] | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.reviews:
            raise ValueError("rendered candidate selection cannot be empty")
        indices = tuple(item.candidate.candidate_index for item in self.reviews)
        if len(indices) != len(set(indices)):
            raise ValueError("rendered candidate selection contains duplicates")
        if self.selected_candidate_index not in indices:
            raise ValueError("selected candidate was not reviewed")
        if self.requested_review_count < len(self.reviews):
            raise ValueError("reviewed more candidates than requested")
        if self.minimum_required_candidate_count < 1:
            raise ValueError("minimum candidate count must be positive")
        if (
            self.professional_cadence_prior is not None
            and self.professional_plateau_rerank is None
        ):
            raise ValueError(
                "professional cadence prior requires professional plateau evidence"
            )

    @property
    def selected(self) -> RenderedCandidateReview:
        matches = [
            item
            for item in self.reviews
            if item.candidate.candidate_index == self.selected_candidate_index
        ]
        if len(matches) != 1:
            raise ValueError("selection must contain exactly one selected candidate")
        return matches[0]

    @property
    def passed(self) -> bool:
        return self.selected.passed

    @property
    def successful_review_count(self) -> int:
        return sum(item.review_payload is not None for item in self.reviews)

    @property
    def passing_candidate_count(self) -> int:
        return sum(item.passed for item in self.reviews)

    @property
    def minimum_candidate_count_satisfied(self) -> bool:
        return len(self.reviews) >= self.minimum_required_candidate_count

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "analysis_basis": self.analysis_basis,
            "selection_basis": (
                LEARNED_REVIEWER_RERANK_SELECTION_BASIS
                if self.learned_reviewer_rerank is not None
                else PROFESSIONAL_CADENCE_SELECTION_BASIS
                if self.professional_cadence_prior is not None
                else PROFESSIONAL_PLATEAU_SELECTION_BASIS
                if self.professional_plateau_rerank is not None
                else SELECTION_BASIS
            ),
            "storage_policy": {
                "one_proxy_at_a_time": True,
                "temporary_proxy_removed_after_each_review": True,
                "decoded_frames_persisted": False,
            },
            "requested_review_count": self.requested_review_count,
            "reviewed_candidate_count": len(self.reviews),
            "minimum_required_candidate_count": (
                self.minimum_required_candidate_count
            ),
            "minimum_candidate_count_satisfied": (
                self.minimum_candidate_count_satisfied
            ),
            "successful_review_count": self.successful_review_count,
            "passing_candidate_count": self.passing_candidate_count,
            "selected_candidate_index": self.selected_candidate_index,
            "passed": self.passed,
            "candidates": [
                item.to_dict(
                    selected=(
                        item.candidate.candidate_index
                        == self.selected_candidate_index
                    )
                )
                for item in self.reviews
            ],
        }
        if self.professional_plateau_rerank is not None:
            payload["professional_plateau_rerank"] = deepcopy(
                self.professional_plateau_rerank
            )
        if self.professional_cadence_prior is not None:
            payload["professional_cadence_prior"] = deepcopy(
                self.professional_cadence_prior
            )
        if self.learned_reviewer_rerank is not None:
            payload["learned_reviewer_rerank"] = deepcopy(
                self.learned_reviewer_rerank
            )
        return payload


RenderReviewFunction = Callable[
    [RenderPlan, Path], tuple[RenderPlan, RenderedProxyReview]
]


def _rank(item: RenderedCandidateReview) -> tuple[Any, ...]:
    return (
        item.passed,
        item.review_payload is not None,
        item.ranking_weighted_score,
        -item.issue_fraction,
        item.boundary_p10,
        item.candidate.pre_render_score,
        -item.candidate.candidate_index,
    )


def _learned_reviewer_selection(
    reviews: Sequence[RenderedCandidateReview],
    *,
    deterministic_selected: RenderedCandidateReview,
    session: LearnedReviewerProductionSession,
    professional_plateau_rerank: Mapping[str, Any] | None,
) -> tuple[RenderedCandidateReview, dict[str, Any]]:
    """Rerank only the deterministic hard-safe pool with learned quality."""

    if not isinstance(session, LearnedReviewerProductionSession):
        raise TypeError(
            "learned_reviewer_session must be a production session"
        )
    professional_evidence: dict[int, Mapping[str, Any]] = {}
    professional_active_pool = "decoded_review_passed"
    if professional_plateau_rerank is not None:
        raw_evidence = professional_plateau_rerank.get("candidates")
        if not isinstance(raw_evidence, list):
            raise ValueError(
                "professional plateau evidence is malformed for learned rerank"
            )
        for item in raw_evidence:
            if not isinstance(item, Mapping):
                raise ValueError(
                    "professional candidate evidence must be an object"
                )
            candidate_index = item.get("candidate_index")
            if (
                isinstance(candidate_index, bool)
                or not isinstance(candidate_index, int)
            ):
                raise ValueError(
                    "professional candidate evidence has an invalid index"
                )
            professional_evidence[candidate_index] = item
        dynamics_count = professional_plateau_rerank.get(
            "dynamics_rerank_eligible_candidate_count"
        )
        plateau_cut_count = professional_plateau_rerank.get(
            "quality_plateau_and_cut_envelope_candidate_count"
        )
        professional_count = professional_plateau_rerank.get(
            "professional_eligible_candidate_count"
        )
        for name, value in (
            ("dynamics candidate count", dynamics_count),
            ("plateau/cut candidate count", plateau_cut_count),
            ("professional candidate count", professional_count),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"professional plateau evidence has invalid {name}"
                )
        if dynamics_count > 0:
            professional_active_pool = "dynamics_rerank_eligible"
        elif plateau_cut_count > 0:
            professional_active_pool = (
                "quality_plateau_and_human_visible_cut_envelope"
            )
        elif professional_count > 0:
            professional_active_pool = "professional_eligible"
        else:
            professional_active_pool = "none"
    learned_by_index: dict[int, Any] = {}
    candidate_audits: list[dict[str, Any]] = []
    eligible_reviews: list[RenderedCandidateReview] = []
    for review in reviews:
        candidate_index = review.candidate.candidate_index
        if professional_plateau_rerank is None:
            hard_eligible = review.passed
            professional_eligible: bool | None = None
        else:
            evidence = professional_evidence.get(candidate_index)
            if evidence is None:
                raise ValueError(
                    "professional evidence is missing a reviewed candidate"
                )
            professional_eligible = evidence.get("professional_eligible") is True
            if professional_active_pool == "dynamics_rerank_eligible":
                in_active_pool = (
                    evidence.get("dynamics_rerank_eligible") is True
                )
            elif professional_active_pool == (
                "quality_plateau_and_human_visible_cut_envelope"
            ):
                in_active_pool = bool(
                    evidence.get("within_quality_plateau") is True
                    and evidence.get(
                        "within_human_visible_cut_count_envelope"
                    )
                    is True
                )
            elif professional_active_pool == "professional_eligible":
                in_active_pool = professional_eligible
            else:
                in_active_pool = False
            hard_eligible = bool(
                review.passed and professional_eligible and in_active_pool
            )
        if review.review_payload is None:
            raise ValueError(
                "configured learned reviewer requires every candidate to have "
                f"decoded evidence; candidate {candidate_index} is unavailable"
            )
        rate = (
            review.candidate.rank_metadata.source_switches_per_minute
            if review.candidate.rank_metadata is not None
            else None
        )
        learned = session.review_candidate(
            candidate_index=candidate_index,
            render_plan=review.candidate.render_plan,
            decoded_review=review.review_payload,
            source_switches_per_minute=rate,
        )
        learned_by_index[candidate_index] = learned
        exclusion_reasons: list[str] = []
        if not review.passed:
            exclusion_reasons.append("deterministic_decoded_review_failed")
        if professional_eligible is False:
            exclusion_reasons.append("professional_hard_gate_failed")
        if professional_plateau_rerank is not None and not in_active_pool:
            exclusion_reasons.append(
                "outside_existing_professional_active_safe_pool"
            )
        if not learned.materialization.technical_zero_tolerance_passed:
            exclusion_reasons.append(
                "decoded_technical_zero_tolerance_failed"
            )
        eligible = bool(
            hard_eligible
            and learned.materialization.technical_zero_tolerance_passed
        )
        if eligible:
            eligible_reviews.append(review)
        candidate_audits.append(
            {
                "candidate_index": candidate_index,
                "deterministic_decoded_passed": review.passed,
                "professional_hard_eligible": professional_eligible,
                "deterministic_hard_eligible": hard_eligible,
                "learned_review_available": True,
                "learned_ordering_eligible": eligible,
                "selected": False,
                "exclusion_reasons": exclusion_reasons,
                "learned_review": learned.to_dict(),
            }
        )
    if eligible_reviews:
        selected = max(
            eligible_reviews,
            key=lambda review: (
                learned_by_index[
                    review.candidate.candidate_index
                ].prediction.learned_quality_probability,
                _rank(review),
            ),
        )
        selection_mode = "learned_quality_within_deterministic_hard_safe_pool"
    else:
        selected = deterministic_selected
        selection_mode = "deterministic_fallback_no_hard_safe_learned_candidate"
    for item in candidate_audits:
        item["selected"] = item["candidate_index"] == selected.candidate.candidate_index
    selected_learned = learned_by_index.get(selected.candidate.candidate_index)
    audit = {
        "schema_version": 1,
        "analysis_basis": LEARNED_REVIEWER_RERANK_ANALYSIS_BASIS,
        "activated": True,
        "policy": {
            "ordering_scope": "deterministic_hard_safe_candidates_only",
            "professional_active_safe_pool": professional_active_pool,
            "decoded_review_final_hard_authority": True,
            "professional_gate_final_hard_authority": (
                professional_plateau_rerank is not None
            ),
            "technical_zero_tolerance": True,
            "failed_candidate_can_be_rescued": False,
            "hard_safe_candidate_set_can_be_expanded": False,
            "learned_output_usage": [
                "candidate_ordering",
                "stage_edit_repair_issue_window_suggestions",
            ],
            "learned_output_can_override_pass_fail": False,
        },
        "session": session.provenance_dict(),
        "deterministic_selected_candidate_index": (
            deterministic_selected.candidate.candidate_index
        ),
        "selected_candidate_index": selected.candidate.candidate_index,
        "selection_mode": selection_mode,
        "learned_order_changed_candidate": (
            selected.candidate.candidate_index
            != deterministic_selected.candidate.candidate_index
        ),
        "deterministic_hard_safe_candidate_count": len(eligible_reviews),
        "selected_repair_issue_boundary_indices": (
            list(selected_learned.repair_issue_boundary_indices)
            if selected_learned is not None
            and selected_learned.materialization.technical_zero_tolerance_passed
            else []
        ),
        "selected_repair_issues": (
            [issue.to_dict() for issue in selected_learned.repair_issues]
            if selected_learned is not None
            and selected_learned.materialization.technical_zero_tolerance_passed
            else []
        ),
        "candidates": candidate_audits,
    }
    return selected, audit


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = tuple(sorted(float(value) for value in values))
    if not ordered:
        raise ValueError("professional score quantile needs evidence")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    alpha = position - low
    return ordered[low] * (1.0 - alpha) + ordered[high] * alpha


def _professional_reference_envelope(
    profile: ProfessionalReferenceProfile,
) -> dict[str, Any]:
    reference_score_p10_values = tuple(
        _quantile(
            tuple(cut.score for cut in reference.cuts),
            0.10,
        )
        for reference in profile.references
    )
    reference_cut_counts = tuple(
        len(reference.cuts) for reference in profile.references
    )
    score_p10_range = max(reference_score_p10_values) - min(
        reference_score_p10_values
    )
    uncapped_plateau = (
        PROFESSIONAL_PLATEAU_RANGE_FRACTION * score_p10_range
    )
    plateau_width = min(
        PROFESSIONAL_PLATEAU_MAXIMUM_POINTS,
        max(PROFESSIONAL_PLATEAU_MINIMUM_POINTS, uncapped_plateau),
    )
    return {
        "reference_score_p10_values": list(reference_score_p10_values),
        "reference_score_p10_range": score_p10_range,
        "plateau_range_fraction": PROFESSIONAL_PLATEAU_RANGE_FRACTION,
        "uncapped_plateau_width_points": uncapped_plateau,
        "minimum_plateau_width_points": (
            PROFESSIONAL_PLATEAU_MINIMUM_POINTS
        ),
        "maximum_plateau_width_points": (
            PROFESSIONAL_PLATEAU_MAXIMUM_POINTS
        ),
        "plateau_width_points": plateau_width,
        "reference_visible_cut_counts": list(reference_cut_counts),
        "visible_cut_count_minimum": min(reference_cut_counts),
        "visible_cut_count_maximum": max(reference_cut_counts),
    }


def _non_negative_integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _professional_gate_evidence(
    review: RenderedCandidateReview,
) -> dict[str, Any]:
    payload = review.review_payload
    gate = (
        payload.get("professional_reference_gate")
        if isinstance(payload, Mapping)
        else None
    )
    technical = gate.get("technical_integrity") if isinstance(gate, Mapping) else None
    technical_issue_count = (
        _non_negative_integer(technical.get("issue_count"))
        if isinstance(technical, Mapping)
        else None
    )
    gate_passed = bool(gate.get("passed")) if isinstance(gate, Mapping) else False
    technical_passed = bool(
        isinstance(technical, Mapping)
        and technical.get("passed") is True
        and technical_issue_count == 0
    )
    overall_review_passed = review.passed
    professional_eligible = bool(
        overall_review_passed and gate_passed and technical_passed
    )
    introduced_count = (
        _non_negative_integer(payload.get("introduced_cross_source_count"))
        if isinstance(payload, Mapping)
        else None
    )
    native_count = (
        _non_negative_integer(payload.get("native_source_cut_count"))
        if isinstance(payload, Mapping)
        else None
    )
    visible_cut_count = (
        introduced_count + native_count
        if introduced_count is not None and native_count is not None
        else None
    )
    source_switch_rate = (
        review.candidate.rank_metadata.source_switches_per_minute
        if review.candidate.rank_metadata is not None
        else None
    )
    exclusion_reasons: list[str] = []
    if payload is None:
        exclusion_reasons.append("decoded_review_unavailable")
    if not overall_review_passed:
        exclusion_reasons.append("overall_decoded_review_failed")
    if not gate_passed:
        exclusion_reasons.append("professional_reference_gate_failed_or_missing")
    if not technical_passed:
        exclusion_reasons.append("professional_technical_zero_tolerance_failed")
    return {
        "candidate_index": review.candidate.candidate_index,
        "decoded_review_available": payload is not None,
        "overall_decoded_review_passed": overall_review_passed,
        "professional_reference_gate_passed": gate_passed,
        "professional_technical_issue_count": technical_issue_count,
        "professional_technical_zero_tolerance_passed": technical_passed,
        "professional_eligible": professional_eligible,
        "decoded_weighted_score": (
            review.weighted_score
            if math.isfinite(review.weighted_score)
            else None
        ),
        "pre_render_score": review.candidate.pre_render_score,
        "introduced_cross_source_count": introduced_count,
        "native_source_cut_count": native_count,
        "rendered_visible_cut_count": visible_cut_count,
        "source_switches_per_minute": source_switch_rate,
        "exclusion_reasons": exclusion_reasons,
    }


def _strict_candidate_master_cut_clock(
    review: RenderedCandidateReview,
    cadence_profile: ProfessionalReferenceCadenceProfile,
) -> dict[str, Any]:
    """Map decoded source switches onto the canonical master clock.

    The proxy is allowed normal frame quantization, but its declared FPS,
    decoded frame count, duration, canonical span, and every source-switch
    boundary must agree with the exact render plan.  A configured cadence
    profile therefore fails closed instead of scoring a stale/misaligned
    review payload.
    """

    payload = review.review_payload
    if not isinstance(payload, Mapping):
        raise ValueError("candidate decoded review is unavailable")
    proxy = payload.get("proxy")
    if not isinstance(proxy, Mapping):
        raise ValueError("candidate review lacks strict proxy clock metadata")
    plan = review.candidate.render_plan
    fps = float(proxy.get("fps", math.nan))
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("candidate proxy fps must be positive and finite")
    if not math.isclose(fps, plan.fps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("candidate proxy fps does not match render-plan fps")
    decoded_frame_count = _non_negative_integer(
        proxy.get("decoded_frame_count")
    )
    if decoded_frame_count != plan.output_frame_count:
        raise ValueError(
            "candidate proxy frame count does not match render-plan clock"
        )
    proxy_duration = float(proxy.get("duration_seconds", math.nan))
    if not math.isfinite(proxy_duration) or not math.isclose(
        proxy_duration,
        plan.render_duration,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "candidate proxy duration does not match render-plan duration"
        )

    canonical_start = float(plan.master_audio_start)
    canonical_end = canonical_start + float(plan.duration)
    frame_tolerance = 1.5 / fps
    if abs(proxy_duration - (canonical_end - canonical_start)) > frame_tolerance:
        raise ValueError(
            "candidate proxy duration does not match canonical master span"
        )
    profile_duration = cadence_profile.model.master_duration_seconds
    if canonical_start < -1e-9 or canonical_end > profile_duration + frame_tolerance:
        raise ValueError(
            "candidate canonical master span exceeds cadence-profile domain"
        )

    expected_switches = tuple(
        (
            canonical_start + right.master_start,
            left.source_index,
            right.source_index,
        )
        for left, right in zip(plan.segments, plan.segments[1:])
        if left.source_index != right.source_index
    )
    raw_boundaries = payload.get("boundaries")
    if not isinstance(raw_boundaries, list):
        raise ValueError("candidate review boundaries must be an array")
    decoded_switches: list[tuple[float, int, int, int]] = []
    for raw_boundary in raw_boundaries:
        if not isinstance(raw_boundary, Mapping):
            raise ValueError("candidate review boundary must be an object")
        outgoing = _non_negative_integer(
            raw_boundary.get("outgoing_source_index")
        )
        incoming = _non_negative_integer(
            raw_boundary.get("incoming_source_index")
        )
        frame_index = _non_negative_integer(raw_boundary.get("cut_frame_index"))
        if outgoing is None or incoming is None or frame_index is None:
            raise ValueError("candidate review boundary clock is malformed")
        if frame_index <= 0 or frame_index >= decoded_frame_count:
            raise ValueError("candidate review boundary lies outside proxy span")
        if outgoing == incoming:
            continue
        decoded_switches.append(
            (
                canonical_start + frame_index / fps,
                outgoing,
                incoming,
                frame_index,
            )
        )
    if len(decoded_switches) != len(expected_switches):
        raise ValueError(
            "decoded source-switch count does not match render-plan timeline"
        )
    introduced_count = _non_negative_integer(
        payload.get("introduced_cross_source_count")
    )
    if (
        introduced_count is not None
        and introduced_count != len(decoded_switches)
    ):
        raise ValueError(
            "decoded source-switch count does not match professional review domain"
        )
    for expected, decoded in zip(
        expected_switches,
        decoded_switches,
        strict=True,
    ):
        expected_seconds, expected_outgoing, expected_incoming = expected
        decoded_seconds, decoded_outgoing, decoded_incoming, _frame_index = decoded
        if (
            decoded_outgoing != expected_outgoing
            or decoded_incoming != expected_incoming
            or abs(decoded_seconds - expected_seconds) > frame_tolerance
        ):
            raise ValueError(
                "decoded source-switch clock does not match render-plan timeline"
            )
    boundary_times = tuple(item[0] for item in decoded_switches)
    if any(
        right <= left
        for left, right in zip(boundary_times, boundary_times[1:])
    ):
        raise ValueError("candidate master source-switch clock is not increasing")
    return {
        "clock_basis": (
            "proxy_output_cut_frame_plus_render_plan_canonical_master_start"
        ),
        "proxy_fps": fps,
        "decoded_frame_count": decoded_frame_count,
        "proxy_duration_seconds": proxy_duration,
        "canonical_master_start_seconds": canonical_start,
        "canonical_master_end_seconds": canonical_end,
        "canonical_span_matches_proxy": True,
        "render_plan_source_switch_count": len(expected_switches),
        "decoded_source_switch_count": len(decoded_switches),
        "professional_review_introduced_cross_source_count": introduced_count,
        "source_switch_clock_matches_render_plan": True,
        "boundary_times_seconds": list(boundary_times),
    }


def _professional_plateau_selection(
    reviews: Sequence[RenderedCandidateReview],
    profile: ProfessionalReferenceProfile,
    cadence_profile: ProfessionalReferenceCadenceProfile | None = None,
) -> tuple[
    RenderedCandidateReview,
    dict[str, Any],
    dict[str, Any] | None,
]:
    """Prefer human-like dynamics only inside a conservative quality plateau."""

    default_selected = max(reviews, key=_rank)
    envelope = _professional_reference_envelope(profile)
    evidence = [_professional_gate_evidence(review) for review in reviews]
    eligible_reviews = tuple(
        review
        for review, item in zip(reviews, evidence, strict=True)
        if item["professional_eligible"]
    )
    if eligible_reviews:
        best_quality_review = max(eligible_reviews, key=_rank)
        best_weighted_score = best_quality_review.weighted_score
        plateau_floor = best_weighted_score - float(
            envelope["plateau_width_points"]
        )
    else:
        best_quality_review = default_selected
        best_weighted_score = None
        plateau_floor = None

    evidence_by_index = {
        int(item["candidate_index"]): item for item in evidence
    }
    plateau_cut_candidates: list[RenderedCandidateReview] = []
    dynamics_candidates: list[RenderedCandidateReview] = []
    preferred_minimum, preferred_maximum = (
        PREFERRED_SOURCE_SWITCHES_PER_MINUTE
    )
    for review in reviews:
        item = evidence_by_index[review.candidate.candidate_index]
        weighted_score = item["decoded_weighted_score"]
        within_plateau = bool(
            item["professional_eligible"]
            and plateau_floor is not None
            and weighted_score is not None
            and float(weighted_score) >= plateau_floor - 1e-9
        )
        visible_cut_count = item["rendered_visible_cut_count"]
        within_cut_envelope = bool(
            visible_cut_count is not None
            and int(envelope["visible_cut_count_minimum"])
            <= int(visible_cut_count)
            <= int(envelope["visible_cut_count_maximum"])
        )
        switch_rate = item["source_switches_per_minute"]
        within_preferred_dynamics = bool(
            switch_rate is not None
            and preferred_minimum <= float(switch_rate) <= preferred_maximum
        )
        dynamics_eligible = bool(
            within_plateau
            and within_cut_envelope
            and within_preferred_dynamics
        )
        item.update(
            {
                "quality_distance_from_best_points": (
                    best_weighted_score - float(weighted_score)
                    if best_weighted_score is not None
                    and weighted_score is not None
                    else None
                ),
                "within_quality_plateau": within_plateau,
                "within_human_visible_cut_count_envelope": (
                    within_cut_envelope
                ),
                "within_preferred_source_switch_rate": (
                    within_preferred_dynamics
                ),
                "dynamics_rerank_eligible": dynamics_eligible,
            }
        )
        if item["professional_eligible"] and not within_plateau:
            item["exclusion_reasons"].append("outside_quality_plateau")
        if within_plateau and not within_cut_envelope:
            item["exclusion_reasons"].append(
                "outside_human_visible_cut_count_envelope"
            )
        if within_plateau and within_cut_envelope and not within_preferred_dynamics:
            item["exclusion_reasons"].append(
                "outside_preferred_source_switch_rate"
            )
        if dynamics_eligible:
            dynamics_candidates.append(review)
        if within_plateau and within_cut_envelope:
            plateau_cut_candidates.append(review)

    cadence_audit: dict[str, Any] | None = None
    cadence_scores_by_index: dict[int, dict[str, Any]] = {}
    if cadence_profile is not None:
        for review in reviews:
            item = evidence_by_index[review.candidate.candidate_index]
            dynamics_eligible = bool(item["dynamics_rerank_eligible"])
            if review.review_payload is None:
                cadence_item = {
                    "candidate_index": review.candidate.candidate_index,
                    "clock_validation": None,
                    "raw_prior_0_to_1": None,
                    "applied_bonus_points": 0.0,
                    "eligible_for_ranking_bonus": False,
                    "exclusion_reason": "decoded_review_unavailable",
                }
            else:
                clock = _strict_candidate_master_cut_clock(
                    review,
                    cadence_profile,
                )
                switch_rate = item["source_switches_per_minute"]
                if switch_rate is None:
                    raise ValueError(
                        "cadence rerank requires source-switch rate metadata"
                    )
                score = cadence_profile.score_candidate_boundaries(
                    tuple(float(value) for value in clock["boundary_times_seconds"]),
                    hard_safety_passed=dynamics_eligible,
                    base_score=float(switch_rate),
                ).to_dict()
                cadence_item = {
                    "candidate_index": review.candidate.candidate_index,
                    "clock_validation": clock,
                    **score,
                    "ranking_base_source_switches_per_minute": float(
                        switch_rate
                    ),
                    "cadence_adjusted_dynamics_score": score[
                        "adjusted_score"
                    ],
                    "exclusion_reason": (
                        None
                        if dynamics_eligible
                        else "outside_preexisting_professional_safe_candidate_set"
                    ),
                }
            cadence_scores_by_index[review.candidate.candidate_index] = cadence_item
            item["professional_cadence_prior"] = deepcopy(cadence_item)
        cadence_audit = {
            "analysis_basis": PROFESSIONAL_CADENCE_PRIOR_ANALYSIS_BASIS,
            "activated": True,
            "profile": cadence_profile.provenance_dict(),
            "policy": {
                "candidate_clock": (
                    "decoded_proxy_source_switch_frame_mapped_to_render_plan_"
                    "canonical_master_clock"
                ),
                "clock_validation": (
                    "strict_proxy_fps_frame_count_duration_span_and_render_plan_"
                    "source_switch_correspondence"
                ),
                "eligibility": (
                    "preexisting_professional_gate_pass_and_technical_zero_"
                    "and_quality_plateau_and_human_visible_cut_envelope_and_"
                    "preferred_source_switch_rate"
                ),
                "ranking": (
                    "source_switches_per_minute_plus_bounded_positive_only_"
                    "cadence_bonus_then_decoded_quality_then_pre_render_then_"
                    "lower_candidate_index"
                ),
                "failed_candidate_can_be_rescued": False,
                "professional_safe_candidate_set_can_be_expanded": False,
            },
            "eligible_candidate_count": len(dynamics_candidates),
            "candidates": [
                deepcopy(cadence_scores_by_index[review.candidate.candidate_index])
                for review in reviews
            ],
        }

    if dynamics_candidates:
        if cadence_profile is None:
            selected = max(
                dynamics_candidates,
                key=lambda review: (
                    float(
                        review.candidate.rank_metadata.source_switches_per_minute
                    ),
                    review.ranking_weighted_score,
                    review.candidate.pre_render_score,
                    -review.candidate.candidate_index,
                ),
            )
            selection_mode = (
                "preferred_dynamics_within_professional_quality_plateau"
            )
        else:
            selected = max(
                dynamics_candidates,
                key=lambda review: (
                    float(
                        cadence_scores_by_index[
                            review.candidate.candidate_index
                        ]["cadence_adjusted_dynamics_score"]
                    ),
                    review.ranking_weighted_score,
                    review.candidate.pre_render_score,
                    -review.candidate.candidate_index,
                ),
            )
            selection_mode = (
                "bounded_positive_cadence_prior_within_preexisting_"
                "professional_safe_dynamics_set"
            )
    elif plateau_cut_candidates:
        selected = max(plateau_cut_candidates, key=_rank)
        selection_mode = (
            "default_quality_rank_within_human_cut_envelope_"
            "no_preferred_dynamics_candidate"
        )
    elif eligible_reviews:
        selected = best_quality_review
        selection_mode = (
            "default_quality_rank_no_human_cut_envelope_candidate"
        )
    else:
        selected = default_selected
        selection_mode = "default_rank_no_professional_eligible_candidate"

    for item in evidence:
        item["selected"] = (
            item["candidate_index"] == selected.candidate.candidate_index
        )
    audit = {
        "analysis_basis": PROFESSIONAL_PLATEAU_ANALYSIS_BASIS,
        "activated": True,
        "profile": profile.provenance_dict(),
        "policy": {
            "professional_eligibility": (
                "overall_pass_and_professional_gate_pass_and_technical_zero"
            ),
            "quality_plateau": (
                "best_decoded_weighted_score_minus_human_score_p10_range_fraction"
            ),
            "rendered_visible_cut_count": (
                "introduced_cross_source_plus_native_source_cut"
            ),
            "preferred_source_switches_per_minute": [
                preferred_minimum,
                preferred_maximum,
            ],
            "outside_preferred_dynamics_policy": (
                "not_dynamics_eligible_default_quality_rank_if_none_eligible"
            ),
            "stable_tie_break": [
                "higher_source_switches_per_minute",
                "higher_editorial_ranking_weighted_score",
                "higher_pre_render_score",
                "lower_candidate_index",
            ],
        },
        "human_envelope": envelope,
        "default_selected_candidate_index": (
            default_selected.candidate.candidate_index
        ),
        "best_professional_quality_candidate_index": (
            best_quality_review.candidate.candidate_index
            if eligible_reviews
            else None
        ),
        "best_professional_decoded_weighted_score": best_weighted_score,
        "quality_plateau_floor": plateau_floor,
        "professional_eligible_candidate_count": len(eligible_reviews),
        "quality_plateau_and_cut_envelope_candidate_count": len(
            plateau_cut_candidates
        ),
        "dynamics_rerank_eligible_candidate_count": len(dynamics_candidates),
        "selection_mode": selection_mode,
        "selected_candidate_index": selected.candidate.candidate_index,
        "candidates": evidence,
    }
    if cadence_audit is not None:
        cadence_audit["selected_candidate_index"] = (
            selected.candidate.candidate_index
        )
    return (selected, audit, cadence_audit)


def review_rendered_candidates(
    candidates: Sequence[RenderedCandidate],
    *,
    scratch_directory: Path,
    maximum_review_count: int = 8,
    minimum_required_candidate_count: int = 8,
    render_review: RenderReviewFunction = render_and_review_proxy,
    diagnostic_sink: Callable[[Path, RenderPlan, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    professional_reference_profile: ProfessionalReferenceProfile | None = None,
    professional_reference_cadence_profile: (
        ProfessionalReferenceCadenceProfile | None
    ) = None,
    learned_reviewer_session: LearnedReviewerProductionSession | None = None,
) -> RenderedCandidateSelection:
    """Render top-N complete plans sequentially and select decoded-output best."""

    if (
        isinstance(maximum_review_count, bool)
        or not isinstance(maximum_review_count, int)
        or not 1 <= maximum_review_count <= 32
    ):
        raise ValueError("maximum_review_count must be between one and 32")
    if (
        isinstance(minimum_required_candidate_count, bool)
        or not isinstance(minimum_required_candidate_count, int)
        or not 1 <= minimum_required_candidate_count <= 32
    ):
        raise ValueError(
            "minimum_required_candidate_count must be between one and 32"
        )
    if (
        professional_reference_cadence_profile is not None
        and professional_reference_profile is None
    ):
        raise ValueError(
            "professional cadence prior requires a professional quality profile"
        )
    ordered = tuple(candidates[:maximum_review_count])
    if not ordered:
        raise ValueError("at least one rendered candidate is required")
    indices = tuple(item.candidate_index for item in ordered)
    if len(indices) != len(set(indices)):
        raise ValueError("rendered candidate indices must be unique")
    reviews: list[RenderedCandidateReview] = []
    for position, candidate in enumerate(ordered, start=1):
        if progress_callback is not None:
            progress_callback(position, len(ordered), candidate.candidate_index)
        reviews.append(
            render_one_candidate_review(
                candidate,
                scratch_directory=scratch_directory,
                render_review=render_review,
                diagnostic_sink=diagnostic_sink,
            )
        )
    return select_rendered_candidate_reviews(
        reviews,
        requested_review_count=maximum_review_count,
        minimum_required_candidate_count=minimum_required_candidate_count,
        professional_reference_profile=professional_reference_profile,
        professional_reference_cadence_profile=(
            professional_reference_cadence_profile
        ),
        learned_reviewer_session=learned_reviewer_session,
    )


def render_one_candidate_review(
    candidate: RenderedCandidate,
    *,
    scratch_directory: Path,
    render_review: RenderReviewFunction = render_and_review_proxy,
    diagnostic_sink: Callable[[Path, RenderPlan, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> RenderedCandidateReview:
    """Render and decode-review exactly one candidate with bounded storage."""

    if not isinstance(candidate, RenderedCandidate):
        raise TypeError("candidate must be a RenderedCandidate")
    scratch = Path(scratch_directory)
    scratch.mkdir(parents=True, exist_ok=True)
    # A separate context per candidate makes cleanup a sequencing barrier: the
    # next proxy cannot start until every file from the prior review is gone.
    with tempfile.TemporaryDirectory(
        prefix=(
            ".ive-crosscut-candidate-proxy-"
            f"{candidate.candidate_index:02d}-"
        ),
        dir=scratch,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        proxy_path = temporary_root / (
            f"candidate-{candidate.candidate_index:02d}.mp4"
        )
        try:
            _proxy_plan, review = render_review(
                candidate.render_plan,
                proxy_path,
            )
            payload = deepcopy(review.payload)
            if diagnostic_sink is not None:
                try:
                    payload["diagnostic_retention"] = dict(
                        diagnostic_sink(proxy_path, candidate.render_plan, payload)
                    )
                except (OSError, TypeError, ValueError) as exc:
                    # Retention failure cannot replace the actual QA verdict.
                    payload["diagnostic_retention"] = {
                        "retained": False, "error": f"{type(exc).__name__}: {exc}",
                        "final_media_authority": False,
                    }
            proxy = payload.get("proxy")
            if isinstance(proxy, dict):
                proxy["path"] = None
                proxy["temporary_proxy_removed_after_review"] = True
            return RenderedCandidateReview(candidate, payload)
        except (OSError, RuntimeError, TypeError, ValueError, cv2.error) as exc:
            exception_name = (
                "cv2.error" if isinstance(exc, cv2.error) else type(exc).__name__
            )
            return RenderedCandidateReview(
                candidate,
                None,
                f"{exception_name}: {exc}",
            )
        finally:
            # Do not suppress failure: advancing with a live prior proxy would
            # violate the serialized one-proxy-at-a-time storage guarantee.
            proxy_path.unlink(missing_ok=True)


def select_rendered_candidate_reviews(
    reviews: Sequence[RenderedCandidateReview],
    *,
    requested_review_count: int,
    minimum_required_candidate_count: int,
    professional_reference_profile: ProfessionalReferenceProfile | None = None,
    professional_reference_cadence_profile: (
        ProfessionalReferenceCadenceProfile | None
    ) = None,
    learned_reviewer_session: LearnedReviewerProductionSession | None = None,
) -> RenderedCandidateSelection:
    """Rank already decoded reviews without rendering their proxies again."""

    values = tuple(reviews)
    if not values:
        raise ValueError("at least one rendered candidate review is required")
    if (
        isinstance(requested_review_count, bool)
        or not isinstance(requested_review_count, int)
        or not 1 <= requested_review_count <= 32
    ):
        raise ValueError("requested_review_count must be between one and 32")
    if (
        isinstance(minimum_required_candidate_count, bool)
        or not isinstance(minimum_required_candidate_count, int)
        or not 1 <= minimum_required_candidate_count <= 32
    ):
        raise ValueError(
            "minimum_required_candidate_count must be between one and 32"
        )
    if len(values) > requested_review_count:
        raise ValueError("reviewed more candidates than requested")
    if any(not isinstance(item, RenderedCandidateReview) for item in values):
        raise TypeError("reviews must contain RenderedCandidateReview values")
    indices = tuple(item.candidate.candidate_index for item in values)
    if len(indices) != len(set(indices)):
        raise ValueError("rendered candidate reviews contain duplicate indices")
    if (
        professional_reference_cadence_profile is not None
        and professional_reference_profile is None
    ):
        raise ValueError(
            "professional cadence prior requires a professional quality profile"
        )
    professional_plateau_rerank: dict[str, Any] | None = None
    professional_cadence_prior: dict[str, Any] | None = None
    if professional_reference_profile is None:
        selected = max(values, key=_rank)
    else:
        (
            selected,
            professional_plateau_rerank,
            professional_cadence_prior,
        ) = _professional_plateau_selection(
            values,
            professional_reference_profile,
            professional_reference_cadence_profile,
        )
    learned_reviewer_rerank: dict[str, Any] | None = None
    if learned_reviewer_session is not None:
        selected, learned_reviewer_rerank = _learned_reviewer_selection(
            values,
            deterministic_selected=selected,
            session=learned_reviewer_session,
            professional_plateau_rerank=professional_plateau_rerank,
        )
    return RenderedCandidateSelection(
        values,
        selected.candidate.candidate_index,
        requested_review_count,
        minimum_required_candidate_count,
        analysis_basis=(
            LEARNED_REVIEWER_RERANK_ANALYSIS_BASIS
            if learned_reviewer_session is not None
            else PROFESSIONAL_CADENCE_ANALYSIS_BASIS
            if professional_reference_cadence_profile is not None
            else PROFESSIONAL_PLATEAU_ANALYSIS_BASIS
            if professional_reference_profile is not None
            else ANALYSIS_BASIS
        ),
        professional_plateau_rerank=professional_plateau_rerank,
        professional_cadence_prior=professional_cadence_prior,
        learned_reviewer_rerank=learned_reviewer_rerank,
    )


__all__ = [
    "ANALYSIS_BASIS",
    "PREFERRED_SOURCE_SWITCHES_PER_MINUTE",
    "PROFESSIONAL_CADENCE_ANALYSIS_BASIS",
    "PROFESSIONAL_CADENCE_SELECTION_BASIS",
    "PROFESSIONAL_PLATEAU_ANALYSIS_BASIS",
    "PROFESSIONAL_PLATEAU_SELECTION_BASIS",
    "RenderedCandidate",
    "RenderedCandidateRankMetadata",
    "RenderedCandidateReview",
    "RenderedCandidateSelection",
    "SELECTION_BASIS",
    "render_one_candidate_review",
    "review_rendered_candidates",
    "select_rendered_candidate_reviews",
]
