"""Fail-closed classification for lazily measured visual splice edges.

The boundary repairer records one scalar-only evidence payload for every
source-local offset it actually measures.  This module turns that audit into a
small state machine without re-decoding media.  Only an exhaustive set of
non-waivable failures may become an optimizer no-good; missing evidence stays
indeterminate, while focus/scale-only failures remain eligible for final
output-domain crop geometry proof.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

from .timeline_review import TimelineReviewConfig


SPLICE_FEASIBILITY_CONTRACT = "lazy_exact_visual_splice_feasibility_v2"
DEFAULT_WIDE_FRAME_OFFSETS = tuple(range(-8, 9))


class SpliceFeasibilityStatus(str, Enum):
    RAW_PASS = "raw_pass"
    RAW_GEOMETRY_PROOF_REQUIRED = "raw_geometry_proof_required"
    HARD_INCOMPATIBLE = "hard_incompatible"
    INDETERMINATE = "indeterminate"


_GEOMETRY_PROOF_FAILURE_CODES = frozenset({"scale", "focus"})


@dataclass(frozen=True, slots=True)
class SpliceOffsetAssessment:
    frame_offset: int
    production_gate_passed: bool
    failure_codes: tuple[str, ...]
    evidence_complete: bool

    def __post_init__(self) -> None:
        if isinstance(self.frame_offset, bool) or not isinstance(
            self.frame_offset, int
        ):
            raise ValueError("splice frame offset must be an integer")
        if not isinstance(self.production_gate_passed, bool):
            raise TypeError("splice production gate result must be a bool")
        if not isinstance(self.evidence_complete, bool):
            raise TypeError("splice evidence completeness must be a bool")
        codes = tuple(sorted(set(self.failure_codes)))
        if any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in codes
        ):
            raise ValueError("splice failure codes must be exact identifiers")
        if self.production_gate_passed and codes:
            raise ValueError("passing splice offset cannot carry failure codes")
        if self.production_gate_passed and not self.evidence_complete:
            raise ValueError("passing splice offset needs complete evidence")
        object.__setattr__(self, "failure_codes", codes)

    @property
    def geometry_proof_eligible(self) -> bool:
        return bool(
            self.evidence_complete
            and self.failure_codes
            and set(self.failure_codes).issubset(
                _GEOMETRY_PROOF_FAILURE_CODES
            )
        )

    @property
    def hard_incompatible(self) -> bool:
        return bool(
            self.evidence_complete
            and self.failure_codes
            and not self.geometry_proof_eligible
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_offset": self.frame_offset,
            "production_gate_passed": self.production_gate_passed,
            "failure_codes": list(self.failure_codes),
            "evidence_complete": self.evidence_complete,
            "geometry_proof_eligible": self.geometry_proof_eligible,
            "hard_incompatible": self.hard_incompatible,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpliceOffsetAssessment":
        if not isinstance(payload, Mapping):
            raise TypeError("splice offset assessment payload must be a mapping")
        raw_codes = payload.get("failure_codes")
        if isinstance(raw_codes, (str, bytes)) or not isinstance(
            raw_codes, Sequence
        ):
            raise ValueError("splice offset failure_codes are invalid")
        value = cls(
            frame_offset=payload.get("frame_offset"),
            production_gate_passed=payload.get("production_gate_passed"),
            failure_codes=tuple(raw_codes),
            evidence_complete=payload.get("evidence_complete"),
        )
        if dict(payload) != value.to_dict():
            raise ValueError("splice offset assessment payload is not canonical")
        return value


def _canonical_frame_offsets(
    values: Sequence[int],
    *,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    offsets = tuple(values)
    if not offsets:
        raise ValueError(f"{name} cannot be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in offsets):
        raise ValueError(f"{name} must contain integers")
    if len(set(offsets)) != len(offsets):
        raise ValueError(f"{name} cannot contain duplicates")
    return tuple(sorted(offsets))


def _derived_feasibility_status(
    expected_frame_offsets: tuple[int, ...],
    offsets: tuple[SpliceOffsetAssessment, ...],
) -> tuple[SpliceFeasibilityStatus, str]:
    expected = set(expected_frame_offsets)
    observed = {item.frame_offset for item in offsets}
    if not observed.issubset(expected):
        return (
            SpliceFeasibilityStatus.INDETERMINATE,
            "unexpected_frame_offset_evidence",
        )
    if any(item.production_gate_passed for item in offsets):
        return SpliceFeasibilityStatus.RAW_PASS, "at_least_one_raw_offset_passed"
    if any(item.geometry_proof_eligible for item in offsets):
        return (
            SpliceFeasibilityStatus.RAW_GEOMETRY_PROOF_REQUIRED,
            "focus_or_scale_only_offset_available",
        )
    if observed != expected:
        return (
            SpliceFeasibilityStatus.INDETERMINATE,
            "allowed_offset_search_incomplete",
        )
    if any(not item.evidence_complete for item in offsets):
        return SpliceFeasibilityStatus.INDETERMINATE, "offset_evidence_incomplete"
    if offsets and all(item.hard_incompatible for item in offsets):
        return (
            SpliceFeasibilityStatus.HARD_INCOMPATIBLE,
            "all_allowed_offsets_nonwaivable_failure",
        )
    return (
        SpliceFeasibilityStatus.INDETERMINATE,
        "splice_evidence_did_not_reach_terminal_class",
    )


@dataclass(frozen=True, slots=True)
class SpliceFeasibilityAssessment:
    status: SpliceFeasibilityStatus
    expected_frame_offsets: tuple[int, ...]
    offsets: tuple[SpliceOffsetAssessment, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, SpliceFeasibilityStatus):
            raise TypeError("splice feasibility status is invalid")
        expected = _canonical_frame_offsets(
            self.expected_frame_offsets,
            name="expected splice frame offsets",
        )
        offsets = tuple(sorted(self.offsets, key=lambda item: item.frame_offset))
        if any(not isinstance(item, SpliceOffsetAssessment) for item in offsets):
            raise TypeError("splice offset assessments are invalid")
        if len({item.frame_offset for item in offsets}) != len(offsets):
            raise ValueError("splice offset assessments cannot be duplicated")
        reasons = tuple(sorted(set(self.reason_codes)))
        if any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in reasons
        ):
            raise ValueError("splice feasibility reason codes are invalid")
        derived_status, derived_reason = _derived_feasibility_status(
            expected,
            offsets,
        )
        if self.status is not derived_status:
            raise ValueError(
                "splice feasibility status contradicts canonical offset evidence"
            )
        if reasons != (derived_reason,):
            raise ValueError(
                "splice feasibility reasons contradict canonical offset evidence"
            )
        object.__setattr__(self, "expected_frame_offsets", expected)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "reason_codes", reasons)

    @property
    def expected_offset_count(self) -> int:
        return len(self.expected_frame_offsets)

    @property
    def observed_frame_offsets(self) -> tuple[int, ...]:
        return tuple(item.frame_offset for item in self.offsets)

    @property
    def search_complete(self) -> bool:
        return self.observed_frame_offsets == self.expected_frame_offsets

    @property
    def terminal_for_expected_domain(self) -> bool:
        return bool(
            self.status is not SpliceFeasibilityStatus.INDETERMINATE
            or self.search_complete
        )

    @property
    def optimizer_no_good_eligible(self) -> bool:
        return self.status is SpliceFeasibilityStatus.HARD_INCOMPATIBLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": SPLICE_FEASIBILITY_CONTRACT,
            "status": self.status.value,
            "optimizer_no_good_eligible": self.optimizer_no_good_eligible,
            "expected_frame_offsets": list(self.expected_frame_offsets),
            "expected_offset_count": self.expected_offset_count,
            "evaluated_offset_count": len(self.offsets),
            "search_complete": self.search_complete,
            "terminal_for_expected_domain": self.terminal_for_expected_domain,
            "reason_codes": list(self.reason_codes),
            "offsets": [item.to_dict() for item in self.offsets],
            "thresholds_lowered": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpliceFeasibilityAssessment":
        if not isinstance(payload, Mapping):
            raise TypeError("splice feasibility payload must be a mapping")
        if payload.get("contract") != SPLICE_FEASIBILITY_CONTRACT:
            raise ValueError("splice feasibility contract is invalid")
        try:
            status = SpliceFeasibilityStatus(payload.get("status"))
        except (TypeError, ValueError) as exc:
            raise ValueError("splice feasibility status is invalid") from exc
        raw_expected = payload.get("expected_frame_offsets")
        raw_offsets = payload.get("offsets")
        raw_reasons = payload.get("reason_codes")
        if isinstance(raw_expected, (str, bytes)) or not isinstance(
            raw_expected, Sequence
        ):
            raise ValueError("expected_frame_offsets are invalid")
        if isinstance(raw_offsets, (str, bytes)) or not isinstance(
            raw_offsets, Sequence
        ):
            raise ValueError("splice offsets are invalid")
        if isinstance(raw_reasons, (str, bytes)) or not isinstance(
            raw_reasons, Sequence
        ):
            raise ValueError("splice reason_codes are invalid")
        value = cls(
            status=status,
            expected_frame_offsets=tuple(raw_expected),
            offsets=tuple(SpliceOffsetAssessment.from_dict(item) for item in raw_offsets),
            reason_codes=tuple(raw_reasons),
        )
        if dict(payload) != value.to_dict():
            raise ValueError("splice feasibility payload is not canonical")
        return value


def build_splice_feasibility_assessment(
    *,
    expected_frame_offsets: Sequence[int],
    offsets: Sequence[SpliceOffsetAssessment],
) -> SpliceFeasibilityAssessment:
    expected = _canonical_frame_offsets(
        expected_frame_offsets,
        name="expected_frame_offsets",
    )
    canonical_offsets = tuple(sorted(offsets, key=lambda item: item.frame_offset))
    if any(not isinstance(item, SpliceOffsetAssessment) for item in canonical_offsets):
        raise TypeError("splice offset assessments are invalid")
    if len({item.frame_offset for item in canonical_offsets}) != len(
        canonical_offsets
    ):
        raise ValueError("splice offset assessments cannot be duplicated")
    status, reason = _derived_feasibility_status(expected, canonical_offsets)
    return SpliceFeasibilityAssessment(
        status=status,
        expected_frame_offsets=expected,
        offsets=canonical_offsets,
        reason_codes=(reason,),
    )


def merge_splice_feasibility_assessments(
    earlier: SpliceFeasibilityAssessment,
    later: SpliceFeasibilityAssessment,
) -> SpliceFeasibilityAssessment:
    """Monotonically union partial offset evidence for one exact edge."""

    if not isinstance(earlier, SpliceFeasibilityAssessment) or not isinstance(
        later, SpliceFeasibilityAssessment
    ):
        raise TypeError("splice assessments must be SpliceFeasibilityAssessment")
    if earlier.expected_frame_offsets != later.expected_frame_offsets:
        raise ValueError("splice assessment expected offset domains disagree")
    merged = {item.frame_offset: item for item in earlier.offsets}
    for item in later.offsets:
        previous = merged.get(item.frame_offset)
        if previous is not None and previous != item:
            raise ValueError("conflicting splice evidence for one exact frame offset")
        merged[item.frame_offset] = item
    return build_splice_feasibility_assessment(
        expected_frame_offsets=earlier.expected_frame_offsets,
        offsets=tuple(merged.values()),
    )


def _optional_finite_number(
    payload: Mapping[str, Any],
    key: str,
) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"splice evidence {key} must be numeric or null")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"splice evidence {key} must be finite")
    return numeric


def _side_stability_failure_codes(
    payload: Mapping[str, Any],
    *,
    side: str,
    config: TimelineReviewConfig,
) -> tuple[str, ...]:
    raw = payload.get(f"{side}_side_stability")
    if not isinstance(raw, Mapping):
        return (f"missing_{side}_side_stability",)
    checks = (
        (
            "scale_delta",
            config.maximum_boundary_side_scale_delta,
        ),
        (
            "structure_mismatch",
            config.maximum_boundary_side_structure_mismatch,
        ),
        (
            "pose_mismatch",
            config.maximum_boundary_side_pose_mismatch,
        ),
        (
            "shot_type_mismatch",
            config.maximum_boundary_side_shot_type_mismatch,
        ),
    )
    exceeded: list[str] = []
    for name, maximum in checks:
        value = _optional_finite_number(raw, name)
        if value is not None and value > maximum + 1e-9:
            exceeded.append(f"{side}_side_{name}")
    return tuple(exceeded)


def _offset_assessment(
    payload: Mapping[str, Any],
    *,
    config: TimelineReviewConfig,
) -> SpliceOffsetAssessment:
    frame_offset = payload.get("frame_offset")
    if isinstance(frame_offset, bool) or not isinstance(frame_offset, int):
        raise ValueError("splice audit frame_offset must be an integer")
    gate_passed = payload.get("production_transition_gate_passed")
    if not isinstance(gate_passed, bool):
        raise ValueError("splice audit production gate result is missing")
    if gate_passed:
        return SpliceOffsetAssessment(frame_offset, True, (), True)

    visual = payload.get("visual_evidence")
    professional_gate = payload.get("professional_gate_passed")
    if not isinstance(visual, Mapping) or professional_gate is not True:
        return SpliceOffsetAssessment(
            frame_offset,
            False,
            ("insufficient_combined_geometry_evidence",),
            False,
        )

    failures: list[str] = []
    checks = (
        ("shot_scale_delta", config.maximum_scale_mismatch, "scale"),
        ("focus_mismatch", config.maximum_focus_mismatch, "focus"),
        ("layout_mismatch", config.maximum_layout_mismatch, "layout"),
        (
            "persisted_layout_mismatch",
            config.maximum_layout_mismatch,
            "persisted_layout",
        ),
        (
            "silhouette_mismatch",
            config.maximum_silhouette_mismatch,
            "silhouette",
        ),
        (
            "formation_mismatch",
            config.maximum_formation_mismatch,
            "formation",
        ),
        ("pose_mismatch", config.maximum_pose_mismatch, "pose"),
    )
    for name, maximum, code in checks:
        value = _optional_finite_number(visual, name)
        if value is not None and value > maximum + 1e-9:
            failures.append(code)
    failures.extend(
        _side_stability_failure_codes(
            visual,
            side="outgoing",
            config=config,
        )
    )
    failures.extend(
        _side_stability_failure_codes(
            visual,
            side="incoming",
            config=config,
        )
    )
    missing = tuple(
        value for value in failures if value.startswith("missing_")
    )
    if not failures:
        # The repairer rejected this offset for a reason not represented by the
        # scalar audit.  Treat that gap as unknown rather than inventing a hard
        # incompatibility certificate.
        return SpliceOffsetAssessment(
            frame_offset,
            False,
            ("unexplained_production_gate_rejection",),
            False,
        )
    return SpliceOffsetAssessment(
        frame_offset,
        False,
        tuple(failures),
        not missing,
    )


def classify_splice_evidence_audit(
    candidate_evidence_audit: Sequence[Mapping[str, Any]],
    *,
    expected_frame_offsets: Sequence[int] = DEFAULT_WIDE_FRAME_OFFSETS,
    config: TimelineReviewConfig | None = None,
) -> SpliceFeasibilityAssessment:
    """Classify one exact edge after its full allowed offset search.

    ``expected_frame_offsets`` is the exact legal search domain.  A matching
    count is insufficient: shifted, missing, or out-of-range evidence remains
    indeterminate and can never become a permanent no-good.
    """

    expected = _canonical_frame_offsets(
        expected_frame_offsets,
        name="expected_frame_offsets",
    )
    if isinstance(candidate_evidence_audit, (str, bytes)) or not isinstance(
        candidate_evidence_audit, Sequence
    ):
        raise TypeError("candidate_evidence_audit must be a sequence")
    settings = config or TimelineReviewConfig()
    offsets = tuple(
        _offset_assessment(item, config=settings)
        for item in candidate_evidence_audit
    )
    if len({item.frame_offset for item in offsets}) != len(offsets):
        raise ValueError("splice audit contains duplicate frame offsets")

    return build_splice_feasibility_assessment(
        expected_frame_offsets=expected,
        offsets=offsets,
    )


__all__ = [
    "DEFAULT_WIDE_FRAME_OFFSETS",
    "SPLICE_FEASIBILITY_CONTRACT",
    "SpliceFeasibilityAssessment",
    "SpliceFeasibilityStatus",
    "SpliceOffsetAssessment",
    "build_splice_feasibility_assessment",
    "classify_splice_evidence_audit",
    "merge_splice_feasibility_assessments",
]
