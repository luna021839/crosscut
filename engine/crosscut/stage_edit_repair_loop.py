"""Deterministic, reviewer-compatible partial repair for :class:`StageEdit`.

This module is intentionally orchestration-only.  It neither runs a model nor
opens/renders media.  A caller supplies a beam redecoder for one bounded edit
window and a full-edit review scorer.  The loop commits a splice only when all
three learned/professional objectives improve strictly and the scorer reports
zero technical errors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

from .render import TransitionKind
from .stage_edit import StageBoundary, StageEdit, StageShot


REPAIR_LOOP_SCHEMA_VERSION = 1
_TIME_EPSILON = 1e-6
_CROP_MODES = frozenset({"static_crop", "pan_crop", "editorial_transform"})


class StageEditRepairError(RuntimeError):
    """Base class for fail-closed repair-loop errors."""


class StageEditRepairInputError(StageEditRepairError, ValueError):
    """Raised for a malformed repair contract."""


class StageEditRepairCallbackError(StageEditRepairError):
    """Raised when a decoder/reviewer callback violates its contract."""


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StageEditRepairInputError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise StageEditRepairInputError(f"{name} must be finite")
    return number


def _strings(values: Sequence[str], name: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise StageEditRepairInputError(f"{name} cannot contain blank values")
    if len(set(normalized)) != len(normalized):
        raise StageEditRepairInputError(f"{name} values must be unique")
    return normalized


def _canonical_audit(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise StageEditRepairInputError(
            f"{name} must be finite JSON-compatible evidence"
        ) from exc
    if not isinstance(decoded, dict):
        raise StageEditRepairInputError(f"{name} must be an object")
    return decoded


@dataclass(frozen=True, slots=True)
class StageEditRepairIssue:
    """One decoded reviewer issue on the absolute canonical master clock."""

    issue_id: str
    start_seconds: float
    end_seconds: float
    causes: tuple[str, ...]
    severity: str = "error"

    def __post_init__(self) -> None:
        issue_id = str(self.issue_id).strip()
        if not issue_id:
            raise StageEditRepairInputError("repair issue_id cannot be blank")
        object.__setattr__(self, "issue_id", issue_id)
        start = _finite(self.start_seconds, "repair issue start_seconds")
        end = _finite(self.end_seconds, "repair issue end_seconds")
        if start < 0.0 or end < start:
            raise StageEditRepairInputError("repair issue span is invalid")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        causes = _strings(self.causes, "repair issue causes")
        if not causes:
            raise StageEditRepairInputError("repair issue needs at least one cause")
        object.__setattr__(self, "causes", causes)
        severity = str(self.severity).strip().lower()
        if severity not in {"warning", "error", "critical"}:
            raise StageEditRepairInputError("repair issue severity is invalid")
        object.__setattr__(self, "severity", severity)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StageEditRepairIssue":
        raw_causes = payload.get("causes", payload.get("issue_codes", ()))
        if not isinstance(raw_causes, (list, tuple)):
            raise StageEditRepairInputError("repair issue causes must be an array")
        return cls(
            issue_id=str(payload.get("issue_id", payload.get("code", ""))),
            start_seconds=payload.get("start_seconds"),
            end_seconds=payload.get("end_seconds"),
            causes=tuple(str(value) for value in raw_causes),
            severity=str(payload.get("severity", "error")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StageEditCutExclusion:
    start_seconds: float
    end_seconds: float
    reason: str

    def __post_init__(self) -> None:
        start = _finite(self.start_seconds, "cut exclusion start_seconds")
        end = _finite(self.end_seconds, "cut exclusion end_seconds")
        if start < 0.0 or end < start:
            raise StageEditRepairInputError("cut exclusion span is invalid")
        reason = str(self.reason).strip()
        if not reason:
            raise StageEditRepairInputError("cut exclusion reason cannot be blank")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True, slots=True)
class StageEditRepairExclusions:
    """Hard exclusions enforced again after the external beam callback."""

    source_indices: tuple[int, ...] = ()
    cut_windows: tuple[StageEditCutExclusion, ...] = ()
    transition_kinds: tuple[str, ...] = ()
    transition_labels: tuple[str, ...] = ()
    crop_modes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        indices = tuple(self.source_indices)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in indices
        ):
            raise StageEditRepairInputError(
                "excluded source indices must be non-negative integers"
            )
        if len(set(indices)) != len(indices):
            raise StageEditRepairInputError("excluded source indices must be unique")
        object.__setattr__(self, "source_indices", tuple(sorted(indices)))
        windows = tuple(self.cut_windows)
        if any(not isinstance(value, StageEditCutExclusion) for value in windows):
            raise StageEditRepairInputError(
                "cut exclusions must be StageEditCutExclusion values"
            )
        object.__setattr__(
            self,
            "cut_windows",
            tuple(
                sorted(
                    windows,
                    key=lambda value: (
                        value.start_seconds,
                        value.end_seconds,
                        value.reason,
                    ),
                )
            ),
        )
        kinds = tuple(str(value).strip() for value in self.transition_kinds)
        valid_kinds = {value.value for value in TransitionKind}
        if any(value not in valid_kinds for value in kinds) or len(set(kinds)) != len(kinds):
            raise StageEditRepairInputError("excluded transition kinds are invalid")
        object.__setattr__(self, "transition_kinds", tuple(sorted(kinds)))
        labels = _strings(self.transition_labels, "excluded transition labels")
        object.__setattr__(self, "transition_labels", tuple(sorted(labels)))
        crop_modes = tuple(str(value).strip() for value in self.crop_modes)
        if any(value not in _CROP_MODES for value in crop_modes) or len(
            set(crop_modes)
        ) != len(crop_modes):
            raise StageEditRepairInputError("excluded crop modes are invalid")
        object.__setattr__(self, "crop_modes", tuple(sorted(crop_modes)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_indices": list(self.source_indices),
            "cut_windows": [asdict(value) for value in self.cut_windows],
            "transition_kinds": list(self.transition_kinds),
            "transition_labels": list(self.transition_labels),
            "crop_modes": list(self.crop_modes),
        }


@dataclass(frozen=True, slots=True)
class StageEditRepairConfig:
    context_seconds: float = 3.0
    maximum_iterations: int = 4
    maximum_consecutive_no_improvement: int = 1
    maximum_proposals_per_iteration: int = 32
    strict_improvement_epsilon: float = 1e-9
    canonical_boundary_tolerance_seconds: float = 1e-6

    def __post_init__(self) -> None:
        context = _finite(self.context_seconds, "repair context_seconds")
        if context < 0.0:
            raise StageEditRepairInputError("repair context_seconds cannot be negative")
        object.__setattr__(self, "context_seconds", context)
        for name, maximum in (
            ("maximum_iterations", 32),
            ("maximum_consecutive_no_improvement", 8),
            ("maximum_proposals_per_iteration", 32),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= maximum
            ):
                raise StageEditRepairInputError(
                    f"{name} must be an integer in 1...{maximum}"
                )
        epsilon = _finite(
            self.strict_improvement_epsilon, "strict_improvement_epsilon"
        )
        if epsilon < 0.0:
            raise StageEditRepairInputError(
                "strict_improvement_epsilon cannot be negative"
            )
        object.__setattr__(self, "strict_improvement_epsilon", epsilon)
        tolerance = _finite(
            self.canonical_boundary_tolerance_seconds,
            "canonical_boundary_tolerance_seconds",
        )
        if tolerance <= 0.0:
            raise StageEditRepairInputError(
                "canonical_boundary_tolerance_seconds must be positive"
            )
        object.__setattr__(
            self, "canonical_boundary_tolerance_seconds", tolerance
        )


@dataclass(frozen=True, slots=True)
class StageEditRepairReview:
    decoded_objective: float
    professional_objective: float
    cadence_objective: float
    technical_errors: tuple[str, ...] = ()
    issue_windows: tuple[StageEditRepairIssue, ...] = ()
    audit: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "decoded_objective",
            "professional_objective",
            "cadence_objective",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(
            self,
            "technical_errors",
            tuple(sorted(_strings(self.technical_errors, "technical_errors"))),
        )
        issues = tuple(self.issue_windows)
        if any(not isinstance(value, StageEditRepairIssue) for value in issues):
            raise StageEditRepairInputError(
                "review issue_windows must contain StageEditRepairIssue values"
            )
        issue_ids = tuple(value.issue_id for value in issues)
        if len(set(issue_ids)) != len(issue_ids):
            raise StageEditRepairInputError("review issue ids must be unique")
        object.__setattr__(
            self,
            "issue_windows",
            tuple(sorted(issues, key=_issue_sort_key)),
        )
        if not isinstance(self.audit, Mapping):
            raise StageEditRepairInputError("review audit must be a mapping")
        object.__setattr__(
            self,
            "audit",
            _canonical_audit(self.audit, "review audit"),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StageEditRepairReview":
        required = (
            "decoded_objective",
            "professional_objective",
            "cadence_objective",
            "technical_errors",
        )
        missing = tuple(name for name in required if name not in payload)
        if missing:
            raise StageEditRepairInputError(
                f"review mapping is missing required fields: {', '.join(missing)}"
            )
        raw_errors = payload["technical_errors"]
        raw_issues = payload.get("issue_windows", ())
        if not isinstance(raw_errors, (list, tuple)) or not isinstance(
            raw_issues, (list, tuple)
        ):
            raise StageEditRepairInputError(
                "review technical_errors/issue_windows must be arrays"
            )
        issues = tuple(
            value
            if isinstance(value, StageEditRepairIssue)
            else StageEditRepairIssue.from_mapping(value)
            if isinstance(value, Mapping)
            else (_raise_bad_review_issue())
            for value in raw_issues
        )
        audit = payload.get("audit", {})
        if not isinstance(audit, Mapping):
            raise StageEditRepairInputError("review audit must be a mapping")
        return cls(
            decoded_objective=payload["decoded_objective"],
            professional_objective=payload["professional_objective"],
            cadence_objective=payload["cadence_objective"],
            technical_errors=tuple(str(value) for value in raw_errors),
            issue_windows=issues,
            audit=dict(audit),
        )

    @property
    def technical_zero(self) -> bool:
        return not self.technical_errors

    @property
    def aggregate_objective(self) -> float:
        return (
            self.decoded_objective
            + self.professional_objective
            + self.cadence_objective
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoded_objective": self.decoded_objective,
            "professional_objective": self.professional_objective,
            "cadence_objective": self.cadence_objective,
            "aggregate_objective": self.aggregate_objective,
            "technical_errors": list(self.technical_errors),
            "technical_zero": self.technical_zero,
            "issue_windows": [value.to_dict() for value in self.issue_windows],
            "audit": dict(self.audit),
        }


def _raise_bad_review_issue() -> StageEditRepairIssue:
    raise StageEditRepairInputError("review issue window has an invalid type")


@dataclass(frozen=True, slots=True)
class StageEditRepairWindow:
    issue_ids: tuple[str, ...]
    causes: tuple[str, ...]
    requested_start_seconds: float
    requested_end_seconds: float
    context_start_seconds: float
    context_end_seconds: float
    canonical_start_seconds: float
    canonical_end_seconds: float
    actual_start_seconds: float
    actual_end_seconds: float
    start_shot: int
    end_shot_exclusive: int

    @property
    def duration_seconds(self) -> float:
        return self.actual_end_seconds - self.actual_start_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True, slots=True)
class StageEditRepairRequest:
    iteration: int
    base_edit: StageEdit = field(repr=False)
    current_review: StageEditRepairReview
    window: StageEditRepairWindow
    canonical_boundaries_seconds: tuple[float, ...]
    exclusions: StageEditRepairExclusions
    forbidden_edit_sha256s: tuple[str, ...]

    @property
    def left_context_shot(self) -> StageShot | None:
        return (
            self.base_edit.shots[self.window.start_shot - 1]
            if self.window.start_shot > 0
            else None
        )

    @property
    def right_context_shot(self) -> StageShot | None:
        return (
            self.base_edit.shots[self.window.end_shot_exclusive]
            if self.window.end_shot_exclusive < len(self.base_edit.shots)
            else None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "base_edit_sha256": _edit_signature(self.base_edit),
            "current_review": self.current_review.to_dict(),
            "window": self.window.to_dict(),
            "canonical_boundaries_seconds": list(
                self.canonical_boundaries_seconds
            ),
            "exclusions": self.exclusions.to_dict(),
            "forbidden_edit_sha256s": list(self.forbidden_edit_sha256s),
            "left_context_decision_id": (
                self.left_context_shot.decision_id
                if self.left_context_shot is not None
                else None
            ),
            "right_context_decision_id": (
                self.right_context_shot.decision_id
                if self.right_context_shot is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class StageEditRepairProposal:
    proposal_id: str
    rank: int
    replacement_shots: tuple[StageShot, ...]
    replacement_boundaries: tuple[StageBoundary, ...]
    audit: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        proposal_id = str(self.proposal_id).strip()
        if not proposal_id:
            raise StageEditRepairInputError("repair proposal_id cannot be blank")
        object.__setattr__(self, "proposal_id", proposal_id)
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise StageEditRepairInputError("repair proposal rank must be positive")
        shots = tuple(self.replacement_shots)
        boundaries = tuple(self.replacement_boundaries)
        if not shots or any(not isinstance(value, StageShot) for value in shots):
            raise StageEditRepairInputError(
                "repair proposal needs StageShot replacement values"
            )
        if any(not isinstance(value, StageBoundary) for value in boundaries):
            raise StageEditRepairInputError(
                "repair proposal boundaries must be StageBoundary values"
            )
        object.__setattr__(self, "replacement_shots", shots)
        object.__setattr__(self, "replacement_boundaries", boundaries)
        if not isinstance(self.audit, Mapping):
            raise StageEditRepairInputError("repair proposal audit must be a mapping")
        object.__setattr__(
            self,
            "audit",
            _canonical_audit(self.audit, "repair proposal audit"),
        )


@dataclass(frozen=True, slots=True)
class StageEditRepairIterationAudit:
    iteration: int
    request: Mapping[str, Any]
    proposal_audits: tuple[Mapping[str, Any], ...]
    committed_proposal_id: str | None
    committed_edit_sha256: str | None
    no_improvement_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "request": dict(self.request),
            "proposal_audits": [dict(value) for value in self.proposal_audits],
            "committed_proposal_id": self.committed_proposal_id,
            "committed_edit_sha256": self.committed_edit_sha256,
            "no_improvement_count": self.no_improvement_count,
        }


@dataclass(frozen=True, slots=True)
class StageEditRepairAudit:
    schema_version: int
    orchestrator: str
    input_sha256: str
    config: Mapping[str, Any]
    exclusions: Mapping[str, Any]
    canonical_boundaries_seconds: tuple[float, ...]
    candidate_reviews: tuple[Mapping[str, Any], ...]
    selected_candidate_index: int
    initial_edit_sha256: str
    final_edit_sha256: str
    initial_review: Mapping[str, Any]
    final_review: Mapping[str, Any]
    iterations: tuple[StageEditRepairIterationAudit, ...]
    termination_reason: str
    committed_iteration_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "orchestrator": self.orchestrator,
            "input_sha256": self.input_sha256,
            "config": dict(self.config),
            "exclusions": dict(self.exclusions),
            "canonical_boundaries_seconds": list(
                self.canonical_boundaries_seconds
            ),
            "candidate_reviews": [dict(value) for value in self.candidate_reviews],
            "selected_candidate_index": self.selected_candidate_index,
            "initial_edit_sha256": self.initial_edit_sha256,
            "final_edit_sha256": self.final_edit_sha256,
            "initial_review": dict(self.initial_review),
            "final_review": dict(self.final_review),
            "iterations": [value.to_dict() for value in self.iterations],
            "termination_reason": self.termination_reason,
            "committed_iteration_count": self.committed_iteration_count,
        }


@dataclass(frozen=True, slots=True)
class StageEditRepairResult:
    original_edit: StageEdit
    edit: StageEdit
    changed: bool
    audit: StageEditRepairAudit

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "original_edit": self.original_edit.to_dict(),
            "edit": self.edit.to_dict(),
            "audit": self.audit.to_dict(),
        }


BeamRedecoder = Callable[
    [StageEditRepairRequest], Sequence[StageEditRepairProposal]
]
ReviewScorer = Callable[
    [StageEdit], StageEditRepairReview | Mapping[str, Any]
]


def _issue_sort_key(issue: StageEditRepairIssue) -> tuple[Any, ...]:
    severity_rank = {"critical": 0, "error": 1, "warning": 2}
    return (
        issue.start_seconds,
        issue.end_seconds,
        severity_rank[issue.severity],
        issue.issue_id,
    )


def _edit_signature(edit: StageEdit) -> str:
    payload = edit.to_render_plan().to_json()
    payload.pop("output", None)
    payload["shot_section_indices"] = [value.section_index for value in edit.shots]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _coerce_review(value: StageEditRepairReview | Mapping[str, Any]) -> StageEditRepairReview:
    if isinstance(value, StageEditRepairReview):
        return value
    if isinstance(value, Mapping):
        return StageEditRepairReview.from_mapping(value)
    raise StageEditRepairCallbackError(
        "review scorer must return StageEditRepairReview or a mapping"
    )


def _score_edit(review_scorer: ReviewScorer, edit: StageEdit) -> StageEditRepairReview:
    try:
        value = review_scorer(edit)
        return _coerce_review(value)
    except StageEditRepairCallbackError:
        raise
    except Exception as exc:
        raise StageEditRepairCallbackError(
            f"review scorer failed zero-tolerance validation: {exc}"
        ) from exc


def _candidate_compatibility_key(edit: StageEdit) -> tuple[Any, ...]:
    return (
        tuple(edit.sources),
        edit.reference_source,
        edit.aspect_ratio,
        edit.width,
        edit.height,
        edit.fps,
        edit.master_audio_start,
        edit.duration_seconds,
        json.dumps(edit.section_plan.to_dict(), sort_keys=True),
    )


def _canonical_boundaries(
    values: Sequence[float],
    edit: StageEdit,
    tolerance: float,
) -> tuple[float, ...]:
    result = tuple(_finite(value, "canonical shot boundary") for value in values)
    if not result or any(right <= left for left, right in zip(result, result[1:])):
        raise StageEditRepairInputError(
            "canonical shot boundaries must be non-empty and strictly increasing"
        )
    start = edit.master_audio_start
    end = start + edit.duration_seconds
    if abs(result[0] - start) > tolerance or abs(result[-1] - end) > tolerance:
        raise StageEditRepairInputError(
            "canonical shot boundaries must include the exact edit endpoints"
        )
    if result[0] < -tolerance:
        raise StageEditRepairInputError("canonical boundaries cannot be negative")
    return result


def _absolute_shot_spans(edit: StageEdit) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            edit.master_audio_start + shot.segment.master_start,
            edit.master_audio_start
            + shot.segment.master_start
            + shot.segment.duration,
        )
        for shot in edit.shots
    )


def _snap_outward(
    start: float,
    end: float,
    boundaries: tuple[float, ...],
) -> tuple[float, float]:
    left = max(
        (value for value in boundaries if value <= start + _TIME_EPSILON),
        default=boundaries[0],
    )
    right = min(
        (value for value in boundaries if value >= end - _TIME_EPSILON),
        default=boundaries[-1],
    )
    return left, right


def _window_for_issue(
    issue: StageEditRepairIssue,
    edit: StageEdit,
    canonical: tuple[float, ...],
    context_seconds: float,
) -> StageEditRepairWindow:
    edit_start = edit.master_audio_start
    edit_end = edit_start + edit.duration_seconds
    if (
        issue.end_seconds < edit_start - _TIME_EPSILON
        or issue.start_seconds > edit_end + _TIME_EPSILON
    ):
        raise StageEditRepairInputError(
            f"issue {issue.issue_id!r} is outside the StageEdit timeline"
        )
    requested_start = max(edit_start, issue.start_seconds)
    requested_end = min(edit_end, issue.end_seconds)
    context_start = max(edit_start, requested_start - context_seconds)
    context_end = min(edit_end, requested_end + context_seconds)
    canonical_start, canonical_end = _snap_outward(
        context_start, context_end, canonical
    )
    spans = _absolute_shot_spans(edit)
    selected = tuple(
        index
        for index, (start, end) in enumerate(spans)
        if end > canonical_start + _TIME_EPSILON
        and start < canonical_end - _TIME_EPSILON
    )
    if not selected:
        # A zero-width boundary issue belongs to both touching shots where
        # possible, which keeps the splice's exterior transitions reviewable.
        touching = tuple(
            index
            for index, (start, end) in enumerate(spans)
            if start - _TIME_EPSILON <= canonical_start <= end + _TIME_EPSILON
        )
        selected = touching
    if not selected:
        raise StageEditRepairInputError(
            f"issue {issue.issue_id!r} cannot be mapped to StageEdit shots"
        )
    start_shot = min(selected)
    end_shot = max(selected) + 1
    return StageEditRepairWindow(
        issue_ids=(issue.issue_id,),
        causes=issue.causes,
        requested_start_seconds=requested_start,
        requested_end_seconds=requested_end,
        context_start_seconds=context_start,
        context_end_seconds=context_end,
        canonical_start_seconds=canonical_start,
        canonical_end_seconds=canonical_end,
        actual_start_seconds=spans[start_shot][0],
        actual_end_seconds=spans[end_shot - 1][1],
        start_shot=start_shot,
        end_shot_exclusive=end_shot,
    )


def _merge_windows(
    windows: Sequence[StageEditRepairWindow],
    edit: StageEdit,
) -> tuple[StageEditRepairWindow, ...]:
    ordered = sorted(
        windows,
        key=lambda value: (
            value.start_shot,
            value.end_shot_exclusive,
            value.issue_ids,
        ),
    )
    merged: list[StageEditRepairWindow] = []
    spans = _absolute_shot_spans(edit)
    for window in ordered:
        if not merged or window.start_shot > merged[-1].end_shot_exclusive:
            merged.append(window)
            continue
        previous = merged.pop()
        start_shot = min(previous.start_shot, window.start_shot)
        end_shot = max(previous.end_shot_exclusive, window.end_shot_exclusive)
        merged.append(
            StageEditRepairWindow(
                issue_ids=tuple(sorted(set(previous.issue_ids + window.issue_ids))),
                causes=tuple(sorted(set(previous.causes + window.causes))),
                requested_start_seconds=min(
                    previous.requested_start_seconds,
                    window.requested_start_seconds,
                ),
                requested_end_seconds=max(
                    previous.requested_end_seconds,
                    window.requested_end_seconds,
                ),
                context_start_seconds=min(
                    previous.context_start_seconds,
                    window.context_start_seconds,
                ),
                context_end_seconds=max(
                    previous.context_end_seconds,
                    window.context_end_seconds,
                ),
                canonical_start_seconds=min(
                    previous.canonical_start_seconds,
                    window.canonical_start_seconds,
                ),
                canonical_end_seconds=max(
                    previous.canonical_end_seconds,
                    window.canonical_end_seconds,
                ),
                actual_start_seconds=spans[start_shot][0],
                actual_end_seconds=spans[end_shot - 1][1],
                start_shot=start_shot,
                end_shot_exclusive=end_shot,
            )
        )
    return tuple(merged)


def _repair_windows(
    issues: Sequence[StageEditRepairIssue],
    edit: StageEdit,
    canonical: tuple[float, ...],
    context_seconds: float,
) -> tuple[StageEditRepairWindow, ...]:
    return _merge_windows(
        tuple(
            _window_for_issue(issue, edit, canonical, context_seconds)
            for issue in sorted(issues, key=_issue_sort_key)
        ),
        edit,
    )


def _proposal_exclusion_reasons(
    proposal: StageEditRepairProposal,
    request: StageEditRepairRequest,
) -> tuple[str, ...]:
    reasons: set[str] = set()
    exclusions = request.exclusions
    for shot in proposal.replacement_shots:
        if shot.segment.source_index in exclusions.source_indices:
            reasons.add(f"excluded_source:{shot.segment.source_index}")
        modes = {
            name
            for name, present in (
                ("static_crop", shot.segment.crop is not None),
                ("pan_crop", shot.segment.pan_crop is not None),
                ("editorial_transform", shot.segment.editorial_transform is not None),
            )
            if present
        }
        for mode in sorted(modes.intersection(exclusions.crop_modes)):
            reasons.add(f"excluded_crop_mode:{mode}")

    # Only replacement-internal cut times are free decisions.  The two window
    # edges are fixed continuity anchors and are not reclassified as new cuts.
    for shot in proposal.replacement_shots[1:]:
        cut_time = request.base_edit.master_audio_start + shot.segment.master_start
        for window in exclusions.cut_windows:
            if (
                window.start_seconds - _TIME_EPSILON
                <= cut_time
                <= window.end_seconds + _TIME_EPSILON
            ):
                reasons.add(f"excluded_cut_window:{window.reason}")

    lowered_labels = tuple(value.casefold() for value in exclusions.transition_labels)
    for boundary in proposal.replacement_boundaries:
        transition = boundary.transition
        if transition.kind.value in exclusions.transition_kinds:
            reasons.add(f"excluded_transition_kind:{transition.kind.value}")
        lowered_reason = transition.reason.casefold()
        for original, lowered in zip(exclusions.transition_labels, lowered_labels, strict=True):
            if lowered in lowered_reason:
                reasons.add(f"excluded_transition_label:{original}")
    return tuple(sorted(reasons))


def _shot_ir_without_index(shot: StageShot) -> tuple[Any, ...]:
    return (
        shot.section_index,
        shot.segment,
        shot.decision_id,
        shot.decision_confidence,
        dict(shot.color_grade) if shot.color_grade is not None else None,
    )


def _boundary_ir_without_index(boundary: StageBoundary) -> tuple[Any, ...]:
    return (
        boundary.transition,
        boundary.reviewer_status,
        boundary.issue_codes,
    )


def _outside_ir_is_unchanged(
    before: StageEdit,
    after: StageEdit,
    *,
    start_shot: int,
    end_shot_exclusive: int,
    replacement_shot_count: int,
) -> bool:
    old_prefix = tuple(
        _shot_ir_without_index(value) for value in before.shots[:start_shot]
    )
    new_prefix = tuple(
        _shot_ir_without_index(value) for value in after.shots[:start_shot]
    )
    new_suffix_start = start_shot + replacement_shot_count
    old_suffix = tuple(
        _shot_ir_without_index(value)
        for value in before.shots[end_shot_exclusive:]
    )
    new_suffix = tuple(
        _shot_ir_without_index(value) for value in after.shots[new_suffix_start:]
    )
    old_boundary_start = max(0, start_shot - 1)
    old_boundary_end = min(len(before.boundaries), end_shot_exclusive)
    replacement_boundary_count = (
        replacement_shot_count
        - 1
        + (1 if start_shot > 0 else 0)
        + (1 if end_shot_exclusive < len(before.shots) else 0)
    )
    new_boundary_end = old_boundary_start + replacement_boundary_count
    old_boundary_prefix = tuple(
        _boundary_ir_without_index(value)
        for value in before.boundaries[:old_boundary_start]
    )
    new_boundary_prefix = tuple(
        _boundary_ir_without_index(value)
        for value in after.boundaries[:old_boundary_start]
    )
    old_boundary_suffix = tuple(
        _boundary_ir_without_index(value)
        for value in before.boundaries[old_boundary_end:]
    )
    new_boundary_suffix = tuple(
        _boundary_ir_without_index(value)
        for value in after.boundaries[new_boundary_end:]
    )
    return (
        old_prefix == new_prefix
        and old_suffix == new_suffix
        and old_boundary_prefix == new_boundary_prefix
        and old_boundary_suffix == new_boundary_suffix
    )


def _strict_deltas(
    before: StageEditRepairReview,
    after: StageEditRepairReview,
) -> dict[str, float]:
    return {
        "decoded_objective": after.decoded_objective - before.decoded_objective,
        "professional_objective": (
            after.professional_objective - before.professional_objective
        ),
        "cadence_objective": after.cadence_objective - before.cadence_objective,
    }


def _strictly_improves(
    deltas: Mapping[str, float],
    epsilon: float,
) -> bool:
    return all(value > epsilon for value in deltas.values())


def _review_decision_contract(
    review: StageEditRepairReview,
) -> tuple[Any, ...]:
    return (
        review.decoded_objective,
        review.professional_objective,
        review.cadence_objective,
        review.technical_errors,
        review.issue_windows,
    )


def _pending_after_commit(
    pending: Sequence[StageEditRepairIssue],
    scored_issues: Sequence[StageEditRepairIssue],
    window: StageEditRepairWindow,
) -> tuple[StageEditRepairIssue, ...]:
    remaining = tuple(
        issue
        for issue in pending
        if issue.end_seconds < window.actual_start_seconds - _TIME_EPSILON
        or issue.start_seconds > window.actual_end_seconds + _TIME_EPSILON
    )
    combined = remaining + tuple(scored_issues)
    unique: dict[tuple[Any, ...], StageEditRepairIssue] = {}
    for issue in combined:
        key = (
            issue.issue_id,
            issue.start_seconds,
            issue.end_seconds,
            issue.causes,
            issue.severity,
        )
        unique[key] = issue
    return tuple(sorted(unique.values(), key=_issue_sort_key))


def _input_digest(
    candidates: Sequence[StageEdit],
    issues: Sequence[StageEditRepairIssue],
    canonical: Sequence[float],
    exclusions: StageEditRepairExclusions,
    config: StageEditRepairConfig,
) -> str:
    payload = {
        "candidate_sha256s": [_edit_signature(value) for value in candidates],
        "issues": [value.to_dict() for value in issues],
        "canonical_boundaries_seconds": list(canonical),
        "exclusions": exclusions.to_dict(),
        "config": asdict(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def repair_stage_edit_candidates(
    *,
    candidates: Sequence[StageEdit],
    reviewer_issues: Sequence[StageEditRepairIssue | Mapping[str, Any]],
    canonical_shot_boundaries_seconds: Sequence[float],
    beam_redecoder: BeamRedecoder,
    review_scorer: ReviewScorer,
    exclusions: StageEditRepairExclusions = StageEditRepairExclusions(),
    config: StageEditRepairConfig = StageEditRepairConfig(),
) -> StageEditRepairResult:
    """Repair the best compatible StageEdit without mutating media or pipeline state."""

    edits = tuple(candidates)
    if not edits or any(not isinstance(value, StageEdit) for value in edits):
        raise StageEditRepairInputError(
            "candidates must contain at least one StageEdit"
        )
    compatibility = _candidate_compatibility_key(edits[0])
    if any(_candidate_compatibility_key(value) != compatibility for value in edits[1:]):
        raise StageEditRepairInputError(
            "StageEdit candidates must share sources, clock, format, and section plan"
        )
    if not callable(beam_redecoder) or not callable(review_scorer):
        raise StageEditRepairInputError(
            "beam_redecoder and review_scorer must be callable"
        )
    if any(index >= len(edits[0].sources) for index in exclusions.source_indices):
        raise StageEditRepairInputError(
            "excluded source index is outside the StageEdit source list"
        )
    issues = tuple(
        value
        if isinstance(value, StageEditRepairIssue)
        else StageEditRepairIssue.from_mapping(value)
        if isinstance(value, Mapping)
        else (_raise_bad_review_issue())
        for value in reviewer_issues
    )
    issue_ids = tuple(value.issue_id for value in issues)
    if len(set(issue_ids)) != len(issue_ids):
        raise StageEditRepairInputError("initial reviewer issue ids must be unique")

    canonical = _canonical_boundaries(
        canonical_shot_boundaries_seconds,
        edits[0],
        config.canonical_boundary_tolerance_seconds,
    )
    candidate_reviews: list[StageEditRepairReview] = []
    candidate_audit: list[dict[str, Any]] = []
    for index, edit in enumerate(edits):
        review = _score_edit(review_scorer, edit)
        candidate_reviews.append(review)
        candidate_audit.append(
            {
                "candidate_index": index,
                "edit_sha256": _edit_signature(edit),
                "review": review.to_dict(),
            }
        )
    selected_index = max(
        range(len(edits)),
        key=lambda index: (
            candidate_reviews[index].technical_zero,
            candidate_reviews[index].decoded_objective,
            candidate_reviews[index].professional_objective,
            candidate_reviews[index].cadence_objective,
            -index,
        ),
    )
    original = edits[selected_index]
    current = original
    initial_review = candidate_reviews[selected_index]
    current_review = initial_review
    initial_signature = _edit_signature(original)
    seen_signatures = {initial_signature}
    pending_issues = tuple(sorted(issues, key=_issue_sort_key))
    iteration_audits: list[StageEditRepairIterationAudit] = []
    no_improvement_count = 0
    committed_count = 0
    termination_reason = "maximum_iterations_reached"

    if not pending_issues:
        termination_reason = "no_reviewer_issues"

    for iteration in range(1, config.maximum_iterations + 1):
        if not pending_issues:
            termination_reason = (
                "reviewer_clean_no_remaining_issues"
                if committed_count
                else "no_reviewer_issues"
            )
            break
        windows = _repair_windows(
            pending_issues,
            current,
            canonical,
            config.context_seconds,
        )
        if not windows:
            termination_reason = "no_repairable_issue_window"
            break
        window = windows[0]
        request = StageEditRepairRequest(
            iteration=iteration,
            base_edit=current,
            current_review=current_review,
            window=window,
            canonical_boundaries_seconds=canonical,
            exclusions=exclusions,
            forbidden_edit_sha256s=tuple(sorted(seen_signatures)),
        )
        try:
            raw_proposals = beam_redecoder(request)
        except Exception as exc:
            raise StageEditRepairCallbackError(
                f"beam redecoder failed at iteration {iteration}: {exc}"
            ) from exc
        if isinstance(raw_proposals, (str, bytes)) or not isinstance(
            raw_proposals, Sequence
        ):
            raise StageEditRepairCallbackError(
                "beam redecoder must return a proposal sequence"
            )
        proposals = tuple(raw_proposals)
        if len(proposals) > config.maximum_proposals_per_iteration:
            raise StageEditRepairCallbackError(
                "beam redecoder exceeded maximum_proposals_per_iteration"
            )
        if any(not isinstance(value, StageEditRepairProposal) for value in proposals):
            raise StageEditRepairCallbackError(
                "beam redecoder returned a non-StageEditRepairProposal value"
            )
        proposal_ids = tuple(value.proposal_id for value in proposals)
        proposal_ranks = tuple(value.rank for value in proposals)
        if len(set(proposal_ids)) != len(proposal_ids) or len(
            set(proposal_ranks)
        ) != len(proposal_ranks):
            raise StageEditRepairCallbackError(
                "beam proposal ids and ranks must be unique"
            )
        ordered = tuple(sorted(proposals, key=lambda value: (value.rank, value.proposal_id)))

        evaluated: list[
            tuple[StageEditRepairProposal, StageEdit, StageEditRepairReview, dict[str, float]]
        ] = []
        proposal_audits: list[dict[str, Any]] = []
        for proposal in ordered:
            record: dict[str, Any] = {
                "proposal_id": proposal.proposal_id,
                "rank": proposal.rank,
                "decoder_audit": dict(proposal.audit),
                "replacement_shot_count": len(proposal.replacement_shots),
                "replacement_boundary_count": len(proposal.replacement_boundaries),
                "validated": False,
                "committed": False,
                "rejection_reasons": [],
                "edit_sha256": None,
                "review": None,
                "objective_deltas": None,
                "post_commit_review_verified": False,
            }
            exclusion_reasons = _proposal_exclusion_reasons(proposal, request)
            if exclusion_reasons:
                record["rejection_reasons"] = list(exclusion_reasons)
                proposal_audits.append(record)
                continue
            try:
                trial = current.replace_window(
                    start_shot=window.start_shot,
                    end_shot_exclusive=window.end_shot_exclusive,
                    replacement_shots=proposal.replacement_shots,
                    replacement_boundaries=proposal.replacement_boundaries,
                    reason=(
                        "stage_edit_repair_trial:"
                        + ",".join(window.causes)
                    ),
                    score_before=None,
                    score_after=None,
                )
            except (TypeError, ValueError) as exc:
                record["rejection_reasons"] = [
                    f"splice_or_ir_validation_failed:{exc}"
                ]
                proposal_audits.append(record)
                continue
            if not _outside_ir_is_unchanged(
                current,
                trial,
                start_shot=window.start_shot,
                end_shot_exclusive=window.end_shot_exclusive,
                replacement_shot_count=len(proposal.replacement_shots),
            ):
                record["rejection_reasons"] = [
                    "outside_window_ir_continuity_changed"
                ]
                proposal_audits.append(record)
                continue
            signature = _edit_signature(trial)
            record["edit_sha256"] = signature
            if signature in seen_signatures:
                record["rejection_reasons"] = ["repeated_edit_result"]
                proposal_audits.append(record)
                continue
            seen_signatures.add(signature)
            after_review = _score_edit(review_scorer, trial)
            deltas = _strict_deltas(current_review, after_review)
            record["review"] = after_review.to_dict()
            record["objective_deltas"] = dict(deltas)
            record["validated"] = True
            if after_review.technical_errors:
                record["rejection_reasons"] = [
                    "technical_zero_tolerance_failed"
                ]
                proposal_audits.append(record)
                continue
            if not _strictly_improves(
                deltas, config.strict_improvement_epsilon
            ):
                record["rejection_reasons"] = [
                    "all_objectives_must_strictly_improve"
                ]
                proposal_audits.append(record)
                continue
            evaluated.append((proposal, trial, after_review, deltas))
            proposal_audits.append(record)

        committed_proposal_id: str | None = None
        committed_signature: str | None = None
        if evaluated:
            selected = max(
                evaluated,
                key=lambda value: (
                    sum(value[3].values()),
                    value[2].decoded_objective,
                    value[2].professional_objective,
                    value[2].cadence_objective,
                    -value[0].rank,
                    value[0].proposal_id,
                ),
            )
            proposal, _trial, after_review, _deltas = selected
            committed = current.replace_window(
                start_shot=window.start_shot,
                end_shot_exclusive=window.end_shot_exclusive,
                replacement_shots=proposal.replacement_shots,
                replacement_boundaries=proposal.replacement_boundaries,
                reason=(
                    "stage_edit_repair:"
                    + ",".join(window.causes)
                    + f":{proposal.proposal_id}"
                ),
                score_before=current_review.aggregate_objective,
                score_after=after_review.aggregate_objective,
            ).with_reviewer_result(
                {
                    "repair_iteration": iteration,
                    "proposal_id": proposal.proposal_id,
                    "strict_objective_improvement": True,
                    "technical_zero": True,
                    "review": after_review.to_dict(),
                }
            )
            verified_review = _score_edit(review_scorer, committed)
            if _review_decision_contract(verified_review) != _review_decision_contract(
                after_review
            ):
                raise StageEditRepairCallbackError(
                    "review scorer changed its decision for identical rendered IR"
                )
            committed_signature = _edit_signature(committed)
            current = committed
            current_review = verified_review
            committed_proposal_id = proposal.proposal_id
            committed_count += 1
            no_improvement_count = 0
            pending_issues = _pending_after_commit(
                pending_issues,
                verified_review.issue_windows,
                window,
            )
            for record in proposal_audits:
                if record["proposal_id"] == proposal.proposal_id:
                    record["committed"] = True
                    record["post_commit_review_verified"] = True
                    record["rejection_reasons"] = []
                    break
        else:
            no_improvement_count += 1

        iteration_audits.append(
            StageEditRepairIterationAudit(
                iteration=iteration,
                request=request.to_dict(),
                proposal_audits=tuple(proposal_audits),
                committed_proposal_id=committed_proposal_id,
                committed_edit_sha256=committed_signature,
                no_improvement_count=no_improvement_count,
            )
        )
        if committed_proposal_id is not None and not pending_issues:
            termination_reason = "reviewer_clean_no_remaining_issues"
            break
        if no_improvement_count >= config.maximum_consecutive_no_improvement:
            termination_reason = "no_strict_improvement"
            break
    else:
        termination_reason = "maximum_iterations_reached"

    final_signature = _edit_signature(current)
    digest = _input_digest(edits, issues, canonical, exclusions, config)
    audit = StageEditRepairAudit(
        schema_version=REPAIR_LOOP_SCHEMA_VERSION,
        orchestrator="stage_edit_repair_loop_v1",
        input_sha256=digest,
        config=asdict(config),
        exclusions=exclusions.to_dict(),
        canonical_boundaries_seconds=canonical,
        candidate_reviews=tuple(candidate_audit),
        selected_candidate_index=selected_index,
        initial_edit_sha256=initial_signature,
        final_edit_sha256=final_signature,
        initial_review=initial_review.to_dict(),
        final_review=current_review.to_dict(),
        iterations=tuple(iteration_audits),
        termination_reason=termination_reason,
        committed_iteration_count=committed_count,
    )
    return StageEditRepairResult(
        original_edit=original,
        edit=current,
        changed=final_signature != initial_signature,
        audit=audit,
    )


__all__ = [
    "BeamRedecoder",
    "REPAIR_LOOP_SCHEMA_VERSION",
    "ReviewScorer",
    "StageEditCutExclusion",
    "StageEditRepairAudit",
    "StageEditRepairCallbackError",
    "StageEditRepairConfig",
    "StageEditRepairError",
    "StageEditRepairExclusions",
    "StageEditRepairInputError",
    "StageEditRepairIssue",
    "StageEditRepairIterationAudit",
    "StageEditRepairProposal",
    "StageEditRepairRequest",
    "StageEditRepairResult",
    "StageEditRepairReview",
    "StageEditRepairWindow",
    "repair_stage_edit_candidates",
]
