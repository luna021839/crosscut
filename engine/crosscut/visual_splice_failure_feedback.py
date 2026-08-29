"""Translate exact boundary-repair evidence into optimizer feedback.

The pipeline may run a normal and then a wider repair search for the same
boundary.  This adapter keeps those attempts grouped by the exact
boundary/source/shot edge and exports a no-good only when a complete wide
search proves non-waivable incompatibility.  A later raw pass or a
focus/scale-only option always wins over an earlier failure certificate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from .timeline_no_good import (
    TimelineNoGoodEdgeKey,
    build_timeline_no_good_edge_key,
)
from .visual_splice_feasibility import (
    DEFAULT_WIDE_FRAME_OFFSETS,
    SpliceFeasibilityAssessment,
    SpliceFeasibilityStatus,
    classify_splice_evidence_audit,
    merge_splice_feasibility_assessments,
)


SPLICE_FAILURE_FEEDBACK_CONTRACT = (
    "exact_wide_boundary_failure_to_optimizer_feedback_v3"
)
SPLICE_EDGE_CONTEXT_CONTRACT = "exact_splice_clock_span_edge_context_v1"
SPLICE_RESULT_CONTEXT_CONTRACT = (
    "exact_splice_selected_boundary_visual_proof_context_v1"
)


@dataclass(frozen=True, slots=True)
class SpliceEdgeFeedback:
    edge: TimelineNoGoodEdgeKey
    edge_context_identity: str
    status: SpliceFeasibilityStatus
    assessments: tuple[SpliceFeasibilityAssessment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.edge, TimelineNoGoodEdgeKey):
            raise TypeError("splice feedback edge is invalid")
        if (
            not isinstance(self.edge_context_identity, str)
            or len(self.edge_context_identity) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.edge_context_identity
            )
        ):
            raise ValueError("splice feedback edge context identity is invalid")
        if not isinstance(self.status, SpliceFeasibilityStatus):
            raise TypeError("splice feedback status is invalid")
        values = tuple(
            sorted(
                self.assessments,
                key=lambda item: json.dumps(
                    item.to_dict(),
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        )
        if not values or any(
            not isinstance(item, SpliceFeasibilityAssessment)
            for item in values
        ):
            raise TypeError("splice feedback assessments are invalid")
        merged = values[0]
        for assessment in values[1:]:
            merged = merge_splice_feasibility_assessments(merged, assessment)
        if self.status is not merged.status:
            raise ValueError(
                "splice feedback status contradicts merged exact evidence"
            )
        object.__setattr__(self, "assessments", values)

    @property
    def optimizer_no_good_eligible(self) -> bool:
        return self.status is SpliceFeasibilityStatus.HARD_INCOMPATIBLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge": self.edge.to_dict(),
            "edge_context_identity": self.edge_context_identity,
            "status": self.status.value,
            "optimizer_no_good_eligible": self.optimizer_no_good_eligible,
            "assessment_count": len(self.assessments),
            "assessments": [item.to_dict() for item in self.assessments],
        }


@dataclass(frozen=True, slots=True)
class SpliceFailureFeedback:
    edges: tuple[SpliceEdgeFeedback, ...]

    def __post_init__(self) -> None:
        values = tuple(
            sorted(
                self.edges,
                key=lambda item: (item.edge, item.edge_context_identity),
            )
        )
        if any(not isinstance(item, SpliceEdgeFeedback) for item in values):
            raise TypeError("splice failure feedback entries are invalid")
        if len(
            {(item.edge, item.edge_context_identity) for item in values}
        ) != len(values):
            raise ValueError("splice failure feedback contexts cannot repeat")
        object.__setattr__(self, "edges", values)

    @property
    def hard_contexts(self) -> tuple[SpliceEdgeFeedback, ...]:
        return tuple(
            item for item in self.edges if item.optimizer_no_good_eligible
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": SPLICE_FAILURE_FEEDBACK_CONTRACT,
            "edge_count": len(self.edges),
            "hard_context_count": len(self.hard_contexts),
            "hard_contexts": [
                {
                    "edge": item.edge.to_dict(),
                    "edge_context_identity": item.edge_context_identity,
                }
                for item in self.hard_contexts
            ],
            "edges": [item.to_dict() for item in self.edges],
            "compact_optimizer_projection_requires_verified_raw_context": True,
            "thresholds_lowered": False,
        }


def _choices(plan_or_choices: object) -> tuple[object, ...]:
    raw = (
        plan_or_choices.get("choices")
        if isinstance(plan_or_choices, Mapping)
        else getattr(plan_or_choices, "choices", plan_or_choices)
    )
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("visual splice feedback needs plan choices")
    values = tuple(raw)
    if len(values) < 2:
        raise ValueError("visual splice feedback needs adjacent choices")
    return values


def _integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"splice failure {key} is invalid")
    return value


def _source_index(choice: object) -> int:
    if isinstance(choice, Mapping):
        value = choice.get("source_index", choice.get("source_id"))
    else:
        value = getattr(choice, "source_index", None)
        if value is None:
            value = getattr(choice, "source_id", None)
    if isinstance(value, str) and value.isdecimal():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("visual plan choice source index is invalid")
    return value


def _choice_value(choice: object, name: str) -> object:
    if isinstance(choice, Mapping):
        if name not in choice:
            raise TypeError(f"visual plan choice must expose {name}")
        return choice[name]
    if not hasattr(choice, name):
        raise TypeError(f"visual plan choice must expose {name}")
    return getattr(choice, name)


def _non_negative_choice_integer(choice: object, name: str) -> int:
    value = _choice_value(choice, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"visual plan choice {name} is invalid")
    return value


def _exact_clock_seconds(choice: object, name: str) -> str:
    value = _choice_value(choice, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"visual plan choice {name} must be numeric")
    seconds = float(value)
    if not math.isfinite(seconds):
        raise ValueError(f"visual plan choice {name} must be finite")
    if seconds == 0.0:
        seconds = 0.0
    return seconds.hex()


def _exact_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-blank exact string")
    return value


def _shot_id(choice: object) -> str:
    value = (
        choice.get("shot_id")
        if isinstance(choice, Mapping)
        else getattr(choice, "shot_id", None)
    )
    return _exact_text(value, "visual plan choice shot_id")


def _plan_analysis_basis(plan_or_choices: object) -> str:
    value = (
        plan_or_choices.get("analysis_basis")
        if isinstance(plan_or_choices, Mapping)
        else getattr(plan_or_choices, "analysis_basis", None)
    )
    return _exact_text(value, "visual plan analysis_basis")


def _plan_source_clock_basis(plan_or_choices: object) -> str:
    value = (
        plan_or_choices.get("source_clock_basis")
        if isinstance(plan_or_choices, Mapping)
        else getattr(plan_or_choices, "source_clock_basis", None)
    )
    return _exact_text(value, "visual plan source_clock_basis")


def _choice_identity_payload(choice: object) -> dict[str, object]:
    return {
        "segment_index": _non_negative_choice_integer(choice, "segment_index"),
        "source_index": _source_index(choice),
        "shot_id": _shot_id(choice),
        "canonical_master_start_hex": _exact_clock_seconds(
            choice,
            "canonical_master_start",
        ),
        "canonical_master_end_hex": _exact_clock_seconds(
            choice,
            "canonical_master_end",
        ),
        "source_local_start_hex": _exact_clock_seconds(
            choice,
            "source_local_start",
        ),
        "source_local_end_hex": _exact_clock_seconds(
            choice,
            "source_local_end",
        ),
    }


def _json_identity_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _json_identity_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_identity_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"splice identity payload contains {type(value)!r}")


def splice_feedback_context_identity(payload: Mapping[str, object]) -> str:
    if not isinstance(payload, Mapping):
        raise TypeError("splice feedback context must be a mapping")
    encoded = json.dumps(
        _json_identity_value(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_identity(payload: Mapping[str, object]) -> str:
    return splice_feedback_context_identity(payload)


def splice_feedback_plan_context(plan_or_choices: object) -> dict[str, object]:
    """Return the canonical clock/span context for one analyzed path."""

    choices = _choices(plan_or_choices)
    return {
        "contract": SPLICE_FAILURE_FEEDBACK_CONTRACT,
        "analysis_basis": _plan_analysis_basis(plan_or_choices),
        "source_clock_basis": _plan_source_clock_basis(plan_or_choices),
        "choices": [_choice_identity_payload(choice) for choice in choices],
    }


def splice_feedback_plan_identity(plan_or_choices: object) -> str:
    """Bind persisted repair evidence to one exact analyzed path."""

    return _payload_identity(splice_feedback_plan_context(plan_or_choices))


def splice_feedback_edge_context(
    plan_or_choices: object,
    boundary_index: int,
) -> dict[str, object]:
    """Return the exact local clock/span context measured at one edge."""

    choices = _choices(plan_or_choices)
    if (
        isinstance(boundary_index, bool)
        or not isinstance(boundary_index, int)
        or boundary_index < 0
    ):
        raise ValueError("splice feedback boundary_index is invalid")
    if boundary_index + 1 >= len(choices):
        raise ValueError("splice feedback boundary is outside the plan")
    return {
        "contract": SPLICE_EDGE_CONTEXT_CONTRACT,
        "source_clock_basis": _plan_source_clock_basis(plan_or_choices),
        "boundary_index": boundary_index,
        "outgoing": _choice_identity_payload(choices[boundary_index]),
        "incoming": _choice_identity_payload(choices[boundary_index + 1]),
    }


def splice_feedback_edge_context_identity(
    plan_or_choices: object,
    boundary_index: int,
) -> str:
    return _payload_identity(
        splice_feedback_edge_context(plan_or_choices, boundary_index)
    )


def splice_feedback_result_context(
    plan_or_choices: object,
    boundary_index: int,
) -> dict[str, object]:
    """Bind positive evidence only to the exact displayed cut instant.

    Boundary descriptors and temporal pose windows are sampled around the
    selected outgoing end/incoming start.  Adjacent edits may change the far
    end of either segment without changing those measured frames.  Minimum
    duration and visible scene-cut safety remain mandatory final timeline
    review gates, so they are not duplicated in this positive cache key.
    """

    choices = _choices(plan_or_choices)
    if (
        isinstance(boundary_index, bool)
        or not isinstance(boundary_index, int)
        or boundary_index < 0
    ):
        raise ValueError("splice result boundary_index is invalid")
    if boundary_index + 1 >= len(choices):
        raise ValueError("splice result boundary is outside the plan")
    outgoing = choices[boundary_index]
    incoming = choices[boundary_index + 1]
    outgoing_master = _exact_clock_seconds(
        outgoing,
        "canonical_master_end",
    )
    incoming_master = _exact_clock_seconds(
        incoming,
        "canonical_master_start",
    )
    if outgoing_master != incoming_master:
        raise ValueError("splice result choices do not share a master boundary")
    return {
        "contract": SPLICE_RESULT_CONTEXT_CONTRACT,
        "source_clock_basis": _plan_source_clock_basis(plan_or_choices),
        "boundary_index": boundary_index,
        "canonical_master_boundary_hex": outgoing_master,
        "outgoing": {
            "segment_index": _non_negative_choice_integer(
                outgoing,
                "segment_index",
            ),
            "source_index": _source_index(outgoing),
            "shot_id": _shot_id(outgoing),
            "source_local_boundary_hex": _exact_clock_seconds(
                outgoing,
                "source_local_end",
            ),
        },
        "incoming": {
            "segment_index": _non_negative_choice_integer(
                incoming,
                "segment_index",
            ),
            "source_index": _source_index(incoming),
            "shot_id": _shot_id(incoming),
            "source_local_boundary_hex": _exact_clock_seconds(
                incoming,
                "source_local_start",
            ),
        },
    }


def splice_feedback_result_context_identity(
    plan_or_choices: object,
    boundary_index: int,
) -> str:
    return _payload_identity(
        splice_feedback_result_context(plan_or_choices, boundary_index)
    )


def _aggregate_status(
    assessments: Sequence[SpliceFeasibilityAssessment],
) -> SpliceFeasibilityStatus:
    if not assessments:
        raise ValueError("splice feedback assessments have no status")
    merged = assessments[0]
    for assessment in assessments[1:]:
        merged = merge_splice_feasibility_assessments(merged, assessment)
    return merged.status


def build_splice_failure_feedback(
    plan_or_choices: object,
    failure_records: Sequence[Mapping[str, Any]],
    *,
    expected_frame_offsets: Sequence[int] = DEFAULT_WIDE_FRAME_OFFSETS,
) -> SpliceFailureFeedback:
    """Classify persisted repair failures against their exact optimizer plan."""

    choices = _choices(plan_or_choices)
    plan_analysis_basis = _plan_analysis_basis(plan_or_choices)
    plan_source_clock_basis = _plan_source_clock_basis(plan_or_choices)
    plan_identity = splice_feedback_plan_identity(plan_or_choices)
    if isinstance(failure_records, (str, bytes)) or not isinstance(
        failure_records, Sequence
    ):
        raise TypeError("failure_records must be a sequence")
    grouped: dict[
        tuple[TimelineNoGoodEdgeKey, str],
        list[SpliceFeasibilityAssessment],
    ] = {}
    for record in failure_records:
        if not isinstance(record, Mapping):
            raise TypeError("splice failure record must be a mapping")
        boundary_index = _integer(record, "boundary_index")
        if boundary_index + 1 >= len(choices):
            raise ValueError("splice failure boundary is outside the plan")
        outgoing = choices[boundary_index]
        incoming = choices[boundary_index + 1]
        outgoing_source = _integer(record, "outgoing_source_index")
        incoming_source = _integer(record, "incoming_source_index")
        if (
            outgoing_source != _source_index(outgoing)
            or incoming_source != _source_index(incoming)
        ):
            raise ValueError(
                "splice failure source direction does not match the plan"
            )
        if outgoing_source == incoming_source:
            raise ValueError("splice failure cannot describe a same-source edge")
        outgoing_shot_id = _exact_text(
            record.get("outgoing_shot_id"),
            "splice failure outgoing_shot_id",
        )
        incoming_shot_id = _exact_text(
            record.get("incoming_shot_id"),
            "splice failure incoming_shot_id",
        )
        if (
            outgoing_shot_id != _shot_id(outgoing)
            or incoming_shot_id != _shot_id(incoming)
        ):
            raise ValueError("splice failure shot edge does not match the plan")
        if _exact_text(
            record.get("timeline_analysis_basis"),
            "splice failure timeline_analysis_basis",
        ) != plan_analysis_basis:
            raise ValueError("splice failure analysis basis does not match the plan")
        if _exact_text(
            record.get("timeline_source_clock_basis"),
            "splice failure timeline_source_clock_basis",
        ) != plan_source_clock_basis:
            raise ValueError(
                "splice failure source clock basis does not match the plan"
            )
        if _exact_text(
            record.get("timeline_path_identity"),
            "splice failure timeline_path_identity",
        ) != plan_identity:
            raise ValueError("splice failure path identity does not match the plan")
        expected_edge_context = splice_feedback_edge_context(
            plan_or_choices,
            boundary_index,
        )
        recorded_edge_context = record.get("optimizer_edge_context")
        if not isinstance(recorded_edge_context, Mapping):
            raise TypeError("splice failure optimizer edge context is invalid")
        if dict(recorded_edge_context) != expected_edge_context:
            raise ValueError(
                "splice failure optimizer edge context does not match the plan"
            )
        expected_edge_context_identity = _payload_identity(
            expected_edge_context
        )
        if _exact_text(
            record.get("optimizer_edge_context_identity"),
            "splice failure optimizer_edge_context_identity",
        ) != expected_edge_context_identity:
            raise ValueError("splice failure optimizer edge identity is stale")
        observed_edge_context = record.get("observed_edge_context")
        if not isinstance(observed_edge_context, Mapping):
            raise TypeError("splice failure observed edge context is invalid")
        observed_edge_context_identity = _payload_identity(
            dict(observed_edge_context)
        )
        if _exact_text(
            record.get("observed_edge_context_identity"),
            "splice failure observed_edge_context_identity",
        ) != observed_edge_context_identity:
            raise ValueError("splice failure observed edge identity is stale")
        context_matches_optimizer = (
            dict(observed_edge_context) == expected_edge_context
        )
        recorded_match = record.get("optimizer_edge_context_match")
        if not isinstance(recorded_match, bool):
            raise TypeError("splice failure optimizer edge match is invalid")
        if recorded_match is not context_matches_optimizer:
            raise ValueError("splice failure optimizer edge match is contradictory")
        audit = record.get("candidate_evidence_audit")
        if not isinstance(audit, Sequence) or isinstance(audit, (str, bytes)):
            # An exception with no scalar evidence is an explicit unknown, not
            # permission to poison the optimizer graph.
            audit = ()
        if not context_matches_optimizer:
            # Evidence measured after an adjacent boundary repair belongs to
            # that transient path.  It may explain the rejection, but cannot
            # become a permanent constraint on the raw optimizer edge.
            audit = ()
        assessment = classify_splice_evidence_audit(
            tuple(audit),
            expected_frame_offsets=expected_frame_offsets,
        )
        edge = build_timeline_no_good_edge_key(
            boundary_index,
            outgoing,
            incoming,
        )
        grouped.setdefault(
            (edge, observed_edge_context_identity),
            [],
        ).append(assessment)
    return SpliceFailureFeedback(
        tuple(
            SpliceEdgeFeedback(
                edge=edge,
                edge_context_identity=edge_context_identity,
                status=_aggregate_status(assessments),
                assessments=tuple(assessments),
            )
            for (
                edge,
                edge_context_identity,
            ), assessments in grouped.items()
        )
    )


__all__ = [
    "SPLICE_FAILURE_FEEDBACK_CONTRACT",
    "SPLICE_EDGE_CONTEXT_CONTRACT",
    "SPLICE_RESULT_CONTEXT_CONTRACT",
    "SpliceEdgeFeedback",
    "SpliceFailureFeedback",
    "build_splice_failure_feedback",
    "splice_feedback_edge_context",
    "splice_feedback_edge_context_identity",
    "splice_feedback_context_identity",
    "splice_feedback_plan_context",
    "splice_feedback_plan_identity",
    "splice_feedback_result_context",
    "splice_feedback_result_context_identity",
]
