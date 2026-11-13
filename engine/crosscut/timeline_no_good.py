"""Exact, deterministic no-good keys for failed timeline boundary edges.

The optimizer can use these keys to avoid re-proposing an edge that already
failed fresh boundary evidence.  An edge is deliberately more specific than a
source transition: its identity includes the canonical boundary and both the
source and native-shot identities on each side.  Therefore evidence against
one shot pairing cannot accidentally ban another shot pairing from the same
two sources.

This module is intentionally independent of ``visual_score``.  The extraction
helpers use a small candidate-like protocol so both ``TimelineCandidate``
objects and materialized ``VisualTimelineChoice`` objects can be translated
without creating a circular import.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any


TIMELINE_NO_GOOD_CONTRACT = "exact_boundary_source_shot_edge_no_good_v1"


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _exact_identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    if value != value.strip():
        raise ValueError(f"{name} cannot have leading or trailing whitespace")
    return value


@dataclass(frozen=True, order=True, slots=True)
class TimelineNoGoodEdgeKey:
    """The exact source-and-shot edge rejected at one timeline boundary."""

    boundary_index: int
    outgoing_source_id: str
    outgoing_shot_id: str
    incoming_source_id: str
    incoming_shot_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "boundary_index",
            _non_negative_integer(self.boundary_index, "boundary_index"),
        )
        for name in (
            "outgoing_source_id",
            "outgoing_shot_id",
            "incoming_source_id",
            "incoming_shot_id",
        ):
            object.__setattr__(
                self,
                name,
                _exact_identifier(getattr(self, name), name),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_index": self.boundary_index,
            "outgoing_source_id": self.outgoing_source_id,
            "outgoing_shot_id": self.outgoing_shot_id,
            "incoming_source_id": self.incoming_source_id,
            "incoming_shot_id": self.incoming_shot_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TimelineNoGoodEdgeKey":
        if not isinstance(payload, Mapping):
            raise TypeError("timeline no-good edge payload must be a mapping")
        value = cls(
            boundary_index=payload.get("boundary_index"),
            outgoing_source_id=payload.get("outgoing_source_id"),
            outgoing_shot_id=payload.get("outgoing_shot_id"),
            incoming_source_id=payload.get("incoming_source_id"),
            incoming_shot_id=payload.get("incoming_shot_id"),
        )
        if dict(payload) != value.to_dict():
            raise ValueError("timeline no-good edge payload is not canonical")
        return value


def _candidate_payload(value: object) -> object:
    """Unwrap a ``TimelineChoice``-like object without concrete imports."""

    if isinstance(value, Mapping):
        if "source_id" in value or "source_index" in value:
            return value
        if "candidate" in value:
            return value["candidate"]
        return value
    if hasattr(value, "source_id") or hasattr(value, "source_index"):
        return value
    if hasattr(value, "candidate"):
        return getattr(value, "candidate")
    return value


def _candidate_field(value: object, name: str) -> tuple[bool, object]:
    if isinstance(value, Mapping):
        if name in value:
            return True, value[name]
        return False, None
    if hasattr(value, name):
        return True, getattr(value, name)
    return False, None


def _candidate_identity(value: object, side: str) -> tuple[str, str]:
    candidate = _candidate_payload(value)
    has_source_id, source_id_value = _candidate_field(candidate, "source_id")
    has_source_index, source_index_value = _candidate_field(
        candidate,
        "source_index",
    )
    has_shot_id, shot_id_value = _candidate_field(candidate, "shot_id")

    if not has_source_id and not has_source_index:
        raise TypeError(
            f"{side} candidate must expose source_id or source_index"
        )
    if not has_shot_id:
        raise TypeError(f"{side} candidate must expose shot_id")

    if has_source_id:
        source_id = _exact_identifier(
            source_id_value,
            f"{side}_source_id",
        )
    else:
        source_index = _non_negative_integer(
            source_index_value,
            f"{side}_source_index",
        )
        source_id = str(source_index)

    if has_source_index:
        source_index = _non_negative_integer(
            source_index_value,
            f"{side}_source_index",
        )
        if has_source_id and source_id != str(source_index):
            raise ValueError(
                f"{side} candidate source_id and source_index disagree"
            )

    shot_id = _exact_identifier(shot_id_value, f"{side}_shot_id")
    return source_id, shot_id


def build_timeline_no_good_edge_key(
    boundary_index: int,
    outgoing: object,
    incoming: object,
) -> TimelineNoGoodEdgeKey:
    """Build an exact edge key from candidate- or visual-choice-like values."""

    boundary_index = _non_negative_integer(boundary_index, "boundary_index")
    outgoing_source_id, outgoing_shot_id = _candidate_identity(
        outgoing,
        "outgoing",
    )
    incoming_source_id, incoming_shot_id = _candidate_identity(
        incoming,
        "incoming",
    )
    return TimelineNoGoodEdgeKey(
        boundary_index=boundary_index,
        outgoing_source_id=outgoing_source_id,
        outgoing_shot_id=outgoing_shot_id,
        incoming_source_id=incoming_source_id,
        incoming_shot_id=incoming_shot_id,
    )


def _plan_choices(plan_or_choices: object) -> tuple[object, ...]:
    if isinstance(plan_or_choices, Mapping):
        if "choices" not in plan_or_choices:
            raise TypeError("plan mapping must contain choices")
        choices = plan_or_choices["choices"]
    elif hasattr(plan_or_choices, "choices"):
        choices = getattr(plan_or_choices, "choices")
    else:
        choices = plan_or_choices

    if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence):
        raise TypeError("plan choices must be a sequence")
    return tuple(choices)


def extract_timeline_no_good_edge_keys(
    plan_or_choices: object,
    *,
    boundary_indices: Iterable[int] | None = None,
) -> tuple[TimelineNoGoodEdgeKey, ...]:
    """Extract canonical exact edges at selected plan boundaries.

    ``boundary_indices`` refers to the transition from ``choices[index]`` to
    ``choices[index + 1]``.  Omission extracts every adjacent edge.  Explicit
    duplicates are collapsed and the result is sorted by the edge key's stable
    field order.
    """

    choices = _plan_choices(plan_or_choices)
    if boundary_indices is None:
        indices = tuple(range(max(0, len(choices) - 1)))
    else:
        if isinstance(boundary_indices, (str, bytes)):
            raise TypeError("boundary_indices must be an iterable of integers")
        try:
            supplied_indices = tuple(boundary_indices)
        except TypeError as exc:
            raise TypeError(
                "boundary_indices must be an iterable of integers"
            ) from exc
        indices = tuple(
            sorted(
                {
                    _non_negative_integer(value, "boundary_index")
                    for value in supplied_indices
                }
            )
        )

    if any(index + 1 >= len(choices) for index in indices):
        raise ValueError("boundary_index is outside the plan choices")
    return tuple(
        sorted(
            {
                build_timeline_no_good_edge_key(
                    index,
                    choices[index],
                    choices[index + 1],
                )
                for index in indices
            }
        )
    )


@dataclass(frozen=True, slots=True)
class TimelineNoGoodSet:
    """A canonical, duplicate-free set with a checkpoint-safe fingerprint."""

    edges: tuple[TimelineNoGoodEdgeKey, ...] = ()
    _edge_index: frozenset[TimelineNoGoodEdgeKey] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if isinstance(self.edges, (str, bytes)):
            raise TypeError("edges must contain TimelineNoGoodEdgeKey values")
        try:
            edges = tuple(self.edges)
        except TypeError as exc:
            raise TypeError(
                "edges must contain TimelineNoGoodEdgeKey values"
            ) from exc
        if any(not isinstance(edge, TimelineNoGoodEdgeKey) for edge in edges):
            raise TypeError("edges must contain TimelineNoGoodEdgeKey values")
        canonical_edges = tuple(sorted(set(edges)))
        object.__setattr__(self, "edges", canonical_edges)
        object.__setattr__(self, "_edge_index", frozenset(canonical_edges))

    def __iter__(self) -> Iterator[TimelineNoGoodEdgeKey]:
        return iter(self.edges)

    def __len__(self) -> int:
        return len(self.edges)

    def __contains__(self, edge: object) -> bool:
        return isinstance(edge, TimelineNoGoodEdgeKey) and edge in self._edge_index

    @classmethod
    def from_plan_boundaries(
        cls,
        plan_or_choices: object,
        *,
        boundary_indices: Iterable[int] | None = None,
    ) -> "TimelineNoGoodSet":
        return cls(
            extract_timeline_no_good_edge_keys(
                plan_or_choices,
                boundary_indices=boundary_indices,
            )
        )

    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "contract": TIMELINE_NO_GOOD_CONTRACT,
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self._fingerprint_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._fingerprint_payload(),
            "edge_count": len(self.edges),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TimelineNoGoodSet":
        if not isinstance(payload, Mapping):
            raise TypeError("timeline no-good set payload must be a mapping")
        if payload.get("contract") != TIMELINE_NO_GOOD_CONTRACT:
            raise ValueError("timeline no-good set contract is invalid")
        raw_edges = payload.get("edges")
        if isinstance(raw_edges, (str, bytes)) or not isinstance(
            raw_edges, Sequence
        ):
            raise ValueError("timeline no-good set edges are invalid")
        value = cls(tuple(TimelineNoGoodEdgeKey.from_dict(item) for item in raw_edges))
        if dict(payload) != value.to_dict():
            raise ValueError("timeline no-good set payload is not canonical")
        return value
