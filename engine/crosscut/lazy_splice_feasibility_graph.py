"""Analysis-bound memoization for exact visual splice edge feasibility.

The graph contains no media access and performs no feasibility inference of
its own.  It binds already classified scalar evidence to one exact analysis
basis and one :class:`TimelineNoGoodEdgeKey`.  This keeps cache reuse safe:
changing the descriptor, threshold, or source-analysis basis requires a new
graph identity, while a conflicting result for an already measured exact edge
is rejected instead of silently replacing evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
import hashlib
import json
from threading import RLock, get_ident
from typing import Any

from .timeline_no_good import TimelineNoGoodEdgeKey, TimelineNoGoodSet
from .visual_splice_feasibility import (
    SpliceFeasibilityAssessment,
    SpliceFeasibilityStatus,
    merge_splice_feasibility_assessments,
)


LAZY_SPLICE_FEASIBILITY_GRAPH_CONTRACT = (
    "analysis_bound_exact_splice_feasibility_graph_v4"
)


def _exact_analysis_basis_identity(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("analysis_basis_identity must be a string")
    if not value or not value.strip():
        raise ValueError("analysis_basis_identity must be non-blank")
    if value != value.strip():
        raise ValueError(
            "analysis_basis_identity cannot have leading or trailing whitespace"
        )
    return value


def _validated_edge(value: object) -> TimelineNoGoodEdgeKey:
    if not isinstance(value, TimelineNoGoodEdgeKey):
        raise TypeError("edge must be a TimelineNoGoodEdgeKey")
    return value


def _validated_assessment(value: object) -> SpliceFeasibilityAssessment:
    if not isinstance(value, SpliceFeasibilityAssessment):
        raise TypeError("assessment must be a SpliceFeasibilityAssessment")
    return value


def _exact_edge_context_identity(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("edge_context_identity must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            "edge_context_identity must be a lowercase SHA-256 identity"
        )
    return value


@dataclass(frozen=True, order=True, slots=True)
class LazySpliceFeasibilityKey:
    """One source/shot edge under one exact clock/span context."""

    edge: TimelineNoGoodEdgeKey
    edge_context_identity: str

    def __post_init__(self) -> None:
        _validated_edge(self.edge)
        object.__setattr__(
            self,
            "edge_context_identity",
            _exact_edge_context_identity(self.edge_context_identity),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge": self.edge.to_dict(),
            "edge_context_identity": self.edge_context_identity,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "LazySpliceFeasibilityKey":
        if not isinstance(payload, Mapping):
            raise TypeError("splice feasibility key payload must be a mapping")
        value = cls(
            edge=TimelineNoGoodEdgeKey.from_dict(payload.get("edge")),
            edge_context_identity=payload.get("edge_context_identity"),
        )
        if dict(payload) != value.to_dict():
            raise ValueError("splice feasibility key payload is not canonical")
        return value


@dataclass(frozen=True, slots=True)
class LazySpliceFeasibilityEntry:
    """One exact edge assessment retained by the graph."""

    key: LazySpliceFeasibilityKey
    assessment: SpliceFeasibilityAssessment

    def __post_init__(self) -> None:
        if not isinstance(self.key, LazySpliceFeasibilityKey):
            raise TypeError("splice feasibility entry key is invalid")
        _validated_assessment(self.assessment)

    @property
    def edge(self) -> TimelineNoGoodEdgeKey:
        return self.key.edge

    @property
    def edge_context_identity(self) -> str:
        return self.key.edge_context_identity

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "assessment": self.assessment.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LazySpliceFeasibilityEntry":
        if not isinstance(payload, Mapping):
            raise TypeError("splice graph entry payload must be a mapping")
        value = cls(
            key=LazySpliceFeasibilityKey.from_dict(payload.get("key")),
            assessment=SpliceFeasibilityAssessment.from_dict(
                payload.get("assessment")
            ),
        )
        if dict(payload) != value.to_dict():
            raise ValueError("splice graph entry payload is not canonical")
        return value


@dataclass(frozen=True, slots=True)
class _InFlightAssessment:
    future: Future[SpliceFeasibilityAssessment]
    owner_thread_id: int


class LazySpliceFeasibilityGraph:
    """A deterministic, analysis-basis-bound cache of exact splice edges.

    ``get_or_assess`` serializes the cache miss check and assessment write, so
    one successful edge assessment is evaluated at most once per graph even
    when multiple callers race for it.  The assessor is intentionally the only
    injected side effect; this module itself remains independent of media and
    descriptor implementations.
    """

    def __init__(
        self,
        analysis_basis_identity: str,
        assessments: (
            Mapping[LazySpliceFeasibilityKey, SpliceFeasibilityAssessment]
            | Iterable[
                tuple[LazySpliceFeasibilityKey, SpliceFeasibilityAssessment]
            ]
        ) = (),
    ) -> None:
        self._analysis_basis_identity = _exact_analysis_basis_identity(
            analysis_basis_identity
        )
        self._assessments: dict[
            LazySpliceFeasibilityKey,
            SpliceFeasibilityAssessment,
        ] = {}
        self._lock = RLock()
        self._inflight: dict[LazySpliceFeasibilityKey, _InFlightAssessment] = {}

        if isinstance(assessments, Mapping):
            initial = tuple(assessments.items())
        else:
            if isinstance(assessments, (str, bytes)):
                raise TypeError("assessments must contain edge-assessment pairs")
            try:
                initial = tuple(assessments)
            except TypeError as exc:
                raise TypeError(
                    "assessments must contain edge-assessment pairs"
                ) from exc
        for item in initial:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
            ):
                raise TypeError(
                    "assessments must contain edge-assessment pairs"
                )
            self.record_assessment(
                item[0].edge,
                item[0].edge_context_identity,
                item[1],
            )

    @property
    def analysis_basis_identity(self) -> str:
        return self._analysis_basis_identity

    def __len__(self) -> int:
        with self._lock:
            return len(self._assessments)

    def __contains__(self, key: object) -> bool:
        return bool(
            isinstance(key, LazySpliceFeasibilityKey)
            and self.assessment_for(
                key.edge,
                key.edge_context_identity,
            )
            is not None
        )

    def assessment_for(
        self,
        edge: TimelineNoGoodEdgeKey,
        edge_context_identity: str,
    ) -> SpliceFeasibilityAssessment | None:
        key = LazySpliceFeasibilityKey(
            _validated_edge(edge),
            edge_context_identity,
        )
        with self._lock:
            return self._assessments.get(key)

    def record_assessment(
        self,
        edge: TimelineNoGoodEdgeKey,
        edge_context_identity: str,
        assessment: SpliceFeasibilityAssessment,
    ) -> SpliceFeasibilityAssessment:
        """Record one result idempotently and reject contradictory evidence."""

        key = LazySpliceFeasibilityKey(
            _validated_edge(edge),
            edge_context_identity,
        )
        exact_assessment = _validated_assessment(assessment)
        with self._lock:
            existing = self._assessments.get(key)
            if existing is not None:
                if existing == exact_assessment:
                    return existing
                merged = merge_splice_feasibility_assessments(
                    existing,
                    exact_assessment,
                )
                self._assessments[key] = merged
                return merged
            self._assessments[key] = exact_assessment
            return exact_assessment

    def needs_assessment(
        self,
        edge: TimelineNoGoodEdgeKey,
        edge_context_identity: str,
    ) -> bool:
        assessment = self.assessment_for(edge, edge_context_identity)
        return bool(
            assessment is None or not assessment.terminal_for_expected_domain
        )

    def get_or_assess(
        self,
        edge: TimelineNoGoodEdgeKey,
        edge_context_identity: str,
        assessor: Callable[
            [LazySpliceFeasibilityKey],
            SpliceFeasibilityAssessment,
        ],
    ) -> SpliceFeasibilityAssessment:
        """Return cached evidence or invoke ``assessor`` exactly once on a hit."""

        key = LazySpliceFeasibilityKey(
            _validated_edge(edge),
            edge_context_identity,
        )
        if not callable(assessor):
            raise TypeError("assessor must be callable")
        with self._lock:
            existing = self._assessments.get(key)
            if existing is not None and existing.terminal_for_expected_domain:
                return existing
            inflight = self._inflight.get(key)
            owner = inflight is None
            if owner:
                inflight = _InFlightAssessment(Future(), get_ident())
                self._inflight[key] = inflight
            elif inflight.owner_thread_id == get_ident():
                raise RuntimeError(
                    "recursive same-edge splice assessment is not allowed"
                )
        assert inflight is not None
        pending = inflight.future
        if not owner:
            return pending.result()

        try:
            assessed = _validated_assessment(assessor(key))
            with self._lock:
                before = self._assessments.get(key)
            stored = self.record_assessment(
                key.edge,
                key.edge_context_identity,
                assessed,
            )
            if (
                before is not None
                and stored == before
                and not stored.terminal_for_expected_domain
            ):
                raise ValueError(
                    "splice feasibility refinement added no new evidence"
                )
        except BaseException as exc:
            pending.set_exception(exc)
            with self._lock:
                if self._inflight.get(key) is inflight:
                    del self._inflight[key]
            raise
        pending.set_result(stored)
        with self._lock:
            if self._inflight.get(key) is inflight:
                del self._inflight[key]
        return stored

    @property
    def entries(self) -> tuple[LazySpliceFeasibilityEntry, ...]:
        with self._lock:
            return tuple(
                LazySpliceFeasibilityEntry(key=key, assessment=assessment)
                for key, assessment in sorted(self._assessments.items())
            )

    def entries_with_status(
        self,
        status: SpliceFeasibilityStatus,
    ) -> tuple[LazySpliceFeasibilityEntry, ...]:
        if not isinstance(status, SpliceFeasibilityStatus):
            raise TypeError("status must be a SpliceFeasibilityStatus")
        return tuple(
            entry for entry in self.entries if entry.assessment.status is status
        )

    def edges_with_status(
        self,
        status: SpliceFeasibilityStatus,
    ) -> tuple[TimelineNoGoodEdgeKey, ...]:
        return tuple(
            sorted(
                {
                    entry.edge
                    for entry in self.entries_with_status(status)
                }
            )
        )

    def keys_with_status(
        self,
        status: SpliceFeasibilityStatus,
    ) -> tuple[LazySpliceFeasibilityKey, ...]:
        return tuple(
            entry.key for entry in self.entries_with_status(status)
        )

    @property
    def raw_pass_edges(self) -> tuple[TimelineNoGoodEdgeKey, ...]:
        return self.edges_with_status(SpliceFeasibilityStatus.RAW_PASS)

    @property
    def geometry_proof_required_edges(
        self,
    ) -> tuple[TimelineNoGoodEdgeKey, ...]:
        return self.edges_with_status(
            SpliceFeasibilityStatus.RAW_GEOMETRY_PROOF_REQUIRED
        )

    @property
    def indeterminate_edges(self) -> tuple[TimelineNoGoodEdgeKey, ...]:
        return self.edges_with_status(SpliceFeasibilityStatus.INDETERMINATE)

    @property
    def terminal_indeterminate_edges(self) -> tuple[TimelineNoGoodEdgeKey, ...]:
        return tuple(
            sorted(
                {
                    entry.edge
                    for entry in self.entries_with_status(
                        SpliceFeasibilityStatus.INDETERMINATE
                    )
                    if entry.assessment.terminal_for_expected_domain
                }
            )
        )

    @property
    def incomplete_edges(self) -> tuple[TimelineNoGoodEdgeKey, ...]:
        return tuple(
            sorted(
                {
                    entry.edge
                    for entry in self.entries
                    if not entry.assessment.terminal_for_expected_domain
                }
            )
        )

    @property
    def hard_incompatible_edges(self) -> tuple[TimelineNoGoodEdgeKey, ...]:
        return self.edges_with_status(SpliceFeasibilityStatus.HARD_INCOMPATIBLE)

    def hard_no_goods_for_contexts(
        self,
        keys: Iterable[LazySpliceFeasibilityKey],
    ) -> TimelineNoGoodSet:
        """Project hard failures only for caller-verified raw contexts."""

        requested = tuple(keys)
        if any(
            not isinstance(key, LazySpliceFeasibilityKey)
            for key in requested
        ):
            raise TypeError("hard splice contexts must be exact graph keys")
        hard = set(
            self.keys_with_status(
                SpliceFeasibilityStatus.HARD_INCOMPATIBLE
            )
        )
        return TimelineNoGoodSet(
            tuple(key.edge for key in requested if key in hard)
        )

    @property
    def run_local_blocked_keys(self) -> tuple[LazySpliceFeasibilityKey, ...]:
        """Return exact contexts unusable for production under this basis.

        Indeterminate evidence is not a permanent no-good, but it still cannot
        authorize a production edit.  Keeping this set distinct prevents a
        missing descriptor from poisoning future analyses while allowing the
        current basis-bound optimizer to stop proposing an edge that will
        necessarily fail closed again.  Focus/scale-only edges remain allowed
        because final output-domain crop proof may resolve them.
        """

        return tuple(
            entry.key
            for entry in self.entries
            if (
                entry.assessment.status
                is SpliceFeasibilityStatus.HARD_INCOMPATIBLE
                or (
                    entry.assessment.status
                    is SpliceFeasibilityStatus.INDETERMINATE
                    and entry.assessment.terminal_for_expected_domain
                )
            )
        )

    def blocked_set_for_contexts(
        self,
        keys: Iterable[LazySpliceFeasibilityKey],
    ) -> TimelineNoGoodSet:
        """Project only explicitly supplied exact contexts to raw no-goods."""

        requested = tuple(keys)
        if any(
            not isinstance(key, LazySpliceFeasibilityKey)
            for key in requested
        ):
            raise TypeError("blocked splice contexts must be exact graph keys")
        blocked = set(self.run_local_blocked_keys)
        return TimelineNoGoodSet(
            tuple(key.edge for key in requested if key in blocked)
        )

    def for_analysis_basis(
        self,
        analysis_basis_identity: str,
    ) -> "LazySpliceFeasibilityGraph":
        """Reuse this graph only for the identical basis; otherwise start empty."""

        exact_identity = _exact_analysis_basis_identity(
            analysis_basis_identity
        )
        if exact_identity == self.analysis_basis_identity:
            return self
        return type(self)(exact_identity)

    def for_new_run(self) -> "LazySpliceFeasibilityGraph":
        """Reuse stable evidence while reopening run-local unknowns.

        Descriptor absence, decode interruption, and other indeterminate
        results are deliberately not cross-run authority.  Their serialized
        entries remain useful audit evidence, but a healthy later run must be
        allowed to measure the exact context again.
        """

        reusable = tuple(
            (entry.key, entry.assessment)
            for entry in self.entries
            if entry.assessment.status
            is not SpliceFeasibilityStatus.INDETERMINATE
        )
        return type(self)(self.analysis_basis_identity, reusable)

    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "contract": LAZY_SPLICE_FEASIBILITY_GRAPH_CONTRACT,
            "analysis_basis_identity": self.analysis_basis_identity,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def fingerprint(self) -> str:
        with self._lock:
            encoded = json.dumps(
                self._fingerprint_payload(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            status_edges = {
                status.value: [
                    key.to_dict() for key in self.keys_with_status(status)
                ]
                for status in SpliceFeasibilityStatus
            }
            return {
                **self._fingerprint_payload(),
                "assessment_count": len(self),
                "status_contexts": status_edges,
                "hard_incompatible_contexts": [
                    key.to_dict()
                    for key in self.keys_with_status(
                        SpliceFeasibilityStatus.HARD_INCOMPATIBLE
                    )
                ],
                "run_local_blocked_contexts": [
                    key.to_dict() for key in self.run_local_blocked_keys
                ],
                "fingerprint": self.fingerprint,
            }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LazySpliceFeasibilityGraph":
        if not isinstance(payload, Mapping):
            raise TypeError("splice feasibility graph payload must be a mapping")
        if payload.get("contract") != LAZY_SPLICE_FEASIBILITY_GRAPH_CONTRACT:
            raise ValueError("splice feasibility graph contract is invalid")
        raw_entries = payload.get("entries")
        if isinstance(raw_entries, (str, bytes)) or not isinstance(
            raw_entries, list
        ):
            raise ValueError("splice feasibility graph entries are invalid")
        entries = tuple(LazySpliceFeasibilityEntry.from_dict(item) for item in raw_entries)
        value = cls(
            _exact_analysis_basis_identity(payload.get("analysis_basis_identity")),
            tuple((entry.key, entry.assessment) for entry in entries),
        )
        if dict(payload) != value.to_dict():
            raise ValueError("splice feasibility graph payload is not canonical")
        return value


__all__ = [
    "LAZY_SPLICE_FEASIBILITY_GRAPH_CONTRACT",
    "LazySpliceFeasibilityEntry",
    "LazySpliceFeasibilityGraph",
    "LazySpliceFeasibilityKey",
]
