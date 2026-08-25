"""Conservative causal IVE center and active-performer sensing.

This module is a scalar-only sensing layer above :mod:`member_state_graph`
and :mod:`causal_shooting_grammar`.  It does not inspect media and it does not
claim to identify a singer.  For every exact shooting-grammar source/time key
it emits two probability distributions over all six :class:`IVEMember` values
plus an explicit unknown state:

* ``center_member`` uses exact-time member geometry, prominence, composition,
  headroom, and shot confidence; and
* ``active_performer`` needs per-observation mouth/body activity corroborated
  by another independent activity, composition, or optional vocal signal.

Known mass is deliberately capped and is never renormalized to one merely
because one member has the largest score.  Missing, ambiguous, occluded, or
non-vocal evidence therefore remains in ``unknown``.  Computation is
source-local and exact-time only, so appending future observations cannot
change an existing output record.

The persistent artifact is immutable and content addressed.  It stores only
normalized scalars, opaque observation identifiers, hashes, and provenance;
raw media, pixels, descriptors, and face/body embeddings are neither accepted
nor serialized.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .causal_shooting_grammar import CausalShootingGrammarRecord
from .ive import IVEMember
from .member_state_graph import (
    CausalMemberObservation,
    MemberProbabilityMass,
    MemberStateFeatureAdapter,
    MemberStateGraph,
)


IVE_PERFORMER_STATE_SCHEMA_VERSION = 1
IVE_PERFORMER_STATE_CONTRACT = "ive_performer_state_v1"
IVE_PERFORMER_STATE_ARTIFACT_KIND = "causal_ive_performer_state"
IVE_PERFORMER_STATE_PRODUCER = "crosscut.ive_performer_state_v1"
IVE_PERFORMER_STATE_ANALYSIS_BASIS = (
    "exact_source_timestamp_past_current_scalar_evidence_only_v1"
)
IVE_PERFORMER_STATE_CONTENT_HASH_BASIS = (
    "sha256_canonical_json_with_content_sha256_null_v1"
)
STAGE_EDIT_IVE_PERFORMER_STATE_ADAPTER = (
    "stage_edit_named_ive_performer_context_source_features_v1"
)

# These names and their order are the reviewed StageEdit hand-off contract.
# Every value has one same-position missing bit.  Explicit distribution
# unknown mass is data, not missingness; only the optional vocal input can be
# absent in this contract.
STAGE_EDIT_IVE_PERFORMER_STATE_FEATURE_NAMES = (
    "ive_center_member_probability_anyujin",
    "ive_center_member_probability_gaeul",
    "ive_center_member_probability_rei",
    "ive_center_member_probability_jangwonyoung",
    "ive_center_member_probability_liz",
    "ive_center_member_probability_leeseo",
    "ive_center_member_unknown_probability",
    "ive_active_performer_probability_anyujin",
    "ive_active_performer_probability_gaeul",
    "ive_active_performer_probability_rei",
    "ive_active_performer_probability_jangwonyoung",
    "ive_active_performer_probability_liz",
    "ive_active_performer_probability_leeseo",
    "ive_active_performer_unknown_probability",
    "ive_vocal_probability",
    "ive_center_member_known_mass",
    "ive_active_performer_known_mass",
)

MAX_CENTER_KNOWN_MASS = 0.95
MAX_ACTIVE_PERFORMER_KNOWN_MASS = 0.95

_MEMBERS = tuple(IVEMember)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROVENANCE_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,127}")
_MAXIMUM_ARTIFACT_BYTES = 32 * 1024 * 1024
_EPSILON = 1e-15
_FORBIDDEN_PERSISTED_KEY_PARTS = (
    "raw_frame",
    "frame_bytes",
    "image_bytes",
    "pixels",
    "visual_descriptor",
    "face_embedding",
    "body_embedding",
    "appearance_embedding",
    "identity_embedding",
)

_STORAGE_POLICY: Mapping[str, bool] = {
    "raw_frames_persisted": False,
    "compressed_frames_persisted": False,
    "pixels_persisted": False,
    "visual_descriptors_persisted": False,
    "identity_or_appearance_embeddings_persisted": False,
    "normalized_scalars_hashes_and_opaque_ids_only": True,
}


class IVEPerformerStateError(ValueError):
    """Base error for invalid performer-state input or derivation."""


class IVEPerformerStateArtifactError(IVEPerformerStateError):
    """Raised when a persistent performer-state artifact is unsafe."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IVEPerformerStateError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise IVEPerformerStateError(f"{name} must be finite")
    return numeric


def _probability(value: object, name: str) -> float:
    numeric = _finite(value, name)
    if not 0.0 <= numeric <= 1.0:
        raise IVEPerformerStateError(f"{name} must be normalized to [0, 1]")
    return numeric


def _optional_probability(value: object | None, name: str) -> float | None:
    return None if value is None else _probability(value, name)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise IVEPerformerStateError(f"{name} must be a string")
    if not value or value != value.strip() or len(value) > 256:
        raise IVEPerformerStateError(f"{name} must be a canonical identifier")
    if any(ord(character) < 32 for character in value):
        raise IVEPerformerStateError(f"{name} cannot contain control characters")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IVEPerformerStateError(f"{name} must be a lowercase SHA-256")
    return value


def _strict_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IVEPerformerStateArtifactError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise IVEPerformerStateArtifactError(f"{name} keys must be strings")
    return value


def _strict_keys(
    value: Mapping[str, Any], expected: Iterable[str], name: str
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        raise IVEPerformerStateArtifactError(
            f"{name} keys mismatch; "
            f"missing={sorted(expected_set - actual_set)!r}, "
            f"extra={sorted(actual_set - expected_set)!r}"
        )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IVEPerformerStateArtifactError(
                f"artifact JSON repeats key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise IVEPerformerStateArtifactError(
        f"artifact JSON contains non-finite constant {value!r}"
    )


def _reject_forbidden_persisted_keys(value: object, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise IVEPerformerStateArtifactError(
                    f"{path} contains a non-string key"
                )
            folded = raw_key.casefold()
            privacy_declaration = folded.endswith("_persisted") and isinstance(
                item, bool
            )
            if not privacy_declaration and any(
                part in folded for part in _FORBIDDEN_PERSISTED_KEY_PARTS
            ):
                raise IVEPerformerStateArtifactError(
                    f"{path}.{raw_key} is forbidden in performer-state artifacts"
                )
            _reject_forbidden_persisted_keys(item, f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_persisted_keys(item, f"{path}[{index}]")


def _member_mass_from_dict(value: object, name: str) -> MemberProbabilityMass:
    mapping = _strict_mapping(value, name)
    _strict_keys(mapping, ("members", "unknown_mass"), name)
    raw_members = mapping["members"]
    if not isinstance(raw_members, list) or len(raw_members) != len(_MEMBERS):
        raise IVEPerformerStateArtifactError(
            f"{name} must contain exactly six member entries"
        )
    serialized_order: list[object] = []
    for index, item in enumerate(raw_members):
        record = _strict_mapping(item, f"{name}.members[{index}]")
        _strict_keys(
            record,
            ("member", "probability"),
            f"{name}.members[{index}]",
        )
        serialized_order.append(record["member"])
    if tuple(serialized_order) != tuple(member.value for member in _MEMBERS):
        raise IVEPerformerStateArtifactError(
            f"{name} member order does not match the IVEMember contract"
        )
    try:
        return MemberProbabilityMass.from_dict(mapping)
    except (TypeError, ValueError) as exc:
        raise IVEPerformerStateArtifactError(f"{name} is invalid") from exc


def _unknown_distribution() -> MemberProbabilityMass:
    return MemberProbabilityMass.unknown()


def _distribution_from_scores(
    scores: Mapping[IVEMember, float],
    *,
    requested_known_mass: float,
    maximum_known_mass: float,
) -> MemberProbabilityMass:
    if tuple(scores) != _MEMBERS:
        raise IVEPerformerStateError(
            "distribution scores must explicitly follow all six IVEMember values"
        )
    normalized_scores: dict[IVEMember, float] = {}
    for member, raw_score in scores.items():
        score = _finite(raw_score, f"score for {member.value}")
        if score < 0.0:
            raise IVEPerformerStateError("distribution scores cannot be negative")
        normalized_scores[member] = score
    known_mass = min(
        _probability(requested_known_mass, "requested_known_mass"),
        _probability(maximum_known_mass, "maximum_known_mass"),
    )
    score_total = math.fsum(normalized_scores.values())
    if score_total <= _EPSILON or known_mass <= _EPSILON:
        return _unknown_distribution()
    probabilities = {
        member: known_mass * normalized_scores[member] / score_total
        for member in _MEMBERS
    }
    actual_known = math.fsum(probabilities.values())
    unknown = max(0.0, min(1.0, 1.0 - actual_known))
    return MemberProbabilityMass(probabilities, unknown_mass=unknown)


@dataclass(frozen=True, slots=True)
class ObservationActivityEvidence:
    """Scalar activity evidence for one graph observation at its exact time."""

    observation_id: str
    mouth_activity_probability: float | None = None
    body_activity_probability: float | None = None
    activity_confidence: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            _identifier(self.observation_id, "observation_id"),
        )
        mouth = _optional_probability(
            self.mouth_activity_probability,
            "mouth_activity_probability",
        )
        body = _optional_probability(
            self.body_activity_probability,
            "body_activity_probability",
        )
        if mouth is None and body is None:
            raise IVEPerformerStateError(
                "activity evidence needs mouth or body activity"
            )
        object.__setattr__(self, "mouth_activity_probability", mouth)
        object.__setattr__(self, "body_activity_probability", body)
        object.__setattr__(
            self,
            "activity_confidence",
            _probability(self.activity_confidence, "activity_confidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "mouth_activity_probability": self.mouth_activity_probability,
            "body_activity_probability": self.body_activity_probability,
            "activity_confidence": self.activity_confidence,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ObservationActivityEvidence":
        mapping = _strict_mapping(value, "observation activity evidence")
        _strict_keys(
            mapping,
            (
                "observation_id",
                "mouth_activity_probability",
                "body_activity_probability",
                "activity_confidence",
            ),
            "observation activity evidence",
        )
        try:
            return cls(**mapping)
        except (TypeError, IVEPerformerStateError) as exc:
            raise IVEPerformerStateArtifactError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class CausalPerformerEvidence:
    """Optional vocal context and exact-time per-observation activity.

    Records are snapshots, not intervals.  A vocal probability describes only
    whether the current source/time is vocal; it is never a member label.
    """

    source_id: str
    master_time_seconds: float
    vocal_probability: float | None = None
    observation_activity: tuple[ObservationActivityEvidence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        timestamp = _finite(self.master_time_seconds, "master_time_seconds")
        if timestamp < 0.0:
            raise IVEPerformerStateError(
                "master_time_seconds must be non-negative"
            )
        object.__setattr__(self, "master_time_seconds", timestamp)
        object.__setattr__(
            self,
            "vocal_probability",
            _optional_probability(self.vocal_probability, "vocal_probability"),
        )
        activities = tuple(self.observation_activity)
        if any(not isinstance(item, ObservationActivityEvidence) for item in activities):
            raise IVEPerformerStateError(
                "observation_activity must contain typed evidence"
            )
        identifiers = tuple(item.observation_id for item in activities)
        if identifiers != tuple(sorted(identifiers)) or len(set(identifiers)) != len(
            identifiers
        ):
            raise IVEPerformerStateError(
                "observation activity must be unique and observation-ID ordered"
            )
        object.__setattr__(self, "observation_activity", activities)

    @property
    def key(self) -> tuple[float, str]:
        return (self.master_time_seconds, self.source_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "master_time_seconds": self.master_time_seconds,
            "vocal_probability": self.vocal_probability,
            "observation_activity": [
                item.to_dict() for item in self.observation_activity
            ],
        }

    @classmethod
    def from_dict(cls, value: object) -> "CausalPerformerEvidence":
        mapping = _strict_mapping(value, "causal performer evidence")
        _strict_keys(
            mapping,
            (
                "source_id",
                "master_time_seconds",
                "vocal_probability",
                "observation_activity",
            ),
            "causal performer evidence",
        )
        raw_activity = mapping["observation_activity"]
        if not isinstance(raw_activity, list):
            raise IVEPerformerStateArtifactError(
                "observation_activity must be an array"
            )
        try:
            return cls(
                source_id=mapping["source_id"],
                master_time_seconds=mapping["master_time_seconds"],
                vocal_probability=mapping["vocal_probability"],
                observation_activity=tuple(
                    ObservationActivityEvidence.from_dict(item)
                    for item in raw_activity
                ),
            )
        except IVEPerformerStateError as exc:
            raise IVEPerformerStateArtifactError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class IVEPerformerStateSourceProvenance:
    """Content identity for one source, without a path or media payload."""

    source_id: str
    source_content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "source_content_sha256",
            _sha256(self.source_content_sha256, "source_content_sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "source_content_sha256": self.source_content_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "IVEPerformerStateSourceProvenance":
        mapping = _strict_mapping(value, "performer-state source provenance")
        _strict_keys(
            mapping,
            ("source_id", "source_content_sha256"),
            "performer-state source provenance",
        )
        try:
            return cls(**mapping)
        except (TypeError, IVEPerformerStateError) as exc:
            raise IVEPerformerStateArtifactError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class IVEPerformerStateProvenance:
    """Analyzer and exact source identities for a deterministic build."""

    analyzer_fingerprint_sha256: str
    sources: tuple[IVEPerformerStateSourceProvenance, ...]
    contracts: tuple[str, ...] = ()
    producer: str = IVE_PERFORMER_STATE_PRODUCER

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "analyzer_fingerprint_sha256",
            _sha256(
                self.analyzer_fingerprint_sha256,
                "analyzer_fingerprint_sha256",
            ),
        )
        sources = tuple(self.sources)
        if any(
            not isinstance(item, IVEPerformerStateSourceProvenance)
            for item in sources
        ):
            raise IVEPerformerStateError("sources must be typed provenance records")
        canonical_sources = tuple(sorted(sources, key=lambda item: item.source_id))
        if sources != canonical_sources or len(
            {item.source_id for item in sources}
        ) != len(sources):
            raise IVEPerformerStateError(
                "source provenance must be unique and source-ID ordered"
            )
        if not sources:
            raise IVEPerformerStateError("source provenance cannot be empty")
        object.__setattr__(self, "sources", sources)
        if any(not isinstance(item, str) for item in self.contracts):
            raise IVEPerformerStateError("provenance contracts must be strings")
        contracts = tuple(sorted(set(self.contracts)))
        if self.contracts != contracts or any(
            not item or _PROVENANCE_NAME.fullmatch(item) is None
            for item in contracts
        ):
            raise IVEPerformerStateError(
                "provenance contracts must be unique canonical names"
            )
        object.__setattr__(self, "contracts", contracts)
        if self.producer != IVE_PERFORMER_STATE_PRODUCER:
            raise IVEPerformerStateError("performer-state producer is unsupported")

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(item.source_id for item in self.sources)

    def validate_source_ids(self, source_ids: Sequence[str]) -> None:
        if tuple(source_ids) != self.source_ids:
            raise IVEPerformerStateError(
                "source provenance must exactly bind all performer-state sources"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "analyzer_fingerprint_sha256": self.analyzer_fingerprint_sha256,
            "contracts": list(self.contracts),
            "sources": [item.to_dict() for item in self.sources],
        }

    @classmethod
    def from_dict(cls, value: object) -> "IVEPerformerStateProvenance":
        mapping = _strict_mapping(value, "performer-state provenance")
        _strict_keys(
            mapping,
            ("producer", "analyzer_fingerprint_sha256", "contracts", "sources"),
            "performer-state provenance",
        )
        raw_contracts = mapping["contracts"]
        raw_sources = mapping["sources"]
        if not isinstance(raw_contracts, list) or not isinstance(raw_sources, list):
            raise IVEPerformerStateArtifactError(
                "performer-state provenance arrays are invalid"
            )
        try:
            return cls(
                producer=mapping["producer"],
                analyzer_fingerprint_sha256=mapping[
                    "analyzer_fingerprint_sha256"
                ],
                contracts=tuple(raw_contracts),
                sources=tuple(
                    IVEPerformerStateSourceProvenance.from_dict(item)
                    for item in raw_sources
                ),
            )
        except IVEPerformerStateError as exc:
            raise IVEPerformerStateArtifactError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class IVEPerformerStateInputBinding:
    """Canonical digests of every scalar input to the sensing layer."""

    member_state_graph_sha256: str
    causal_shooting_grammar_records_sha256: str
    causal_performer_evidence_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "member_state_graph_sha256",
            "causal_shooting_grammar_records_sha256",
            "causal_performer_evidence_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "member_state_graph_sha256": self.member_state_graph_sha256,
            "causal_shooting_grammar_records_sha256": (
                self.causal_shooting_grammar_records_sha256
            ),
            "causal_performer_evidence_sha256": (
                self.causal_performer_evidence_sha256
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> "IVEPerformerStateInputBinding":
        mapping = _strict_mapping(value, "performer-state input binding")
        _strict_keys(
            mapping,
            (
                "member_state_graph_sha256",
                "causal_shooting_grammar_records_sha256",
                "causal_performer_evidence_sha256",
            ),
            "performer-state input binding",
        )
        try:
            return cls(**mapping)
        except (TypeError, IVEPerformerStateError) as exc:
            raise IVEPerformerStateArtifactError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class IVEPerformerStateRecord:
    """Two normalized distributions for one exact source/master timestamp."""

    source_id: str
    master_time_seconds: float
    center_member: MemberProbabilityMass
    active_performer: MemberProbabilityMass
    vocal_probability: float | None
    center_member_known_mass: float
    active_performer_known_mass: float
    member_observation_ids: tuple[str, ...]
    activity_observation_ids: tuple[str, ...]
    corroborated_observation_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        timestamp = _finite(self.master_time_seconds, "master_time_seconds")
        if timestamp < 0.0:
            raise IVEPerformerStateError(
                "master_time_seconds must be non-negative"
            )
        object.__setattr__(self, "master_time_seconds", timestamp)
        if not isinstance(self.center_member, MemberProbabilityMass) or not isinstance(
            self.active_performer, MemberProbabilityMass
        ):
            raise IVEPerformerStateError(
                "center_member and active_performer must be normalized typed masses"
            )
        center_known = _probability(
            self.center_member_known_mass,
            "center_member_known_mass",
        )
        active_known = _probability(
            self.active_performer_known_mass,
            "active_performer_known_mass",
        )
        if not math.isclose(
            center_known,
            self.center_member.known_mass,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise IVEPerformerStateError(
                "center known mass must match center distribution unknown mass"
            )
        if not math.isclose(
            active_known,
            self.active_performer.known_mass,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise IVEPerformerStateError(
                "active known mass must match active distribution unknown mass"
            )
        if center_known > MAX_CENTER_KNOWN_MASS + 1e-12:
            raise IVEPerformerStateError("center known mass exceeds inference cap")
        if active_known > MAX_ACTIVE_PERFORMER_KNOWN_MASS + 1e-12:
            raise IVEPerformerStateError(
                "active performer known mass exceeds inference cap"
            )
        object.__setattr__(self, "center_member_known_mass", center_known)
        object.__setattr__(self, "active_performer_known_mass", active_known)
        vocal = _optional_probability(self.vocal_probability, "vocal_probability")
        object.__setattr__(self, "vocal_probability", vocal)
        member_ids = tuple(
            _identifier(item, "member_observation_id")
            for item in self.member_observation_ids
        )
        activity_ids = tuple(
            _identifier(item, "activity_observation_id")
            for item in self.activity_observation_ids
        )
        if member_ids != tuple(sorted(member_ids)) or len(set(member_ids)) != len(
            member_ids
        ):
            raise IVEPerformerStateError(
                "member observation IDs must be unique and canonical"
            )
        if activity_ids != tuple(sorted(activity_ids)) or len(
            set(activity_ids)
        ) != len(activity_ids):
            raise IVEPerformerStateError(
                "activity observation IDs must be unique and canonical"
            )
        if not set(activity_ids).issubset(member_ids):
            raise IVEPerformerStateError(
                "activity observations must be bound to exact-time graph observations"
            )
        object.__setattr__(self, "member_observation_ids", member_ids)
        object.__setattr__(self, "activity_observation_ids", activity_ids)
        count = self.corroborated_observation_count
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or count > len(activity_ids)
        ):
            raise IVEPerformerStateError(
                "corroborated_observation_count is invalid"
            )
        if not member_ids and center_known > 1e-12:
            raise IVEPerformerStateError(
                "missing member observations must leave center unknown"
            )
        if (not activity_ids or count == 0) and active_known > 1e-12:
            raise IVEPerformerStateError(
                "uncorroborated activity must leave active performer unknown"
            )
        if vocal == 0.0 and active_known > 1e-12:
            raise IVEPerformerStateError(
                "explicit non-vocal evidence must leave active performer unknown"
            )

    @property
    def key(self) -> tuple[float, str]:
        return (self.master_time_seconds, self.source_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "master_time_seconds": self.master_time_seconds,
            "center_member": self.center_member.to_dict(),
            "active_performer": self.active_performer.to_dict(),
            "vocal_probability": self.vocal_probability,
            "center_member_known_mass": self.center_member_known_mass,
            "active_performer_known_mass": self.active_performer_known_mass,
            "member_observation_ids": list(self.member_observation_ids),
            "activity_observation_ids": list(self.activity_observation_ids),
            "corroborated_observation_count": self.corroborated_observation_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "IVEPerformerStateRecord":
        mapping = _strict_mapping(value, "performer-state record")
        _strict_keys(
            mapping,
            (
                "source_id",
                "master_time_seconds",
                "center_member",
                "active_performer",
                "vocal_probability",
                "center_member_known_mass",
                "active_performer_known_mass",
                "member_observation_ids",
                "activity_observation_ids",
                "corroborated_observation_count",
            ),
            "performer-state record",
        )
        raw_member_ids = mapping["member_observation_ids"]
        raw_activity_ids = mapping["activity_observation_ids"]
        if not isinstance(raw_member_ids, list) or not isinstance(
            raw_activity_ids, list
        ):
            raise IVEPerformerStateArtifactError(
                "performer-state observation IDs must be arrays"
            )
        try:
            return cls(
                source_id=mapping["source_id"],
                master_time_seconds=mapping["master_time_seconds"],
                center_member=_member_mass_from_dict(
                    mapping["center_member"], "center_member"
                ),
                active_performer=_member_mass_from_dict(
                    mapping["active_performer"], "active_performer"
                ),
                vocal_probability=mapping["vocal_probability"],
                center_member_known_mass=mapping["center_member_known_mass"],
                active_performer_known_mass=mapping[
                    "active_performer_known_mass"
                ],
                member_observation_ids=tuple(raw_member_ids),
                activity_observation_ids=tuple(raw_activity_ids),
                corroborated_observation_count=mapping[
                    "corroborated_observation_count"
                ],
            )
        except IVEPerformerStateError as exc:
            raise IVEPerformerStateArtifactError(str(exc)) from exc


def _visible_geometry(observation: CausalMemberObservation) -> float:
    if observation.association_bbox is None:
        return 0.0
    return max(
        0.0,
        min(
            1.0,
            (1.0 - observation.occlusion_probability)
            * observation.geometry_confidence
            * observation.shot_confidence,
        ),
    )


def _center_closeness(center_x: float) -> float:
    return max(0.0, 1.0 - min(1.0, abs(center_x - 0.5) / 0.5))


def _headroom_quality(headroom: float | None) -> float:
    if headroom is None:
        return 0.0
    # A modest top margin is a useful center-composition cue.  This is a
    # quality weight, not a claim that one exact headroom is universally ideal.
    return max(0.0, 1.0 - min(1.0, abs(headroom - 0.18) / 0.82))


def _shot_scale_support(record: CausalShootingGrammarRecord) -> float:
    if record.shot_closeup_probability is None:
        return 0.0
    return max(
        0.0,
        min(
            1.0,
            record.shot_closeup_probability
            + 0.65 * (record.shot_medium_probability or 0.0)
            + 0.20 * (record.shot_wide_probability or 0.0),
        ),
    )


def _identity_known_quality(
    observations: Sequence[CausalMemberObservation],
    weights: Mapping[str, float] | None = None,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for observation in observations:
        weight = (
            _visible_geometry(observation)
            if weights is None
            else weights.get(observation.observation_id, 0.0)
        )
        numerator += weight * (1.0 - observation.member_probabilities.unknown_mass)
        denominator += weight
    return numerator / denominator if denominator > _EPSILON else 0.0


def _center_member_distribution(
    *,
    adapter: MemberStateFeatureAdapter,
    observations: Sequence[CausalMemberObservation],
    grammar: CausalShootingGrammarRecord,
) -> MemberProbabilityMass:
    if not observations or not any(item.has_geometry for item in observations):
        return _unknown_distribution()
    features = {
        member: adapter.at_timestamp(
            source_id=grammar.source_id,
            timestamp_seconds=grammar.master_time_seconds,
            target_member=member,
        )
        for member in _MEMBERS
    }
    scores: dict[IVEMember, float] = {}
    for member in _MEMBERS:
        feature = features[member]
        if feature.center_xy is None:
            scores[member] = 0.0
            continue
        centrality = _center_closeness(feature.center_xy[0])
        spatial = 0.65 * centrality + 0.35 * feature.prominence_probability
        certainty_weight = 0.5 + 0.5 * (1.0 - feature.uncertainty_probability)
        scores[member] = (
            feature.visibility_probability * spatial * certainty_weight
        )
    score_total = math.fsum(scores.values())
    if score_total <= _EPSILON:
        return _unknown_distribution()

    visible_observations = tuple(
        item for item in observations if _visible_geometry(item) > 0.0
    )
    identity_quality = _identity_known_quality(visible_observations)
    maximum_visibility = max(
        (feature.visibility_probability for feature in features.values()),
        default=0.0,
    )
    dominance = max(scores.values()) / score_total
    detected_count = max(
        grammar.detected_person_count or 0,
        len(visible_observations),
    )
    isolation = 1.0 / max(1, detected_count)
    best_observation_shot_confidence = max(
        (item.shot_confidence for item in visible_observations),
        default=0.0,
    )
    grammar_confidence = (
        grammar.shooting_grammar_confidence
        if grammar.shooting_grammar_confidence is not None
        else 0.5 * best_observation_shot_confidence
    )
    requested_known_mass = (
        identity_quality
        * maximum_visibility
        * (0.30 + 0.70 * _shot_scale_support(grammar))
        * (0.30 + 0.70 * isolation)
        * (0.50 + 0.50 * _headroom_quality(grammar.subject_headroom_normalized))
        * (0.50 + 0.50 * grammar_confidence)
        * (0.50 + 0.50 * dominance)
    )
    return _distribution_from_scores(
        scores,
        requested_known_mass=min(1.0, requested_known_mass),
        maximum_known_mass=MAX_CENTER_KNOWN_MASS,
    )


def _activity_composition_strengths(
    observations: Sequence[CausalMemberObservation],
) -> dict[str, float]:
    visible = {
        item.observation_id: _visible_geometry(item) for item in observations
    }
    total_visible_area = math.fsum(
        visible[item.observation_id] * item.association_bbox.area
        for item in observations
        if item.association_bbox is not None
    )
    strengths: dict[str, float] = {}
    for observation in observations:
        box = observation.association_bbox
        visibility = visible[observation.observation_id]
        if box is None or visibility <= 0.0:
            strengths[observation.observation_id] = 0.0
            continue
        prominence = (
            visibility * box.area / total_visible_area
            if total_visible_area > _EPSILON
            else 0.0
        )
        strengths[observation.observation_id] = visibility * (
            0.60 * _center_closeness(box.center[0]) + 0.40 * prominence
        )
    return strengths


def _corroborated_activity_support(
    *,
    evidence: ObservationActivityEvidence,
    composition_strength: float,
    visible_geometry: float,
    vocal_probability: float | None,
) -> float:
    mouth = evidence.mouth_activity_probability
    body = evidence.body_activity_probability
    pairs: list[float] = []
    if mouth is not None and body is not None:
        pairs.append(mouth * body)
    if mouth is not None:
        pairs.append(mouth * composition_strength)
        if vocal_probability is not None:
            pairs.append(mouth * vocal_probability)
    if body is not None:
        pairs.append(body * composition_strength)
        if vocal_probability is not None:
            pairs.append(body * vocal_probability)
    if not pairs:
        return 0.0
    support = max(pairs) * evidence.activity_confidence * visible_geometry
    # Explicitly non-vocal context is stronger than a coincidental movement;
    # absent vocal context is neutral rather than guessed.
    if vocal_probability is not None:
        support *= vocal_probability
    return max(0.0, min(1.0, support))


def _active_performer_distribution(
    *,
    observations: Sequence[CausalMemberObservation],
    evidence: CausalPerformerEvidence | None,
    grammar: CausalShootingGrammarRecord,
    center_member: MemberProbabilityMass,
) -> tuple[MemberProbabilityMass, int]:
    if evidence is None or not evidence.observation_activity:
        return (_unknown_distribution(), 0)
    if evidence.vocal_probability == 0.0:
        return (_unknown_distribution(), 0)
    observation_by_id = {item.observation_id: item for item in observations}
    composition = _activity_composition_strengths(observations)
    scores = {member: 0.0 for member in _MEMBERS}
    support_by_observation: dict[str, float] = {}
    for activity in evidence.observation_activity:
        observation = observation_by_id[activity.observation_id]
        support = _corroborated_activity_support(
            evidence=activity,
            composition_strength=composition[activity.observation_id],
            visible_geometry=_visible_geometry(observation),
            vocal_probability=evidence.vocal_probability,
        )
        support_by_observation[activity.observation_id] = support
        for member in _MEMBERS:
            scores[member] += support * observation.member_probabilities.probability_for(
                member
            )
    corroborated_count = sum(
        support > _EPSILON for support in support_by_observation.values()
    )
    if corroborated_count == 0 or math.fsum(scores.values()) <= _EPSILON:
        return (_unknown_distribution(), 0)
    identity_quality = _identity_known_quality(
        observations,
        support_by_observation,
    )
    maximum_support = max(support_by_observation.values(), default=0.0)
    best_observation_shot_confidence = max(
        (item.shot_confidence for item in observations),
        default=0.0,
    )
    grammar_confidence = (
        grammar.shooting_grammar_confidence
        if grammar.shooting_grammar_confidence is not None
        else 0.5 * best_observation_shot_confidence
    )
    requested_known_mass = (
        maximum_support
        * identity_quality
        * (0.40 + 0.60 * grammar_confidence)
        * (0.35 + 0.65 * center_member.known_mass)
    )
    return (
        _distribution_from_scores(
            scores,
            requested_known_mass=min(1.0, requested_known_mass),
            maximum_known_mass=MAX_ACTIVE_PERFORMER_KNOWN_MASS,
        ),
        corroborated_count,
    )


def _validate_grammar_records(
    records: Sequence[CausalShootingGrammarRecord],
) -> tuple[CausalShootingGrammarRecord, ...]:
    items = tuple(records)
    if not items:
        raise IVEPerformerStateError(
            "performer-state derivation needs shooting-grammar records"
        )
    if any(not isinstance(item, CausalShootingGrammarRecord) for item in items):
        raise IVEPerformerStateError(
            "shooting_grammar_records must contain causal typed records"
        )
    previous_key: tuple[float, str] | None = None
    for item in items:
        key = (item.master_time_seconds, item.source_id)
        if previous_key is not None and key <= previous_key:
            raise IVEPerformerStateError(
                "shooting-grammar records must be strict master-time/source ordered"
            )
        previous_key = key
    return items


def _validate_causal_evidence(
    evidence: Sequence[CausalPerformerEvidence],
) -> tuple[CausalPerformerEvidence, ...]:
    items = tuple(evidence)
    if any(not isinstance(item, CausalPerformerEvidence) for item in items):
        raise IVEPerformerStateError(
            "performer_evidence must contain causal typed records"
        )
    previous_key: tuple[float, str] | None = None
    for item in items:
        if previous_key is not None and item.key <= previous_key:
            raise IVEPerformerStateError(
                "performer evidence must be strict master-time/source ordered"
            )
        previous_key = item.key
    return items


def derive_ive_performer_state_records(
    graph: MemberStateGraph,
    shooting_grammar_records: Sequence[CausalShootingGrammarRecord],
    *,
    performer_evidence: Sequence[CausalPerformerEvidence] = (),
) -> tuple[IVEPerformerStateRecord, ...]:
    """Derive deterministic exact-time records without future look-ahead."""

    if not isinstance(graph, MemberStateGraph):
        raise IVEPerformerStateError("graph must be MemberStateGraph")
    grammar_records = _validate_grammar_records(shooting_grammar_records)
    evidence_records = _validate_causal_evidence(performer_evidence)
    grammar_keys = {
        (item.master_time_seconds, item.source_id) for item in grammar_records
    }
    grammar_source_ids = tuple(sorted({item.source_id for item in grammar_records}))
    if grammar_source_ids != graph.source_ids:
        raise IVEPerformerStateError(
            "member graph and shooting grammar must bind the exact same sources"
        )
    evidence_by_key: dict[tuple[float, str], CausalPerformerEvidence] = {}
    for item in evidence_records:
        if item.key not in grammar_keys:
            raise IVEPerformerStateError(
                "performer evidence must bind an exact shooting-grammar source/time"
            )
        evidence_by_key[item.key] = item

    observations_by_key: dict[tuple[float, str], tuple[CausalMemberObservation, ...]] = {}
    mutable_observations: dict[
        tuple[float, str], list[CausalMemberObservation]
    ] = {}
    for observation in graph.observations:
        mutable_observations.setdefault(
            (observation.timestamp_seconds, observation.source_id), []
        ).append(observation)
    for key, values in mutable_observations.items():
        observations_by_key[key] = tuple(
            sorted(values, key=lambda item: item.causal_order)
        )

    adapter = MemberStateFeatureAdapter(graph)
    output: list[IVEPerformerStateRecord] = []
    for grammar in grammar_records:
        key = (grammar.master_time_seconds, grammar.source_id)
        observations = observations_by_key.get(key, ())
        evidence = evidence_by_key.get(key)
        observation_by_id = {item.observation_id: item for item in observations}
        if evidence is not None:
            for activity in evidence.observation_activity:
                if activity.observation_id not in observation_by_id:
                    raise IVEPerformerStateError(
                        "activity evidence observation is not present at its exact "
                        "source/time"
                    )
        center = _center_member_distribution(
            adapter=adapter,
            observations=observations,
            grammar=grammar,
        )
        active, corroborated_count = _active_performer_distribution(
            observations=observations,
            evidence=evidence,
            grammar=grammar,
            center_member=center,
        )
        output.append(
            IVEPerformerStateRecord(
                source_id=grammar.source_id,
                master_time_seconds=grammar.master_time_seconds,
                center_member=center,
                active_performer=active,
                vocal_probability=(
                    evidence.vocal_probability if evidence is not None else None
                ),
                center_member_known_mass=center.known_mass,
                active_performer_known_mass=active.known_mass,
                member_observation_ids=tuple(sorted(observation_by_id)),
                activity_observation_ids=(
                    tuple(
                        item.observation_id for item in evidence.observation_activity
                    )
                    if evidence is not None
                    else ()
                ),
                corroborated_observation_count=corroborated_count,
            )
        )
    return tuple(output)


def build_ive_performer_state_input_binding(
    graph: MemberStateGraph,
    shooting_grammar_records: Sequence[CausalShootingGrammarRecord],
    *,
    performer_evidence: Sequence[CausalPerformerEvidence] = (),
) -> IVEPerformerStateInputBinding:
    """Return deterministic scalar-input identities used for replay binding."""

    if not isinstance(graph, MemberStateGraph):
        raise IVEPerformerStateError("graph must be MemberStateGraph")
    grammar = _validate_grammar_records(shooting_grammar_records)
    evidence = _validate_causal_evidence(performer_evidence)
    grammar_keys = {
        (item.master_time_seconds, item.source_id) for item in grammar
    }
    grammar_source_ids = tuple(sorted({item.source_id for item in grammar}))
    if grammar_source_ids != graph.source_ids:
        raise IVEPerformerStateError(
            "member graph and shooting grammar must bind the exact same sources"
        )
    observations_by_key = {
        (item.timestamp_seconds, item.source_id, item.observation_id)
        for item in graph.observations
    }
    for evidence_record in evidence:
        if evidence_record.key not in grammar_keys:
            raise IVEPerformerStateError(
                "performer evidence must bind an exact shooting-grammar source/time"
            )
        for activity in evidence_record.observation_activity:
            if (
                evidence_record.master_time_seconds,
                evidence_record.source_id,
                activity.observation_id,
            ) not in observations_by_key:
                raise IVEPerformerStateError(
                    "activity evidence observation is not present at its exact "
                    "source/time"
                )
    return IVEPerformerStateInputBinding(
        member_state_graph_sha256=_canonical_sha256(graph.to_dict()),
        causal_shooting_grammar_records_sha256=_canonical_sha256(
            [item.to_dict() for item in grammar]
        ),
        causal_performer_evidence_sha256=_canonical_sha256(
            [item.to_dict() for item in evidence]
        ),
    )


@dataclass(frozen=True, slots=True)
class StageEditNamedIVEPerformerStateFeatures:
    """Named StageEdit source/context columns with a 1:1 missing mask."""

    source_id: str
    master_time_seconds: float
    values: tuple[float, ...]
    missing_mask: tuple[bool, ...]
    feature_names: tuple[str, ...] = STAGE_EDIT_IVE_PERFORMER_STATE_FEATURE_NAMES
    adapter: str = STAGE_EDIT_IVE_PERFORMER_STATE_ADAPTER

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        timestamp = _finite(self.master_time_seconds, "master_time_seconds")
        if timestamp < 0.0:
            raise IVEPerformerStateError(
                "master_time_seconds must be non-negative"
            )
        object.__setattr__(self, "master_time_seconds", timestamp)
        if self.feature_names != STAGE_EDIT_IVE_PERFORMER_STATE_FEATURE_NAMES:
            raise IVEPerformerStateError(
                "performer-state StageEdit feature names changed"
            )
        if self.adapter != STAGE_EDIT_IVE_PERFORMER_STATE_ADAPTER:
            raise IVEPerformerStateError(
                "performer-state StageEdit adapter identity changed"
            )
        values = tuple(
            _finite(value, f"StageEdit feature {index}")
            for index, value in enumerate(self.values)
        )
        missing = tuple(self.missing_mask)
        if len(values) != len(self.feature_names) or len(missing) != len(
            self.feature_names
        ):
            raise IVEPerformerStateError(
                "StageEdit values and missing mask must match exact columns"
            )
        if any(not isinstance(value, bool) for value in missing):
            raise IVEPerformerStateError("StageEdit missing mask must be boolean")
        if any(
            value != 0.0
            for value, is_missing in zip(values, missing, strict=True)
            if is_missing
        ):
            raise IVEPerformerStateError(
                "missing StageEdit features must use a zero placeholder"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "missing_mask", missing)

    def named_values(self) -> dict[str, tuple[float, bool]]:
        return {
            name: (value, missing)
            for name, value, missing in zip(
                self.feature_names,
                self.values,
                self.missing_mask,
                strict=True,
            )
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "source_id": self.source_id,
            "master_time_seconds": self.master_time_seconds,
            "feature_names": list(self.feature_names),
            "values": list(self.values),
            "missing_mask": list(self.missing_mask),
        }


def adapt_ive_performer_state_for_stage_edit(
    record: IVEPerformerStateRecord,
) -> StageEditNamedIVEPerformerStateFeatures:
    """Adapt one record to the exact named source/context feature contract."""

    if not isinstance(record, IVEPerformerStateRecord):
        raise TypeError("StageEdit adapter needs IVEPerformerStateRecord")
    center_values = tuple(
        record.center_member.probability_for(member) for member in _MEMBERS
    ) + (record.center_member.unknown_mass,)
    active_values = tuple(
        record.active_performer.probability_for(member) for member in _MEMBERS
    ) + (record.active_performer.unknown_mass,)
    vocal_missing = record.vocal_probability is None
    values = center_values + active_values + (
        0.0 if vocal_missing else record.vocal_probability,
        record.center_member_known_mass,
        record.active_performer_known_mass,
    )
    missing = (False,) * 14 + (vocal_missing, False, False)
    return StageEditNamedIVEPerformerStateFeatures(
        source_id=record.source_id,
        master_time_seconds=record.master_time_seconds,
        values=values,
        missing_mask=missing,
    )


def _temporal_policy() -> dict[str, Any]:
    return {
        "context": "exact_source_timestamp_current_evidence_only",
        "future_member_observations_used": False,
        "future_shooting_grammar_records_used": False,
        "future_activity_or_vocal_evidence_used": False,
        "interpolation_used": False,
        "input_order": "strict_master_time_then_source_id",
        "source_binding": "exact_source_id_and_float_timestamp",
    }


def _computation_policy() -> dict[str, Any]:
    return {
        "center_distribution": (
            "member_geometry_prominence_headroom_shot_confidence_v1"
        ),
        "active_distribution": (
            "per_observation_activity_two_family_corroboration_v1"
        ),
        "vocal_semantics": "optional_context_probability_never_member_label",
        "explicit_nonvocal_semantics": "zero_forces_active_known_mass_zero",
        "insufficient_evidence_semantics": "retain_probability_mass_in_unknown",
        "hard_singer_label_emitted": False,
        "maximum_center_known_mass": MAX_CENTER_KNOWN_MASS,
        "maximum_active_performer_known_mass": MAX_ACTIVE_PERFORMER_KNOWN_MASS,
    }


def _feature_adapter_contract() -> dict[str, Any]:
    return {
        "name": STAGE_EDIT_IVE_PERFORMER_STATE_ADAPTER,
        "feature_names": list(STAGE_EDIT_IVE_PERFORMER_STATE_FEATURE_NAMES),
        "missing_mask_semantics": "true_means_unavailable",
        "missing_numeric_placeholder": 0.0,
        "distribution_unknown_is_explicit_not_missing": True,
        "axis": "source_context_at_exact_master_timestamp",
    }


def _artifact_unsigned_payload(
    *,
    records: Sequence[IVEPerformerStateRecord],
    provenance: IVEPerformerStateProvenance,
    input_binding: IVEPerformerStateInputBinding,
) -> dict[str, Any]:
    source_ids = tuple(sorted({record.source_id for record in records}))
    return {
        "contract": IVE_PERFORMER_STATE_CONTRACT,
        "schema_version": IVE_PERFORMER_STATE_SCHEMA_VERSION,
        "artifact_kind": IVE_PERFORMER_STATE_ARTIFACT_KIND,
        "analysis_basis": IVE_PERFORMER_STATE_ANALYSIS_BASIS,
        "content_hash_basis": IVE_PERFORMER_STATE_CONTENT_HASH_BASIS,
        "content_sha256": None,
        "temporal_policy": _temporal_policy(),
        "computation_policy": _computation_policy(),
        "feature_adapter": _feature_adapter_contract(),
        "storage_policy": dict(_STORAGE_POLICY),
        "source_ids": list(source_ids),
        "source_count": len(source_ids),
        "record_count": len(records),
        "input_binding": input_binding.to_dict(),
        "provenance": provenance.to_dict(),
        "records": [record.to_dict() for record in records],
    }


def _artifact_content_sha256(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized["content_sha256"] = None
    return _canonical_sha256(normalized)


@dataclass(frozen=True, slots=True)
class IVEPerformerStateArtifact:
    """Self-validating immutable performer-state artifact value."""

    records: tuple[IVEPerformerStateRecord, ...]
    provenance: IVEPerformerStateProvenance
    input_binding: IVEPerformerStateInputBinding
    content_sha256: str
    schema_version: int = IVE_PERFORMER_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records or any(
            not isinstance(item, IVEPerformerStateRecord) for item in records
        ):
            raise IVEPerformerStateArtifactError(
                "artifact needs typed performer-state records"
            )
        previous_key: tuple[float, str] | None = None
        for record in records:
            if previous_key is not None and record.key <= previous_key:
                raise IVEPerformerStateArtifactError(
                    "artifact records must be strict master-time/source ordered"
                )
            previous_key = record.key
        if not isinstance(self.provenance, IVEPerformerStateProvenance):
            raise IVEPerformerStateArtifactError("artifact provenance must be typed")
        if not isinstance(self.input_binding, IVEPerformerStateInputBinding):
            raise IVEPerformerStateArtifactError(
                "artifact input binding must be typed"
            )
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != IVE_PERFORMER_STATE_SCHEMA_VERSION
        ):
            raise IVEPerformerStateArtifactError(
                "unsupported performer-state schema version"
            )
        source_ids = tuple(sorted({record.source_id for record in records}))
        self.provenance.validate_source_ids(source_ids)
        object.__setattr__(self, "records", records)
        digest = _sha256(self.content_sha256, "content_sha256")
        unsigned = _artifact_unsigned_payload(
            records=records,
            provenance=self.provenance,
            input_binding=self.input_binding,
        )
        if not hmac.compare_digest(_artifact_content_sha256(unsigned), digest):
            raise IVEPerformerStateArtifactError(
                "performer-state artifact content hash mismatch"
            )
        object.__setattr__(self, "content_sha256", digest)

    @classmethod
    def build(
        cls,
        graph: MemberStateGraph,
        shooting_grammar_records: Sequence[CausalShootingGrammarRecord],
        *,
        provenance: IVEPerformerStateProvenance,
        performer_evidence: Sequence[CausalPerformerEvidence] = (),
    ) -> "IVEPerformerStateArtifact":
        grammar = _validate_grammar_records(shooting_grammar_records)
        evidence = _validate_causal_evidence(performer_evidence)
        records = derive_ive_performer_state_records(
            graph,
            grammar,
            performer_evidence=evidence,
        )
        if not isinstance(provenance, IVEPerformerStateProvenance):
            raise IVEPerformerStateError(
                "provenance must be IVEPerformerStateProvenance"
            )
        source_ids = tuple(sorted({record.source_id for record in records}))
        provenance.validate_source_ids(source_ids)
        binding = build_ive_performer_state_input_binding(
            graph,
            grammar,
            performer_evidence=evidence,
        )
        unsigned = _artifact_unsigned_payload(
            records=records,
            provenance=provenance,
            input_binding=binding,
        )
        return cls(
            records=records,
            provenance=provenance,
            input_binding=binding,
            content_sha256=_artifact_content_sha256(unsigned),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = _artifact_unsigned_payload(
            records=self.records,
            provenance=self.provenance,
            input_binding=self.input_binding,
        )
        digest = _artifact_content_sha256(payload)
        if not hmac.compare_digest(digest, self.content_sha256):
            raise IVEPerformerStateArtifactError(
                "performer-state artifact mutated after hashing"
            )
        payload["content_sha256"] = self.content_sha256
        _reject_forbidden_persisted_keys(payload)
        return payload

    @classmethod
    def from_dict(cls, value: object) -> "IVEPerformerStateArtifact":
        payload = _strict_mapping(value, "performer-state artifact")
        expected = (
            "contract",
            "schema_version",
            "artifact_kind",
            "analysis_basis",
            "content_hash_basis",
            "content_sha256",
            "temporal_policy",
            "computation_policy",
            "feature_adapter",
            "storage_policy",
            "source_ids",
            "source_count",
            "record_count",
            "input_binding",
            "provenance",
            "records",
        )
        _strict_keys(payload, expected, "performer-state artifact")
        _reject_forbidden_persisted_keys(payload)
        if payload["contract"] != IVE_PERFORMER_STATE_CONTRACT:
            raise IVEPerformerStateArtifactError("artifact contract mismatch")
        if payload["artifact_kind"] != IVE_PERFORMER_STATE_ARTIFACT_KIND:
            raise IVEPerformerStateArtifactError("artifact kind mismatch")
        if payload["analysis_basis"] != IVE_PERFORMER_STATE_ANALYSIS_BASIS:
            raise IVEPerformerStateArtifactError("artifact analysis basis mismatch")
        if (
            payload["content_hash_basis"]
            != IVE_PERFORMER_STATE_CONTENT_HASH_BASIS
        ):
            raise IVEPerformerStateArtifactError(
                "artifact content hash basis mismatch"
            )
        if _canonical_json_bytes(payload["temporal_policy"]) != (
            _canonical_json_bytes(_temporal_policy())
        ):
            raise IVEPerformerStateArtifactError("artifact temporal policy is unsafe")
        if _canonical_json_bytes(payload["computation_policy"]) != (
            _canonical_json_bytes(_computation_policy())
        ):
            raise IVEPerformerStateArtifactError(
                "artifact computation policy changed"
            )
        if _canonical_json_bytes(payload["feature_adapter"]) != (
            _canonical_json_bytes(_feature_adapter_contract())
        ):
            raise IVEPerformerStateArtifactError(
                "artifact feature adapter contract changed"
            )
        if _canonical_json_bytes(payload["storage_policy"]) != (
            _canonical_json_bytes(dict(_STORAGE_POLICY))
        ):
            raise IVEPerformerStateArtifactError("artifact storage policy is unsafe")
        if (
            isinstance(payload["schema_version"], bool)
            or not isinstance(payload["schema_version"], int)
            or payload["schema_version"] != IVE_PERFORMER_STATE_SCHEMA_VERSION
        ):
            raise IVEPerformerStateArtifactError("artifact schema version mismatch")
        raw_records = payload["records"]
        raw_source_ids = payload["source_ids"]
        if not isinstance(raw_records, list) or not isinstance(raw_source_ids, list):
            raise IVEPerformerStateArtifactError(
                "artifact records/source_ids must be arrays"
            )
        for name in ("source_count", "record_count"):
            count = payload[name]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise IVEPerformerStateArtifactError(
                    f"artifact {name} must be a non-negative integer"
                )
        records = tuple(IVEPerformerStateRecord.from_dict(item) for item in raw_records)
        artifact = cls(
            records=records,
            provenance=IVEPerformerStateProvenance.from_dict(payload["provenance"]),
            input_binding=IVEPerformerStateInputBinding.from_dict(
                payload["input_binding"]
            ),
            content_sha256=payload["content_sha256"],
            schema_version=payload["schema_version"],
        )
        expected_source_ids = sorted({record.source_id for record in records})
        if raw_source_ids != expected_source_ids:
            raise IVEPerformerStateArtifactError(
                "artifact source IDs are not canonical"
            )
        if payload["source_count"] != len(expected_source_ids) or payload[
            "record_count"
        ] != len(records):
            raise IVEPerformerStateArtifactError("artifact counts disagree")
        if _canonical_json_bytes(artifact.to_dict()) != _canonical_json_bytes(
            dict(payload)
        ):
            raise IVEPerformerStateArtifactError(
                "artifact serialized derived values disagree"
            )
        return artifact


def build_ive_performer_state_artifact(
    graph: MemberStateGraph,
    shooting_grammar_records: Sequence[CausalShootingGrammarRecord],
    *,
    provenance: IVEPerformerStateProvenance,
    performer_evidence: Sequence[CausalPerformerEvidence] = (),
) -> IVEPerformerStateArtifact:
    return IVEPerformerStateArtifact.build(
        graph,
        shooting_grammar_records,
        provenance=provenance,
        performer_evidence=performer_evidence,
    )


def _safe_directory(parent: Path, name: str) -> Path:
    child = parent / name
    if child.is_symlink():
        raise IVEPerformerStateArtifactError(
            f"artifact directory is a symlink: {child}"
        )
    child.mkdir(exist_ok=True)
    if not child.is_dir():
        raise IVEPerformerStateArtifactError(
            f"artifact path is not a directory: {child}"
        )
    return child


def write_ive_performer_state_artifact(
    artifact_directory: Path,
    artifact: IVEPerformerStateArtifact,
) -> Path:
    """Atomically publish an immutable content-addressed JSON artifact."""

    if not isinstance(artifact, IVEPerformerStateArtifact):
        raise TypeError("writer needs IVEPerformerStateArtifact")
    root = Path(artifact_directory).expanduser()
    if root.is_symlink():
        raise IVEPerformerStateArtifactError(
            "artifact root cannot be a symbolic link"
        )
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise IVEPerformerStateArtifactError("artifact root must be a directory")
    contract_directory = _safe_directory(root, IVE_PERFORMER_STATE_CONTRACT)
    shard = _safe_directory(contract_directory, artifact.content_sha256[:2])
    destination = shard / f"{artifact.content_sha256}.json"
    encoded = _canonical_json_bytes(artifact.to_dict())
    if not 1 <= len(encoded) <= _MAXIMUM_ARTIFACT_BYTES:
        raise IVEPerformerStateArtifactError("artifact size is invalid")
    if destination.is_symlink():
        raise IVEPerformerStateArtifactError(
            "artifact destination cannot be a symbolic link"
        )
    if destination.exists():
        existing = destination.read_bytes()
        if not hmac.compare_digest(existing, encoded):
            raise IVEPerformerStateArtifactError(
                "content-addressed destination contains different bytes"
            )
        return destination

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact.content_sha256}.",
        suffix=".tmp",
        dir=shard,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = destination.read_bytes()
            if not hmac.compare_digest(existing, encoded):
                raise IVEPerformerStateArtifactError(
                    "concurrent immutable artifact differs"
                )
        directory_fd = os.open(shard, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except IVEPerformerStateArtifactError:
        raise
    except OSError as exc:
        raise IVEPerformerStateArtifactError(
            "failed to publish performer-state artifact"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    verified = load_ive_performer_state_artifact(
        destination,
        expected_content_sha256=artifact.content_sha256,
        expected_input_binding=artifact.input_binding,
    )
    if verified.to_dict() != artifact.to_dict():
        raise IVEPerformerStateArtifactError(
            "published performer-state artifact verification failed"
        )
    return destination


def load_ive_performer_state_artifact(
    path: Path,
    *,
    expected_content_sha256: str | None = None,
    expected_input_binding: IVEPerformerStateInputBinding | None = None,
    expected_source_content_sha256: Mapping[str, str] | None = None,
) -> IVEPerformerStateArtifact:
    """Load, verify, and optionally reject replay against different inputs."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise IVEPerformerStateArtifactError(
            "content-addressed artifact path cannot be a symlink"
        )
    try:
        before = requested.stat()
    except OSError as exc:
        raise IVEPerformerStateArtifactError("artifact path is not readable") from exc
    if not stat.S_ISREG(before.st_mode):
        raise IVEPerformerStateArtifactError("artifact path is not a regular file")
    if not 1 <= before.st_size <= _MAXIMUM_ARTIFACT_BYTES:
        raise IVEPerformerStateArtifactError("artifact size is invalid")
    try:
        encoded = requested.read_bytes()
        after = requested.stat()
    except OSError as exc:
        raise IVEPerformerStateArtifactError("cannot read artifact") from exc
    stable_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    stable_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if stable_before != stable_after or len(encoded) != before.st_size:
        raise IVEPerformerStateArtifactError("artifact changed while being read")
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except IVEPerformerStateArtifactError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IVEPerformerStateArtifactError("cannot decode artifact JSON") from exc
    artifact = IVEPerformerStateArtifact.from_dict(payload)
    if requested.name != f"{artifact.content_sha256}.json":
        raise IVEPerformerStateArtifactError(
            "artifact filename is not content addressed"
        )
    if expected_content_sha256 is not None:
        expected = _sha256(
            expected_content_sha256,
            "expected_content_sha256",
        )
        if not hmac.compare_digest(expected, artifact.content_sha256):
            raise IVEPerformerStateArtifactError(
                "artifact does not match expected content SHA-256"
            )
    if expected_input_binding is not None:
        if not isinstance(expected_input_binding, IVEPerformerStateInputBinding):
            raise TypeError(
                "expected_input_binding must be IVEPerformerStateInputBinding"
            )
        if artifact.input_binding != expected_input_binding:
            raise IVEPerformerStateArtifactError(
                "artifact replay/input binding mismatch"
            )
    if expected_source_content_sha256 is not None:
        if not isinstance(expected_source_content_sha256, Mapping):
            raise TypeError("expected source content identities must be a mapping")
        expected_sources = {
            _identifier(source_id, "expected source_id"): _sha256(
                digest,
                "expected source_content_sha256",
            )
            for source_id, digest in expected_source_content_sha256.items()
        }
        actual_sources = {
            item.source_id: item.source_content_sha256
            for item in artifact.provenance.sources
        }
        if expected_sources != actual_sources:
            raise IVEPerformerStateArtifactError(
                "artifact source-content replay binding mismatch"
            )
    return artifact


__all__ = [
    "CausalPerformerEvidence",
    "IVE_PERFORMER_STATE_ANALYSIS_BASIS",
    "IVE_PERFORMER_STATE_ARTIFACT_KIND",
    "IVE_PERFORMER_STATE_CONTRACT",
    "IVE_PERFORMER_STATE_CONTENT_HASH_BASIS",
    "IVE_PERFORMER_STATE_PRODUCER",
    "IVE_PERFORMER_STATE_SCHEMA_VERSION",
    "IVEPerformerStateArtifact",
    "IVEPerformerStateArtifactError",
    "IVEPerformerStateError",
    "IVEPerformerStateInputBinding",
    "IVEPerformerStateProvenance",
    "IVEPerformerStateRecord",
    "IVEPerformerStateSourceProvenance",
    "MAX_ACTIVE_PERFORMER_KNOWN_MASS",
    "MAX_CENTER_KNOWN_MASS",
    "ObservationActivityEvidence",
    "STAGE_EDIT_IVE_PERFORMER_STATE_ADAPTER",
    "STAGE_EDIT_IVE_PERFORMER_STATE_FEATURE_NAMES",
    "StageEditNamedIVEPerformerStateFeatures",
    "adapt_ive_performer_state_for_stage_edit",
    "build_ive_performer_state_artifact",
    "build_ive_performer_state_input_binding",
    "derive_ive_performer_state_records",
    "load_ive_performer_state_artifact",
    "write_ive_performer_state_artifact",
]
