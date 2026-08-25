"""Typed, fail-closed IVE member-state graph primitives.

This module does not inspect pixels and does not claim to solve automatic
identity extraction.  It accepts verified or caller-supplied observations,
keeps probability mass for every IVE member plus an explicit unknown state,
and builds only conservative topology:

* source-local tracklets use deterministic temporal/geometry association;
* recovery across missing geometry is bounded and needs a continuing local
  tracker or pose token;
* cross-tracklet and cross-source identity links need an explicit claim with
  corroborating evidence from at least two modalities; and
* no association copies or smooths member probability into another
  observation.

The persistent form is immutable and content addressed.  It stores normalized
geometry, scalar confidence, opaque evidence digests, and provenance, but never
raw frames, image bytes, face embeddings, or body appearance embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import uuid

from .ive import IVEMember


MEMBER_STATE_GRAPH_CONTRACT = "ive_member_state_graph_v1"
MEMBER_STATE_GRAPH_SCHEMA_VERSION = 1
OBSERVATION_BASIS = (
    "verified_or_caller_supplied_observations_no_raw_media_inference_v1"
)
_CONTENT_HASH_BASIS = "sha256_canonical_json_with_content_sha256_null_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAXIMUM_ARTIFACT_BYTES = 32 * 1024 * 1024
_MEMBERS = tuple(IVEMember)
_FORBIDDEN_PERSISTED_KEY_PARTS = (
    "raw_frame",
    "frame_bytes",
    "image_bytes",
    "pixels",
    "sface_embedding",
    "face_embedding",
    "appearance_embedding",
    "body_embedding",
)


class MemberStateGraphError(ValueError):
    """Base error for invalid state-graph input or artifacts."""


class MemberStateArtifactError(MemberStateGraphError):
    """Raised when a persistent graph artifact is invalid or mutable."""


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MemberStateGraphError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise MemberStateGraphError(f"{name} must be finite")
    return numeric


def _probability(value: object, name: str) -> float:
    numeric = _finite(value, name)
    if not 0.0 <= numeric <= 1.0:
        raise MemberStateGraphError(f"{name} must be normalized to [0, 1]")
    return numeric


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MemberStateGraphError(f"{name} must be a non-negative integer")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise MemberStateGraphError(f"{name} must be a string")
    if not value or value != value.strip() or len(value) > 256:
        raise MemberStateGraphError(f"{name} must be a non-empty canonical string")
    if any(ord(character) < 32 for character in value):
        raise MemberStateGraphError(f"{name} cannot contain control characters")
    return value


def _optional_identifier(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, name)


def _ive_member(value: object, name: str) -> IVEMember:
    if isinstance(value, IVEMember):
        return value
    if not isinstance(value, str):
        raise MemberStateGraphError(f"{name} must be an IVE member string")
    try:
        return IVEMember.parse(value)
    except ValueError as exc:
        raise MemberStateGraphError(f"{name} is not an IVE member") from exc


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MemberStateArtifactError(f"artifact JSON repeats key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise MemberStateArtifactError(
        f"artifact JSON contains non-finite constant {value!r}"
    )


def _content_sha256(value: Mapping[str, Any]) -> str:
    normalized = dict(value)
    normalized["content_sha256"] = None
    return hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()


def _strict_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MemberStateArtifactError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise MemberStateArtifactError(f"{name} keys must be strings")
    return value


def _strict_keys(
    value: Mapping[str, Any], expected: Iterable[str], name: str
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise MemberStateArtifactError(
            f"{name} keys mismatch; missing={missing!r}, extra={extra!r}"
        )


def _reject_forbidden_persisted_keys(value: object, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise MemberStateArtifactError(f"{path} contains a non-string key")
            folded = raw_key.casefold()
            privacy_declaration = folded.endswith("_persisted") and isinstance(
                item, bool
            )
            if not privacy_declaration and any(
                part in folded for part in _FORBIDDEN_PERSISTED_KEY_PARTS
            ):
                raise MemberStateArtifactError(
                    f"{path}.{raw_key} is forbidden in member-state artifacts"
                )
            _reject_forbidden_persisted_keys(item, f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_persisted_keys(item, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class NormalizedBBox:
    """Normalized ``(left, top, right, bottom)`` geometry."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = tuple(
            _finite(value, "bbox coordinate")
            for value in (self.left, self.top, self.right, self.bottom)
        )
        if not (
            0.0 <= values[0] < values[2] <= 1.0
            and 0.0 <= values[1] < values[3] <= 1.0
        ):
            raise MemberStateGraphError(
                "bbox must satisfy 0 <= left < right <= 1 and "
                "0 <= top < bottom <= 1"
            )
        for field_name, value in zip(
            ("left", "top", "right", "bottom"), values, strict=True
        ):
            object.__setattr__(self, field_name, value)

    @classmethod
    def from_xyxy(cls, value: Sequence[float]) -> "NormalizedBBox":
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 4
        ):
            raise MemberStateGraphError("bbox must contain four coordinates")
        return cls(*value)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)

    @property
    def area(self) -> float:
        return (self.right - self.left) * (self.bottom - self.top)

    def intersection_over_union(self, other: "NormalizedBBox") -> float:
        width = max(0.0, min(self.right, other.right) - max(self.left, other.left))
        height = max(0.0, min(self.bottom, other.bottom) - max(self.top, other.top))
        intersection = width * height
        union = self.area + other.area - intersection
        return intersection / union if union > 0.0 else 0.0

    def to_list(self) -> list[float]:
        return [self.left, self.top, self.right, self.bottom]


@dataclass(frozen=True, slots=True, init=False)
class MemberProbabilityMass:
    """Probability for all six IVE members and one explicit unknown state."""

    _probabilities: tuple[float, ...]
    unknown_mass: float

    def __init__(
        self,
        member_probabilities: Mapping[IVEMember | str, float],
        *,
        unknown_mass: float,
    ) -> None:
        if not isinstance(member_probabilities, Mapping):
            raise MemberStateGraphError("member_probabilities must be a mapping")
        parsed: dict[IVEMember, float] = {}
        for raw_member, raw_probability in member_probabilities.items():
            try:
                member = _ive_member(raw_member, "member probability key")
            except MemberStateGraphError as exc:
                raise MemberStateGraphError(
                    f"invalid IVE probability key {raw_member!r}"
                ) from exc
            if member in parsed:
                raise MemberStateGraphError(
                    f"duplicate IVE member probability for {member.value}"
                )
            parsed[member] = _probability(
                raw_probability, f"probability for {member.value}"
            )
        missing = tuple(member.value for member in _MEMBERS if member not in parsed)
        extra_count = len(parsed) - len(_MEMBERS)
        if missing or extra_count:
            raise MemberStateGraphError(
                "member_probabilities must explicitly contain all six IVE "
                f"members; missing={missing!r}"
            )
        unknown = _probability(unknown_mass, "unknown_mass")
        probabilities = tuple(parsed[member] for member in _MEMBERS)
        total = math.fsum((*probabilities, unknown))
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise MemberStateGraphError(
                "IVE member probabilities plus unknown_mass must sum to 1"
            )
        object.__setattr__(self, "_probabilities", probabilities)
        object.__setattr__(self, "unknown_mass", unknown)

    @classmethod
    def unknown(cls) -> "MemberProbabilityMass":
        return cls({member: 0.0 for member in _MEMBERS}, unknown_mass=1.0)

    @classmethod
    def certain(cls, member: IVEMember | str) -> "MemberProbabilityMass":
        target = _ive_member(member, "certain member")
        return cls(
            {candidate: float(candidate is target) for candidate in _MEMBERS},
            unknown_mass=0.0,
        )

    @property
    def member_probabilities(self) -> Mapping[IVEMember, float]:
        return MappingProxyType(dict(zip(_MEMBERS, self._probabilities, strict=True)))

    @property
    def known_mass(self) -> float:
        return 1.0 - self.unknown_mass

    def probability_for(self, member: IVEMember | str) -> float:
        target = _ive_member(member, "probability member")
        return self._probabilities[_MEMBERS.index(target)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "members": [
                {"member": member.value, "probability": probability}
                for member, probability in zip(
                    _MEMBERS, self._probabilities, strict=True
                )
            ],
            "unknown_mass": self.unknown_mass,
        }

    @classmethod
    def from_dict(cls, value: object) -> "MemberProbabilityMass":
        mapping = _strict_mapping(value, "member probability mass")
        _strict_keys(mapping, ("members", "unknown_mass"), "member probability mass")
        raw_members = mapping["members"]
        if not isinstance(raw_members, list) or len(raw_members) != len(_MEMBERS):
            raise MemberStateArtifactError(
                "member probability artifact must contain exactly six members"
            )
        probabilities: dict[IVEMember, float] = {}
        for index, item in enumerate(raw_members):
            record = _strict_mapping(item, f"member probability {index}")
            _strict_keys(
                record,
                ("member", "probability"),
                f"member probability {index}",
            )
            try:
                member = _ive_member(record["member"], "artifact member")
            except MemberStateGraphError as exc:
                raise MemberStateArtifactError("invalid artifact IVE member") from exc
            probabilities[member] = record["probability"]
        return cls(probabilities, unknown_mass=mapping["unknown_mass"])


@dataclass(frozen=True, slots=True)
class CausalMemberObservation:
    """One verified/caller-supplied person state at an exact source time.

    Local tracker and pose tokens are opaque source/shot-local identifiers.  A
    face identity token is also opaque and never contains an embedding.  The
    graph does not infer any of these values from media.
    """

    observation_id: str
    source_id: str
    shot_id: str
    timestamp_seconds: float
    frame_index: int
    causal_order: int
    member_probabilities: MemberProbabilityMass
    face_bbox: NormalizedBBox | Sequence[float] | None = None
    body_bbox: NormalizedBBox | Sequence[float] | None = None
    face_identity_id: str | None = None
    pose_tracklet_id: str | None = None
    tracker_tracklet_id: str | None = None
    occlusion_probability: float = 0.0
    geometry_confidence: float = 1.0
    shot_confidence: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", _identifier(self.observation_id, "observation_id")
        )
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        object.__setattr__(self, "shot_id", _identifier(self.shot_id, "shot_id"))
        timestamp = _finite(self.timestamp_seconds, "timestamp_seconds")
        if timestamp < 0.0:
            raise MemberStateGraphError("timestamp_seconds must be non-negative")
        object.__setattr__(self, "timestamp_seconds", timestamp)
        object.__setattr__(
            self, "frame_index", _non_negative_integer(self.frame_index, "frame_index")
        )
        object.__setattr__(
            self, "causal_order", _non_negative_integer(self.causal_order, "causal_order")
        )
        if not isinstance(self.member_probabilities, MemberProbabilityMass):
            raise MemberStateGraphError(
                "member_probabilities must be MemberProbabilityMass"
            )
        for field_name in ("face_bbox", "body_bbox"):
            raw_box = getattr(self, field_name)
            if raw_box is not None and not isinstance(raw_box, NormalizedBBox):
                object.__setattr__(
                    self, field_name, NormalizedBBox.from_xyxy(raw_box)
                )
        for field_name in (
            "face_identity_id",
            "pose_tracklet_id",
            "tracker_tracklet_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_identifier(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "occlusion_probability",
            _probability(self.occlusion_probability, "occlusion_probability"),
        )
        object.__setattr__(
            self,
            "geometry_confidence",
            _probability(self.geometry_confidence, "geometry_confidence"),
        )
        object.__setattr__(
            self,
            "shot_confidence",
            _probability(self.shot_confidence, "shot_confidence"),
        )
        if self.face_identity_id is not None and self.face_bbox is None:
            raise MemberStateGraphError("face identity needs observed face geometry")
        if self.face_bbox is not None and self.body_bbox is not None:
            center_x, center_y = self.face_bbox.center
            if not (
                self.body_bbox.left <= center_x <= self.body_bbox.right
                and self.body_bbox.top <= center_y <= self.body_bbox.bottom
                and self.face_bbox.area <= self.body_bbox.area
            ):
                raise MemberStateGraphError(
                    "face bbox must be geometrically contained by the body bbox"
                )
        if not self.has_geometry:
            if not math.isclose(self.occlusion_probability, 1.0, abs_tol=1e-12):
                raise MemberStateGraphError(
                    "geometry-free observations must declare complete occlusion"
                )
            if self.geometry_confidence != 0.0:
                raise MemberStateGraphError(
                    "geometry-free observations must have zero geometry confidence"
                )
            if not math.isclose(
                self.member_probabilities.unknown_mass, 1.0, abs_tol=1e-12
            ):
                raise MemberStateGraphError(
                    "missing geometry cannot invent identity; unknown_mass must be 1"
                )
        elif self.geometry_confidence <= 0.0:
            raise MemberStateGraphError(
                "observed geometry needs positive geometry confidence"
            )

    @property
    def has_geometry(self) -> bool:
        return self.body_bbox is not None or self.face_bbox is not None

    @property
    def association_bbox(self) -> NormalizedBBox | None:
        return self.body_bbox or self.face_bbox

    @property
    def local_tokens(self) -> tuple[str, ...]:
        return tuple(
            value
            for value in (self.tracker_tracklet_id, self.pose_tracklet_id)
            if value is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "source_id": self.source_id,
            "shot_id": self.shot_id,
            "timestamp_seconds": self.timestamp_seconds,
            "frame_index": self.frame_index,
            "causal_order": self.causal_order,
            "member_probabilities": self.member_probabilities.to_dict(),
            "face_bbox_xyxy": self.face_bbox.to_list() if self.face_bbox else None,
            "body_bbox_xyxy": self.body_bbox.to_list() if self.body_bbox else None,
            "face_identity_id": self.face_identity_id,
            "pose_tracklet_id": self.pose_tracklet_id,
            "tracker_tracklet_id": self.tracker_tracklet_id,
            "occlusion_probability": self.occlusion_probability,
            "geometry_confidence": self.geometry_confidence,
            "shot_confidence": self.shot_confidence,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CausalMemberObservation":
        mapping = _strict_mapping(value, "member observation")
        keys = (
            "observation_id",
            "source_id",
            "shot_id",
            "timestamp_seconds",
            "frame_index",
            "causal_order",
            "member_probabilities",
            "face_bbox_xyxy",
            "body_bbox_xyxy",
            "face_identity_id",
            "pose_tracklet_id",
            "tracker_tracklet_id",
            "occlusion_probability",
            "geometry_confidence",
            "shot_confidence",
        )
        _strict_keys(mapping, keys, "member observation")
        try:
            return cls(
                observation_id=mapping["observation_id"],
                source_id=mapping["source_id"],
                shot_id=mapping["shot_id"],
                timestamp_seconds=mapping["timestamp_seconds"],
                frame_index=mapping["frame_index"],
                causal_order=mapping["causal_order"],
                member_probabilities=MemberProbabilityMass.from_dict(
                    mapping["member_probabilities"]
                ),
                face_bbox=mapping["face_bbox_xyxy"],
                body_bbox=mapping["body_bbox_xyxy"],
                face_identity_id=mapping["face_identity_id"],
                pose_tracklet_id=mapping["pose_tracklet_id"],
                tracker_tracklet_id=mapping["tracker_tracklet_id"],
                occlusion_probability=mapping["occlusion_probability"],
                geometry_confidence=mapping["geometry_confidence"],
                shot_confidence=mapping["shot_confidence"],
            )
        except (MemberStateGraphError, TypeError, ValueError) as exc:
            raise MemberStateArtifactError(str(exc)) from exc


class IdentityEvidenceKind(StrEnum):
    MANUAL_IDENTITY = "manual_identity"
    FACE_IDENTITY_MATCH = "face_identity_match"
    BODY_APPEARANCE_MATCH = "body_appearance_match"
    COSTUME_METADATA_MATCH = "costume_metadata_match"


@dataclass(frozen=True, slots=True)
class CorroboratedIdentityEvidence:
    """One scalar, provenance-bound re-identification modality.

    ``evidence_sha256`` identifies an external audit record.  The record itself
    and any embeddings used to produce its scalar confidence stay outside this
    graph artifact.
    """

    kind: IdentityEvidenceKind | str
    confidence: float
    provider_id: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, IdentityEvidenceKind):
            try:
                object.__setattr__(self, "kind", IdentityEvidenceKind(self.kind))
            except (TypeError, ValueError) as exc:
                raise MemberStateGraphError("invalid identity evidence kind") from exc
        object.__setattr__(
            self, "confidence", _probability(self.confidence, "evidence confidence")
        )
        object.__setattr__(
            self, "provider_id", _identifier(self.provider_id, "provider_id")
        )
        if not isinstance(self.evidence_sha256, str) or _SHA256.fullmatch(
            self.evidence_sha256
        ) is None:
            raise MemberStateGraphError(
                "identity evidence must reference a lowercase SHA-256"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "confidence": self.confidence,
            "provider_id": self.provider_id,
            "evidence_sha256": self.evidence_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CorroboratedIdentityEvidence":
        mapping = _strict_mapping(value, "identity evidence")
        _strict_keys(
            mapping,
            ("kind", "confidence", "provider_id", "evidence_sha256"),
            "identity evidence",
        )
        try:
            return cls(
                kind=mapping["kind"],
                confidence=mapping["confidence"],
                provider_id=mapping["provider_id"],
                evidence_sha256=mapping["evidence_sha256"],
            )
        except MemberStateGraphError as exc:
            raise MemberStateArtifactError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class ReIdentificationClaim:
    """Caller-supplied request for one conservative cross-tracklet edge."""

    left_observation_id: str
    right_observation_id: str
    member: IVEMember | str
    evidence: tuple[CorroboratedIdentityEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "left_observation_id",
            _identifier(self.left_observation_id, "left_observation_id"),
        )
        object.__setattr__(
            self,
            "right_observation_id",
            _identifier(self.right_observation_id, "right_observation_id"),
        )
        if self.left_observation_id == self.right_observation_id:
            raise MemberStateGraphError("re-identification endpoints must differ")
        if not isinstance(self.member, IVEMember):
            try:
                object.__setattr__(
                    self, "member", _ive_member(self.member, "re-identification member")
                )
            except MemberStateGraphError as exc:
                raise MemberStateGraphError("invalid re-identification member") from exc
        evidence = tuple(self.evidence)
        if not all(isinstance(item, CorroboratedIdentityEvidence) for item in evidence):
            raise MemberStateGraphError(
                "re-identification evidence must be typed scalar evidence"
            )
        object.__setattr__(
            self,
            "evidence",
            tuple(
                sorted(
                    evidence,
                    key=lambda item: (
                        item.kind.value,
                        item.provider_id,
                        item.evidence_sha256,
                    ),
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class MemberStateGraphConfig:
    maximum_association_gap_seconds: float = 0.35
    maximum_association_gap_frames: int = 12
    maximum_occlusion_gap_seconds: float = 1.0
    maximum_occlusion_gap_frames: int = 36
    maximum_occluded_observations: int = 12
    minimum_bbox_iou: float = 0.05
    maximum_center_distance: float = 0.20
    maximum_recovery_center_distance: float = 0.30
    maximum_area_ratio: float = 3.0
    maximum_recovery_area_ratio: float = 4.0
    minimum_reid_evidence_confidence: float = 0.88
    minimum_cross_source_evidence_confidence: float = 0.92
    minimum_reid_member_probability: float = 0.72
    maximum_reid_unknown_mass: float = 0.20
    minimum_reid_modalities: int = 2
    minimum_center_visibility: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "maximum_association_gap_seconds",
            "maximum_occlusion_gap_seconds",
            "maximum_center_distance",
            "maximum_recovery_center_distance",
            "maximum_area_ratio",
            "maximum_recovery_area_ratio",
        ):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise MemberStateGraphError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in (
            "maximum_association_gap_frames",
            "maximum_occlusion_gap_frames",
            "maximum_occluded_observations",
            "minimum_reid_modalities",
        ):
            value = _non_negative_integer(getattr(self, name), name)
            if value < 1:
                raise MemberStateGraphError(f"{name} must be at least 1")
        for name in (
            "minimum_bbox_iou",
            "minimum_reid_evidence_confidence",
            "minimum_cross_source_evidence_confidence",
            "minimum_reid_member_probability",
            "maximum_reid_unknown_mass",
            "minimum_center_visibility",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if self.maximum_occlusion_gap_seconds < self.maximum_association_gap_seconds:
            raise MemberStateGraphError(
                "occlusion gap seconds cannot be smaller than association gap"
            )
        if self.maximum_occlusion_gap_frames < self.maximum_association_gap_frames:
            raise MemberStateGraphError(
                "occlusion gap frames cannot be smaller than association gap"
            )
        if self.maximum_recovery_center_distance < self.maximum_center_distance:
            raise MemberStateGraphError(
                "recovery center bound cannot be tighter than association bound"
            )
        if self.maximum_recovery_area_ratio < self.maximum_area_ratio:
            raise MemberStateGraphError(
                "recovery area bound cannot be tighter than association bound"
            )
        if (
            self.minimum_cross_source_evidence_confidence
            < self.minimum_reid_evidence_confidence
        ):
            raise MemberStateGraphError(
                "cross-source re-ID threshold cannot be weaker than local re-ID"
            )
        if self.minimum_reid_modalities < 2:
            raise MemberStateGraphError(
                "re-identification requires at least two evidence modalities"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> "MemberStateGraphConfig":
        mapping = _strict_mapping(value, "member-state graph config")
        expected = tuple(cls.__dataclass_fields__)
        _strict_keys(mapping, expected, "member-state graph config")
        try:
            return cls(**{name: mapping[name] for name in expected})
        except MemberStateGraphError as exc:
            raise MemberStateArtifactError(str(exc)) from exc


class MemberStateEdgeKind(StrEnum):
    WITHIN_TRACKLET = "within_tracklet"
    OCCLUSION_RECOVERY = "occlusion_recovery"
    CORROBORATED_REIDENTIFICATION = "corroborated_reidentification"


@dataclass(frozen=True, slots=True)
class MemberStateEdge:
    edge_id: str
    kind: MemberStateEdgeKind | str
    left_observation_id: str
    right_observation_id: str
    left_tracklet_id: str
    right_tracklet_id: str
    confidence: float
    member: IVEMember | str | None = None
    evidence: tuple[CorroboratedIdentityEvidence, ...] = ()
    probability_mass_propagated: bool = False

    def __post_init__(self) -> None:
        for name in (
            "edge_id",
            "left_observation_id",
            "right_observation_id",
            "left_tracklet_id",
            "right_tracklet_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if not isinstance(self.kind, MemberStateEdgeKind):
            try:
                object.__setattr__(self, "kind", MemberStateEdgeKind(self.kind))
            except (TypeError, ValueError) as exc:
                raise MemberStateGraphError("invalid member-state edge kind") from exc
        object.__setattr__(
            self, "confidence", _probability(self.confidence, "edge confidence")
        )
        if not isinstance(self.probability_mass_propagated, bool):
            raise MemberStateGraphError("probability_mass_propagated must be boolean")
        if self.probability_mass_propagated:
            raise MemberStateGraphError(
                "member-state edges cannot propagate probability mass"
            )
        evidence = tuple(self.evidence)
        if self.kind is MemberStateEdgeKind.CORROBORATED_REIDENTIFICATION:
            if self.left_tracklet_id == self.right_tracklet_id:
                raise MemberStateGraphError("re-ID edge must join different tracklets")
            if self.member is None:
                raise MemberStateGraphError("re-ID edge needs an IVE member")
            if not isinstance(self.member, IVEMember):
                object.__setattr__(
                    self, "member", _ive_member(self.member, "edge member")
                )
            if not evidence:
                raise MemberStateGraphError("re-ID edge needs corroborated evidence")
        else:
            if self.left_tracklet_id != self.right_tracklet_id:
                raise MemberStateGraphError("local edge must remain inside one tracklet")
            if self.member is not None or evidence:
                raise MemberStateGraphError(
                    "local association edges cannot claim durable identity"
                )
        if not all(isinstance(item, CorroboratedIdentityEvidence) for item in evidence):
            raise MemberStateGraphError("edge evidence must be typed")
        object.__setattr__(self, "evidence", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "kind": self.kind.value,
            "left_observation_id": self.left_observation_id,
            "right_observation_id": self.right_observation_id,
            "left_tracklet_id": self.left_tracklet_id,
            "right_tracklet_id": self.right_tracklet_id,
            "confidence": self.confidence,
            "member": self.member.value if self.member is not None else None,
            "evidence": [item.to_dict() for item in self.evidence],
            "probability_mass_propagated": self.probability_mass_propagated,
        }

    @classmethod
    def from_dict(cls, value: object) -> "MemberStateEdge":
        mapping = _strict_mapping(value, "member-state edge")
        keys = (
            "edge_id",
            "kind",
            "left_observation_id",
            "right_observation_id",
            "left_tracklet_id",
            "right_tracklet_id",
            "confidence",
            "member",
            "evidence",
            "probability_mass_propagated",
        )
        _strict_keys(mapping, keys, "member-state edge")
        raw_evidence = mapping["evidence"]
        if not isinstance(raw_evidence, list):
            raise MemberStateArtifactError("edge evidence must be an array")
        try:
            return cls(
                edge_id=mapping["edge_id"],
                kind=mapping["kind"],
                left_observation_id=mapping["left_observation_id"],
                right_observation_id=mapping["right_observation_id"],
                left_tracklet_id=mapping["left_tracklet_id"],
                right_tracklet_id=mapping["right_tracklet_id"],
                confidence=mapping["confidence"],
                member=mapping["member"],
                evidence=tuple(
                    CorroboratedIdentityEvidence.from_dict(item)
                    for item in raw_evidence
                ),
                probability_mass_propagated=mapping[
                    "probability_mass_propagated"
                ],
            )
        except (MemberStateGraphError, TypeError, ValueError) as exc:
            raise MemberStateArtifactError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class MemberTracklet:
    tracklet_id: str
    source_id: str
    shot_id: str
    observation_ids: tuple[str, ...]
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    occlusion_recovery_count: int

    def __post_init__(self) -> None:
        for name in ("tracklet_id", "source_id", "shot_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        observations = tuple(
            _identifier(item, "tracklet observation_id")
            for item in self.observation_ids
        )
        if not observations or len(observations) != len(set(observations)):
            raise MemberStateGraphError(
                "tracklet observations must be non-empty and unique"
            )
        object.__setattr__(self, "observation_ids", observations)
        start = _finite(self.start_timestamp_seconds, "tracklet start timestamp")
        end = _finite(self.end_timestamp_seconds, "tracklet end timestamp")
        if start < 0.0 or end < start:
            raise MemberStateGraphError("tracklet timestamps must be ordered")
        object.__setattr__(self, "start_timestamp_seconds", start)
        object.__setattr__(self, "end_timestamp_seconds", end)
        object.__setattr__(
            self,
            "occlusion_recovery_count",
            _non_negative_integer(
                self.occlusion_recovery_count, "occlusion_recovery_count"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracklet_id": self.tracklet_id,
            "source_id": self.source_id,
            "shot_id": self.shot_id,
            "observation_ids": list(self.observation_ids),
            "start_timestamp_seconds": self.start_timestamp_seconds,
            "end_timestamp_seconds": self.end_timestamp_seconds,
            "occlusion_recovery_count": self.occlusion_recovery_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "MemberTracklet":
        mapping = _strict_mapping(value, "member tracklet")
        keys = (
            "tracklet_id",
            "source_id",
            "shot_id",
            "observation_ids",
            "start_timestamp_seconds",
            "end_timestamp_seconds",
            "occlusion_recovery_count",
        )
        _strict_keys(mapping, keys, "member tracklet")
        raw_ids = mapping["observation_ids"]
        if not isinstance(raw_ids, list):
            raise MemberStateArtifactError("tracklet observation_ids must be an array")
        try:
            return cls(
                tracklet_id=mapping["tracklet_id"],
                source_id=mapping["source_id"],
                shot_id=mapping["shot_id"],
                observation_ids=tuple(raw_ids),
                start_timestamp_seconds=mapping["start_timestamp_seconds"],
                end_timestamp_seconds=mapping["end_timestamp_seconds"],
                occlusion_recovery_count=mapping["occlusion_recovery_count"],
            )
        except MemberStateGraphError as exc:
            raise MemberStateArtifactError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class MemberStateGraph:
    config: MemberStateGraphConfig
    observations: tuple[CausalMemberObservation, ...]
    tracklets: tuple[MemberTracklet, ...]
    edges: tuple[MemberStateEdge, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.config, MemberStateGraphConfig):
            raise MemberStateGraphError("graph config must be typed")
        observations = tuple(self.observations)
        tracklets = tuple(self.tracklets)
        edges = tuple(self.edges)
        if not all(isinstance(item, CausalMemberObservation) for item in observations):
            raise MemberStateGraphError("graph observations must be typed")
        if not all(isinstance(item, MemberTracklet) for item in tracklets):
            raise MemberStateGraphError("graph tracklets must be typed")
        if not all(isinstance(item, MemberStateEdge) for item in edges):
            raise MemberStateGraphError("graph edges must be typed")
        expected_observations = tuple(
            sorted(observations, key=lambda item: (item.source_id, item.causal_order))
        )
        if observations != expected_observations:
            raise MemberStateGraphError("graph observations are not canonical")
        if tracklets != tuple(sorted(tracklets, key=lambda item: item.tracklet_id)):
            raise MemberStateGraphError("graph tracklets are not canonical")
        if edges != tuple(sorted(edges, key=lambda item: item.edge_id)):
            raise MemberStateGraphError("graph edges are not canonical")
        observation_by_id = {item.observation_id: item for item in observations}
        if len(observation_by_id) != len(observations):
            raise MemberStateGraphError("graph observation IDs must be unique")
        tracklet_by_id = {item.tracklet_id: item for item in tracklets}
        if len(tracklet_by_id) != len(tracklets):
            raise MemberStateGraphError("graph tracklet IDs must be unique")
        assigned: dict[str, str] = {}
        for tracklet in tracklets:
            prior_time = -1.0
            prior_order = -1
            for observation_id in tracklet.observation_ids:
                if observation_id not in observation_by_id:
                    raise MemberStateGraphError(
                        "tracklet references an unknown observation"
                    )
                if observation_id in assigned:
                    raise MemberStateGraphError(
                        "an observation cannot belong to multiple tracklets"
                    )
                observation = observation_by_id[observation_id]
                if (
                    observation.source_id != tracklet.source_id
                    or observation.shot_id != tracklet.shot_id
                ):
                    raise MemberStateGraphError("tracklet source/shot mismatch")
                if (
                    observation.timestamp_seconds < prior_time
                    or observation.causal_order <= prior_order
                ):
                    raise MemberStateGraphError(
                        "tracklet observations are not causally ordered"
                    )
                prior_time = observation.timestamp_seconds
                prior_order = observation.causal_order
                assigned[observation_id] = tracklet.tracklet_id
        if set(assigned) != set(observation_by_id):
            raise MemberStateGraphError("every graph observation needs one tracklet")
        if len({item.edge_id for item in edges}) != len(edges):
            raise MemberStateGraphError("graph edge IDs must be unique")
        for edge in edges:
            if (
                edge.left_observation_id not in observation_by_id
                or edge.right_observation_id not in observation_by_id
                or edge.left_tracklet_id not in tracklet_by_id
                or edge.right_tracklet_id not in tracklet_by_id
            ):
                raise MemberStateGraphError("edge references an unknown graph object")
            if assigned[edge.left_observation_id] != edge.left_tracklet_id or assigned[
                edge.right_observation_id
            ] != edge.right_tracklet_id:
                raise MemberStateGraphError("edge observation/tracklet mismatch")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "tracklets", tracklets)
        object.__setattr__(self, "edges", edges)

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.source_id for item in self.observations}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "observations": [item.to_dict() for item in self.observations],
            "tracklets": [item.to_dict() for item in self.tracklets],
            "edges": [item.to_dict() for item in self.edges],
        }

    @classmethod
    def from_dict(cls, value: object) -> "MemberStateGraph":
        mapping = _strict_mapping(value, "member-state graph")
        _strict_keys(
            mapping,
            ("config", "observations", "tracklets", "edges"),
            "member-state graph",
        )
        arrays: dict[str, list[Any]] = {}
        for key in ("observations", "tracklets", "edges"):
            raw = mapping[key]
            if not isinstance(raw, list):
                raise MemberStateArtifactError(f"graph {key} must be an array")
            arrays[key] = raw
        try:
            graph = cls(
                config=MemberStateGraphConfig.from_dict(mapping["config"]),
                observations=tuple(
                    CausalMemberObservation.from_dict(item)
                    for item in arrays["observations"]
                ),
                tracklets=tuple(
                    MemberTracklet.from_dict(item) for item in arrays["tracklets"]
                ),
                edges=tuple(
                    MemberStateEdge.from_dict(item) for item in arrays["edges"]
                ),
            )
            claims = tuple(
                ReIdentificationClaim(
                    left_observation_id=edge.left_observation_id,
                    right_observation_id=edge.right_observation_id,
                    member=edge.member,
                    evidence=edge.evidence,
                )
                for edge in graph.edges
                if edge.kind
                is MemberStateEdgeKind.CORROBORATED_REIDENTIFICATION
            )
            rebuilt = build_member_state_graph(
                graph.observations,
                reidentification_claims=claims,
                config=graph.config,
            )
            if rebuilt != graph:
                raise MemberStateArtifactError(
                    "artifact graph topology is not reproducible from its observations"
                )
            return graph
        except (MemberStateGraphError, TypeError, ValueError) as exc:
            raise MemberStateArtifactError(str(exc)) from exc


class _TrackState:
    def __init__(self, tracklet_id: str, observation: CausalMemberObservation) -> None:
        self.tracklet_id = tracklet_id
        self.observations = [observation]
        self.last = observation
        self.last_visible = observation if observation.has_geometry else None
        self.tracker_tokens = {
            observation.tracker_tracklet_id
        } - {None}
        self.pose_tokens = {observation.pose_tracklet_id} - {None}
        self.missing_run = 0 if observation.has_geometry else 1
        self.recovery_count = 0


def _tracklet_id(observation: CausalMemberObservation) -> str:
    payload = {
        "source_id": observation.source_id,
        "shot_id": observation.shot_id,
        "first_observation_id": observation.observation_id,
    }
    return "tracklet_" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _edge_id(
    kind: MemberStateEdgeKind,
    left_observation_id: str,
    right_observation_id: str,
    member: IVEMember | None = None,
) -> str:
    payload = {
        "kind": kind.value,
        "left_observation_id": left_observation_id,
        "right_observation_id": right_observation_id,
        "member": member.value if member is not None else None,
    }
    return "edge_" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _center_distance(left: NormalizedBBox, right: NormalizedBBox) -> float:
    left_x, left_y = left.center
    right_x, right_y = right.center
    return math.hypot(left_x - right_x, left_y - right_y)


def _area_ratio(left: NormalizedBBox, right: NormalizedBBox) -> float:
    return max(left.area, right.area) / min(left.area, right.area)


def _geometry_match(
    left: CausalMemberObservation,
    right: CausalMemberObservation,
    *,
    recovery: bool,
    config: MemberStateGraphConfig,
) -> tuple[bool, float, float]:
    left_box = left.association_bbox
    right_box = right.association_bbox
    if left_box is None or right_box is None:
        return (False, 0.0, math.inf)
    iou = left_box.intersection_over_union(right_box)
    distance = _center_distance(left_box, right_box)
    maximum_distance = (
        config.maximum_recovery_center_distance
        if recovery
        else config.maximum_center_distance
    )
    maximum_ratio = (
        config.maximum_recovery_area_ratio
        if recovery
        else config.maximum_area_ratio
    )
    compatible = (
        iou >= config.minimum_bbox_iou
        and distance <= maximum_distance
        and _area_ratio(left_box, right_box) <= maximum_ratio
    )
    return (compatible, iou, distance)


def _within_gap(
    left: CausalMemberObservation,
    right: CausalMemberObservation,
    *,
    occlusion: bool,
    config: MemberStateGraphConfig,
) -> bool:
    seconds = right.timestamp_seconds - left.timestamp_seconds
    frames = right.frame_index - left.frame_index
    if seconds <= 0.0 or frames <= 0:
        return False
    if occlusion:
        return (
            seconds <= config.maximum_occlusion_gap_seconds
            and frames <= config.maximum_occlusion_gap_frames
        )
    return (
        seconds <= config.maximum_association_gap_seconds
        and frames <= config.maximum_association_gap_frames
    )


def _validate_observation_order(
    observations: Sequence[CausalMemberObservation],
) -> None:
    seen_ids: set[str] = set()
    last_by_source: dict[str, CausalMemberObservation] = {}
    active_shot_by_source: dict[str, str] = {}
    closed_shots_by_source: dict[str, set[str]] = {}
    for observation in observations:
        if not isinstance(observation, CausalMemberObservation):
            raise MemberStateGraphError("observations must be typed")
        if observation.observation_id in seen_ids:
            raise MemberStateGraphError("observation IDs must be globally unique")
        seen_ids.add(observation.observation_id)
        previous = last_by_source.get(observation.source_id)
        if previous is not None:
            if observation.causal_order <= previous.causal_order:
                raise MemberStateGraphError(
                    f"source {observation.source_id} causal order regressed"
                )
            if observation.timestamp_seconds < previous.timestamp_seconds:
                raise MemberStateGraphError(
                    f"source {observation.source_id} timestamp order regressed"
                )
            if observation.frame_index < previous.frame_index:
                raise MemberStateGraphError(
                    f"source {observation.source_id} frame order regressed"
                )
            same_timestamp = (
                observation.timestamp_seconds == previous.timestamp_seconds
            )
            same_frame = observation.frame_index == previous.frame_index
            if same_timestamp != same_frame:
                raise MemberStateGraphError(
                    "same-frame observations must share an exact timestamp and "
                    "distinct-frame observations must advance it"
                )
            if same_frame and observation.shot_id != previous.shot_id:
                raise MemberStateGraphError("one source frame cannot span two shots")
        prior_shot = active_shot_by_source.get(observation.source_id)
        if prior_shot is None:
            active_shot_by_source[observation.source_id] = observation.shot_id
            closed_shots_by_source[observation.source_id] = set()
        elif observation.shot_id != prior_shot:
            closed = closed_shots_by_source[observation.source_id]
            closed.add(prior_shot)
            if observation.shot_id in closed:
                raise MemberStateGraphError(
                    f"source {observation.source_id} returned to a closed shot"
                )
            active_shot_by_source[observation.source_id] = observation.shot_id
        last_by_source[observation.source_id] = observation


def _frame_groups(
    observations: Sequence[CausalMemberObservation],
) -> tuple[tuple[CausalMemberObservation, ...], ...]:
    groups: list[list[CausalMemberObservation]] = []
    current_key: tuple[int, float, str] | None = None
    for observation in observations:
        key = (
            observation.frame_index,
            observation.timestamp_seconds,
            observation.shot_id,
        )
        if key != current_key:
            groups.append([])
            current_key = key
        groups[-1].append(observation)
    return tuple(tuple(group) for group in groups)


def _ensure_frame_token_uniqueness(
    frame: Sequence[CausalMemberObservation],
) -> None:
    for field_name in ("tracker_tracklet_id", "pose_tracklet_id"):
        tokens = [
            getattr(observation, field_name)
            for observation in frame
            if getattr(observation, field_name) is not None
        ]
        if len(tokens) != len(set(tokens)):
            raise MemberStateGraphError(
                f"one source frame reuses {field_name} for multiple people"
            )


def _association_candidate(
    observation: CausalMemberObservation,
    state: _TrackState,
    config: MemberStateGraphConfig,
) -> tuple[int, float, float, MemberStateEdgeKind, float] | None:
    tracker_match = (
        observation.tracker_tracklet_id is not None
        and observation.tracker_tracklet_id in state.tracker_tokens
    )
    pose_match = (
        observation.pose_tracklet_id is not None
        and observation.pose_tracklet_id in state.pose_tokens
    )
    explicit_match = tracker_match or pose_match
    local_gap = _within_gap(state.last, observation, occlusion=False, config=config)
    anchor = state.last_visible or state.last
    recovery_gap = _within_gap(anchor, observation, occlusion=True, config=config)
    recovery = state.last_visible is None or not state.last.has_geometry or not local_gap

    if not observation.has_geometry:
        if (
            not explicit_match
            or not recovery_gap
            or state.missing_run >= config.maximum_occluded_observations
        ):
            return None
        priority = 0 if tracker_match else 1
        return (priority, 0.0, 0.0, MemberStateEdgeKind.WITHIN_TRACKLET, 0.0)

    if explicit_match:
        if recovery:
            if not recovery_gap or state.missing_run > config.maximum_occluded_observations:
                return None
        elif not local_gap:
            return None
        if state.last_visible is not None:
            compatible, iou, distance = _geometry_match(
                state.last_visible,
                observation,
                recovery=recovery,
                config=config,
            )
            if not compatible:
                raise MemberStateGraphError(
                    "a continuing local tracklet token violates geometry bounds"
                )
        else:
            iou, distance = 0.0, 0.0
        kind = (
            MemberStateEdgeKind.OCCLUSION_RECOVERY
            if recovery
            else MemberStateEdgeKind.WITHIN_TRACKLET
        )
        confidence = min(
            observation.geometry_confidence,
            observation.shot_confidence,
            0.98 if tracker_match else 0.94,
        )
        return (0 if tracker_match else 1, -iou, distance, kind, confidence)

    if observation.local_tokens or state.last.local_tokens:
        return None
    if not local_gap or state.last_visible is None:
        return None
    compatible, iou, distance = _geometry_match(
        state.last_visible, observation, recovery=False, config=config
    )
    if not compatible:
        return None
    confidence = min(
        observation.geometry_confidence,
        observation.shot_confidence,
        max(0.0, min(1.0, 0.5 * iou + 0.5 * (1.0 - distance))),
    )
    return (2, -iou, distance, MemberStateEdgeKind.WITHIN_TRACKLET, confidence)


def _local_graph(
    observations: Sequence[CausalMemberObservation],
    config: MemberStateGraphConfig,
) -> tuple[list[_TrackState], list[MemberStateEdge]]:
    states: list[_TrackState] = []
    edges: list[MemberStateEdge] = []
    by_source: dict[str, list[CausalMemberObservation]] = {}
    for observation in observations:
        by_source.setdefault(observation.source_id, []).append(observation)

    for source_id in sorted(by_source):
        source_states: list[_TrackState] = []
        current_shot: str | None = None
        for frame in _frame_groups(by_source[source_id]):
            _ensure_frame_token_uniqueness(frame)
            if current_shot != frame[0].shot_id:
                current_shot = frame[0].shot_id
                source_states = []
            else:
                # Expired states remain in the immutable output but cannot
                # participate in later matching.  Besides bounding work, this
                # prevents a recycled local tracker token from reaching across
                # the configured occlusion horizon.
                source_states = [
                    state
                    for state in source_states
                    if _within_gap(
                        state.last_visible or state.last,
                        frame[0],
                        occlusion=True,
                        config=config,
                    )
                ]
            candidates: list[
                tuple[
                    int,
                    float,
                    float,
                    str,
                    str,
                    MemberStateEdgeKind,
                    float,
                    CausalMemberObservation,
                    _TrackState,
                ]
            ] = []
            for observation in frame:
                explicit_matches: set[str] = set()
                for state in source_states:
                    candidate = _association_candidate(observation, state, config)
                    if candidate is None:
                        continue
                    priority, negative_iou, distance, kind, confidence = candidate
                    if priority < 2:
                        explicit_matches.add(state.tracklet_id)
                    candidates.append(
                        (
                            priority,
                            negative_iou,
                            distance,
                            state.tracklet_id,
                            observation.observation_id,
                            kind,
                            confidence,
                            observation,
                            state,
                        )
                    )
                if len(explicit_matches) > 1:
                    raise MemberStateGraphError(
                        "local tracker/pose tokens ambiguously join two active tracklets"
                    )
            used_observations: set[str] = set()
            used_tracklets: set[str] = set()
            for candidate in sorted(candidates, key=lambda item: item[:5]):
                (
                    _priority,
                    _negative_iou,
                    _distance,
                    tracklet_id,
                    observation_id,
                    kind,
                    confidence,
                    observation,
                    state,
                ) = candidate
                if observation_id in used_observations or tracklet_id in used_tracklets:
                    continue
                previous = state.last
                edges.append(
                    MemberStateEdge(
                        edge_id=_edge_id(
                            kind, previous.observation_id, observation.observation_id
                        ),
                        kind=kind,
                        left_observation_id=previous.observation_id,
                        right_observation_id=observation.observation_id,
                        left_tracklet_id=state.tracklet_id,
                        right_tracklet_id=state.tracklet_id,
                        confidence=confidence,
                    )
                )
                state.observations.append(observation)
                state.last = observation
                if observation.has_geometry:
                    if kind is MemberStateEdgeKind.OCCLUSION_RECOVERY:
                        state.recovery_count += 1
                    state.last_visible = observation
                    state.missing_run = 0
                else:
                    state.missing_run += 1
                if observation.tracker_tracklet_id is not None:
                    state.tracker_tokens.add(observation.tracker_tracklet_id)
                if observation.pose_tracklet_id is not None:
                    state.pose_tokens.add(observation.pose_tracklet_id)
                used_observations.add(observation_id)
                used_tracklets.add(tracklet_id)
            for observation in frame:
                if observation.observation_id in used_observations:
                    continue
                state = _TrackState(_tracklet_id(observation), observation)
                source_states.append(state)
                states.append(state)
    return (states, edges)


def _validate_reid_claim(
    claim: ReIdentificationClaim,
    *,
    observation_by_id: Mapping[str, CausalMemberObservation],
    tracklet_by_observation: Mapping[str, str],
    config: MemberStateGraphConfig,
) -> tuple[CausalMemberObservation, CausalMemberObservation, float]:
    if not isinstance(claim, ReIdentificationClaim):
        raise MemberStateGraphError("re-identification claims must be typed")
    try:
        left = observation_by_id[claim.left_observation_id]
        right = observation_by_id[claim.right_observation_id]
    except KeyError as exc:
        raise MemberStateGraphError(
            "re-identification claim references an unknown observation"
        ) from exc
    if tracklet_by_observation[left.observation_id] == tracklet_by_observation[
        right.observation_id
    ]:
        raise MemberStateGraphError(
            "re-identification claim cannot target one existing tracklet"
        )
    kinds = {item.kind for item in claim.evidence}
    hashes = {item.evidence_sha256 for item in claim.evidence}
    if len(kinds) < config.minimum_reid_modalities or len(hashes) != len(
        claim.evidence
    ):
        raise MemberStateGraphError(
            "re-identification needs distinct corroborating modalities and records"
        )
    if not kinds.intersection(
        {IdentityEvidenceKind.MANUAL_IDENTITY, IdentityEvidenceKind.FACE_IDENTITY_MATCH}
    ):
        raise MemberStateGraphError(
            "re-identification needs a manual or face-identity anchor"
        )
    if IdentityEvidenceKind.FACE_IDENTITY_MATCH in kinds and (
        left.face_bbox is None or right.face_bbox is None
    ):
        raise MemberStateGraphError("face re-identification evidence needs face boxes")
    if IdentityEvidenceKind.BODY_APPEARANCE_MATCH in kinds and (
        left.body_bbox is None or right.body_bbox is None
    ):
        raise MemberStateGraphError("body re-identification evidence needs body boxes")
    cross_source = left.source_id != right.source_id
    threshold = (
        config.minimum_cross_source_evidence_confidence
        if cross_source
        else config.minimum_reid_evidence_confidence
    )
    if any(item.confidence < threshold for item in claim.evidence):
        raise MemberStateGraphError(
            "re-identification evidence does not pass the conservative threshold"
        )
    endpoint_probabilities = (
        left.member_probabilities.probability_for(claim.member),
        right.member_probabilities.probability_for(claim.member),
    )
    if min(endpoint_probabilities) < config.minimum_reid_member_probability:
        raise MemberStateGraphError(
            "re-identification endpoint member probability is too low"
        )
    if max(
        left.member_probabilities.unknown_mass,
        right.member_probabilities.unknown_mass,
    ) > config.maximum_reid_unknown_mass:
        raise MemberStateGraphError(
            "re-identification endpoint carries too much unknown mass"
        )
    confidence = min(
        *endpoint_probabilities,
        *(item.confidence for item in claim.evidence),
        left.shot_confidence,
        right.shot_confidence,
    )
    return (left, right, confidence)


def build_member_state_graph(
    observations: Sequence[CausalMemberObservation],
    *,
    reidentification_claims: Sequence[ReIdentificationClaim] = (),
    config: MemberStateGraphConfig | None = None,
) -> MemberStateGraph:
    """Build deterministic topology without estimating identity from pixels."""

    settings = config or MemberStateGraphConfig()
    if not isinstance(settings, MemberStateGraphConfig):
        raise MemberStateGraphError("config must be MemberStateGraphConfig")
    supplied = tuple(observations)
    _validate_observation_order(supplied)
    canonical = tuple(
        sorted(supplied, key=lambda item: (item.source_id, item.causal_order))
    )
    states, local_edges = _local_graph(canonical, settings)
    tracklets = tuple(
        sorted(
            (
                MemberTracklet(
                    tracklet_id=state.tracklet_id,
                    source_id=state.observations[0].source_id,
                    shot_id=state.observations[0].shot_id,
                    observation_ids=tuple(
                        item.observation_id for item in state.observations
                    ),
                    start_timestamp_seconds=state.observations[0].timestamp_seconds,
                    end_timestamp_seconds=state.observations[-1].timestamp_seconds,
                    occlusion_recovery_count=state.recovery_count,
                )
                for state in states
            ),
            key=lambda item: item.tracklet_id,
        )
    )
    observation_by_id = {item.observation_id: item for item in canonical}
    tracklet_by_observation = {
        observation_id: tracklet.tracklet_id
        for tracklet in tracklets
        for observation_id in tracklet.observation_ids
    }
    reid_edges: list[MemberStateEdge] = []
    claimed_pairs: set[tuple[str, str, IVEMember]] = set()
    for claim in tuple(reidentification_claims):
        left, right, confidence = _validate_reid_claim(
            claim,
            observation_by_id=observation_by_id,
            tracklet_by_observation=tracklet_by_observation,
            config=settings,
        )
        left_tracklet = tracklet_by_observation[left.observation_id]
        right_tracklet = tracklet_by_observation[right.observation_id]
        pair = tuple(sorted((left_tracklet, right_tracklet)))
        pair_key = (pair[0], pair[1], claim.member)
        if pair_key in claimed_pairs:
            raise MemberStateGraphError(
                "duplicate re-identification claim for one tracklet pair"
            )
        claimed_pairs.add(pair_key)
        if left.source_id == right.source_id:
            endpoint_key = lambda item: (item.causal_order, item.observation_id)
        else:
            endpoint_key = lambda item: (item.source_id, item.observation_id)
        if endpoint_key(right) < endpoint_key(left):
            left, right = right, left
            left_tracklet, right_tracklet = right_tracklet, left_tracklet
        reid_edges.append(
            MemberStateEdge(
                edge_id=_edge_id(
                    MemberStateEdgeKind.CORROBORATED_REIDENTIFICATION,
                    left.observation_id,
                    right.observation_id,
                    claim.member,
                ),
                kind=MemberStateEdgeKind.CORROBORATED_REIDENTIFICATION,
                left_observation_id=left.observation_id,
                right_observation_id=right.observation_id,
                left_tracklet_id=left_tracklet,
                right_tracklet_id=right_tracklet,
                confidence=confidence,
                member=claim.member,
                evidence=claim.evidence,
            )
        )
    return MemberStateGraph(
        config=settings,
        observations=canonical,
        tracklets=tracklets,
        edges=tuple(
            sorted((*local_edges, *reid_edges), key=lambda item: item.edge_id)
        ),
    )


@dataclass(frozen=True, slots=True)
class TargetMemberTimestampFeatures:
    """Scalar member-state features ready for a later StageEdit bridge."""

    source_id: str
    timestamp_seconds: float
    target_member: IVEMember | str
    target_member_probability: float
    visibility_probability: float
    center_xy: tuple[float, float] | None
    prominence_probability: float
    uncertainty_probability: float
    state_observed: bool
    observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        timestamp = _finite(self.timestamp_seconds, "timestamp_seconds")
        if timestamp < 0.0:
            raise MemberStateGraphError("timestamp_seconds must be non-negative")
        object.__setattr__(self, "timestamp_seconds", timestamp)
        if not isinstance(self.target_member, IVEMember):
            object.__setattr__(
                self, "target_member", _ive_member(self.target_member, "target_member")
            )
        for name in (
            "target_member_probability",
            "visibility_probability",
            "prominence_probability",
            "uncertainty_probability",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if self.center_xy is not None:
            if len(self.center_xy) != 2:
                raise MemberStateGraphError("center_xy must contain two coordinates")
            center = tuple(_probability(value, "center coordinate") for value in self.center_xy)
            object.__setattr__(self, "center_xy", center)
        if not isinstance(self.state_observed, bool):
            raise MemberStateGraphError("state_observed must be boolean")
        ids = tuple(_identifier(item, "feature observation_id") for item in self.observation_ids)
        if len(ids) != len(set(ids)):
            raise MemberStateGraphError("feature observation IDs must be unique")
        if self.state_observed != bool(ids):
            raise MemberStateGraphError(
                "state_observed must exactly describe observation provenance"
            )
        if self.center_xy is not None and self.visibility_probability <= 0.0:
            raise MemberStateGraphError("a target center needs visibility evidence")
        object.__setattr__(self, "observation_ids", ids)

    @property
    def center_x(self) -> float | None:
        return self.center_xy[0] if self.center_xy is not None else None

    @property
    def center_y(self) -> float | None:
        return self.center_xy[1] if self.center_xy is not None else None

    def to_stage_edit_features(self) -> dict[str, Any]:
        """Return only scalar/normalized fields for a future materializer."""

        return {
            "target_member_probability": self.target_member_probability,
            "target_visibility_probability": self.visibility_probability,
            "target_center_x": self.center_x,
            "target_center_y": self.center_y,
            "target_prominence_probability": self.prominence_probability,
            "target_member_uncertainty_probability": self.uncertainty_probability,
            "target_member_state_observed": self.state_observed,
        }


class MemberStateFeatureAdapter:
    """Exact-timestamp adapter; it never interpolates identity through gaps."""

    def __init__(self, graph: MemberStateGraph) -> None:
        if not isinstance(graph, MemberStateGraph):
            raise MemberStateGraphError("feature adapter needs MemberStateGraph")
        self._graph = graph
        indexed: dict[tuple[str, float], list[CausalMemberObservation]] = {}
        for observation in graph.observations:
            indexed.setdefault(
                (observation.source_id, observation.timestamp_seconds), []
            ).append(observation)
        self._observations_by_time = {
            key: tuple(sorted(value, key=lambda item: item.causal_order))
            for key, value in indexed.items()
        }

    @staticmethod
    def _union_probability(values: Iterable[float]) -> float:
        complement = 1.0
        for value in values:
            complement *= 1.0 - min(1.0, max(0.0, value))
        return min(1.0, max(0.0, 1.0 - complement))

    def at_timestamp(
        self,
        *,
        source_id: str,
        timestamp_seconds: float,
        target_member: IVEMember | str,
    ) -> TargetMemberTimestampFeatures:
        source = _identifier(source_id, "source_id")
        if source not in self._graph.source_ids:
            raise MemberStateGraphError(f"unknown graph source {source!r}")
        timestamp = _finite(timestamp_seconds, "timestamp_seconds")
        if timestamp < 0.0:
            raise MemberStateGraphError("timestamp_seconds must be non-negative")
        target = _ive_member(target_member, "target_member")
        observations = self._observations_by_time.get((source, timestamp), ())
        if not observations:
            return TargetMemberTimestampFeatures(
                source_id=source,
                timestamp_seconds=timestamp,
                target_member=target,
                target_member_probability=0.0,
                visibility_probability=0.0,
                center_xy=None,
                prominence_probability=0.0,
                uncertainty_probability=1.0,
                state_observed=False,
                observation_ids=(),
            )

        target_probabilities = tuple(
            item.member_probabilities.probability_for(target)
            for item in observations
        )
        target_probability = self._union_probability(target_probabilities)
        visibility_weights: list[float] = []
        center_terms: list[tuple[float, float, float]] = []
        target_area = 0.0
        observed_area = 0.0
        for observation, target_probability_for_person in zip(
            observations, target_probabilities, strict=True
        ):
            box = observation.association_bbox
            visible_geometry = (
                (1.0 - observation.occlusion_probability)
                * observation.geometry_confidence
                * observation.shot_confidence
                if box is not None
                else 0.0
            )
            target_visibility = target_probability_for_person * visible_geometry
            visibility_weights.append(target_visibility)
            if box is not None:
                observed_area += visible_geometry * box.area
                target_area += target_visibility * box.area
                if target_visibility > 0.0:
                    center_x, center_y = box.center
                    center_terms.append(
                        (target_visibility, center_x, center_y)
                    )
        visibility = self._union_probability(visibility_weights)
        total_center_weight = math.fsum(item[0] for item in center_terms)
        center: tuple[float, float] | None = None
        if (
            visibility >= self._graph.config.minimum_center_visibility
            and total_center_weight > 0.0
        ):
            center = (
                math.fsum(weight * x for weight, x, _y in center_terms)
                / total_center_weight,
                math.fsum(weight * y for weight, _x, y in center_terms)
                / total_center_weight,
            )
        prominence = (
            min(1.0, max(0.0, target_area / observed_area))
            if observed_area > 0.0
            else 0.0
        )
        unknown_union = self._union_probability(
            item.member_probabilities.unknown_mass for item in observations
        )
        target_ambiguity = 4.0 * target_probability * (1.0 - target_probability)
        shot_coverage = max(item.shot_confidence for item in observations)
        uncertainty = 1.0 - (
            (1.0 - unknown_union)
            * (1.0 - target_ambiguity)
            * shot_coverage
        )
        return TargetMemberTimestampFeatures(
            source_id=source,
            timestamp_seconds=timestamp,
            target_member=target,
            target_member_probability=target_probability,
            visibility_probability=visibility,
            center_xy=center,
            prominence_probability=prominence,
            uncertainty_probability=min(1.0, max(0.0, uncertainty)),
            state_observed=True,
            observation_ids=tuple(item.observation_id for item in observations),
        )

    def for_target(
        self, target_member: IVEMember | str
    ) -> tuple[TargetMemberTimestampFeatures, ...]:
        target = _ive_member(target_member, "target_member")
        return tuple(
            self.at_timestamp(
                source_id=source_id,
                timestamp_seconds=timestamp,
                target_member=target,
            )
            for source_id, timestamp in sorted(self._observations_by_time)
        )


def adapt_target_member_features(
    graph: MemberStateGraph,
    target_member: IVEMember | str,
) -> tuple[TargetMemberTimestampFeatures, ...]:
    return MemberStateFeatureAdapter(graph).for_target(target_member)


@dataclass(frozen=True, slots=True)
class MemberStateSourceProvenance:
    source_id: str
    source_content_sha256: str
    observation_manifest_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        for name in ("source_content_sha256", "observation_manifest_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise MemberStateGraphError(f"{name} must be a lowercase SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "source_content_sha256": self.source_content_sha256,
            "observation_manifest_sha256": self.observation_manifest_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "MemberStateSourceProvenance":
        mapping = _strict_mapping(value, "source provenance")
        _strict_keys(
            mapping,
            (
                "source_id",
                "source_content_sha256",
                "observation_manifest_sha256",
            ),
            "source provenance",
        )
        try:
            return cls(
                source_id=mapping["source_id"],
                source_content_sha256=mapping["source_content_sha256"],
                observation_manifest_sha256=mapping[
                    "observation_manifest_sha256"
                ],
            )
        except MemberStateGraphError as exc:
            raise MemberStateArtifactError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class MemberStateArtifactProvenance:
    """Complete caller and source identity for an immutable graph artifact."""

    producer_id: str
    producer_version: str
    created_at_utc: str
    sources: tuple[MemberStateSourceProvenance, ...]
    observation_basis: str = OBSERVATION_BASIS

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "producer_id", _identifier(self.producer_id, "producer_id")
        )
        object.__setattr__(
            self,
            "producer_version",
            _identifier(self.producer_version, "producer_version"),
        )
        if not isinstance(self.created_at_utc, str):
            raise MemberStateGraphError("created_at_utc must be an RFC3339 string")
        try:
            parsed = datetime.fromisoformat(self.created_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise MemberStateGraphError("created_at_utc must be valid RFC3339") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise MemberStateGraphError("created_at_utc must use UTC")
        if not self.created_at_utc.endswith("Z"):
            raise MemberStateGraphError("created_at_utc must use canonical Z suffix")
        supplied_sources = tuple(self.sources)
        if not all(
            isinstance(item, MemberStateSourceProvenance)
            for item in supplied_sources
        ):
            raise MemberStateGraphError("sources must be typed provenance records")
        sources = tuple(sorted(supplied_sources, key=lambda item: item.source_id))
        if len({item.source_id for item in sources}) != len(sources):
            raise MemberStateGraphError("source provenance IDs must be unique")
        object.__setattr__(self, "sources", sources)
        if self.observation_basis != OBSERVATION_BASIS:
            raise MemberStateGraphError(
                "artifact must declare the caller-supplied observation basis"
            )

    def validate_for_graph(self, graph: MemberStateGraph) -> None:
        if tuple(item.source_id for item in self.sources) != graph.source_ids:
            raise MemberStateGraphError(
                "artifact source provenance must exactly cover graph sources"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer_id": self.producer_id,
            "producer_version": self.producer_version,
            "created_at_utc": self.created_at_utc,
            "observation_basis": self.observation_basis,
            "sources": [item.to_dict() for item in self.sources],
        }

    @classmethod
    def from_dict(cls, value: object) -> "MemberStateArtifactProvenance":
        mapping = _strict_mapping(value, "member-state provenance")
        _strict_keys(
            mapping,
            (
                "producer_id",
                "producer_version",
                "created_at_utc",
                "observation_basis",
                "sources",
            ),
            "member-state provenance",
        )
        raw_sources = mapping["sources"]
        if not isinstance(raw_sources, list):
            raise MemberStateArtifactError("provenance sources must be an array")
        try:
            return cls(
                producer_id=mapping["producer_id"],
                producer_version=mapping["producer_version"],
                created_at_utc=mapping["created_at_utc"],
                observation_basis=mapping["observation_basis"],
                sources=tuple(
                    MemberStateSourceProvenance.from_dict(item)
                    for item in raw_sources
                ),
            )
        except MemberStateGraphError as exc:
            raise MemberStateArtifactError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class MemberStateGraphArtifact:
    path: Path
    content_sha256: str
    graph: MemberStateGraph
    provenance: MemberStateArtifactProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))
        if not isinstance(self.content_sha256, str) or _SHA256.fullmatch(
            self.content_sha256
        ) is None:
            raise MemberStateArtifactError("artifact content SHA-256 is invalid")
        if not isinstance(self.graph, MemberStateGraph) or not isinstance(
            self.provenance, MemberStateArtifactProvenance
        ):
            raise MemberStateArtifactError("artifact graph/provenance must be typed")
        self.provenance.validate_for_graph(self.graph)


def _artifact_payload(
    graph: MemberStateGraph,
    provenance: MemberStateArtifactProvenance,
) -> tuple[dict[str, Any], str, bytes]:
    provenance.validate_for_graph(graph)
    payload: dict[str, Any] = {
        "contract": MEMBER_STATE_GRAPH_CONTRACT,
        "schema_version": MEMBER_STATE_GRAPH_SCHEMA_VERSION,
        "content_hash_basis": _CONTENT_HASH_BASIS,
        "content_sha256": None,
        "provenance": provenance.to_dict(),
        "graph": graph.to_dict(),
        "privacy": {
            "raw_frames_persisted": False,
            "compressed_frames_persisted": False,
            "face_identity_embeddings_persisted": False,
            "body_appearance_embeddings_persisted": False,
        },
    }
    _reject_forbidden_persisted_keys(payload)
    digest = _content_sha256(payload)
    payload["content_sha256"] = digest
    encoded = _canonical_json_bytes(payload)
    if not 1 <= len(encoded) <= _MAXIMUM_ARTIFACT_BYTES:
        raise MemberStateArtifactError("member-state artifact size is invalid")
    return (payload, digest, encoded)


def _safe_directory(parent: Path, name: str) -> Path:
    child = parent / name
    if child.is_symlink():
        raise MemberStateArtifactError(f"artifact directory is a symlink: {child}")
    child.mkdir(exist_ok=True)
    if not child.is_dir():
        raise MemberStateArtifactError(f"artifact path is not a directory: {child}")
    return child


def write_member_state_graph_artifact(
    artifact_directory: Path,
    graph: MemberStateGraph,
    provenance: MemberStateArtifactProvenance,
) -> MemberStateGraphArtifact:
    """Atomically publish one immutable content-addressed JSON artifact."""

    if not isinstance(graph, MemberStateGraph):
        raise MemberStateArtifactError("graph must be MemberStateGraph")
    if not isinstance(provenance, MemberStateArtifactProvenance):
        raise MemberStateArtifactError(
            "provenance must be MemberStateArtifactProvenance"
        )
    _payload, digest, encoded = _artifact_payload(graph, provenance)
    root = Path(artifact_directory).expanduser()
    if root.is_symlink():
        raise MemberStateArtifactError("artifact root cannot be a symbolic link")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise MemberStateArtifactError("artifact root must be a directory")
    contract_directory = _safe_directory(root, MEMBER_STATE_GRAPH_CONTRACT)
    shard = _safe_directory(contract_directory, digest[:2])
    path = shard / f"{digest}.json"
    if path.is_symlink():
        raise MemberStateArtifactError("artifact path cannot be a symbolic link")
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise MemberStateArtifactError("cannot read existing artifact") from exc
        if not hmac.compare_digest(existing, encoded):
            raise MemberStateArtifactError(
                "content-addressed artifact path already contains different bytes"
            )
        return MemberStateGraphArtifact(path, digest, graph, provenance)

    temporary = shard / f".{digest}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = path.read_bytes()
            if not hmac.compare_digest(existing, encoded):
                raise MemberStateArtifactError(
                    "concurrent artifact writer published different bytes"
                )
        directory_fd = os.open(shard, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (OSError, MemberStateArtifactError) as exc:
        if isinstance(exc, MemberStateArtifactError):
            raise
        raise MemberStateArtifactError("failed to publish member-state artifact") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
    return MemberStateGraphArtifact(path, digest, graph, provenance)


def load_member_state_graph_artifact(
    path: Path,
    *,
    expected_content_sha256: str | None = None,
) -> MemberStateGraphArtifact:
    """Load and semantically validate a content-addressed graph artifact."""

    artifact_path = Path(path).expanduser()
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise MemberStateArtifactError("artifact must be a non-symlink regular file")
    try:
        size = artifact_path.stat().st_size
        if not 1 <= size <= _MAXIMUM_ARTIFACT_BYTES:
            raise MemberStateArtifactError("artifact size is invalid")
        encoded = artifact_path.read_bytes()
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except MemberStateArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MemberStateArtifactError("cannot decode member-state artifact") from exc
    mapping = _strict_mapping(payload, "member-state artifact")
    _strict_keys(
        mapping,
        (
            "contract",
            "schema_version",
            "content_hash_basis",
            "content_sha256",
            "provenance",
            "graph",
            "privacy",
        ),
        "member-state artifact",
    )
    _reject_forbidden_persisted_keys(mapping)
    if mapping["contract"] != MEMBER_STATE_GRAPH_CONTRACT:
        raise MemberStateArtifactError("member-state artifact contract mismatch")
    if (
        isinstance(mapping["schema_version"], bool)
        or not isinstance(mapping["schema_version"], int)
        or mapping["schema_version"] != MEMBER_STATE_GRAPH_SCHEMA_VERSION
    ):
        raise MemberStateArtifactError("member-state artifact schema mismatch")
    if mapping["content_hash_basis"] != _CONTENT_HASH_BASIS:
        raise MemberStateArtifactError("member-state content hash basis mismatch")
    claimed_digest = mapping["content_sha256"]
    if not isinstance(claimed_digest, str) or _SHA256.fullmatch(claimed_digest) is None:
        raise MemberStateArtifactError("member-state content digest is invalid")
    try:
        actual_digest = _content_sha256(mapping)
    except (TypeError, ValueError) as exc:
        raise MemberStateArtifactError(
            "member-state artifact is not canonical finite JSON"
        ) from exc
    if not hmac.compare_digest(claimed_digest, actual_digest):
        raise MemberStateArtifactError("member-state artifact checksum mismatch")
    if artifact_path.name != f"{claimed_digest}.json":
        raise MemberStateArtifactError("artifact filename is not content addressed")
    if expected_content_sha256 is not None:
        if (
            not isinstance(expected_content_sha256, str)
            or _SHA256.fullmatch(expected_content_sha256) is None
            or not hmac.compare_digest(expected_content_sha256, claimed_digest)
        ):
            raise MemberStateArtifactError("artifact does not match expected SHA-256")
    privacy = _strict_mapping(mapping["privacy"], "artifact privacy declaration")
    expected_privacy = {
        "raw_frames_persisted": False,
        "compressed_frames_persisted": False,
        "face_identity_embeddings_persisted": False,
        "body_appearance_embeddings_persisted": False,
    }
    _strict_keys(privacy, expected_privacy, "artifact privacy declaration")
    if any(privacy[key] is not False for key in expected_privacy):
        raise MemberStateArtifactError("artifact privacy declaration is not fail-closed")
    graph = MemberStateGraph.from_dict(mapping["graph"])
    provenance = MemberStateArtifactProvenance.from_dict(mapping["provenance"])
    provenance.validate_for_graph(graph)
    return MemberStateGraphArtifact(
        path=artifact_path,
        content_sha256=claimed_digest,
        graph=graph,
        provenance=provenance,
    )


__all__ = [
    "CausalMemberObservation",
    "CorroboratedIdentityEvidence",
    "IdentityEvidenceKind",
    "MEMBER_STATE_GRAPH_CONTRACT",
    "MEMBER_STATE_GRAPH_SCHEMA_VERSION",
    "MemberProbabilityMass",
    "MemberStateArtifactError",
    "MemberStateArtifactProvenance",
    "MemberStateEdge",
    "MemberStateEdgeKind",
    "MemberStateFeatureAdapter",
    "MemberStateGraph",
    "MemberStateGraphArtifact",
    "MemberStateGraphConfig",
    "MemberStateGraphError",
    "MemberStateSourceProvenance",
    "MemberTracklet",
    "NormalizedBBox",
    "OBSERVATION_BASIS",
    "ReIdentificationClaim",
    "TargetMemberTimestampFeatures",
    "adapt_target_member_features",
    "build_member_state_graph",
    "load_member_state_graph_artifact",
    "write_member_state_graph_artifact",
]
