"""Private-local IVE face gallery and conservative open-set classification.

This module is deliberately separated from StageEdit feature snapshots and
general audit reports.  It accepts only typed, metadata-authorized portrait
fancam samples, stores the minimum SFace prototype vectors needed for local
inference in a content-addressed private artifact, and emits only scalar IVE
member probability mass plus an opaque evidence token.

The gallery is not an authentication system.  A query is accepted only when
both its absolute SFace similarity and its lead over the second-best IVE
member pass fixed gates.  Every weak or ambiguous query remains explicitly
unknown; there is no nearest-neighbour hard guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
from statistics import median
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np

from .direct_member_transition import (
    OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT,
)
from .face_identity import (
    SFACE_BACKEND,
    SFACE_EMBEDDING_SIZE,
    SFACE_SHA256,
    cosine_similarity,
    extract_sface_embedding,
    sface_model_asset_status,
)
from .ive import IVEMember
from .member_state_graph import (
    CorroboratedIdentityEvidence,
    IdentityEvidenceKind,
    MemberProbabilityMass,
)
from .visual_score import (
    YUNET_EXPECTED_SHA256,
    describe_frame,
    yunet_model_asset_status,
)


IVE_FACE_IDENTITY_GALLERY_CONTRACT = "ive_private_face_identity_gallery_v1"
IVE_FACE_IDENTITY_GALLERY_SCHEMA_VERSION = 1
PRIVATE_LOCAL_DISTRIBUTION_POLICY = (
    "private_local_only_never_stageedit_snapshot_or_general_audit_v1"
)
PORTRAIT_FANCAM_MEDIA_CONTRACT = "automatic_member_portrait_9_16_media_v1"
CLOSEUP_FACECAM_MEDIA_CONTRACT = (
    "official_target_member_closeup_facecam_any_aspect_identity_source_v1"
)
TITLE_MEMBER_AUTHORIZATION_CONTRACT = (
    "verified_official_target_member_title_signal_v1"
)
FACECAM_TITLE_MEMBER_AUTHORIZATION_CONTRACT = (
    "verified_official_target_member_closeup_facecam_title_signal_v1"
)
SINGLE_DOMINANT_YUNET_FACE_CONTRACT = (
    "verified_yunet_single_dominant_face_for_private_identity_gallery_v1"
)
IDENTITY_PROVIDER_ID = "ive_private_sface_gallery_v1"

_CONTENT_HASH_BASIS = "sha256_canonical_json_with_content_sha256_null_v1"
_MAXIMUM_ARTIFACT_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VIDEO_ID = re.compile(r"[0-9A-Za-z_-]{6,32}")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_MEMBERS = tuple(IVEMember)

_MEMBER_TITLE_SIGNALS: Mapping[IVEMember, tuple[str, ...]] = MappingProxyType(
    {
        IVEMember.ANYUJIN: ("안유진", "유진", "anyujin", "yujin"),
        IVEMember.GAEUL: ("가을", "gaeul"),
        IVEMember.REI: ("레이", "rei"),
        IVEMember.JANGWONYOUNG: (
            "장원영",
            "원영",
            "jangwonyoung",
            "wonyoung",
        ),
        IVEMember.LIZ: ("리즈", "liz"),
        IVEMember.LEESEO: ("이서", "leeseo"),
    }
)


class IVEFaceGalleryError(ValueError):
    """Invalid gallery inputs or a failed conservative build gate."""


class IVEFaceGalleryArtifactError(IVEFaceGalleryError):
    """Invalid, mutable, or tampered private-local gallery artifact."""


class IVEFaceGallerySourceAuthority(StrEnum):
    """The only two media authorities allowed to seed the private gallery."""

    OFFICIAL_TARGET_MEMBER_PORTRAIT_FANCAM = (
        "official_target_member_portrait_fancam_9_16"
    )
    OFFICIAL_TARGET_MEMBER_CLOSEUP_FACECAM = (
        "official_target_member_closeup_facecam_any_aspect"
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IVEFaceGalleryArtifactError(
                f"gallery JSON repeats key {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise IVEFaceGalleryArtifactError(
        f"gallery JSON contains non-finite constant {value!r}"
    )


def _strict_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise IVEFaceGalleryArtifactError(f"{name} must be an object")
    return value


def _strict_keys(
    value: Mapping[str, Any], expected: Iterable[str], name: str
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        raise IVEFaceGalleryArtifactError(
            f"{name} keys mismatch; missing={sorted(expected_set - actual_set)!r}, "
            f"extra={sorted(actual_set - expected_set)!r}"
        )


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IVEFaceGalleryError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise IVEFaceGalleryError(f"{name} must be finite")
    return numeric


def _probability(value: object, name: str) -> float:
    numeric = _finite(value, name)
    if not 0.0 <= numeric <= 1.0:
        raise IVEFaceGalleryError(f"{name} must be normalized to [0, 1]")
    return numeric


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise IVEFaceGalleryError(f"{name} must be a positive integer")
    return value


def _identifier(value: object, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str):
        raise IVEFaceGalleryError(f"{name} must be a string")
    if not value or value != value.strip() or len(value) > maximum:
        raise IVEFaceGalleryError(f"{name} must be a canonical non-empty string")
    if any(ord(character) < 32 for character in value):
        raise IVEFaceGalleryError(f"{name} cannot contain control characters")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IVEFaceGalleryError(f"{name} must be a lowercase SHA-256")
    return value


def _member(value: object, name: str = "member") -> IVEMember:
    if isinstance(value, IVEMember):
        return value
    if not isinstance(value, str):
        raise IVEFaceGalleryError(f"{name} must be an IVE member")
    try:
        return IVEMember.parse(value)
    except ValueError as exc:
        raise IVEFaceGalleryError(f"{name} must be one of the six IVE members") from exc


def _normalized_embedding(
    value: Sequence[float], *, name: str = "SFace embedding"
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != SFACE_EMBEDDING_SIZE:
        raise IVEFaceGalleryError(
            f"{name} must contain exactly {SFACE_EMBEDDING_SIZE} values"
        )
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (SFACE_EMBEDDING_SIZE,) or not np.all(np.isfinite(array)):
        raise IVEFaceGalleryError(f"{name} must contain finite values")
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise IVEFaceGalleryError(f"{name} must not be empty")
    # Preserve already-normalized serialized vectors byte-for-byte so strict
    # artifact replay cannot drift through repeated harmless normalization.
    normalized = array if abs(norm - 1.0) <= 1e-12 else array / norm
    return tuple(float(item) for item in normalized)


def _mean_normalized(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        raise IVEFaceGalleryError("cannot average an empty vector collection")
    array = np.asarray(vectors, dtype=np.float64)
    mean = np.mean(array, axis=0)
    return _normalized_embedding(mean, name="prototype centroid")


def _similarity(left: Sequence[float], right: Sequence[float]) -> float:
    value = cosine_similarity(left, right)
    if value is None:
        raise IVEFaceGalleryError("SFace similarity evidence is unavailable")
    return value


def _title_has_authorized_signals(title: str, member: IVEMember) -> bool:
    folded = "".join(character for character in title.casefold() if character.isalnum())
    group_signal = "ive" in folded or "아이브" in folded
    fancam_signal = any(
        token in folded
        for token in (
            "직캠",
            "fancam",
            "facecam",
            "fanfacecam",
            "closeup",
            "클로즈업",
        )
    )
    member_signal = any(
        "".join(character for character in token.casefold() if character.isalnum())
        in folded
        for token in _MEMBER_TITLE_SIGNALS[member]
    )
    return group_signal and fancam_signal and member_signal


def _title_has_closeup_facecam_signal(title: str) -> bool:
    folded = "".join(character for character in title.casefold() if character.isalnum())
    return any(
        signal in folded
        for signal in (
            "facecam",
            "facefancam",
            "얼빡직캠",
            "얼굴직캠",
            "클로즈업직캠",
            "closeup",
            "클로즈업",
        )
    )


@dataclass(frozen=True, slots=True)
class OfficialTargetMemberFancamProvenance:
    """Typed source authority; a local filename is intentionally absent."""

    song: str
    target_member: IVEMember | str
    video_id: str
    title: str
    broadcast_date: str
    broadcast_program: str
    channel_id: str
    metadata_authority_sha256: str
    acquisition_manifest_sha256: str
    media_content_sha256: str
    media_size_bytes: int
    media_width: int
    media_height: int
    source_authority: IVEFaceGallerySourceAuthority | str = (
        IVEFaceGallerySourceAuthority.OFFICIAL_TARGET_MEMBER_PORTRAIT_FANCAM
    )
    metadata_contract: str = OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
    source_media_contract: str = PORTRAIT_FANCAM_MEDIA_CONTRACT
    title_member_authorization_contract: str = (
        TITLE_MEMBER_AUTHORIZATION_CONTRACT
    )
    group: str = "IVE"

    def __post_init__(self) -> None:
        if self.group != "IVE":
            raise IVEFaceGalleryError("face gallery accepts IVE sources only")
        object.__setattr__(self, "song", _identifier(self.song, "song"))
        object.__setattr__(
            self, "target_member", _member(self.target_member, "target_member")
        )
        video_id = _identifier(self.video_id, "video_id", maximum=32)
        if _VIDEO_ID.fullmatch(video_id) is None:
            raise IVEFaceGalleryError("video_id is invalid")
        object.__setattr__(self, "video_id", video_id)
        title = _identifier(self.title, "title", maximum=1024)
        if not _title_has_authorized_signals(title, self.target_member):
            raise IVEFaceGalleryError(
                "official metadata title lacks IVE, fancam, or target-member signal"
            )
        object.__setattr__(self, "title", title)
        if not isinstance(self.broadcast_date, str) or _DATE.fullmatch(
            self.broadcast_date
        ) is None:
            raise IVEFaceGalleryError("broadcast_date must use YYYY-MM-DD")
        object.__setattr__(
            self,
            "broadcast_program",
            _identifier(self.broadcast_program, "broadcast_program"),
        )
        object.__setattr__(
            self, "channel_id", _identifier(self.channel_id, "channel_id")
        )
        for field_name in (
            "metadata_authority_sha256",
            "acquisition_manifest_sha256",
            "media_content_sha256",
        ):
            object.__setattr__(
                self, field_name, _sha256(getattr(self, field_name), field_name)
            )
        object.__setattr__(
            self,
            "media_size_bytes",
            _positive_integer(self.media_size_bytes, "media_size_bytes"),
        )
        object.__setattr__(
            self, "media_width", _positive_integer(self.media_width, "media_width")
        )
        object.__setattr__(
            self,
            "media_height",
            _positive_integer(self.media_height, "media_height"),
        )
        if not isinstance(self.source_authority, IVEFaceGallerySourceAuthority):
            try:
                object.__setattr__(
                    self,
                    "source_authority",
                    IVEFaceGallerySourceAuthority(self.source_authority),
                )
            except (TypeError, ValueError) as exc:
                raise IVEFaceGalleryError("source media authority is invalid") from exc
        if self.metadata_contract != OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT:
            raise IVEFaceGalleryError("official target-member metadata contract mismatch")
        if (
            self.source_authority
            is IVEFaceGallerySourceAuthority.OFFICIAL_TARGET_MEMBER_PORTRAIT_FANCAM
        ):
            if self.source_media_contract != PORTRAIT_FANCAM_MEDIA_CONTRACT:
                raise IVEFaceGalleryError("portrait fancam media contract mismatch")
            if (
                self.title_member_authorization_contract
                != TITLE_MEMBER_AUTHORIZATION_CONTRACT
            ):
                raise IVEFaceGalleryError(
                    "portrait title/member authorization contract mismatch"
                )
            aspect_ratio = self.media_width / self.media_height
            if not (
                self.media_height > self.media_width
                and abs(aspect_ratio - (9.0 / 16.0)) <= 0.01
            ):
                raise IVEFaceGalleryError(
                    "portrait fancam authority requires verified 9:16 media"
                )
        else:
            if self.source_media_contract != CLOSEUP_FACECAM_MEDIA_CONTRACT:
                raise IVEFaceGalleryError("close-up FaceCam media contract mismatch")
            if (
                self.title_member_authorization_contract
                != FACECAM_TITLE_MEMBER_AUTHORIZATION_CONTRACT
            ):
                raise IVEFaceGalleryError(
                    "FaceCam title/member authorization contract mismatch"
                )
            if not _title_has_closeup_facecam_signal(self.title):
                raise IVEFaceGalleryError(
                    "close-up source title lacks FaceCam or 얼빡직캠 signal"
                )

    @property
    def provenance_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "song": self.song,
            "target_member": self.target_member.value,
            "video_id": self.video_id,
            "title": self.title,
            "broadcast_date": self.broadcast_date,
            "broadcast_program": self.broadcast_program,
            "channel_id": self.channel_id,
            "source_authority": self.source_authority.value,
            "metadata_contract": self.metadata_contract,
            "source_media_contract": self.source_media_contract,
            "title_member_authorization_contract": (
                self.title_member_authorization_contract
            ),
            "metadata_authority_sha256": self.metadata_authority_sha256,
            "acquisition_manifest_sha256": self.acquisition_manifest_sha256,
            "media_content_sha256": self.media_content_sha256,
            "media_size_bytes": self.media_size_bytes,
            "media_width": self.media_width,
            "media_height": self.media_height,
        }

    @classmethod
    def from_dict(cls, value: object) -> "OfficialTargetMemberFancamProvenance":
        mapping = _strict_mapping(value, "gallery source provenance")
        keys = (
            "group",
            "song",
            "target_member",
            "video_id",
            "title",
            "broadcast_date",
            "broadcast_program",
            "channel_id",
            "source_authority",
            "metadata_contract",
            "source_media_contract",
            "title_member_authorization_contract",
            "metadata_authority_sha256",
            "acquisition_manifest_sha256",
            "media_content_sha256",
            "media_size_bytes",
            "media_width",
            "media_height",
        )
        _strict_keys(mapping, keys, "gallery source provenance")
        try:
            return cls(**{key: mapping[key] for key in keys})
        except (IVEFaceGalleryError, TypeError) as exc:
            raise IVEFaceGalleryArtifactError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class AuthorizedFaceEmbeddingSample:
    """One RAM/local-only SFace sample bound to authorized source metadata."""

    provenance: OfficialTargetMemberFancamProvenance
    timestamp_seconds: float
    frame_index: int
    embedding: tuple[float, ...] | Sequence[float]
    face_observation_sha256: str
    face_detector_backend: str = "opencv_zoo_yunet_2023mar"
    face_detector_model_sha256: str = YUNET_EXPECTED_SHA256
    detected_face_count: int = 1
    dominant_face_contract: str = SINGLE_DOMINANT_YUNET_FACE_CONTRACT

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, OfficialTargetMemberFancamProvenance):
            raise IVEFaceGalleryError("sample provenance must be typed")
        timestamp = _finite(self.timestamp_seconds, "timestamp_seconds")
        if timestamp < 0.0:
            raise IVEFaceGalleryError("timestamp_seconds must be non-negative")
        object.__setattr__(self, "timestamp_seconds", timestamp)
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise IVEFaceGalleryError("frame_index must be a non-negative integer")
        object.__setattr__(
            self, "embedding", _normalized_embedding(self.embedding)
        )
        object.__setattr__(
            self,
            "face_observation_sha256",
            _sha256(self.face_observation_sha256, "face_observation_sha256"),
        )
        if self.face_detector_backend != "opencv_zoo_yunet_2023mar":
            raise IVEFaceGalleryError(
                "gallery sample requires the verified YuNet detector backend"
            )
        object.__setattr__(
            self,
            "face_detector_model_sha256",
            _sha256(
                self.face_detector_model_sha256,
                "face_detector_model_sha256",
            ),
        )
        if self.face_detector_model_sha256 != YUNET_EXPECTED_SHA256:
            raise IVEFaceGalleryError("gallery sample YuNet model identity mismatch")
        if (
            isinstance(self.detected_face_count, bool)
            or not isinstance(self.detected_face_count, int)
            or self.detected_face_count != 1
        ):
            raise IVEFaceGalleryError(
                "gallery sample requires exactly one dominant YuNet face"
            )
        if self.dominant_face_contract != SINGLE_DOMINANT_YUNET_FACE_CONTRACT:
            raise IVEFaceGalleryError("single dominant YuNet face contract mismatch")

    @property
    def sample_sha256(self) -> str:
        return _canonical_sha256(
            {
                "provenance_sha256": self.provenance.provenance_sha256,
                "timestamp_seconds": self.timestamp_seconds,
                "frame_index": self.frame_index,
                "embedding_sha256": _canonical_sha256(list(self.embedding)),
                "face_observation_sha256": self.face_observation_sha256,
                "face_detector_backend": self.face_detector_backend,
                "face_detector_model_sha256": self.face_detector_model_sha256,
                "detected_face_count": self.detected_face_count,
                "dominant_face_contract": self.dominant_face_contract,
                "embedding_backend": SFACE_BACKEND,
            }
        )


def extract_authorized_face_sample(
    frame: np.ndarray,
    *,
    provenance: OfficialTargetMemberFancamProvenance,
    timestamp_seconds: float,
    frame_index: int,
    face_observation_sha256: str,
) -> AuthorizedFaceEmbeddingSample:
    """Use the existing verified YuNet descriptor and SFace embedder APIs.

    Only a single unambiguous YuNet face with five landmarks is accepted.  No
    pixels or descriptor vectors are persisted by this helper.
    """

    yunet = yunet_model_asset_status()
    sface = sface_model_asset_status()
    if not bool(yunet.get("digest_valid")) or not sface.digest_valid:
        raise IVEFaceGalleryError("verified YuNet and SFace model assets are required")
    descriptor = describe_frame(frame, include_subject_geometry=False)
    if not (
        descriptor.face_detector_backend == "yunet"
        and descriptor.face_count == 1
        and descriptor.primary_face_bbox is not None
        and descriptor.primary_face_landmarks_xy is not None
        and len(descriptor.primary_face_landmarks_xy) == 5
    ):
        raise IVEFaceGalleryError(
            "gallery samples require one unambiguous verified YuNet face"
        )
    embedding = extract_sface_embedding(
        frame,
        bbox_xywh=descriptor.primary_face_bbox,
        landmarks_xy=descriptor.primary_face_landmarks_xy,
    )
    if not embedding:
        raise IVEFaceGalleryError("verified SFace embedding extraction failed")
    return AuthorizedFaceEmbeddingSample(
        provenance=provenance,
        timestamp_seconds=timestamp_seconds,
        frame_index=frame_index,
        embedding=embedding,
        face_observation_sha256=face_observation_sha256,
    )


@dataclass(frozen=True, slots=True)
class IVEFaceGalleryConfig:
    minimum_videos_per_member: int = 2
    minimum_samples_per_video: int = 3
    maximum_samples_per_video: int = 12
    maximum_videos_per_member: int = 12
    minimum_temporal_separation_seconds: float = 0.40
    minimum_within_video_similarity: float = 0.45
    minimum_same_member_video_similarity: float = 0.42
    maximum_inter_member_similarity: float = 0.78
    minimum_open_set_similarity: float = 0.48
    minimum_open_set_top_two_margin: float = 0.08
    probability_temperature: float = 0.08

    def __post_init__(self) -> None:
        for name in (
            "minimum_videos_per_member",
            "minimum_samples_per_video",
            "maximum_samples_per_video",
            "maximum_videos_per_member",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name)
            )
        if self.maximum_samples_per_video < self.minimum_samples_per_video:
            raise IVEFaceGalleryError(
                "maximum_samples_per_video cannot be below its minimum"
            )
        if self.maximum_videos_per_member < self.minimum_videos_per_member:
            raise IVEFaceGalleryError(
                "maximum_videos_per_member cannot be below its minimum"
            )
        separation = _finite(
            self.minimum_temporal_separation_seconds,
            "minimum_temporal_separation_seconds",
        )
        if separation <= 0.0:
            raise IVEFaceGalleryError(
                "minimum_temporal_separation_seconds must be positive"
            )
        object.__setattr__(
            self, "minimum_temporal_separation_seconds", separation
        )
        for name in (
            "minimum_within_video_similarity",
            "minimum_same_member_video_similarity",
            "maximum_inter_member_similarity",
            "minimum_open_set_similarity",
            "minimum_open_set_top_two_margin",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        temperature = _finite(self.probability_temperature, "probability_temperature")
        if temperature <= 0.0:
            raise IVEFaceGalleryError("probability_temperature must be positive")
        object.__setattr__(self, "probability_temperature", temperature)
        if self.maximum_inter_member_similarity <= self.minimum_open_set_similarity:
            raise IVEFaceGalleryError(
                "inter-member ambiguity ceiling must exceed open-set similarity gate"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> "IVEFaceGalleryConfig":
        mapping = _strict_mapping(value, "gallery config")
        keys = tuple(cls.__dataclass_fields__)
        _strict_keys(mapping, keys, "gallery config")
        try:
            return cls(**{key: mapping[key] for key in keys})
        except (IVEFaceGalleryError, TypeError) as exc:
            raise IVEFaceGalleryArtifactError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class IVEFacePrototype:
    member: IVEMember | str
    prototype_id: str
    source_provenance_sha256: str
    source_media_sha256: str
    source_video_id_sha256: str
    contributing_sample_sha256: tuple[str, ...]
    embedding: tuple[float, ...] | Sequence[float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "member", _member(self.member))
        object.__setattr__(
            self, "prototype_id", _sha256(self.prototype_id, "prototype_id")
        )
        for name in (
            "source_provenance_sha256",
            "source_media_sha256",
            "source_video_id_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        samples = tuple(self.contributing_sample_sha256)
        if not samples or len(samples) != len(set(samples)):
            raise IVEFaceGalleryError(
                "prototype needs unique contributing sample digests"
            )
        for digest in samples:
            _sha256(digest, "contributing sample SHA-256")
        object.__setattr__(self, "contributing_sample_sha256", tuple(sorted(samples)))
        object.__setattr__(
            self, "embedding", _normalized_embedding(self.embedding, name="prototype")
        )
        expected_id = _canonical_sha256(
            {
                "member": self.member.value,
                "source_provenance_sha256": self.source_provenance_sha256,
                "source_media_sha256": self.source_media_sha256,
                "source_video_id_sha256": self.source_video_id_sha256,
                "contributing_sample_sha256": list(
                    self.contributing_sample_sha256
                ),
                "embedding": list(self.embedding),
            }
        )
        if not hmac.compare_digest(self.prototype_id, expected_id):
            raise IVEFaceGalleryError(
                "prototype ID does not reproduce from its bound content"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "member": self.member.value,
            "prototype_id": self.prototype_id,
            "source_provenance_sha256": self.source_provenance_sha256,
            "source_media_sha256": self.source_media_sha256,
            "source_video_id_sha256": self.source_video_id_sha256,
            "contributing_sample_sha256": list(
                self.contributing_sample_sha256
            ),
            "embedding": list(self.embedding),
        }

    @classmethod
    def from_dict(cls, value: object) -> "IVEFacePrototype":
        mapping = _strict_mapping(value, "face prototype")
        keys = (
            "member",
            "prototype_id",
            "source_provenance_sha256",
            "source_media_sha256",
            "source_video_id_sha256",
            "contributing_sample_sha256",
            "embedding",
        )
        _strict_keys(mapping, keys, "face prototype")
        contributing = mapping["contributing_sample_sha256"]
        if not isinstance(contributing, list):
            raise IVEFaceGalleryArtifactError(
                "prototype contributing samples must be an array"
            )
        try:
            return cls(
                member=mapping["member"],
                prototype_id=mapping["prototype_id"],
                source_provenance_sha256=mapping["source_provenance_sha256"],
                source_media_sha256=mapping["source_media_sha256"],
                source_video_id_sha256=mapping["source_video_id_sha256"],
                contributing_sample_sha256=tuple(contributing),
                embedding=mapping["embedding"],
            )
        except (IVEFaceGalleryError, TypeError) as exc:
            raise IVEFaceGalleryArtifactError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class IVEFaceGalleryRejection:
    member: IVEMember | str
    source_provenance_sha256: str
    sample_sha256: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "member", _member(self.member))
        object.__setattr__(
            self,
            "source_provenance_sha256",
            _sha256(self.source_provenance_sha256, "source_provenance_sha256"),
        )
        object.__setattr__(
            self, "sample_sha256", _sha256(self.sample_sha256, "sample_sha256")
        )
        object.__setattr__(self, "reason", _identifier(self.reason, "reason"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "member": self.member.value,
            "source_provenance_sha256": self.source_provenance_sha256,
            "sample_sha256": self.sample_sha256,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: object) -> "IVEFaceGalleryRejection":
        mapping = _strict_mapping(value, "gallery rejection")
        keys = (
            "member",
            "source_provenance_sha256",
            "sample_sha256",
            "reason",
        )
        _strict_keys(mapping, keys, "gallery rejection")
        try:
            return cls(**{key: mapping[key] for key in keys})
        except (IVEFaceGalleryError, TypeError) as exc:
            raise IVEFaceGalleryArtifactError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class IVEFaceIdentityGallery:
    config: IVEFaceGalleryConfig
    sources: tuple[OfficialTargetMemberFancamProvenance, ...]
    prototypes: tuple[IVEFacePrototype, ...]
    rejections: tuple[IVEFaceGalleryRejection, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.config, IVEFaceGalleryConfig):
            raise IVEFaceGalleryError("gallery config must be typed")
        sources = tuple(
            sorted(self.sources, key=lambda item: item.provenance_sha256)
        )
        if not sources or not all(
            isinstance(item, OfficialTargetMemberFancamProvenance)
            for item in sources
        ):
            raise IVEFaceGalleryError("gallery sources must be typed")
        source_by_digest = {item.provenance_sha256: item for item in sources}
        if len(source_by_digest) != len(sources):
            raise IVEFaceGalleryError("gallery source provenance is duplicated")
        media_labels: dict[str, IVEMember] = {}
        video_labels: dict[str, IVEMember] = {}
        for source in sources:
            for key, labels in (
                (source.media_content_sha256, media_labels),
                (source.video_id, video_labels),
            ):
                previous = labels.get(key)
                if previous is not None and previous is not source.target_member:
                    raise IVEFaceGalleryError(
                        "one source cannot be authorized for two IVE members"
                    )
                labels[key] = source.target_member
        prototypes = tuple(
            sorted(
                self.prototypes,
                key=lambda item: (_MEMBERS.index(item.member), item.prototype_id),
            )
        )
        if not prototypes or not all(
            isinstance(item, IVEFacePrototype) for item in prototypes
        ):
            raise IVEFaceGalleryError("gallery prototypes must be typed")
        if len({item.prototype_id for item in prototypes}) != len(prototypes):
            raise IVEFaceGalleryError("gallery prototype IDs are duplicated")
        counts: dict[IVEMember, set[str]] = {member: set() for member in _MEMBERS}
        for prototype in prototypes:
            source = source_by_digest.get(prototype.source_provenance_sha256)
            if source is None or source.target_member is not prototype.member:
                raise IVEFaceGalleryError(
                    "prototype source does not match its IVE member"
                )
            if source.media_content_sha256 != prototype.source_media_sha256:
                raise IVEFaceGalleryError("prototype media binding mismatch")
            expected_video_digest = hashlib.sha256(
                source.video_id.encode("utf-8")
            ).hexdigest()
            if expected_video_digest != prototype.source_video_id_sha256:
                raise IVEFaceGalleryError("prototype video binding mismatch")
            counts[prototype.member].add(prototype.source_media_sha256)
        missing = [
            member.value
            for member in _MEMBERS
            if len(counts[member]) < self.config.minimum_videos_per_member
        ]
        if missing:
            raise IVEFaceGalleryError(
                f"gallery lacks independent videos for IVE members: {missing!r}"
            )
        for index, left in enumerate(prototypes):
            for right in prototypes[index + 1 :]:
                if left.member is right.member:
                    continue
                if _similarity(left.embedding, right.embedding) >= (
                    self.config.maximum_inter_member_similarity
                ):
                    raise IVEFaceGalleryError(
                        "inter-member SFace prototypes are ambiguous"
                    )
        rejections = tuple(
            sorted(
                self.rejections,
                key=lambda item: (
                    _MEMBERS.index(item.member),
                    item.source_provenance_sha256,
                    item.sample_sha256,
                    item.reason,
                ),
            )
        )
        if not all(isinstance(item, IVEFaceGalleryRejection) for item in rejections):
            raise IVEFaceGalleryError("gallery rejection audit must be typed")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "prototypes", prototypes)
        object.__setattr__(self, "rejections", rejections)

    @property
    def fingerprint_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def prototypes_for(self, member: IVEMember | str) -> tuple[IVEFacePrototype, ...]:
        target = _member(member)
        return tuple(item for item in self.prototypes if item.member is target)

    def to_dict(self) -> dict[str, Any]:
        prototype_counts = {
            member.value: len(self.prototypes_for(member)) for member in _MEMBERS
        }
        video_counts = {
            member.value: len(
                {
                    item.source_media_sha256
                    for item in self.prototypes_for(member)
                }
            )
            for member in _MEMBERS
        }
        authority_counts = {
            authority.value: sum(
                source.source_authority is authority for source in self.sources
            )
            for authority in IVEFaceGallerySourceAuthority
        }
        return {
            "scope": {
                "group": "IVE",
                "members": [member.value for member in _MEMBERS],
                "open_set_unknown_required": True,
                "accepted_source_authorities": [
                    authority.value for authority in IVEFaceGallerySourceAuthority
                ],
            },
            "models": {
                "face_detector": "opencv_zoo_yunet_2023mar",
                "face_detector_model_sha256": YUNET_EXPECTED_SHA256,
                "face_embedding": SFACE_BACKEND,
                "face_embedding_model_sha256": SFACE_SHA256,
                "embedding_dimensions": SFACE_EMBEDDING_SIZE,
            },
            "config": self.config.to_dict(),
            "sources": [item.to_dict() for item in self.sources],
            "prototypes": [item.to_dict() for item in self.prototypes],
            "rejections": [item.to_dict() for item in self.rejections],
            "summary": {
                "prototype_count_by_member": prototype_counts,
                "independent_video_count_by_member": video_counts,
                "source_authority_count": authority_counts,
                "rejection_count": len(self.rejections),
                "video_equal_weighting_at_inference": True,
            },
        }

    @classmethod
    def from_dict(cls, value: object) -> "IVEFaceIdentityGallery":
        mapping = _strict_mapping(value, "face identity gallery")
        keys = (
            "scope",
            "models",
            "config",
            "sources",
            "prototypes",
            "rejections",
            "summary",
        )
        _strict_keys(mapping, keys, "face identity gallery")
        scope = _strict_mapping(mapping["scope"], "gallery scope")
        _strict_keys(
            scope,
            (
                "group",
                "members",
                "open_set_unknown_required",
                "accepted_source_authorities",
            ),
            "gallery scope",
        )
        if scope != {
            "group": "IVE",
            "members": [member.value for member in _MEMBERS],
            "open_set_unknown_required": True,
            "accepted_source_authorities": [
                authority.value for authority in IVEFaceGallerySourceAuthority
            ],
        }:
            raise IVEFaceGalleryArtifactError("gallery IVE-only scope mismatch")
        models = _strict_mapping(mapping["models"], "gallery models")
        expected_models = {
            "face_detector": "opencv_zoo_yunet_2023mar",
            "face_detector_model_sha256": YUNET_EXPECTED_SHA256,
            "face_embedding": SFACE_BACKEND,
            "face_embedding_model_sha256": SFACE_SHA256,
            "embedding_dimensions": SFACE_EMBEDDING_SIZE,
        }
        _strict_keys(models, expected_models, "gallery models")
        if dict(models) != expected_models:
            raise IVEFaceGalleryArtifactError("gallery model identity mismatch")
        for name in ("sources", "prototypes", "rejections"):
            if not isinstance(mapping[name], list):
                raise IVEFaceGalleryArtifactError(f"gallery {name} must be an array")
        try:
            gallery = cls(
                config=IVEFaceGalleryConfig.from_dict(mapping["config"]),
                sources=tuple(
                    OfficialTargetMemberFancamProvenance.from_dict(item)
                    for item in mapping["sources"]
                ),
                prototypes=tuple(
                    IVEFacePrototype.from_dict(item)
                    for item in mapping["prototypes"]
                ),
                rejections=tuple(
                    IVEFaceGalleryRejection.from_dict(item)
                    for item in mapping["rejections"]
                ),
            )
        except IVEFaceGalleryArtifactError:
            raise
        except IVEFaceGalleryError as exc:
            raise IVEFaceGalleryArtifactError(str(exc)) from exc
        if mapping["summary"] != gallery.to_dict()["summary"]:
            raise IVEFaceGalleryArtifactError("gallery summary is not reproducible")
        return gallery


def _sample_sort_key(sample: AuthorizedFaceEmbeddingSample) -> tuple[Any, ...]:
    return (
        sample.timestamp_seconds,
        sample.frame_index,
        sample.sample_sha256,
    )


def _evenly_bounded_samples(
    samples: Sequence[AuthorizedFaceEmbeddingSample], maximum: int
) -> tuple[AuthorizedFaceEmbeddingSample, ...]:
    if len(samples) <= maximum:
        return tuple(samples)
    if maximum == 1:
        return (samples[len(samples) // 2],)
    indexes = tuple(
        round(index * (len(samples) - 1) / (maximum - 1))
        for index in range(maximum)
    )
    return tuple(samples[index] for index in indexes)


def _robust_medoid_index(vectors: Sequence[Sequence[float]]) -> int:
    if not vectors:
        raise IVEFaceGalleryError("cannot select a medoid from no vectors")
    scored: list[tuple[float, float, int]] = []
    for index, vector in enumerate(vectors):
        similarities = [
            _similarity(vector, other)
            for other_index, other in enumerate(vectors)
            if other_index != index
        ]
        if not similarities:
            scored.append((1.0, 1.0, index))
        else:
            scored.append((median(similarities), math.fsum(similarities), index))
    return max(scored, key=lambda item: (item[0], item[1], -item[2]))[2]


def build_ive_face_identity_gallery(
    samples: Sequence[AuthorizedFaceEmbeddingSample],
    *,
    config: IVEFaceGalleryConfig | None = None,
) -> IVEFaceIdentityGallery:
    """Build equal-video-weighted, outlier-filtered IVE prototypes."""

    policy = config or IVEFaceGalleryConfig()
    provided = tuple(samples)
    if not provided or not all(
        isinstance(item, AuthorizedFaceEmbeddingSample) for item in provided
    ):
        raise IVEFaceGalleryError("gallery needs typed authorized face samples")
    sample_digests = [item.sample_sha256 for item in provided]
    if len(sample_digests) != len(set(sample_digests)):
        raise IVEFaceGalleryError("gallery sample content is duplicated")

    sources = {
        item.provenance.provenance_sha256: item.provenance for item in provided
    }
    media_labels: dict[str, IVEMember] = {}
    video_labels: dict[str, IVEMember] = {}
    for item in provided:
        provenance = item.provenance
        for key, labels in (
            (provenance.media_content_sha256, media_labels),
            (provenance.video_id, video_labels),
        ):
            previous = labels.get(key)
            if previous is not None and previous is not provenance.target_member:
                raise IVEFaceGalleryError(
                    "one authorized source is labelled as multiple IVE members"
                )
            labels[key] = provenance.target_member

    rejections: list[IVEFaceGalleryRejection] = []
    candidate_centroids: dict[
        IVEMember,
        list[
            tuple[
                OfficialTargetMemberFancamProvenance,
                tuple[float, ...],
                tuple[AuthorizedFaceEmbeddingSample, ...],
            ]
        ],
    ] = {member: [] for member in _MEMBERS}

    for member in _MEMBERS:
        member_samples = [
            item for item in provided if item.provenance.target_member is member
        ]
        by_source: dict[str, list[AuthorizedFaceEmbeddingSample]] = {}
        for item in member_samples:
            by_source.setdefault(item.provenance.provenance_sha256, []).append(item)
        if len(by_source) < policy.minimum_videos_per_member:
            raise IVEFaceGalleryError(
                f"{member.value} needs at least {policy.minimum_videos_per_member} "
                "independent authorized videos"
            )
        selected_source_digests = tuple(sorted(by_source))[
            : policy.maximum_videos_per_member
        ]
        for source_digest in sorted(set(by_source) - set(selected_source_digests)):
            for sample in by_source[source_digest]:
                rejections.append(
                    IVEFaceGalleryRejection(
                        member,
                        source_digest,
                        sample.sample_sha256,
                        "member_video_cap",
                    )
                )

        for source_digest in selected_source_digests:
            ordered = sorted(by_source[source_digest], key=_sample_sort_key)
            temporally_spaced: list[AuthorizedFaceEmbeddingSample] = []
            for sample in ordered:
                if (
                    temporally_spaced
                    and sample.timestamp_seconds
                    - temporally_spaced[-1].timestamp_seconds
                    < policy.minimum_temporal_separation_seconds
                ):
                    rejections.append(
                        IVEFaceGalleryRejection(
                            member,
                            source_digest,
                            sample.sample_sha256,
                            "temporal_separation_cap",
                        )
                    )
                    continue
                temporally_spaced.append(sample)
            bounded = _evenly_bounded_samples(
                temporally_spaced, policy.maximum_samples_per_video
            )
            bounded_hashes = {item.sample_sha256 for item in bounded}
            for sample in temporally_spaced:
                if sample.sample_sha256 not in bounded_hashes:
                    rejections.append(
                        IVEFaceGalleryRejection(
                            member,
                            source_digest,
                            sample.sample_sha256,
                            "per_video_sample_cap",
                        )
                    )
            if len(bounded) < policy.minimum_samples_per_video:
                for sample in bounded:
                    rejections.append(
                        IVEFaceGalleryRejection(
                            member,
                            source_digest,
                            sample.sample_sha256,
                            "insufficient_temporally_independent_samples",
                        )
                    )
                continue
            medoid = bounded[_robust_medoid_index([item.embedding for item in bounded])]
            inliers: list[AuthorizedFaceEmbeddingSample] = []
            for sample in bounded:
                if _similarity(sample.embedding, medoid.embedding) < (
                    policy.minimum_within_video_similarity
                ):
                    rejections.append(
                        IVEFaceGalleryRejection(
                            member,
                            source_digest,
                            sample.sample_sha256,
                            "within_video_identity_outlier",
                        )
                    )
                else:
                    inliers.append(sample)
            if len(inliers) < policy.minimum_samples_per_video:
                for sample in inliers:
                    rejections.append(
                        IVEFaceGalleryRejection(
                            member,
                            source_digest,
                            sample.sample_sha256,
                            "insufficient_inliers_after_outlier_rejection",
                        )
                    )
                continue
            candidate_centroids[member].append(
                (
                    inliers[0].provenance,
                    _mean_normalized([item.embedding for item in inliers]),
                    tuple(inliers),
                )
            )

    accepted_centroids: dict[
        IVEMember,
        list[
            tuple[
                OfficialTargetMemberFancamProvenance,
                tuple[float, ...],
                tuple[AuthorizedFaceEmbeddingSample, ...],
            ]
        ],
    ] = {member: [] for member in _MEMBERS}
    for member in _MEMBERS:
        candidates = candidate_centroids[member]
        if len(candidates) < policy.minimum_videos_per_member:
            raise IVEFaceGalleryError(
                f"{member.value} lacks enough valid per-video prototypes"
            )
        medoid_index = _robust_medoid_index([item[1] for item in candidates])
        medoid_vector = candidates[medoid_index][1]
        for provenance, vector, inliers in candidates:
            if _similarity(vector, medoid_vector) < (
                policy.minimum_same_member_video_similarity
            ):
                for sample in inliers:
                    rejections.append(
                        IVEFaceGalleryRejection(
                            member,
                            provenance.provenance_sha256,
                            sample.sample_sha256,
                            "same_member_video_outlier",
                        )
                    )
            else:
                accepted_centroids[member].append(
                    (provenance, vector, inliers)
                )
        if len(accepted_centroids[member]) < policy.minimum_videos_per_member:
            raise IVEFaceGalleryError(
                f"{member.value} same-member video cluster is not robust"
            )

    for left_index, left_member in enumerate(_MEMBERS):
        for right_member in _MEMBERS[left_index + 1 :]:
            for _left_source, left_vector, _left_samples in accepted_centroids[
                left_member
            ]:
                for _right_source, right_vector, _right_samples in accepted_centroids[
                    right_member
                ]:
                    if _similarity(left_vector, right_vector) >= (
                        policy.maximum_inter_member_similarity
                    ):
                        raise IVEFaceGalleryError(
                            "inter-member ambiguity gate rejected the gallery"
                        )

    prototypes: list[IVEFacePrototype] = []
    for member in _MEMBERS:
        for provenance, vector, inliers in accepted_centroids[member]:
            unsigned = {
                "member": member.value,
                "source_provenance_sha256": provenance.provenance_sha256,
                "source_media_sha256": provenance.media_content_sha256,
                "source_video_id_sha256": hashlib.sha256(
                    provenance.video_id.encode("utf-8")
                ).hexdigest(),
                "contributing_sample_sha256": sorted(
                    sample.sample_sha256 for sample in inliers
                ),
                "embedding": list(vector),
            }
            prototypes.append(
                IVEFacePrototype(
                    member=member,
                    prototype_id=_canonical_sha256(unsigned),
                    source_provenance_sha256=provenance.provenance_sha256,
                    source_media_sha256=provenance.media_content_sha256,
                    source_video_id_sha256=unsigned["source_video_id_sha256"],
                    contributing_sample_sha256=tuple(
                        unsigned["contributing_sample_sha256"]
                    ),
                    embedding=vector,
                )
            )
    return IVEFaceIdentityGallery(
        config=policy,
        sources=tuple(sources.values()),
        prototypes=tuple(prototypes),
        rejections=tuple(rejections),
    )


@dataclass(frozen=True, slots=True)
class OpenSetIVEIdentityResult:
    member_probabilities: MemberProbabilityMass
    accepted_member: IVEMember | None
    opaque_identity_token: str | None
    evidence_sha256: str
    member_similarity: tuple[float, ...]
    top_similarity: float
    top_two_margin: float
    decision_reason: str
    gallery_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.member_probabilities, MemberProbabilityMass):
            raise IVEFaceGalleryError("identity result probability mass must be typed")
        if self.accepted_member is not None and not isinstance(
            self.accepted_member, IVEMember
        ):
            object.__setattr__(self, "accepted_member", _member(self.accepted_member))
        if len(self.member_similarity) != len(_MEMBERS):
            raise IVEFaceGalleryError("identity result needs six member similarities")
        similarities = tuple(
            _finite(item, "member similarity") for item in self.member_similarity
        )
        if any(not -1.0 <= item <= 1.0 for item in similarities):
            raise IVEFaceGalleryError("member similarity must be in [-1, 1]")
        object.__setattr__(self, "member_similarity", similarities)
        object.__setattr__(
            self, "top_similarity", _finite(self.top_similarity, "top_similarity")
        )
        object.__setattr__(
            self, "top_two_margin", _finite(self.top_two_margin, "top_two_margin")
        )
        object.__setattr__(
            self, "evidence_sha256", _sha256(self.evidence_sha256, "evidence_sha256")
        )
        object.__setattr__(
            self,
            "gallery_fingerprint_sha256",
            _sha256(
                self.gallery_fingerprint_sha256,
                "gallery_fingerprint_sha256",
            ),
        )
        object.__setattr__(
            self, "decision_reason", _identifier(self.decision_reason, "decision_reason")
        )
        if self.accepted_member is None:
            if self.opaque_identity_token is not None:
                raise IVEFaceGalleryError("unknown identity cannot carry an identity token")
            if not math.isclose(
                self.member_probabilities.unknown_mass, 1.0, abs_tol=1e-12
            ):
                raise IVEFaceGalleryError("rejected identity must remain fully unknown")
        else:
            token = _identifier(self.opaque_identity_token, "opaque_identity_token")
            if not token.startswith("iveface:"):
                raise IVEFaceGalleryError("identity token must stay opaque")
            object.__setattr__(self, "opaque_identity_token", token)

    def to_scalar_dict(self) -> dict[str, Any]:
        """Safe member-state/audit form; no prototype or query vector."""

        return {
            "member_probabilities": self.member_probabilities.to_dict(),
            "accepted_member": (
                self.accepted_member.value if self.accepted_member is not None else None
            ),
            "opaque_identity_token": self.opaque_identity_token,
            "evidence_sha256": self.evidence_sha256,
            "member_similarity": [
                {"member": member.value, "similarity": similarity}
                for member, similarity in zip(
                    _MEMBERS, self.member_similarity, strict=True
                )
            ],
            "top_similarity": self.top_similarity,
            "top_two_margin": self.top_two_margin,
            "decision_reason": self.decision_reason,
            "gallery_fingerprint_sha256": self.gallery_fingerprint_sha256,
            "raw_query_embedding_persisted": False,
            "gallery_prototype_embeddings_persisted": False,
        }


def classify_open_set_ive_identity(
    gallery: IVEFaceIdentityGallery,
    query_embedding: Sequence[float],
) -> OpenSetIVEIdentityResult:
    """Classify one face without ever forcing a nearest IVE member."""

    if not isinstance(gallery, IVEFaceIdentityGallery):
        raise IVEFaceGalleryError("gallery must be IVEFaceIdentityGallery")
    query = _normalized_embedding(query_embedding, name="query SFace embedding")
    scores: list[float] = []
    for member in _MEMBERS:
        # One centroid is retained per source video.  Averaging those video
        # scores means a heavily sampled fancam can never dominate a member.
        per_video: dict[str, float] = {}
        for prototype in gallery.prototypes_for(member):
            similarity = _similarity(query, prototype.embedding)
            current = per_video.get(prototype.source_media_sha256)
            if current is None or similarity > current:
                per_video[prototype.source_media_sha256] = similarity
        if not per_video:
            raise IVEFaceGalleryError("gallery member has no prototypes")
        scores.append(math.fsum(per_video.values()) / len(per_video))
    ranked = sorted(
        range(len(_MEMBERS)), key=lambda index: (-scores[index], index)
    )
    top_index, runner_up_index = ranked[:2]
    top_similarity = scores[top_index]
    margin = top_similarity - scores[runner_up_index]
    if top_similarity < gallery.config.minimum_open_set_similarity:
        accepted: IVEMember | None = None
        reason = "below_absolute_similarity_gate"
    elif margin < gallery.config.minimum_open_set_top_two_margin:
        accepted = None
        reason = "below_top_two_margin_gate"
    else:
        accepted = _MEMBERS[top_index]
        reason = "accepted_similarity_and_margin"

    gallery_fingerprint = gallery.fingerprint_sha256
    query_sha256 = _canonical_sha256(list(query))
    evidence_sha256 = _canonical_sha256(
        {
            "provider_id": IDENTITY_PROVIDER_ID,
            "gallery_fingerprint_sha256": gallery_fingerprint,
            "query_embedding_sha256": query_sha256,
            "member_similarity": [
                {"member": member.value, "similarity": score}
                for member, score in zip(_MEMBERS, scores, strict=True)
            ],
            "decision_reason": reason,
        }
    )
    if accepted is None:
        probability_mass = MemberProbabilityMass.unknown()
        token = None
    else:
        similarity_strength = min(
            1.0,
            max(
                0.0,
                (top_similarity - gallery.config.minimum_open_set_similarity)
                / (1.0 - gallery.config.minimum_open_set_similarity),
            ),
        )
        margin_strength = min(
            1.0,
            max(
                0.0,
                (margin - gallery.config.minimum_open_set_top_two_margin)
                / (1.0 - gallery.config.minimum_open_set_top_two_margin),
            ),
        )
        combined_strength = math.sqrt(similarity_strength * margin_strength)
        known_mass = 0.82 + 0.16 * combined_strength
        top_share = 0.90 + 0.08 * margin_strength
        probabilities = {member: 0.0 for member in _MEMBERS}
        probabilities[accepted] = known_mass * top_share
        others = [index for index in range(len(_MEMBERS)) if index != top_index]
        maximum_other = max(scores[index] for index in others)
        weights = [
            math.exp(
                (scores[index] - maximum_other)
                / gallery.config.probability_temperature
            )
            for index in others
        ]
        weight_total = math.fsum(weights)
        residual = known_mass - probabilities[accepted]
        for index, weight in zip(others, weights, strict=True):
            probabilities[_MEMBERS[index]] = residual * weight / weight_total
        probability_mass = MemberProbabilityMass(
            probabilities, unknown_mass=1.0 - known_mass
        )
        token = f"iveface:{evidence_sha256[:32]}"
    return OpenSetIVEIdentityResult(
        member_probabilities=probability_mass,
        accepted_member=accepted,
        opaque_identity_token=token,
        evidence_sha256=evidence_sha256,
        member_similarity=tuple(scores),
        top_similarity=top_similarity,
        top_two_margin=margin,
        decision_reason=reason,
        gallery_fingerprint_sha256=gallery_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class MemberStateIdentityAdapter:
    """Only the scalar/opaque values allowed into ``member_state_graph``."""

    member_probabilities: MemberProbabilityMass
    face_identity_id: str | None
    evidence: CorroboratedIdentityEvidence | None


def adapt_gallery_identity_for_member_state(
    result: OpenSetIVEIdentityResult,
) -> MemberStateIdentityAdapter:
    if not isinstance(result, OpenSetIVEIdentityResult):
        raise IVEFaceGalleryError("identity result must be typed")
    evidence = (
        CorroboratedIdentityEvidence(
            kind=IdentityEvidenceKind.FACE_IDENTITY_MATCH,
            confidence=result.member_probabilities.probability_for(
                result.accepted_member
            ),
            provider_id=IDENTITY_PROVIDER_ID,
            evidence_sha256=result.evidence_sha256,
        )
        if result.accepted_member is not None
        else None
    )
    return MemberStateIdentityAdapter(
        member_probabilities=result.member_probabilities,
        face_identity_id=result.opaque_identity_token,
        evidence=evidence,
    )


@dataclass(frozen=True, slots=True)
class IVEFaceGalleryArtifact:
    path: Path
    content_sha256: str
    gallery: IVEFaceIdentityGallery

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(
            self, "content_sha256", _sha256(self.content_sha256, "content_sha256")
        )
        if not isinstance(self.gallery, IVEFaceIdentityGallery):
            raise IVEFaceGalleryArtifactError("artifact gallery must be typed")


def _artifact_content_sha256(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized["content_sha256"] = None
    return _canonical_sha256(normalized)


def _artifact_payload(
    gallery: IVEFaceIdentityGallery,
) -> tuple[dict[str, Any], str, bytes]:
    if not isinstance(gallery, IVEFaceIdentityGallery):
        raise IVEFaceGalleryArtifactError("gallery must be typed")
    payload: dict[str, Any] = {
        "contract": IVE_FACE_IDENTITY_GALLERY_CONTRACT,
        "schema_version": IVE_FACE_IDENTITY_GALLERY_SCHEMA_VERSION,
        "content_hash_basis": _CONTENT_HASH_BASIS,
        "content_sha256": None,
        "distribution_policy": PRIVATE_LOCAL_DISTRIBUTION_POLICY,
        "privacy": {
            "private_local_only": True,
            "contains_sface_prototype_vectors": True,
            "raw_frames_persisted": False,
            "source_media_paths_persisted": False,
            "raw_query_embeddings_persisted": False,
            "stageedit_snapshot_export_permitted": False,
            "general_audit_json_export_permitted": False,
        },
        "gallery": gallery.to_dict(),
    }
    digest = _artifact_content_sha256(payload)
    payload["content_sha256"] = digest
    encoded = _canonical_json_bytes(payload)
    if not 1 <= len(encoded) <= _MAXIMUM_ARTIFACT_BYTES:
        raise IVEFaceGalleryArtifactError("gallery artifact size is invalid")
    return (payload, digest, encoded)


def _safe_directory(parent: Path, name: str) -> Path:
    child = parent / name
    if child.is_symlink():
        raise IVEFaceGalleryArtifactError(
            f"gallery artifact directory is a symlink: {child}"
        )
    child.mkdir(exist_ok=True)
    if child.is_symlink() or not child.is_dir():
        raise IVEFaceGalleryArtifactError(
            f"gallery artifact path is not a safe directory: {child}"
        )
    return child


def write_ive_face_identity_gallery_artifact(
    artifact_directory: Path,
    gallery: IVEFaceIdentityGallery,
) -> IVEFaceGalleryArtifact:
    """Publish immutable private-local gallery bytes by their content hash."""

    _payload, digest, encoded = _artifact_payload(gallery)
    root = Path(artifact_directory).expanduser()
    if root.is_symlink():
        raise IVEFaceGalleryArtifactError("gallery artifact root is a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise IVEFaceGalleryArtifactError("gallery artifact root is not a directory")
    contract = _safe_directory(root, IVE_FACE_IDENTITY_GALLERY_CONTRACT)
    shard = _safe_directory(contract, digest[:2])
    destination = shard / f"{digest}.json"
    if destination.is_symlink():
        raise IVEFaceGalleryArtifactError("gallery artifact path is a symlink")
    if destination.exists():
        try:
            existing = _read_regular_nonsymlink(destination)
        except (OSError, IVEFaceGalleryArtifactError) as exc:
            raise IVEFaceGalleryArtifactError(
                "cannot read existing gallery artifact"
            ) from exc
        if not hmac.compare_digest(existing, encoded):
            raise IVEFaceGalleryArtifactError(
                "content-addressed gallery path contains different bytes"
            )
        return IVEFaceGalleryArtifact(destination, digest, gallery)

    temporary = shard / f".{digest}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o400)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = _read_regular_nonsymlink(destination)
            if not hmac.compare_digest(existing, encoded):
                raise IVEFaceGalleryArtifactError(
                    "concurrent gallery writer published different bytes"
                )
        directory_fd = os.open(shard, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except IVEFaceGalleryArtifactError:
        raise
    except OSError as exc:
        raise IVEFaceGalleryArtifactError("failed to publish gallery artifact") from exc
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return IVEFaceGalleryArtifact(destination, digest, gallery)


def _read_regular_nonsymlink(path: Path) -> bytes:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise IVEFaceGalleryArtifactError("gallery artifact is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise IVEFaceGalleryArtifactError(
            "gallery artifact must be a non-symlink regular file"
        )
    if not 1 <= before.st_size <= _MAXIMUM_ARTIFACT_BYTES:
        raise IVEFaceGalleryArtifactError("gallery artifact size is invalid")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                if sum(len(item) for item in chunks) > _MAXIMUM_ARTIFACT_BYTES:
                    raise IVEFaceGalleryArtifactError(
                        "gallery artifact exceeds its size limit"
                    )
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except IVEFaceGalleryArtifactError:
        raise
    except OSError as exc:
        raise IVEFaceGalleryArtifactError("cannot read gallery artifact") from exc
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise IVEFaceGalleryArtifactError("gallery artifact changed while reading")
    return b"".join(chunks)


def load_ive_face_identity_gallery_artifact(
    path: Path,
    *,
    expected_content_sha256: str | None = None,
) -> IVEFaceGalleryArtifact:
    """Strictly load and semantically revalidate a private-local gallery."""

    artifact_path = Path(path).expanduser()
    encoded = _read_regular_nonsymlink(artifact_path)
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except IVEFaceGalleryArtifactError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IVEFaceGalleryArtifactError("cannot decode gallery artifact") from exc
    mapping = _strict_mapping(payload, "gallery artifact")
    keys = (
        "contract",
        "schema_version",
        "content_hash_basis",
        "content_sha256",
        "distribution_policy",
        "privacy",
        "gallery",
    )
    _strict_keys(mapping, keys, "gallery artifact")
    if mapping["contract"] != IVE_FACE_IDENTITY_GALLERY_CONTRACT:
        raise IVEFaceGalleryArtifactError("gallery artifact contract mismatch")
    if (
        isinstance(mapping["schema_version"], bool)
        or not isinstance(mapping["schema_version"], int)
        or mapping["schema_version"] != IVE_FACE_IDENTITY_GALLERY_SCHEMA_VERSION
    ):
        raise IVEFaceGalleryArtifactError("gallery artifact schema mismatch")
    if mapping["content_hash_basis"] != _CONTENT_HASH_BASIS:
        raise IVEFaceGalleryArtifactError("gallery content hash basis mismatch")
    if mapping["distribution_policy"] != PRIVATE_LOCAL_DISTRIBUTION_POLICY:
        raise IVEFaceGalleryArtifactError("gallery private-local policy mismatch")
    claimed = _sha256(mapping["content_sha256"], "content_sha256")
    if not hmac.compare_digest(claimed, _artifact_content_sha256(mapping)):
        raise IVEFaceGalleryArtifactError("gallery artifact checksum mismatch")
    if artifact_path.name != f"{claimed}.json":
        raise IVEFaceGalleryArtifactError("gallery filename is not content addressed")
    if expected_content_sha256 is not None and not hmac.compare_digest(
        claimed, _sha256(expected_content_sha256, "expected_content_sha256")
    ):
        raise IVEFaceGalleryArtifactError("gallery content SHA-256 mismatch")
    privacy = _strict_mapping(mapping["privacy"], "gallery privacy")
    expected_privacy = {
        "private_local_only": True,
        "contains_sface_prototype_vectors": True,
        "raw_frames_persisted": False,
        "source_media_paths_persisted": False,
        "raw_query_embeddings_persisted": False,
        "stageedit_snapshot_export_permitted": False,
        "general_audit_json_export_permitted": False,
    }
    _strict_keys(privacy, expected_privacy, "gallery privacy")
    if dict(privacy) != expected_privacy:
        raise IVEFaceGalleryArtifactError("gallery privacy declaration mismatch")
    try:
        canonical = _canonical_json_bytes(mapping)
    except (TypeError, ValueError) as exc:
        raise IVEFaceGalleryArtifactError(
            "gallery artifact contains non-finite or non-canonical data"
        ) from exc
    if not hmac.compare_digest(encoded, canonical):
        raise IVEFaceGalleryArtifactError("gallery artifact JSON is not canonical")
    gallery = IVEFaceIdentityGallery.from_dict(mapping["gallery"])
    return IVEFaceGalleryArtifact(artifact_path, claimed, gallery)


__all__ = [
    "CLOSEUP_FACECAM_MEDIA_CONTRACT",
    "FACECAM_TITLE_MEMBER_AUTHORIZATION_CONTRACT",
    "IDENTITY_PROVIDER_ID",
    "IVE_FACE_IDENTITY_GALLERY_CONTRACT",
    "IVE_FACE_IDENTITY_GALLERY_SCHEMA_VERSION",
    "PRIVATE_LOCAL_DISTRIBUTION_POLICY",
    "PORTRAIT_FANCAM_MEDIA_CONTRACT",
    "SINGLE_DOMINANT_YUNET_FACE_CONTRACT",
    "TITLE_MEMBER_AUTHORIZATION_CONTRACT",
    "AuthorizedFaceEmbeddingSample",
    "IVEFaceGalleryArtifact",
    "IVEFaceGalleryArtifactError",
    "IVEFaceGalleryConfig",
    "IVEFaceGalleryError",
    "IVEFaceGalleryRejection",
    "IVEFaceGallerySourceAuthority",
    "IVEFaceIdentityGallery",
    "IVEFacePrototype",
    "MemberStateIdentityAdapter",
    "OfficialTargetMemberFancamProvenance",
    "OpenSetIVEIdentityResult",
    "adapt_gallery_identity_for_member_state",
    "build_ive_face_identity_gallery",
    "classify_open_set_ive_identity",
    "extract_authorized_face_sample",
    "load_ive_face_identity_gallery_artifact",
    "write_ive_face_identity_gallery_artifact",
]
