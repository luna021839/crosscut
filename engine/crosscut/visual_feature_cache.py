"""Fail-closed persistent cache for source-local numeric visual features.

The cache deliberately stores no decoded image, compressed frame, SFace
embedding, or other appearance-identity vector.  One record is bound to the
SHA-256 of the complete source file, an exact canonical frame identity, the
descriptor request options, and a fingerprint of the analyzer schema, model
assets, runtime, and implementation files.

Records are independent and atomically replaced.  An interrupted analysis can
therefore reuse every descriptor completed before the interruption without
making the timeline optimizer, reviewer, exact-eight gate, or decoded-proxy
gate persistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Any, Mapping, Sequence
import uuid


VISUAL_FEATURE_CACHE_CONTRACT = "ive_source_local_visual_features_v1"
VISUAL_FEATURE_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAXIMUM_RECORD_BYTES = 16 * 1024 * 1024
_FORBIDDEN_PAYLOAD_KEY_PARTS = (
    "raw_frame",
    "pixels",
    "image_bytes",
    "compressed_frame",
    "sface",
    "face_identity_embedding",
    "appearance_embedding",
    "orb_descriptor",
)


def _stable_sha256(path: Path) -> tuple[str, tuple[int, int, int, int, int]]:
    """Hash one unchanged regular file and return its stable stat identity."""

    resolved = Path(path).expanduser().resolve(strict=True)
    before = resolved.stat()
    if not resolved.is_file():
        raise ValueError(f"visual feature source is not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    after = resolved.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise OSError("visual feature source changed while SHA-256 was computed")
    return (digest.hexdigest(), after_identity)


def _file_digest(path: Path) -> str:
    digest, _identity = _stable_sha256(path)
    return digest


@dataclass(frozen=True, slots=True)
class AnalyzerFingerprint:
    """Stable analyzer identity included in every cache record and path."""

    digest: str
    components: Mapping[str, Any]

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.digest) is None:
            raise ValueError("analyzer fingerprint must be a lowercase SHA-256")


def build_analyzer_fingerprint(
    *,
    descriptor_schema: str,
    code_paths: Sequence[Path],
    model_assets: Mapping[str, str],
    runtime_components: Mapping[str, str],
) -> AnalyzerFingerprint:
    """Fingerprint every input that can alter a numeric frame descriptor."""

    schema = descriptor_schema.strip()
    if not schema:
        raise ValueError("descriptor_schema cannot be blank")
    code: dict[str, str] = {}
    for path in sorted(
        (Path(item).expanduser().resolve(strict=True) for item in code_paths),
        key=lambda item: item.name,
    ):
        if path.name in code:
            raise ValueError("analyzer code path names must be unique")
        code[path.name] = _file_digest(path)
    models = {str(key): str(value) for key, value in sorted(model_assets.items())}
    runtime = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        **{
            str(key): str(value)
            for key, value in sorted(runtime_components.items())
        },
    }
    components: dict[str, Any] = {
        "cache_contract": VISUAL_FEATURE_CACHE_CONTRACT,
        "feature_schema_version": VISUAL_FEATURE_SCHEMA_VERSION,
        "descriptor_schema": schema,
        "code_sha256": code,
        "model_asset_sha256_or_state": models,
        "runtime": runtime,
    }
    encoded = json.dumps(
        components,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return AnalyzerFingerprint(
        digest=hashlib.sha256(encoded).hexdigest(),
        components=components,
    )


@dataclass(frozen=True, slots=True)
class CanonicalFrameIdentity:
    """Exact decoded-frame identity; never a rounded millisecond timestamp."""

    frame_index: int
    fps_numerator: int
    fps_denominator: int
    source_time_numerator: int | None = None
    source_time_denominator: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        if (
            isinstance(self.fps_numerator, bool)
            or not isinstance(self.fps_numerator, int)
            or self.fps_numerator <= 0
            or isinstance(self.fps_denominator, bool)
            or not isinstance(self.fps_denominator, int)
            or self.fps_denominator <= 0
        ):
            raise ValueError("canonical FPS ratio must contain positive integers")
        if self.source_time_numerator is None and self.source_time_denominator is None:
            object.__setattr__(
                self,
                "source_time_numerator",
                self.frame_index * self.fps_denominator,
            )
            object.__setattr__(self, "source_time_denominator", self.fps_numerator)
        if (
            isinstance(self.source_time_numerator, bool)
            or not isinstance(self.source_time_numerator, int)
            or self.source_time_numerator < 0
            or isinstance(self.source_time_denominator, bool)
            or not isinstance(self.source_time_denominator, int)
            or self.source_time_denominator <= 0
        ):
            raise ValueError("canonical source time ratio must be non-negative")

    @classmethod
    def from_seconds(
        cls,
        seconds: float,
        *,
        frame_rate: float,
    ) -> "CanonicalFrameIdentity":
        if not math.isfinite(seconds):
            raise ValueError("frame timestamp must be finite")
        if not math.isfinite(frame_rate) or frame_rate <= 0.0:
            raise ValueError("frame rate must be positive and finite")
        fps = Fraction(str(float(frame_rate))).limit_denominator(1_000_000)
        timestamp = Fraction(str(max(0.0, float(seconds))))
        scaled = timestamp * fps
        # Match the historical nearest-frame intent without reducing the key
        # to milliseconds.  Half frames round away from zero (all inputs are
        # non-negative), so the result is stable across Python versions.
        frame_index = (2 * scaled.numerator + scaled.denominator) // (
            2 * scaled.denominator
        )
        return cls(
            frame_index=frame_index,
            fps_numerator=fps.numerator,
            fps_denominator=fps.denominator,
            source_time_numerator=timestamp.numerator,
            source_time_denominator=timestamp.denominator,
        )

    @property
    def canonical_seconds(self) -> float:
        assert self.source_time_numerator is not None
        assert self.source_time_denominator is not None
        return float(self.source_time_numerator) / float(self.source_time_denominator)

    def to_dict(self) -> dict[str, int]:
        assert self.source_time_numerator is not None
        assert self.source_time_denominator is not None
        return {
            "frame_index": self.frame_index,
            "fps_numerator": self.fps_numerator,
            "fps_denominator": self.fps_denominator,
            "source_time_numerator": self.source_time_numerator,
            "source_time_denominator": self.source_time_denominator,
        }


@dataclass(frozen=True, slots=True)
class DescriptorRequestIdentity:
    primary: CanonicalFrameIdentity
    following: CanonicalFrameIdentity | None
    primary_working_frame_sha256: str
    following_working_frame_sha256: str | None
    include_subject_geometry: bool
    include_formation_pose: bool

    def __post_init__(self) -> None:
        if not isinstance(self.include_subject_geometry, bool) or not isinstance(
            self.include_formation_pose, bool
        ):
            raise TypeError("descriptor feature flags must be booleans")
        if self.include_formation_pose and not self.include_subject_geometry:
            raise ValueError("formation pose requires subject geometry")
        if _SHA256.fullmatch(self.primary_working_frame_sha256) is None:
            raise ValueError("primary working frame must have a SHA-256 identity")
        if (self.following is None) != (
            self.following_working_frame_sha256 is None
        ):
            raise ValueError("following frame identity and SHA-256 must agree")
        if (
            self.following_working_frame_sha256 is not None
            and _SHA256.fullmatch(self.following_working_frame_sha256) is None
        ):
            raise ValueError("following working frame must have a SHA-256 identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.to_dict(),
            "following": (
                self.following.to_dict() if self.following is not None else None
            ),
            "primary_working_frame_sha256": self.primary_working_frame_sha256,
            "following_working_frame_sha256": (
                self.following_working_frame_sha256
            ),
            "include_subject_geometry": self.include_subject_geometry,
            "include_formation_pose": self.include_formation_pose,
        }


def _validate_derived_payload(value: Any, *, key_path: str = "payload") -> None:
    """Reject non-JSON, non-finite, or explicitly forbidden media payloads."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{key_path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_derived_payload(item, key_path=f"{key_path}[{index}]")
        return
    if isinstance(value, dict):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"{key_path} contains a non-string key")
            folded = raw_key.casefold()
            if any(part in folded for part in _FORBIDDEN_PAYLOAD_KEY_PARTS):
                raise ValueError(
                    f"{key_path}.{raw_key} is forbidden in the visual feature cache"
                )
            _validate_derived_payload(item, key_path=f"{key_path}.{raw_key}")
        return
    raise TypeError(f"{key_path} contains unsupported type {type(value).__name__}")


class PersistentVisualFeatureCache:
    """Per-source content-addressed descriptor records with atomic writes."""

    def __init__(
        self,
        *,
        source_cache_directory: Path,
        source_path: Path,
        analyzer: AnalyzerFingerprint,
    ) -> None:
        cache_root = Path(source_cache_directory).expanduser()
        if cache_root.is_symlink():
            raise ValueError("source cache directory cannot be a symbolic link")
        cache_root.mkdir(parents=True, exist_ok=True)
        if not cache_root.is_dir():
            raise ValueError("source cache directory must be a directory")
        self.source_path = Path(source_path).expanduser().resolve(strict=True)
        self.source_sha256, self._source_stat_identity = _stable_sha256(
            self.source_path
        )
        self.analyzer = analyzer
        self._root = self._safe_directory(cache_root, "derived")
        self._root = self._safe_directory(self._root, "visual-v1")
        self._root = self._safe_directory(self._root, self.source_sha256[:2])
        self._root = self._safe_directory(self._root, self.source_sha256)
        self._root = self._safe_directory(self._root, analyzer.digest)
        self._hit_count = 0
        self._miss_count = 0
        self._write_count = 0
        self._invalid_count = 0
        self._disabled_by_source_change = False

    @staticmethod
    def _safe_directory(parent: Path, name: str) -> Path:
        child = parent / name
        if child.is_symlink():
            raise ValueError(f"visual feature cache path is a symlink: {child}")
        child.mkdir(exist_ok=True)
        if not child.is_dir():
            raise ValueError(f"visual feature cache path is not a directory: {child}")
        return child

    def _source_is_unchanged(self) -> bool:
        if self._disabled_by_source_change:
            return False
        try:
            stat = self.source_path.stat()
        except OSError:
            self._disabled_by_source_change = True
            return False
        identity = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        if identity != self._source_stat_identity:
            self._disabled_by_source_change = True
            return False
        return True

    def _record_identity(
        self,
        request: DescriptorRequestIdentity,
    ) -> dict[str, Any]:
        return {
            "cache_contract": VISUAL_FEATURE_CACHE_CONTRACT,
            "feature_schema_version": VISUAL_FEATURE_SCHEMA_VERSION,
            "source_content": {
                "algorithm": "sha256",
                "sha256": self.source_sha256,
                "size_bytes": self._source_stat_identity[2],
            },
            "analyzer": {
                "fingerprint_sha256": self.analyzer.digest,
                "components": dict(self.analyzer.components),
            },
            "descriptor_request": request.to_dict(),
        }

    def _record_path(self, request: DescriptorRequestIdentity) -> Path:
        identity = self._record_identity(request)
        encoded = json.dumps(
            identity,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()
        shard = self._safe_directory(self._root, key[:2])
        return shard / f"{key}.json"

    def get(self, request: DescriptorRequestIdentity) -> dict[str, Any] | None:
        if not self._source_is_unchanged():
            self._miss_count += 1
            return None
        path = self._record_path(request)
        if path.is_symlink() or not path.is_file():
            self._miss_count += 1
            return None
        try:
            size = path.stat().st_size
            if not 1 <= size <= _MAXIMUM_RECORD_BYTES:
                raise ValueError("visual feature cache record size is invalid")
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("visual feature cache record is not an object")
            expected = self._record_identity(request)
            actual_identity = record.get("identity")
            if actual_identity != expected:
                raise ValueError("visual feature cache identity mismatch")
            payload = record.get("numeric_features")
            if not isinstance(payload, dict):
                raise ValueError("visual feature cache payload is not an object")
            _validate_derived_payload(payload)
            expected_payload_sha256 = record.get("numeric_features_sha256")
            actual_payload_sha256 = hashlib.sha256(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if (
                not isinstance(expected_payload_sha256, str)
                or not hmac.compare_digest(
                    expected_payload_sha256,
                    actual_payload_sha256,
                )
            ):
                raise ValueError("visual feature cache payload checksum mismatch")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            self._invalid_count += 1
            self._miss_count += 1
            return None
        self._hit_count += 1
        return payload

    def put(
        self,
        request: DescriptorRequestIdentity,
        payload: Mapping[str, Any],
    ) -> bool:
        if not self._source_is_unchanged():
            return False
        safe_payload = dict(payload)
        _validate_derived_payload(safe_payload)
        payload_encoding = json.dumps(
            safe_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        record = {
            "identity": self._record_identity(request),
            "numeric_features": safe_payload,
            "numeric_features_sha256": hashlib.sha256(
                payload_encoding
            ).hexdigest(),
            "privacy": {
                "raw_frames_persisted": False,
                "compressed_frames_persisted": False,
                "face_identity_embeddings_persisted": False,
                "appearance_embeddings_persisted": False,
            },
        }
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if not 1 <= len(encoded) <= _MAXIMUM_RECORD_BYTES:
            raise ValueError("visual feature cache record is too large")
        path = self._record_path(request)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except OSError:
            # A partial temporary is deliberately ignored by readers.  Do not
            # delete it here: interruption evidence remains recoverable and no
            # cleanup operation is needed for correctness.
            return False
        self._write_count += 1
        return True

    def diagnostics(self) -> dict[str, Any]:
        return {
            "cache_contract": VISUAL_FEATURE_CACHE_CONTRACT,
            "source_sha256": self.source_sha256,
            "analyzer_fingerprint_sha256": self.analyzer.digest,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "write_count": self._write_count,
            "invalid_record_count": self._invalid_count,
            "disabled_by_source_change": self._disabled_by_source_change,
            "raw_frames_persisted": False,
            "face_identity_embeddings_persisted": False,
        }


__all__ = [
    "AnalyzerFingerprint",
    "CanonicalFrameIdentity",
    "DescriptorRequestIdentity",
    "PersistentVisualFeatureCache",
    "VISUAL_FEATURE_CACHE_CONTRACT",
    "VISUAL_FEATURE_SCHEMA_VERSION",
    "build_analyzer_fingerprint",
]
