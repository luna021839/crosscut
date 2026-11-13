"""Content-addressed cache for non-media scalar source analyses.

The cache stores only JSON-native derived measurements. It never stores PCM,
decoded frames, compressed media, identity embeddings, or analyzer model
outputs that are not explicitly represented as finite scalar/list payloads.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping
import uuid

from .visual_feature_cache import (
    PersistentVisualFeatureCacheStorageUnavailableError,
    _stable_sha256,
    assert_visual_feature_derived_cache_root_available,
)


SOURCE_SCALAR_CACHE_CONTRACT = "ive_source_scalar_analysis_v1"
SOURCE_SCALAR_CACHE_SCHEMA_VERSION = 1
_SAFE_NAMESPACE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_MAXIMUM_RECORD_BYTES = 32 * 1024 * 1024
_FORBIDDEN_KEY_PARTS = (
    "raw_frame",
    "pixels",
    "image_bytes",
    "compressed_frame",
    "raw_pcm",
    "audio_samples",
    "sface",
    "face_identity_embedding",
    "appearance_embedding",
)


class PersistentSourceScalarCacheStorageUnavailableError(RuntimeError):
    """An explicitly selected direct derived-cache root was lost."""


def source_content_sha256(path: Path) -> str:
    """Return the stable full-file digest shared by scalar and visual caches."""

    return _stable_sha256(Path(path))[0]


def _validate_payload(value: Any, *, key_path: str = "payload") -> None:
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
            _validate_payload(item, key_path=f"{key_path}[{index}]")
        return
    if isinstance(value, dict):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"{key_path} contains a non-string key")
            folded = raw_key.casefold()
            if any(part in folded for part in _FORBIDDEN_KEY_PARTS):
                raise ValueError(f"{key_path}.{raw_key} is forbidden")
            _validate_payload(item, key_path=f"{key_path}.{raw_key}")
        return
    raise TypeError(f"{key_path} contains unsupported type {type(value).__name__}")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    _validate_payload(dict(value), key_path="analyzer")
    return hashlib.sha256(
        json.dumps(
            dict(value),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class PersistentSourceScalarCache:
    """One atomic scalar record per source content and analyzer identity."""

    def __init__(
        self,
        *,
        source_cache_directory: Path,
        source_path: Path,
        namespace: str,
        analyzer: Mapping[str, Any],
        derived_cache_directory: Path | None = None,
    ) -> None:
        if _SAFE_NAMESPACE.fullmatch(namespace) is None:
            raise ValueError("scalar cache namespace is invalid")
        self._direct_derived_cache_configured = (
            derived_cache_directory is not None
        )
        self._storage_root: Path | None = None
        self._storage_root_identity: tuple[int, int] | None = None
        self._storage_io_failure_count = 0
        if derived_cache_directory is None:
            root = Path(source_cache_directory).expanduser()
            if root.is_symlink():
                raise ValueError(
                    "source cache directory cannot be a symbolic link"
                )
            root.mkdir(parents=True, exist_ok=True)
            if not root.is_dir():
                raise ValueError("source cache directory must be a directory")
            derived_root = self._safe_directory(root, "derived")
        else:
            direct_root = Path(derived_cache_directory).expanduser()
            if direct_root.is_symlink():
                raise PersistentSourceScalarCacheStorageUnavailableError(
                    "required direct source scalar cache storage is unavailable"
                )
            try:
                (
                    derived_root,
                    self._storage_root_identity,
                ) = assert_visual_feature_derived_cache_root_available(
                    direct_root,
                )
            except PersistentVisualFeatureCacheStorageUnavailableError:
                raise PersistentSourceScalarCacheStorageUnavailableError(
                    "required direct source scalar cache storage is unavailable"
                ) from None
            self._storage_root = derived_root
        self.source_path = Path(source_path).expanduser().resolve(strict=True)
        self.source_sha256, self._source_stat_identity = _stable_sha256(
            self.source_path
        )
        self.namespace = namespace
        self.analyzer = {
            "cache_contract": SOURCE_SCALAR_CACHE_CONTRACT,
            "schema_version": SOURCE_SCALAR_CACHE_SCHEMA_VERSION,
            **dict(analyzer),
        }
        self.analyzer_sha256 = _canonical_sha256(self.analyzer)
        try:
            if self._direct_derived_cache_configured:
                assert self._storage_root is not None
                assert self._storage_root_identity is not None
                assert_visual_feature_derived_cache_root_available(
                    self._storage_root,
                    expected_identity=self._storage_root_identity,
                )
            self._root = self._safe_directory(derived_root, "scalar-v1")
            self._root = self._safe_directory(self._root, namespace)
            self._root = self._safe_directory(
                self._root,
                self.source_sha256[:2],
            )
            self._root = self._safe_directory(self._root, self.source_sha256)
        except PersistentVisualFeatureCacheStorageUnavailableError:
            raise PersistentSourceScalarCacheStorageUnavailableError(
                "required direct source scalar cache storage is unavailable"
            ) from None
        except (OSError, ValueError):
            if self._direct_derived_cache_configured:
                raise PersistentSourceScalarCacheStorageUnavailableError(
                    "required direct source scalar cache storage is unavailable"
                ) from None
            raise
        self._path = self._root / f"{self.analyzer_sha256}.json"
        self._hit_count = 0
        self._miss_count = 0
        self._write_count = 0
        self._invalid_count = 0
        self._disabled_by_source_change = False

    @staticmethod
    def _safe_directory(parent: Path, name: str) -> Path:
        child = parent / name
        if child.is_symlink():
            raise ValueError(f"scalar cache path is a symlink: {child}")
        child.mkdir(exist_ok=True)
        if not child.is_dir():
            raise ValueError(f"scalar cache path is not a directory: {child}")
        return child

    def _source_is_unchanged(self) -> bool:
        if self._disabled_by_source_change:
            return False
        try:
            current = self.source_path.stat()
        except OSError:
            self._disabled_by_source_change = True
            return False
        identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if identity != self._source_stat_identity:
            self._disabled_by_source_change = True
            return False
        return True

    def assert_storage_root_available(self) -> None:
        """Attest a required direct root without weakening legacy caching."""

        if not self._direct_derived_cache_configured:
            return
        assert self._storage_root is not None
        assert self._storage_root_identity is not None
        try:
            assert_visual_feature_derived_cache_root_available(
                self._storage_root,
                expected_identity=self._storage_root_identity,
            )
        except PersistentVisualFeatureCacheStorageUnavailableError:
            self._storage_io_failure_count += 1
            raise PersistentSourceScalarCacheStorageUnavailableError(
                "required direct source scalar cache storage is unavailable"
            ) from None

    def _identity(self) -> dict[str, Any]:
        return {
            "cache_contract": SOURCE_SCALAR_CACHE_CONTRACT,
            "schema_version": SOURCE_SCALAR_CACHE_SCHEMA_VERSION,
            "namespace": self.namespace,
            "source_content": {
                "algorithm": "sha256",
                "sha256": self.source_sha256,
                "size_bytes": self._source_stat_identity[2],
            },
            "analyzer": self.analyzer,
            "analyzer_sha256": self.analyzer_sha256,
        }

    def get(self) -> dict[str, Any] | None:
        self.assert_storage_root_available()
        if not self._source_is_unchanged():
            self._miss_count += 1
            return None
        if self._path.is_symlink() or not self._path.is_file():
            self.assert_storage_root_available()
            self._miss_count += 1
            return None
        try:
            size = self._path.stat().st_size
            if not 1 <= size <= _MAXIMUM_RECORD_BYTES:
                raise ValueError("scalar cache record size is invalid")
            record = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("scalar cache record is not an object")
            if record.get("identity") != self._identity():
                raise ValueError("scalar cache identity mismatch")
            payload = record.get("scalar_analysis")
            if not isinstance(payload, dict):
                raise ValueError("scalar cache payload is not an object")
            _validate_payload(payload)
            actual = hashlib.sha256(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            expected = record.get("scalar_analysis_sha256")
            if not isinstance(expected, str) or not hmac.compare_digest(
                expected,
                actual,
            ):
                raise ValueError("scalar cache payload checksum mismatch")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            self.assert_storage_root_available()
            self._invalid_count += 1
            self._miss_count += 1
            return None
        self._hit_count += 1
        return payload

    def put(self, payload: Mapping[str, Any]) -> bool:
        self.assert_storage_root_available()
        if not self._source_is_unchanged():
            return False
        safe_payload = dict(payload)
        _validate_payload(safe_payload)
        payload_bytes = json.dumps(
            safe_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        record = {
            "identity": self._identity(),
            "scalar_analysis": safe_payload,
            "scalar_analysis_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "privacy": {
                "raw_media_persisted": False,
                "decoded_frames_persisted": False,
                "raw_pcm_persisted": False,
                "identity_embeddings_persisted": False,
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
            raise ValueError("scalar cache record is too large")
        temporary = self._path.with_name(
            f".{self._path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
            directory_fd = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            self.assert_storage_root_available()
            return False
        self._write_count += 1
        return True

    def diagnostics(self) -> dict[str, Any]:
        return {
            "cache_contract": SOURCE_SCALAR_CACHE_CONTRACT,
            "namespace": self.namespace,
            "source_sha256": self.source_sha256,
            "analyzer_sha256": self.analyzer_sha256,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "write_count": self._write_count,
            "invalid_record_count": self._invalid_count,
            "disabled_by_source_change": self._disabled_by_source_change,
            "required": self._direct_derived_cache_configured,
            "direct_derived_cache_configured": (
                self._direct_derived_cache_configured
            ),
            "storage_io_failure_count": self._storage_io_failure_count,
            "local_paths_exposed": False,
            "raw_media_persisted": False,
        }


__all__ = [
    "PersistentSourceScalarCache",
    "PersistentSourceScalarCacheStorageUnavailableError",
    "SOURCE_SCALAR_CACHE_CONTRACT",
    "SOURCE_SCALAR_CACHE_SCHEMA_VERSION",
    "source_content_sha256",
]
