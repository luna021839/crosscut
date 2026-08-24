"""Deduplicated immutable-input verification for crop-v2 artifact batches.

One song can contain hundreds of per-shot crop artifacts that all point to the
same reference edit, attribution/evidence JSON, and ten synchronized sources.
Hashing those large videos once per artifact adds no integrity and can reread
hundreds of gigabytes before training.  This helper preserves the exact SHA-256
contract while caching only verified path digests for the lifetime of one
manifest or corpus build.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping


_INPUT_DIGEST_FIELDS = (
    ("attribution", "file_sha256"),
    ("evidence", "file_sha256"),
    ("reference_media", "media_sha256"),
    ("source_media", "media_sha256"),
)

_FileIdentity = tuple[int, int, int, int, int]


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _stable_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
        after = os.fstat(handle.fileno())
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ValueError(f"dynamic crop input changed while hashing: {path}")
    if _file_identity(path) != identity_after:
        raise ValueError(f"dynamic crop input path changed while hashing: {path}")
    return digest.hexdigest()


def _file_identity(path: Path) -> _FileIdentity:
    observed = path.stat()
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _input_mismatch_message(
    *,
    name: str,
    path: Path,
    expected_sha256: object,
    observed_sha256: object,
) -> str:
    return (
        "dynamic crop input no longer matches its provenance "
        f"name={name} path={path} expected_sha256={expected_sha256} "
        f"observed_sha256={observed_sha256}"
    )


def verify_dynamic_crop_input_files(
    payload: Mapping[str, Any],
    *,
    verified_sha256_cache: MutableMapping[str, str] | None = None,
    verified_identity_cache: MutableMapping[str, _FileIdentity] | None = None,
) -> None:
    """Verify every provenance input, hashing each absolute path at most once."""

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("dynamic crop provenance is missing")
    cache = verified_sha256_cache if verified_sha256_cache is not None else {}
    for name, digest_name in _INPUT_DIGEST_FIELDS:
        item = provenance.get(name)
        if not isinstance(item, Mapping):
            raise ValueError(f"dynamic crop {name} provenance is invalid")
        expected = item.get(digest_name)
        if not _is_sha256(expected):
            raise ValueError(f"dynamic crop {name} provenance is invalid")
        path = Path(str(item.get("path") or ""))
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError(
                _input_mismatch_message(
                    name=name,
                    path=path,
                    expected_sha256=expected,
                    observed_sha256="unavailable",
                )
            )
        cache_key = str(path)
        observed = cache.get(cache_key)
        if observed is None:
            try:
                identity_before = (
                    _file_identity(path)
                    if verified_identity_cache is not None
                    else None
                )
                observed = _stable_file_sha256(path)
                identity_after = (
                    _file_identity(path)
                    if verified_identity_cache is not None
                    else None
                )
                if identity_before != identity_after:
                    raise ValueError(
                        f"dynamic crop input identity changed while hashing: {path}"
                    )
            except (OSError, ValueError) as exc:
                raise ValueError(
                    _input_mismatch_message(
                        name=name,
                        path=path,
                        expected_sha256=expected,
                        observed_sha256="unavailable",
                    )
                    + f" verification_error={exc}"
                ) from exc
            cache[cache_key] = observed
            if verified_identity_cache is not None:
                assert identity_after is not None
                verified_identity_cache[cache_key] = identity_after
        elif verified_identity_cache is not None:
            expected_identity = verified_identity_cache.get(cache_key)
            if expected_identity is None:
                raise ValueError(
                    "dynamic crop digest cache is missing its file identity "
                    f"path={path}"
                )
            try:
                current_identity = _file_identity(path)
            except OSError as exc:
                raise ValueError(
                    _input_mismatch_message(
                        name=name,
                        path=path,
                        expected_sha256=expected,
                        observed_sha256="unavailable",
                    )
                    + f" verification_error={exc}"
                ) from exc
            if current_identity != expected_identity:
                raise ValueError(
                    "dynamic crop input identity changed after hashing "
                    f"name={name} path={path}"
                )
        if observed != expected:
            raise ValueError(
                _input_mismatch_message(
                    name=name,
                    path=path,
                    expected_sha256=expected,
                    observed_sha256=observed,
                )
            )


__all__ = ["verify_dynamic_crop_input_files"]
