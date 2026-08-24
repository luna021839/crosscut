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
from pathlib import Path
from typing import Any, Mapping, MutableMapping


_INPUT_DIGEST_FIELDS = (
    ("attribution", "file_sha256"),
    ("evidence", "file_sha256"),
    ("reference_media", "media_sha256"),
    ("source_media", "media_sha256"),
)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _stable_file_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise ValueError(f"dynamic crop input changed while hashing: {path}")
    return digest.hexdigest()


def verify_dynamic_crop_input_files(
    payload: Mapping[str, Any],
    *,
    verified_sha256_cache: MutableMapping[str, str] | None = None,
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
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"dynamic crop {name} no longer matches its input")
        cache_key = str(path)
        observed = cache.get(cache_key)
        if observed is None:
            observed = _stable_file_sha256(path)
            cache[cache_key] = observed
        if observed != expected:
            raise ValueError(f"dynamic crop {name} no longer matches its input")


__all__ = ["verify_dynamic_crop_input_files"]
