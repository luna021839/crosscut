"""Bounded, streaming I/O for immutable causal scalar artifacts.

The causal sensing artifacts are intentionally plain JSON so their schema and
content hashes remain reviewable. Production grids are nevertheless large
enough that building a second ``bytes`` copy merely to publish or compare an
artifact is wasteful on an M4-class machine. This module keeps the JSON schema
unchanged while providing a finite 128 MiB hard cap, streaming canonical-JSON
hashing/publication, stable regular-file checks, and ExFAT-safe no-clobber
publication with streaming byte-for-byte collision checks and readback hashes.

It is deliberately media-agnostic. Callers still own semantic validation and
privacy-key rejection for their typed payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable, Sequence, TypeVar
import uuid


CAUSAL_SCALAR_ARTIFACT_MAXIMUM_BYTES = 128 * 1024 * 1024
CAUSAL_SCALAR_ARTIFACT_STREAM_BLOCK_BYTES = 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}")
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    {
        errno.EINVAL,
        errno.ENOTSUP,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
        getattr(errno, "ENOTTY", errno.EINVAL),
    }
)
_UNSUPPORTED_CHMOD_ERRNOS = frozenset(
    {
        errno.EPERM,
        errno.ENOTSUP,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
    }
)


class CausalArtifactIOError(ValueError):
    """Raised when bounded immutable artifact I/O cannot be proven safe."""


_Error = TypeVar("_Error", bound=Exception)


@dataclass(frozen=True, slots=True)
class ArtifactFileIdentity:
    size_bytes: int
    file_sha256: str
    stat_identity: tuple[int, int, int, int]


class _HashingTextSink:
    """Minimal text writer used by ``json.dump`` without materializing bytes."""

    def __init__(self, maximum_bytes: int, error_type: type[_Error]) -> None:
        self._maximum_bytes = maximum_bytes
        self._error_type = error_type
        self._digest = hashlib.sha256()
        self.size_bytes = 0

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError("JSON text sink only accepts strings")
        encoded = value.encode("utf-8")
        next_size = self.size_bytes + len(encoded)
        if next_size > self._maximum_bytes:
            raise self._error_type(
                "causal scalar artifact exceeds the finite byte cap"
            )
        self._digest.update(encoded)
        self.size_bytes = next_size
        return len(value)

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()


class _BoundedArtifactTextWriter(_HashingTextSink):
    """Hash UTF-8 JSON while buffering only one bounded output block."""

    def __init__(
        self,
        handle: Any,
        maximum_bytes: int,
        error_type: type[_Error],
    ) -> None:
        super().__init__(maximum_bytes, error_type)
        self._handle = handle
        self._buffer = bytearray()

    @staticmethod
    def _write_all(handle: Any, encoded: bytes) -> None:
        view = memoryview(encoded)
        written = 0
        while written < len(view):
            count = handle.write(view[written:])
            if not isinstance(count, int) or count <= 0:
                raise OSError("artifact write made no forward progress")
            written += count

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        encoded = bytes(self._buffer)
        self._buffer.clear()
        self._write_all(self._handle, encoded)

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError("JSON text writer only accepts strings")
        encoded = value.encode("utf-8")
        next_size = self.size_bytes + len(encoded)
        if next_size > self._maximum_bytes:
            raise self._error_type(
                "causal scalar artifact exceeds the finite byte cap"
            )
        self._digest.update(encoded)
        self.size_bytes = next_size
        if len(encoded) >= CAUSAL_SCALAR_ARTIFACT_STREAM_BLOCK_BYTES:
            self._flush_buffer()
            self._write_all(self._handle, encoded)
        else:
            self._buffer.extend(encoded)
            if len(self._buffer) >= CAUSAL_SCALAR_ARTIFACT_STREAM_BLOCK_BYTES:
                self._flush_buffer()
        return len(value)

    def flush(self) -> None:
        self._flush_buffer()


def _validate_maximum_bytes(
    maximum_bytes: int,
    error_type: type[_Error],
) -> int:
    if (
        isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or maximum_bytes <= 0
        or maximum_bytes > CAUSAL_SCALAR_ARTIFACT_MAXIMUM_BYTES
    ):
        raise error_type(
            "causal scalar artifact byte cap must be in (0, 128 MiB]"
        )
    return maximum_bytes


def streaming_canonical_json_identity(
    value: object,
    *,
    maximum_bytes: int = CAUSAL_SCALAR_ARTIFACT_MAXIMUM_BYTES,
    error_type: type[_Error] = CausalArtifactIOError,
) -> tuple[str, int]:
    """Return canonical JSON SHA-256 and byte size without a bytes copy."""

    limit = _validate_maximum_bytes(maximum_bytes, error_type)
    sink = _HashingTextSink(limit, error_type)
    try:
        json.dump(
            value,
            sink,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except error_type:
        raise
    except (TypeError, ValueError, UnicodeError) as exc:
        raise error_type("causal scalar artifact is not canonical finite JSON") from exc
    if sink.size_bytes <= 0:
        raise error_type("causal scalar artifact cannot be empty")
    return sink.hexdigest, sink.size_bytes


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def stream_regular_file_identity(
    path: Path,
    *,
    maximum_bytes: int = CAUSAL_SCALAR_ARTIFACT_MAXIMUM_BYTES,
    error_type: type[_Error] = CausalArtifactIOError,
) -> ArtifactFileIdentity:
    """Hash one stable non-symlink regular file with bounded block residency."""

    limit = _validate_maximum_bytes(maximum_bytes, error_type)
    requested = Path(path).expanduser()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as exc:
        raise error_type("causal scalar artifact path is not readable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise error_type("causal scalar artifact must be a regular file")
        if not 1 <= before.st_size <= limit:
            raise error_type("causal scalar artifact size is outside the finite cap")
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, CAUSAL_SCALAR_ARTIFACT_STREAM_BLOCK_BYTES)
            if not block:
                break
            total += len(block)
            if total > limit:
                raise error_type(
                    "causal scalar artifact exceeded its declared byte cap"
                )
            digest.update(block)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after) or total != before.st_size:
            raise error_type("causal scalar artifact changed while being hashed")
        try:
            current = os.lstat(requested)
        except OSError as exc:
            raise error_type(
                "causal scalar artifact pathname changed while being hashed"
            ) from exc
        if not stat.S_ISREG(current.st_mode) or _stat_identity(current) != (
            _stat_identity(after)
        ):
            raise error_type(
                "causal scalar artifact pathname changed while being hashed"
            )
        return ArtifactFileIdentity(
            size_bytes=total,
            file_sha256=digest.hexdigest(),
            stat_identity=_stat_identity(after),
        )
    except error_type:
        raise
    except OSError as exc:
        raise error_type("cannot hash causal scalar artifact") from exc
    finally:
        os.close(descriptor)


def _stream_files_equal(
    left: Path,
    right: Path,
    *,
    maximum_bytes: int,
    error_type: type[_Error],
) -> bool:
    left_identity = stream_regular_file_identity(
        left,
        maximum_bytes=maximum_bytes,
        error_type=error_type,
    )
    right_identity = stream_regular_file_identity(
        right,
        maximum_bytes=maximum_bytes,
        error_type=error_type,
    )
    if left_identity.size_bytes != right_identity.size_bytes or not hmac.compare_digest(
        left_identity.file_sha256,
        right_identity.file_sha256,
    ):
        return False
    left_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    right_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    left_fd = os.open(left, left_flags)
    try:
        right_fd = os.open(right, right_flags)
        try:
            left_before = os.fstat(left_fd)
            right_before = os.fstat(right_fd)
            if (
                _stat_identity(left_before) != left_identity.stat_identity
                or _stat_identity(right_before) != right_identity.stat_identity
            ):
                raise error_type(
                    "causal artifact changed before exact collision comparison"
                )
            while True:
                left_block = os.read(
                    left_fd,
                    CAUSAL_SCALAR_ARTIFACT_STREAM_BLOCK_BYTES,
                )
                right_block = os.read(
                    right_fd,
                    CAUSAL_SCALAR_ARTIFACT_STREAM_BLOCK_BYTES,
                )
                if not hmac.compare_digest(left_block, right_block):
                    return False
                if not left_block:
                    break
            left_after = os.fstat(left_fd)
            right_after = os.fstat(right_fd)
            if (
                _stat_identity(left_before) != _stat_identity(left_after)
                or _stat_identity(right_before) != _stat_identity(right_after)
            ):
                raise error_type(
                    "causal artifact changed during exact collision comparison"
                )
            try:
                left_path = os.lstat(left)
                right_path = os.lstat(right)
            except OSError as exc:
                raise error_type(
                    "causal artifact pathname changed during collision comparison"
                ) from exc
            if (
                not stat.S_ISREG(left_path.st_mode)
                or not stat.S_ISREG(right_path.st_mode)
                or _stat_identity(left_path) != _stat_identity(left_after)
                or _stat_identity(right_path) != _stat_identity(right_after)
            ):
                raise error_type(
                    "causal artifact pathname changed during collision comparison"
                )
            return True
        finally:
            os.close(right_fd)
    finally:
        os.close(left_fd)


def _fsync_directory(path: Path, error_type: type[_Error]) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                raise error_type("cannot flush causal artifact directory") from exc
    finally:
        os.close(descriptor)


def _best_effort_read_only(path: Path, error_type: type[_Error]) -> None:
    try:
        path.chmod(0o400)
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_CHMOD_ERRNOS:
            raise error_type("cannot protect immutable causal artifact") from exc


def _owned_lock(
    lock: Path,
    *,
    lock_identity: tuple[int, int] | None,
    marker: bytes,
) -> bool:
    try:
        current = os.lstat(lock)
    except OSError:
        return False
    if not stat.S_ISREG(current.st_mode):
        return False
    if lock_identity is not None and (current.st_dev, current.st_ino) == lock_identity:
        return True
    # ExFAT can report unstable inode numbers. Authenticate the unique marker
    # before removing a lock whose inode identity did not round-trip.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock, flags)
        try:
            opened = os.fstat(descriptor)
            observed = os.read(descriptor, len(marker) + 1)
        finally:
            os.close(descriptor)
    except OSError:
        return False
    return bool(opened.st_size == len(marker) and observed == marker)


def _require_expected_file_identity(
    path: Path,
    expected: ArtifactFileIdentity,
    *,
    maximum_bytes: int,
    error_type: type[_Error],
) -> ArtifactFileIdentity:
    observed = stream_regular_file_identity(
        path,
        maximum_bytes=maximum_bytes,
        error_type=error_type,
    )
    if observed.size_bytes != expected.size_bytes or not hmac.compare_digest(
        observed.file_sha256,
        expected.file_sha256,
    ):
        raise error_type("published causal scalar artifact readback differs")
    return observed


def publish_json_content_addressed(
    destination: Path,
    payload: object,
    *,
    content_sha256: str,
    maximum_bytes: int = CAUSAL_SCALAR_ARTIFACT_MAXIMUM_BYTES,
    error_type: type[_Error] = CausalArtifactIOError,
    indent: int | None = None,
    trailing_newline: bool = False,
) -> ArtifactFileIdentity:
    """Stream, lock, no-clobber publish, and hash-readback one JSON file.

    Publication locks are never timed out, stolen, or silently removed. A
    lock with no final artifact fails closed for explicit operator recovery;
    an already valid immutable destination can be reused without that lock.
    """

    limit = _validate_maximum_bytes(maximum_bytes, error_type)
    digest = str(content_sha256).strip()
    final = Path(destination).expanduser()
    if _SHA256.fullmatch(digest) is None or final.name != f"{digest}.json":
        raise error_type("causal scalar artifact destination is not content addressed")
    if final.parent.is_symlink() or not final.parent.is_dir():
        raise error_type("causal scalar artifact parent directory is unsafe")
    if final.is_symlink():
        raise error_type("causal scalar artifact destination cannot be a symlink")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{digest}.",
        suffix=".tmp",
        dir=final.parent,
    )
    temporary = Path(temporary_name)
    lock = final.parent / f".{digest}.publish-lock"
    lock_descriptor: int | None = None
    lock_identity: tuple[int, int] | None = None
    marker = f"{os.getpid()}:{uuid.uuid4().hex}\n".encode("ascii")
    expected: ArtifactFileIdentity | None = None
    try:
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                writer = _BoundedArtifactTextWriter(handle, limit, error_type)
                json.dump(
                    payload,
                    writer,
                    allow_nan=False,
                    ensure_ascii=False,
                    indent=indent,
                    separators=None if indent is not None else (",", ":"),
                    sort_keys=True,
                )
                if trailing_newline:
                    writer.write("\n")
                writer.flush()
                handle.flush()
                os.fsync(handle.fileno())
                if writer.size_bytes <= 0:
                    raise error_type("causal scalar artifact cannot be empty")
                temporary_stat = os.fstat(handle.fileno())
                expected = ArtifactFileIdentity(
                    size_bytes=writer.size_bytes,
                    file_sha256=writer.hexdigest,
                    stat_identity=_stat_identity(temporary_stat),
                )
        except error_type:
            raise
        except (OSError, TypeError, ValueError, UnicodeError) as exc:
            raise error_type("cannot serialize causal scalar artifact") from exc

        _best_effort_read_only(temporary, error_type)
        if final.exists():
            if not _stream_files_equal(
                temporary,
                final,
                maximum_bytes=limit,
                error_type=error_type,
            ):
                raise error_type(
                    "content-addressed destination contains different bytes"
                )
            assert expected is not None
            return _require_expected_file_identity(
                final,
                expected,
                maximum_bytes=limit,
                error_type=error_type,
            )

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            lock_descriptor = os.open(lock, flags, 0o600)
        except FileExistsError as exc:
            if final.exists() and not final.is_symlink() and _stream_files_equal(
                temporary,
                final,
                maximum_bytes=limit,
                error_type=error_type,
            ):
                assert expected is not None
                return _require_expected_file_identity(
                    final,
                    expected,
                    maximum_bytes=limit,
                    error_type=error_type,
                )
            raise error_type("causal artifact publication lock is already held") from exc
        lock_stat = os.fstat(lock_descriptor)
        lock_identity = (lock_stat.st_dev, lock_stat.st_ino)
        if os.write(lock_descriptor, marker) != len(marker):
            raise error_type("causal artifact publication lock write was incomplete")
        os.fsync(lock_descriptor)

        if final.is_symlink():
            raise error_type("causal artifact destination became a symlink")
        if final.exists():
            if not _stream_files_equal(
                temporary,
                final,
                maximum_bytes=limit,
                error_type=error_type,
            ):
                raise error_type(
                    "concurrent causal artifact writer published different bytes"
                )
        else:
            os.replace(temporary, final)
            _best_effort_read_only(final, error_type)
        _fsync_directory(final.parent, error_type)
        assert expected is not None
        observed = _require_expected_file_identity(
            final,
            expected,
            maximum_bytes=limit,
            error_type=error_type,
        )
        return observed
    except error_type:
        raise
    except OSError as exc:
        raise error_type("failed to publish immutable causal artifact") from exc
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if _owned_lock(lock, lock_identity=lock_identity, marker=marker):
            try:
                lock.unlink()
            except OSError:
                pass
        try:
            temporary.unlink()
        except OSError:
            pass


def load_bounded_json_artifact(
    path: Path,
    *,
    maximum_bytes: int = CAUSAL_SCALAR_ARTIFACT_MAXIMUM_BYTES,
    error_type: type[_Error] = CausalArtifactIOError,
    object_pairs_hook: Callable[[Sequence[tuple[str, Any]]], Any] | None = None,
    parse_constant: Callable[[str], Any] | None = None,
) -> Any:
    """Parse one stable regular JSON file after enforcing the hard cap."""

    limit = _validate_maximum_bytes(maximum_bytes, error_type)
    requested = Path(path).expanduser()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as exc:
        raise error_type("causal scalar artifact path is not readable") from exc
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise error_type("causal scalar artifact must be a regular file")
            if not 1 <= before.st_size <= limit:
                raise error_type(
                    "causal scalar artifact size is outside the finite cap"
                )
            payload = json.load(
                handle,
                object_pairs_hook=object_pairs_hook,
                parse_constant=parse_constant,
            )
            after = os.fstat(handle.fileno())
            if _stat_identity(before) != _stat_identity(after):
                raise error_type("causal scalar artifact changed while being parsed")
            try:
                current = os.lstat(requested)
            except OSError as exc:
                raise error_type(
                    "causal scalar artifact pathname changed while being parsed"
                ) from exc
            if not stat.S_ISREG(current.st_mode) or _stat_identity(current) != (
                _stat_identity(after)
            ):
                raise error_type(
                    "causal scalar artifact pathname changed while being parsed"
                )
            return payload
    except error_type:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise error_type("cannot decode causal scalar artifact JSON") from exc


__all__ = [
    "ArtifactFileIdentity",
    "CAUSAL_SCALAR_ARTIFACT_MAXIMUM_BYTES",
    "CAUSAL_SCALAR_ARTIFACT_STREAM_BLOCK_BYTES",
    "CausalArtifactIOError",
    "load_bounded_json_artifact",
    "publish_json_content_addressed",
    "stream_regular_file_identity",
    "streaming_canonical_json_identity",
]
