"""Hash-bound local installation pointer for private IVE face identity.

The native application only asks for an artist/song request. The private
gallery used by real StageEdit causal sensing is installed once per machine,
much like the production StageEdit model bundle, instead of being hard-coded
to one developer artifact path.

An installation pointer contains no face vectors. It binds one absolute local
gallery artifact path to the gallery's canonical content SHA-256 and binds the
pointer itself with a canonical SHA-256. Discovery is fail-closed: an explicit
environment override is mandatory when set, multiple default installations are
ambiguous, and every load revalidates both pointer bytes and gallery semantics.
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
from typing import Any, Mapping
import uuid

from .ive_face_identity_gallery import (
    IVEFaceGalleryArtifact,
    load_ive_face_identity_gallery_artifact,
)


IVE_CAUSAL_SENSING_INSTALLATION_SCHEMA_VERSION = 1
IVE_CAUSAL_SENSING_INSTALLATION_CONTRACT = (
    "ive_private_face_gallery_local_installation_v1"
)
IVE_CAUSAL_SENSING_INSTALLATION_ACTIVATION = (
    "automatic_ive_stage_edit_real_causal_sensing_only"
)
IVE_CAUSAL_SENSING_INSTALLATION_ENV = (
    "CROSSCUT_IVE_CAUSAL_SENSING_GALLERY_INSTALLATION"
)
IVE_CAUSAL_SENSING_INSTALLATION_FILENAME = "active-private-gallery.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAXIMUM_POINTER_BYTES = 64 * 1024
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    {
        errno.EINVAL,
        errno.ENOTSUP,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
        getattr(errno, "ENOTTY", errno.EINVAL),
    }
)


class IVECausalSensingInstallationError(RuntimeError):
    """A configured private-gallery installation cannot be trusted."""


class IVECausalSensingInstallationNotFoundError(
    IVECausalSensingInstallationError
):
    """No private-gallery installation pointer is available."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IVECausalSensingInstallationError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_stable_regular_file(path: Path) -> bytes:
    source = Path(path)
    descriptor: int | None = None
    try:
        path_before = source.lstat()
        if not stat.S_ISREG(path_before.st_mode) or source.is_symlink():
            raise IVECausalSensingInstallationError(
                "installation pointer must be a non-symlink regular file"
            )
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(source, flags)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or (path_before.st_dev, path_before.st_ino)
                != (before.st_dev, before.st_ino)
            ):
                raise IVECausalSensingInstallationError(
                    "installation pointer changed before reading"
                )
            encoded = stream.read(_MAXIMUM_POINTER_BYTES + 1)
            after = os.fstat(stream.fileno())
        path_after = source.lstat()
    except IVECausalSensingInstallationError:
        raise
    except OSError as exc:
        raise IVECausalSensingInstallationError(
            f"installation pointer cannot be read: {source}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(encoded) > _MAXIMUM_POINTER_BYTES:
        raise IVECausalSensingInstallationError(
            "installation pointer exceeds its size limit"
        )
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
    path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if (
        before_identity != after_identity
        or after_identity != path_identity
        or len(encoded) != after.st_size
        or source.is_symlink()
    ):
        raise IVECausalSensingInstallationError(
            "installation pointer changed while reading"
        )
    return encoded


def default_ive_causal_sensing_installation_paths() -> tuple[Path, ...]:
    """Return the unambiguous SSD-first and per-user pointer locations."""

    return (
        Path("/Volumes/CrossCutMedia/CrossCut/Models/CausalSensing")
        / IVE_CAUSAL_SENSING_INSTALLATION_FILENAME,
        Path.home()
        / "Library/Application Support/CrossCut/CausalSensing"
        / IVE_CAUSAL_SENSING_INSTALLATION_FILENAME,
    )


def preferred_ive_causal_sensing_installation_path() -> Path:
    """Choose the configured path, otherwise writable SSD then local storage."""

    configured = os.environ.get(IVE_CAUSAL_SENSING_INSTALLATION_ENV)
    if configured is not None:
        if not configured.strip():
            raise IVECausalSensingInstallationError(
                f"{IVE_CAUSAL_SENSING_INSTALLATION_ENV} cannot be blank"
            )
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            raise IVECausalSensingInstallationError(
                f"{IVE_CAUSAL_SENSING_INSTALLATION_ENV} must be absolute"
            )
        return candidate
    external, local = default_ive_causal_sensing_installation_paths()
    volume = Path("/Volumes/CrossCutMedia")
    return external if volume.is_dir() and os.access(volume, os.W_OK) else local


@dataclass(frozen=True, slots=True)
class InstalledIVECausalSensingGallery:
    pointer_path: Path
    pointer_file_sha256: str
    installation_sha256: str
    gallery_artifact_path: Path
    gallery_content_sha256: str
    gallery_artifact: IVEFaceGalleryArtifact

    def __post_init__(self) -> None:
        pointer = Path(self.pointer_path)
        gallery = Path(self.gallery_artifact_path)
        if not pointer.is_absolute() or not gallery.is_absolute():
            raise ValueError("installed IVE causal-sensing paths must be absolute")
        _require_sha256(self.pointer_file_sha256, "pointer_file_sha256")
        _require_sha256(self.installation_sha256, "installation_sha256")
        _require_sha256(self.gallery_content_sha256, "gallery_content_sha256")
        if not isinstance(self.gallery_artifact, IVEFaceGalleryArtifact):
            raise TypeError("gallery_artifact must be a verified gallery artifact")
        if self.gallery_artifact.content_sha256 != self.gallery_content_sha256:
            raise ValueError("verified gallery content identity differs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": IVE_CAUSAL_SENSING_INSTALLATION_CONTRACT,
            "pointer_path": str(self.pointer_path),
            "pointer_file_sha256": self.pointer_file_sha256,
            "installation_sha256": self.installation_sha256,
            "gallery_artifact_path": str(self.gallery_artifact_path),
            "gallery_content_sha256": self.gallery_content_sha256,
            "private_gallery_vectors_copied_into_pointer": False,
        }


def _installation_payload(
    gallery_artifact_path: Path,
    gallery_content_sha256: str,
) -> dict[str, Any]:
    unsigned: dict[str, Any] = {
        "schema_version": IVE_CAUSAL_SENSING_INSTALLATION_SCHEMA_VERSION,
        "contract": IVE_CAUSAL_SENSING_INSTALLATION_CONTRACT,
        "activation": IVE_CAUSAL_SENSING_INSTALLATION_ACTIVATION,
        "scope": {"artist": "IVE", "private_local_only": True},
        "gallery": {
            "artifact_path": str(gallery_artifact_path),
            "content_sha256": gallery_content_sha256,
        },
        "privacy": {
            "face_vectors_embedded": False,
            "raw_frames_embedded": False,
            "query_embeddings_embedded": False,
        },
    }
    return {**unsigned, "installation_sha256": _canonical_sha256(unsigned)}


def load_ive_causal_sensing_installation(
    pointer_path: Path,
    *,
    expected_pointer_file_sha256: str | None = None,
) -> InstalledIVECausalSensingGallery:
    """Strictly replay one pointer and its exact private gallery artifact."""

    raw_path = Path(pointer_path).expanduser()
    if not raw_path.is_absolute():
        raise IVECausalSensingInstallationError(
            "installation pointer path must be absolute"
        )
    encoded = _read_stable_regular_file(raw_path)
    pointer_file_sha256 = _file_sha256(encoded)
    if expected_pointer_file_sha256 is not None and not hmac.compare_digest(
        _require_sha256(
            expected_pointer_file_sha256,
            "expected_pointer_file_sha256",
        ),
        pointer_file_sha256,
    ):
        raise IVECausalSensingInstallationError(
            "installation pointer file SHA-256 mismatch"
        )
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IVECausalSensingInstallationError(
            "installation pointer is not canonical JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise IVECausalSensingInstallationError(
            "installation pointer root must be an object"
        )
    expected_keys = {
        "schema_version",
        "contract",
        "activation",
        "scope",
        "gallery",
        "privacy",
        "installation_sha256",
    }
    if set(payload) != expected_keys:
        raise IVECausalSensingInstallationError(
            "installation pointer fields are unsupported"
        )
    if (
        payload.get("schema_version")
        != IVE_CAUSAL_SENSING_INSTALLATION_SCHEMA_VERSION
        or payload.get("contract") != IVE_CAUSAL_SENSING_INSTALLATION_CONTRACT
        or payload.get("activation")
        != IVE_CAUSAL_SENSING_INSTALLATION_ACTIVATION
        or payload.get("scope")
        != {"artist": "IVE", "private_local_only": True}
        or payload.get("privacy")
        != {
            "face_vectors_embedded": False,
            "raw_frames_embedded": False,
            "query_embeddings_embedded": False,
        }
    ):
        raise IVECausalSensingInstallationError(
            "installation pointer policy is unsupported"
        )
    gallery = payload.get("gallery")
    if not isinstance(gallery, Mapping) or set(gallery) != {
        "artifact_path",
        "content_sha256",
    }:
        raise IVECausalSensingInstallationError(
            "installation gallery authority is invalid"
        )
    installation_sha256 = _require_sha256(
        payload.get("installation_sha256"),
        "installation_sha256",
    )
    unsigned = dict(payload)
    unsigned.pop("installation_sha256")
    if not hmac.compare_digest(
        installation_sha256,
        _canonical_sha256(unsigned),
    ):
        raise IVECausalSensingInstallationError(
            "installation pointer canonical SHA-256 mismatch"
        )
    raw_gallery_path = gallery.get("artifact_path")
    if not isinstance(raw_gallery_path, str) or not raw_gallery_path:
        raise IVECausalSensingInstallationError(
            "installation gallery path is invalid"
        )
    gallery_path = Path(raw_gallery_path).expanduser()
    if not gallery_path.is_absolute():
        raise IVECausalSensingInstallationError(
            "installation gallery path must be absolute"
        )
    gallery_content_sha256 = _require_sha256(
        gallery.get("content_sha256"),
        "gallery.content_sha256",
    )
    try:
        verified_gallery = load_ive_face_identity_gallery_artifact(
            gallery_path,
            expected_content_sha256=gallery_content_sha256,
        )
    except Exception as exc:
        raise IVECausalSensingInstallationError(
            f"installed private gallery failed strict replay: {exc}"
        ) from exc
    resolved_pointer = raw_path.resolve(strict=True)
    resolved_gallery = gallery_path.resolve(strict=True)
    if verified_gallery.path.resolve(strict=True) != resolved_gallery:
        raise IVECausalSensingInstallationError(
            "installed gallery resolved to a different artifact"
        )
    return InstalledIVECausalSensingGallery(
        pointer_path=resolved_pointer,
        pointer_file_sha256=pointer_file_sha256,
        installation_sha256=installation_sha256,
        gallery_artifact_path=resolved_gallery,
        gallery_content_sha256=gallery_content_sha256,
        gallery_artifact=verified_gallery,
    )


def discover_ive_causal_sensing_installation() -> InstalledIVECausalSensingGallery:
    """Discover exactly one configured installation and replay it strictly."""

    configured = os.environ.get(IVE_CAUSAL_SENSING_INSTALLATION_ENV)
    if configured is not None:
        if not configured.strip():
            raise IVECausalSensingInstallationError(
                f"{IVE_CAUSAL_SENSING_INSTALLATION_ENV} cannot be blank"
            )
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise IVECausalSensingInstallationError(
                f"{IVE_CAUSAL_SENSING_INSTALLATION_ENV} must be absolute"
            )
        if not path.exists() and not path.is_symlink():
            raise IVECausalSensingInstallationNotFoundError(
                "configured IVE causal-sensing installation is missing"
            )
        return load_ive_causal_sensing_installation(path)

    candidates = tuple(
        path
        for path in default_ive_causal_sensing_installation_paths()
        if path.exists() or path.is_symlink()
    )
    if not candidates:
        raise IVECausalSensingInstallationNotFoundError(
            "IVE causal-sensing private gallery is not installed"
        )
    if len(candidates) != 1:
        raise IVECausalSensingInstallationError(
            "multiple IVE causal-sensing gallery installations are active"
        )
    return load_ive_causal_sensing_installation(candidates[0])


def register_ive_causal_sensing_installation(
    *,
    gallery_artifact_path: Path,
    gallery_content_sha256: str,
    pointer_path: Path | None = None,
) -> InstalledIVECausalSensingGallery:
    """Immutably install one verified private-gallery authority pointer."""

    expected_gallery_sha256 = _require_sha256(
        gallery_content_sha256,
        "gallery_content_sha256",
    )
    raw_gallery_path = Path(gallery_artifact_path).expanduser()
    if not raw_gallery_path.is_absolute():
        raise IVECausalSensingInstallationError(
            "gallery artifact path must be absolute"
        )
    try:
        verified_gallery = load_ive_face_identity_gallery_artifact(
            raw_gallery_path,
            expected_content_sha256=expected_gallery_sha256,
        )
    except Exception as exc:
        raise IVECausalSensingInstallationError(
            f"private gallery cannot be installed: {exc}"
        ) from exc
    resolved_gallery = verified_gallery.path.resolve(strict=True)
    destination = Path(
        pointer_path
        if pointer_path is not None
        else preferred_ive_causal_sensing_installation_path()
    ).expanduser()
    if not destination.is_absolute():
        raise IVECausalSensingInstallationError(
            "installation pointer path must be absolute"
        )
    destination = destination.resolve(strict=False)
    payload = _installation_payload(
        resolved_gallery,
        expected_gallery_sha256,
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8")
    if destination.exists() or destination.is_symlink():
        installed = load_ive_causal_sensing_installation(destination)
        if (
            installed.gallery_artifact_path == resolved_gallery
            and installed.gallery_content_sha256 == expected_gallery_sha256
            and installed.installation_sha256
            == payload["installation_sha256"]
        ):
            return installed
        raise IVECausalSensingInstallationError(
            "a different IVE causal-sensing installation already exists"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.parent.is_symlink()
        or not destination.parent.is_dir()
        or destination.parent.resolve(strict=True) != destination.parent
    ):
        raise IVECausalSensingInstallationError(
            "installation pointer directory is unsafe"
        )
    lock = destination.with_name(f".{destination.name}.lock")
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    lock_descriptor: int | None = None
    owns_lock = False
    try:
        lock_descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        owns_lock = True
        os.write(lock_descriptor, b"ive-causal-sensing-installation-v1\n")
        os.fsync(lock_descriptor)
        os.close(lock_descriptor)
        lock_descriptor = None
        if destination.exists() or destination.is_symlink():
            installed = load_ive_causal_sensing_installation(destination)
            if (
                installed.gallery_artifact_path == resolved_gallery
                and installed.gallery_content_sha256 == expected_gallery_sha256
            ):
                return installed
            raise IVECausalSensingInstallationError(
                "a different IVE causal-sensing installation appeared"
            )
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        _fsync_directory(destination.parent)
    except FileExistsError as exc:
        raise IVECausalSensingInstallationError(
            "IVE causal-sensing installation publication is already active"
        ) from exc
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        if owns_lock and lock.exists() and not lock.is_symlink():
            lock.unlink()
    return load_ive_causal_sensing_installation(
        destination,
        expected_pointer_file_sha256=_file_sha256(encoded),
    )


__all__ = [
    "IVE_CAUSAL_SENSING_INSTALLATION_ACTIVATION",
    "IVE_CAUSAL_SENSING_INSTALLATION_CONTRACT",
    "IVE_CAUSAL_SENSING_INSTALLATION_ENV",
    "IVE_CAUSAL_SENSING_INSTALLATION_FILENAME",
    "IVE_CAUSAL_SENSING_INSTALLATION_SCHEMA_VERSION",
    "InstalledIVECausalSensingGallery",
    "IVECausalSensingInstallationError",
    "IVECausalSensingInstallationNotFoundError",
    "default_ive_causal_sensing_installation_paths",
    "discover_ive_causal_sensing_installation",
    "load_ive_causal_sensing_installation",
    "preferred_ive_causal_sensing_installation_path",
    "register_ive_causal_sensing_installation",
]
