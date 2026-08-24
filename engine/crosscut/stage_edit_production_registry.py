"""Immutable local registry for an active production StageEdit bundle.

The native app intentionally asks for only ``IVE`` and a song title.  Model
installation is therefore a machine-level concern, not a per-edit request
field.  This module provides one fail-closed discovery boundary:

* no registry means the optional learned editor is not installed;
* an existing registry is never ignored when it is malformed or stale;
* every checkpoint file is rebound to the hashes recorded at registration;
* the exact professional IVE montage profile is rebound by path plus file and
  canonical-content hashes on every load;
* only a Dynamic Crop v2 bundle that passes the production readiness report is
  admitted; and
* registering a different bundle never overwrites an existing active record.

Large checkpoints naturally live on the preferred external SSD.  A single
environment override remains available for development and deterministic
tests, but setting it makes the named registry mandatory rather than optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .professional_montage_policy import (
    ProfessionalMontagePolicyError,
    load_professional_montage_policy,
)
from .stage_edit_dynamic_crop_head_v2 import (
    DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME,
    DYNAMIC_CROP_V2_MODEL_CONTRACT,
    StageEditDynamicCropCheckpointMetadataV2,
)
from .stage_edit_transformer import (
    PRODUCTION_TRAINING_REPORT_FILENAME,
    StageEditReadinessReport,
    stage_edit_readiness_report,
)


STAGE_EDIT_PRODUCTION_REGISTRY_SCHEMA_VERSION = 2
STAGE_EDIT_PRODUCTION_REGISTRY_KIND = (
    "ive_stage_edit_dynamic_crop_v2_active_production_bundle"
)
STAGE_EDIT_PRODUCTION_REGISTRY_ENV = "CROSSCUT_STAGE_EDIT_PRODUCTION_REGISTRY"
STAGE_EDIT_PRODUCTION_REGISTRY_FILENAME = "active-production-bundle.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class StageEditProductionRegistryError(RuntimeError):
    """An installed registry cannot be trusted or resolved unambiguously."""


class StageEditProductionRegistryNotReadyError(StageEditProductionRegistryError):
    """A registered checkpoint no longer satisfies production readiness."""


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def default_stage_edit_production_registry_paths() -> tuple[Path, ...]:
    """Return ordered external-SSD then per-user registry locations."""

    return (
        Path("/Volumes/CrossCutMedia/CrossCut/Models/StageEdit")
        / STAGE_EDIT_PRODUCTION_REGISTRY_FILENAME,
        Path.home()
        / "Library/Application Support/CrossCut/StageEdit"
        / STAGE_EDIT_PRODUCTION_REGISTRY_FILENAME,
    )


def preferred_stage_edit_production_registry_path() -> Path:
    """Choose the SSD registry when its volume is mounted and writable."""

    configured = os.environ.get(STAGE_EDIT_PRODUCTION_REGISTRY_ENV)
    if configured is not None:
        if not configured.strip():
            raise StageEditProductionRegistryError(
                f"{STAGE_EDIT_PRODUCTION_REGISTRY_ENV} cannot be blank"
            )
        return Path(configured).expanduser().resolve()
    external, local = default_stage_edit_production_registry_paths()
    volume = Path("/Volumes/CrossCutMedia")
    if volume.is_dir() and os.access(volume, os.W_OK):
        return external
    return local


@dataclass(frozen=True, slots=True)
class RegisteredStageEditProductionBundle:
    registry_path: Path
    checkpoint_directory: Path
    checkpoint_id: str
    token_schema_fingerprint: str
    maximum_sequence_length: int
    maximum_sources: int
    professional_montage_grammar_profile_path: Path
    professional_montage_grammar_profile_file_sha256: str
    professional_montage_grammar_profile_content_sha256: str
    professional_montage_grammar_profile_analysis_basis: str
    professional_montage_grammar_profile_schema_version: int
    metadata_sha256: str
    weights_sha256: str
    training_report_sha256: str
    registry_sha256: str
    readiness: StageEditReadinessReport

    def __post_init__(self) -> None:
        registry = Path(self.registry_path)
        checkpoint = Path(self.checkpoint_directory)
        montage_profile = Path(self.professional_montage_grammar_profile_path)
        if not (
            registry.is_absolute()
            and checkpoint.is_absolute()
            and montage_profile.is_absolute()
        ):
            raise ValueError("registered StageEdit paths must be absolute")
        if not self.checkpoint_id.strip():
            raise ValueError("registered StageEdit checkpoint id cannot be blank")
        for value in (
            self.token_schema_fingerprint,
            self.professional_montage_grammar_profile_file_sha256,
            self.professional_montage_grammar_profile_content_sha256,
            self.metadata_sha256,
            self.weights_sha256,
            self.training_report_sha256,
            self.registry_sha256,
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError("registered StageEdit identity must be SHA-256")
        if self.maximum_sequence_length < 1 or self.maximum_sources < 2:
            raise ValueError("registered StageEdit model limits are invalid")
        if not self.professional_montage_grammar_profile_analysis_basis.strip():
            raise ValueError(
                "registered professional montage profile analysis basis is invalid"
            )
        if (
            isinstance(
                self.professional_montage_grammar_profile_schema_version,
                bool,
            )
            or not isinstance(
                self.professional_montage_grammar_profile_schema_version,
                int,
            )
            or self.professional_montage_grammar_profile_schema_version < 1
        ):
            raise ValueError(
                "registered professional montage profile schema version is invalid"
            )
        if not (
            self.readiness.architecture_available
            and self.readiness.learned_inference_ready
            and self.readiness.status == "production_checkpoint_ready"
            and self.readiness.checkpoint_path == str(checkpoint)
            and self.readiness.training_steps > 0
        ):
            raise ValueError("registered StageEdit bundle must be production ready")

    @property
    def professional_montage_grammar_profile_identity(self) -> dict[str, Any]:
        """Return the exact immutable montage-profile identity."""

        return {
            "path": str(self.professional_montage_grammar_profile_path),
            "analysis_basis": (
                self.professional_montage_grammar_profile_analysis_basis
            ),
            "schema_version": (
                self.professional_montage_grammar_profile_schema_version
            ),
            "file_sha256": (
                self.professional_montage_grammar_profile_file_sha256
            ),
            "content_sha256": (
                self.professional_montage_grammar_profile_content_sha256
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation": "automatic_registered_production_bundle",
            "registry_path": str(self.registry_path),
            "registry_sha256": self.registry_sha256,
            "checkpoint_directory": str(self.checkpoint_directory),
            "checkpoint_id": self.checkpoint_id,
            "token_schema_fingerprint": self.token_schema_fingerprint,
            "maximum_sequence_length": self.maximum_sequence_length,
            "maximum_sources": self.maximum_sources,
            "professional_montage_grammar_profile": (
                self.professional_montage_grammar_profile_identity
            ),
            "metadata_sha256": self.metadata_sha256,
            "weights_sha256": self.weights_sha256,
            "training_report_sha256": self.training_report_sha256,
            "readiness": self.readiness.to_dict(),
            "silent_fallback_allowed": False,
        }


def _strict_registry_payload(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StageEditProductionRegistryError(
            f"production registry is missing or unsafe: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageEditProductionRegistryError(
            f"production registry cannot be read: {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise StageEditProductionRegistryError(
            "production registry root must be an object"
        )
    expected = {
        "schema_version",
        "registry_kind",
        "active",
        "registered_at",
        "checkpoint_directory",
        "checkpoint_id",
        "model_contract",
        "token_schema_fingerprint",
        "maximum_sequence_length",
        "maximum_sources",
        "metadata",
        "weights",
        "training_report",
        "professional_montage_grammar_profile",
        "storage_policy",
        "registry_sha256",
    }
    if set(payload) != expected:
        raise StageEditProductionRegistryError(
            "production registry schema does not match"
        )
    unsigned = dict(payload)
    digest = unsigned.pop("registry_sha256", None)
    if not isinstance(digest, str) or not hmac_compare_digest(
        digest, _canonical_sha256(unsigned)
    ):
        raise StageEditProductionRegistryError(
            "production registry SHA-256 mismatch"
        )
    if (
        payload.get("schema_version")
        != STAGE_EDIT_PRODUCTION_REGISTRY_SCHEMA_VERSION
        or payload.get("registry_kind") != STAGE_EDIT_PRODUCTION_REGISTRY_KIND
        or payload.get("active") is not True
        or payload.get("model_contract") != DYNAMIC_CROP_V2_MODEL_CONTRACT
    ):
        raise StageEditProductionRegistryError(
            "production registry activation contract is unsupported"
        )
    expected_storage_policy = {
        "external_ssd_preferred": True,
        "checkpoint_absolute_path_required": True,
        "professional_montage_profile_absolute_path_required": True,
        "symlinks_allowed": False,
        "different_active_record_overwrite_allowed": False,
    }
    if payload.get("storage_policy") != expected_storage_policy:
        raise StageEditProductionRegistryError(
            "production registry storage policy is unsupported"
        )
    return payload


def hmac_compare_digest(left: str, right: str) -> bool:
    """Small local wrapper to keep digest comparison intent explicit."""

    import hmac

    return hmac.compare_digest(left, right)


def _registered_montage_profile_identity(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    raw = payload.get("professional_montage_grammar_profile")
    expected_keys = {
        "path",
        "analysis_basis",
        "schema_version",
        "file_sha256",
        "content_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise StageEditProductionRegistryError(
            "production registry professional montage profile evidence is invalid"
        )
    raw_path = raw.get("path")
    analysis_basis = raw.get("analysis_basis")
    schema_version = raw.get("schema_version")
    file_sha256 = raw.get("file_sha256")
    content_sha256 = raw.get("content_sha256")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise StageEditProductionRegistryError(
            "production registry professional montage profile path is invalid"
        )
    profile_path = Path(raw_path)
    if not profile_path.is_absolute():
        raise StageEditProductionRegistryError(
            "production registry professional montage profile path must be absolute"
        )
    if not isinstance(analysis_basis, str) or not analysis_basis.strip():
        raise StageEditProductionRegistryError(
            "production registry professional montage profile analysis basis is invalid"
        )
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 1
    ):
        raise StageEditProductionRegistryError(
            "production registry professional montage profile schema is invalid"
        )
    for name, value in (
        ("file", file_sha256),
        ("content", content_sha256),
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise StageEditProductionRegistryError(
                "production registry professional montage profile "
                f"{name} hash is invalid"
            )

    try:
        policy = load_professional_montage_policy(
            profile_path,
            expected_file_sha256=file_sha256,
        )
    except ProfessionalMontagePolicyError as exc:
        raise StageEditProductionRegistryNotReadyError(
            "registered professional montage grammar profile is missing, "
            f"changed, or invalid: {exc}"
        ) from exc
    actual = policy.profile_identity
    expected = {
        "path": raw_path,
        "analysis_basis": analysis_basis,
        "schema_version": schema_version,
        "file_sha256": file_sha256,
        "content_sha256": content_sha256,
    }
    if actual != expected:
        raise StageEditProductionRegistryNotReadyError(
            "registered professional montage grammar profile identity changed"
        )
    return expected


def load_registered_stage_edit_production_bundle(
    registry_path: Path,
) -> RegisteredStageEditProductionBundle:
    """Load and re-verify one immutable active registry record."""

    raw_path = Path(registry_path).expanduser()
    if raw_path.is_symlink():
        raise StageEditProductionRegistryError(
            f"production registry cannot be a symlink: {raw_path}"
        )
    path = raw_path.resolve()
    payload = _strict_registry_payload(path)
    raw_checkpoint = payload.get("checkpoint_directory")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint.strip():
        raise StageEditProductionRegistryError(
            "production registry checkpoint path is invalid"
        )
    raw_checkpoint_path = Path(raw_checkpoint).expanduser()
    if not raw_checkpoint_path.is_absolute():
        raise StageEditProductionRegistryError(
            "production registry checkpoint path must be absolute"
        )
    if raw_checkpoint_path.is_symlink():
        raise StageEditProductionRegistryNotReadyError(
            f"registered checkpoint cannot be a symlink: {raw_checkpoint_path}"
        )
    checkpoint = raw_checkpoint_path.resolve()
    if not checkpoint.is_dir():
        raise StageEditProductionRegistryNotReadyError(
            f"registered checkpoint is missing or unsafe: {checkpoint}"
        )

    montage_profile = _registered_montage_profile_identity(payload)

    evidence_fields = (
        (
            "metadata",
            DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
        ),
        ("weights", DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME),
        ("training_report", PRODUCTION_TRAINING_REPORT_FILENAME),
    )
    evidence: dict[str, tuple[Path, str]] = {}
    for field, expected_name in evidence_fields:
        raw = payload.get(field)
        if not isinstance(raw, Mapping) or set(raw) != {"filename", "sha256"}:
            raise StageEditProductionRegistryError(
                f"production registry {field} evidence is invalid"
            )
        if raw.get("filename") != expected_name:
            raise StageEditProductionRegistryError(
                f"production registry {field} filename is unsupported"
            )
        expected_sha = raw.get("sha256")
        if not isinstance(expected_sha, str) or _SHA256.fullmatch(expected_sha) is None:
            raise StageEditProductionRegistryError(
                f"production registry {field} hash is invalid"
            )
        file_path = checkpoint / expected_name
        if file_path.is_symlink() or not file_path.is_file():
            raise StageEditProductionRegistryNotReadyError(
                f"registered checkpoint evidence is missing: {file_path}"
            )
        actual_sha = _file_sha256(file_path)
        if not hmac_compare_digest(expected_sha, actual_sha):
            raise StageEditProductionRegistryNotReadyError(
                f"registered checkpoint evidence changed: {expected_name}"
            )
        evidence[field] = (file_path, actual_sha)

    try:
        metadata_payload = json.loads(
            evidence["metadata"][0].read_text(encoding="utf-8")
        )
        if not isinstance(metadata_payload, Mapping):
            raise ValueError("metadata root must be an object")
        metadata = StageEditDynamicCropCheckpointMetadataV2.from_dict(
            metadata_payload
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StageEditProductionRegistryNotReadyError(
            f"registered Dynamic Crop v2 metadata is invalid: {exc}"
        ) from exc

    checkpoint_id = str(payload.get("checkpoint_id") or "")
    token_schema = str(payload.get("token_schema_fingerprint") or "")
    maximum_sequence_length = payload.get("maximum_sequence_length")
    maximum_sources = payload.get("maximum_sources")
    if (
        checkpoint_id != metadata.checkpoint_id
        or token_schema != metadata.token_schema.fingerprint
        or maximum_sequence_length != metadata.base_model_config.max_sequence_length
        or maximum_sources != metadata.base_model_config.max_sources
        or evidence["weights"][1] != metadata.weight_sha256
    ):
        raise StageEditProductionRegistryNotReadyError(
            "registered checkpoint identity differs from Dynamic Crop v2 metadata"
        )

    readiness = stage_edit_readiness_report(checkpoint)
    if not (
        readiness.architecture_available
        and readiness.learned_inference_ready
        and readiness.status == "production_checkpoint_ready"
        and readiness.checkpoint_path == str(checkpoint)
        and readiness.training_steps == metadata.training_steps
        and readiness.training_steps > 0
    ):
        raise StageEditProductionRegistryNotReadyError(
            "registered Dynamic Crop v2 checkpoint is not production-ready: "
            f"{readiness.status}: {readiness.reason}"
        )

    return RegisteredStageEditProductionBundle(
        registry_path=path,
        checkpoint_directory=checkpoint,
        checkpoint_id=checkpoint_id,
        token_schema_fingerprint=token_schema,
        maximum_sequence_length=int(maximum_sequence_length),
        maximum_sources=int(maximum_sources),
        professional_montage_grammar_profile_path=Path(
            montage_profile["path"]
        ),
        professional_montage_grammar_profile_file_sha256=str(
            montage_profile["file_sha256"]
        ),
        professional_montage_grammar_profile_content_sha256=str(
            montage_profile["content_sha256"]
        ),
        professional_montage_grammar_profile_analysis_basis=str(
            montage_profile["analysis_basis"]
        ),
        professional_montage_grammar_profile_schema_version=int(
            montage_profile["schema_version"]
        ),
        metadata_sha256=evidence["metadata"][1],
        weights_sha256=evidence["weights"][1],
        training_report_sha256=evidence["training_report"][1],
        registry_sha256=str(payload["registry_sha256"]),
        readiness=readiness,
    )


def discover_registered_stage_edit_production_bundle(
) -> RegisteredStageEditProductionBundle | None:
    """Discover the one active local bundle, or return ``None`` if absent.

    An environment override is an explicit installation declaration.  A
    missing override path therefore fails instead of looking elsewhere.
    """

    configured = os.environ.get(STAGE_EDIT_PRODUCTION_REGISTRY_ENV)
    if configured is not None:
        if not configured.strip():
            raise StageEditProductionRegistryError(
                f"{STAGE_EDIT_PRODUCTION_REGISTRY_ENV} cannot be blank"
            )
        return load_registered_stage_edit_production_bundle(
            Path(configured).expanduser()
        )

    existing = tuple(
        path
        for path in default_stage_edit_production_registry_paths()
        if path.exists() or path.is_symlink()
    )
    if not existing:
        return None
    if len(existing) != 1:
        raise StageEditProductionRegistryError(
            "multiple active StageEdit production registries are installed: "
            + ", ".join(str(path) for path in existing)
        )
    return load_registered_stage_edit_production_bundle(existing[0])


def register_stage_edit_production_bundle(
    checkpoint_directory: Path,
    *,
    professional_montage_grammar_profile_path: Path,
    registry_path: Path | None = None,
) -> Path:
    """Install one verified bundle without replacing another active record."""

    raw_checkpoint = Path(checkpoint_directory).expanduser()
    if raw_checkpoint.is_symlink():
        raise StageEditProductionRegistryNotReadyError(
            "checkpoint directory cannot be a symlink"
        )
    checkpoint = raw_checkpoint.resolve()
    if not checkpoint.is_dir():
        raise StageEditProductionRegistryNotReadyError(
            "checkpoint directory is missing or unsafe"
        )
    raw_montage_profile = Path(professional_montage_grammar_profile_path)
    if not raw_montage_profile.is_absolute():
        raise StageEditProductionRegistryNotReadyError(
            "professional montage grammar profile path must be explicit and absolute"
        )
    try:
        montage_policy = load_professional_montage_policy(raw_montage_profile)
    except ProfessionalMontagePolicyError as exc:
        raise StageEditProductionRegistryNotReadyError(
            "professional montage grammar profile cannot be registered: "
            f"{exc}"
        ) from exc
    montage_profile = montage_policy.profile_identity
    if montage_profile.get("path") != str(raw_montage_profile):
        raise StageEditProductionRegistryNotReadyError(
            "professional montage grammar profile path identity changed"
        )
    metadata_path = checkpoint / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME
    weights_path = checkpoint / DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME
    report_path = checkpoint / PRODUCTION_TRAINING_REPORT_FILENAME
    for evidence_path in (metadata_path, weights_path, report_path):
        if evidence_path.is_symlink() or not evidence_path.is_file():
            raise StageEditProductionRegistryNotReadyError(
                f"checkpoint evidence is missing or unsafe: {evidence_path}"
            )
    try:
        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("metadata root must be an object")
        metadata = StageEditDynamicCropCheckpointMetadataV2.from_dict(raw_metadata)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StageEditProductionRegistryNotReadyError(
            f"Dynamic Crop v2 metadata is invalid: {exc}"
        ) from exc
    readiness = stage_edit_readiness_report(checkpoint)
    if not (
        readiness.architecture_available
        and readiness.learned_inference_ready
        and readiness.status == "production_checkpoint_ready"
        and readiness.checkpoint_path == str(checkpoint)
        and readiness.training_steps == metadata.training_steps
        and readiness.training_steps > 0
    ):
        raise StageEditProductionRegistryNotReadyError(
            "checkpoint cannot be registered before every production gate passes: "
            f"{readiness.status}: {readiness.reason}"
        )

    metadata_sha = _file_sha256(metadata_path)
    weights_sha = _file_sha256(weights_path)
    report_sha = _file_sha256(report_path)
    if metadata.weight_sha256 != weights_sha:
        raise StageEditProductionRegistryNotReadyError(
            "checkpoint metadata weight hash differs from installed weights"
        )
    destination = Path(
        registry_path or preferred_stage_edit_production_registry_path()
    ).expanduser()
    if destination.is_symlink():
        raise StageEditProductionRegistryError(
            f"production registry path cannot be a symlink: {destination}"
        )
    destination = destination.resolve()
    if destination.exists():
        installed = load_registered_stage_edit_production_bundle(destination)
        if (
            installed.checkpoint_directory == checkpoint
            and installed.checkpoint_id == metadata.checkpoint_id
            and installed.metadata_sha256 == metadata_sha
            and installed.weights_sha256 == weights_sha
            and installed.training_report_sha256 == report_sha
            and installed.professional_montage_grammar_profile_identity
            == montage_profile
        ):
            return destination
        raise StageEditProductionRegistryError(
            "a different active production registry already exists"
        )

    unsigned: dict[str, Any] = {
        "schema_version": STAGE_EDIT_PRODUCTION_REGISTRY_SCHEMA_VERSION,
        "registry_kind": STAGE_EDIT_PRODUCTION_REGISTRY_KIND,
        "active": True,
        "registered_at": datetime.now(UTC).isoformat(),
        "checkpoint_directory": str(checkpoint),
        "checkpoint_id": metadata.checkpoint_id,
        "model_contract": metadata.model_contract,
        "token_schema_fingerprint": metadata.token_schema.fingerprint,
        "maximum_sequence_length": metadata.base_model_config.max_sequence_length,
        "maximum_sources": metadata.base_model_config.max_sources,
        "metadata": {
            "filename": metadata_path.name,
            "sha256": metadata_sha,
        },
        "weights": {
            "filename": weights_path.name,
            "sha256": weights_sha,
        },
        "training_report": {
            "filename": report_path.name,
            "sha256": report_sha,
        },
        "professional_montage_grammar_profile": montage_profile,
        "storage_policy": {
            "external_ssd_preferred": True,
            "checkpoint_absolute_path_required": True,
            "professional_montage_profile_absolute_path_required": True,
            "symlinks_allowed": False,
            "different_active_record_overwrite_allowed": False,
        },
    }
    payload = {**unsigned, "registry_sha256": _canonical_sha256(unsigned)}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise StageEditProductionRegistryError(
            "production registry parent cannot be a symlink"
        )
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise StageEditProductionRegistryError(
            f"stale production registry temporary file: {temporary}"
        )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
        temporary.replace(destination)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    # Re-read every evidence path through the public loader before returning.
    load_registered_stage_edit_production_bundle(destination)
    return destination


__all__ = [
    "RegisteredStageEditProductionBundle",
    "STAGE_EDIT_PRODUCTION_REGISTRY_ENV",
    "STAGE_EDIT_PRODUCTION_REGISTRY_FILENAME",
    "STAGE_EDIT_PRODUCTION_REGISTRY_KIND",
    "STAGE_EDIT_PRODUCTION_REGISTRY_SCHEMA_VERSION",
    "StageEditProductionRegistryError",
    "StageEditProductionRegistryNotReadyError",
    "default_stage_edit_production_registry_paths",
    "discover_registered_stage_edit_production_bundle",
    "load_registered_stage_edit_production_bundle",
    "preferred_stage_edit_production_registry_path",
    "register_stage_edit_production_bundle",
]
