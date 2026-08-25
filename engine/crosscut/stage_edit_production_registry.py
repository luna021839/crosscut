"""Immutable local registry for an active production StageEdit bundle.

The native app intentionally asks for only ``IVE`` and a song title.  Model
installation is therefore a machine-level concern, not a per-edit request
field.  This module provides one fail-closed discovery boundary:

* no registry means the optional learned editor is not installed;
* an existing registry is never ignored when it is malformed or stale;
* every checkpoint file is rebound to the hashes recorded at registration;
* the exact professional IVE montage profile is rebound by path plus file and
  canonical-content hashes on every load;
* legacy Dynamic Crop v2 registries remain loadable, while v3 records one
  explicit continuity authority and binds source-pair evidence only when the
  production learned authority is selected;
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
import math
import os
from pathlib import Path
import stat as stat_module
import re
from typing import Any, Mapping, Sequence

from .professional_montage_policy import (
    ProfessionalMontagePolicyError,
    load_professional_montage_policy,
)
from .stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
    PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
    STAGE_EDIT_CONTINUITY_AUTHORITIES,
)
from .stage_edit_dynamic_crop_head_v2 import (
    DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME,
    DYNAMIC_CROP_V2_MODEL_CONTRACT,
    StageEditDynamicCropCheckpointMetadataV2,
)
from .stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
    DYNAMIC_CROP_V3_MODEL_CONTRACT,
    StageEditDynamicCropCheckpointMetadataV3,
)
from .stage_edit_source_pair_production import (
    STAGE_EDIT_SOURCE_PAIR_CROSS_VALIDATION_CONTRACT,
    STAGE_EDIT_SOURCE_PAIR_PRODUCTION_REPORT_CONTRACT,
    STAGE_EDIT_SOURCE_PAIR_PRODUCTION_REPORT_FILENAME,
    STAGE_EDIT_SOURCE_PAIR_PRODUCTION_SCHEMA_VERSION,
)
from .stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME,
    StageEditSourcePairCheckpointError,
    load_stage_edit_source_pair_checkpoint,
)
from .stage_edit_source_pair_training import (
    STAGE_EDIT_SOURCE_PAIR_ACTIVATION_GATE_CONTRACT,
)
from .stage_edit_source_pair_contextual_v2 import (
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FILENAME,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CORPUS_FILENAME,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_FILENAME,
    StageEditContextualTrainingCorpusV2,
)
from .stage_edit_source_pair_contextual_v2_production import (
    StageEditContextualSourcePairCheckpointV2,
    StageEditContextualSourcePairProductionReportV2,
    load_stage_edit_contextual_source_pair_production_bundle_v2,
)
from .stage_edit_transformer import (
    DYNAMIC_CROP_V3_TRAINING_CORPUS_CONTRACT,
    DYNAMIC_CROP_V3_VALIDATION_SELECTION_GATE_CONTRACT,
    PRODUCTION_TRAINING_REPORT_FILENAME,
    StageEditReadinessReport,
    stage_edit_readiness_report,
    valid_dynamic_crop_v3_activation_checkpointing_evidence,
)
from .stage_edit_transition_authority import (
    RENDERER_OWNED_TRANSITION_AUTHORITY,
    StageEditTransitionAuthority,
)


STAGE_EDIT_PRODUCTION_REGISTRY_SCHEMA_VERSION = 3
STAGE_EDIT_PRODUCTION_REGISTRY_KIND = (
    "ive_stage_edit_dynamic_crop_v2_active_production_bundle"
)
STAGE_EDIT_PRODUCTION_REGISTRY_V3_SCHEMA_VERSION = 4
STAGE_EDIT_PRODUCTION_REGISTRY_V3_KIND = (
    "ive_stage_edit_dynamic_crop_v3_active_production_bundle"
)
STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_V2_SCHEMA_VERSION = 5
STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_V2_KIND = (
    "ive_stage_edit_dynamic_crop_v3_contextual_v2_active_production_bundle"
)
STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_AUTHORITY_V3_SCHEMA_VERSION = 6
STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_AUTHORITY_V3_KIND = (
    "ive_stage_edit_dynamic_crop_v3_release_freeze_motion_authority_bundle"
)
STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_LOGIT_THRESHOLD_V3 = 0.0
STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_THRESHOLD_FREEZE_CONTRACT_V3 = (
    "zero_logit_predeclared_before_loso_heldout_evaluation_v1"
)
STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_BEAM_QA_MANIFEST_CONTRACT = (
    "adapter_replayable_contextual_beam_qa_registry_manifest_v1"
)
STAGE_EDIT_PRODUCTION_REGISTRY_ENV = "CROSSCUT_STAGE_EDIT_PRODUCTION_REGISTRY"
STAGE_EDIT_PRODUCTION_REGISTRY_FILENAME = "active-production-bundle.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT_OID = re.compile(r"[0-9a-f]{40}")


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


def _canonical_digest_matches(expected: object, value: Any) -> bool:
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        return False
    try:
        actual = _canonical_sha256(value)
    except (TypeError, ValueError):
        return False
    return hmac_compare_digest(expected, actual)


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
class RegisteredStageEditMotionAuthorityIdentityV3:
    """Exact LOSO release-freeze identity that may authorize v3 motion.

    This is deliberately separate from the checkpoint identity.  A v3
    checkpoint name, weight, or legacy registry record cannot manufacture
    motion authority; only the strictly replayed production-promotion lineage
    can populate this value.
    """

    release_freeze_path: Path
    release_freeze_file_sha256: str
    release_freeze_content_sha256: str
    source_commit_oid: str
    candidate_id: str
    motion_logit_threshold: float
    threshold_freeze_contract: str

    def __post_init__(self) -> None:
        path = Path(self.release_freeze_path)
        if not path.is_absolute() or path.is_symlink():
            raise ValueError(
                "registered v3 motion release-freeze path must be absolute "
                "and non-symlink"
            )
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                "registered v3 motion release-freeze path is missing"
            ) from exc
        if resolved != path or not path.is_file():
            raise ValueError(
                "registered v3 motion release-freeze path must be normalized "
                "and regular"
            )
        for value in (
            self.release_freeze_file_sha256,
            self.release_freeze_content_sha256,
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(
                    "registered v3 motion release-freeze identity must be SHA-256"
                )
        if (
            not isinstance(self.source_commit_oid, str)
            or _COMMIT_OID.fullmatch(self.source_commit_oid) is None
        ):
            raise ValueError(
                "registered v3 motion source commit must be a full commit OID"
            )
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("registered v3 motion candidate id cannot be blank")
        if (
            isinstance(self.motion_logit_threshold, bool)
            or not isinstance(self.motion_logit_threshold, float)
            or not math.isfinite(self.motion_logit_threshold)
            or self.motion_logit_threshold
            != STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_LOGIT_THRESHOLD_V3
            or math.copysign(1.0, self.motion_logit_threshold) < 0.0
        ):
            raise ValueError(
                "registered v3 motion logit threshold is not the frozen value"
            )
        if (
            self.threshold_freeze_contract
            != STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_THRESHOLD_FREEZE_CONTRACT_V3
        ):
            raise ValueError(
                "registered v3 motion threshold freeze contract is unsupported"
            )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "RegisteredStageEditMotionAuthorityIdentityV3":
        expected = {
            "release_freeze_path",
            "release_freeze_file_sha256",
            "release_freeze_content_sha256",
            "source_commit_oid",
            "candidate_id",
            "motion_logit_threshold",
            "threshold_freeze_contract",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(
                "registered v3 motion authority identity schema is invalid"
            )
        raw_path = value.get("release_freeze_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                "registered v3 motion release-freeze path is invalid"
            )
        file_sha256 = value.get("release_freeze_file_sha256")
        content_sha256 = value.get("release_freeze_content_sha256")
        source_commit_oid = value.get("source_commit_oid")
        candidate_id = value.get("candidate_id")
        motion_logit_threshold = value.get("motion_logit_threshold")
        threshold_freeze_contract = value.get("threshold_freeze_contract")
        if not all(
            isinstance(item, str)
            for item in (
                file_sha256,
                content_sha256,
                source_commit_oid,
                candidate_id,
                threshold_freeze_contract,
            )
        ) or not isinstance(motion_logit_threshold, float):
            raise ValueError(
                "registered v3 motion authority identity types are invalid"
            )
        assert isinstance(file_sha256, str)
        assert isinstance(content_sha256, str)
        assert isinstance(source_commit_oid, str)
        assert isinstance(candidate_id, str)
        assert isinstance(motion_logit_threshold, float)
        assert isinstance(threshold_freeze_contract, str)
        return cls(
            release_freeze_path=Path(raw_path),
            release_freeze_file_sha256=file_sha256,
            release_freeze_content_sha256=content_sha256,
            source_commit_oid=source_commit_oid,
            candidate_id=candidate_id,
            motion_logit_threshold=motion_logit_threshold,
            threshold_freeze_contract=threshold_freeze_contract,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_freeze_path": str(self.release_freeze_path),
            "release_freeze_file_sha256": self.release_freeze_file_sha256,
            "release_freeze_content_sha256": (
                self.release_freeze_content_sha256
            ),
            "source_commit_oid": self.source_commit_oid,
            "candidate_id": self.candidate_id,
            "motion_logit_threshold": self.motion_logit_threshold,
            "threshold_freeze_contract": self.threshold_freeze_contract,
        }


@dataclass(frozen=True, slots=True)
class RegisteredStageEditProductionBundle:
    registry_path: Path
    checkpoint_directory: Path
    checkpoint_id: str
    dynamic_crop_version: int
    continuity_authority: str | None
    token_schema_fingerprint: str
    maximum_sequence_length: int
    maximum_sources: int
    transition_authority: Mapping[str, Any]
    professional_montage_grammar_profile_path: Path
    professional_montage_grammar_profile_file_sha256: str
    professional_montage_grammar_profile_content_sha256: str
    professional_montage_grammar_profile_analysis_basis: str
    professional_montage_grammar_profile_schema_version: int
    metadata_sha256: str
    weights_sha256: str
    training_report_sha256: str
    source_pair_scorer_path: Path | None
    source_pair_production_report_path: Path | None
    source_pair_scorer_file_sha256: str | None
    source_pair_production_report_file_sha256: str | None
    source_pair_production_report_sha256: str | None
    source_pair_scorer_fingerprint: str | None
    source_pair_checkpoint_sha256: str | None
    source_pair_corpus_sha256: str | None
    contextual_source_pair_bundle_directory: Path | None
    contextual_source_pair_checkpoint_sha256: str | None
    contextual_source_pair_checkpoint_file_sha256: str | None
    contextual_source_pair_report_sha256: str | None
    contextual_source_pair_report_file_sha256: str | None
    contextual_source_pair_beam_qa_report_sha256: str | None
    contextual_source_pair_corpus_sha256: str | None
    contextual_source_pair_corpus_file_sha256: str | None
    contextual_source_pair_token_schema_fingerprint: str | None
    contextual_source_pair_scorer_fingerprint: str | None
    contextual_source_pair_verified_beam_qa_cases: tuple[object, ...]
    motion_authority_identity_v3: (
        RegisteredStageEditMotionAuthorityIdentityV3 | None
    )
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
        if (
            isinstance(self.dynamic_crop_version, bool)
            or not isinstance(self.dynamic_crop_version, int)
            or self.dynamic_crop_version not in (2, 3)
        ):
            raise ValueError("registered Dynamic Crop version is unsupported")
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
        pair_values = (
            self.source_pair_scorer_file_sha256,
            self.source_pair_production_report_file_sha256,
            self.source_pair_production_report_sha256,
            self.source_pair_scorer_fingerprint,
            self.source_pair_checkpoint_sha256,
            self.source_pair_corpus_sha256,
        )
        pair_paths = (
            self.source_pair_scorer_path,
            self.source_pair_production_report_path,
        )
        contextual_values = (
            self.contextual_source_pair_checkpoint_sha256,
            self.contextual_source_pair_checkpoint_file_sha256,
            self.contextual_source_pair_report_sha256,
            self.contextual_source_pair_report_file_sha256,
            self.contextual_source_pair_beam_qa_report_sha256,
            self.contextual_source_pair_corpus_sha256,
            self.contextual_source_pair_corpus_file_sha256,
            self.contextual_source_pair_token_schema_fingerprint,
            self.contextual_source_pair_scorer_fingerprint,
        )
        contextual_paths = (self.contextual_source_pair_bundle_directory,)
        contextual_cases = tuple(
            self.contextual_source_pair_verified_beam_qa_cases
        )
        if self.dynamic_crop_version == 2:
            if self.continuity_authority is not None:
                raise ValueError(
                    "Dynamic Crop v2 cannot bind a continuity authority"
                )
            if self.motion_authority_identity_v3 is not None:
                raise ValueError(
                    "Dynamic Crop v2 cannot bind v3 motion authority"
                )
            if any(
                value is not None
                for value in (
                    *pair_paths,
                    *pair_values,
                    *contextual_paths,
                    *contextual_values,
                )
            ) or contextual_cases:
                raise ValueError(
                    "Dynamic Crop v2 cannot bind source-pair v3 evidence"
                )
        elif (
            not isinstance(self.continuity_authority, str)
            or self.continuity_authority not in STAGE_EDIT_CONTINUITY_AUTHORITIES
        ):
            raise ValueError("Dynamic Crop v3 continuity authority is invalid")
        elif (
            self.continuity_authority
            == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
        ):
            if any(value is None for value in (*pair_paths, *pair_values)):
                raise ValueError(
                    "learned Dynamic Crop v3 continuity requires source-pair evidence"
                )
            if any(not Path(value).is_absolute() for value in pair_paths):
                raise ValueError("registered source-pair paths must be absolute")
            if any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
                for value in pair_values
            ):
                raise ValueError("registered source-pair identity must be SHA-256")
            if any(
                value is not None
                for value in (*contextual_paths, *contextual_values)
            ) or contextual_cases:
                raise ValueError(
                    "learned-v1 Dynamic Crop v3 continuity cannot bind "
                    "contextual-v2 evidence"
                )
        elif (
            self.continuity_authority
            == PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
        ):
            if any(value is not None for value in (*pair_paths, *pair_values)):
                raise ValueError(
                    "contextual-v2 Dynamic Crop v3 continuity cannot bind "
                    "learned-v1 evidence"
                )
            if any(
                value is None
                for value in (*contextual_paths, *contextual_values)
            ) or not contextual_cases:
                raise ValueError(
                    "contextual-v2 Dynamic Crop v3 continuity requires its "
                    "complete production bundle and verified Beam-QA evidence"
                )
            if not Path(self.contextual_source_pair_bundle_directory).is_absolute():
                raise ValueError(
                    "registered contextual-v2 bundle path must be absolute"
                )
            if any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
                for value in contextual_values
            ):
                raise ValueError(
                    "registered contextual-v2 identity must be SHA-256"
                )
            from .stage_edit_contextual_beam_qa_adapter_v2 import (
                is_verified_stage_edit_contextual_beam_qa_case_v2,
            )

            if any(
                not is_verified_stage_edit_contextual_beam_qa_case_v2(value)
                for value in contextual_cases
            ):
                raise ValueError(
                    "registered contextual-v2 Beam-QA cases must be "
                    "adapter-issued envelopes"
                )
        elif any(
            value is not None
            for value in (
                *pair_paths,
                *pair_values,
                *contextual_paths,
                *contextual_values,
            )
        ) or contextual_cases:
            raise ValueError(
                "handcrafted Dynamic Crop v3 continuity cannot bind "
                "source-pair evidence"
            )
        if self.maximum_sequence_length < 1 or self.maximum_sources < 2:
            raise ValueError("registered StageEdit model limits are invalid")
        if dict(self.transition_authority) != StageEditTransitionAuthority(
            mode=RENDERER_OWNED_TRANSITION_AUTHORITY
        ).to_dict():
            raise ValueError("registered transition authority is not renderer-owned")
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
            and (
                self.dynamic_crop_version == 2
                or (
                    self.readiness.dynamic_crop_version == 3
                    and self.readiness.checkpoint_format == "dynamic_crop_v3"
                    and self.readiness.checkpoint_metadata_filename
                    == DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
                )
            )
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

    @property
    def dynamic_crop_v3_motion_authority_qualified(self) -> bool:
        """Whether this registry carries replayed v3 release-freeze authority."""

        return bool(
            self.dynamic_crop_version == 3
            and self.motion_authority_identity_v3 is not None
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "activation": "automatic_registered_production_bundle",
            "registry_path": str(self.registry_path),
            "registry_sha256": self.registry_sha256,
            "checkpoint_directory": str(self.checkpoint_directory),
            "checkpoint_id": self.checkpoint_id,
            "token_schema_fingerprint": self.token_schema_fingerprint,
            "maximum_sequence_length": self.maximum_sequence_length,
            "maximum_sources": self.maximum_sources,
            "transition_authority": dict(self.transition_authority),
            "professional_montage_grammar_profile": (
                self.professional_montage_grammar_profile_identity
            ),
            "metadata_sha256": self.metadata_sha256,
            "weights_sha256": self.weights_sha256,
            "training_report_sha256": self.training_report_sha256,
            "readiness": self.readiness.to_dict(),
            "dynamic_crop_v3_motion_authority": {
                "qualified": self.dynamic_crop_v3_motion_authority_qualified,
                "legacy_without_motion_authority": (
                    self.motion_authority_identity_v3 is None
                ),
                "identity": (
                    self.motion_authority_identity_v3.to_dict()
                    if self.motion_authority_identity_v3 is not None
                    else None
                ),
            },
            "silent_fallback_allowed": False,
        }
        if self.dynamic_crop_version == 3:
            payload["dynamic_crop_version"] = 3
            payload["continuity_authority"] = self.continuity_authority
        if (
            self.continuity_authority
            == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
        ):
            payload["source_pair_scorer"] = {
                "checkpoint_path": str(self.source_pair_scorer_path),
                "production_report_path": str(
                    self.source_pair_production_report_path
                ),
                "checkpoint_file_sha256": (
                    self.source_pair_scorer_file_sha256
                ),
                "production_report_file_sha256": (
                    self.source_pair_production_report_file_sha256
                ),
                "production_report_sha256": (
                    self.source_pair_production_report_sha256
                ),
                "scorer_fingerprint": self.source_pair_scorer_fingerprint,
                "checkpoint_sha256": self.source_pair_checkpoint_sha256,
                "corpus_sha256": self.source_pair_corpus_sha256,
            }
        elif (
            self.continuity_authority
            == PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
        ):
            payload["contextual_source_pair_v2"] = {
                "bundle_directory": str(
                    self.contextual_source_pair_bundle_directory
                ),
                "checkpoint_sha256": (
                    self.contextual_source_pair_checkpoint_sha256
                ),
                "checkpoint_file_sha256": (
                    self.contextual_source_pair_checkpoint_file_sha256
                ),
                "report_sha256": self.contextual_source_pair_report_sha256,
                "report_file_sha256": (
                    self.contextual_source_pair_report_file_sha256
                ),
                "beam_qa_report_sha256": (
                    self.contextual_source_pair_beam_qa_report_sha256
                ),
                "corpus_sha256": self.contextual_source_pair_corpus_sha256,
                "corpus_file_sha256": (
                    self.contextual_source_pair_corpus_file_sha256
                ),
                "token_schema_fingerprint": (
                    self.contextual_source_pair_token_schema_fingerprint
                ),
                "scorer_fingerprint": (
                    self.contextual_source_pair_scorer_fingerprint
                ),
                "verified_beam_qa_case_count": len(
                    self.contextual_source_pair_verified_beam_qa_cases
                ),
                "verified_beam_qa_cases": [
                    {
                        "case_id": value.case.case_id,
                        "song_id": value.case.song_id,
                        "case_sha256": value.case.case_sha256,
                        "derivation_sha256": (
                            value.derivation.derivation_sha256
                        ),
                    }
                    for value in self.contextual_source_pair_verified_beam_qa_cases
                ],
            }
        return payload


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
    v2_expected = {
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
        "transition_authority",
        "metadata",
        "weights",
        "training_report",
        "professional_montage_grammar_profile",
        "storage_policy",
        "registry_sha256",
    }
    v3_expected = v2_expected | {
        "dynamic_crop_version",
        "continuity_authority",
    }
    schema_version = payload.get("schema_version")
    dynamic_crop_version = payload.get("dynamic_crop_version", 2)
    is_v2 = (
        schema_version == STAGE_EDIT_PRODUCTION_REGISTRY_SCHEMA_VERSION
        and dynamic_crop_version == 2
    )
    is_v3 = (
        schema_version == STAGE_EDIT_PRODUCTION_REGISTRY_V3_SCHEMA_VERSION
        and dynamic_crop_version == 3
    )
    is_contextual_v2 = (
        schema_version
        == STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_V2_SCHEMA_VERSION
        and dynamic_crop_version == 3
    )
    is_motion_authority_v3 = (
        schema_version
        == STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_AUTHORITY_V3_SCHEMA_VERSION
        and dynamic_crop_version == 3
    )
    if not any((is_v2, is_v3, is_contextual_v2, is_motion_authority_v3)):
        raise StageEditProductionRegistryError(
            "production registry schema does not match"
        )
    continuity_authority = (
        payload.get("continuity_authority")
        if is_v3 or is_contextual_v2 or is_motion_authority_v3
        else None
    )
    if (is_v3 or is_contextual_v2 or is_motion_authority_v3) and (
        not isinstance(continuity_authority, str)
        or continuity_authority not in STAGE_EDIT_CONTINUITY_AUTHORITIES
    ):
        raise StageEditProductionRegistryError(
            "production registry continuity authority is unsupported"
        )
    if is_v3 and continuity_authority == (
        PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
    ):
        raise StageEditProductionRegistryError(
            "contextual-v2 continuity requires its strict registry schema"
        )
    if is_contextual_v2 and continuity_authority != (
        PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
    ):
        raise StageEditProductionRegistryError(
            "contextual-v2 registry continuity authority is unsupported"
        )
    if continuity_authority == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY:
        v3_expected |= {
            "source_pair_scorer",
            "source_pair_production_report",
        }
    contextual_expected = v3_expected | {
        "contextual_source_pair_v2",
        "contextual_beam_qa_cases",
    }
    motion_authority_expected = (
        contextual_expected
        if continuity_authority
        == PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
        else v3_expected
    ) | {"dynamic_crop_v3_motion_authority"}
    expected = (
        v2_expected
        if is_v2
        else contextual_expected
        if is_contextual_v2
        else motion_authority_expected
        if is_motion_authority_v3
        else v3_expected
        if is_v3
        else set()
    )
    if set(payload) != expected:
        raise StageEditProductionRegistryError(
            "production registry schema does not match"
        )
    unsigned = dict(payload)
    digest = unsigned.pop("registry_sha256", None)
    if not _canonical_digest_matches(digest, unsigned):
        raise StageEditProductionRegistryError(
            "production registry SHA-256 mismatch"
        )
    if is_v2:
        supported_contract = bool(
            payload.get("registry_kind")
            == STAGE_EDIT_PRODUCTION_REGISTRY_KIND
            and payload.get("model_contract")
            == DYNAMIC_CROP_V2_MODEL_CONTRACT
        )
    elif is_contextual_v2:
        supported_contract = bool(
            payload.get("registry_kind")
            == STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_V2_KIND
            and payload.get("model_contract") == DYNAMIC_CROP_V3_MODEL_CONTRACT
        )
    elif is_v3:
        supported_contract = bool(
            payload.get("registry_kind")
            == STAGE_EDIT_PRODUCTION_REGISTRY_V3_KIND
            and payload.get("model_contract")
            == DYNAMIC_CROP_V3_MODEL_CONTRACT
        )
    elif is_motion_authority_v3:
        supported_contract = bool(
            payload.get("registry_kind")
            == STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_AUTHORITY_V3_KIND
            and payload.get("model_contract")
            == DYNAMIC_CROP_V3_MODEL_CONTRACT
        )
    else:  # The exact schema check above already rejects this branch.
        supported_contract = False
    if payload.get("active") is not True or not supported_contract:
        raise StageEditProductionRegistryError(
            "production registry activation contract is unsupported"
        )
    if payload.get("transition_authority") != StageEditTransitionAuthority(
        mode=RENDERER_OWNED_TRANSITION_AUTHORITY
    ).to_dict():
        raise StageEditProductionRegistryError(
            "production registry transition authority is unsupported"
        )
    expected_storage_policy: dict[str, Any] = {
        "external_ssd_preferred": True,
        "checkpoint_absolute_path_required": True,
        "professional_montage_profile_absolute_path_required": True,
        "symlinks_allowed": False,
        "different_active_record_overwrite_allowed": False,
    }
    if continuity_authority == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY:
        expected_storage_policy["source_pair_absolute_paths_required"] = True
    elif continuity_authority == PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2:
        expected_storage_policy[
            "contextual_source_pair_bundle_absolute_path_required"
        ] = True
        expected_storage_policy[
            "contextual_beam_qa_artifact_absolute_paths_required"
        ] = True
    if is_motion_authority_v3:
        expected_storage_policy[
            "dynamic_crop_v3_release_freeze_absolute_path_required"
        ] = True
    if payload.get("storage_policy") != expected_storage_policy:
        raise StageEditProductionRegistryError(
            "production registry storage policy is unsupported"
        )
    return payload


def hmac_compare_digest(left: str, right: str) -> bool:
    """Small local wrapper to keep digest comparison intent explicit."""

    import hmac

    return hmac.compare_digest(left, right)


def _checkpoint_namespace_version(
    checkpoint: Path,
    *,
    expected_version: int,
) -> int:
    namespaces = {
        2: (
            checkpoint / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
            checkpoint / DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME,
        ),
        3: (
            checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
            checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
        ),
    }
    present = {
        version: tuple(path.exists() or path.is_symlink() for path in paths)
        for version, paths in namespaces.items()
    }
    if any(any(values) and not all(values) for values in present.values()):
        raise StageEditProductionRegistryNotReadyError(
            "Dynamic Crop checkpoint namespace is partial"
        )
    complete = tuple(
        version for version, values in present.items() if all(values)
    )
    if len(complete) != 1:
        raise StageEditProductionRegistryNotReadyError(
            "Dynamic Crop v2/v3 checkpoint namespaces are mixed or missing"
        )
    observed = complete[0]
    if observed != expected_version:
        raise StageEditProductionRegistryNotReadyError(
            f"Dynamic Crop v{observed} checkpoint cannot be interpreted as "
            f"v{expected_version}"
        )
    return observed


def _strict_json_mapping(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageEditProductionRegistryNotReadyError(
            f"{name} cannot be read: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise StageEditProductionRegistryNotReadyError(
            f"{name} root must be an object"
        )
    return value


def _valid_v3_selection_gate(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected_keys = {
        "contract",
        "selectable",
        "reasons",
        "score_component",
        "renderer_decode_attempt_count",
        "renderer_decode_eligible_count",
        "renderer_decode_eligible_rate",
        "minimum_safe_zoom_projection_count",
        "target_source_slot_mismatch_count",
        "aggregate_normalized_improvement_over_static_prior",
        "static_prior_can_be_selected",
        "heldout_metrics_consumed",
    }
    attempts = value.get("renderer_decode_attempt_count")
    eligible = value.get("renderer_decode_eligible_count")
    rate = value.get("renderer_decode_eligible_rate")
    improvement = value.get(
        "aggregate_normalized_improvement_over_static_prior"
    )
    score = value.get("score_component")
    return bool(
        set(value) == expected_keys
        and value.get("contract")
        == DYNAMIC_CROP_V3_VALIDATION_SELECTION_GATE_CONTRACT
        and value.get("selectable") is True
        and value.get("reasons") == []
        and value.get("heldout_metrics_consumed") is False
        and value.get("static_prior_can_be_selected") is False
        and isinstance(attempts, int)
        and not isinstance(attempts, bool)
        and attempts > 0
        and isinstance(eligible, int)
        and not isinstance(eligible, bool)
        and eligible == attempts
        and isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and math.isfinite(float(rate))
        and float(rate) == 1.0
        and isinstance(
            value.get("minimum_safe_zoom_projection_count"), int
        )
        and not isinstance(
            value.get("minimum_safe_zoom_projection_count"), bool
        )
        and value.get("minimum_safe_zoom_projection_count") == 0
        and isinstance(value.get("target_source_slot_mismatch_count"), int)
        and not isinstance(
            value.get("target_source_slot_mismatch_count"), bool
        )
        and value.get("target_source_slot_mismatch_count") == 0
        and isinstance(improvement, (int, float))
        and not isinstance(improvement, bool)
        and math.isfinite(float(improvement))
        and float(improvement) > 0.0
        and isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(float(score))
        and float(score) == float(improvement)
    )


def _validate_v3_training_report_declarations(
    report: Mapping[str, Any],
    *,
    metadata: StageEditDynamicCropCheckpointMetadataV3,
    weight_sha256: str,
    checkpoint_directory: Path | None = None,
) -> None:
    # A promoted all-reference final-fit weight cannot truthfully satisfy the
    # per-fold report's "same weight had one held-out song" contract.  Admit
    # only the alternate exact run kind through its own strict validator;
    # every other report continues through the unchanged fold declarations.
    from .stage_edit_dynamic_crop_v3_production_promotion import (
        PRODUCTION_PROMOTION_RUN_KIND,
        production_promotion_report_errors,
    )

    if report.get("run_kind") == PRODUCTION_PROMOTION_RUN_KIND:
        errors = production_promotion_report_errors(
            report,
            metadata=metadata,
            actual_weight_sha256=weight_sha256,
            checkpoint_directory=checkpoint_directory,
        )
        if errors:
            raise StageEditProductionRegistryNotReadyError("; ".join(errors))
        return
    checkpoint = report.get("checkpoint")
    model = report.get("model")
    data = report.get("data")
    optimization = report.get("optimization")
    heldout = report.get("held_out_evaluation")
    if not all(
        isinstance(value, Mapping)
        for value in (checkpoint, model, data, optimization, heldout)
    ):
        raise StageEditProductionRegistryNotReadyError(
            "Dynamic Crop v3 production report declarations are incomplete"
        )
    assert isinstance(checkpoint, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(data, Mapping)
    assert isinstance(optimization, Mapping)
    assert isinstance(heldout, Mapping)
    checkpoint_expected = {
        "checkpoint_id": metadata.checkpoint_id,
        "training_steps": metadata.training_steps,
        "weight_sha256": weight_sha256,
        "production_activation_authorized": True,
        "production_supervision_gate_passed": True,
        "checkpoint_format": "dynamic_crop_v3",
        "dynamic_crop_version": 3,
        "metadata_filename": DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
        "dynamic_crop_v2_checkpoint": False,
        "dynamic_crop_v3_checkpoint": True,
        "dynamic_crop_model_contract": metadata.model_contract,
        "dynamic_crop_target_contract": metadata.target_contract,
        "dynamic_crop_output_contract": metadata.output_contract,
    }
    if any(
        checkpoint.get(name) != value
        for name, value in checkpoint_expected.items()
    ):
        raise StageEditProductionRegistryNotReadyError(
            "production training report does not declare the exact v3 checkpoint"
        )
    data_id = checkpoint.get("dynamic_crop_v3_data_id")
    if not isinstance(data_id, str) or not data_id.strip():
        raise StageEditProductionRegistryNotReadyError(
            "production training report lacks the v3 corpus id"
        )
    if (
        model.get("dynamic_crop_version") != 3
        or model.get("dynamic_crop_v2_enabled") is not False
        or model.get("dynamic_crop_v3_enabled") is not True
        or data.get("dynamic_crop_version") != 3
        or data.get("dynamic_crop_v2") is not None
    ):
        raise StageEditProductionRegistryNotReadyError(
            "production training report mixes v2 and v3 declarations"
        )
    if not valid_dynamic_crop_v3_activation_checkpointing_evidence(
        model.get("activation_checkpointing")
    ):
        raise StageEditProductionRegistryNotReadyError(
            "production training report activation checkpointing evidence differs"
        )
    dynamic_data = data.get("dynamic_crop_v3")
    if (
        not isinstance(dynamic_data, Mapping)
        or dynamic_data.get("data_id") != data_id
        or dynamic_data.get("contract") != DYNAMIC_CROP_V3_TRAINING_CORPUS_CONTRACT
        or dynamic_data.get("target_contract") != metadata.target_contract
    ):
        raise StageEditProductionRegistryNotReadyError(
            "production training report v3 corpus declaration is invalid"
        )
    selected = optimization.get("best_validation_selection_evidence")
    reloaded = optimization.get(
        "reloaded_checkpoint_validation_selection_evidence"
    )
    if (
        optimization.get("dynamic_crop_v3_model_selection")
        != DYNAMIC_CROP_V3_VALIDATION_SELECTION_GATE_CONTRACT
        or not isinstance(selected, Mapping)
        or not isinstance(reloaded, Mapping)
        or not _valid_v3_selection_gate(
            selected.get("dynamic_crop_v3_selection_gate")
        )
        or not _valid_v3_selection_gate(
            reloaded.get("dynamic_crop_v3_selection_gate")
        )
    ):
        raise StageEditProductionRegistryNotReadyError(
            "production training report v3 validation gate is invalid"
        )
    if (
        heldout.get("passed") is not True
        or heldout.get("dynamic_crop_version") != 3
        or heldout.get("dynamic_crop_v2_metrics") is not None
        or not isinstance(heldout.get("dynamic_crop_v3_metrics"), Mapping)
        or heldout.get(
            "heldout_dynamic_crop_metrics_used_for_checkpoint_selection"
        )
        is not False
    ):
        raise StageEditProductionRegistryNotReadyError(
            "production training report v3 held-out evidence is invalid"
        )


def _motion_authority_identity_from_verified_v3_training_report(
    report: Mapping[str, Any],
) -> RegisteredStageEditMotionAuthorityIdentityV3 | None:
    """Replay and derive motion identity after strict report validation.

    The caller must first pass ``report`` through
    ``_validate_v3_training_report_declarations``.  That validates the complete
    production-promotion lineage against the exact installed checkpoint.  We
    then independently replay the release freeze here and bind the small
    runtime-facing identity; no checkpoint filename or id participates in the
    authority decision.
    """

    from .stage_edit_dynamic_crop_v3_production_promotion import (
        PRODUCTION_PROMOTION_RUN_KIND,
    )

    if report.get("run_kind") != PRODUCTION_PROMOTION_RUN_KIND:
        return None
    from .stage_edit_dynamic_crop_v3_final_release_launcher import (
        load_dynamic_crop_v3_release_freeze,
    )

    lineage = report.get("lineage")
    if not isinstance(lineage, Mapping):
        raise StageEditProductionRegistryNotReadyError(
            "production-promotion report lacks release-freeze lineage"
        )
    pointer = lineage.get("release_freeze")
    if not isinstance(pointer, Mapping) or set(pointer) != {
        "path",
        "file_sha256",
    }:
        raise StageEditProductionRegistryNotReadyError(
            "production-promotion release-freeze pointer is invalid"
        )
    raw_path_value = pointer.get("path")
    pointer_file_sha256 = pointer.get("file_sha256")
    if (
        not isinstance(raw_path_value, str)
        or not raw_path_value.strip()
        or not isinstance(pointer_file_sha256, str)
        or _SHA256.fullmatch(pointer_file_sha256) is None
    ):
        raise StageEditProductionRegistryNotReadyError(
            "production-promotion release-freeze identity is invalid"
        )
    raw_path = Path(raw_path_value)
    if not raw_path.is_absolute() or raw_path.is_symlink():
        raise StageEditProductionRegistryNotReadyError(
            "production-promotion release-freeze path must be absolute and "
            "non-symlink"
        )
    try:
        release_freeze_path = raw_path.resolve(strict=True)
    except OSError as exc:
        raise StageEditProductionRegistryNotReadyError(
            f"production-promotion release freeze is missing: {exc}"
        ) from exc
    if release_freeze_path != raw_path or not release_freeze_path.is_file():
        raise StageEditProductionRegistryNotReadyError(
            "production-promotion release-freeze path is not normalized or safe"
        )
    before = release_freeze_path.lstat()
    if not stat_module.S_ISREG(before.st_mode):
        raise StageEditProductionRegistryNotReadyError(
            "production-promotion release freeze must be a regular file"
        )
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    actual_file_sha256 = _file_sha256(release_freeze_path)
    if not hmac_compare_digest(pointer_file_sha256, actual_file_sha256):
        raise StageEditProductionRegistryNotReadyError(
            "production-promotion release-freeze file identity changed"
        )
    try:
        freeze = load_dynamic_crop_v3_release_freeze(release_freeze_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise StageEditProductionRegistryNotReadyError(
            f"production-promotion release-freeze replay failed: {exc}"
        ) from exc
    try:
        after = release_freeze_path.lstat()
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        after_file_sha256 = _file_sha256(release_freeze_path)
    except OSError as exc:
        raise StageEditProductionRegistryNotReadyError(
            f"production-promotion release freeze changed during replay: {exc}"
        ) from exc
    if (
        not stat_module.S_ISREG(after.st_mode)
        or after_identity != before_identity
        or not hmac_compare_digest(actual_file_sha256, after_file_sha256)
    ):
        raise StageEditProductionRegistryNotReadyError(
            "production-promotion release freeze changed during replay"
        )
    source = freeze.get("source_provenance")
    source_commit_oid = (
        source.get("head_commit_oid")
        if isinstance(source, Mapping)
        else None
    )
    release_freeze_content_sha256 = freeze.get("freeze_content_sha256")
    candidate_id = freeze.get("candidate_id")
    if (
        not isinstance(release_freeze_content_sha256, str)
        or _SHA256.fullmatch(release_freeze_content_sha256) is None
        or lineage.get("release_freeze_content_sha256")
        != release_freeze_content_sha256
        or not isinstance(source_commit_oid, str)
        or _COMMIT_OID.fullmatch(source_commit_oid) is None
        or report.get("source_commit") != source_commit_oid
        or not isinstance(candidate_id, str)
        or not candidate_id.strip()
    ):
        raise StageEditProductionRegistryNotReadyError(
            "production-promotion release-freeze lineage identity differs"
        )
    return RegisteredStageEditMotionAuthorityIdentityV3(
        release_freeze_path=release_freeze_path,
        release_freeze_file_sha256=actual_file_sha256,
        release_freeze_content_sha256=release_freeze_content_sha256,
        source_commit_oid=source_commit_oid,
        candidate_id=candidate_id,
        motion_logit_threshold=(
            STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_LOGIT_THRESHOLD_V3
        ),
        threshold_freeze_contract=(
            STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_THRESHOLD_FREEZE_CONTRACT_V3
        ),
    )


def _validate_source_pair_production_report(
    report_path: Path,
    *,
    checkpoint: Any,
) -> tuple[str, str]:
    report = _strict_json_mapping(
        report_path, "source-pair production report"
    )
    expected_keys = {
        "schema_version",
        "contract",
        "cross_validation",
        "final_full_corpus_training",
        "checkpoint",
        "production_authorization",
        "report_sha256",
    }
    if set(report) != expected_keys:
        raise StageEditProductionRegistryNotReadyError(
            "source-pair production report schema is invalid"
        )
    unsigned_report = dict(report)
    report_sha256 = unsigned_report.pop("report_sha256", None)
    if (
        not _canonical_digest_matches(report_sha256, unsigned_report)
        or report.get("schema_version")
        != STAGE_EDIT_SOURCE_PAIR_PRODUCTION_SCHEMA_VERSION
        or report.get("contract")
        != STAGE_EDIT_SOURCE_PAIR_PRODUCTION_REPORT_CONTRACT
    ):
        raise StageEditProductionRegistryNotReadyError(
            "source-pair production report self-hash or contract is invalid"
        )
    if report.get("checkpoint") != checkpoint.to_dict():
        raise StageEditProductionRegistryNotReadyError(
            "source-pair report checkpoint does not round-trip exactly"
        )
    final_training = report.get("final_full_corpus_training")
    final_training_keys = {
        "training_steps",
        "reference_count",
        "group_count",
        "initial_loss",
        "final_loss",
        "margin",
        "corpus_sha256",
        "scorer",
    }
    if (
        not isinstance(final_training, Mapping)
        or set(final_training) != final_training_keys
        or final_training.get("corpus_sha256") != checkpoint.corpus_sha256
        or final_training.get("scorer") != checkpoint.scorer.to_dict()
        or final_training.get("training_steps") != checkpoint.training_steps
        or final_training.get("reference_count") != checkpoint.reference_count
        or final_training.get("group_count") != checkpoint.group_count
    ):
        raise StageEditProductionRegistryNotReadyError(
            "source-pair corpus or scorer does not round-trip exactly"
        )
    cross_validation = report.get("cross_validation")
    expected_cross_validation_keys = {
        "schema_version",
        "contract",
        "protocol",
        "configuration",
        "source_corpus",
        "folds",
        "reference_balanced_heldout_aggregate",
        "aggregate_activation_gate",
        "production_authorized",
        "production_denial_reasons",
        "leakage_contract",
        "validation_sha256",
    }
    if (
        not isinstance(cross_validation, Mapping)
        or set(cross_validation) != expected_cross_validation_keys
    ):
        raise StageEditProductionRegistryNotReadyError(
            "source-pair cross-validation evidence schema is invalid"
        )
    unsigned_validation = dict(cross_validation)
    validation_sha256 = unsigned_validation.pop("validation_sha256", None)
    folds = cross_validation.get("folds")
    aggregate_gate = cross_validation.get("aggregate_activation_gate")
    source_corpus = cross_validation.get("source_corpus")
    configuration = cross_validation.get("configuration")
    aggregate_metrics = cross_validation.get(
        "reference_balanced_heldout_aggregate"
    )
    fold_keys = {
        "fold_index",
        "heldout_song_id",
        "training_song_ids",
        "training_reference_ids",
        "heldout_reference_ids",
        "training_group_ids",
        "heldout_group_ids",
        "training",
        "heldout_ranking",
        "activation_gate",
    }
    folds_are_structurally_valid = bool(
        isinstance(folds, list)
        and folds
        and all(
            isinstance(fold, Mapping)
            and set(fold) == fold_keys
            and isinstance(fold.get("fold_index"), int)
            and not isinstance(fold.get("fold_index"), bool)
            and isinstance(fold.get("heldout_song_id"), str)
            and bool(str(fold.get("heldout_song_id")).strip())
            for fold in folds
        )
        and tuple(fold.get("fold_index") for fold in folds)
        == tuple(range(len(folds)))
    )
    fold_gates = (
        [fold.get("activation_gate") for fold in folds]
        if folds_are_structurally_valid and isinstance(folds, list)
        else []
    )
    all_fold_gates = bool(fold_gates) and all(
        isinstance(gate, Mapping)
        and set(gate) == {"contract", "authorized", "reasons"}
        and gate.get("contract")
        == STAGE_EDIT_SOURCE_PAIR_ACTIVATION_GATE_CONTRACT
        and gate.get("authorized") is True
        and gate.get("reasons") == []
        for gate in fold_gates
    )
    aggregate_authorized = bool(
        isinstance(aggregate_gate, Mapping)
        and set(aggregate_gate) == {"contract", "authorized", "reasons"}
        and aggregate_gate.get("contract")
        == STAGE_EDIT_SOURCE_PAIR_ACTIVATION_GATE_CONTRACT
        and aggregate_gate.get("authorized") is True
        and aggregate_gate.get("reasons") == []
    )
    expected_leakage_contract = {
        "materialized_once_before_folds": True,
        "training_and_heldout_songs_disjoint": True,
        "training_and_heldout_references_disjoint": True,
        "heldout_groups_used_for_training": False,
        "every_song_held_out_once": True,
        "every_reference_aggregated_once": True,
        "in_sample_fit_gate_used_for_authorization": False,
    }
    source_corpus_matches_checkpoint = bool(
        isinstance(source_corpus, Mapping)
        and isinstance(folds, list)
        and isinstance(source_corpus.get("song_count"), int)
        and not isinstance(source_corpus.get("song_count"), bool)
        and isinstance(source_corpus.get("reference_count"), int)
        and not isinstance(source_corpus.get("reference_count"), bool)
        and isinstance(source_corpus.get("group_count"), int)
        and not isinstance(source_corpus.get("group_count"), bool)
        and source_corpus.get("song_count") == len(folds)
        and source_corpus.get("reference_count") == checkpoint.reference_count
        and source_corpus.get("group_count") == checkpoint.group_count
        and source_corpus.get("song_ids")
        == [fold.get("heldout_song_id") for fold in folds]
    )
    if (
        not _canonical_digest_matches(validation_sha256, unsigned_validation)
        or cross_validation.get("schema_version")
        != STAGE_EDIT_SOURCE_PAIR_PRODUCTION_SCHEMA_VERSION
        or cross_validation.get("contract")
        != STAGE_EDIT_SOURCE_PAIR_CROSS_VALIDATION_CONTRACT
        or cross_validation.get("protocol")
        != "song_disjoint_leave_one_song_out"
        or not isinstance(configuration, Mapping)
        or set(configuration) != {"learning_rate", "activation_tolerance"}
        or not isinstance(aggregate_metrics, Mapping)
        or set(aggregate_metrics) != {"learned", "uniform", "handcrafted"}
        or cross_validation.get("production_authorized") is not True
        or cross_validation.get("production_denial_reasons") != []
        or cross_validation.get("leakage_contract")
        != expected_leakage_contract
        or not folds_are_structurally_valid
        or not source_corpus_matches_checkpoint
        or not all_fold_gates
        or not aggregate_authorized
    ):
        raise StageEditProductionRegistryNotReadyError(
            "source-pair all-fold or aggregate production gate is invalid"
        )
    authorization = report.get("production_authorization")
    expected_authorization = {
        "authorized": True,
        "all_fold_gates_authorized": all_fold_gates,
        "overall_gate_authorized": aggregate_authorized,
        "final_scorer_trained_after_cross_validation_gate": True,
        "in_sample_fit_gate_used_for_authorization": False,
        "materialization_pass_count": 1,
    }
    if authorization != expected_authorization:
        raise StageEditProductionRegistryNotReadyError(
            "source-pair production authorization does not round-trip"
        )
    return report_sha256, validation_sha256


def _validated_source_pair_identity(
    scorer_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    scorer = Path(scorer_path)
    report = Path(report_path)
    for path, expected_name in (
        (scorer, STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME),
        (report, STAGE_EDIT_SOURCE_PAIR_PRODUCTION_REPORT_FILENAME),
    ):
        if not path.is_absolute():
            raise StageEditProductionRegistryNotReadyError(
                "source-pair artifact paths must be explicit and absolute"
            )
        if path.name != expected_name or path.is_symlink() or not path.is_file():
            raise StageEditProductionRegistryNotReadyError(
                f"source-pair artifact is missing or unsafe: {path}"
            )
    try:
        pair_scorer, pair_checkpoint = load_stage_edit_source_pair_checkpoint(
            scorer.parent
        )
    except (StageEditSourcePairCheckpointError, TypeError, ValueError) as exc:
        raise StageEditProductionRegistryNotReadyError(
            f"source-pair scorer checkpoint is invalid: {exc}"
        ) from exc
    report_sha256, validation_sha256 = _validate_source_pair_production_report(
        report,
        checkpoint=pair_checkpoint,
    )
    return {
        "scorer_path": str(scorer),
        "production_report_path": str(report),
        "scorer_file_sha256": _file_sha256(scorer),
        "production_report_file_sha256": _file_sha256(report),
        "production_report_sha256": report_sha256,
        "cross_validation_sha256": validation_sha256,
        "scorer_fingerprint": pair_scorer.fingerprint,
        "checkpoint_sha256": pair_checkpoint.checkpoint_sha256,
        "corpus_sha256": pair_checkpoint.corpus_sha256,
    }


def _registered_source_pair_identity(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    raw_scorer = payload.get("source_pair_scorer")
    raw_report = payload.get("source_pair_production_report")
    scorer_keys = {
        "path",
        "sha256",
        "scorer_fingerprint",
        "checkpoint_sha256",
        "corpus_sha256",
    }
    report_keys = {
        "path",
        "sha256",
        "report_sha256",
        "cross_validation_sha256",
    }
    if (
        not isinstance(raw_scorer, Mapping)
        or set(raw_scorer) != scorer_keys
        or not isinstance(raw_report, Mapping)
        or set(raw_report) != report_keys
    ):
        raise StageEditProductionRegistryError(
            "production registry source-pair evidence is invalid"
        )
    scorer_path = raw_scorer.get("path")
    report_path = raw_report.get("path")
    if (
        not isinstance(scorer_path, str)
        or not scorer_path.strip()
        or not Path(scorer_path).is_absolute()
        or not isinstance(report_path, str)
        or not report_path.strip()
        or not Path(report_path).is_absolute()
    ):
        raise StageEditProductionRegistryError(
            "production registry source-pair paths must be absolute"
        )
    actual = _validated_source_pair_identity(
        Path(scorer_path), Path(report_path)
    )
    expected = {
        "scorer_path": scorer_path,
        "production_report_path": report_path,
        "scorer_file_sha256": raw_scorer.get("sha256"),
        "production_report_file_sha256": raw_report.get("sha256"),
        "production_report_sha256": raw_report.get("report_sha256"),
        "cross_validation_sha256": raw_report.get("cross_validation_sha256"),
        "scorer_fingerprint": raw_scorer.get("scorer_fingerprint"),
        "checkpoint_sha256": raw_scorer.get("checkpoint_sha256"),
        "corpus_sha256": raw_scorer.get("corpus_sha256"),
    }
    if actual != expected:
        raise StageEditProductionRegistryNotReadyError(
            "registered source-pair artifact identity changed"
        )
    return actual


def _contextual_beam_qa_manifest_from_verified_cases(
    verified_cases: Sequence[object],
) -> dict[str, Any]:
    """Serialize only adapter-issued evidence needed for restart replay."""

    from .stage_edit_contextual_beam_qa_adapter_v2 import (
        require_verified_stage_edit_contextual_beam_qa_case_v2,
    )

    if isinstance(verified_cases, (str, bytes)) or not isinstance(
        verified_cases, Sequence
    ):
        raise StageEditProductionRegistryNotReadyError(
            "contextual-v2 Beam-QA evidence must be a sequence of verified envelopes"
        )
    ordered = tuple(
        sorted(
            tuple(verified_cases),
            key=lambda value: (
                getattr(getattr(value, "case", None), "song_id", ""),
                getattr(getattr(value, "case", None), "case_id", ""),
            ),
        )
    )
    if not ordered:
        raise StageEditProductionRegistryNotReadyError(
            "contextual-v2 Beam-QA evidence cannot be empty"
        )
    entries: list[dict[str, Any]] = []
    case_sha256s: list[str] = []
    for value in ordered:
        try:
            case = require_verified_stage_edit_contextual_beam_qa_case_v2(value)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise StageEditProductionRegistryNotReadyError(
                "contextual-v2 registration rejects raw or stale Beam-QA DTOs: "
                f"{exc}"
            ) from exc
        derivation = getattr(value, "derivation", None)
        if derivation is None or not callable(getattr(derivation, "to_dict", None)):
            raise StageEditProductionRegistryNotReadyError(
                "contextual-v2 verified Beam-QA envelope lacks derivation evidence"
            )
        if not isinstance(case.case_sha256, str):
            raise StageEditProductionRegistryNotReadyError(
                "contextual-v2 verified Beam-QA case lacks its digest"
            )
        case_sha256s.append(case.case_sha256)
        entries.append(
            {
                "case": case.to_dict(),
                "derivation": derivation.to_dict(),
            }
        )
    if len(case_sha256s) != len(set(case_sha256s)):
        raise StageEditProductionRegistryNotReadyError(
            "contextual-v2 Beam-QA case identities must be unique"
        )
    return {
        "contract": (
            STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_BEAM_QA_MANIFEST_CONTRACT
        ),
        "cases": entries,
    }


def _validated_contextual_source_pair_identity(
    bundle_directory: Path,
    verified_beam_qa_cases: Sequence[object],
) -> dict[str, Any]:
    """Replay one complete contextual-v2 bundle from fixed sibling artifacts."""

    if isinstance(verified_beam_qa_cases, (str, bytes)) or not isinstance(
        verified_beam_qa_cases, Sequence
    ):
        raise StageEditProductionRegistryNotReadyError(
            "contextual-v2 Beam-QA evidence must be a sequence of verified envelopes"
        )
    ordered_cases = tuple(
        sorted(
            tuple(verified_beam_qa_cases),
            key=lambda value: (
                getattr(getattr(value, "case", None), "song_id", ""),
                getattr(getattr(value, "case", None), "case_id", ""),
            ),
        )
    )
    raw_directory = Path(bundle_directory)
    raw_spelling = os.fspath(raw_directory)
    if (
        not raw_directory.is_absolute()
        or os.path.normpath(raw_spelling) != raw_spelling
        or raw_directory.is_symlink()
        or not raw_directory.is_dir()
    ):
        raise StageEditProductionRegistryNotReadyError(
            "contextual-v2 bundle directory must be one absolute, normalized, "
            "non-symlink directory"
        )
    artifact_paths = {
        "corpus": raw_directory
        / STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CORPUS_FILENAME,
        "checkpoint": raw_directory
        / STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FILENAME,
        "report": raw_directory
        / STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_FILENAME,
    }
    for name, path in artifact_paths.items():
        if path.is_symlink() or not path.is_file():
            raise StageEditProductionRegistryNotReadyError(
                f"contextual-v2 {name} artifact is missing or unsafe: {path}"
            )
    try:
        corpus = StageEditContextualTrainingCorpusV2.from_dict(
            _strict_json_mapping(
                artifact_paths["corpus"], "contextual-v2 training corpus"
            )
        )
        checkpoint = StageEditContextualSourcePairCheckpointV2.from_dict(
            _strict_json_mapping(
                artifact_paths["checkpoint"], "contextual-v2 checkpoint"
            )
        )
        report = StageEditContextualSourcePairProductionReportV2.from_dict(
            _strict_json_mapping(
                artifact_paths["report"], "contextual-v2 production report"
            )
        )
        file_sha256s = {
            name: _file_sha256(path) for name, path in artifact_paths.items()
        }
        bundle = load_stage_edit_contextual_source_pair_production_bundle_v2(
            raw_directory,
            expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
            expected_checkpoint_file_sha256=file_sha256s["checkpoint"],
            expected_report_sha256=report.report_sha256,
            expected_report_file_sha256=file_sha256s["report"],
            expected_beam_qa_report_sha256=(
                report.beam_level_qa.qa_report_sha256
            ),
            expected_corpus_sha256=corpus.content_sha256,
            expected_corpus_file_sha256=file_sha256s["corpus"],
            expected_token_schema_fingerprint=(
                checkpoint.token_schema_fingerprint
            ),
            verified_beam_qa_cases=ordered_cases,
        )
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise StageEditProductionRegistryNotReadyError(
            f"contextual-v2 production bundle is invalid: {exc}"
        ) from exc
    manifest = _contextual_beam_qa_manifest_from_verified_cases(
        ordered_cases
    )
    return {
        "bundle_directory": str(bundle.directory),
        "checkpoint_sha256": bundle.checkpoint.checkpoint_sha256,
        "checkpoint_file_sha256": bundle.checkpoint_file_sha256,
        "report_sha256": bundle.report.report_sha256,
        "report_file_sha256": bundle.report_file_sha256,
        "beam_qa_report_sha256": (
            bundle.report.beam_level_qa.qa_report_sha256
        ),
        "corpus_sha256": bundle.corpus.content_sha256,
        "corpus_file_sha256": bundle.corpus_file_sha256,
        "token_schema_fingerprint": (
            bundle.checkpoint.token_schema_fingerprint
        ),
        "scorer_fingerprint": bundle.scorer.fingerprint,
        "verified_beam_qa_cases": ordered_cases,
        "beam_qa_manifest": manifest,
    }


def _registered_contextual_source_pair_identity(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Reissue QA envelopes and reverify every contextual-v2 artifact."""

    raw_identity = payload.get("contextual_source_pair_v2")
    raw_manifest = payload.get("contextual_beam_qa_cases")
    identity_keys = {
        "bundle_directory",
        "checkpoint",
        "production_report",
        "training_corpus",
        "token_schema_fingerprint",
        "scorer_fingerprint",
    }
    checkpoint_keys = {"filename", "content_sha256", "file_sha256"}
    report_keys = {
        "filename",
        "content_sha256",
        "file_sha256",
        "beam_qa_report_sha256",
    }
    corpus_keys = {"filename", "content_sha256", "file_sha256"}
    if (
        not isinstance(raw_identity, Mapping)
        or set(raw_identity) != identity_keys
        or not isinstance(raw_identity.get("checkpoint"), Mapping)
        or set(raw_identity["checkpoint"]) != checkpoint_keys
        or not isinstance(raw_identity.get("production_report"), Mapping)
        or set(raw_identity["production_report"]) != report_keys
        or not isinstance(raw_identity.get("training_corpus"), Mapping)
        or set(raw_identity["training_corpus"]) != corpus_keys
        or not isinstance(raw_manifest, Mapping)
        or set(raw_manifest) != {"contract", "cases"}
        or raw_manifest.get("contract")
        != STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_BEAM_QA_MANIFEST_CONTRACT
        or not isinstance(raw_manifest.get("cases"), list)
        or not raw_manifest["cases"]
    ):
        raise StageEditProductionRegistryError(
            "production registry contextual-v2 evidence is invalid"
        )
    directory = raw_identity.get("bundle_directory")
    if (
        not isinstance(directory, str)
        or not directory
        or not os.path.isabs(directory)
        or os.path.normpath(directory) != directory
    ):
        raise StageEditProductionRegistryError(
            "production registry contextual-v2 bundle path must be absolute and normalized"
        )
    checkpoint_record = raw_identity["checkpoint"]
    report_record = raw_identity["production_report"]
    corpus_record = raw_identity["training_corpus"]
    if (
        checkpoint_record.get("filename")
        != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FILENAME
        or report_record.get("filename")
        != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_FILENAME
        or corpus_record.get("filename")
        != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CORPUS_FILENAME
    ):
        raise StageEditProductionRegistryError(
            "production registry contextual-v2 artifact filenames are unsupported"
        )
    for value in (
        checkpoint_record.get("content_sha256"),
        checkpoint_record.get("file_sha256"),
        report_record.get("content_sha256"),
        report_record.get("file_sha256"),
        report_record.get("beam_qa_report_sha256"),
        corpus_record.get("content_sha256"),
        corpus_record.get("file_sha256"),
        raw_identity.get("token_schema_fingerprint"),
        raw_identity.get("scorer_fingerprint"),
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise StageEditProductionRegistryError(
                "production registry contextual-v2 SHA-256 identity is invalid"
            )
    from .stage_edit_contextual_beam_qa_adapter_v2 import (
        rehydrate_stage_edit_contextual_beam_qa_case_v2,
    )

    verified_cases: list[object] = []
    for entry in raw_manifest["cases"]:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"case", "derivation"}
            or not isinstance(entry.get("case"), Mapping)
            or not isinstance(entry.get("derivation"), Mapping)
        ):
            raise StageEditProductionRegistryError(
                "production registry contextual-v2 Beam-QA case manifest is invalid"
            )
        try:
            verified_cases.append(
                rehydrate_stage_edit_contextual_beam_qa_case_v2(
                    case_payload=entry["case"],
                    derivation_payload=entry["derivation"],
                )
            )
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise StageEditProductionRegistryNotReadyError(
                f"registered contextual-v2 Beam-QA replay failed: {exc}"
            ) from exc
    actual = _validated_contextual_source_pair_identity(
        Path(directory), tuple(verified_cases)
    )
    expected = {
        "bundle_directory": directory,
        "checkpoint_sha256": checkpoint_record.get("content_sha256"),
        "checkpoint_file_sha256": checkpoint_record.get("file_sha256"),
        "report_sha256": report_record.get("content_sha256"),
        "report_file_sha256": report_record.get("file_sha256"),
        "beam_qa_report_sha256": report_record.get("beam_qa_report_sha256"),
        "corpus_sha256": corpus_record.get("content_sha256"),
        "corpus_file_sha256": corpus_record.get("file_sha256"),
        "token_schema_fingerprint": raw_identity.get(
            "token_schema_fingerprint"
        ),
        "scorer_fingerprint": raw_identity.get("scorer_fingerprint"),
    }
    observed = {key: actual[key] for key in expected}
    if observed != expected or actual["beam_qa_manifest"] != dict(raw_manifest):
        raise StageEditProductionRegistryNotReadyError(
            "registered contextual-v2 artifact identity changed"
        )
    return actual


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

    dynamic_crop_version = int(payload.get("dynamic_crop_version", 2))
    continuity_authority = (
        str(payload["continuity_authority"])
        if dynamic_crop_version == 3
        else None
    )
    _checkpoint_namespace_version(
        checkpoint,
        expected_version=dynamic_crop_version,
    )
    montage_profile = _registered_montage_profile_identity(payload)
    pair_identity = (
        _registered_source_pair_identity(payload)
        if continuity_authority == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
        else None
    )
    contextual_identity = (
        _registered_contextual_source_pair_identity(payload)
        if continuity_authority
        == PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
        else None
    )

    metadata_filename = (
        DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
        if dynamic_crop_version == 3
        else DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME
    )
    weights_filename = (
        DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
        if dynamic_crop_version == 3
        else DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME
    )
    evidence_fields = (
        ("metadata", metadata_filename),
        ("weights", weights_filename),
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
        metadata = (
            StageEditDynamicCropCheckpointMetadataV3.from_dict(metadata_payload)
            if dynamic_crop_version == 3
            else StageEditDynamicCropCheckpointMetadataV2.from_dict(
                metadata_payload
            )
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        raise StageEditProductionRegistryNotReadyError(
            "registered Dynamic Crop "
            f"v{dynamic_crop_version} metadata is invalid: {exc}"
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
        or metadata.base_model_config.transition_authority
        != RENDERER_OWNED_TRANSITION_AUTHORITY
        or payload.get("transition_authority")
        != StageEditTransitionAuthority(
            mode=metadata.base_model_config.transition_authority
        ).to_dict()
        or evidence["weights"][1] != metadata.weight_sha256
    ):
        raise StageEditProductionRegistryNotReadyError(
            "registered checkpoint identity differs from Dynamic Crop "
            f"v{dynamic_crop_version} metadata"
        )

    motion_authority_identity_v3: (
        RegisteredStageEditMotionAuthorityIdentityV3 | None
    ) = None
    if dynamic_crop_version == 3:
        training_report = _strict_json_mapping(
            evidence["training_report"][0],
            "Dynamic Crop v3 production training report",
        )
        assert isinstance(metadata, StageEditDynamicCropCheckpointMetadataV3)
        _validate_v3_training_report_declarations(
            training_report,
            metadata=metadata,
            weight_sha256=evidence["weights"][1],
            checkpoint_directory=checkpoint,
        )
        if (
            payload.get("schema_version")
            == STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_AUTHORITY_V3_SCHEMA_VERSION
        ):
            observed_motion_identity = (
                _motion_authority_identity_from_verified_v3_training_report(
                    training_report
                )
            )
            try:
                registered_motion_identity = (
                    RegisteredStageEditMotionAuthorityIdentityV3.from_dict(
                        payload.get("dynamic_crop_v3_motion_authority")
                    )
                )
            except (TypeError, ValueError) as exc:
                raise StageEditProductionRegistryError(
                    "production registry v3 motion authority identity is invalid"
                ) from exc
            if (
                observed_motion_identity is None
                or registered_motion_identity != observed_motion_identity
            ):
                raise StageEditProductionRegistryNotReadyError(
                    "registered v3 motion authority no longer matches the "
                    "strictly replayed production-promotion release freeze"
                )
            motion_authority_identity_v3 = observed_motion_identity

    readiness = stage_edit_readiness_report(checkpoint)
    if not (
        readiness.architecture_available
        and readiness.learned_inference_ready
        and readiness.status == "production_checkpoint_ready"
        and readiness.checkpoint_path == str(checkpoint)
        and readiness.training_steps == metadata.training_steps
        and readiness.training_steps > 0
        and (
            dynamic_crop_version == 2
            or (
                readiness.dynamic_crop_version == 3
                and readiness.checkpoint_format == "dynamic_crop_v3"
                and readiness.checkpoint_metadata_filename
                == DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
            )
        )
    ):
        raise StageEditProductionRegistryNotReadyError(
            f"registered Dynamic Crop v{dynamic_crop_version} checkpoint is not "
            "production-ready: "
            f"{readiness.status}: {readiness.reason}"
        )

    return RegisteredStageEditProductionBundle(
        registry_path=path,
        checkpoint_directory=checkpoint,
        checkpoint_id=checkpoint_id,
        dynamic_crop_version=dynamic_crop_version,
        continuity_authority=continuity_authority,
        token_schema_fingerprint=token_schema,
        maximum_sequence_length=int(maximum_sequence_length),
        maximum_sources=int(maximum_sources),
        transition_authority=dict(payload["transition_authority"]),
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
        source_pair_scorer_path=(
            Path(pair_identity["scorer_path"])
            if pair_identity is not None
            else None
        ),
        source_pair_production_report_path=(
            Path(pair_identity["production_report_path"])
            if pair_identity is not None
            else None
        ),
        source_pair_scorer_file_sha256=(
            pair_identity["scorer_file_sha256"]
            if pair_identity is not None
            else None
        ),
        source_pair_production_report_file_sha256=(
            pair_identity["production_report_file_sha256"]
            if pair_identity is not None
            else None
        ),
        source_pair_production_report_sha256=(
            pair_identity["production_report_sha256"]
            if pair_identity is not None
            else None
        ),
        source_pair_scorer_fingerprint=(
            pair_identity["scorer_fingerprint"]
            if pair_identity is not None
            else None
        ),
        source_pair_checkpoint_sha256=(
            pair_identity["checkpoint_sha256"]
            if pair_identity is not None
            else None
        ),
        source_pair_corpus_sha256=(
            pair_identity["corpus_sha256"]
            if pair_identity is not None
            else None
        ),
        contextual_source_pair_bundle_directory=(
            Path(contextual_identity["bundle_directory"])
            if contextual_identity is not None
            else None
        ),
        contextual_source_pair_checkpoint_sha256=(
            contextual_identity["checkpoint_sha256"]
            if contextual_identity is not None
            else None
        ),
        contextual_source_pair_checkpoint_file_sha256=(
            contextual_identity["checkpoint_file_sha256"]
            if contextual_identity is not None
            else None
        ),
        contextual_source_pair_report_sha256=(
            contextual_identity["report_sha256"]
            if contextual_identity is not None
            else None
        ),
        contextual_source_pair_report_file_sha256=(
            contextual_identity["report_file_sha256"]
            if contextual_identity is not None
            else None
        ),
        contextual_source_pair_beam_qa_report_sha256=(
            contextual_identity["beam_qa_report_sha256"]
            if contextual_identity is not None
            else None
        ),
        contextual_source_pair_corpus_sha256=(
            contextual_identity["corpus_sha256"]
            if contextual_identity is not None
            else None
        ),
        contextual_source_pair_corpus_file_sha256=(
            contextual_identity["corpus_file_sha256"]
            if contextual_identity is not None
            else None
        ),
        contextual_source_pair_token_schema_fingerprint=(
            contextual_identity["token_schema_fingerprint"]
            if contextual_identity is not None
            else None
        ),
        contextual_source_pair_scorer_fingerprint=(
            contextual_identity["scorer_fingerprint"]
            if contextual_identity is not None
            else None
        ),
        contextual_source_pair_verified_beam_qa_cases=(
            tuple(contextual_identity["verified_beam_qa_cases"])
            if contextual_identity is not None
            else ()
        ),
        motion_authority_identity_v3=motion_authority_identity_v3,
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
    dynamic_crop_version: int = 2,
    continuity_authority: str | None = None,
    source_pair_scorer_path: Path | None = None,
    source_pair_production_report_path: Path | None = None,
    contextual_source_pair_bundle_directory: Path | None = None,
    contextual_source_pair_verified_beam_qa_cases: Sequence[object] | None = None,
) -> Path:
    """Install one verified bundle without replacing another active record."""

    if (
        isinstance(dynamic_crop_version, bool)
        or not isinstance(dynamic_crop_version, int)
        or dynamic_crop_version not in (2, 3)
    ):
        raise StageEditProductionRegistryNotReadyError(
            "dynamic_crop_version must be exactly 2 or 3"
        )
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
    _checkpoint_namespace_version(
        checkpoint,
        expected_version=dynamic_crop_version,
    )
    if dynamic_crop_version == 2:
        if continuity_authority is not None:
            raise StageEditProductionRegistryNotReadyError(
                "Dynamic Crop v2 cannot register a continuity authority"
            )
        if (
            source_pair_scorer_path is not None
            or source_pair_production_report_path is not None
            or contextual_source_pair_bundle_directory is not None
            or contextual_source_pair_verified_beam_qa_cases is not None
        ):
            raise StageEditProductionRegistryNotReadyError(
                "Dynamic Crop v2 cannot register source-pair v3 artifacts"
            )
        pair_identity: dict[str, Any] | None = None
        contextual_identity: dict[str, Any] | None = None
    elif (
        not isinstance(continuity_authority, str)
        or continuity_authority not in STAGE_EDIT_CONTINUITY_AUTHORITIES
    ):
        raise StageEditProductionRegistryNotReadyError(
            "Dynamic Crop v3 requires an explicit supported continuity authority"
        )
    elif continuity_authority == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY:
        if (
            contextual_source_pair_bundle_directory is not None
            or contextual_source_pair_verified_beam_qa_cases is not None
        ):
            raise StageEditProductionRegistryNotReadyError(
                "learned-v1 Dynamic Crop v3 continuity cannot register "
                "contextual-v2 artifacts"
            )
        if (
            source_pair_scorer_path is None
            or source_pair_production_report_path is None
        ):
            raise StageEditProductionRegistryNotReadyError(
                "Dynamic Crop v3 requires both source-pair scorer and "
                "production report paths"
            )
        pair_identity = _validated_source_pair_identity(
            Path(source_pair_scorer_path),
            Path(source_pair_production_report_path),
        )
        contextual_identity: dict[str, Any] | None = None
    elif continuity_authority == PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2:
        if (
            source_pair_scorer_path is not None
            or source_pair_production_report_path is not None
        ):
            raise StageEditProductionRegistryNotReadyError(
                "contextual-v2 Dynamic Crop v3 continuity cannot register "
                "learned-v1 artifacts"
            )
        if (
            contextual_source_pair_bundle_directory is None
            or contextual_source_pair_verified_beam_qa_cases is None
        ):
            raise StageEditProductionRegistryNotReadyError(
                "contextual-v2 Dynamic Crop v3 continuity requires its bundle "
                "directory and adapter-issued Beam-QA envelopes"
            )
        pair_identity = None
        contextual_identity = _validated_contextual_source_pair_identity(
            Path(contextual_source_pair_bundle_directory),
            contextual_source_pair_verified_beam_qa_cases,
        )
    else:
        if (
            source_pair_scorer_path is not None
            or source_pair_production_report_path is not None
            or contextual_source_pair_bundle_directory is not None
            or contextual_source_pair_verified_beam_qa_cases is not None
        ):
            raise StageEditProductionRegistryNotReadyError(
                "handcrafted Dynamic Crop v3 continuity cannot register "
                "source-pair artifacts"
            )
        pair_identity = None
        contextual_identity = None
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
    metadata_path = checkpoint / (
        DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
        if dynamic_crop_version == 3
        else DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME
    )
    weights_path = checkpoint / (
        DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
        if dynamic_crop_version == 3
        else DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME
    )
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
        metadata = (
            StageEditDynamicCropCheckpointMetadataV3.from_dict(raw_metadata)
            if dynamic_crop_version == 3
            else StageEditDynamicCropCheckpointMetadataV2.from_dict(
                raw_metadata
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StageEditProductionRegistryNotReadyError(
            f"Dynamic Crop v{dynamic_crop_version} metadata is invalid: {exc}"
        ) from exc
    if (
        metadata.base_model_config.transition_authority
        != RENDERER_OWNED_TRANSITION_AUTHORITY
    ):
        raise StageEditProductionRegistryNotReadyError(
            "IVE production registry requires renderer-owned transition authority"
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
    motion_authority_identity_v3: (
        RegisteredStageEditMotionAuthorityIdentityV3 | None
    ) = None
    if dynamic_crop_version == 3:
        assert isinstance(metadata, StageEditDynamicCropCheckpointMetadataV3)
        training_report = _strict_json_mapping(
            report_path,
            "Dynamic Crop v3 production training report",
        )
        _validate_v3_training_report_declarations(
            training_report,
            metadata=metadata,
            weight_sha256=weights_sha,
            checkpoint_directory=checkpoint,
        )
        motion_authority_identity_v3 = (
            _motion_authority_identity_from_verified_v3_training_report(
                training_report
            )
        )
        if not (
            readiness.dynamic_crop_version == 3
            and readiness.checkpoint_format == "dynamic_crop_v3"
            and readiness.checkpoint_metadata_filename
            == DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
        ):
            raise StageEditProductionRegistryNotReadyError(
                "Dynamic Crop v3 readiness does not identify the v3 namespace"
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
            and installed.dynamic_crop_version == dynamic_crop_version
            and installed.continuity_authority == continuity_authority
            and installed.metadata_sha256 == metadata_sha
            and installed.weights_sha256 == weights_sha
            and installed.training_report_sha256 == report_sha
            and installed.motion_authority_identity_v3
            == motion_authority_identity_v3
            and installed.professional_montage_grammar_profile_identity
            == montage_profile
            and (
                pair_identity is None
                or (
                    installed.source_pair_scorer_path
                    == Path(pair_identity["scorer_path"])
                    and installed.source_pair_production_report_path
                    == Path(pair_identity["production_report_path"])
                    and installed.source_pair_scorer_file_sha256
                    == pair_identity["scorer_file_sha256"]
                    and installed.source_pair_production_report_file_sha256
                    == pair_identity["production_report_file_sha256"]
                    and installed.source_pair_production_report_sha256
                    == pair_identity["production_report_sha256"]
                    and installed.source_pair_scorer_fingerprint
                    == pair_identity["scorer_fingerprint"]
                    and installed.source_pair_checkpoint_sha256
                    == pair_identity["checkpoint_sha256"]
                    and installed.source_pair_corpus_sha256
                        == pair_identity["corpus_sha256"]
                )
            )
            and (
                contextual_identity is None
                or (
                    installed.contextual_source_pair_bundle_directory
                    == Path(contextual_identity["bundle_directory"])
                    and installed.contextual_source_pair_checkpoint_sha256
                    == contextual_identity["checkpoint_sha256"]
                    and installed.contextual_source_pair_checkpoint_file_sha256
                    == contextual_identity["checkpoint_file_sha256"]
                    and installed.contextual_source_pair_report_sha256
                    == contextual_identity["report_sha256"]
                    and installed.contextual_source_pair_report_file_sha256
                    == contextual_identity["report_file_sha256"]
                    and installed.contextual_source_pair_beam_qa_report_sha256
                    == contextual_identity["beam_qa_report_sha256"]
                    and installed.contextual_source_pair_corpus_sha256
                    == contextual_identity["corpus_sha256"]
                    and installed.contextual_source_pair_corpus_file_sha256
                    == contextual_identity["corpus_file_sha256"]
                    and installed.contextual_source_pair_token_schema_fingerprint
                    == contextual_identity["token_schema_fingerprint"]
                    and installed.contextual_source_pair_scorer_fingerprint
                    == contextual_identity["scorer_fingerprint"]
                    and [
                        {
                            "case": value.case.to_dict(),
                            "derivation": value.derivation.to_dict(),
                        }
                        for value in installed.contextual_source_pair_verified_beam_qa_cases
                    ]
                    == [
                        {
                            "case": value.case.to_dict(),
                            "derivation": value.derivation.to_dict(),
                        }
                        for value in contextual_identity[
                            "verified_beam_qa_cases"
                        ]
                    ]
                )
            )
        ):
            return destination
        raise StageEditProductionRegistryError(
            "a different active production registry already exists"
        )

    unsigned: dict[str, Any] = {
        "schema_version": (
            STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_AUTHORITY_V3_SCHEMA_VERSION
            if motion_authority_identity_v3 is not None
            else STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_V2_SCHEMA_VERSION
            if continuity_authority
            == PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
            else STAGE_EDIT_PRODUCTION_REGISTRY_V3_SCHEMA_VERSION
            if dynamic_crop_version == 3
            else STAGE_EDIT_PRODUCTION_REGISTRY_SCHEMA_VERSION
        ),
        "registry_kind": (
            STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_AUTHORITY_V3_KIND
            if motion_authority_identity_v3 is not None
            else STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_V2_KIND
            if continuity_authority
            == PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
            else STAGE_EDIT_PRODUCTION_REGISTRY_V3_KIND
            if dynamic_crop_version == 3
            else STAGE_EDIT_PRODUCTION_REGISTRY_KIND
        ),
        "active": True,
        "registered_at": datetime.now(UTC).isoformat(),
        "checkpoint_directory": str(checkpoint),
        "checkpoint_id": metadata.checkpoint_id,
        "model_contract": metadata.model_contract,
        "token_schema_fingerprint": metadata.token_schema.fingerprint,
        "maximum_sequence_length": metadata.base_model_config.max_sequence_length,
        "maximum_sources": metadata.base_model_config.max_sources,
        "transition_authority": StageEditTransitionAuthority(
            mode=metadata.base_model_config.transition_authority
        ).to_dict(),
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
    if dynamic_crop_version == 3:
        unsigned["dynamic_crop_version"] = 3
        unsigned["continuity_authority"] = continuity_authority
    if motion_authority_identity_v3 is not None:
        unsigned["dynamic_crop_v3_motion_authority"] = (
            motion_authority_identity_v3.to_dict()
        )
        unsigned["storage_policy"][
            "dynamic_crop_v3_release_freeze_absolute_path_required"
        ] = True
    if continuity_authority == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY:
        assert pair_identity is not None
        unsigned["source_pair_scorer"] = {
            "path": pair_identity["scorer_path"],
            "sha256": pair_identity["scorer_file_sha256"],
            "scorer_fingerprint": pair_identity["scorer_fingerprint"],
            "checkpoint_sha256": pair_identity["checkpoint_sha256"],
            "corpus_sha256": pair_identity["corpus_sha256"],
        }
        unsigned["source_pair_production_report"] = {
            "path": pair_identity["production_report_path"],
            "sha256": pair_identity["production_report_file_sha256"],
            "report_sha256": pair_identity["production_report_sha256"],
            "cross_validation_sha256": pair_identity[
                "cross_validation_sha256"
            ],
        }
        unsigned["storage_policy"][
            "source_pair_absolute_paths_required"
        ] = True
    elif continuity_authority == PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2:
        assert contextual_identity is not None
        unsigned["contextual_source_pair_v2"] = {
            "bundle_directory": contextual_identity["bundle_directory"],
            "checkpoint": {
                "filename": (
                    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FILENAME
                ),
                "content_sha256": contextual_identity["checkpoint_sha256"],
                "file_sha256": contextual_identity[
                    "checkpoint_file_sha256"
                ],
            },
            "production_report": {
                "filename": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_FILENAME,
                "content_sha256": contextual_identity["report_sha256"],
                "file_sha256": contextual_identity["report_file_sha256"],
                "beam_qa_report_sha256": contextual_identity[
                    "beam_qa_report_sha256"
                ],
            },
            "training_corpus": {
                "filename": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CORPUS_FILENAME,
                "content_sha256": contextual_identity["corpus_sha256"],
                "file_sha256": contextual_identity["corpus_file_sha256"],
            },
            "token_schema_fingerprint": contextual_identity[
                "token_schema_fingerprint"
            ],
            "scorer_fingerprint": contextual_identity[
                "scorer_fingerprint"
            ],
        }
        unsigned["contextual_beam_qa_cases"] = contextual_identity[
            "beam_qa_manifest"
        ]
        unsigned["storage_policy"][
            "contextual_source_pair_bundle_absolute_path_required"
        ] = True
        unsigned["storage_policy"][
            "contextual_beam_qa_artifact_absolute_paths_required"
        ] = True
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
    installed_new_record = False
    installed_record_identity: tuple[int, int, int] | None = None
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        staged_stat = temporary.lstat()
        staged_identity = (
            staged_stat.st_dev,
            staged_stat.st_ino,
            staged_stat.st_size,
        )
        temporary.replace(destination)
        installed_new_record = True
        installed_stat = destination.lstat()
        installed_record_identity = (
            installed_stat.st_dev,
            installed_stat.st_ino,
            installed_stat.st_size,
        )
        if installed_record_identity != staged_identity:
            raise StageEditProductionRegistryError(
                "new production registry identity changed during installation"
            )
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        # Re-read every evidence path through the public loader before
        # returning.  If replay fails, remove only the exact bytes installed
        # by this call so a caller can retry safely; never touch a pre-existing
        # registry.
        load_registered_stage_edit_production_bundle(destination)
    except BaseException:
        if (
            installed_new_record
            and installed_record_identity is not None
        ):
            try:
                observed_before = destination.lstat()
                observed_identity = (
                    observed_before.st_dev,
                    observed_before.st_ino,
                    observed_before.st_size,
                )
                observed_bytes = destination.read_bytes()
                observed_after = destination.lstat()
                if (
                    stat_module.S_ISREG(observed_before.st_mode)
                    and observed_identity == installed_record_identity
                    and observed_identity
                    == (
                        observed_after.st_dev,
                        observed_after.st_ino,
                        observed_after.st_size,
                    )
                    and observed_bytes == encoded
                ):
                    destination.unlink()
            except OSError:
                pass
        raise
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return destination


__all__ = [
    "RegisteredStageEditMotionAuthorityIdentityV3",
    "RegisteredStageEditProductionBundle",
    "STAGE_EDIT_PRODUCTION_REGISTRY_ENV",
    "STAGE_EDIT_PRODUCTION_REGISTRY_FILENAME",
    "STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_BEAM_QA_MANIFEST_CONTRACT",
    "STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_V2_KIND",
    "STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_V2_SCHEMA_VERSION",
    "STAGE_EDIT_PRODUCTION_REGISTRY_KIND",
    "STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_AUTHORITY_V3_KIND",
    "STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_AUTHORITY_V3_SCHEMA_VERSION",
    "STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_LOGIT_THRESHOLD_V3",
    "STAGE_EDIT_PRODUCTION_REGISTRY_MOTION_THRESHOLD_FREEZE_CONTRACT_V3",
    "STAGE_EDIT_PRODUCTION_REGISTRY_SCHEMA_VERSION",
    "STAGE_EDIT_PRODUCTION_REGISTRY_V3_KIND",
    "STAGE_EDIT_PRODUCTION_REGISTRY_V3_SCHEMA_VERSION",
    "StageEditProductionRegistryError",
    "StageEditProductionRegistryNotReadyError",
    "default_stage_edit_production_registry_paths",
    "discover_registered_stage_edit_production_bundle",
    "load_registered_stage_edit_production_bundle",
    "preferred_stage_edit_production_registry_path",
    "register_stage_edit_production_bundle",
]
