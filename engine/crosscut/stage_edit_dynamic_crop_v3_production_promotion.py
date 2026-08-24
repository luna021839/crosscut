"""Promote one immutable Dynamic Crop v3 all-reference final fit.

The three LOSO checkpoints prove that the frozen architecture, objective and
threshold policy generalize, but none of those fold weights is the all-reference
final-fit weight.  This module keeps those identities separate.  It copies the
inactive final-fit checkpoint byte-for-byte, evaluates that exact copied weight
on two predeclared non-blind IVE evidence songs, and grants registry authority
only when every frozen evaluator, renderer and structured-head gate passes.

No optimizer is constructed here.  XOXZ is explicitly recorded as previously
observed development evidence, ATTITUDE is explicitly recorded as predeclared
non-blind evidence, and neither blind-holdout nor human-parity claims are made.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat as stat_module
from typing import Any, Mapping, Sequence

from .professional_montage_policy import (
    ProfessionalMontagePolicyError,
    load_professional_montage_policy,
)
from .stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
)
from .stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
    DYNAMIC_CROP_V3_LOSS_CONTRACT,
    DYNAMIC_CROP_V3_MODEL_CONTRACT,
    DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
    DYNAMIC_CROP_V3_SCHEMA_VERSION,
    DYNAMIC_CROP_V3_TARGET_CONTRACT,
    DynamicCropV3LossContract,
    StageEditDynamicCropCheckpointMetadataV3,
    evaluate_stage_edit_dynamic_crop_v3,
    load_stage_edit_dynamic_crop_checkpoint_v3,
    tensorize_stage_edit_dynamic_crop_supervision_v3,
)
from .stage_edit_dynamic_crop_training import (
    build_stage_edit_dynamic_crop_training_corpus_v2,
)
from .stage_edit_dynamic_crop_training_v3 import (
    build_stage_edit_dynamic_crop_training_corpus_v3,
)
from . import stage_edit_dynamic_crop_v3_final_release_launcher as final_release
from .stage_edit_dynamic_crop_v3_capacity_fold_launcher import (
    load_dynamic_crop_v3_capacity_fold_evidence,
)
from . import stage_edit_dynamic_crop_capacity_fold_launcher as provenance
from . import stage_edit_production_fold_launcher as manifests
from .stage_edit_production_registry import (
    load_registered_stage_edit_production_bundle,
    preferred_stage_edit_production_registry_path,
    register_stage_edit_production_bundle,
)
from .stage_edit_production_train_cli import (
    _dynamic_artifact_map,
    _snapshot_map,
)
from .stage_edit_production_training import (
    ProductionQualityThresholds,
    STRUCTURED_HEAD_COVERAGE_CONTRACT,
    _structured_head_coverage_gate,
    evaluate_stage_edit_production_quality,
    prepare_stage_edit_production_input_sequences,
)
from . import stage_edit_production_training as production
from .stage_edit_reference_adapter import (
    ReferenceTrainingCorpus,
    load_reference_training_corpus,
)
from .stage_edit_training import evaluate_stage_edit_model
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    PRODUCTION_TRAINING_REPORT_FILENAME,
    STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT,
    StageEditLossContract,
    stage_edit_readiness_report,
    tensorize_stage_edit_batch,
    tensorize_stage_edit_supervision,
)
from .stage_edit_transition_authority import transition_authority_contract


PROMOTION_PLAN_SCHEMA_VERSION = 1
PROMOTION_PLAN_ANALYSIS_BASIS = (
    "ive_dynamic_crop_v3_final_fit_exact_weight_production_promotion_plan_v1"
)
PROMOTION_PLAN_CONTRACT = (
    "three_loso_plus_all_reference_final_fit_exact_weight_two_song_evaluation_v1"
)
PROMOTION_PLAN_PREFIX = "dynamic-crop-v3-production-promotion-plan-"

PRODUCTION_PROMOTION_REPORT_SCHEMA_VERSION = 1
PRODUCTION_PROMOTION_RUN_KIND = (
    "dynamic_crop_v3_final_fit_exact_weight_production_promotion"
)
PRODUCTION_PROMOTION_REPORT_CONTRACT = (
    "copied_final_fit_weight_frozen_two_song_nonblind_quality_gate_v1"
)
PRODUCTION_PROMOTION_CHECKPOINT_SCOPE = (
    "all_15_reference_final_fit_exact_weight_promoted_after_frozen_external_evaluation"
)

PROMOTION_RECEIPT_SCHEMA_VERSION = 1
PROMOTION_RECEIPT_ANALYSIS_BASIS = (
    "ive_dynamic_crop_v3_registered_exact_final_fit_weight_receipt_v1"
)
PROMOTION_RECEIPT_CONTRACT = (
    "checkpoint_report_registry_and_lineage_immutable_binding_v1"
)
PROMOTION_RECEIPT_PREFIX = "dynamic-crop-v3-production-promotion-receipt-"

XOXZ = "IVE-XOXZ"
ATTITUDE = "IVE-ATTITUDE"
EVALUATION_SONG_IDS = (XOXZ, ATTITUDE)
EVIDENCE_STATUS = {
    XOXZ: "previously_observed_development_evidence_non_blind",
    ATTITUDE: "predeclared_release_evidence_non_blind",
}
PRODUCTION_VOLUME_ROOT = Path("/Volumes/CrossCutMedia")
MINIMUM_FREE_BYTES = 8 * 1024**3

SOURCE_RELATIVE_PATHS = (
    "engine/crosscut/stage_edit_dynamic_crop_v3_production_promotion.py",
    "engine/crosscut/stage_edit_dynamic_crop_v3_final_fit.py",
    "engine/crosscut/stage_edit_dynamic_crop_v3_final_release_launcher.py",
    "engine/crosscut/stage_edit_dynamic_crop_v3_capacity_fold_launcher.py",
    "engine/crosscut/stage_edit_production_training.py",
    "engine/crosscut/stage_edit_transformer.py",
    "engine/crosscut/stage_edit_production_registry.py",
    "engine/crosscut/stage_edit_dynamic_crop_head_v3.py",
    "engine/crosscut/stage_edit_dynamic_crop_training.py",
    "engine/crosscut/stage_edit_dynamic_crop_training_v3.py",
    "engine/crosscut/stage_edit_reference_adapter.py",
    "engine/crosscut/stage_edit_production_fold_launcher.py",
    "engine/crosscut/professional_montage_policy.py",
)


class DynamicCropV3ProductionPromotionError(ValueError):
    """Promotion lineage, evidence, checkpoint or report is invalid."""


@dataclass(frozen=True, slots=True)
class PromotionSongEvidence:
    song_id: str
    evidence_status: str
    materialization_manifest_path: Path
    feature_manifest_path: Path
    dynamic_crop_manifest_path: Path
    reference_ids: tuple[str, ...]
    supervision_paths: Mapping[str, Path]
    snapshot_paths: Mapping[str, Path]


@dataclass(frozen=True, slots=True)
class DynamicCropV3ProductionPromotionPreflight:
    plan_path: Path
    plan_content_sha256: str
    source_checkpoint_directory: Path
    destination_checkpoint_directory: Path
    evidence_directory: Path
    production_registry_path: Path
    professional_montage_profile_path: Path
    checkpoint_id: str
    checkpoint_weight_sha256: str
    frozen_cut_probability_threshold: float
    quality_thresholds: Mapping[str, Any]
    song_evidence: Mapping[str, PromotionSongEvidence]
    free_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_path": str(self.plan_path),
            "plan_content_sha256": self.plan_content_sha256,
            "source_checkpoint_directory": str(
                self.source_checkpoint_directory
            ),
            "destination_checkpoint_directory": str(
                self.destination_checkpoint_directory
            ),
            "evidence_directory": str(self.evidence_directory),
            "production_registry_path": str(self.production_registry_path),
            "professional_montage_profile_path": str(
                self.professional_montage_profile_path
            ),
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_weight_sha256": self.checkpoint_weight_sha256,
            "frozen_cut_probability_threshold": (
                self.frozen_cut_probability_threshold
            ),
            "quality_thresholds": dict(self.quality_thresholds),
            "evaluation_songs": {
                song_id: {
                    "evidence_status": evidence.evidence_status,
                    "reference_ids": list(evidence.reference_ids),
                }
                for song_id, evidence in sorted(self.song_evidence.items())
            },
            "free_bytes": self.free_bytes,
            "optimizer_or_gradient_step_authorized": False,
            "model_threshold_or_policy_reselection_authorized": False,
            "blind_holdout_claim_authorized": False,
            "human_parity_claim_authorized": False,
            "continuity_authority": HANDCRAFTED_CONTINUITY_AUTHORITY,
        }


@dataclass(frozen=True, slots=True)
class DynamicCropV3ProductionPromotionResult:
    checkpoint_directory: Path
    report_path: Path
    registry_path: Path
    receipt_path: Path
    checkpoint_id: str
    checkpoint_weight_sha256: str
    report: Mapping[str, Any]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _is_commit_oid(value: object) -> bool:
    if not isinstance(value, str) or len(value) not in (40, 64):
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicCropV3ProductionPromotionError(
            f"{label} must be an object"
        )
    return dict(value)


def _file_sha256(path: Path, label: str) -> str:
    try:
        return provenance._stable_file_sha256(
            Path(path).expanduser().resolve(), label
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3ProductionPromotionError(str(exc)) from exc


def _pointer(path: Path, label: str) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(resolved),
        "file_sha256": _file_sha256(resolved, label),
    }


def _verify_pointer(value: object, label: str) -> Path:
    raw = _mapping(value, label)
    if set(raw) != {"path", "file_sha256"}:
        raise DynamicCropV3ProductionPromotionError(
            f"{label} pointer keys are not exact"
        )
    raw_path = raw.get("path")
    expected = raw.get("file_sha256")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise DynamicCropV3ProductionPromotionError(
            f"{label} pointer path must be absolute"
        )
    if not _is_sha256(expected):
        raise DynamicCropV3ProductionPromotionError(
            f"{label} pointer hash is invalid"
        )
    path = Path(raw_path).resolve()
    if _file_sha256(path, label) != expected:
        raise DynamicCropV3ProductionPromotionError(
            f"{label} pointer hash mismatch"
        )
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise DynamicCropV3ProductionPromotionError(
            f"{label} must be a regular non-symlink file"
        )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DynamicCropV3ProductionPromotionError(
            f"{label} cannot be read: {exc}"
        ) from exc
    return _mapping(value, label)


def _under_production_volume(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    volume = PRODUCTION_VOLUME_ROOT.resolve()
    if resolved != volume and volume not in resolved.parents:
        raise DynamicCropV3ProductionPromotionError(
            f"{label} must be under {volume}"
        )
    for ancestor in (resolved, *resolved.parents):
        if ancestor == volume.parent:
            break
        if ancestor.is_symlink():
            raise DynamicCropV3ProductionPromotionError(
                f"{label} must not traverse a symlink: {ancestor}"
            )
    return resolved


def _outside_repository(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    if resolved == repository or repository in resolved.parents:
        raise DynamicCropV3ProductionPromotionError(
            f"{label} must stay outside the source repository"
        )
    return resolved


def _capture_source_provenance(
    repository_root: Path | None = None,
) -> dict[str, Any]:
    try:
        return provenance._capture_source_provenance(
            repository_root,
            source_relative_paths=SOURCE_RELATIVE_PATHS,
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3ProductionPromotionError(str(exc)) from exc


def _validate_source_provenance(value: object) -> dict[str, Any]:
    payload = _mapping(value, "promotion source provenance")
    if set(payload) != {
        "contract",
        "repository_root",
        "head_commit_oid",
        "repository_status_clean",
        "repository_status_porcelain_sha256",
        "source_files",
    }:
        raise DynamicCropV3ProductionPromotionError(
            "promotion source provenance keys are not exact"
        )
    files = _mapping(payload.get("source_files"), "promotion source files")
    if set(files) != set(SOURCE_RELATIVE_PATHS):
        raise DynamicCropV3ProductionPromotionError(
            "promotion source provenance file set differs"
        )
    root = Path(str(payload.get("repository_root") or ""))
    observed = _capture_source_provenance(root)
    if observed != payload:
        raise DynamicCropV3ProductionPromotionError(
            "current clean promotion source differs from the plan"
        )
    return payload


def _write_content_addressed(
    *,
    output_directory: Path,
    prefix: str,
    unsigned: Mapping[str, Any],
    hash_field: str,
) -> Path:
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    digest = _canonical_sha256(unsigned)
    payload = {**dict(unsigned), hash_field: digest}
    destination = output / f"{prefix}{digest}.json"
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        with destination.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        existing = _read_json(destination, "existing content-addressed artifact")
        if existing != payload:
            raise DynamicCropV3ProductionPromotionError(
                f"refusing to overwrite different artifact: {destination}"
            ) from exc
    directory_fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination


def _replay_lineage(final_fit_receipt_path: Path) -> dict[str, Any]:
    """Fully replay final fit, release freeze and each of the three LOSO runs."""

    receipt_path = Path(final_fit_receipt_path).expanduser().resolve()
    try:
        receipt = final_release.load_dynamic_crop_v3_final_fit_receipt(
            receipt_path
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DynamicCropV3ProductionPromotionError(
            f"final-fit receipt replay failed: {exc}"
        ) from exc
    final_fit_plan_path = _verify_pointer(receipt.get("plan"), "final-fit plan")
    final_fit_plan = final_release._strict_final_fit_plan(final_fit_plan_path)
    freeze_path = _verify_pointer(
        receipt.get("release_freeze"), "release freeze"
    )
    freeze = final_release._strict_release_freeze(freeze_path)
    folds = freeze.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise DynamicCropV3ProductionPromotionError(
            "release freeze must contain exactly three LOSO folds"
        )
    fold_receipts: list[tuple[Path, dict[str, Any]]] = []
    for raw in folds:
        record = _mapping(raw, "release-freeze LOSO fold")
        path = _verify_pointer(record.get("receipt"), "LOSO fold receipt")
        try:
            payload = load_dynamic_crop_v3_capacity_fold_evidence(path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise DynamicCropV3ProductionPromotionError(
                f"LOSO receipt replay failed: {exc}"
            ) from exc
        fold_receipts.append((path, payload))
    heldout = {payload.get("validation_song_id") for _, payload in fold_receipts}
    if heldout != set(final_release.SELECTION_SONG_IDS):
        raise DynamicCropV3ProductionPromotionError(
            "replayed LOSO receipts do not cover all three frozen songs"
        )
    metadata_path = _verify_pointer(
        receipt.get("checkpoint_metadata"), "final-fit metadata"
    )
    weights_path = _verify_pointer(
        receipt.get("checkpoint_weights"), "final-fit weights"
    )
    report_path = _verify_pointer(
        receipt.get("final_fit_report"), "final-fit report"
    )
    metadata = StageEditDynamicCropCheckpointMetadataV3.from_dict(
        _read_json(metadata_path, "final-fit metadata")
    )
    if (
        metadata.weight_sha256 != _file_sha256(weights_path, "final-fit weights")
        or receipt.get("checkpoint_id") != metadata.checkpoint_id
        or receipt.get("checkpoint_weight_sha256") != metadata.weight_sha256
        or metadata.production_activation_authorized is not False
    ):
        raise DynamicCropV3ProductionPromotionError(
            "final-fit receipt checkpoint identity differs"
        )
    source = _mapping(
        final_fit_plan.get("source_provenance"), "final-fit source provenance"
    )
    return {
        "receipt_path": receipt_path,
        "receipt": receipt,
        "final_fit_plan_path": final_fit_plan_path,
        "final_fit_plan": final_fit_plan,
        "release_freeze_path": freeze_path,
        "release_freeze": freeze,
        "fold_receipts": fold_receipts,
        "metadata_path": metadata_path,
        "weights_path": weights_path,
        "final_fit_report_path": report_path,
        "final_fit_report": final_release.load_dynamic_crop_v3_final_fit_report(
            report_path
        ),
        "metadata": metadata,
        "source_commit": source.get("head_commit_oid"),
    }


def _lineage_plan_record(lineage: Mapping[str, Any]) -> dict[str, Any]:
    metadata = lineage["metadata"]
    assert isinstance(metadata, StageEditDynamicCropCheckpointMetadataV3)
    folds = sorted(
        lineage["fold_receipts"],
        key=lambda item: str(item[1]["validation_song_id"]),
    )
    return {
        "final_fit_receipt": _pointer(
            lineage["receipt_path"], "final-fit receipt"
        ),
        "final_fit_receipt_content_sha256": lineage["receipt"].get(
            "artifact_content_sha256"
        ),
        "final_fit_plan": _pointer(
            lineage["final_fit_plan_path"], "final-fit plan"
        ),
        "release_freeze": _pointer(
            lineage["release_freeze_path"], "release freeze"
        ),
        "release_freeze_content_sha256": lineage["release_freeze"].get(
            "freeze_content_sha256"
        ),
        "loso_fold_receipts": [
            {
                "validation_song_id": payload["validation_song_id"],
                "checkpoint_id": payload["checkpoint_id"],
                "checkpoint_weight_sha256": payload[
                    "checkpoint_weight_sha256"
                ],
                "receipt": _pointer(path, "LOSO fold receipt"),
                "receipt_content_sha256": payload["artifact_content_sha256"],
            }
            for path, payload in folds
        ],
        "source_checkpoint": {
            "directory": str(lineage["metadata_path"].parent),
            "metadata": _pointer(lineage["metadata_path"], "final-fit metadata"),
            "weights": _pointer(lineage["weights_path"], "final-fit weights"),
            "final_fit_report": _pointer(
                lineage["final_fit_report_path"], "final-fit report"
            ),
            "checkpoint_id": metadata.checkpoint_id,
            "training_steps": metadata.training_steps,
            "weight_sha256": metadata.weight_sha256,
            "production_activation_authorized_in_metadata": False,
        },
    }


def _manifest_pointer(path: Path, label: str) -> dict[str, str]:
    return _pointer(Path(path).expanduser().resolve(), label)


def _validate_song_evidence_record(
    value: object,
    *,
    expected_song_id: str,
) -> PromotionSongEvidence:
    record = _mapping(value, f"{expected_song_id} evidence")
    if set(record) != {
        "song_id",
        "evidence_status",
        "blind_holdout_claim_authorized",
        "human_parity_claim_authorized",
        "materialization_manifest",
        "feature_manifest",
        "dynamic_crop_manifest",
    }:
        raise DynamicCropV3ProductionPromotionError(
            f"{expected_song_id} evidence keys are not exact"
        )
    if (
        record.get("song_id") != expected_song_id
        or record.get("evidence_status") != EVIDENCE_STATUS[expected_song_id]
        or record.get("blind_holdout_claim_authorized") is not False
        or record.get("human_parity_claim_authorized") is not False
    ):
        raise DynamicCropV3ProductionPromotionError(
            f"{expected_song_id} evidence claim status differs"
        )
    try:
        materialization_pointer, materialization = (
            manifests._validate_materialization_manifest(
                record.get("materialization_manifest"),
                song_id=expected_song_id,
            )
        )
        feature_pointer, feature = manifests._validate_feature_manifest(
            record.get("feature_manifest"),
            song_id=expected_song_id,
            materialization=materialization,
        )
        dynamic_pointer = manifests._validate_dynamic_crop_manifest(
            record.get("dynamic_crop_manifest"),
            song_id=expected_song_id,
            materialization=materialization,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DynamicCropV3ProductionPromotionError(
            f"{expected_song_id} evidence validation failed: {exc}"
        ) from exc
    return PromotionSongEvidence(
        song_id=expected_song_id,
        evidence_status=EVIDENCE_STATUS[expected_song_id],
        materialization_manifest_path=materialization_pointer.path,
        feature_manifest_path=feature_pointer.path,
        dynamic_crop_manifest_path=dynamic_pointer.path,
        reference_ids=materialization.reference_ids,
        supervision_paths=dict(materialization.supervision_paths),
        snapshot_paths=dict(feature.snapshot_paths),
    )


def _evidence_plan_record(
    song_id: str,
    *,
    materialization_manifest_path: Path,
    feature_manifest_path: Path,
    dynamic_crop_manifest_path: Path,
) -> dict[str, Any]:
    record = {
        "song_id": song_id,
        "evidence_status": EVIDENCE_STATUS[song_id],
        "blind_holdout_claim_authorized": False,
        "human_parity_claim_authorized": False,
        "materialization_manifest": _manifest_pointer(
            materialization_manifest_path,
            f"{song_id} materialization manifest",
        ),
        "feature_manifest": _manifest_pointer(
            feature_manifest_path, f"{song_id} feature manifest"
        ),
        "dynamic_crop_manifest": _manifest_pointer(
            dynamic_crop_manifest_path, f"{song_id} dynamic crop manifest"
        ),
    }
    _validate_song_evidence_record(record, expected_song_id=song_id)
    return record


def _promotion_authority_contract() -> dict[str, Any]:
    return {
        "production_ready": True,
        "production_registry_activation_authorized": True,
        "production_registry_activation_requested": True,
        "continuity_authority": HANDCRAFTED_CONTINUITY_AUTHORITY,
        "learned_source_pair_authority_present": False,
        "blind_holdout_quality_claim_authorized": False,
        "xoxz_evidence_status": EVIDENCE_STATUS[XOXZ],
        "attitude_evidence_status": EVIDENCE_STATUS[ATTITUDE],
        "human_parity_claim_authorized": False,
        "final_fit_weight_retrained_or_modified": False,
    }


def _evaluation_protocol_contract(
    *,
    checkpoint_id: str,
    checkpoint_weight_sha256: str,
    cut_probability_threshold: float,
    quality_thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "evaluation_song_ids": list(EVALUATION_SONG_IDS),
        "evaluated_checkpoint_id": checkpoint_id,
        "evaluated_checkpoint_weight_sha256": checkpoint_weight_sha256,
        "cut_probability_threshold": cut_probability_threshold,
        "quality_thresholds": dict(quality_thresholds),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "gradient_computation_performed": False,
        "gradient_updates_performed": False,
        "checkpoint_or_model_reselection_performed": False,
        "threshold_reselection_performed": False,
        "quality_threshold_reselection_performed": False,
        "evidence_song_used_for_training": False,
        "evaluation_results_used_for_tuning": False,
        "single_exact_copied_weight_evaluated": True,
    }


def create_dynamic_crop_v3_production_promotion_plan(
    *,
    final_fit_receipt_path: Path,
    materialization_manifests: Mapping[str, Path],
    feature_manifests: Mapping[str, Path],
    dynamic_crop_manifests: Mapping[str, Path],
    checkpoint_directory: Path,
    evidence_directory: Path,
    output_directory: Path,
    production_registry_path: Path | None = None,
) -> Path:
    """Predeclare both evaluation songs before any promotion inference."""

    for mapping_value, label in (
        (materialization_manifests, "materialization manifests"),
        (feature_manifests, "feature manifests"),
        (dynamic_crop_manifests, "dynamic crop manifests"),
    ):
        if set(mapping_value) != set(EVALUATION_SONG_IDS):
            raise DynamicCropV3ProductionPromotionError(
                f"{label} must contain exactly XOXZ and ATTITUDE"
            )
    lineage = _replay_lineage(final_fit_receipt_path)
    source = _capture_source_provenance()
    if lineage.get("source_commit") != source.get("head_commit_oid"):
        raise DynamicCropV3ProductionPromotionError(
            "final fit and promotion source must share one clean commit"
        )
    destination = _under_production_volume(
        checkpoint_directory, "promotion checkpoint directory"
    )
    evidence = _under_production_volume(
        evidence_directory, "promotion evidence directory"
    )
    output = _outside_repository(output_directory, "promotion plan directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite promotion checkpoint: {destination}"
        )
    registry = Path(
        production_registry_path or preferred_stage_edit_production_registry_path()
    ).expanduser().resolve()
    if not registry.is_absolute() or registry.is_symlink():
        raise DynamicCropV3ProductionPromotionError(
            "production registry path must be absolute and non-symlink"
        )
    if any(
        left == right or left in right.parents or right in left.parents
        for left, right in (
            (destination, evidence),
            (destination, output),
            (destination, registry),
        )
    ):
        raise DynamicCropV3ProductionPromotionError(
            "checkpoint, evidence, plan and registry paths must be isolated"
        )
    evaluation_records = [
        _evidence_plan_record(
            song_id,
            materialization_manifest_path=materialization_manifests[song_id],
            feature_manifest_path=feature_manifests[song_id],
            dynamic_crop_manifest_path=dynamic_crop_manifests[song_id],
        )
        for song_id in EVALUATION_SONG_IDS
    ]
    freeze = lineage["release_freeze"]
    metadata = lineage["metadata"]
    assert isinstance(metadata, StageEditDynamicCropCheckpointMetadataV3)
    profile_path = _verify_pointer(
        freeze.get("professional_montage_profile"),
        "professional montage profile",
    )
    policy = load_professional_montage_policy(profile_path)
    if policy.identity.to_dict() != freeze.get(
        "professional_montage_profile_identity"
    ):
        raise DynamicCropV3ProductionPromotionError(
            "professional montage profile differs from release freeze"
        )
    report = lineage["final_fit_report"]
    frozen = _mapping(report.get("frozen_decisions"), "final-fit frozen decisions")
    protocol = _evaluation_protocol_contract(
        checkpoint_id=metadata.checkpoint_id,
        checkpoint_weight_sha256=str(metadata.weight_sha256),
        cut_probability_threshold=float(frozen["cut_probability_threshold"]),
        quality_thresholds=_mapping(
            frozen.get("quality_thresholds"), "frozen quality thresholds"
        ),
    )
    unsigned = {
        "schema_version": PROMOTION_PLAN_SCHEMA_VERSION,
        "analysis_basis": PROMOTION_PLAN_ANALYSIS_BASIS,
        "contract": PROMOTION_PLAN_CONTRACT,
        "source_provenance": source,
        "lineage": _lineage_plan_record(lineage),
        "evaluation_evidence": evaluation_records,
        "evaluation_protocol": protocol,
        "destination_checkpoint_directory": str(destination),
        "evidence_directory": str(evidence),
        "production_registry_path": str(registry),
        "professional_montage_profile": policy.identity.to_dict(),
        "authority": _promotion_authority_contract(),
        "plan_content_sha256": None,
    }
    return _write_content_addressed(
        output_directory=output,
        prefix=PROMOTION_PLAN_PREFIX,
        unsigned=unsigned,
        hash_field="plan_content_sha256",
    )


_PLAN_KEYS = {
    "schema_version",
    "analysis_basis",
    "contract",
    "source_provenance",
    "lineage",
    "evaluation_evidence",
    "evaluation_protocol",
    "destination_checkpoint_directory",
    "evidence_directory",
    "production_registry_path",
    "professional_montage_profile",
    "authority",
    "plan_content_sha256",
}


def _strict_plan(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = _read_json(resolved, "production promotion plan")
    if set(payload) != _PLAN_KEYS:
        raise DynamicCropV3ProductionPromotionError(
            "production promotion plan keys are not exact"
        )
    digest = payload.get("plan_content_sha256")
    unsigned = deepcopy(payload)
    unsigned["plan_content_sha256"] = None
    if (
        not _is_sha256(digest)
        or _canonical_sha256(unsigned) != digest
        or resolved.name != f"{PROMOTION_PLAN_PREFIX}{digest}.json"
        or payload.get("schema_version") != PROMOTION_PLAN_SCHEMA_VERSION
        or payload.get("analysis_basis") != PROMOTION_PLAN_ANALYSIS_BASIS
        or payload.get("contract") != PROMOTION_PLAN_CONTRACT
        or payload.get("authority") != _promotion_authority_contract()
    ):
        raise DynamicCropV3ProductionPromotionError(
            "production promotion plan contract or content address differs"
        )
    source = _validate_source_provenance(payload.get("source_provenance"))
    lineage_record = _mapping(payload.get("lineage"), "promotion lineage")
    receipt_path = _verify_pointer(
        lineage_record.get("final_fit_receipt"), "final-fit receipt"
    )
    replayed = _replay_lineage(receipt_path)
    if (
        _lineage_plan_record(replayed) != lineage_record
        or replayed.get("source_commit") != source.get("head_commit_oid")
    ):
        raise DynamicCropV3ProductionPromotionError(
            "promotion lineage no longer replays exactly"
        )
    raw_destination = payload.get("destination_checkpoint_directory")
    raw_evidence = payload.get("evidence_directory")
    raw_registry = payload.get("production_registry_path")
    if any(
        not isinstance(value, str) or not Path(value).is_absolute()
        for value in (raw_destination, raw_evidence, raw_registry)
    ):
        raise DynamicCropV3ProductionPromotionError(
            "promotion output and registry paths must be explicit absolute paths"
        )
    destination = _under_production_volume(
        Path(str(raw_destination)), "promotion checkpoint directory"
    )
    evidence = _under_production_volume(
        Path(str(raw_evidence)), "promotion evidence directory"
    )
    registry = Path(str(raw_registry)).resolve()
    if registry.is_symlink() or any(
        left == right or left in right.parents or right in left.parents
        for left, right in (
            (destination, evidence),
            (destination, registry),
        )
    ):
        raise DynamicCropV3ProductionPromotionError(
            "promotion checkpoint, evidence, and registry paths are unsafe"
        )
    metadata = replayed["metadata"]
    assert isinstance(metadata, StageEditDynamicCropCheckpointMetadataV3)
    final_report = replayed["final_fit_report"]
    frozen = _mapping(
        final_report.get("frozen_decisions"), "final-fit frozen decisions"
    )
    expected_protocol = _evaluation_protocol_contract(
        checkpoint_id=metadata.checkpoint_id,
        checkpoint_weight_sha256=str(metadata.weight_sha256),
        cut_probability_threshold=float(frozen["cut_probability_threshold"]),
        quality_thresholds=_mapping(
            frozen.get("quality_thresholds"), "frozen quality thresholds"
        ),
    )
    if payload.get("evaluation_protocol") != expected_protocol:
        raise DynamicCropV3ProductionPromotionError(
            "promotion evaluation protocol differs from final-fit freeze"
        )
    evidence_records = payload.get("evaluation_evidence")
    if not isinstance(evidence_records, list) or len(evidence_records) != 2:
        raise DynamicCropV3ProductionPromotionError(
            "promotion requires exactly two predeclared evidence songs"
        )
    if [record.get("song_id") for record in evidence_records] != list(
        EVALUATION_SONG_IDS
    ):
        raise DynamicCropV3ProductionPromotionError(
            "promotion evidence song order or identity differs"
        )
    for song_id, record in zip(EVALUATION_SONG_IDS, evidence_records):
        _validate_song_evidence_record(record, expected_song_id=song_id)
    profile = _mapping(
        payload.get("professional_montage_profile"),
        "professional montage profile",
    )
    try:
        policy = load_professional_montage_policy(Path(str(profile.get("path"))))
    except (OSError, ProfessionalMontagePolicyError, ValueError) as exc:
        raise DynamicCropV3ProductionPromotionError(str(exc)) from exc
    if policy.identity.to_dict() != profile:
        raise DynamicCropV3ProductionPromotionError(
            "professional montage profile changed"
        )
    if profile != replayed["release_freeze"].get(
        "professional_montage_profile_identity"
    ):
        raise DynamicCropV3ProductionPromotionError(
            "promotion montage profile differs from the LOSO release freeze"
        )
    return payload


def preflight_dynamic_crop_v3_production_promotion(
    plan_path: Path,
) -> DynamicCropV3ProductionPromotionPreflight:
    resolved = Path(plan_path).expanduser().resolve()
    plan = _strict_plan(resolved)
    destination = _under_production_volume(
        Path(str(plan["destination_checkpoint_directory"])),
        "promotion checkpoint directory",
    )
    evidence_directory = _under_production_volume(
        Path(str(plan["evidence_directory"])),
        "promotion evidence directory",
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite promotion checkpoint: {destination}"
        )
    if not PRODUCTION_VOLUME_ROOT.is_dir() or not os.path.ismount(
        PRODUCTION_VOLUME_ROOT
    ):
        raise DynamicCropV3ProductionPromotionError(
            f"required external SSD is not mounted: {PRODUCTION_VOLUME_ROOT}"
        )
    free_bytes = shutil.disk_usage(PRODUCTION_VOLUME_ROOT).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise DynamicCropV3ProductionPromotionError(
            "external SSD free space is below the promotion minimum"
        )
    registry = Path(str(plan["production_registry_path"])).resolve()
    if registry.exists() or registry.is_symlink():
        raise DynamicCropV3ProductionPromotionError(
            "a production registry already exists before promotion"
        )
    evidence_records = plan["evaluation_evidence"]
    song_evidence = {
        song_id: _validate_song_evidence_record(
            record, expected_song_id=song_id
        )
        for song_id, record in zip(EVALUATION_SONG_IDS, evidence_records)
    }
    lineage = _mapping(plan["lineage"], "promotion lineage")
    source_checkpoint = _mapping(
        lineage.get("source_checkpoint"), "source checkpoint"
    )
    metadata_path = _verify_pointer(
        source_checkpoint.get("metadata"), "source checkpoint metadata"
    )
    metadata = StageEditDynamicCropCheckpointMetadataV3.from_dict(
        _read_json(metadata_path, "source checkpoint metadata")
    )
    protocol = _mapping(plan["evaluation_protocol"], "evaluation protocol")
    return DynamicCropV3ProductionPromotionPreflight(
        plan_path=resolved,
        plan_content_sha256=str(plan["plan_content_sha256"]),
        source_checkpoint_directory=metadata_path.parent,
        destination_checkpoint_directory=destination,
        evidence_directory=evidence_directory,
        production_registry_path=registry,
        professional_montage_profile_path=Path(
            str(plan["professional_montage_profile"]["path"])
        ).resolve(),
        checkpoint_id=metadata.checkpoint_id,
        checkpoint_weight_sha256=str(metadata.weight_sha256),
        frozen_cut_probability_threshold=float(
            protocol["cut_probability_threshold"]
        ),
        quality_thresholds=dict(protocol["quality_thresholds"]),
        song_evidence=song_evidence,
        free_bytes=free_bytes,
    )


def _build_evaluation_corpus(
    preflight: DynamicCropV3ProductionPromotionPreflight,
) -> tuple[ReferenceTrainingCorpus, Any]:
    supervision_paths: list[Path] = []
    song_ids_by_reference: dict[str, str] = {}
    snapshots: list[str] = []
    dynamic_manifests: list[Path] = []
    for song_id in EVALUATION_SONG_IDS:
        evidence = preflight.song_evidence[song_id]
        dynamic_manifests.append(evidence.dynamic_crop_manifest_path)
        for reference_id in evidence.reference_ids:
            supervision_paths.append(evidence.supervision_paths[reference_id])
            song_ids_by_reference[reference_id] = song_id
            snapshots.append(
                f"{reference_id}={evidence.snapshot_paths[reference_id]}"
            )
    corpus = load_reference_training_corpus(
        supervision_paths,
        feature_snapshot_by_reference=_snapshot_map(snapshots),
        song_id_by_reference=song_ids_by_reference,
        negative_samples_per_hold=int(
            manifests.PRODUCTION_SAMPLING_CONTRACT["negative_samples_per_hold"]
        ),
        boundary_guard_seconds=float(
            manifests.PRODUCTION_SAMPLING_CONTRACT["boundary_guard_seconds"]
        ),
        max_sources=int(manifests.PRODUCTION_SAMPLING_CONTRACT["max_sources"]),
    )
    v2_corpus = build_stage_edit_dynamic_crop_training_corpus_v2(
        corpus.examples,
        artifact_paths_by_reference=_dynamic_artifact_map(
            [], dynamic_manifests
        ),
    )
    return corpus, build_stage_edit_dynamic_crop_training_corpus_v3(v2_corpus)


def _build_final_fit_training_corpus(plan: Mapping[str, Any]) -> ReferenceTrainingCorpus:
    lineage = _mapping(plan.get("lineage"), "promotion lineage")
    final_fit_plan_path = _verify_pointer(
        lineage.get("final_fit_plan"), "final-fit plan"
    )
    final_fit_plan = final_release._strict_final_fit_plan(final_fit_plan_path)
    materialization, feature, dynamic = final_release._manifest_maps_from_plan(
        final_fit_plan.get("songs")
    )
    dataset_path = _verify_pointer(
        final_fit_plan.get("selection_dataset_manifest"),
        "final-fit selection dataset",
    )
    validated = final_release._validate_manifest_set(
        materialization_manifests=materialization,
        feature_manifests=feature,
        dynamic_crop_manifests=dynamic,
        selection_dataset_manifest_path=dataset_path,
    )
    supervision_paths: list[Path] = []
    song_ids_by_reference: dict[str, str] = {}
    snapshot_arguments: list[str] = []
    for song_id in final_release.SELECTION_SONG_IDS:
        for reference_id in validated.reference_ids_by_song[song_id]:
            supervision_paths.append(
                validated.supervision_paths_by_song[song_id][reference_id]
            )
            song_ids_by_reference[reference_id] = song_id
            snapshot_arguments.append(
                f"{reference_id}="
                f"{validated.snapshot_paths_by_song[song_id][reference_id]}"
            )
    return load_reference_training_corpus(
        supervision_paths,
        feature_snapshot_by_reference=_snapshot_map(snapshot_arguments),
        song_id_by_reference=song_ids_by_reference,
        negative_samples_per_hold=int(
            manifests.PRODUCTION_SAMPLING_CONTRACT["negative_samples_per_hold"]
        ),
        boundary_guard_seconds=float(
            manifests.PRODUCTION_SAMPLING_CONTRACT["boundary_guard_seconds"]
        ),
        max_sources=int(manifests.PRODUCTION_SAMPLING_CONTRACT["max_sources"]),
    )


def _loss_contracts_from_final_fit_report(
    final_fit_report: Mapping[str, Any],
) -> tuple[StageEditLossContract, DynamicCropV3LossContract]:
    training = _mapping(final_fit_report.get("training"), "final-fit training")
    base_payload = _mapping(
        training.get("effective_base_loss_contract"),
        "effective base loss contract",
    )
    dynamic_payload = _mapping(
        training.get("effective_dynamic_crop_loss_contract"),
        "effective Dynamic Crop v3 loss contract",
    )
    dynamic_base = _mapping(
        dynamic_payload.get("base"), "effective Dynamic Crop v3 base loss"
    )
    base = StageEditLossContract(**base_payload)
    dynamic = DynamicCropV3LossContract(
        **{**dynamic_payload, "base": StageEditLossContract(**dynamic_base)}
    )
    if dynamic.base != base or dynamic.contract != DYNAMIC_CROP_V3_LOSS_CONTRACT:
        raise DynamicCropV3ProductionPromotionError(
            "final-fit effective loss contracts differ"
        )
    return base, dynamic


def _evaluate_exact_copied_checkpoint(
    *,
    checkpoint_directory: Path,
    preflight: DynamicCropV3ProductionPromotionPreflight,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Forward-only evaluation; this function never constructs an optimizer."""

    model, metadata = load_stage_edit_dynamic_crop_checkpoint_v3(
        checkpoint_directory,
        expected_schema=DEFAULT_STAGE_EDIT_SCHEMA,
    )
    if (
        metadata.checkpoint_id != preflight.checkpoint_id
        or metadata.weight_sha256 != preflight.checkpoint_weight_sha256
    ):
        raise DynamicCropV3ProductionPromotionError(
            "copied checkpoint identity differs before evaluation"
        )
    lineage = _mapping(plan.get("lineage"), "promotion lineage")
    final_fit_report_path = _verify_pointer(
        _mapping(lineage.get("source_checkpoint"), "source checkpoint").get(
            "final_fit_report"
        ),
        "final-fit report",
    )
    final_fit_report = final_release.load_dynamic_crop_v3_final_fit_report(
        final_fit_report_path
    )
    base_loss, dynamic_loss = _loss_contracts_from_final_fit_report(
        final_fit_report
    )
    thresholds = ProductionQualityThresholds(**dict(preflight.quality_thresholds))
    evaluation_corpus, dynamic_corpus = _build_evaluation_corpus(preflight)
    final_fit_training_corpus = _build_final_fit_training_corpus(plan)
    active_device = production.mx.default_device()
    planned_evidence = {
        str(record["song_id"]): record
        for record in plan["evaluation_evidence"]
    }
    evaluations: list[dict[str, Any]] = []
    for song_id in EVALUATION_SONG_IDS:
        examples = tuple(
            example
            for example in evaluation_corpus.examples
            if str(example.song_id) == song_id
        )
        if len(examples) != 5:
            raise DynamicCropV3ProductionPromotionError(
                f"{song_id} evaluation requires exactly five references"
            )
        with production.mx.stream(active_device):
            batch = tensorize_stage_edit_batch(
                prepare_stage_edit_production_input_sequences(
                    examples, schema=DEFAULT_STAGE_EDIT_SCHEMA
                ),
                DEFAULT_STAGE_EDIT_SCHEMA,
                metadata.base_model_config,
                device=active_device,
            )
            supervision = tensorize_stage_edit_supervision(
                tuple(example.targets for example in examples),
                batch,
                DEFAULT_STAGE_EDIT_SCHEMA,
                device=active_device,
            )
            base_metrics = evaluate_stage_edit_model(
                model,
                batch,
                supervision,
                loss_contract=base_loss,
                cut_probability_threshold=(
                    preflight.frozen_cut_probability_threshold
                ),
            )
            base_metrics = {
                **base_metrics,
                "cut": {
                    **base_metrics["cut"],
                    "balanced_accuracy_at_frozen_validation_threshold": (
                        base_metrics["cut"][
                            "balanced_accuracy_at_threshold"
                        ]
                    ),
                    "frozen_validation_threshold": (
                        preflight.frozen_cut_probability_threshold
                    ),
                    "threshold_contract": (
                        STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT
                    ),
                },
            }
            dynamic_supervision = (
                tensorize_stage_edit_dynamic_crop_supervision_v3(
                    tuple(
                        dynamic_corpus.sequence_for(
                            example.sequence.sequence_id
                        )
                        for example in examples
                    ),
                    batch,
                    device=active_device,
                )
            )
            dynamic_metrics = evaluate_stage_edit_dynamic_crop_v3(
                model,
                batch,
                supervision,
                dynamic_supervision,
                contract=dynamic_loss,
            )
        quality = evaluate_stage_edit_production_quality(
            base_metrics,
            dynamic_crop_v3_metrics=dynamic_metrics,
            thresholds=thresholds,
            transition_authority=(
                metadata.base_model_config.transition_authority
            ),
        )
        structured = _structured_head_coverage_gate(
            final_fit_training_corpus.examples,
            base_metrics,
            thresholds=thresholds,
            transition_authority=(
                metadata.base_model_config.transition_authority
            ),
        )
        evidence = preflight.song_evidence[song_id]
        evaluations.append(
            {
                "song_id": song_id,
                "evidence_status": evidence.evidence_status,
                "blind_holdout_claim_authorized": False,
                "human_parity_claim_authorized": False,
                "reference_ids": list(evidence.reference_ids),
                "reference_count": len(evidence.reference_ids),
                "evidence_manifests": {
                    name: deepcopy(planned_evidence[song_id][name])
                    for name in (
                        "materialization_manifest",
                        "feature_manifest",
                        "dynamic_crop_manifest",
                    )
                },
                "checkpoint_id": metadata.checkpoint_id,
                "checkpoint_weight_sha256": metadata.weight_sha256,
                "base_metrics": base_metrics,
                "dynamic_crop_v3_metrics": dynamic_metrics,
                "quality_gate": quality.to_dict(),
                "structured_head_coverage_gate": structured,
                "evaluation_used_for_optimizer_or_gradient": False,
                "evaluation_used_for_model_threshold_or_policy_reselection": False,
            }
        )
    return evaluations


def _copy_file_byte_identical(source: Path, destination: Path, label: str) -> str:
    source_hash = _file_sha256(source, f"source {label}")
    before = source.stat()
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        while block := input_handle.read(4 * 1024 * 1024):
            output_handle.write(block)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    after = source.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise DynamicCropV3ProductionPromotionError(
            f"source {label} changed while copying"
        )
    if destination.stat().st_size != before.st_size:
        raise DynamicCropV3ProductionPromotionError(
            f"copied {label} size differs"
        )
    destination_hash = _file_sha256(destination, f"copied {label}")
    if destination_hash != source_hash:
        raise DynamicCropV3ProductionPromotionError(
            f"copied {label} digest differs"
        )
    return source_hash


def _owned_file_identity(path: Path, label: str) -> tuple[int, int, int, str]:
    candidate = Path(path)
    before = candidate.lstat()
    if not stat_module.S_ISREG(before.st_mode):
        raise DynamicCropV3ProductionPromotionError(
            f"{label} is not an owned regular file"
        )
    digest = _file_sha256(candidate, label)
    after = candidate.lstat()
    identity = (before.st_dev, before.st_ino, before.st_size)
    if identity != (after.st_dev, after.st_ino, after.st_size):
        raise DynamicCropV3ProductionPromotionError(
            f"{label} changed while capturing rollback identity"
        )
    return (*identity, digest)


def _owned_directory_identity(
    path: Path,
    label: str,
    *,
    include_files: bool,
) -> tuple[Any, ...]:
    candidate = Path(path)
    before = candidate.lstat()
    if not stat_module.S_ISDIR(before.st_mode):
        raise DynamicCropV3ProductionPromotionError(
            f"{label} is not an owned directory"
        )
    files: tuple[tuple[str, tuple[int, int, int, str]], ...] = ()
    if include_files:
        entries = tuple(sorted(candidate.iterdir(), key=lambda item: item.name))
        files = tuple(
            (entry.name, _owned_file_identity(entry, f"{label} entry"))
            for entry in entries
        )
    after = candidate.lstat()
    identity = (before.st_dev, before.st_ino)
    if identity != (after.st_dev, after.st_ino):
        raise DynamicCropV3ProductionPromotionError(
            f"{label} changed while capturing rollback identity"
        )
    return (*identity, files)


def _require_owned_identity(
    path: Path,
    expected: tuple[Any, ...] | None,
    label: str,
    *,
    directory: bool,
    include_files: bool = False,
) -> None:
    if expected is None:
        raise DynamicCropV3ProductionPromotionError(
            f"{label} rollback identity was never captured"
        )
    observed = (
        _owned_directory_identity(
            path, label, include_files=include_files
        )
        if directory
        else _owned_file_identity(path, label)
    )
    if observed != expected:
        raise DynamicCropV3ProductionPromotionError(
            f"{label} identity changed; unsafe rollback refused"
        )


def _write_report_exclusive(path: Path, payload: Mapping[str, Any]) -> Path:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return path


_REPORT_KEYS = {
    "schema_version",
    "run_kind",
    "contract",
    "checkpoint_scope",
    "created_at",
    "source_commit",
    "production_ready",
    "promotion_plan",
    "promotion_plan_content_sha256",
    "lineage",
    "checkpoint",
    "evaluation_protocol",
    "evaluations",
    "aggregate_gate",
    "authority",
    "report_sha256",
}


def _light_pointer_errors(value: object, label: str) -> list[str]:
    errors: list[str] = []
    try:
        _verify_pointer(value, label)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"{label} changed or is invalid: {exc}")
    return errors


def _lineage_report_errors(value: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return ["promotion lineage is missing"]
    if set(value) != {
        "final_fit_receipt",
        "final_fit_receipt_content_sha256",
        "final_fit_plan",
        "release_freeze",
        "release_freeze_content_sha256",
        "loso_fold_receipts",
        "source_checkpoint",
    }:
        return ["promotion lineage keys are not exact"]
    lineage_paths: dict[str, Path] = {}
    for field in ("final_fit_receipt", "final_fit_plan", "release_freeze"):
        errors.extend(_light_pointer_errors(value.get(field), field))
        try:
            lineage_paths[field] = _verify_pointer(value.get(field), field)
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
    if not _is_sha256(value.get("final_fit_receipt_content_sha256")):
        errors.append("final-fit receipt content hash is invalid")
    if not _is_sha256(value.get("release_freeze_content_sha256")):
        errors.append("release freeze content hash is invalid")
    try:
        receipt_payload = _read_json(
            lineage_paths["final_fit_receipt"], "final-fit receipt"
        )
        if receipt_payload.get("artifact_content_sha256") != value.get(
            "final_fit_receipt_content_sha256"
        ):
            errors.append("final-fit receipt content identity differs")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        pass
    try:
        freeze_payload = _read_json(
            lineage_paths["release_freeze"], "release freeze"
        )
        if freeze_payload.get("freeze_content_sha256") != value.get(
            "release_freeze_content_sha256"
        ):
            errors.append("release freeze content identity differs")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        pass
    folds = value.get("loso_fold_receipts")
    if not isinstance(folds, list) or len(folds) != 3:
        errors.append("promotion lineage does not bind exactly three LOSO receipts")
    else:
        songs: list[str] = []
        for index, raw in enumerate(folds):
            if not isinstance(raw, Mapping) or set(raw) != {
                "validation_song_id",
                "checkpoint_id",
                "checkpoint_weight_sha256",
                "receipt",
                "receipt_content_sha256",
            }:
                errors.append(f"LOSO lineage record {index} is invalid")
                continue
            songs.append(str(raw.get("validation_song_id")))
            if not _is_sha256(raw.get("checkpoint_weight_sha256")) or not _is_sha256(
                raw.get("receipt_content_sha256")
            ):
                errors.append(f"LOSO lineage record {index} hashes are invalid")
            errors.extend(
                _light_pointer_errors(raw.get("receipt"), f"LOSO receipt {index}")
            )
            try:
                fold_receipt_path = _verify_pointer(
                    raw.get("receipt"), f"LOSO receipt {index}"
                )
                fold_receipt = _read_json(
                    fold_receipt_path, f"LOSO receipt {index}"
                )
                if any(
                    fold_receipt.get(name) != raw.get(name)
                    for name in (
                        "validation_song_id",
                        "checkpoint_id",
                        "checkpoint_weight_sha256",
                    )
                ) or fold_receipt.get("artifact_content_sha256") != raw.get(
                    "receipt_content_sha256"
                ):
                    errors.append(
                        f"LOSO lineage record {index} content identity differs"
                    )
            except (OSError, RuntimeError, TypeError, ValueError):
                pass
        if set(songs) != set(final_release.SELECTION_SONG_IDS):
            errors.append("LOSO lineage songs differ")
    source = value.get("source_checkpoint")
    if not isinstance(source, Mapping) or set(source) != {
        "directory",
        "metadata",
        "weights",
        "final_fit_report",
        "checkpoint_id",
        "training_steps",
        "weight_sha256",
        "production_activation_authorized_in_metadata",
    }:
        errors.append("source final-fit checkpoint lineage is invalid")
    else:
        resolved_source_paths: dict[str, Path] = {}
        for field in ("metadata", "weights", "final_fit_report"):
            errors.extend(
                _light_pointer_errors(source.get(field), f"source checkpoint {field}")
            )
            try:
                resolved_source_paths[field] = Path(
                    str(_mapping(source.get(field), field)["path"])
                ).resolve()
            except (KeyError, TypeError, ValueError):
                pass
        if (
            not _is_sha256(source.get("weight_sha256"))
            or not isinstance(source.get("checkpoint_id"), str)
            or not source.get("checkpoint_id")
            or not isinstance(source.get("training_steps"), int)
            or isinstance(source.get("training_steps"), bool)
            or source.get("training_steps") <= 0
            or source.get("production_activation_authorized_in_metadata")
            is not False
        ):
            errors.append("source final-fit checkpoint authority differs")
        raw_directory = source.get("directory")
        if (
            not isinstance(raw_directory, str)
            or not Path(raw_directory).is_absolute()
            or len(resolved_source_paths) != 3
            or any(
                path.parent != Path(raw_directory).resolve()
                for path in resolved_source_paths.values()
            )
        ):
            errors.append("source final-fit checkpoint paths are not co-located")
        elif (
            resolved_source_paths["metadata"].name
            != DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
            or resolved_source_paths["weights"].name
            != DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
            or resolved_source_paths["final_fit_report"].name
            != final_release.FINAL_FIT_REPORT_FILENAME
        ):
            errors.append("source final-fit checkpoint filenames differ")
        else:
            try:
                source_metadata = StageEditDynamicCropCheckpointMetadataV3.from_dict(
                    _read_json(
                        resolved_source_paths["metadata"],
                        "source final-fit checkpoint metadata",
                    )
                )
                source_weight_sha = _file_sha256(
                    resolved_source_paths["weights"],
                    "source final-fit checkpoint weights",
                )
                source_report = _read_json(
                    resolved_source_paths["final_fit_report"],
                    "source final-fit report",
                )
                source_report_checkpoint = source_report.get("checkpoint")
                if (
                    source_metadata.checkpoint_id != source.get("checkpoint_id")
                    or source_metadata.training_steps != source.get("training_steps")
                    or source_metadata.weight_sha256 != source.get("weight_sha256")
                    or source_metadata.production_activation_authorized is not False
                    or source_weight_sha != source.get("weight_sha256")
                    or not isinstance(source_report_checkpoint, Mapping)
                    or source_report_checkpoint.get("checkpoint_id")
                    != source.get("checkpoint_id")
                    or source_report_checkpoint.get("weight_sha256")
                    != source.get("weight_sha256")
                    or source_report_checkpoint.get("directory")
                    != str(Path(str(raw_directory)).resolve())
                ):
                    errors.append("source final-fit checkpoint identity differs")
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                errors.append(
                    f"source final-fit checkpoint cannot be replayed: {exc}"
                )
    return errors


def production_promotion_report_errors(
    payload: object,
    *,
    metadata: Any,
    actual_weight_sha256: str | None,
    checkpoint_directory: Path | None = None,
    checkpoint_files_directory: Path | None = None,
) -> tuple[str, ...]:
    """Validate the alternate promotion report without weakening fold reports."""

    if not isinstance(payload, Mapping):
        return ("production promotion report root must be an object",)
    errors: list[str] = []
    if set(payload) != _REPORT_KEYS:
        errors.append("production promotion report keys are not exact")
    if (
        payload.get("schema_version") != PRODUCTION_PROMOTION_REPORT_SCHEMA_VERSION
        or payload.get("run_kind") != PRODUCTION_PROMOTION_RUN_KIND
        or payload.get("contract") != PRODUCTION_PROMOTION_REPORT_CONTRACT
        or payload.get("checkpoint_scope")
        != PRODUCTION_PROMOTION_CHECKPOINT_SCOPE
        or payload.get("production_ready") is not True
    ):
        errors.append("production promotion report contract differs")
    unsigned = dict(payload)
    digest = unsigned.pop("report_sha256", None)
    if not _is_sha256(digest) or _canonical_sha256(unsigned) != digest:
        errors.append("production promotion report SHA-256 differs")
    if not _is_commit_oid(payload.get("source_commit")):
        errors.append("production promotion source commit is invalid")
    try:
        created_at = datetime.fromisoformat(str(payload.get("created_at")))
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("timestamp lacks a UTC offset")
    except ValueError:
        errors.append("production promotion timestamp is invalid")

    plan: Mapping[str, Any] | None = None
    planned_song_evidence: dict[str, PromotionSongEvidence] = {}
    final_fit_training_examples: Sequence[Any] | None = None
    try:
        plan_path = _verify_pointer(
            payload.get("promotion_plan"), "promotion plan"
        )
        plan = _strict_plan(plan_path)
        if payload.get("promotion_plan_content_sha256") != plan.get(
            "plan_content_sha256"
        ):
            errors.append("production promotion plan content identity differs")
        planned_records = plan.get("evaluation_evidence")
        if not isinstance(planned_records, list) or len(planned_records) != 2:
            errors.append("production promotion plan evidence is invalid")
        else:
            for song_id, record in zip(EVALUATION_SONG_IDS, planned_records):
                planned_song_evidence[song_id] = (
                    _validate_song_evidence_record(
                        record, expected_song_id=song_id
                    )
                )
        final_fit_training_examples = tuple(
            _build_final_fit_training_corpus(plan).examples
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"production promotion plan replay failed: {exc}")

    lineage = payload.get("lineage")
    errors.extend(_lineage_report_errors(lineage))

    if isinstance(plan, Mapping):
        if payload.get("source_commit") != _mapping(
            plan.get("source_provenance"), "promotion source provenance"
        ).get("head_commit_oid"):
            errors.append("production promotion source commit differs from plan")
        if lineage != plan.get("lineage"):
            errors.append("production promotion lineage differs from plan")

    source_lineage = (
        lineage.get("source_checkpoint")
        if isinstance(lineage, Mapping)
        else None
    )
    release_freeze: Mapping[str, Any] | None = None
    if isinstance(lineage, Mapping):
        try:
            release_freeze_path = _verify_pointer(
                lineage.get("release_freeze"), "release freeze"
            )
            release_freeze = _read_json(
                release_freeze_path, "release freeze"
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
    if isinstance(release_freeze, Mapping):
        freeze_source = release_freeze.get("source_provenance")
        if (
            not isinstance(freeze_source, Mapping)
            or payload.get("source_commit")
            != freeze_source.get("head_commit_oid")
        ):
            errors.append("production promotion source commit differs from freeze")

    checkpoint = payload.get("checkpoint")
    resolved_checkpoint = (
        Path(checkpoint_directory).resolve()
        if checkpoint_directory is not None
        else None
    )
    resolved_checkpoint_files = (
        Path(checkpoint_files_directory).resolve()
        if checkpoint_files_directory is not None
        else resolved_checkpoint
    )
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "directory",
        "checkpoint_format",
        "dynamic_crop_version",
        "metadata_filename",
        "weight_filename",
        "checkpoint_id",
        "training_steps",
        "weight_sha256",
        "model_contract",
        "target_contract",
        "output_contract",
        "production_activation_authorized",
        "copy_integrity",
    }:
        errors.append("production promotion checkpoint evidence is invalid")
    else:
        copy_integrity = checkpoint.get("copy_integrity")
        raw_checkpoint_directory = checkpoint.get("directory")
        declared_checkpoint: Path | None = None
        if (
            not isinstance(raw_checkpoint_directory, str)
            or not Path(raw_checkpoint_directory).is_absolute()
        ):
            errors.append("production promotion checkpoint directory is invalid")
        else:
            declared_checkpoint = Path(raw_checkpoint_directory).resolve()
            if resolved_checkpoint is None:
                resolved_checkpoint = declared_checkpoint
            if resolved_checkpoint_files is None:
                resolved_checkpoint_files = declared_checkpoint
        expected_directory = str(resolved_checkpoint) if resolved_checkpoint else None
        expected_checkpoint_fields = {
            "checkpoint_format": "dynamic_crop_v3",
            "dynamic_crop_version": 3,
            "metadata_filename": DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
            "weight_filename": DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
            "checkpoint_id": getattr(metadata, "checkpoint_id", None),
            "training_steps": getattr(metadata, "training_steps", None),
            "weight_sha256": actual_weight_sha256,
            "model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
            "target_contract": DYNAMIC_CROP_V3_TARGET_CONTRACT,
            "output_contract": DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
            "production_activation_authorized": True,
        }
        mismatched_checkpoint_fields = [
            name
            for name, expected in expected_checkpoint_fields.items()
            if checkpoint.get(name) != expected
        ]
        if expected_directory is not None and checkpoint.get(
            "directory"
        ) != expected_directory:
            mismatched_checkpoint_fields.append("directory")
        if isinstance(plan, Mapping) and checkpoint.get("directory") != plan.get(
            "destination_checkpoint_directory"
        ):
            mismatched_checkpoint_fields.append("planned_directory")
        if actual_weight_sha256 is None:
            mismatched_checkpoint_fields.append("actual_weight_sha256")
        if mismatched_checkpoint_fields:
            errors.append(
                "production promotion checkpoint identity differs: "
                + ", ".join(sorted(set(mismatched_checkpoint_fields)))
            )
        if not isinstance(copy_integrity, Mapping) or set(copy_integrity) != {
            "source_checkpoint_directory",
            "source_metadata_sha256",
            "source_weight_sha256",
            "destination_metadata_sha256",
            "destination_weight_sha256",
            "metadata_byte_identical",
            "weights_byte_identical",
            "atomically_installed",
        }:
            errors.append("production promotion copy integrity is invalid")
        else:
            source_metadata_pointer = (
                source_lineage.get("metadata")
                if isinstance(source_lineage, Mapping)
                else None
            )
            source_weights_pointer = (
                source_lineage.get("weights")
                if isinstance(source_lineage, Mapping)
                else None
            )
            source_metadata_sha = (
                source_metadata_pointer.get("file_sha256")
                if isinstance(source_metadata_pointer, Mapping)
                else None
            )
            source_weight_sha = (
                source_weights_pointer.get("file_sha256")
                if isinstance(source_weights_pointer, Mapping)
                else None
            )
            source_checkpoint_directory = (
                str(Path(str(source_lineage.get("directory"))).resolve())
                if isinstance(source_lineage, Mapping)
                and isinstance(source_lineage.get("directory"), str)
                and Path(str(source_lineage.get("directory"))).is_absolute()
                else None
            )
            actual_destination_metadata_sha: str | None = None
            actual_destination_weight_sha: str | None = None
            if resolved_checkpoint_files is not None:
                try:
                    actual_destination_metadata_sha = _file_sha256(
                        resolved_checkpoint_files
                        / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
                        "promoted checkpoint metadata",
                    )
                    actual_destination_weight_sha = _file_sha256(
                        resolved_checkpoint_files
                        / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
                        "promoted checkpoint weights",
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    errors.append(
                        f"production promotion destination copy cannot be read: {exc}"
                    )
            if (
                copy_integrity.get("source_checkpoint_directory")
                != source_checkpoint_directory
                or copy_integrity.get("source_metadata_sha256")
                != source_metadata_sha
                or copy_integrity.get("source_weight_sha256")
                != source_weight_sha
                or copy_integrity.get("source_weight_sha256")
                != (
                    source_lineage.get("weight_sha256")
                    if isinstance(source_lineage, Mapping)
                    else None
                )
                or copy_integrity.get("source_metadata_sha256")
                != copy_integrity.get("destination_metadata_sha256")
                or copy_integrity.get("source_weight_sha256")
                != copy_integrity.get("destination_weight_sha256")
                or copy_integrity.get("destination_metadata_sha256")
                != actual_destination_metadata_sha
                or copy_integrity.get("destination_weight_sha256")
                != actual_destination_weight_sha
                or copy_integrity.get("destination_weight_sha256")
                != actual_weight_sha256
                or copy_integrity.get("metadata_byte_identical") is not True
                or copy_integrity.get("weights_byte_identical") is not True
                or copy_integrity.get("atomically_installed") is not True
            ):
                errors.append("production promotion copy is not byte-identical")

    protocol = payload.get("evaluation_protocol")
    if not isinstance(protocol, Mapping):
        errors.append("production promotion evaluation protocol is missing")
    else:
        expected_protocol_keys = {
            "evaluation_song_ids",
            "evaluated_checkpoint_id",
            "evaluated_checkpoint_weight_sha256",
            "cut_probability_threshold",
            "quality_thresholds",
            "optimizer_constructed",
            "optimizer_steps",
            "gradient_computation_performed",
            "gradient_updates_performed",
            "checkpoint_or_model_reselection_performed",
            "threshold_reselection_performed",
            "quality_threshold_reselection_performed",
            "evidence_song_used_for_training",
            "evaluation_results_used_for_tuning",
            "single_exact_copied_weight_evaluated",
        }
        threshold = protocol.get("cut_probability_threshold")
        if (
            set(protocol) != expected_protocol_keys
            or protocol.get("evaluation_song_ids") != list(EVALUATION_SONG_IDS)
            or protocol.get("evaluated_checkpoint_id")
            != getattr(metadata, "checkpoint_id", None)
            or protocol.get("evaluated_checkpoint_weight_sha256")
            != actual_weight_sha256
            or not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
            or not 0.0 < float(threshold) < 1.0
            or not isinstance(protocol.get("quality_thresholds"), Mapping)
            or protocol.get("optimizer_constructed") is not False
            or protocol.get("optimizer_steps") != 0
            or protocol.get("gradient_computation_performed") is not False
            or protocol.get("gradient_updates_performed") is not False
            or protocol.get("checkpoint_or_model_reselection_performed") is not False
            or protocol.get("threshold_reselection_performed") is not False
            or protocol.get("quality_threshold_reselection_performed") is not False
            or protocol.get("evidence_song_used_for_training") is not False
            or protocol.get("evaluation_results_used_for_tuning") is not False
            or protocol.get("single_exact_copied_weight_evaluated") is not True
        ):
            errors.append("production promotion forward-only protocol differs")
        if isinstance(release_freeze, Mapping) and (
            protocol.get("cut_probability_threshold")
            != release_freeze.get("frozen_cut_probability_threshold")
            or protocol.get("quality_thresholds")
            != release_freeze.get("quality_thresholds")
        ):
            errors.append("production promotion protocol differs from release freeze")
        if isinstance(plan, Mapping) and protocol != plan.get(
            "evaluation_protocol"
        ):
            errors.append("production promotion protocol differs from plan")

    evaluations = payload.get("evaluations")
    renderer_exact_pass_count = 0
    if not isinstance(evaluations, list) or len(evaluations) != 2:
        errors.append("production promotion requires exactly two evaluations")
    else:
        for song_id, evaluation in zip(EVALUATION_SONG_IDS, evaluations):
            renderer_exact = bool(
                isinstance(evaluation, Mapping)
                and _renderer_exact_pass(evaluation)
            )
            renderer_exact_pass_count += int(renderer_exact)
            if not isinstance(evaluation, Mapping) or set(evaluation) != {
                "song_id",
                "evidence_status",
                "blind_holdout_claim_authorized",
                "human_parity_claim_authorized",
                "reference_ids",
                "reference_count",
                "evidence_manifests",
                "checkpoint_id",
                "checkpoint_weight_sha256",
                "base_metrics",
                "dynamic_crop_v3_metrics",
                "quality_gate",
                "structured_head_coverage_gate",
                "evaluation_used_for_optimizer_or_gradient",
                "evaluation_used_for_model_threshold_or_policy_reselection",
            }:
                errors.append(f"{song_id} promotion evaluation schema differs")
                continue
            if not renderer_exact:
                errors.append(f"{song_id} renderer-exact gate did not pass")
            base_metrics = evaluation.get("base_metrics")
            dynamic_metrics = evaluation.get("dynamic_crop_v3_metrics")
            quality = evaluation.get("quality_gate")
            structured = evaluation.get("structured_head_coverage_gate")
            references = evaluation.get("reference_ids")
            evidence_manifests = evaluation.get("evidence_manifests")
            if (
                evaluation.get("song_id") != song_id
                or evaluation.get("evidence_status") != EVIDENCE_STATUS[song_id]
                or evaluation.get("blind_holdout_claim_authorized") is not False
                or evaluation.get("human_parity_claim_authorized") is not False
                or not isinstance(references, list)
                or len(references) != 5
                or len(set(references)) != 5
                or evaluation.get("reference_count") != 5
                or evaluation.get("checkpoint_id")
                != getattr(metadata, "checkpoint_id", None)
                or evaluation.get("checkpoint_weight_sha256")
                != actual_weight_sha256
                or evaluation.get("evaluation_used_for_optimizer_or_gradient") is not False
                or evaluation.get(
                    "evaluation_used_for_model_threshold_or_policy_reselection"
                )
                is not False
            ):
                errors.append(f"{song_id} promotion identity or claims differ")
            cut_metrics = (
                base_metrics.get("cut")
                if isinstance(base_metrics, Mapping)
                else None
            )
            if (
                not isinstance(cut_metrics, Mapping)
                or not isinstance(protocol, Mapping)
                or cut_metrics.get("frozen_validation_threshold")
                != protocol.get("cut_probability_threshold")
                or cut_metrics.get("threshold_contract")
                != STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT
            ):
                errors.append(f"{song_id} frozen cut threshold binding differs")
            if not isinstance(evidence_manifests, Mapping) or set(
                evidence_manifests
            ) != {
                "materialization_manifest",
                "feature_manifest",
                "dynamic_crop_manifest",
            }:
                errors.append(f"{song_id} evidence manifest binding is invalid")
            else:
                for name, pointer in evidence_manifests.items():
                    errors.extend(
                        _light_pointer_errors(
                            pointer, f"{song_id} {name.replace('_', ' ')}"
                        )
                    )
                planned_record = None
                if isinstance(plan, Mapping):
                    plan_records = plan.get("evaluation_evidence")
                    if isinstance(plan_records, list):
                        planned_record = next(
                            (
                                record
                                for record in plan_records
                                if isinstance(record, Mapping)
                                and record.get("song_id") == song_id
                            ),
                            None,
                        )
                expected_manifests = (
                    {
                        name: deepcopy(planned_record[name])
                        for name in (
                            "materialization_manifest",
                            "feature_manifest",
                            "dynamic_crop_manifest",
                        )
                    }
                    if isinstance(planned_record, Mapping)
                    else None
                )
                if evidence_manifests != expected_manifests:
                    errors.append(
                        f"{song_id} evidence manifests differ from plan"
                    )
            planned_evidence = planned_song_evidence.get(song_id)
            if planned_evidence is None or references != list(
                planned_evidence.reference_ids
            ):
                errors.append(f"{song_id} reference ids differ from plan")
            if (
                not isinstance(dynamic_metrics, Mapping)
                or dynamic_metrics.get("schema_version")
                != DYNAMIC_CROP_V3_SCHEMA_VERSION
                or dynamic_metrics.get("model_contract")
                != DYNAMIC_CROP_V3_MODEL_CONTRACT
                or dynamic_metrics.get("target_contract")
                != DYNAMIC_CROP_V3_TARGET_CONTRACT
                or dynamic_metrics.get("output_contract")
                != DYNAMIC_CROP_V3_OUTPUT_CONTRACT
                or dynamic_metrics.get("legacy_v2_checkpoint_or_output_used")
                is not False
                or dynamic_metrics.get(
                    "legacy_static_crop_head_used_for_trajectory"
                )
                is not False
            ):
                errors.append(f"{song_id} Dynamic Crop v3 evaluator contract differs")
            if (
                not isinstance(quality, Mapping)
                or quality.get("passed") is not True
                or quality.get("reasons") != []
                or not isinstance(base_metrics, Mapping)
                or not isinstance(protocol, Mapping)
            ):
                errors.append(f"{song_id} quality gate did not pass")
            else:
                try:
                    recomputed = evaluate_stage_edit_production_quality(
                        base_metrics,
                        dynamic_crop_v3_metrics=dynamic_metrics,
                        thresholds=ProductionQualityThresholds(
                            **dict(protocol["quality_thresholds"])
                        ),
                        transition_authority=(
                            getattr(metadata, "base_model_config").transition_authority
                        ),
                    ).to_dict()
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"{song_id} quality gate cannot replay: {exc}")
                else:
                    if recomputed != quality or not recomputed["passed"]:
                        errors.append(f"{song_id} quality gate replay differs")
            if (
                not isinstance(structured, Mapping)
                or structured.get("contract")
                != STRUCTURED_HEAD_COVERAGE_CONTRACT
                or structured.get("passed") is not True
                or structured.get("reasons") != []
                or structured.get("heldout_used_for_model_selection") is not False
                or structured.get("reference_validation_labels_consumed") is not False
            ):
                errors.append(f"{song_id} structured-head gate did not pass")
            elif isinstance(protocol, Mapping):
                quality_thresholds = protocol.get("quality_thresholds")
                expected_structured_thresholds = (
                    {
                        "minimum_duration_observed_classes": quality_thresholds[
                            "minimum_duration_observed_classes"
                        ],
                        "minimum_duration_label_entropy_bits": quality_thresholds[
                            "minimum_duration_label_entropy_bits"
                        ],
                        "minimum_transition_observed_classes": quality_thresholds[
                            "minimum_transition_observed_classes"
                        ],
                        "minimum_transition_label_entropy_bits": quality_thresholds[
                            "minimum_transition_label_entropy_bits"
                        ],
                    }
                    if isinstance(quality_thresholds, Mapping)
                    and all(
                        name in quality_thresholds
                        for name in (
                            "minimum_duration_observed_classes",
                            "minimum_duration_label_entropy_bits",
                            "minimum_transition_observed_classes",
                            "minimum_transition_label_entropy_bits",
                        )
                    )
                    else None
                )
                if (
                    structured.get("thresholds")
                    != expected_structured_thresholds
                    or structured.get("transition_authority")
                    != transition_authority_contract(
                        getattr(metadata, "base_model_config").transition_authority
                    )
                    or not isinstance(structured.get("training_partition"), Mapping)
                    or not isinstance(structured.get("heldout"), Mapping)
                ):
                    errors.append(
                        f"{song_id} structured-head evidence binding differs"
                    )
                if (
                    final_fit_training_examples is None
                    or not isinstance(base_metrics, Mapping)
                ):
                    errors.append(
                        f"{song_id} structured-head gate cannot replay"
                    )
                else:
                    try:
                        recomputed_structured = _structured_head_coverage_gate(
                            final_fit_training_examples,
                            base_metrics,
                            thresholds=ProductionQualityThresholds(
                                **dict(protocol["quality_thresholds"])
                            ),
                            transition_authority=(
                                getattr(
                                    metadata, "base_model_config"
                                ).transition_authority
                            ),
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        errors.append(
                            f"{song_id} structured-head gate cannot replay: {exc}"
                        )
                    else:
                        if recomputed_structured != structured:
                            errors.append(
                                f"{song_id} structured-head gate replay differs"
                            )

    aggregate = payload.get("aggregate_gate")
    expected_aggregate = {
        "passed": True,
        "evaluation_song_ids": list(EVALUATION_SONG_IDS),
        "quality_gate_pass_count": 2,
        "structured_head_gate_pass_count": 2,
        "renderer_exact_gate_pass_count": renderer_exact_pass_count,
        "all_evaluations_used_exact_copied_weight": True,
        "registry_activation_authorized": True,
    }
    if (
        renderer_exact_pass_count != len(EVALUATION_SONG_IDS)
        or aggregate != expected_aggregate
    ):
        errors.append("production promotion aggregate gate differs")
    if payload.get("authority") != _promotion_authority_contract():
        errors.append("production promotion authority or claims differ")
    if isinstance(plan, Mapping) and payload.get("authority") != plan.get(
        "authority"
    ):
        errors.append("production promotion authority differs from plan")
    return tuple(errors)


def load_dynamic_crop_v3_production_promotion_report(
    path: Path,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = _read_json(resolved, "production promotion report")
    checkpoint = resolved.parent
    metadata = StageEditDynamicCropCheckpointMetadataV3.from_dict(
        _read_json(
            checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
            "promoted checkpoint metadata",
        )
    )
    weight_sha = _file_sha256(
        checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
        "promoted checkpoint weights",
    )
    errors = production_promotion_report_errors(
        payload,
        metadata=metadata,
        actual_weight_sha256=weight_sha,
        checkpoint_directory=checkpoint,
    )
    if errors:
        raise DynamicCropV3ProductionPromotionError("; ".join(errors))
    return payload


def _renderer_exact_pass(evaluation: Mapping[str, Any]) -> bool:
    dynamic = evaluation.get("dynamic_crop_v3_metrics")
    trajectory = (
        dynamic.get("trajectory") if isinstance(dynamic, Mapping) else None
    )
    renderer = (
        trajectory.get("renderer_decode")
        if isinstance(trajectory, Mapping)
        else None
    )
    attempts = renderer.get("attempt_count") if isinstance(renderer, Mapping) else None
    eligible = renderer.get("eligible_count") if isinstance(renderer, Mapping) else None
    eligible_rate = renderer.get("eligible_rate") if isinstance(renderer, Mapping) else None
    projection_count = (
        renderer.get("minimum_safe_zoom_projection_count")
        if isinstance(renderer, Mapping)
        else None
    )
    source_slot_mismatch_count = (
        trajectory.get("target_source_slot_mismatch_count")
        if isinstance(trajectory, Mapping)
        else None
    )
    improvement = (
        trajectory.get("aggregate_normalized_improvement_over_static_prior")
        if isinstance(trajectory, Mapping)
        else None
    )
    return bool(
        isinstance(attempts, int)
        and not isinstance(attempts, bool)
        and attempts > 0
        and isinstance(eligible, int)
        and not isinstance(eligible, bool)
        and eligible == attempts
        and isinstance(eligible_rate, (int, float))
        and not isinstance(eligible_rate, bool)
        and math.isfinite(float(eligible_rate))
        and float(eligible_rate) == 1.0
        and isinstance(projection_count, int)
        and not isinstance(projection_count, bool)
        and projection_count == 0
        and isinstance(source_slot_mismatch_count, int)
        and not isinstance(source_slot_mismatch_count, bool)
        and source_slot_mismatch_count == 0
        and isinstance(improvement, (int, float))
        and not isinstance(improvement, bool)
        and math.isfinite(float(improvement))
        and float(improvement) > 0.0
    )


def _expected_registry_binding(
    *,
    plan: Mapping[str, Any],
    checkpoint_id: str,
    checkpoint_weight_sha256: str,
) -> dict[str, Any]:
    return {
        "path": str(Path(str(plan["production_registry_path"])).resolve()),
        "checkpoint_directory": str(
            Path(str(plan["destination_checkpoint_directory"])).resolve()
        ),
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_sha256": checkpoint_weight_sha256,
        "dynamic_crop_version": 3,
        "continuity_authority": HANDCRAFTED_CONTINUITY_AUTHORITY,
        "learned_source_pair_authority_present": False,
        "professional_montage_profile": deepcopy(
            plan["professional_montage_profile"]
        ),
    }


def _promotion_receipt_unsigned(
    *,
    preflight: DynamicCropV3ProductionPromotionPreflight,
    report_path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = preflight.destination_checkpoint_directory
    plan = _strict_plan(preflight.plan_path)
    return {
        "schema_version": PROMOTION_RECEIPT_SCHEMA_VERSION,
        "analysis_basis": PROMOTION_RECEIPT_ANALYSIS_BASIS,
        "contract": PROMOTION_RECEIPT_CONTRACT,
        "plan": _pointer(preflight.plan_path, "promotion plan"),
        "plan_content_sha256": preflight.plan_content_sha256,
        "promotion_report": _pointer(report_path, "promotion report"),
        "checkpoint_metadata": _pointer(
            checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
            "promoted checkpoint metadata",
        ),
        "checkpoint_weights": _pointer(
            checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
            "promoted checkpoint weights",
        ),
        "production_registry_binding": _expected_registry_binding(
            plan=plan,
            checkpoint_id=preflight.checkpoint_id,
            checkpoint_weight_sha256=preflight.checkpoint_weight_sha256,
        ),
        "checkpoint_id": preflight.checkpoint_id,
        "checkpoint_weight_sha256": preflight.checkpoint_weight_sha256,
        "report_sha256": report.get("report_sha256"),
        "continuity_authority": HANDCRAFTED_CONTINUITY_AUTHORITY,
        "learned_source_pair_authority_present": False,
        "blind_holdout_quality_claim_authorized": False,
        "human_parity_claim_authorized": False,
        "artifact_content_sha256": None,
    }


def _record_promotion_receipt(
    *,
    preflight: DynamicCropV3ProductionPromotionPreflight,
    report_path: Path,
    report: Mapping[str, Any],
) -> tuple[Path, bool]:
    """Durably stage the exact activation receipt before registry commit."""

    unsigned = _promotion_receipt_unsigned(
        preflight=preflight,
        report_path=report_path,
        report=report,
    )
    digest = _canonical_sha256(unsigned)
    expected_payload = {
        **dict(unsigned),
        "artifact_content_sha256": digest,
    }
    expected_encoded = json.dumps(
        expected_payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    expected_file_sha256 = hashlib.sha256(expected_encoded).hexdigest()
    expected = (
        preflight.evidence_directory
        / f"{PROMOTION_RECEIPT_PREFIX}{digest}.json"
    )
    existed = expected.exists() or expected.is_symlink()
    receipt = _write_content_addressed(
        output_directory=preflight.evidence_directory,
        prefix=PROMOTION_RECEIPT_PREFIX,
        unsigned=unsigned,
        hash_field="artifact_content_sha256",
    )
    created = not existed
    created_identity = (
        _owned_file_identity(receipt, "promotion receipt")
        if created
        else None
    )
    if created and (
        created_identity is None
        or created_identity[3] != expected_file_sha256
    ):
        raise DynamicCropV3ProductionPromotionError(
            "promotion receipt identity changed during installation; "
            "unsafe rollback refused"
        )
    try:
        _load_dynamic_crop_v3_production_promotion_receipt(
            receipt, require_registry=False
        )
    except BaseException as failure:
        if created:
            try:
                _require_owned_identity(
                    receipt,
                    created_identity,
                    "promotion receipt",
                    directory=False,
                )
                receipt.unlink()
            except BaseException as rollback_error:
                raise DynamicCropV3ProductionPromotionError(
                    "promotion receipt validation failed and unsafe rollback "
                    f"was refused: {rollback_error}"
                ) from failure
        raise
    return receipt, created


_RECEIPT_KEYS = {
    "schema_version",
    "analysis_basis",
    "contract",
    "plan",
    "plan_content_sha256",
    "promotion_report",
    "checkpoint_metadata",
    "checkpoint_weights",
    "production_registry_binding",
    "checkpoint_id",
    "checkpoint_weight_sha256",
    "report_sha256",
    "continuity_authority",
    "learned_source_pair_authority_present",
    "blind_holdout_quality_claim_authorized",
    "human_parity_claim_authorized",
    "artifact_content_sha256",
}


def _load_dynamic_crop_v3_production_promotion_receipt(
    path: Path,
    *,
    require_registry: bool,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = _read_json(resolved, "production promotion receipt")
    if set(payload) != _RECEIPT_KEYS:
        raise DynamicCropV3ProductionPromotionError(
            "production promotion receipt keys are not exact"
        )
    digest = payload.get("artifact_content_sha256")
    unsigned = deepcopy(payload)
    unsigned["artifact_content_sha256"] = None
    if (
        not _is_sha256(digest)
        or _canonical_sha256(unsigned) != digest
        or resolved.name != f"{PROMOTION_RECEIPT_PREFIX}{digest}.json"
        or payload.get("schema_version") != PROMOTION_RECEIPT_SCHEMA_VERSION
        or payload.get("analysis_basis") != PROMOTION_RECEIPT_ANALYSIS_BASIS
        or payload.get("contract") != PROMOTION_RECEIPT_CONTRACT
        or payload.get("continuity_authority")
        != HANDCRAFTED_CONTINUITY_AUTHORITY
        or payload.get("learned_source_pair_authority_present") is not False
        or payload.get("blind_holdout_quality_claim_authorized") is not False
        or payload.get("human_parity_claim_authorized") is not False
    ):
        raise DynamicCropV3ProductionPromotionError(
            "production promotion receipt contract differs"
        )
    plan_path = _verify_pointer(payload.get("plan"), "promotion plan")
    plan = _strict_plan(plan_path)
    if payload.get("plan_content_sha256") != plan.get("plan_content_sha256"):
        raise DynamicCropV3ProductionPromotionError(
            "production promotion receipt plan identity differs"
        )
    report_path = _verify_pointer(
        payload.get("promotion_report"), "promotion report"
    )
    metadata_path = _verify_pointer(
        payload.get("checkpoint_metadata"), "promoted checkpoint metadata"
    )
    weights_path = _verify_pointer(
        payload.get("checkpoint_weights"), "promoted checkpoint weights"
    )
    checkpoint = Path(str(plan["destination_checkpoint_directory"])).resolve()
    registry_path = Path(str(plan["production_registry_path"])).resolve()
    if (
        resolved.parent != Path(str(plan["evidence_directory"])).resolve()
        or report_path != checkpoint / PRODUCTION_TRAINING_REPORT_FILENAME
        or metadata_path
        != checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
        or weights_path != checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
    ):
        raise DynamicCropV3ProductionPromotionError(
            "production promotion receipt paths differ from plan"
        )
    report = load_dynamic_crop_v3_production_promotion_report(report_path)
    metadata = StageEditDynamicCropCheckpointMetadataV3.from_dict(
        _read_json(metadata_path, "promoted checkpoint metadata")
    )
    weight_sha = _file_sha256(weights_path, "promoted checkpoint weights")
    report_plan_pointer = report.get("promotion_plan")
    expected_plan_pointer = payload.get("plan")
    expected_binding = _expected_registry_binding(
        plan=plan,
        checkpoint_id=metadata.checkpoint_id,
        checkpoint_weight_sha256=weight_sha,
    )
    if (
        report_plan_pointer != expected_plan_pointer
        or report.get("promotion_plan_content_sha256")
        != payload.get("plan_content_sha256")
        or payload.get("production_registry_binding") != expected_binding
        or payload.get("checkpoint_id") != metadata.checkpoint_id
        or payload.get("checkpoint_weight_sha256") != metadata.weight_sha256
        or payload.get("checkpoint_weight_sha256") != weight_sha
        or payload.get("report_sha256") != report.get("report_sha256")
    ):
        raise DynamicCropV3ProductionPromotionError(
            "production promotion receipt identity differs"
        )
    if require_registry:
        bundle = load_registered_stage_edit_production_bundle(registry_path)
        if (
            bundle.registry_path != registry_path
            or bundle.checkpoint_directory != checkpoint
            or bundle.checkpoint_id != metadata.checkpoint_id
            or bundle.weights_sha256 != weight_sha
            or bundle.dynamic_crop_version != 3
            or bundle.continuity_authority
            != HANDCRAFTED_CONTINUITY_AUTHORITY
            or bundle.source_pair_scorer_path is not None
            or bundle.source_pair_production_report_path is not None
            or bundle.professional_montage_grammar_profile_identity
            != plan["professional_montage_profile"]
        ):
            raise DynamicCropV3ProductionPromotionError(
                "registered production bundle differs from promotion receipt"
            )
    return payload


def load_dynamic_crop_v3_production_promotion_receipt(
    path: Path,
) -> dict[str, Any]:
    return _load_dynamic_crop_v3_production_promotion_receipt(
        path, require_registry=True
    )


def _preflight_from_existing_plan(
    plan_path: Path,
    plan: Mapping[str, Any],
) -> DynamicCropV3ProductionPromotionPreflight:
    lineage = _mapping(plan.get("lineage"), "promotion lineage")
    source_checkpoint = _mapping(
        lineage.get("source_checkpoint"), "source checkpoint"
    )
    metadata_path = _verify_pointer(
        source_checkpoint.get("metadata"), "source checkpoint metadata"
    )
    metadata = StageEditDynamicCropCheckpointMetadataV3.from_dict(
        _read_json(metadata_path, "source checkpoint metadata")
    )
    records = plan.get("evaluation_evidence")
    if not isinstance(records, list) or len(records) != 2:
        raise DynamicCropV3ProductionPromotionError(
            "promotion plan evidence is invalid"
        )
    song_evidence = {
        song_id: _validate_song_evidence_record(
            record, expected_song_id=song_id
        )
        for song_id, record in zip(EVALUATION_SONG_IDS, records)
    }
    protocol = _mapping(plan.get("evaluation_protocol"), "evaluation protocol")
    free_bytes = (
        shutil.disk_usage(PRODUCTION_VOLUME_ROOT).free
        if PRODUCTION_VOLUME_ROOT.is_dir()
        and os.path.ismount(PRODUCTION_VOLUME_ROOT)
        else 0
    )
    return DynamicCropV3ProductionPromotionPreflight(
        plan_path=Path(plan_path).resolve(),
        plan_content_sha256=str(plan["plan_content_sha256"]),
        source_checkpoint_directory=metadata_path.parent,
        destination_checkpoint_directory=Path(
            str(plan["destination_checkpoint_directory"])
        ).resolve(),
        evidence_directory=Path(str(plan["evidence_directory"])).resolve(),
        production_registry_path=Path(
            str(plan["production_registry_path"])
        ).resolve(),
        professional_montage_profile_path=Path(
            str(plan["professional_montage_profile"]["path"])
        ).resolve(),
        checkpoint_id=metadata.checkpoint_id,
        checkpoint_weight_sha256=str(metadata.weight_sha256),
        frozen_cut_probability_threshold=float(
            protocol["cut_probability_threshold"]
        ),
        quality_thresholds=dict(protocol["quality_thresholds"]),
        song_evidence=song_evidence,
        free_bytes=free_bytes,
    )


def _expected_receipt_path(
    *,
    preflight: DynamicCropV3ProductionPromotionPreflight,
    report_path: Path,
    report: Mapping[str, Any],
) -> Path:
    unsigned = _promotion_receipt_unsigned(
        preflight=preflight,
        report_path=report_path,
        report=report,
    )
    return (
        preflight.evidence_directory
        / f"{PROMOTION_RECEIPT_PREFIX}{_canonical_sha256(unsigned)}.json"
    )


def _resume_or_return_existing_promotion(
    plan_path: Path,
) -> DynamicCropV3ProductionPromotionResult | None:
    """Strictly resume our staged commit or return an exact completed run."""

    resolved_plan = Path(plan_path).expanduser().resolve()
    plan = _strict_plan(resolved_plan)
    destination = Path(
        str(plan.get("destination_checkpoint_directory") or "")
    ).resolve()
    registry = Path(str(plan.get("production_registry_path") or "")).resolve()
    destination_present = destination.exists() or destination.is_symlink()
    registry_present = registry.exists() or registry.is_symlink()
    if not destination_present and not registry_present:
        return None
    preflight = _preflight_from_existing_plan(resolved_plan, plan)
    destination = preflight.destination_checkpoint_directory
    registry = preflight.production_registry_path
    if not destination_present or destination.is_symlink() or not destination.is_dir():
        raise DynamicCropV3ProductionPromotionError(
            "promotion registry/checkpoint state is incomplete or unsafe"
        )
    if registry.is_symlink():
        raise DynamicCropV3ProductionPromotionError(
            "promotion registry state is unsafe"
        )

    report_path = destination / PRODUCTION_TRAINING_REPORT_FILENAME
    report = load_dynamic_crop_v3_production_promotion_report(report_path)
    readiness = stage_edit_readiness_report(destination)
    if not (
        readiness.architecture_available
        and readiness.learned_inference_ready
        and readiness.status == "production_checkpoint_ready"
        and readiness.dynamic_crop_version == 3
        and readiness.checkpoint_path == str(destination)
    ):
        raise DynamicCropV3ProductionPromotionError(
            "existing promotion checkpoint is not ready: "
            f"{readiness.status}: {readiness.reason}"
        )
    receipt_path = _expected_receipt_path(
        preflight=preflight,
        report_path=report_path,
        report=report,
    )
    receipt_present = receipt_path.exists() or receipt_path.is_symlink()
    if receipt_path.is_symlink():
        raise DynamicCropV3ProductionPromotionError(
            "promotion receipt state is unsafe"
        )
    receipt_created = False
    receipt_rollback_identity: tuple[Any, ...] | None = None
    if receipt_present:
        _load_dynamic_crop_v3_production_promotion_receipt(
            receipt_path, require_registry=registry_present
        )
    elif registry_present:
        raise DynamicCropV3ProductionPromotionError(
            "active promotion registry exists without its exact receipt"
        )
    else:
        receipt_path, receipt_created = _record_promotion_receipt(
            preflight=preflight,
            report_path=report_path,
            report=report,
        )
        if receipt_created:
            receipt_rollback_identity = _owned_file_identity(
                receipt_path, "promotion receipt"
            )

    if not registry_present:
        try:
            registry = register_stage_edit_production_bundle(
                destination,
                professional_montage_grammar_profile_path=(
                    preflight.professional_montage_profile_path
                ),
                registry_path=preflight.production_registry_path,
                dynamic_crop_version=3,
                continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
                source_pair_scorer_path=None,
                source_pair_production_report_path=None,
            )
        except BaseException as failure:
            if receipt_created:
                try:
                    _require_owned_identity(
                        receipt_path,
                        receipt_rollback_identity,
                        "promotion receipt",
                        directory=False,
                    )
                    receipt_path.unlink()
                except BaseException as rollback_error:
                    raise DynamicCropV3ProductionPromotionError(
                        "promotion resume failed and unsafe receipt rollback "
                        f"was refused: {rollback_error}"
                    ) from failure
            raise
        # Registry installation is the final failure-capable operation.  The
        # exact receipt was already staged and locally replayed above.
    return DynamicCropV3ProductionPromotionResult(
        checkpoint_directory=destination,
        report_path=report_path,
        registry_path=registry,
        receipt_path=receipt_path,
        checkpoint_id=preflight.checkpoint_id,
        checkpoint_weight_sha256=preflight.checkpoint_weight_sha256,
        report=report,
    )


def launch_dynamic_crop_v3_production_promotion(
    plan_path: Path,
) -> DynamicCropV3ProductionPromotionResult:
    """Copy, forward-evaluate, atomically install and finally register."""

    existing = _resume_or_return_existing_promotion(plan_path)
    if existing is not None:
        return existing
    preflight = preflight_dynamic_crop_v3_production_promotion(plan_path)
    plan = _strict_plan(preflight.plan_path)
    destination = preflight.destination_checkpoint_directory
    source = preflight.source_checkpoint_directory
    temporary = destination.with_name(
        f".{destination.name}.promotion-tmp-{os.getpid()}"
    )
    if temporary.exists() or temporary.is_symlink():
        raise DynamicCropV3ProductionPromotionError(
            f"stale promotion temporary checkpoint exists: {temporary}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(mode=0o700)
    temporary_rollback_identity = _owned_directory_identity(
        temporary,
        "promotion temporary checkpoint",
        include_files=False,
    )
    installed = False
    installed_rollback_identity: tuple[Any, ...] | None = None
    receipt_path: Path | None = None
    receipt_created = False
    receipt_rollback_identity: tuple[Any, ...] | None = None
    try:
        source_metadata = source / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
        source_weights = source / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
        copied_metadata = temporary / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
        copied_weights = temporary / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
        metadata_sha = _copy_file_byte_identical(
            source_metadata, copied_metadata, "Dynamic Crop v3 metadata"
        )
        weights_sha = _copy_file_byte_identical(
            source_weights, copied_weights, "Dynamic Crop v3 weights"
        )
        if weights_sha != preflight.checkpoint_weight_sha256:
            raise DynamicCropV3ProductionPromotionError(
                "copied weights differ from final-fit receipt"
            )
        evaluations = _evaluate_exact_copied_checkpoint(
            checkpoint_directory=temporary,
            preflight=preflight,
            plan=plan,
        )
        quality_passes = sum(
            evaluation["quality_gate"].get("passed") is True
            and evaluation["quality_gate"].get("reasons") == []
            for evaluation in evaluations
        )
        structured_passes = sum(
            evaluation["structured_head_coverage_gate"].get("passed") is True
            and evaluation["structured_head_coverage_gate"].get("reasons") == []
            for evaluation in evaluations
        )
        renderer_passes = sum(
            _renderer_exact_pass(evaluation) for evaluation in evaluations
        )
        if (quality_passes, structured_passes, renderer_passes) != (2, 2, 2):
            reasons = {
                evaluation["song_id"]: {
                    "quality": evaluation["quality_gate"].get("reasons"),
                    "structured": evaluation[
                        "structured_head_coverage_gate"
                    ].get("reasons"),
                    "renderer_exact": _renderer_exact_pass(evaluation),
                }
                for evaluation in evaluations
            }
            raise DynamicCropV3ProductionPromotionError(
                "promotion evaluation gates failed: "
                + json.dumps(reasons, ensure_ascii=False, sort_keys=True)
            )
        revalidated_plan = _strict_plan(preflight.plan_path)
        if (
            revalidated_plan.get("plan_content_sha256")
            != preflight.plan_content_sha256
            or revalidated_plan != plan
        ):
            raise DynamicCropV3ProductionPromotionError(
                "promotion plan, lineage, source, or evidence changed during evaluation"
            )
        plan = revalidated_plan
        lineage = deepcopy(plan["lineage"])
        protocol = deepcopy(plan["evaluation_protocol"])
        checkpoint_evidence = {
            "directory": str(destination),
            "checkpoint_format": "dynamic_crop_v3",
            "dynamic_crop_version": 3,
            "metadata_filename": DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
            "weight_filename": DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
            "checkpoint_id": preflight.checkpoint_id,
            "training_steps": int(
                _mapping(
                    lineage["source_checkpoint"], "source checkpoint"
                )["training_steps"]
            ),
            "weight_sha256": weights_sha,
            "model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
            "target_contract": DYNAMIC_CROP_V3_TARGET_CONTRACT,
            "output_contract": DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
            "production_activation_authorized": True,
            "copy_integrity": {
                "source_checkpoint_directory": str(source),
                "source_metadata_sha256": metadata_sha,
                "source_weight_sha256": weights_sha,
                "destination_metadata_sha256": metadata_sha,
                "destination_weight_sha256": weights_sha,
                "metadata_byte_identical": True,
                "weights_byte_identical": True,
                "atomically_installed": True,
            },
        }
        unsigned_report = {
            "schema_version": PRODUCTION_PROMOTION_REPORT_SCHEMA_VERSION,
            "run_kind": PRODUCTION_PROMOTION_RUN_KIND,
            "contract": PRODUCTION_PROMOTION_REPORT_CONTRACT,
            "checkpoint_scope": PRODUCTION_PROMOTION_CHECKPOINT_SCOPE,
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": plan["source_provenance"]["head_commit_oid"],
            "production_ready": True,
            "promotion_plan": _pointer(
                preflight.plan_path, "promotion plan"
            ),
            "promotion_plan_content_sha256": preflight.plan_content_sha256,
            "lineage": lineage,
            "checkpoint": checkpoint_evidence,
            "evaluation_protocol": protocol,
            "evaluations": evaluations,
            "aggregate_gate": {
                "passed": True,
                "evaluation_song_ids": list(EVALUATION_SONG_IDS),
                "quality_gate_pass_count": quality_passes,
                "structured_head_gate_pass_count": structured_passes,
                "renderer_exact_gate_pass_count": renderer_passes,
                "all_evaluations_used_exact_copied_weight": True,
                "registry_activation_authorized": True,
            },
            "authority": _promotion_authority_contract(),
        }
        report = {
            **unsigned_report,
            "report_sha256": _canonical_sha256(unsigned_report),
        }
        report_path_in_temporary = _write_report_exclusive(
            temporary / PRODUCTION_TRAINING_REPORT_FILENAME, report
        )
        metadata = StageEditDynamicCropCheckpointMetadataV3.from_dict(
            _read_json(copied_metadata, "copied checkpoint metadata")
        )
        report_errors = production_promotion_report_errors(
            report,
            metadata=metadata,
            actual_weight_sha256=weights_sha,
            checkpoint_directory=destination,
            checkpoint_files_directory=temporary,
        )
        if report_errors:
            raise DynamicCropV3ProductionPromotionError(
                "generated promotion report is invalid: "
                + "; ".join(report_errors)
            )
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite promotion checkpoint: {destination}"
            )
        staged_checkpoint_identity = _owned_directory_identity(
            temporary,
            "staged promotion checkpoint",
            include_files=True,
        )
        temporary.rename(destination)
        installed = True
        installed_rollback_identity = staged_checkpoint_identity
        installed_identity = _owned_directory_identity(
            destination,
            "installed promotion checkpoint",
            include_files=True,
        )
        if installed_identity != staged_checkpoint_identity:
            raise DynamicCropV3ProductionPromotionError(
                "promotion checkpoint identity changed during installation"
            )
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        report_path = destination / report_path_in_temporary.name
        readiness = stage_edit_readiness_report(destination)
        if not (
            readiness.architecture_available
            and readiness.learned_inference_ready
            and readiness.status == "production_checkpoint_ready"
            and readiness.dynamic_crop_version == 3
            and readiness.checkpoint_path == str(destination)
        ):
            raise DynamicCropV3ProductionPromotionError(
                "shared readiness rejected promotion report: "
                f"{readiness.status}: {readiness.reason}"
            )
        receipt_path, receipt_created = _record_promotion_receipt(
            preflight=preflight,
            report_path=report_path,
            report=report,
        )
        if receipt_created:
            receipt_rollback_identity = _owned_file_identity(
                receipt_path, "promotion receipt"
            )
        # Registry activation is intentionally the last failure-capable side
        # effect.  Its public writer rolls back a newly installed record if
        # its own post-write replay fails.
        registry = register_stage_edit_production_bundle(
            destination,
            professional_montage_grammar_profile_path=(
                preflight.professional_montage_profile_path
            ),
            registry_path=preflight.production_registry_path,
            dynamic_crop_version=3,
            continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
            source_pair_scorer_path=None,
            source_pair_production_report_path=None,
        )
        return DynamicCropV3ProductionPromotionResult(
            checkpoint_directory=destination,
            report_path=report_path,
            registry_path=registry,
            receipt_path=receipt_path,
            checkpoint_id=preflight.checkpoint_id,
            checkpoint_weight_sha256=weights_sha,
            report=report,
        )
    except BaseException as failure:
        # Remove only artifacts exclusively created by this invocation.  The
        # registry is never removed here: its writer owns exact-byte rollback,
        # and a pre-existing registry must never be touched.
        try:
            if receipt_created and receipt_path is not None:
                _require_owned_identity(
                    receipt_path,
                    receipt_rollback_identity,
                    "promotion receipt",
                    directory=False,
                )
            if installed:
                _require_owned_identity(
                    destination,
                    installed_rollback_identity,
                    "installed promotion checkpoint",
                    directory=True,
                    include_files=True,
                )
            if temporary.exists() or temporary.is_symlink():
                _require_owned_identity(
                    temporary,
                    temporary_rollback_identity,
                    "promotion temporary checkpoint",
                    directory=True,
                )
        except BaseException as rollback_identity_error:
            raise DynamicCropV3ProductionPromotionError(
                "promotion failed and unsafe rollback was refused: "
                f"{rollback_identity_error}"
            ) from failure
        try:
            if receipt_created and receipt_path is not None:
                receipt_path.unlink()
            if installed:
                shutil.rmtree(destination)
            if temporary.exists() or temporary.is_symlink():
                shutil.rmtree(temporary)
        except BaseException as rollback_error:
            raise DynamicCropV3ProductionPromotionError(
                f"promotion failed and owned-artifact rollback failed: {rollback_error}"
            ) from failure
        raise


def _keyed_paths(values: Sequence[str], option: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for raw in values:
        song_id, separator, path_text = raw.partition("=")
        song_id = song_id.strip()
        path_text = path_text.strip()
        if (
            not separator
            or song_id not in EVALUATION_SONG_IDS
            or song_id in output
            or not path_text
        ):
            raise DynamicCropV3ProductionPromotionError(
                f"{option} must name XOXZ and ATTITUDE exactly once as SONG=/path"
            )
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            raise DynamicCropV3ProductionPromotionError(
                f"{option} paths must be absolute"
            )
        output[song_id] = path.resolve()
    if set(output) != set(EVALUATION_SONG_IDS):
        raise DynamicCropV3ProductionPromotionError(
            f"{option} must include both XOXZ and ATTITUDE"
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote the exact Dynamic Crop v3 all-reference final-fit weight "
            "only after frozen non-blind XOXZ and ATTITUDE evaluation"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--final-fit-receipt", required=True, type=Path)
    create.add_argument(
        "--materialization-manifest", action="append", required=True
    )
    create.add_argument("--feature-manifest", action="append", required=True)
    create.add_argument(
        "--dynamic-crop-manifest", action="append", required=True
    )
    create.add_argument("--checkpoint-directory", required=True, type=Path)
    create.add_argument("--evidence-directory", required=True, type=Path)
    create.add_argument("--output-directory", required=True, type=Path)
    create.add_argument("--production-registry", type=Path)
    for name in ("preflight", "launch"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True, type=Path)
    verify = commands.add_parser("verify-receipt")
    verify.add_argument("--receipt", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create":
        path = create_dynamic_crop_v3_production_promotion_plan(
            final_fit_receipt_path=args.final_fit_receipt,
            materialization_manifests=_keyed_paths(
                args.materialization_manifest, "--materialization-manifest"
            ),
            feature_manifests=_keyed_paths(
                args.feature_manifest, "--feature-manifest"
            ),
            dynamic_crop_manifests=_keyed_paths(
                args.dynamic_crop_manifest, "--dynamic-crop-manifest"
            ),
            checkpoint_directory=args.checkpoint_directory,
            evidence_directory=args.evidence_directory,
            output_directory=args.output_directory,
            production_registry_path=args.production_registry,
        )
        print(json.dumps({"promotion_plan": str(path)}, indent=2))
        return 0
    if args.command == "preflight":
        preflight = preflight_dynamic_crop_v3_production_promotion(args.plan)
        print(json.dumps(preflight.to_dict(), indent=2))
        return 0
    if args.command == "verify-receipt":
        receipt = load_dynamic_crop_v3_production_promotion_receipt(args.receipt)
        print(json.dumps(receipt, indent=2))
        return 0
    result = launch_dynamic_crop_v3_production_promotion(args.plan)
    print(
        json.dumps(
            {
                "checkpoint_directory": str(result.checkpoint_directory),
                "promotion_report": str(result.report_path),
                "production_registry": str(result.registry_path),
                "promotion_receipt": str(result.receipt_path),
                "checkpoint_id": result.checkpoint_id,
                "checkpoint_weight_sha256": result.checkpoint_weight_sha256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTITUDE",
    "DynamicCropV3ProductionPromotionError",
    "DynamicCropV3ProductionPromotionPreflight",
    "DynamicCropV3ProductionPromotionResult",
    "EVALUATION_SONG_IDS",
    "PRODUCTION_PROMOTION_CHECKPOINT_SCOPE",
    "PRODUCTION_PROMOTION_REPORT_CONTRACT",
    "PRODUCTION_PROMOTION_REPORT_SCHEMA_VERSION",
    "PRODUCTION_PROMOTION_RUN_KIND",
    "PROMOTION_PLAN_CONTRACT",
    "PROMOTION_RECEIPT_CONTRACT",
    "XOXZ",
    "build_parser",
    "create_dynamic_crop_v3_production_promotion_plan",
    "launch_dynamic_crop_v3_production_promotion",
    "load_dynamic_crop_v3_production_promotion_receipt",
    "load_dynamic_crop_v3_production_promotion_report",
    "main",
    "preflight_dynamic_crop_v3_production_promotion",
    "production_promotion_report_errors",
]
