"""Leakage-safe Dynamic Crop v3 LOSO freeze and inactive final refit.

The BANG BANG, BLACKHOLE, and REBEL HEART whole-song folds select exactly one
pinned candidate.  Only after all three same-candidate receipts pass can this
module freeze the configuration, schedule, and median pre-heldout cut
threshold.  A subsequent final-fit plan consumes all fifteen references, but
has no registry authority and makes no XOXZ, ATTITUDE, blind-holdout, or human
parity claim.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .professional_montage_grammar import PROFILE_SCHEMA_VERSION
from .professional_montage_policy import (
    ProfessionalMontagePolicyError,
    load_professional_montage_policy,
)
from .stage_edit_dynamic_crop_capacity_probe import (
    SELECTION_SONG_IDS,
)
from .stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
    DYNAMIC_CROP_V3_LOSS_CONTRACT,
    DYNAMIC_CROP_V3_MODEL_CONTRACT,
    DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
    DYNAMIC_CROP_V3_TARGET_CONTRACT,
    StageEditDynamicCropCheckpointMetadataV3,
    StageEditTransformerDynamicCropV3,
    load_stage_edit_dynamic_crop_checkpoint_v3,
)
from .stage_edit_dynamic_crop_v3_capacity_smoke import (
    V3_PROBE_TRAINING_SCHEDULE as PROBE_TRAINING_SCHEDULE,
)
from .stage_edit_dynamic_crop_training import (
    build_stage_edit_dynamic_crop_training_corpus_v2,
)
from .stage_edit_dynamic_crop_training_v3 import (
    build_stage_edit_dynamic_crop_training_corpus_v3,
)
from .stage_edit_dynamic_crop_v3_capacity_fold_launcher import (
    V3_SOURCE_RELATIVE_PATHS as CAPACITY_V3_SOURCE_RELATIVE_PATHS,
    _candidate_defaults_by_id,
    _manifest_maps_from_plan,
    _strict_plan as _strict_capacity_plan,
    _validate_manifest_set,
    load_dynamic_crop_v3_capacity_fold_evidence,
)
from . import stage_edit_dynamic_crop_v3_capacity_fold_launcher as capacity
from .stage_edit_dynamic_crop_v3_final_fit import (
    FINAL_FIT_CONTRACT,
    FINAL_FIT_REPORT_FILENAME,
    DynamicCropV3FinalFitResult,
    load_dynamic_crop_v3_final_fit_report,
    train_dynamic_crop_v3_final_fit,
)
from .stage_edit_production_train_cli import (
    _dynamic_artifact_map,
    _snapshot_map,
)
from .stage_edit_production_training import (
    DYNAMIC_CROP_V3_PRIOR_CANDIDATE_SELECTION_CONTRACT,
    DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS,
)
from .stage_edit_reference_adapter import load_reference_training_corpus
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    PRODUCTION_TRAINING_REPORT_FILENAME,
)


V3_RELEASE_FREEZE_SCHEMA_VERSION = 1
V3_RELEASE_FREEZE_ANALYSIS_BASIS = (
    "ive_dynamic_crop_v3_three_song_loso_release_freeze_v1"
)
V3_RELEASE_FREEZE_PREFIX = "dynamic-crop-v3-release-freeze-"
V3_RELEASE_FREEZE_CONTRACT = (
    "all_three_same_candidate_clean_loso_receipts_before_final_fit_v1"
)
V3_FINAL_FIT_PLAN_SCHEMA_VERSION = 1
V3_FINAL_FIT_PLAN_ANALYSIS_BASIS = (
    "ive_dynamic_crop_v3_inactive_all_15_reference_final_fit_plan_v1"
)
V3_FINAL_FIT_PLAN_PREFIX = "dynamic-crop-v3-final-fit-plan-"
V3_FINAL_FIT_PLAN_CONTRACT = (
    "frozen_loso_decisions_all_15_reference_inactive_refit_v1"
)
V3_FINAL_FIT_RECEIPT_SCHEMA_VERSION = 1
V3_FINAL_FIT_RECEIPT_ANALYSIS_BASIS = (
    "verified_inactive_dynamic_crop_v3_all_reference_final_fit_receipt_v1"
)
V3_FINAL_FIT_RECEIPT_PREFIX = "dynamic-crop-v3-final-fit-receipt-"
V3_FINAL_FIT_RECEIPT_CONTRACT = (
    "checkpoint_report_weights_bound_without_activation_or_blind_claim_v1"
)
SELECTED_CANDIDATE_NAME = "production-balanced"
PRODUCTION_VOLUME_ROOT = Path("/Volumes/CrossCutMedia")
TRAIN_MODULE = (
    "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher"
)
SOURCE_RELATIVE_PATHS = tuple(
    dict.fromkeys(
        (
            *CAPACITY_V3_SOURCE_RELATIVE_PATHS,
            "engine/crosscut/stage_edit_dynamic_crop_v3_final_fit.py",
            "engine/crosscut/stage_edit_dynamic_crop_v3_final_release_launcher.py",
            "engine/crosscut/professional_montage_policy.py",
            "engine/crosscut/professional_montage_grammar.py",
        )
    )
)


class DynamicCropV3FinalReleaseError(ValueError):
    """Release evidence, plan, or output is unsafe."""


@dataclass(frozen=True, slots=True)
class DynamicCropV3FinalFitPreflight:
    plan_path: Path
    plan_content_sha256: str
    release_freeze_path: Path
    release_freeze_file_sha256: str
    candidate_id: str
    checkpoint_directory: Path
    evidence_directory: Path
    source_commit: str
    frozen_cut_probability_threshold: float
    frozen_dynamic_crop_v3_prior_candidate_step: int
    quality_thresholds: Mapping[str, Any]
    reference_ids_by_song: Mapping[str, tuple[str, ...]]
    supervision_paths_by_song: Mapping[str, Mapping[str, Path]]
    snapshot_paths_by_song: Mapping[str, Mapping[str, Path]]
    dynamic_manifest_paths_by_song: Mapping[str, Path]
    free_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_path": str(self.plan_path),
            "plan_content_sha256": self.plan_content_sha256,
            "release_freeze_path": str(self.release_freeze_path),
            "release_freeze_file_sha256": self.release_freeze_file_sha256,
            "candidate_id": self.candidate_id,
            "candidate_name": SELECTED_CANDIDATE_NAME,
            "checkpoint_directory": str(self.checkpoint_directory),
            "evidence_directory": str(self.evidence_directory),
            "source_commit": self.source_commit,
            "frozen_cut_probability_threshold": (
                self.frozen_cut_probability_threshold
            ),
            "frozen_dynamic_crop_v3_prior_candidate_step": (
                self.frozen_dynamic_crop_v3_prior_candidate_step
            ),
            "quality_thresholds": dict(self.quality_thresholds),
            "reference_ids_by_song": {
                song: list(references)
                for song, references in sorted(
                    self.reference_ids_by_song.items()
                )
            },
            "free_bytes": self.free_bytes,
            "dynamic_crop_version": 3,
            "all_15_reference_fit": True,
            "production_activation_authorized": False,
            "production_registry_activation_requested": False,
            "learned_source_pair_authority_present": False,
            "blind_holdout_quality_claim_authorized": False,
        }


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


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicCropV3FinalReleaseError(f"{label} must be an object")
    return dict(value)


def _exact_json_equal(left: object, right: object) -> bool:
    try:
        return _canonical_bytes(left) == _canonical_bytes(right)
    except (TypeError, ValueError):
        return False


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _file_sha256(path: Path, label: str) -> str:
    try:
        return capacity.v2_launcher._stable_file_sha256(
            Path(path).expanduser().resolve(), label
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3FinalReleaseError(str(exc)) from exc


def _pointer(path: Path, label: str) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(resolved),
        "file_sha256": _file_sha256(resolved, label),
    }


def _verify_pointer(value: object, label: str) -> Path:
    try:
        return capacity.v2_launcher._verify_pointer(value, label)
    except (OSError, ValueError) as exc:
        raise DynamicCropV3FinalReleaseError(str(exc)) from exc


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return capacity.v2_launcher._read_json_object(path, label)
    except (OSError, ValueError) as exc:
        raise DynamicCropV3FinalReleaseError(str(exc)) from exc


def _capture_source_provenance(
    repository_root: Path | None = None,
) -> dict[str, Any]:
    try:
        return capacity.v2_launcher._capture_source_provenance(
            repository_root,
            source_relative_paths=SOURCE_RELATIVE_PATHS,
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3FinalReleaseError(str(exc)) from exc


def _validate_source_provenance(value: object) -> dict[str, Any]:
    payload = _mapping(value, "v3 final-release source provenance")
    required = {
        "contract",
        "repository_root",
        "head_commit_oid",
        "repository_status_clean",
        "repository_status_porcelain_sha256",
        "source_files",
    }
    if set(payload) != required:
        raise DynamicCropV3FinalReleaseError(
            "v3 final-release source provenance keys are not exact"
        )
    files = _mapping(payload.get("source_files"), "v3 release source files")
    if set(files) != set(SOURCE_RELATIVE_PATHS):
        raise DynamicCropV3FinalReleaseError(
            "v3 final-release source file set is incomplete"
        )
    if (
        payload.get("contract")
        != "clean_head_and_exact_interpreter_source_blobs_v1"
        or payload.get("repository_status_clean") is not True
        or payload.get("repository_status_porcelain_sha256")
        != hashlib.sha256(b"").hexdigest()
    ):
        raise DynamicCropV3FinalReleaseError(
            "v3 final-release source provenance is not clean"
        )
    observed = _capture_source_provenance(
        Path(str(payload.get("repository_root") or ""))
    )
    if observed != payload:
        raise DynamicCropV3FinalReleaseError(
            "current v3 final-release code differs from frozen source"
        )
    return payload


def _require_same_source_provenance(value: Mapping[str, Any]) -> None:
    if _validate_source_provenance(value) != dict(value):
        raise DynamicCropV3FinalReleaseError(
            "v3 final-release source provenance changed"
        )


def _outside_clean_repository(path: Path, label: str) -> Path:
    try:
        resolved = capacity.v2_launcher._absolute_safe_path(path, label)
    except (OSError, ValueError) as exc:
        raise DynamicCropV3FinalReleaseError(str(exc)) from exc
    repository = Path(str(_capture_source_provenance()["repository_root"]))
    if resolved == repository or repository in resolved.parents:
        raise DynamicCropV3FinalReleaseError(
            f"{label} must stay outside the clean repository"
        )
    return resolved


def _under_production_volume(path: Path, label: str) -> Path:
    try:
        resolved = capacity.v2_launcher._absolute_safe_path(path, label)
    except (OSError, ValueError) as exc:
        raise DynamicCropV3FinalReleaseError(str(exc)) from exc
    root = PRODUCTION_VOLUME_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise DynamicCropV3FinalReleaseError(f"{label} must be under {root}")
    return resolved


def _atomic_content_addressed_write(
    *,
    output_directory: Path,
    prefix: str,
    unsigned: Mapping[str, Any],
    hash_field: str,
) -> Path:
    digest = _canonical_sha256(unsigned)
    payload = {**unsigned, hash_field: digest}
    try:
        return capacity.v2_launcher._atomic_exclusive_write(
            output_directory / f"{prefix}{digest}.json", payload
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3FinalReleaseError(str(exc)) from exc


def _median_three(values: Sequence[float]) -> float:
    if len(values) != 3:
        raise DynamicCropV3FinalReleaseError(
            "exactly three fold thresholds are required"
        )
    ordered = sorted(float(value) for value in values)
    if any(not 0.0 < value < 1.0 for value in ordered):
        raise DynamicCropV3FinalReleaseError(
            "fold cut thresholds must be strictly between zero and one"
        )
    return ordered[1]


def _median_three_prior_candidate_steps(values: Sequence[int]) -> int:
    if (
        len(values) != 3
        or any(
            isinstance(value, bool)
            or value not in DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
            for value in values
        )
    ):
        raise DynamicCropV3FinalReleaseError(
            "exactly three predeclared prior candidate milestones are required"
        )
    return sorted(int(value) for value in values)[1]


def _inactive_authority_contract() -> dict[str, bool]:
    return {
        "production_activation_authorized": False,
        "production_registry_activation_requested": False,
        "learned_source_pair_authority_present": False,
        "blind_holdout_quality_claim_authorized": False,
        "xoxz_dynamic_crop_v3_blind_evidence_present": False,
        "attitude_dynamic_crop_v3_blind_evidence_present": False,
        "human_parity_claim_authorized": False,
        "experimental_unseen_ive_inference_only": True,
    }


def _loso_scope_contract() -> dict[str, Any]:
    return {
        "song_ids": list(SELECTION_SONG_IDS),
        "whole_song_heldout": True,
        "all_three_heldout_gates_passed": True,
        "candidate_predeclared_before_receipt_admission": True,
        "heldout_used_for_candidate_selection": False,
        "heldout_used_for_release_qualification": True,
        "heldout_used_for_within_fold_checkpoint_selection": False,
        "heldout_used_for_within_fold_threshold_selection": False,
        "target_song_training_required": False,
    }


def _final_fit_contract() -> dict[str, Any]:
    return {
        "contract": FINAL_FIT_CONTRACT,
        "song_ids": list(SELECTION_SONG_IDS),
        "reference_count": 15,
        "all_references_used_for_optimizer_updates": True,
        "candidate_schedule_and_thresholds_frozen_before_fit": True,
        "final_fit_data_used_for_reselection": False,
    }


def _all_reference_plan_contract() -> dict[str, Any]:
    return {
        "song_ids": list(SELECTION_SONG_IDS),
        "reference_count": 15,
        "all_references_consumed_by_optimizer": True,
        "model_selection_during_final_fit": False,
        "schedule_selection_during_final_fit": False,
        "threshold_selection_during_final_fit": False,
    }


def _load_complete_same_candidate_loso(
    receipt_paths: Sequence[Path],
) -> dict[str, Any]:
    if len(receipt_paths) != 3:
        raise DynamicCropV3FinalReleaseError(
            "release freeze requires exactly three fold receipts"
        )
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for raw_path in receipt_paths:
        path = Path(raw_path).expanduser().resolve()
        try:
            payload = load_dynamic_crop_v3_capacity_fold_evidence(path)
        except (OSError, ValueError) as exc:
            raise DynamicCropV3FinalReleaseError(
                f"v3 LOSO receipt is invalid: {exc}"
            ) from exc
        loaded.append((path, payload))
    candidate_ids = {payload.get("candidate_id") for _path, payload in loaded}
    candidate_names = {
        payload.get("candidate_name") for _path, payload in loaded
    }
    validation_songs = {
        payload.get("validation_song_id") for _path, payload in loaded
    }
    if len(candidate_ids) != 1 or candidate_names != {SELECTED_CANDIDATE_NAME}:
        raise DynamicCropV3FinalReleaseError(
            "all LOSO receipts must use the same production-balanced candidate"
        )
    if validation_songs != set(SELECTION_SONG_IDS):
        raise DynamicCropV3FinalReleaseError(
            "LOSO receipts do not cover every fixed held-out song exactly once"
        )
    candidate_id = str(next(iter(candidate_ids)))
    candidate = _candidate_defaults_by_id().get(candidate_id)
    if candidate is None or candidate.candidate_name != SELECTED_CANDIDATE_NAME:
        raise DynamicCropV3FinalReleaseError(
            "LOSO candidate is not the exact pinned production-balanced v3 model"
        )
    plan_paths = {
        str(_verify_pointer(payload.get("plan"), "capacity fold plan"))
        for _path, payload in loaded
    }
    if len(plan_paths) != 1:
        raise DynamicCropV3FinalReleaseError(
            "all LOSO receipts must bind the same immutable capacity plan"
        )
    capacity_plan_path = Path(next(iter(plan_paths)))
    try:
        capacity_plan = _strict_capacity_plan(capacity_plan_path)
    except (OSError, ValueError) as exc:
        raise DynamicCropV3FinalReleaseError(str(exc)) from exc
    plan_source = _mapping(
        capacity_plan.get("source_provenance"), "capacity source provenance"
    )
    planned_candidates = capacity._candidates_from_plan(
        capacity_plan.get("candidates")
    )
    if tuple(item.candidate_id for item in planned_candidates) != (
        candidate.candidate_id,
    ):
        raise DynamicCropV3FinalReleaseError(
            "capacity plan must predeclare only production-balanced before LOSO"
        )
    thresholds: list[float] = []
    prior_candidate_steps: list[int] = []
    fold_records: list[dict[str, Any]] = []
    quality_thresholds: Mapping[str, Any] | None = None
    for receipt_path, receipt in sorted(
        loaded, key=lambda item: str(item[1]["validation_song_id"])
    ):
        report_path = _verify_pointer(
            receipt.get("production_training_report"),
            "fold production report",
        )
        selection_path = _verify_pointer(
            receipt.get("selection_freeze"), "fold selection freeze"
        )
        report = _read_json(report_path, "fold production report")
        selection = _read_json(selection_path, "fold selection freeze")
        heldout = _mapping(report.get("held_out_evaluation"), "fold heldout")
        quality = _mapping(heldout.get("quality_gate"), "fold quality gate")
        split = _mapping(report.get("split"), "fold split")
        optimization = _mapping(report.get("optimization"), "fold optimization")
        prior = _mapping(
            optimization.get("dynamic_crop_v3_prior_pretraining"),
            "fold Dynamic Crop v3 prior",
        )
        prior_selection = _mapping(
            prior.get("candidate_selection"),
            "fold Dynamic Crop v3 prior candidate selection",
        )
        selected_prior_step = prior_selection.get("selected_candidate_step")
        selected_prior_digest = prior_selection.get(
            "selected_candidate_value_bias_sha256"
        )
        threshold_freeze = _mapping(
            report.get("cut_threshold_freeze"), "fold cut threshold freeze"
        )
        selected_threshold = threshold_freeze.get(
            "selected_probability_threshold"
        )
        if (
            isinstance(selected_threshold, bool)
            or not isinstance(selected_threshold, (int, float))
        ):
            raise DynamicCropV3FinalReleaseError(
                "fold cut threshold is not numeric"
            )
        threshold = float(selected_threshold)
        expected_training = sorted(
            set(SELECTION_SONG_IDS) - {receipt["validation_song_id"]}
        )
        if (
            heldout.get("passed") is not True
            or quality.get("passed") is not True
            or split.get("training_song_ids") != expected_training
            or split.get("held_out_song_ids")
            != [receipt["validation_song_id"]]
            or split.get("held_out_used_for_model_selection") is not False
            or optimization.get("unseen_song_used_for_model_selection") is not False
            or heldout.get(
                "heldout_dynamic_crop_metrics_used_for_checkpoint_selection"
            )
            is not False
            or selection.get("heldout_metrics_present") is not False
            or selection.get("heldout_used_for_model_or_threshold_selection")
            is not False
            or threshold_freeze.get("selected_before_heldout_forward") is not True
            or selection.get("cut_threshold_freeze") != threshold_freeze
            or prior_selection.get("contract")
            != DYNAMIC_CROP_V3_PRIOR_CANDIDATE_SELECTION_CONTRACT
            or prior_selection.get("predeclared_candidate_steps")
            != list(DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS)
            or prior_selection.get("retention_authorized") is not True
            or isinstance(selected_prior_step, bool)
            or selected_prior_step not in DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
            or not _is_sha256(selected_prior_digest)
            or selection.get("dynamic_crop_v3_prior_selected_candidate_step")
            != selected_prior_step
            or selection.get(
                "dynamic_crop_v3_prior_selected_candidate_value_bias_sha256"
            )
            != selected_prior_digest
        ):
            raise DynamicCropV3FinalReleaseError(
                "fold held-out data influenced model or threshold selection"
            )
        current_quality_thresholds = quality.get("thresholds")
        if not isinstance(current_quality_thresholds, Mapping):
            raise DynamicCropV3FinalReleaseError(
                "fold quality thresholds are missing"
            )
        if quality_thresholds is None:
            quality_thresholds = dict(current_quality_thresholds)
        elif not _exact_json_equal(
            dict(current_quality_thresholds), dict(quality_thresholds)
        ):
            raise DynamicCropV3FinalReleaseError(
                "fold quality thresholds differ"
            )
        thresholds.append(threshold)
        prior_candidate_steps.append(int(selected_prior_step))
        fold_records.append(
            {
                "validation_song_id": receipt["validation_song_id"],
                "training_song_ids": list(receipt["training_song_ids"]),
                "receipt": _pointer(receipt_path, "v3 LOSO receipt"),
                "production_training_report": dict(
                    receipt["production_training_report"]
                ),
                "selection_freeze": dict(receipt["selection_freeze"]),
                "checkpoint_metadata": dict(receipt["checkpoint_metadata"]),
                "checkpoint_weights": dict(receipt["checkpoint_weights"]),
                "checkpoint_id": receipt["checkpoint_id"],
                "checkpoint_weight_sha256": receipt[
                    "checkpoint_weight_sha256"
                ],
                "cut_probability_threshold": threshold,
                "selected_finetune_optimizer_step": selection.get(
                    "selected_finetune_optimizer_step"
                ),
                "dynamic_crop_v3_prior_selected_candidate_step": (
                    selected_prior_step
                ),
                "dynamic_crop_v3_prior_selected_candidate_value_bias_sha256": (
                    selected_prior_digest
                ),
                "heldout_gate_passed": True,
                "heldout_used_for_selection": False,
            }
        )
    assert quality_thresholds is not None
    return {
        "candidate": candidate,
        "capacity_plan_path": capacity_plan_path,
        "capacity_plan": capacity_plan,
        "capacity_source_commit": plan_source.get("head_commit_oid"),
        "folds": fold_records,
        "fold_cut_probability_thresholds": thresholds,
        "frozen_cut_probability_threshold": _median_three(thresholds),
        "fold_dynamic_crop_v3_prior_selected_candidate_steps": (
            prior_candidate_steps
        ),
        "frozen_dynamic_crop_v3_prior_candidate_step": (
            _median_three_prior_candidate_steps(prior_candidate_steps)
        ),
        "quality_thresholds": dict(quality_thresholds),
    }


def create_dynamic_crop_v3_release_freeze(
    *,
    fold_receipt_paths: Sequence[Path],
    professional_montage_grammar_profile_path: Path,
    output_directory: Path,
) -> Path:
    """Freeze one candidate only after complete same-candidate LOSO passes."""

    source = _capture_source_provenance()
    loso = _load_complete_same_candidate_loso(fold_receipt_paths)
    if loso["capacity_source_commit"] != source["head_commit_oid"]:
        raise DynamicCropV3FinalReleaseError(
            "LOSO receipts and final-release code must share one clean commit"
        )
    profile_path = Path(
        professional_montage_grammar_profile_path
    ).expanduser().resolve()
    try:
        policy = load_professional_montage_policy(profile_path)
    except ProfessionalMontagePolicyError as exc:
        raise DynamicCropV3FinalReleaseError(
            f"professional montage profile is invalid: {exc}"
        ) from exc
    if policy.identity.schema_version != PROFILE_SCHEMA_VERSION:
        raise DynamicCropV3FinalReleaseError(
            "professional montage profile schema differs"
        )
    output = _outside_clean_repository(output_directory, "freeze output directory")
    candidate = loso["candidate"]
    _require_same_source_provenance(source)
    unsigned = {
        "schema_version": V3_RELEASE_FREEZE_SCHEMA_VERSION,
        "analysis_basis": V3_RELEASE_FREEZE_ANALYSIS_BASIS,
        "contract": V3_RELEASE_FREEZE_CONTRACT,
        "source_provenance": source,
        "candidate_name": candidate.candidate_name,
        "candidate_id": candidate.candidate_id,
        "candidate_contract": candidate.contract_dict(),
        "training_schedule": dict(PROBE_TRAINING_SCHEDULE),
        "final_fit_finetune_optimizer_steps": PROBE_TRAINING_SCHEDULE[
            "finetune_steps_planned"
        ],
        "quality_thresholds": loso["quality_thresholds"],
        "fold_cut_probability_thresholds": list(
            loso["fold_cut_probability_thresholds"]
        ),
        "frozen_cut_probability_threshold": loso[
            "frozen_cut_probability_threshold"
        ],
        "cut_threshold_aggregation": "median_of_three_preheldout_thresholds_v1",
        "fold_dynamic_crop_v3_prior_selected_candidate_steps": list(
            loso["fold_dynamic_crop_v3_prior_selected_candidate_steps"]
        ),
        "frozen_dynamic_crop_v3_prior_candidate_step": loso[
            "frozen_dynamic_crop_v3_prior_candidate_step"
        ],
        "dynamic_crop_v3_prior_candidate_step_aggregation": (
            "median_of_three_preheldout_selected_predeclared_milestones_v1"
        ),
        "capacity_plan": _pointer(
            loso["capacity_plan_path"], "v3 capacity plan"
        ),
        "folds": loso["folds"],
        "professional_montage_profile": _pointer(
            profile_path, "professional montage profile"
        ),
        "professional_montage_profile_identity": policy.identity.to_dict(),
        "loso_scope": _loso_scope_contract(),
        "final_fit_contract": _final_fit_contract(),
        "authority": _inactive_authority_contract(),
        "freeze_content_sha256": None,
    }
    return _atomic_content_addressed_write(
        output_directory=output,
        prefix=V3_RELEASE_FREEZE_PREFIX,
        unsigned=unsigned,
        hash_field="freeze_content_sha256",
    )


_FREEZE_KEYS = {
    "schema_version",
    "analysis_basis",
    "contract",
    "source_provenance",
    "candidate_name",
    "candidate_id",
    "candidate_contract",
    "training_schedule",
    "final_fit_finetune_optimizer_steps",
    "quality_thresholds",
    "fold_cut_probability_thresholds",
    "frozen_cut_probability_threshold",
    "cut_threshold_aggregation",
    "fold_dynamic_crop_v3_prior_selected_candidate_steps",
    "frozen_dynamic_crop_v3_prior_candidate_step",
    "dynamic_crop_v3_prior_candidate_step_aggregation",
    "capacity_plan",
    "folds",
    "professional_montage_profile",
    "professional_montage_profile_identity",
    "loso_scope",
    "final_fit_contract",
    "authority",
    "freeze_content_sha256",
}


def _strict_release_freeze(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = _read_json(resolved, "v3 release freeze")
    if set(payload) != _FREEZE_KEYS:
        raise DynamicCropV3FinalReleaseError(
            "v3 release freeze keys are not exact"
        )
    digest = payload.get("freeze_content_sha256")
    unsigned = deepcopy(payload)
    unsigned["freeze_content_sha256"] = None
    if (
        not _is_sha256(digest)
        or _canonical_sha256(unsigned) != digest
        or resolved.name != f"{V3_RELEASE_FREEZE_PREFIX}{digest}.json"
    ):
        raise DynamicCropV3FinalReleaseError(
            "v3 release freeze is not content-addressed"
        )
    if (
        payload.get("schema_version") != V3_RELEASE_FREEZE_SCHEMA_VERSION
        or payload.get("analysis_basis") != V3_RELEASE_FREEZE_ANALYSIS_BASIS
        or payload.get("contract") != V3_RELEASE_FREEZE_CONTRACT
        or payload.get("candidate_name") != SELECTED_CANDIDATE_NAME
        or not _exact_json_equal(
            payload.get("training_schedule"), dict(PROBE_TRAINING_SCHEDULE)
        )
        or payload.get("final_fit_finetune_optimizer_steps")
        != PROBE_TRAINING_SCHEDULE["finetune_steps_planned"]
        or payload.get("cut_threshold_aggregation")
        != "median_of_three_preheldout_thresholds_v1"
        or payload.get("dynamic_crop_v3_prior_candidate_step_aggregation")
        != "median_of_three_preheldout_selected_predeclared_milestones_v1"
        or payload.get("frozen_dynamic_crop_v3_prior_candidate_step")
        not in DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
        or not _exact_json_equal(
            payload.get("loso_scope"), _loso_scope_contract()
        )
        or not _exact_json_equal(
            payload.get("final_fit_contract"), _final_fit_contract()
        )
        or not _exact_json_equal(
            payload.get("authority"), _inactive_authority_contract()
        )
    ):
        raise DynamicCropV3FinalReleaseError(
            "v3 release freeze contract differs"
        )
    source = _validate_source_provenance(payload.get("source_provenance"))
    candidate = _candidate_defaults_by_id().get(
        str(payload.get("candidate_id") or "")
    )
    if (
        candidate is None
        or candidate.candidate_name != SELECTED_CANDIDATE_NAME
        or not _exact_json_equal(
            payload.get("candidate_contract"), candidate.contract_dict()
        )
    ):
        raise DynamicCropV3FinalReleaseError(
            "frozen production-balanced candidate differs"
        )
    fold_paths = [
        _verify_pointer(
            _mapping(raw, "frozen LOSO fold").get("receipt"),
            "v3 LOSO receipt",
        )
        for raw in payload.get("folds", [])
    ]
    loso = _load_complete_same_candidate_loso(fold_paths)
    if (
        loso["candidate"].candidate_id != candidate.candidate_id
        or loso["capacity_source_commit"] != source["head_commit_oid"]
        or payload.get("capacity_plan")
        != _pointer(loso["capacity_plan_path"], "v3 capacity plan")
        or not _exact_json_equal(payload.get("folds"), loso["folds"])
        or not _exact_json_equal(
            payload.get("quality_thresholds"), loso["quality_thresholds"]
        )
        or payload.get("fold_cut_probability_thresholds")
        != loso["fold_cut_probability_thresholds"]
        or payload.get("frozen_cut_probability_threshold")
        != loso["frozen_cut_probability_threshold"]
        or payload.get("fold_dynamic_crop_v3_prior_selected_candidate_steps")
        != loso["fold_dynamic_crop_v3_prior_selected_candidate_steps"]
        or payload.get("frozen_dynamic_crop_v3_prior_candidate_step")
        != loso["frozen_dynamic_crop_v3_prior_candidate_step"]
    ):
        raise DynamicCropV3FinalReleaseError(
            "v3 release freeze no longer replays from LOSO evidence"
        )
    profile_path = _verify_pointer(
        payload.get("professional_montage_profile"),
        "professional montage profile",
    )
    try:
        policy = load_professional_montage_policy(profile_path)
    except ProfessionalMontagePolicyError as exc:
        raise DynamicCropV3FinalReleaseError(str(exc)) from exc
    if (
        policy.identity.schema_version != PROFILE_SCHEMA_VERSION
        or payload.get("professional_montage_profile_identity")
        != policy.identity.to_dict()
    ):
        raise DynamicCropV3FinalReleaseError(
            "professional montage profile identity differs"
        )
    return payload


def create_dynamic_crop_v3_final_fit_plan(
    *,
    release_freeze_path: Path,
    checkpoint_directory: Path,
    evidence_directory: Path,
    output_directory: Path,
) -> Path:
    """Create an immutable inactive all-reference refit plan."""

    freeze_path = Path(release_freeze_path).expanduser().resolve()
    freeze = _strict_release_freeze(freeze_path)
    source = _mapping(freeze.get("source_provenance"), "freeze source")
    checkpoint = _under_production_volume(
        checkpoint_directory, "final-fit checkpoint directory"
    )
    evidence = _under_production_volume(
        evidence_directory, "final-fit evidence directory"
    )
    output = _outside_clean_repository(output_directory, "plan output directory")
    if checkpoint.exists() or checkpoint.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite final-fit checkpoint: {checkpoint}"
        )
    try:
        capacity.v2_launcher._validate_artifact_path_isolation(
            plan_path=output,
            plan_label="final-fit plan output directory",
            evidence_directory=evidence,
            planned_checkpoint_directories=(checkpoint,),
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3FinalReleaseError(str(exc)) from exc
    capacity_plan_path = _verify_pointer(
        freeze.get("capacity_plan"), "v3 capacity plan"
    )
    capacity_plan = _strict_capacity_plan(capacity_plan_path)
    songs = deepcopy(capacity_plan.get("songs"))
    selection_dataset = deepcopy(
        capacity_plan.get("selection_dataset_manifest")
    )
    command = [
        "/usr/bin/env",
        f"PYTHONPATH={Path(str(source['repository_root'])) / 'engine'}",
        "python3",
        "-m",
        TRAIN_MODULE,
        "launch",
        "--plan",
        "__CONTENT_ADDRESSED_PLAN_PATH__",
    ]
    _require_same_source_provenance(source)
    unsigned = {
        "schema_version": V3_FINAL_FIT_PLAN_SCHEMA_VERSION,
        "analysis_basis": V3_FINAL_FIT_PLAN_ANALYSIS_BASIS,
        "contract": V3_FINAL_FIT_PLAN_CONTRACT,
        "source_provenance": source,
        "release_freeze": _pointer(freeze_path, "v3 release freeze"),
        "release_freeze_content_sha256": freeze[
            "freeze_content_sha256"
        ],
        "candidate_name": freeze["candidate_name"],
        "candidate_id": freeze["candidate_id"],
        "candidate_contract": freeze["candidate_contract"],
        "training_schedule": freeze["training_schedule"],
        "frozen_dynamic_crop_v3_prior_candidate_step": freeze[
            "frozen_dynamic_crop_v3_prior_candidate_step"
        ],
        "frozen_cut_probability_threshold": freeze[
            "frozen_cut_probability_threshold"
        ],
        "quality_thresholds": freeze["quality_thresholds"],
        "professional_montage_profile": freeze[
            "professional_montage_profile"
        ],
        "professional_montage_profile_identity": freeze[
            "professional_montage_profile_identity"
        ],
        "selection_dataset_manifest": selection_dataset,
        "selection_dataset_content_sha256": capacity_plan[
            "selection_dataset_content_sha256"
        ],
        "songs": songs,
        "checkpoint_directory": str(checkpoint),
        "evidence_directory": str(evidence),
        "command_argv_template": command,
        "all_reference_fit": _all_reference_plan_contract(),
        "authority": deepcopy(freeze["authority"]),
        "plan_content_sha256": None,
    }
    return _atomic_content_addressed_write(
        output_directory=output,
        prefix=V3_FINAL_FIT_PLAN_PREFIX,
        unsigned=unsigned,
        hash_field="plan_content_sha256",
    )


_PLAN_KEYS = {
    "schema_version",
    "analysis_basis",
    "contract",
    "source_provenance",
    "release_freeze",
    "release_freeze_content_sha256",
    "candidate_name",
    "candidate_id",
    "candidate_contract",
    "training_schedule",
    "frozen_dynamic_crop_v3_prior_candidate_step",
    "frozen_cut_probability_threshold",
    "quality_thresholds",
    "professional_montage_profile",
    "professional_montage_profile_identity",
    "selection_dataset_manifest",
    "selection_dataset_content_sha256",
    "songs",
    "checkpoint_directory",
    "evidence_directory",
    "command_argv_template",
    "all_reference_fit",
    "authority",
    "plan_content_sha256",
}


def _strict_final_fit_plan(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = _read_json(resolved, "v3 final-fit plan")
    if set(payload) != _PLAN_KEYS:
        raise DynamicCropV3FinalReleaseError(
            "v3 final-fit plan keys are not exact"
        )
    digest = payload.get("plan_content_sha256")
    unsigned = deepcopy(payload)
    unsigned["plan_content_sha256"] = None
    if (
        not _is_sha256(digest)
        or _canonical_sha256(unsigned) != digest
        or resolved.name != f"{V3_FINAL_FIT_PLAN_PREFIX}{digest}.json"
    ):
        raise DynamicCropV3FinalReleaseError(
            "v3 final-fit plan is not content-addressed"
        )
    if (
        payload.get("schema_version") != V3_FINAL_FIT_PLAN_SCHEMA_VERSION
        or payload.get("analysis_basis") != V3_FINAL_FIT_PLAN_ANALYSIS_BASIS
        or payload.get("contract") != V3_FINAL_FIT_PLAN_CONTRACT
        or not _exact_json_equal(
            payload.get("all_reference_fit"),
            _all_reference_plan_contract(),
        )
        or not _exact_json_equal(
            payload.get("authority"), _inactive_authority_contract()
        )
    ):
        raise DynamicCropV3FinalReleaseError("v3 final-fit plan contract differs")
    source = _validate_source_provenance(payload.get("source_provenance"))
    freeze_path = _verify_pointer(
        payload.get("release_freeze"), "v3 release freeze"
    )
    freeze = _strict_release_freeze(freeze_path)
    if (
        payload.get("release_freeze_content_sha256")
        != freeze["freeze_content_sha256"]
        or payload.get("source_provenance") != freeze["source_provenance"]
        or payload.get("candidate_name") != freeze["candidate_name"]
        or payload.get("candidate_id") != freeze["candidate_id"]
        or not _exact_json_equal(
            payload.get("candidate_contract"), freeze["candidate_contract"]
        )
        or not _exact_json_equal(
            payload.get("training_schedule"), freeze["training_schedule"]
        )
        or payload.get("frozen_dynamic_crop_v3_prior_candidate_step")
        != freeze["frozen_dynamic_crop_v3_prior_candidate_step"]
        or payload.get("frozen_cut_probability_threshold")
        != freeze["frozen_cut_probability_threshold"]
        or not _exact_json_equal(
            payload.get("quality_thresholds"), freeze["quality_thresholds"]
        )
        or payload.get("professional_montage_profile")
        != freeze["professional_montage_profile"]
        or payload.get("professional_montage_profile_identity")
        != freeze["professional_montage_profile_identity"]
        or payload.get("authority") != freeze["authority"]
    ):
        raise DynamicCropV3FinalReleaseError(
            "v3 final-fit plan differs from release freeze"
        )
    capacity_plan_path = _verify_pointer(
        freeze.get("capacity_plan"), "v3 capacity plan"
    )
    capacity_plan = _strict_capacity_plan(capacity_plan_path)
    if (
        not _exact_json_equal(
            payload.get("songs"), capacity_plan.get("songs")
        )
        or payload.get("selection_dataset_manifest")
        != capacity_plan.get("selection_dataset_manifest")
        or payload.get("selection_dataset_content_sha256")
        != capacity_plan.get("selection_dataset_content_sha256")
    ):
        raise DynamicCropV3FinalReleaseError(
            "v3 final-fit manifest graph differs from LOSO plan"
        )
    expected_command = [
        "/usr/bin/env",
        f"PYTHONPATH={Path(str(source['repository_root'])) / 'engine'}",
        "python3",
        "-m",
        TRAIN_MODULE,
        "launch",
        "--plan",
        "__CONTENT_ADDRESSED_PLAN_PATH__",
    ]
    if payload.get("command_argv_template") != expected_command:
        raise DynamicCropV3FinalReleaseError(
            "v3 final-fit launch command differs"
        )
    return payload


def preflight_dynamic_crop_v3_final_fit(
    plan_path: Path,
) -> DynamicCropV3FinalFitPreflight:
    plan_resolved = Path(plan_path).expanduser().resolve()
    plan = _strict_final_fit_plan(plan_resolved)
    source = _mapping(plan.get("source_provenance"), "plan source")
    freeze_path = _verify_pointer(
        plan.get("release_freeze"), "v3 release freeze"
    )
    freeze_pointer = _mapping(
        plan.get("release_freeze"), "v3 release freeze pointer"
    )
    freeze_file_sha = str(freeze_pointer["file_sha256"])
    checkpoint = _under_production_volume(
        Path(str(plan.get("checkpoint_directory") or "")),
        "final-fit checkpoint directory",
    )
    evidence = _under_production_volume(
        Path(str(plan.get("evidence_directory") or "")),
        "final-fit evidence directory",
    )
    if checkpoint.exists() or checkpoint.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite final-fit checkpoint: {checkpoint}"
        )
    materialization, feature, dynamic = _manifest_maps_from_plan(
        plan.get("songs")
    )
    dataset_path = _verify_pointer(
        plan.get("selection_dataset_manifest"),
        "selection dataset manifest",
    )
    try:
        manifests = _validate_manifest_set(
            materialization_manifests=materialization,
            feature_manifests=feature,
            dynamic_crop_manifests=dynamic,
            selection_dataset_manifest_path=dataset_path,
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3FinalReleaseError(str(exc)) from exc
    if (
        not _exact_json_equal(
            [dict(record) for record in manifests.song_records],
            plan.get("songs"),
        )
        or dict(manifests.selection_dataset_pointer)
        != plan.get("selection_dataset_manifest")
        or manifests.selection_dataset_content_sha256
        != plan.get("selection_dataset_content_sha256")
        or set(manifests.reference_ids_by_song) != set(SELECTION_SONG_IDS)
        or any(
            len(references) != 5
            for references in manifests.reference_ids_by_song.values()
        )
    ):
        raise DynamicCropV3FinalReleaseError(
            "v3 final-fit manifest replay differs"
        )
    _require_same_source_provenance(source)
    try:
        free_bytes = capacity.v2_launcher.legacy._validate_storage(checkpoint)
    except (OSError, ValueError) as exc:
        raise DynamicCropV3FinalReleaseError(str(exc)) from exc
    return DynamicCropV3FinalFitPreflight(
        plan_path=plan_resolved,
        plan_content_sha256=str(plan["plan_content_sha256"]),
        release_freeze_path=freeze_path,
        release_freeze_file_sha256=freeze_file_sha,
        candidate_id=str(plan["candidate_id"]),
        checkpoint_directory=checkpoint,
        evidence_directory=evidence,
        source_commit=str(source["head_commit_oid"]),
        frozen_cut_probability_threshold=float(
            plan["frozen_cut_probability_threshold"]
        ),
        frozen_dynamic_crop_v3_prior_candidate_step=int(
            plan["frozen_dynamic_crop_v3_prior_candidate_step"]
        ),
        quality_thresholds=dict(plan["quality_thresholds"]),
        reference_ids_by_song=dict(manifests.reference_ids_by_song),
        supervision_paths_by_song=dict(
            manifests.supervision_paths_by_song
        ),
        snapshot_paths_by_song=dict(manifests.snapshot_paths_by_song),
        dynamic_manifest_paths_by_song=dict(
            manifests.dynamic_manifest_paths_by_song
        ),
        free_bytes=free_bytes,
    )


def _execute_final_fit(
    preflight: DynamicCropV3FinalFitPreflight,
) -> DynamicCropV3FinalFitResult:
    candidate = _candidate_defaults_by_id().get(preflight.candidate_id)
    if candidate is None or candidate.candidate_name != SELECTED_CANDIDATE_NAME:
        raise DynamicCropV3FinalReleaseError(
            "preflight candidate is not production-balanced v3"
        )
    supervision_paths: list[Path] = []
    song_ids_by_reference: dict[str, str] = {}
    snapshot_arguments: list[str] = []
    for song_id in SELECTION_SONG_IDS:
        for reference_id in preflight.reference_ids_by_song[song_id]:
            supervision_paths.append(
                preflight.supervision_paths_by_song[song_id][reference_id]
            )
            song_ids_by_reference[reference_id] = song_id
            snapshot_arguments.append(
                f"{reference_id}="
                f"{preflight.snapshot_paths_by_song[song_id][reference_id]}"
            )
    corpus = load_reference_training_corpus(
        supervision_paths,
        feature_snapshot_by_reference=_snapshot_map(snapshot_arguments),
        song_id_by_reference=song_ids_by_reference,
        negative_samples_per_hold=(
            capacity.v2_launcher.legacy.PRODUCTION_SAMPLING_CONTRACT[
                "negative_samples_per_hold"
            ]
        ),
        boundary_guard_seconds=(
            capacity.v2_launcher.legacy.PRODUCTION_SAMPLING_CONTRACT[
                "boundary_guard_seconds"
            ]
        ),
        max_sources=capacity.v2_launcher.legacy.PRODUCTION_SAMPLING_CONTRACT[
            "max_sources"
        ],
    )
    v2_corpus = build_stage_edit_dynamic_crop_training_corpus_v2(
        corpus.examples,
        artifact_paths_by_reference=_dynamic_artifact_map(
            [],
            [
                preflight.dynamic_manifest_paths_by_song[song]
                for song in SELECTION_SONG_IDS
            ],
        ),
    )
    dynamic_crop_corpus = build_stage_edit_dynamic_crop_training_corpus_v3(
        v2_corpus
    )
    return train_dynamic_crop_v3_final_fit(
        corpus,
        dynamic_crop_corpus=dynamic_crop_corpus,
        checkpoint_directory=preflight.checkpoint_directory,
        model_config=candidate.model_config,
        base_loss_contract=candidate.base_loss_contract,
        dynamic_crop_loss_contract=candidate.dynamic_crop_loss_contract,
        frozen_training_schedule=PROBE_TRAINING_SCHEDULE,
        frozen_dynamic_crop_v3_prior_candidate_step=(
            preflight.frozen_dynamic_crop_v3_prior_candidate_step
        ),
        frozen_cut_probability_threshold=(
            preflight.frozen_cut_probability_threshold
        ),
        frozen_quality_thresholds=preflight.quality_thresholds,
        release_freeze_path=preflight.release_freeze_path,
        release_freeze_sha256=preflight.release_freeze_file_sha256,
        source_commit=preflight.source_commit,
    )


def record_dynamic_crop_v3_final_fit_receipt(
    preflight: DynamicCropV3FinalFitPreflight,
) -> Path:
    plan = _strict_final_fit_plan(preflight.plan_path)
    plan_freeze_pointer = _mapping(
        plan.get("release_freeze"), "v3 release freeze pointer"
    )
    if (
        plan.get("plan_content_sha256") != preflight.plan_content_sha256
        or plan_freeze_pointer.get("path")
        != str(preflight.release_freeze_path)
        or plan_freeze_pointer.get("file_sha256")
        != preflight.release_freeze_file_sha256
    ):
        raise DynamicCropV3FinalReleaseError(
            "final-fit preflight identity changed before receipt"
        )
    source = _mapping(plan.get("source_provenance"), "plan source")
    _require_same_source_provenance(source)
    checkpoint = preflight.checkpoint_directory
    report_path = checkpoint / FINAL_FIT_REPORT_FILENAME
    metadata_path = checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
    weights_path = checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
    report = load_dynamic_crop_v3_final_fit_report(report_path)
    try:
        metadata = StageEditDynamicCropCheckpointMetadataV3.from_dict(
            _read_json(metadata_path, "v3 final-fit metadata")
        )
    except (TypeError, ValueError) as exc:
        raise DynamicCropV3FinalReleaseError(str(exc)) from exc
    candidate = _candidate_defaults_by_id().get(preflight.candidate_id)
    if candidate is None:
        raise DynamicCropV3FinalReleaseError("final-fit candidate disappeared")
    report_checkpoint = _mapping(
        report.get("checkpoint"), "final-fit report checkpoint"
    )
    report_freeze = _mapping(
        report.get("release_freeze"), "final-fit report freeze"
    )
    if (
        report.get("source_commit") != preflight.source_commit
        or report_freeze
        != {
            "path": str(preflight.release_freeze_path),
            "file_sha256": preflight.release_freeze_file_sha256,
        }
        or report.get("frozen_decisions", {}).get("candidate_config")
        != candidate.model_config.to_dict()
        or report.get("frozen_decisions", {}).get("base_loss_contract")
        != asdict(candidate.base_loss_contract)
        or report.get("frozen_decisions", {}).get(
            "dynamic_crop_loss_contract"
        )
        != asdict(candidate.dynamic_crop_loss_contract)
        or report.get("frozen_decisions", {}).get(
            "cut_probability_threshold"
        )
        != preflight.frozen_cut_probability_threshold
        or report.get("frozen_decisions", {}).get(
            "dynamic_crop_v3_prior_candidate_step"
        )
        != preflight.frozen_dynamic_crop_v3_prior_candidate_step
        or report.get("frozen_decisions", {}).get("quality_thresholds")
        != dict(preflight.quality_thresholds)
        or metadata.base_model_config != candidate.model_config
        or metadata.weight_sha256 != _file_sha256(weights_path, "v3 weights")
        or metadata.weight_filename != DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
        or metadata.production_activation_authorized is not False
        or report_checkpoint.get("checkpoint_id") != metadata.checkpoint_id
        or report_checkpoint.get("weight_sha256") != metadata.weight_sha256
        or report_checkpoint.get("training_steps") != metadata.training_steps
    ):
        raise DynamicCropV3FinalReleaseError(
            "final-fit report, checkpoint, and frozen candidate differ"
        )
    forbidden = (
        checkpoint / PRODUCTION_TRAINING_REPORT_FILENAME,
        checkpoint / "production-registry.json",
    )
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise DynamicCropV3FinalReleaseError(
            "inactive final fit contains an activation artifact"
        )
    pointers = {
        "plan": _pointer(preflight.plan_path, "v3 final-fit plan"),
        "release_freeze": dict(plan_freeze_pointer),
        "final_fit_report": _pointer(report_path, "v3 final-fit report"),
        "checkpoint_metadata": _pointer(
            metadata_path, "v3 final-fit metadata"
        ),
        "checkpoint_weights": _pointer(weights_path, "v3 final-fit weights"),
    }
    unsigned = {
        "schema_version": V3_FINAL_FIT_RECEIPT_SCHEMA_VERSION,
        "analysis_basis": V3_FINAL_FIT_RECEIPT_ANALYSIS_BASIS,
        "contract": V3_FINAL_FIT_RECEIPT_CONTRACT,
        "candidate_name": candidate.candidate_name,
        "candidate_id": candidate.candidate_id,
        "source_commit": preflight.source_commit,
        "dynamic_crop_version": 3,
        "model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
        "target_contract": DYNAMIC_CROP_V3_TARGET_CONTRACT,
        "output_contract": DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
        "loss_contract": DYNAMIC_CROP_V3_LOSS_CONTRACT,
        "checkpoint_id": metadata.checkpoint_id,
        "checkpoint_weight_sha256": metadata.weight_sha256,
        "training_steps": metadata.training_steps,
        "reference_count": 15,
        **pointers,
        "production_activation_authorized": False,
        "production_registry_activation_requested": False,
        "learned_source_pair_authority_present": False,
        "blind_holdout_quality_claim_authorized": False,
        "xoxz_dynamic_crop_v3_blind_evidence_present": False,
        "attitude_dynamic_crop_v3_blind_evidence_present": False,
        "human_parity_claim_authorized": False,
        "experimental_unseen_ive_inference_only": True,
        "artifact_content_sha256": None,
    }
    evidence = _under_production_volume(
        preflight.evidence_directory, "final-fit evidence directory"
    )
    receipt_path = _atomic_content_addressed_write(
        output_directory=evidence,
        prefix=V3_FINAL_FIT_RECEIPT_PREFIX,
        unsigned=unsigned,
        hash_field="artifact_content_sha256",
    )
    load_dynamic_crop_v3_final_fit_receipt(receipt_path)
    return receipt_path


_RECEIPT_KEYS = {
    "schema_version",
    "analysis_basis",
    "contract",
    "candidate_name",
    "candidate_id",
    "source_commit",
    "dynamic_crop_version",
    "model_contract",
    "target_contract",
    "output_contract",
    "loss_contract",
    "checkpoint_id",
    "checkpoint_weight_sha256",
    "training_steps",
    "reference_count",
    "plan",
    "release_freeze",
    "final_fit_report",
    "checkpoint_metadata",
    "checkpoint_weights",
    "production_activation_authorized",
    "production_registry_activation_requested",
    "learned_source_pair_authority_present",
    "blind_holdout_quality_claim_authorized",
    "xoxz_dynamic_crop_v3_blind_evidence_present",
    "attitude_dynamic_crop_v3_blind_evidence_present",
    "human_parity_claim_authorized",
    "experimental_unseen_ive_inference_only",
    "artifact_content_sha256",
}


def load_dynamic_crop_v3_final_fit_receipt(path: Path) -> dict[str, Any]:
    """Replay one inactive final-fit receipt without granting authority."""

    resolved = Path(path).expanduser().resolve()
    payload = _read_json(resolved, "v3 final-fit receipt")
    if set(payload) != _RECEIPT_KEYS:
        raise DynamicCropV3FinalReleaseError(
            "v3 final-fit receipt keys are not exact"
        )
    digest = payload.get("artifact_content_sha256")
    unsigned = deepcopy(payload)
    unsigned["artifact_content_sha256"] = None
    if (
        not _is_sha256(digest)
        or _canonical_sha256(unsigned) != digest
        or resolved.name != f"{V3_FINAL_FIT_RECEIPT_PREFIX}{digest}.json"
    ):
        raise DynamicCropV3FinalReleaseError(
            "v3 final-fit receipt is not content-addressed"
        )
    if (
        payload.get("schema_version") != V3_FINAL_FIT_RECEIPT_SCHEMA_VERSION
        or payload.get("analysis_basis")
        != V3_FINAL_FIT_RECEIPT_ANALYSIS_BASIS
        or payload.get("contract") != V3_FINAL_FIT_RECEIPT_CONTRACT
        or payload.get("candidate_name") != SELECTED_CANDIDATE_NAME
        or payload.get("dynamic_crop_version") != 3
        or payload.get("model_contract") != DYNAMIC_CROP_V3_MODEL_CONTRACT
        or payload.get("target_contract") != DYNAMIC_CROP_V3_TARGET_CONTRACT
        or payload.get("output_contract") != DYNAMIC_CROP_V3_OUTPUT_CONTRACT
        or payload.get("loss_contract") != DYNAMIC_CROP_V3_LOSS_CONTRACT
        or payload.get("reference_count") != 15
        or payload.get("production_activation_authorized") is not False
        or payload.get("production_registry_activation_requested") is not False
        or payload.get("learned_source_pair_authority_present") is not False
        or payload.get("blind_holdout_quality_claim_authorized") is not False
        or payload.get("xoxz_dynamic_crop_v3_blind_evidence_present") is not False
        or payload.get("attitude_dynamic_crop_v3_blind_evidence_present") is not False
        or payload.get("human_parity_claim_authorized") is not False
        or payload.get("experimental_unseen_ive_inference_only") is not True
    ):
        raise DynamicCropV3FinalReleaseError(
            "v3 final-fit receipt contract or authority differs"
        )
    candidate = _candidate_defaults_by_id().get(
        str(payload.get("candidate_id") or "")
    )
    if candidate is None or candidate.candidate_name != SELECTED_CANDIDATE_NAME:
        raise DynamicCropV3FinalReleaseError(
            "v3 final-fit receipt candidate differs"
        )
    paths = {
        key: _verify_pointer(payload.get(key), key.replace("_", " "))
        for key in (
            "plan",
            "release_freeze",
            "final_fit_report",
            "checkpoint_metadata",
            "checkpoint_weights",
        )
    }
    plan = _strict_final_fit_plan(paths["plan"])
    source = _mapping(plan.get("source_provenance"), "plan source")
    checkpoint = Path(str(plan.get("checkpoint_directory") or "")).resolve()
    evidence = Path(str(plan.get("evidence_directory") or "")).resolve()
    expected = {
        "release_freeze": _verify_pointer(
            plan.get("release_freeze"), "v3 release freeze"
        ),
        "final_fit_report": checkpoint / FINAL_FIT_REPORT_FILENAME,
        "checkpoint_metadata": checkpoint
        / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
        "checkpoint_weights": checkpoint
        / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
    }
    if (
        resolved.parent != evidence
        or any(paths[key] != value for key, value in expected.items())
        or payload.get("source_commit") != source.get("head_commit_oid")
        or payload.get("candidate_id") != plan.get("candidate_id")
    ):
        raise DynamicCropV3FinalReleaseError(
            "v3 final-fit receipt paths differ from its immutable plan"
        )
    try:
        report = load_dynamic_crop_v3_final_fit_report(
            paths["final_fit_report"]
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DynamicCropV3FinalReleaseError(
            f"v3 final-fit receipt checkpoint identity differs: {exc}"
        ) from exc
    try:
        metadata = StageEditDynamicCropCheckpointMetadataV3.from_dict(
            _read_json(paths["checkpoint_metadata"], "v3 final-fit metadata")
        )
    except (TypeError, ValueError) as exc:
        raise DynamicCropV3FinalReleaseError(str(exc)) from exc
    report_checkpoint = _mapping(
        report.get("checkpoint"), "final-fit report checkpoint"
    )
    report_frozen = _mapping(
        report.get("frozen_decisions"), "final-fit report frozen decisions"
    )
    report_reference_scope = _mapping(
        report.get("all_reference_fit"), "final-fit report reference scope"
    )
    expected_references_by_song: dict[str, list[str]] = {}
    song_records = plan.get("songs")
    if not isinstance(song_records, list):
        raise DynamicCropV3FinalReleaseError(
            "v3 final-fit plan songs are not a list"
        )
    for raw_record in song_records:
        record = _mapping(raw_record, "v3 final-fit plan song")
        song_id = str(record.get("song_id") or "")
        references = record.get("reference_ids")
        if (
            song_id not in SELECTION_SONG_IDS
            or song_id in expected_references_by_song
            or not isinstance(references, list)
            or len(references) != 5
            or any(not isinstance(item, str) or not item for item in references)
        ):
            raise DynamicCropV3FinalReleaseError(
                "v3 final-fit plan reference scope differs"
            )
        expected_references_by_song[song_id] = sorted(references)
    if set(expected_references_by_song) != set(SELECTION_SONG_IDS):
        raise DynamicCropV3FinalReleaseError(
            "v3 final-fit plan reference songs differ"
        )
    plan_freeze_pointer = _mapping(
        plan.get("release_freeze"), "v3 release freeze pointer"
    )
    try:
        loaded_model, loaded_metadata = (
            load_stage_edit_dynamic_crop_checkpoint_v3(
                checkpoint,
                expected_schema=DEFAULT_STAGE_EDIT_SCHEMA,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DynamicCropV3FinalReleaseError(
            f"standard v3 checkpoint replay failed: {exc}"
        ) from exc
    loaded_encoder = getattr(
        getattr(loaded_model, "base", None), "encoder", None
    )
    if (
        metadata.base_model_config != candidate.model_config
        or metadata.dynamic_crop_config
        != candidate.dynamic_crop_config
        or metadata.weights_trained is not True
        or metadata.weight_filename
        != DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
        or metadata.weight_sha256
        != _file_sha256(paths["checkpoint_weights"], "v3 final-fit weights")
        or metadata.production_activation_authorized is not False
        or payload.get("checkpoint_id") != metadata.checkpoint_id
        or payload.get("checkpoint_weight_sha256") != metadata.weight_sha256
        or payload.get("training_steps") != metadata.training_steps
        or report_checkpoint.get("checkpoint_id") != metadata.checkpoint_id
        or report_checkpoint.get("weight_sha256") != metadata.weight_sha256
        or report_checkpoint.get("directory") != str(checkpoint)
        or report_checkpoint.get("metadata_filename")
        != DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
        or report_checkpoint.get("weight_filename")
        != DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
        or report_checkpoint.get("model_config")
        != candidate.model_config.to_dict()
        or report.get("source_commit") != source.get("head_commit_oid")
        or report.get("release_freeze")
        != plan_freeze_pointer
        or report_reference_scope.get("references_by_song")
        != expected_references_by_song
        or report_frozen.get("candidate_config")
        != candidate.model_config.to_dict()
        or report_frozen.get("base_loss_contract")
        != asdict(candidate.base_loss_contract)
        or report_frozen.get("dynamic_crop_loss_contract")
        != asdict(candidate.dynamic_crop_loss_contract)
        or not _exact_json_equal(
            report_frozen.get("training_schedule"),
            plan.get("training_schedule"),
        )
        or report_frozen.get("dynamic_crop_v3_prior_candidate_step")
        != plan.get("frozen_dynamic_crop_v3_prior_candidate_step")
        or report_frozen.get("cut_probability_threshold")
        != plan.get("frozen_cut_probability_threshold")
        or not _exact_json_equal(
            report_frozen.get("quality_thresholds"),
            plan.get("quality_thresholds"),
        )
        or loaded_metadata != metadata
        or not isinstance(loaded_model, StageEditTransformerDynamicCropV3)
        or loaded_encoder is None
        or getattr(loaded_encoder, "checkpoint", None) is not False
    ):
        raise DynamicCropV3FinalReleaseError(
            "v3 final-fit receipt checkpoint identity differs"
        )
    if (
        (checkpoint / PRODUCTION_TRAINING_REPORT_FILENAME).exists()
        or (checkpoint / PRODUCTION_TRAINING_REPORT_FILENAME).is_symlink()
        or (checkpoint / "production-registry.json").exists()
        or (checkpoint / "production-registry.json").is_symlink()
    ):
        raise DynamicCropV3FinalReleaseError(
            "v3 final-fit receipt checkpoint gained activation artifacts"
        )
    return payload


def launch_dynamic_crop_v3_final_fit(
    plan_path: Path,
) -> tuple[DynamicCropV3FinalFitPreflight, Path]:
    preflight = preflight_dynamic_crop_v3_final_fit(plan_path)
    plan = _strict_final_fit_plan(preflight.plan_path)
    source = _mapping(plan.get("source_provenance"), "plan source")
    _require_same_source_provenance(source)
    _execute_final_fit(preflight)
    plan = _strict_final_fit_plan(preflight.plan_path)
    _require_same_source_provenance(
        _mapping(plan.get("source_provenance"), "plan source")
    )
    receipt = record_dynamic_crop_v3_final_fit_receipt(preflight)
    return preflight, receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze passing Dynamic Crop v3 LOSO evidence and run an inactive "
            "all-reference final fit"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--fold-receipt", action="append", required=True, type=Path)
    freeze.add_argument("--professional-montage-profile", required=True, type=Path)
    freeze.add_argument("--output-directory", required=True, type=Path)
    create = commands.add_parser("create-final-fit")
    create.add_argument("--release-freeze", required=True, type=Path)
    create.add_argument("--checkpoint-directory", required=True, type=Path)
    create.add_argument("--evidence-directory", required=True, type=Path)
    create.add_argument("--output-directory", required=True, type=Path)
    for name in ("preflight", "launch"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        path = create_dynamic_crop_v3_release_freeze(
            fold_receipt_paths=args.fold_receipt,
            professional_montage_grammar_profile_path=(
                args.professional_montage_profile
            ),
            output_directory=args.output_directory,
        )
        print(json.dumps({"release_freeze": str(path)}, indent=2))
        return 0
    if args.command == "create-final-fit":
        path = create_dynamic_crop_v3_final_fit_plan(
            release_freeze_path=args.release_freeze,
            checkpoint_directory=args.checkpoint_directory,
            evidence_directory=args.evidence_directory,
            output_directory=args.output_directory,
        )
        print(json.dumps({"final_fit_plan": str(path)}, indent=2))
        return 0
    if args.command == "preflight":
        preflight = preflight_dynamic_crop_v3_final_fit(args.plan)
        print(json.dumps(preflight.to_dict(), indent=2))
        return 0
    preflight, receipt = launch_dynamic_crop_v3_final_fit(args.plan)
    print(
        json.dumps(
            {
                "preflight": preflight.to_dict(),
                "final_fit_receipt": str(receipt),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DynamicCropV3FinalFitPreflight",
    "DynamicCropV3FinalReleaseError",
    "V3_FINAL_FIT_PLAN_CONTRACT",
    "V3_FINAL_FIT_RECEIPT_CONTRACT",
    "V3_RELEASE_FREEZE_CONTRACT",
    "build_parser",
    "create_dynamic_crop_v3_final_fit_plan",
    "create_dynamic_crop_v3_release_freeze",
    "launch_dynamic_crop_v3_final_fit",
    "load_dynamic_crop_v3_final_fit_receipt",
    "main",
    "preflight_dynamic_crop_v3_final_fit",
    "record_dynamic_crop_v3_final_fit_receipt",
]
