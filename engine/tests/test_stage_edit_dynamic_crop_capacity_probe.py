from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from crosscut.stage_edit_dynamic_crop_capacity_probe import (
    BANG_BANG,
    BLACKHOLE,
    REBEL_HEART,
    FOLD_EVIDENCE_PREFIX,
    SMOKE_EVIDENCE_ANALYSIS_BASIS,
    SMOKE_EVIDENCE_SCHEMA_VERSION,
    SMOKE_EVIDENCE_PREFIX,
    SMOKE_INPUT_PACKING_CONTRACT,
    SMOKE_MEMORY_LIMIT_CONTRACT,
    SMOKE_PROCESS_ISOLATION_CONTRACT,
    SMOKE_RUNNER_CONTRACT,
    SMOKE_RUNNER_LOG_ANALYSIS_BASIS,
    SMOKE_RUNNER_LOG_PREFIX,
    SMOKE_RUNNER_LOG_SCHEMA_VERSION,
    SMOKE_SUPERVISOR_ATTESTATION_CONTRACT,
    SMOKE_TILING_CONTRACT,
    SMOKE_WORKER_EVENT_PREFIX,
    SMOKE_CACHE_LIMIT_BYTES,
    DeviceMemoryBudget,
    DynamicCropCapacityCandidate,
    DynamicCropCapacityProbeError,
    FoldEvaluation,
    SelectionPolicy,
    build_dynamic_crop_capacity_selection as _build_dynamic_crop_capacity_selection,
    default_dynamic_crop_v2_probe_candidates,
    _atomic_exclusive_write,
    _load_fold_evaluation,
    _load_selection_dataset,
    _publish_supervised_mlx_smoke_evidence,
    _validate_schedule,
    _validate_mlx_smoke_evidence_root,
    load_dynamic_crop_capacity_selection as _load_dynamic_crop_capacity_selection,
    write_dynamic_crop_capacity_selection as _write_dynamic_crop_capacity_selection,
    write_fold_run_evidence,
    write_mlx_smoke_evidence,
    write_selection_dataset_manifest,
)
from crosscut.stage_edit_dynamic_crop_head_v2 import (
    DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME,
    DYNAMIC_CROP_V2_LOSS_CONTRACT,
    DYNAMIC_CROP_V2_MODEL_CONTRACT,
    DYNAMIC_CROP_V2_SCHEMA_VERSION,
    DYNAMIC_CROP_V2_TARGET_CONTRACT,
    DynamicCropHeadV2Config,
    DynamicCropV2LossContract,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v2,
    stage_edit_dynamic_crop_v2_parameter_count,
)
from crosscut.stage_edit_production_training import (
    STAGE_EDIT_SELECTION_FREEZE_FILENAME,
    ProductionQualityThresholds,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    PRODUCTION_CHECKPOINT_SCOPE,
    PRODUCTION_TRAINING_REPORT_FILENAME,
    PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
    PRODUCTION_TRAINING_RUN_KIND,
    STAGE_EDIT_VALIDATION_SELECTION_CONTRACT,
    StageEditLossContract,
    StageEditTransformerConfig,
)
from crosscut.stage_edit_transition_authority import (
    RENDERER_OWNED_TRANSITION_AUTHORITY,
    StageEditTransitionAuthority,
)
from crosscut.stage_edit_dynamic_crop_capacity_probe import PROBE_TRAINING_SCHEDULE


SONGS = (BANG_BANG, BLACKHOLE, REBEL_HEART)
REFERENCES = {
    song: tuple(f"{song.lower()}-ref-{index}" for index in range(5))
    for song in SONGS
}
BUDGET = DeviceMemoryBudget(
    device_name="Apple M4 Pro",
    physical_memory_bytes=25_769_803_776,
)


def build_dynamic_crop_capacity_selection(**kwargs):
    """Exercise selector math with the external clean-checkout gate mocked."""

    with patch(
        "crosscut.stage_edit_dynamic_crop_capacity_probe._enforce_live_release_smoke_attestation"
    ), patch(
        "crosscut.stage_edit_dynamic_crop_capacity_probe._enforce_actual_smoke_source_paths"
    ), patch(
        "crosscut.stage_edit_dynamic_crop_capacity_probe._enforce_release_capacity_candidate_set"
    ):
        return _build_dynamic_crop_capacity_selection(**kwargs)


def write_dynamic_crop_capacity_selection(*args, **kwargs):
    with patch(
        "crosscut.stage_edit_dynamic_crop_capacity_probe._enforce_live_release_smoke_attestation"
    ), patch(
        "crosscut.stage_edit_dynamic_crop_capacity_probe._enforce_actual_smoke_source_paths"
    ), patch(
        "crosscut.stage_edit_dynamic_crop_capacity_probe._enforce_release_capacity_candidate_set"
    ):
        return _write_dynamic_crop_capacity_selection(*args, **kwargs)


def load_dynamic_crop_capacity_selection(*args, **kwargs):
    with patch(
        "crosscut.stage_edit_dynamic_crop_capacity_probe._enforce_live_release_smoke_attestation"
    ), patch(
        "crosscut.stage_edit_dynamic_crop_capacity_probe._enforce_actual_smoke_source_paths"
    ), patch(
        "crosscut.stage_edit_dynamic_crop_capacity_probe._enforce_release_capacity_candidate_set"
    ):
        return _load_dynamic_crop_capacity_selection(*args, **kwargs)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_addressed(root: Path, prefix: str, unsigned: dict[str, object]) -> Path:
    payload = {**unsigned, "artifact_content_sha256": None}
    digest = _canonical_hash(payload)
    payload["artifact_content_sha256"] = digest
    path = root / f"{prefix}{digest}.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def _candidate(name: str, dimension: int) -> DynamicCropCapacityCandidate:
    base = StageEditLossContract(
        transition_weight=0.0,
        transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
    )
    return DynamicCropCapacityCandidate(
        candidate_name=name,
        model_config=StageEditTransformerConfig.from_schema(
            DEFAULT_STAGE_EDIT_SCHEMA,
            model_dimension=dimension,
            num_heads=4,
            num_layers=2,
            feed_forward_dimension=dimension * 3,
            max_sequence_length=2048,
            max_sources=32,
            dropout=0.0,
            transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
        ),
        dynamic_crop_config=DynamicCropHeadV2Config(),
        base_loss_contract=base,
        dynamic_crop_loss_contract=DynamicCropV2LossContract(base=base),
    )


def _dataset(root: Path, references=REFERENCES) -> Path:
    manifests = {}
    for song in SONGS:
        outputs = []
        for reference in references[song]:
            supervision = root / f"{reference}-supervision.json"
            supervision.write_text(
                json.dumps({"reference_id": reference}), encoding="utf-8"
            )
            outputs.append(
                {
                    "reference_id": reference,
                    "supervision_path": str(supervision.resolve()),
                    "supervision_file_sha256": _hash_file(supervision),
                }
            )
        path = root / f"{song}-source-manifest.json"
        path.write_text(
            json.dumps(
                {
                    "song_id": song,
                    "input_contract": {
                        "style_id": {
                            BANG_BANG: "bang-bang-human-reference-10source-v2",
                            BLACKHOLE: "blackhole-human-reference-10source-v2",
                            REBEL_HEART: "rebel-heart-human-reference-10source-v3-current-code",
                        }[song],
                        "expected_reference_count": 5,
                        "inputs": [
                            {"reference_id": reference}
                            for reference in references[song]
                        ],
                    },
                    "statistics": {"reference_count": 5},
                    "outputs": outputs,
                }
            ),
            encoding="utf-8",
        )
        manifests[song] = path.resolve()
    return write_selection_dataset_manifest(
        root,
        manifest_paths_by_song=manifests,
        reference_ids_by_song=references,
    )


def _smoke_code_provenance(root: Path, *, release_trusted: bool) -> dict[str, object]:
    roles = (
        "runner",
        "capacity_probe",
        "dynamic_crop_head",
        "transformer",
        "feature_snapshot_loader",
        "feature_snapshot_manifest_contract",
        "reference_adapter",
        "dynamic_crop_training",
        "production_training",
        "dynamic_crop_manifest_validator",
        "dynamic_crop_supervision_loader",
    )
    repo = root / "smoke-code-repo"
    repo.mkdir(exist_ok=True)
    (repo / "engine").mkdir(exist_ok=True)
    if not (repo / ".git").exists():
        subprocess.run(("git", "init", "-q"), cwd=repo, check=True)
        subprocess.run(
            ("git", "config", "user.email", "smoke@example.invalid"),
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ("git", "config", "user.name", "Smoke Fixture"),
            cwd=repo,
            check=True,
        )
        for role in roles:
            (repo / f"{role}.py").write_text(f"# {role}\n", encoding="utf-8")
        subprocess.run(("git", "add", "."), cwd=repo, check=True)
        subprocess.run(
            ("git", "commit", "-q", "-m", "fixture"), cwd=repo, check=True
        )
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    source_files = {}
    for role in roles:
        relative = f"{role}.py"
        path = (repo / relative).resolve()
        file_hash = _hash_file(path)
        source_files[role] = {
            "path": str(path),
            "file_sha256": file_hash,
            "repo_relative_path": relative,
            "observed_commit_blob_sha256": file_hash,
            "matches_observed_commit": True,
        }
    empty_hash = hashlib.sha256(b"").hexdigest()
    return {
        "runner_contract": SMOKE_RUNNER_CONTRACT,
        "expected_git_commit": commit,
        "observed_git_commit": commit,
        "tracked_diff_policy": (
            "require_clean" if release_trusted else "record_development"
        ),
        "tracked_diff_clean": True,
        "tracked_diff_sha256": empty_hash,
        "repository_status_clean": True,
        "repository_status_entry_count": 0,
        "repository_status_porcelain_sha256": empty_hash,
        "untracked_dependency_paths": [],
        "source_files_match_observed_commit": True,
        "release_trust_authorized": release_trusted,
        "source_files": source_files,
    }


def _smoke(
    root: Path,
    candidate: DynamicCropCapacityCandidate,
    *,
    passed: bool = True,
    suite_id: str = "e" * 64,
    partial_oom_peak_bytes: int | None = None,
) -> Path:
    from crosscut.stage_edit_dynamic_crop_mlx_smoke_cli import (
        _execution_from_process,
        _runtime_provenance,
    )

    feature_manifests: dict[str, dict[str, str]] = {}
    materialization_manifests: dict[str, dict[str, str]] = {}
    dynamic_crop_manifests: dict[str, dict[str, str]] = {}
    feature_snapshots: list[dict[str, object]] = []
    reference_supervisions: list[dict[str, object]] = []
    dynamic_crop_artifacts: list[dict[str, object]] = []
    lane_order: list[str] = []
    global_reference_index = 0
    for song in SONGS:
        manifest = root / f"{song}-feature-manifest.json"
        manifest.write_text(json.dumps({"song_id": song}), encoding="utf-8")
        feature_manifests[song] = {
            "path": str(manifest.resolve()),
            "file_sha256": _hash_file(manifest),
        }
        materialization = (root / f"{song}-source-manifest.json").resolve()
        materialization_payload = json.loads(
            materialization.read_text(encoding="utf-8")
        )
        materialization_manifests[song] = {
            "path": str(materialization),
            "file_sha256": _hash_file(materialization),
        }
        outputs_by_reference = {
            output["reference_id"]: output
            for output in materialization_payload["outputs"]
        }
        manifest_artifacts: list[dict[str, object]] = []
        for reference in REFERENCES[song]:
            snapshot = root / f"{reference}-feature-snapshot.json"
            snapshot.write_text(
                json.dumps({"reference_id": reference}), encoding="utf-8"
            )
            feature_snapshots.append(
                {
                    "song_id": song,
                    "reference_id": reference,
                    "token_count": 1,
                    "snapshot_content_sha256": _canonical_hash(
                        {"reference_id": reference}
                    ),
                    "file": {
                        "path": str(snapshot.resolve()),
                        "file_sha256": _hash_file(snapshot),
                    },
                }
            )
            lane_order.append(reference)
            supervision_output = outputs_by_reference[reference]
            reference_supervisions.append(
                {
                    "song_id": song,
                    "reference_id": reference,
                    "file": {
                        "path": supervision_output["supervision_path"],
                        "file_sha256": supervision_output[
                            "supervision_file_sha256"
                        ],
                    },
                }
            )
            shot_count = 39 if global_reference_index == 14 else 40
            for shot_index in range(shot_count):
                artifact_path = root / (
                    f"{reference}-crop-v2-{shot_index:04d}.json"
                )
                artifact_payload: dict[str, object] = {
                    "identity": {
                        "song_id": song,
                        "reference_id": reference,
                        "shot_index": shot_index,
                    },
                    "provenance": {"artifact_content_sha256": None},
                }
                content_hash = _canonical_hash(artifact_payload)
                artifact_payload["provenance"] = {
                    "artifact_content_sha256": content_hash
                }
                artifact_path.write_text(
                    json.dumps(artifact_payload, sort_keys=True), encoding="utf-8"
                )
                file_hash = _hash_file(artifact_path)
                artifact_record = {
                    "song_id": song,
                    "reference_id": reference,
                    "shot_index": shot_index,
                    "file": {
                        "path": str(artifact_path.resolve()),
                        "file_sha256": file_hash,
                    },
                    "content_sha256": content_hash,
                }
                dynamic_crop_artifacts.append(artifact_record)
                manifest_artifacts.append(
                    {
                        "reference_id": reference,
                        "shot_index": shot_index,
                        "artifact": {
                            "path": str(artifact_path.resolve()),
                            "file_sha256": file_hash,
                            "content_sha256": content_hash,
                        },
                    }
                )
            global_reference_index += 1
        dynamic_manifest = root / f"{song}-dynamic-crop-manifest.json"
        dynamic_manifest.write_text(
            json.dumps({"song_id": song, "artifacts": manifest_artifacts}),
            encoding="utf-8",
        )
        dynamic_crop_manifests[song] = {
            "path": str(dynamic_manifest.resolve()),
            "file_sha256": _hash_file(dynamic_manifest),
        }
    dataset = {
        "selection_song_ids": list(SONGS),
        "quarantined_song_ids_opened": False,
        "reference_count": 15,
        "reference_ids_by_song": {
            song: list(REFERENCES[song]) for song in SONGS
        },
        "feature_manifests": feature_manifests,
        "feature_snapshots": feature_snapshots,
        "materialization_manifests": materialization_manifests,
        "dynamic_crop_manifests": dynamic_crop_manifests,
        "reference_supervisions": reference_supervisions,
        "reference_supervision_count": 15,
        "reference_supervision_aggregate_sha256": _canonical_hash(
            reference_supervisions
        ),
        "dynamic_crop_artifacts": dynamic_crop_artifacts,
        "dynamic_crop_artifact_count": 599,
        "dynamic_crop_artifact_aggregate_sha256": _canonical_hash(
            dynamic_crop_artifacts
        ),
    }
    code = _smoke_code_provenance(root, release_trusted=passed)
    runtime = _runtime_provenance(BUDGET, root / "smoke-code-repo")
    hardware = {
        "device_name": BUDGET.device_name,
        "physical_memory_bytes": BUDGET.physical_memory_bytes,
        "max_recommended_working_set_size": BUDGET.max_recommended_working_set_size,
        "max_buffer_length": BUDGET.max_buffer_length,
        "architecture": BUDGET.architecture,
        "os_reserve_bytes": BUDGET.os_reserve_bytes,
        "allowed_training_bytes": BUDGET.allowed_training_bytes,
        "device_info_api": "mlx.core.device_info",
    }
    tensor = {
        "packing_contract": SMOKE_INPUT_PACKING_CONTRACT,
        "tiling_contract": SMOKE_TILING_CONTRACT,
        "batch_size": 2,
        "sequence_length": 2048,
        "max_sources": 32,
        "actual_observed_source_max": 10,
        "context_feature_dimension": candidate.model_config.context_input_dimension,
        "source_feature_dimension": 720,
        "source_tensor_shape": [2, 2048, 32, 720],
        "context_tensor_shape": [
            2,
            2048,
            candidate.model_config.context_input_dimension,
        ],
        "dynamic_values_shape": [2, 2048, 8, 4],
        "dynamic_times_shape": [2, 2048, 8],
        "dynamic_presence_shape": [2, 2048, 8],
        "dtype": "float32",
        "lane_reference_ids": {"A": lane_order[0::2], "B": lane_order[1::2]},
        "lane_original_cycle_token_counts": {"A": 8, "B": 7},
        "lane_expanded_token_counts": {"A": 2048, "B": 2048},
        "lane_repeat_counts": {"A": 256, "B": 292},
        "lane_truncated_prefix_counts": {"A": 0, "B": 4},
        "lane_padding_counts": {"A": 0, "B": 0},
        "base_label_mask_counts": (
            {
                "cut": 100,
                "source_listwise": 100,
                "source_pairwise": 100,
                "duration": 100,
                "transition_metadata_only": 100,
                "crop_coordinates": 400,
            }
            if passed or partial_oom_peak_bytes is not None
            else None
        ),
        "dynamic_label_mask_counts": (
            {"tracks": 100, "keyframes": 200, "presence": 800, "motion": 100}
            if passed or partial_oom_peak_bytes is not None
            else None
        ),
        "parameter_count": stage_edit_dynamic_crop_v2_parameter_count(
            candidate.model_config, candidate.dynamic_crop_config
        ),
        "input_tensor_sha256": (
            "b" * 64 if passed or partial_oom_peak_bytes is not None else None
        ),
        "supervision_tensor_sha256": (
            "c" * 64 if passed or partial_oom_peak_bytes is not None else None
        ),
        "real_input_materialized": passed or partial_oom_peak_bytes is not None,
    }
    provenance_expectations = {
        "candidate_contract_sha256": _canonical_hash(candidate.to_dict()),
        "dataset_provenance_sha256": _canonical_hash(dataset),
        "code_provenance_sha256": _canonical_hash(code),
        "runtime_provenance_sha256": _canonical_hash(runtime),
        "hardware_provenance_sha256": _canonical_hash(hardware),
        "git_commit": code["expected_git_commit"],
    }
    events: list[dict[str, object]] = [
        {
            "event": "provenance_verified",
            "candidate_id": candidate.candidate_id,
            "candidate_contract_sha256": provenance_expectations[
                "candidate_contract_sha256"
            ],
            "expected_dataset_provenance_sha256": provenance_expectations[
                "dataset_provenance_sha256"
            ],
            "observed_dataset_provenance_sha256": provenance_expectations[
                "dataset_provenance_sha256"
            ],
            "expected_code_provenance_sha256": provenance_expectations[
                "code_provenance_sha256"
            ],
            "observed_code_provenance_sha256": provenance_expectations[
                "code_provenance_sha256"
            ],
            "expected_runtime_provenance_sha256": provenance_expectations[
                "runtime_provenance_sha256"
            ],
            "observed_runtime_provenance_sha256": provenance_expectations[
                "runtime_provenance_sha256"
            ],
            "observed_runtime_modules_sha256": _canonical_hash(
                runtime["imported_runtime_modules"]
            ),
            "observed_runtime_distributions_sha256": _canonical_hash(
                runtime["runtime_distributions"]
            ),
            "worker_environment_contract": runtime[
                "worker_environment_contract"
            ],
            "worker_pythonpath_entries": runtime[
                "worker_pythonpath_entries"
            ],
            "expected_hardware_provenance_sha256": provenance_expectations[
                "hardware_provenance_sha256"
            ],
            "observed_hardware_provenance_sha256": provenance_expectations[
                "hardware_provenance_sha256"
            ],
            "expected_git_commit": code["expected_git_commit"],
            "observed_git_commit": code["observed_git_commit"],
            "matched": True,
        }
    ]
    parameter_count = tensor["parameter_count"]
    tree = {
        "leaf_count": 1,
        "element_count": parameter_count,
        "nbytes": parameter_count * 4,
        "key_digest_sha256": "3" * 64,
        "nonfinite_count": 0,
    }
    if passed:
        stages = [
            {
                "stage": stage,
                "synchronized": True,
                "active_memory_bytes": 1024,
                "cache_memory_bytes": 0,
                "peak_memory_bytes": 1024**3,
                "elapsed_seconds": float(index + 1),
            }
            for index, stage in enumerate(
                (
                    "input_materialized",
                    "model_allocated",
                    "forward_backward_completed",
                    "gradient_clip_completed",
                    "optimizer_step_completed",
                )
            )
        ]
        events.extend(
            [
                {
                    "event": "input_and_limits_ready",
                    "tensor_provenance": tensor,
                    "previous_memory_limit_bytes": BUDGET.physical_memory_bytes,
                    "applied_memory_limit_bytes": BUDGET.allowed_training_bytes,
                    "previous_cache_limit_bytes": 0,
                    "applied_cache_limit_bytes": SMOKE_CACHE_LIMIT_BYTES,
                    "memory_limit_contract": SMOKE_MEMORY_LIMIT_CONTRACT,
                    "pre_reset_active_memory_bytes": 0,
                    "pre_reset_cache_memory_bytes": 0,
                },
                {"event": "input_materialized", "memory_stage": stages[0]},
                {
                    "event": "model_allocated",
                    "memory_stage": stages[1],
                    "parameter_tree": tree,
                },
                {
                    "event": "forward_backward_completed",
                    "loss_value_for_execution_only": 1.0,
                    "gradient_tree": {**tree, "key_digest_sha256": "4" * 64},
                    "memory_stage": stages[2],
                },
                {
                    "event": "gradient_clip_completed",
                    "gradient_global_norm_before_clip": 1.0,
                    "memory_stage": stages[3],
                },
                {
                    "event": "optimizer_step_completed",
                    "gradients_finite": True,
                    "optimizer_state_materialized": True,
                    "post_update_parameter_tree": {
                        **tree,
                        "key_digest_sha256": "6" * 64,
                    },
                    "updated_parameters_finite": True,
                    "optimizer_state_tree": {
                        "leaf_count": 3,
                        "element_count": 2 * parameter_count + 1,
                        "nbytes": 8 * parameter_count + 4,
                        "key_digest_sha256": "5" * 64,
                        "nonfinite_count": 0,
                    },
                    "optimizer_state_finite": True,
                    "memory_stage": stages[4],
                },
            ]
        )
        returncode = 0
    else:
        if partial_oom_peak_bytes is not None:
            input_stage = {
                "stage": "input_materialized",
                "synchronized": True,
                "active_memory_bytes": 1024,
                "cache_memory_bytes": 0,
                "peak_memory_bytes": 1024**3,
                "elapsed_seconds": 1.0,
            }
            model_stage = {
                "stage": "model_allocated",
                "synchronized": True,
                "active_memory_bytes": 1024,
                "cache_memory_bytes": 0,
                "peak_memory_bytes": 2 * 1024**3,
                "elapsed_seconds": 2.0,
            }
            events.extend(
                [
                    {
                        "event": "input_and_limits_ready",
                        "tensor_provenance": tensor,
                        "previous_memory_limit_bytes": BUDGET.physical_memory_bytes,
                        "applied_memory_limit_bytes": BUDGET.allowed_training_bytes,
                        "previous_cache_limit_bytes": 0,
                        "applied_cache_limit_bytes": SMOKE_CACHE_LIMIT_BYTES,
                        "memory_limit_contract": SMOKE_MEMORY_LIMIT_CONTRACT,
                        "pre_reset_active_memory_bytes": 0,
                        "pre_reset_cache_memory_bytes": 0,
                    },
                    {"event": "input_materialized", "memory_stage": input_stage},
                    {
                        "event": "model_allocated",
                        "memory_stage": model_stage,
                        "parameter_tree": tree,
                    },
                ]
            )
        events.append(
            {
                "event": "worker_failure",
                "failure_kind": "out_of_memory",
                "exception_type": "MemoryError",
                "exception_message": "fixture OOM",
                "measured_peak_bytes": partial_oom_peak_bytes,
            }
        )
        returncode = 70
    supervisor_pid = 42001
    worker_pid = 42002
    suite_started_at_unix_ns = 500_000
    suite_ended_at_unix_ns = 3_000_000
    run_nonce = "d" * 64
    run_nonce_sha256 = hashlib.sha256(run_nonce.encode("ascii")).hexdigest()
    events = [
        {
            **event,
            "worker_pid": worker_pid,
            "supervisor_run_nonce_sha256": run_nonce_sha256,
            "supervisor_suite_id": suite_id,
        }
        for event in events
    ]
    tensor_from_events, execution = _execution_from_process(
        candidate=candidate,
        budget=BUDGET,
        code_provenance=code,
        provenance_expectations=provenance_expectations,
        tensor_template=tensor,
        events=events,
        returncode=returncode,
    )
    command = [
        sys.executable,
        "-P",
        "-S",
        "-m",
        "crosscut.stage_edit_dynamic_crop_mlx_smoke_cli",
        "--worker",
        "--candidate-id",
        candidate.candidate_id,
        "--suite-id",
        suite_id,
        "--expected-candidate-contract-sha256",
        provenance_expectations["candidate_contract_sha256"],
        "--expected-dataset-provenance-sha256",
        provenance_expectations["dataset_provenance_sha256"],
        "--expected-code-provenance-sha256",
        provenance_expectations["code_provenance_sha256"],
        "--expected-runtime-provenance-sha256",
        provenance_expectations["runtime_provenance_sha256"],
        "--expected-hardware-provenance-sha256",
        provenance_expectations["hardware_provenance_sha256"],
        "--expected-git-commit",
        code["expected_git_commit"],
        "--tracked-diff-policy",
        code["tracked_diff_policy"],
    ]
    for song in SONGS:
        command.extend(
            ["--feature-manifest", f"{song}={feature_manifests[song]['path']}"]
        )
        command.extend(
            [
                "--materialization-manifest",
                f"{song}={materialization_manifests[song]['path']}",
            ]
        )
        command.extend(
            [
                "--dynamic-crop-manifest",
                f"{song}={dynamic_crop_manifests[song]['path']}",
            ]
        )
    stdout = "".join(
        SMOKE_WORKER_EVENT_PREFIX
        + json.dumps(event, sort_keys=True, separators=(",", ":"))
        + "\n"
        for event in events
    )
    started_at_unix_ns = 1_000_000
    ended_at_unix_ns = 2_000_000
    receipt_core = {
        "candidate_id": candidate.candidate_id,
        "command": command,
        "supervisor_pid": supervisor_pid,
        "worker_pid": worker_pid,
        "returncode": returncode,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "run_nonce_sha256": run_nonce_sha256,
        "started_at_unix_ns": started_at_unix_ns,
        "ended_at_unix_ns": ended_at_unix_ns,
        "suite_id": suite_id,
        "suite_started_at_unix_ns": suite_started_at_unix_ns,
        "suite_ended_at_unix_ns": suite_ended_at_unix_ns,
    }
    log = _write_addressed(
        root,
        SMOKE_RUNNER_LOG_PREFIX,
        {
            "schema_version": SMOKE_RUNNER_LOG_SCHEMA_VERSION,
            "analysis_basis": SMOKE_RUNNER_LOG_ANALYSIS_BASIS,
            "candidate_id": candidate.candidate_id,
            "command": command,
            "supervisor_pid": supervisor_pid,
            "worker_pid": worker_pid,
            "returncode": returncode,
            "stdout": stdout,
            "worker_event_count": len(events),
            "stdout_event_sha256": _canonical_hash(events),
            "stderr": "",
            "run_nonce_sha256": run_nonce_sha256,
            "subprocess_receipt_sha256": _canonical_hash(receipt_core),
            "started_at_unix_ns": started_at_unix_ns,
            "ended_at_unix_ns": ended_at_unix_ns,
            "suite_id": suite_id,
            "suite_started_at_unix_ns": suite_started_at_unix_ns,
            "suite_ended_at_unix_ns": suite_ended_at_unix_ns,
        },
    )
    runner_source = code["source_files"]["runner"]
    evidence = {
        "semantic_metrics_authorized": False,
        "candidate": candidate.to_dict(),
        "dataset_provenance": dataset,
        "code_provenance": code,
        "runtime_provenance": runtime,
        "hardware_provenance": hardware,
        "tensor_provenance": tensor_from_events,
        "execution": execution,
        "supervisor_attestation": {
            "contract": SMOKE_SUPERVISOR_ATTESTATION_CONTRACT,
            "candidate_id": candidate.candidate_id,
            "supervisor_pid": supervisor_pid,
            "worker_pid": worker_pid,
            "run_nonce_sha256": run_nonce_sha256,
            "subprocess_receipt_sha256": _canonical_hash(receipt_core),
            "runner_log_artifact_content_sha256": json.loads(
                log.read_text(encoding="utf-8")
            )["artifact_content_sha256"],
            "runner_source_file_sha256": runner_source["file_sha256"],
            "suite_id": suite_id,
            "suite_started_at_unix_ns": suite_started_at_unix_ns,
            "suite_ended_at_unix_ns": suite_ended_at_unix_ns,
        },
        "runner_log": {"path": str(log), "file_sha256": _hash_file(log)},
    }
    unsigned = {
        "schema_version": SMOKE_EVIDENCE_SCHEMA_VERSION,
        "analysis_basis": SMOKE_EVIDENCE_ANALYSIS_BASIS,
        **evidence,
        "artifact_content_sha256": None,
    }
    digest = _canonical_hash(unsigned)
    payload = {**unsigned, "artifact_content_sha256": digest}
    validation_patch = patch(
        "crosscut.stage_edit_dynamic_crop_capacity_probe._enforce_live_release_smoke_attestation"
    )
    if passed:
        with validation_patch:
            _validate_mlx_smoke_evidence_root(
                payload, candidate=candidate, budget=BUDGET
            )
    else:
        _validate_mlx_smoke_evidence_root(
            payload, candidate=candidate, budget=BUDGET
        )
    return _atomic_exclusive_write(
        root / f"{SMOKE_EVIDENCE_PREFIX}{digest}.json", payload
    )


def _full_dynamic_validation_metrics(metric_value: float) -> dict:
    coordinate = {
        "label_count": 20,
        "mean_absolute_error": metric_value,
        "p95_absolute_error": metric_value,
    }
    return {
        "schema_version": DYNAMIC_CROP_V2_SCHEMA_VERSION,
        "model_contract": DYNAMIC_CROP_V2_MODEL_CONTRACT,
        "target_contract": DYNAMIC_CROP_V2_TARGET_CONTRACT,
        "loss_contract": DYNAMIC_CROP_V2_LOSS_CONTRACT,
        "total_loss": metric_value,
        "loss_components": {
            "base_total": metric_value,
            "base_cut": metric_value,
            "base_source_listwise": metric_value,
            "base_source_pairwise": metric_value,
            "base_duration": metric_value,
            "base_transition": metric_value,
            "base_crop": metric_value,
            "dynamic_crop_values": metric_value,
            "dynamic_crop_times": metric_value,
            "dynamic_crop_presence": metric_value,
            "dynamic_crop_motion": metric_value,
            "dynamic_crop_total": metric_value,
            "total": metric_value,
        },
        "trajectory": {
            "track_label_count": 10,
            "keyframe_label_count": 20,
            "coordinates": {
                "zoom": dict(coordinate),
                "center_x": dict(coordinate),
                "center_y": dict(coordinate),
                "roll_degrees": dict(coordinate),
            },
            "normalized_time_mae": metric_value,
            "presence": {
                "label_count": 80,
                "precision": 0.8,
                "recall": 0.8,
                "f1": 0.8,
                "keyframe_count_exact_accuracy": 0.6,
            },
            "motion": {"label_count": 10, "accuracy": 0.7},
            "renderer_decode": {
                "attempt_count": 10,
                "eligible_count": 9,
                "eligible_rate": 0.9,
                "minimum_safe_zoom_projection_count": 0,
                "rejection_reasons": {"unsafe_fixture": 1},
            },
        },
        "legacy_static_crop_head_used_for_trajectory": False,
        "production_activation_authorized": False,
    }


def _fixture_phase_learning_rate(
    phase_step: int,
    *,
    peak: float,
    total_steps: int,
    warmup_steps: int,
) -> float:
    schedule_index = phase_step - 1
    if schedule_index < warmup_steps:
        start = peak * 0.1
        return start + (peak - start) * schedule_index / warmup_steps
    decay_steps = total_steps - warmup_steps
    decay_index = schedule_index - warmup_steps
    end = peak * float(PROBE_TRAINING_SCHEDULE["minimum_learning_rate_ratio"])
    return end + 0.5 * (peak - end) * (
        1.0 + math.cos(math.pi * decay_index / decay_steps)
    )


def _fixture_finetune_learning_rate(phase_step: int) -> float:
    return _fixture_phase_learning_rate(
        phase_step,
        peak=float(PROBE_TRAINING_SCHEDULE["learning_rate"]),
        total_steps=int(PROBE_TRAINING_SCHEDULE["finetune_steps_planned"]),
        warmup_steps=int(PROBE_TRAINING_SCHEDULE["finetune_warmup_steps"]),
    )


def _fixture_source_pretraining(
    training_songs: tuple[str, ...],
    training_refs: tuple[str, ...],
) -> tuple[list[dict], dict]:
    references_by_song = {
        song_id: tuple(
            sorted(set(REFERENCES[song_id]) & set(training_refs))
        )
        for song_id in training_songs
    }
    song_visits = {song_id: 0 for song_id in training_songs}
    per_song_counts = {song_id: 0 for song_id in training_songs}
    per_reference_counts = {reference_id: 0 for reference_id in training_refs}
    reference_song_ids = {
        reference_id: song_id
        for song_id, references in references_by_song.items()
        for reference_id in references
    }
    history = []
    source_steps = int(PROBE_TRAINING_SCHEDULE["source_pretraining_steps"])
    for phase_step in range(1, source_steps + 1):
        song_id = training_songs[(phase_step - 1) % len(training_songs)]
        references = references_by_song[song_id]
        reference_id = references[song_visits[song_id] % len(references)]
        song_visits[song_id] += 1
        per_song_counts[song_id] += 1
        per_reference_counts[reference_id] += 1
        history.append(
            {
                "source_pretrain_step": phase_step,
                "optimizer_step": 500 + phase_step,
                "actual_learning_rate": _fixture_phase_learning_rate(
                    phase_step,
                    peak=float(
                        PROBE_TRAINING_SCHEDULE[
                            "source_pretraining_peak_learning_rate"
                        ]
                    ),
                    total_steps=source_steps,
                    warmup_steps=int(
                        PROBE_TRAINING_SCHEDULE[
                            "source_pretraining_warmup_steps"
                        ]
                    ),
                ),
                "song_id": song_id,
                "reference_id": reference_id,
                "group_id": f"{reference_id}:confirmed-source:1",
                "anchor_token_index": 1,
                "source_only_loss": 1.0,
                "gradient_norm_before_clip": 1.0,
                "checkpoint_selection_eligible": False,
            }
        )
    sampling = {
        "contract": "deterministic_song_reference_group_round_robin_v1",
        "opening_source_excluded_count": len(training_refs),
        "unconfirmed_source_label_excluded_count": 0,
        "confirmed_source_event_count": len(training_refs),
        "reference_song_ids": reference_song_ids,
        "per_reference_confirmed_group_counts": {
            reference_id: 1 for reference_id in training_refs
        },
        "per_song_sampled_counts": per_song_counts,
        "per_reference_sampled_counts": per_reference_counts,
    }
    return history, sampling


def _full_training_history(
    *,
    dynamic_validation_metrics: dict,
    validation_selection: dict,
) -> list[dict]:
    history = []
    for epoch in range(1, 31):
        first_step = (epoch - 1) * 4 + 1
        last_step = epoch * 4
        steps = list(range(first_step, last_step + 1))
        history.append(
            {
                "epoch": epoch,
                "optimizer_step": 1000 + last_step,
                "finetune_optimizer_step_first": first_step,
                "finetune_optimizer_step_last": last_step,
                "finetune_optimizer_steps": steps,
                "finetune_actual_learning_rates": [
                    _fixture_finetune_learning_rate(step) for step in steps
                ],
                "training_song_reference_validation_total_loss": (
                    dynamic_validation_metrics["total_loss"]
                ),
                "training_song_reference_validation_selection_score": (
                    validation_selection["score"]
                ),
                "training_song_reference_validation_selection_contract": (
                    STAGE_EDIT_VALIDATION_SELECTION_CONTRACT
                ),
                "training_song_reference_validation_dynamic_crop_v2": (
                    dynamic_validation_metrics
                ),
                "checkpoint_selection_eligible_after_burn_in": (
                    last_step >= 60
                ),
                "training_song_reference_validation_improved": epoch == 15,
            }
        )
    return history


def _fold(
    root: Path,
    candidate: DynamicCropCapacityCandidate,
    heldout_song: str,
    dataset_path: Path,
    metric_value: float,
    *,
    report_schema_version: int = PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
) -> Path:
    fold_root = root / candidate.candidate_name / heldout_song
    fold_root.mkdir(parents=True)
    weight_path = fold_root / DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME
    weight_path.write_bytes(f"weights:{candidate.candidate_id}:{heldout_song}".encode())
    weight_sha = _hash_file(weight_path)
    checkpoint_id = f"checkpoint-{candidate.candidate_name}-{heldout_song}"
    metadata = new_stage_edit_dynamic_crop_checkpoint_metadata_v2(
        checkpoint_id,
        DEFAULT_STAGE_EDIT_SCHEMA,
        candidate.model_config,
        training_steps=1060,
        weights_trained=True,
        dynamic_crop_config=DynamicCropHeadV2Config(),
    )
    metadata = replace(
        metadata,
        weight_filename=weight_path.name,
        weight_sha256=weight_sha,
    )
    metadata_path = fold_root / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata.to_dict()), encoding="utf-8")

    training_songs = tuple(sorted(set(SONGS) - {heldout_song}))
    internal_validation_refs = tuple(REFERENCES[song][0] for song in training_songs)
    training_refs = tuple(
        sorted(
            ref
            for song in training_songs
            for ref in REFERENCES[song]
            if ref not in internal_validation_refs
        )
    )
    base_effective = asdict(candidate.base_loss_contract)
    base_effective["cut_positive_weight"] = 2.0
    dynamic_effective = asdict(candidate.dynamic_crop_loss_contract)
    dynamic_effective["base"]["cut_positive_weight"] = 2.0
    source_effective = dict(base_effective)
    source_effective.update(
        {
            "source_objective": "eligible_all_hybrid",
            "cut_weight": 0.0,
            "cut_positive_weight": 1.0,
            "duration_weight": 0.0,
            "transition_weight": 0.0,
            "crop_weight": 0.0,
        }
    )
    source_pretraining_history, source_sampling = _fixture_source_pretraining(
        training_songs, training_refs
    )
    dynamic_validation_metrics = _full_dynamic_validation_metrics(metric_value)
    validation_selection = {
        "contract": STAGE_EDIT_VALIDATION_SELECTION_CONTRACT,
        "selectable": True,
        "score": 0.5,
        "weights": {
            "cut_auroc": 0.45,
            "source_mrr": 0.45,
            "bounded_inverse_total_loss": 0.10,
        },
        "components": {
            "cut_auroc": 0.5,
            "source_mrr": 0.5,
            "total_loss": metric_value,
            "bounded_inverse_total_loss": 1.0 / (1.0 + metric_value),
        },
        "formula": "fixture",
    }
    selected_validation_metrics = {
        "total_loss": metric_value,
        "dynamic_crop_v2": dynamic_validation_metrics,
    }
    freeze = {
        "schema_version": 1,
        "contract": "stage_edit_model_config_checkpoint_and_threshold_freeze_v1",
        "model_config": candidate.model_config.to_dict(),
        "model_config_fingerprint_sha256": _canonical_hash(
            candidate.model_config.to_dict()
        ),
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_sha256": weight_sha,
        "selected_training_step": 1060,
        "selected_finetune_optimizer_step": 60,
        "minimum_checkpoint_selection_finetune_steps": 60,
        "selected_after_checkpoint_burn_in": True,
        "validation_selection": validation_selection,
        "selected_validation_metrics": selected_validation_metrics,
        "training_song_ids": list(training_songs),
        "training_reference_ids": list(training_refs),
        "validation_reference_ids": list(internal_validation_refs),
        "held_out_song_ids_reserved_for_one_forward": [heldout_song],
        "heldout_metrics_present": False,
        "heldout_used_for_model_or_threshold_selection": False,
        "style_or_attitude_configuration_mutable_after_freeze": False,
    }
    freeze_path = fold_root / STAGE_EDIT_SELECTION_FREEZE_FILENAME
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    schedule = {
        "contract": PROBE_TRAINING_SCHEDULE["schedule_contract"],
        "cut_pretraining_steps": PROBE_TRAINING_SCHEDULE["cut_pretraining_steps"],
        "cut_pretraining_peak_learning_rate": PROBE_TRAINING_SCHEDULE[
            "cut_pretraining_peak_learning_rate"
        ],
        "cut_pretraining_window_length": PROBE_TRAINING_SCHEDULE[
            "cut_pretraining_window_length"
        ],
        "cut_pretraining_warmup_steps": PROBE_TRAINING_SCHEDULE[
            "cut_pretraining_warmup_steps"
        ],
        "source_pretraining_steps": PROBE_TRAINING_SCHEDULE[
            "source_pretraining_steps"
        ],
        "source_pretraining_peak_learning_rate": PROBE_TRAINING_SCHEDULE[
            "source_pretraining_peak_learning_rate"
        ],
        "source_pretraining_window_length": PROBE_TRAINING_SCHEDULE[
            "source_pretraining_window_length"
        ],
        "source_pretraining_warmup_steps": PROBE_TRAINING_SCHEDULE[
            "source_pretraining_warmup_steps"
        ],
        "finetune_warmup_steps": PROBE_TRAINING_SCHEDULE[
            "finetune_warmup_steps"
        ],
        "finetune_steps_planned": PROBE_TRAINING_SCHEDULE[
            "finetune_steps_planned"
        ],
        "minimum_learning_rate_ratio": PROBE_TRAINING_SCHEDULE[
            "minimum_learning_rate_ratio"
        ],
    }
    report = {
        "schema_version": report_schema_version,
        "run_kind": PRODUCTION_TRAINING_RUN_KIND,
        "checkpoint_scope": PRODUCTION_CHECKPOINT_SCOPE,
        "production_ready": False,
        "production_readiness_reason": "activation_not_explicitly_authorized",
        "transition_authority_contract": StageEditTransitionAuthority(
            mode=RENDERER_OWNED_TRANSITION_AUTHORITY
        ).to_dict(),
        "transition_authority_contract_sha256": StageEditTransitionAuthority(
            mode=RENDERER_OWNED_TRANSITION_AUTHORITY
        ).fingerprint,
        "model": {
            "config": candidate.model_config.to_dict(),
            "dynamic_crop_v2_enabled": True,
        },
        "split": {
            "training_song_ids": list(training_songs),
            "held_out_song_ids": [heldout_song],
            "training_reference_ids": list(training_refs),
            "validation_reference_ids": list(internal_validation_refs),
            "held_out_reference_ids": list(REFERENCES[heldout_song]),
            "held_out_used_for_model_selection": False,
        },
        "checkpoint": {
            "checkpoint_id": checkpoint_id,
            "training_steps": 1060,
            "weight_sha256": weight_sha,
            "production_activation_authorized": False,
        },
        "selection_freeze_evidence": {
            "path": str(freeze_path.resolve()),
            "filename": freeze_path.name,
            "file_sha256": _hash_file(freeze_path),
            "content_sha256": _canonical_hash(freeze),
            "written_before_heldout_forward": True,
        },
        "structured_head_coverage_gate": {"passed": True},
        "optimization": {
            "seed": PROBE_TRAINING_SCHEDULE["seed"],
            "requested_epochs": PROBE_TRAINING_SCHEDULE["requested_epochs"],
            "early_stopping_patience": PROBE_TRAINING_SCHEDULE[
                "early_stopping_patience"
            ],
            "early_stopping_patience_unit": PROBE_TRAINING_SCHEDULE[
                "early_stopping_patience_unit"
            ],
            "batch_size": PROBE_TRAINING_SCHEDULE["batch_size"],
            "learning_rate": PROBE_TRAINING_SCHEDULE["learning_rate"],
            "weight_decay": PROBE_TRAINING_SCHEDULE["weight_decay"],
            "gradient_clip_norm": PROBE_TRAINING_SCHEDULE["gradient_clip_norm"],
            "learning_rate_schedule": schedule,
            "completed_epochs": 30,
            "optimizer_steps": 1120,
            "cut_pretraining_optimizer_steps": 500,
            "source_pretraining_optimizer_steps": 500,
            "multi_head_finetuning_optimizer_steps": 120,
            "source_head_update_steps": 620,
            "minimum_finetune_optimizer_steps": 120,
            "minimum_checkpoint_selection_finetune_steps": 60,
            "finetune_optimizer_initial_step": 0,
            "finetune_optimizer_state_final_step": 120,
            "finetune_optimizer_state_reset_after_cut_pretraining": True,
            "cut_pretraining_optimizer_initial_step": 0,
            "cut_pretraining_optimizer_state_final_step": 500,
            "source_pretraining_optimizer_initial_step": 0,
            "source_pretraining_optimizer_state_final_step": 500,
            "source_pretraining_optimizer_state_reset_after_cut_pretraining": True,
            "finetune_optimizer_state_reset_after_cut_and_source_pretraining": True,
            "checkpoint_writes_before_joint_finetuning": 0,
            "finetune_steps_without_improvement_at_stop": 60,
            "early_stopping_triggered": False,
            "training_stop_reason": "planned_finetune_steps_completed",
            "finetune_schedule_completion_fraction": 1.0,
            "best_epoch": 15,
            "best_training_step": 1060,
            "best_finetune_optimizer_step": 60,
            "best_validation_selection_score": 0.5,
            "best_validation_selection_evidence": validation_selection,
            "selected_checkpoint_validation_metrics": (
                selected_validation_metrics
            ),
            "effective_loss_contract": base_effective,
            "effective_dynamic_crop_loss_contract": dynamic_effective,
            "cut_pretraining": {
                "enabled": True,
                "partition": "training_references_only",
                "balanced_anchor_positive_count_per_step": 1,
                "balanced_anchor_negative_count_per_step": 1,
                "other_head_losses_zero_weighted": True,
                "source_head_losses_zero_weighted": True,
                "checkpoint_selection_eligible": False,
                "positive_weight": 1.0,
                "minimum_step_contract": 500,
                "validation_labels_consumed": False,
                "heldout_labels_consumed": False,
            },
            "source_pretraining": {
                "enabled": True,
                "partition": "training_references_only",
                "confirmed_event_contract": (
                    "target_cut_equals_one_and_selected_source_id_v1"
                ),
                "opening_source_selection_excluded": True,
                "loss_contract": source_effective,
                "listwise_loss_enabled": True,
                "eligible_all_pairwise_loss_enabled": True,
                "other_head_losses_zero_weighted": True,
                "checkpoint_selection_eligible": False,
                "minimum_step_contract": 500,
                "validation_labels_consumed": False,
                "heldout_labels_consumed": False,
                "sampling": source_sampling,
            },
            "unseen_song_used_for_model_selection": False,
        },
        "held_out_evaluation": {
            "passed": True,
            "checkpoint_id": checkpoint_id,
            "weight_sha256": weight_sha,
            "training_song_ids": list(training_songs),
            "held_out_song_ids": [heldout_song],
            "held_out_reference_count": 5,
            "cut_threshold_frozen_before_heldout_forward": True,
            "quality_gate": {
                "passed": True,
                "thresholds": ProductionQualityThresholds().to_dict(),
            },
            "dynamic_crop_v2_metrics": dynamic_validation_metrics,
        },
        "source_pretraining_history": source_pretraining_history,
        "history": _full_training_history(
            dynamic_validation_metrics=dynamic_validation_metrics,
            validation_selection=validation_selection,
        ),
    }
    report_path = fold_root / PRODUCTION_TRAINING_REPORT_FILENAME
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return write_fold_run_evidence(
        root,
        candidate_id=candidate.candidate_id,
        validation_song_id=heldout_song,
        dataset_manifest_path=dataset_path,
        production_training_report_path=report_path.resolve(),
        selection_freeze_path=freeze_path.resolve(),
        checkpoint_metadata_path=metadata_path.resolve(),
        checkpoint_weights_path=weight_path.resolve(),
    )


def _inputs(
    root: Path,
    candidate_values: tuple[tuple[DynamicCropCapacityCandidate, tuple[float, ...]], ...],
):
    dataset_path = _dataset(root)
    folds = []
    smokes = {}
    for candidate, values in candidate_values:
        smokes[candidate.candidate_id] = _smoke(root, candidate)
        folds.extend(
            _fold(root, candidate, song, dataset_path, value)
            for song, value in zip(SONGS, values, strict=True)
        )
    return dataset_path, tuple(folds), smokes


class DynamicCropCapacityProbeTests(unittest.TestCase):
    def test_release_selection_rejects_candidate_omission_and_wrong_baseline(self) -> None:
        defaults = default_dynamic_crop_v2_probe_candidates()
        with self.assertRaisesRegex(
            DynamicCropCapacityProbeError, "exact four pinned candidates"
        ):
            _build_dynamic_crop_capacity_selection(
                candidates=defaults[:-1],
                fold_run_evidence_paths=(),
                selection_dataset_manifest_path=Path("/missing"),
                mlx_smoke_evidence_paths_by_candidate={},
                device_memory_budget=BUDGET,
                baseline_candidate_id=defaults[0].candidate_id,
            )
        with self.assertRaisesRegex(
            DynamicCropCapacityProbeError, "baseline must be compact-balanced"
        ):
            _build_dynamic_crop_capacity_selection(
                candidates=defaults,
                fold_run_evidence_paths=(),
                selection_dataset_manifest_path=Path("/missing"),
                mlx_smoke_evidence_paths_by_candidate={},
                device_memory_budget=BUDGET,
                baseline_candidate_id=defaults[1].candidate_id,
            )

    def test_fold_evaluation_has_no_public_scalar_constructor(self) -> None:
        with self.assertRaises(TypeError):
            FoldEvaluation(candidate_id="a" * 64)  # type: ignore[call-arg]

    def test_schedule_requires_exact_120_steps_and_60_step_burn_in(self) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path = _dataset(root)
            evidence_path = _fold(
                root, candidate, BANG_BANG, dataset_path, 1.0
            )
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            report = json.loads(
                Path(evidence["production_training_report"]["path"])
                .read_text(encoding="utf-8")
            )

        incomplete = json.loads(json.dumps(report))
        incomplete["optimization"][
            "multi_head_finetuning_optimizer_steps"
        ] = 119
        with self.assertRaisesRegex(
            DynamicCropCapacityProbeError, "exact 120-step"
        ):
            _validate_schedule(incomplete)

        before_burn_in = json.loads(json.dumps(report))
        before_burn_in["optimization"]["best_finetune_optimizer_step"] = 59
        before_burn_in["optimization"]["best_training_step"] = 559
        with self.assertRaisesRegex(
            DynamicCropCapacityProbeError, "60-step burn-in"
        ):
            _validate_schedule(before_burn_in)

        cut_source_leak = json.loads(json.dumps(report))
        cut_source_leak["optimization"]["cut_pretraining"][
            "source_head_losses_zero_weighted"
        ] = False
        with self.assertRaisesRegex(
            DynamicCropCapacityProbeError,
            "cut-only pretraining could influence source/checkpoint",
        ):
            _validate_schedule(cut_source_leak)

    def test_schedule_requires_actual_lr_and_full_dynamic_history(self) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path = _dataset(root)
            evidence_path = _fold(
                root, candidate, BANG_BANG, dataset_path, 1.0
            )
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            report = json.loads(
                Path(evidence["production_training_report"]["path"])
                .read_text(encoding="utf-8")
            )

        wrong_lr = json.loads(json.dumps(report))
        wrong_lr["history"][0]["finetune_actual_learning_rates"][0] *= 2.0
        with self.assertRaisesRegex(
            DynamicCropCapacityProbeError, "actual finetune learning-rate"
        ):
            _validate_schedule(wrong_lr)

        wrong_source_lr = json.loads(json.dumps(report))
        wrong_source_lr["source_pretraining_history"][0][
            "actual_learning_rate"
        ] *= 2.0
        with self.assertRaisesRegex(
            DynamicCropCapacityProbeError, "source actual learning-rate"
        ):
            _validate_schedule(wrong_source_lr)

        reused_optimizer = json.loads(json.dumps(report))
        reused_optimizer["optimization"][
            "source_pretraining_optimizer_initial_step"
        ] = 500
        with self.assertRaisesRegex(
            DynamicCropCapacityProbeError, "exact 120-step"
        ):
            _validate_schedule(reused_optimizer)

        truncated_metrics = json.loads(json.dumps(report))
        del truncated_metrics["history"][0][
            "training_song_reference_validation_dynamic_crop_v2"
        ]["trajectory"]
        with self.assertRaisesRegex(
            DynamicCropCapacityProbeError, "keys are not exact"
        ):
            _validate_schedule(truncated_metrics)

    def test_selection_freeze_metrics_must_match_selected_report_evidence(
        self,
    ) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path = _dataset(root)
            evidence_path = _fold(
                root, candidate, BANG_BANG, dataset_path, 1.0
            )
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            report_path = Path(evidence["production_training_report"]["path"])
            freeze_path = Path(evidence["selection_freeze"]["path"])
            metadata_path = Path(evidence["checkpoint_metadata"]["path"])
            weights_path = Path(evidence["checkpoint_weights"]["path"])

            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            freeze["selected_validation_metrics"]["dynamic_crop_v2"][
                "total_loss"
            ] = 2.0
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")

            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["selection_freeze_evidence"]["file_sha256"] = _hash_file(
                freeze_path
            )
            report["selection_freeze_evidence"]["content_sha256"] = (
                _canonical_hash(freeze)
            )
            report_path.write_text(json.dumps(report), encoding="utf-8")
            rebound = write_fold_run_evidence(
                root,
                candidate_id=candidate.candidate_id,
                validation_song_id=BANG_BANG,
                dataset_manifest_path=dataset_path,
                production_training_report_path=report_path,
                selection_freeze_path=freeze_path,
                checkpoint_metadata_path=metadata_path,
                checkpoint_weights_path=weights_path,
            )
            dataset = _load_selection_dataset(dataset_path)
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "selection freeze provenance"
            ):
                _load_fold_evaluation(
                    rebound, candidate=candidate, dataset=dataset
                )

    def test_incomplete_reference_dataset_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            incomplete = dict(REFERENCES)
            incomplete[BANG_BANG] = REFERENCES[BANG_BANG][:-1]
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "exactly five"
            ):
                _dataset(root, incomplete)

    def test_missing_real_mlx_smoke_fails_before_selection(self) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path = _dataset(root)
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "real MLX smoke"
            ):
                build_dynamic_crop_capacity_selection(
                    candidates=(candidate,),
                    fold_run_evidence_paths=(),
                    selection_dataset_manifest_path=dataset_path,
                    mlx_smoke_evidence_paths_by_candidate={},
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=candidate.candidate_id,
                )

    def test_synthetic_passed_smoke_cannot_use_public_writer(self) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _dataset(root)
            attested_fixture = _smoke(root, candidate)
            payload = json.loads(attested_fixture.read_text(encoding="utf-8"))
            evidence = {
                key: value
                for key, value in payload.items()
                if key
                not in {"schema_version", "analysis_basis", "artifact_content_sha256"}
            }
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "fresh-process supervisor"
            ):
                write_mlx_smoke_evidence(
                    root / "public-writer-attempt",
                    candidate=candidate,
                    budget=BUDGET,
                    evidence=evidence,
                )

    def test_private_synthetic_passed_smoke_cannot_authorize_selector(self) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path = _dataset(root)
            synthetic = _smoke(root, candidate)
            with patch(
                "crosscut.stage_edit_dynamic_crop_capacity_probe._enforce_release_capacity_candidate_set"
            ), self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "actual module source"
            ):
                _build_dynamic_crop_capacity_selection(
                    candidates=(candidate,),
                    fold_run_evidence_paths=(),
                    selection_dataset_manifest_path=dataset_path,
                    mlx_smoke_evidence_paths_by_candidate={
                        candidate.candidate_id: synthetic
                    },
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=candidate.candidate_id,
                )

            payload = json.loads(synthetic.read_text(encoding="utf-8"))
            evidence = {
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "schema_version",
                    "analysis_basis",
                    "artifact_content_sha256",
                    "supervisor_attestation",
                }
            }
            from crosscut.stage_edit_dynamic_crop_mlx_smoke_cli import (
                _code_provenance,
            )

            current_head = subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=Path(__file__).resolve().parents[2],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            actual_code, _ = _code_provenance(
                expected_git_commit=current_head,
                tracked_diff_policy="record_development",
            )
            evidence["code_provenance"] = actual_code
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "real supervisor call frame"
            ):
                _publish_supervised_mlx_smoke_evidence(
                    root / "external-private-writer-attempt",
                    candidate=candidate,
                    budget=BUDGET,
                    evidence=evidence,
                    runner_log_path=Path(evidence["runner_log"]["path"]),
                    run_nonce="d" * 64,
                )

    def test_empty_runner_log_cannot_publish_failed_smoke(self) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _dataset(root)
            valid_failed = _smoke(root, candidate, passed=False)
            payload = json.loads(valid_failed.read_text(encoding="utf-8"))
            fake_log = root / "fake-runner-log.json"
            fake_log.write_text("{}\n", encoding="utf-8")
            evidence = {
                key: value
                for key, value in payload.items()
                if key
                not in {"schema_version", "analysis_basis", "artifact_content_sha256"}
            }
            evidence["runner_log"] = {
                "path": str(fake_log.resolve()),
                "file_sha256": _hash_file(fake_log),
            }
            evidence["supervisor_attestation"] = None
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "runner log"
            ):
                write_mlx_smoke_evidence(
                    root / "fake-log-attempt",
                    candidate=candidate,
                    budget=BUDGET,
                    evidence=evidence,
                )

    def test_synthetic_failed_smoke_cannot_eliminate_candidate(self) -> None:
        failed_candidate = _candidate("failed", 64)
        baseline = _candidate("baseline", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path = _dataset(root)
            synthetic_failed = _smoke(root, failed_candidate, passed=False)
            payload = json.loads(synthetic_failed.read_text(encoding="utf-8"))
            diagnostic_evidence = {
                key: value
                for key, value in payload.items()
                if key
                not in {"schema_version", "analysis_basis", "artifact_content_sha256"}
            }
            diagnostic_evidence["supervisor_attestation"] = None
            diagnostic = write_mlx_smoke_evidence(
                root / "diagnostic-only",
                candidate=failed_candidate,
                budget=BUDGET,
                evidence=diagnostic_evidence,
            )
            baseline_smoke = _smoke(root, baseline)
            with patch(
                "crosscut.stage_edit_dynamic_crop_capacity_probe._enforce_release_capacity_candidate_set"
            ), self.assertRaisesRegex(
                DynamicCropCapacityProbeError,
                "selection-authoritative.*attestation",
            ):
                _build_dynamic_crop_capacity_selection(
                    candidates=(failed_candidate, baseline),
                    fold_run_evidence_paths=(),
                    selection_dataset_manifest_path=dataset_path,
                    mlx_smoke_evidence_paths_by_candidate={
                        failed_candidate.candidate_id: diagnostic,
                        baseline.candidate_id: baseline_smoke,
                    },
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=baseline.candidate_id,
                )

    def test_fake_report_schema_and_zero_metrics_cannot_authorize(self) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path = _dataset(root)
            bad_fold = _fold(
                root,
                candidate,
                BANG_BANG,
                dataset_path,
                1.0,
                report_schema_version=0,
            )
            other_folds = tuple(
                _fold(root, candidate, song, dataset_path, 1.0)
                for song in (BLACKHOLE, REBEL_HEART)
            )
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "production report contract"
            ):
                build_dynamic_crop_capacity_selection(
                    candidates=(candidate,),
                    fold_run_evidence_paths=(bad_fold, *other_folds),
                    selection_dataset_manifest_path=dataset_path,
                    mlx_smoke_evidence_paths_by_candidate={
                        candidate.candidate_id: _smoke(root, candidate)
                    },
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=candidate.candidate_id,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path, folds, smokes = _inputs(
                root, ((candidate, (0.0, 0.0, 0.0)),)
            )
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "no candidate passes"
            ):
                build_dynamic_crop_capacity_selection(
                    candidates=(candidate,),
                    fold_run_evidence_paths=folds,
                    selection_dataset_manifest_path=dataset_path,
                    mlx_smoke_evidence_paths_by_candidate=smokes,
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=candidate.candidate_id,
                )

    def test_catastrophic_fold_is_rejected(self) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path, folds, smokes = _inputs(
                root, ((candidate, (1.0, 1.0, 1000.0)),)
            )
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "no candidate passes"
            ):
                build_dynamic_crop_capacity_selection(
                    candidates=(candidate,),
                    fold_run_evidence_paths=folds,
                    selection_dataset_manifest_path=dataset_path,
                    mlx_smoke_evidence_paths_by_candidate=smokes,
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=candidate.candidate_id,
                )

    def test_tiny_loss_difference_prefers_smaller_model(self) -> None:
        small = _candidate("small", 32)
        large = _candidate("large", 64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path, folds, smokes = _inputs(
                root,
                (
                    (large, (1.0, 1.0, 1.0)),
                    (small, (1.0 + 1.0e-8,) * 3),
                ),
            )
            payload = build_dynamic_crop_capacity_selection(
                candidates=(large, small),
                fold_run_evidence_paths=tuple(reversed(folds)),
                selection_dataset_manifest_path=dataset_path,
                mlx_smoke_evidence_paths_by_candidate=smokes,
                device_memory_budget=BUDGET,
                baseline_candidate_id=large.candidate_id,
            )
            self.assertEqual(small.candidate_id, payload["selected_candidate_id"])

    def test_failed_large_smoke_rejects_only_large_candidate(self) -> None:
        small = _candidate("small", 32)
        large = _candidate("large", 64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path, folds, smokes = _inputs(
                root,
                (
                    (small, (1.0, 1.0, 1.0)),
                    (large, (0.9, 0.9, 0.9)),
                ),
            )
            smokes[large.candidate_id] = _smoke(root, large, passed=False)
            payload = build_dynamic_crop_capacity_selection(
                candidates=(small, large),
                # The failed-memory candidate must never consume three costly
                # whole-song fold trainings.
                fold_run_evidence_paths=folds[:3],
                selection_dataset_manifest_path=dataset_path,
                mlx_smoke_evidence_paths_by_candidate=smokes,
                device_memory_budget=BUDGET,
                baseline_candidate_id=small.candidate_id,
            )
        self.assertEqual(small.candidate_id, payload["selected_candidate_id"])
        large_row = next(
            row
            for row in payload["ranking"]
            if row["candidate_id"] == large.candidate_id
        )
        self.assertFalse(large_row["eligible"])
        self.assertIn("real_mlx_smoke_failed", large_row["rejection_reasons"])
        self.assertIsNone(large_row["mean_selection_loss"])
        self.assertIsNone(large_row["fold_selection_losses"])
        self.assertEqual(
            [small.candidate_id],
            payload["selection_protocol"]["memory_eligible_candidate_ids"],
        )

    def test_genuine_smoke_records_from_different_suites_cannot_mix(self) -> None:
        baseline = _candidate("baseline", 32)
        other = _candidate("other", 64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path, folds, smokes = _inputs(
                root,
                (
                    (baseline, (1.0, 1.0, 1.0)),
                    (other, (0.9, 0.9, 0.9)),
                ),
            )
            smokes[other.candidate_id] = _smoke(
                root, other, suite_id="f" * 64
            )
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "one supervisor suite"
            ):
                build_dynamic_crop_capacity_selection(
                    candidates=(baseline, other),
                    fold_run_evidence_paths=folds,
                    selection_dataset_manifest_path=dataset_path,
                    mlx_smoke_evidence_paths_by_candidate=smokes,
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=baseline.candidate_id,
                )

    def test_partial_oom_peak_above_last_sync_is_valid_failed_evidence(self) -> None:
        baseline = _candidate("baseline", 32)
        oom_candidate = _candidate("oom", 64)
        failure_peak = 10 * 1024**3
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path, folds, smokes = _inputs(
                root,
                (
                    (baseline, (1.0, 1.0, 1.0)),
                    (oom_candidate, (0.9, 0.9, 0.9)),
                ),
            )
            smokes[oom_candidate.candidate_id] = _smoke(
                root,
                oom_candidate,
                passed=False,
                partial_oom_peak_bytes=failure_peak,
            )
            payload = build_dynamic_crop_capacity_selection(
                candidates=(baseline, oom_candidate),
                fold_run_evidence_paths=folds[:3],
                selection_dataset_manifest_path=dataset_path,
                mlx_smoke_evidence_paths_by_candidate=smokes,
                device_memory_budget=BUDGET,
                baseline_candidate_id=baseline.candidate_id,
            )
            smoke_payload = json.loads(
                smokes[oom_candidate.candidate_id].read_text(encoding="utf-8")
            )
        self.assertEqual(baseline.candidate_id, payload["selected_candidate_id"])
        oom_row = next(
            row
            for row in payload["ranking"]
            if row["candidate_id"] == oom_candidate.candidate_id
        )
        self.assertIn("real_mlx_smoke_failed", oom_row["rejection_reasons"])
        self.assertEqual(
            failure_peak, smoke_payload["execution"]["measured_peak_bytes"]
        )
        self.assertEqual(
            2 * 1024**3,
            smoke_payload["execution"]["sync_stages"][-1][
                "peak_memory_bytes"
            ],
        )

    def test_failed_baseline_smoke_fails_before_fold_training(self) -> None:
        baseline = _candidate("baseline", 32)
        other = _candidate("other", 64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path = _dataset(root)
            smokes = {
                baseline.candidate_id: _smoke(root, baseline, passed=False),
                other.candidate_id: _smoke(root, other),
            }
            with self.assertRaisesRegex(
                DynamicCropCapacityProbeError, "baseline candidate must pass"
            ):
                build_dynamic_crop_capacity_selection(
                    candidates=(baseline, other),
                    fold_run_evidence_paths=(),
                    selection_dataset_manifest_path=dataset_path,
                    mlx_smoke_evidence_paths_by_candidate=smokes,
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=baseline.candidate_id,
                )

    def test_symlink_evidence_and_exclusive_overwrite_are_rejected(self) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path, folds, smokes = _inputs(
                root, ((candidate, (1.0, 1.0, 1.0)),)
            )
            symlink = root / "dataset-link.json"
            symlink.symlink_to(dataset_path)
            with self.assertRaisesRegex(DynamicCropCapacityProbeError, "symlink"):
                build_dynamic_crop_capacity_selection(
                    candidates=(candidate,),
                    fold_run_evidence_paths=folds,
                    selection_dataset_manifest_path=symlink,
                    mlx_smoke_evidence_paths_by_candidate=smokes,
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=candidate.candidate_id,
                )
            path = write_dynamic_crop_capacity_selection(
                root,
                candidates=(candidate,),
                fold_run_evidence_paths=folds,
                selection_dataset_manifest_path=dataset_path,
                mlx_smoke_evidence_paths_by_candidate=smokes,
                device_memory_budget=BUDGET,
                baseline_candidate_id=candidate.candidate_id,
            )
            self.assertEqual(
                path.read_bytes(),
                path.read_bytes(),
            )
            with self.assertRaises(FileExistsError):
                write_dynamic_crop_capacity_selection(
                    root,
                    candidates=(candidate,),
                    fold_run_evidence_paths=folds,
                    selection_dataset_manifest_path=dataset_path,
                    mlx_smoke_evidence_paths_by_candidate=smokes,
                    device_memory_budget=BUDGET,
                    baseline_candidate_id=candidate.candidate_id,
                )

    def test_round_trip_emits_release_projection_and_origin(self) -> None:
        candidate = _candidate("small", 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dataset_path, folds, smokes = _inputs(
                root, ((candidate, (1.0, 1.0, 1.0)),)
            )
            path = write_dynamic_crop_capacity_selection(
                root,
                candidates=(candidate,),
                fold_run_evidence_paths=folds,
                selection_dataset_manifest_path=dataset_path,
                mlx_smoke_evidence_paths_by_candidate=smokes,
                device_memory_budget=BUDGET,
                baseline_candidate_id=candidate.candidate_id,
            )
            payload = load_dynamic_crop_capacity_selection(path)
        self.assertEqual(
            {
                "contract_version",
                "model_config",
                "model_config_sha256",
                "base_loss_contract",
                "base_loss_contract_sha256",
                "dynamic_crop_loss_contract",
                "dynamic_crop_loss_contract_sha256",
                "source_objective",
                "transition_authority_contract",
                "transition_authority_contract_sha256",
            },
            set(payload["release_model_training_contract"]),
        )
        self.assertEqual(
            payload["selection_identity_sha256"],
            payload["release_model_training_contract_origin"][
                "selection_identity_sha256"
            ],
        )

    def test_default_candidates_include_intermediate_and_forbid_dynamic_config(self) -> None:
        candidates = default_dynamic_crop_v2_probe_candidates()
        self.assertEqual(4, len(candidates))
        self.assertEqual({512, 640, 768}, {c.model_config.model_dimension for c in candidates})
        with self.assertRaisesRegex(
            DynamicCropCapacityProbeError, "non-default"
        ):
            replace(candidates[0], dynamic_crop_config=DynamicCropHeadV2Config(presence_threshold=0.6))
        from crosscut.stage_edit_release_fold_launcher import (
            _validate_model_training_contract,
        )

        for candidate in candidates:
            _validate_model_training_contract(
                candidate.release_model_training_contract()
            )

    def test_memory_budget_cannot_claim_float16_batch1_or_unsafe_fraction(self) -> None:
        for kwargs in (
            {"batch_size": 1},
            {"dtype_bytes": 2},
            {"maximum_training_fraction": 0.8},
        ):
            with self.assertRaises(DynamicCropCapacityProbeError):
                DeviceMemoryBudget(
                    device_name="Apple M4 Pro",
                    physical_memory_bytes=25_769_803_776,
                    **kwargs,
                )


if __name__ == "__main__":
    unittest.main()
