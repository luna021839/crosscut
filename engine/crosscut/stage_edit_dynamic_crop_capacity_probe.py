"""Evidence-bound capacity selection for StageEdit Dynamic Crop v2.

The selector never trains and never opens XOXZ or ATTITUDE. Selection scalars
come only from immutable BANG BANG / BLACKHOLE / REBEL HEART fold-run records
which bind a production report, pre-heldout selection freeze, checkpoint
metadata, and weights. Static memory math is rejection-only; every selectable
candidate also needs a separately recorded real MLX forward/backward/optimizer
smoke run at the release shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import errno
import hashlib
import inspect
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import secrets
import stat
import subprocess
from typing import Any, Mapping, Sequence

from .stage_edit_dynamic_crop_head_v2 import (
    DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME,
    DYNAMIC_CROP_V2_LOSS_CONTRACT,
    DYNAMIC_CROP_V2_MODEL_CONTRACT,
    DYNAMIC_CROP_V2_SCHEMA_VERSION,
    DYNAMIC_CROP_V2_TARGET_CONTRACT,
    DynamicCropHeadV2Config,
    DynamicCropV2LossContract,
    StageEditDynamicCropCheckpointMetadataV2,
    stage_edit_dynamic_crop_v2_parameter_count,
)
from .stage_edit_production_training import (
    STAGE_EDIT_SELECTION_FREEZE_FILENAME,
    ProductionQualityThresholds,
)
from .stage_edit_transition_authority import (
    RENDERER_OWNED_TRANSITION_AUTHORITY,
    StageEditTransitionAuthority,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    PRODUCTION_CHECKPOINT_SCOPE,
    PRODUCTION_TRAINING_REPORT_FILENAME,
    PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
    PRODUCTION_TRAINING_RUN_KIND,
    StageEditLossContract,
    StageEditTransformerConfig,
)


CAPACITY_PROBE_SCHEMA_VERSION = 2
CAPACITY_PROBE_ANALYSIS_BASIS = "verified_dynamic_crop_v2_capacity_probe_v2"
DATASET_ANALYSIS_BASIS = "ive_fixed_capacity_selection_dataset_v1"
FOLD_EVIDENCE_ANALYSIS_BASIS = "verified_stageedit_fold_run_evidence_v1"
SMOKE_EVIDENCE_ANALYSIS_BASIS = "real_mlx_training_memory_smoke_v2"
SMOKE_EVIDENCE_SCHEMA_VERSION = 2
MEMORY_ESTIMATE_CONTRACT = "non_authoritative_float32_source720_bound_v2"
SELECTION_ARTIFACT_PREFIX = "dynamic-crop-v2-capacity-selection-"
DATASET_ARTIFACT_PREFIX = "dynamic-crop-v2-selection-dataset-"
FOLD_EVIDENCE_PREFIX = "dynamic-crop-v2-fold-run-"
SMOKE_EVIDENCE_PREFIX = "dynamic-crop-v2-mlx-smoke-"
SMOKE_INPUT_PACKING_CONTRACT = (
    "fifteen_bbr_reference_feature_snapshots_fixed_lane_ab_2x2048x32x720_v1"
)
SMOKE_TILING_CONTRACT = (
    "repeat_exact_original_lane_cycle_then_prefix_truncate_to_2048_v1"
)
SMOKE_PROCESS_ISOLATION_CONTRACT = "one_fresh_process_per_capacity_candidate_v1"
SMOKE_RUNNER_CONTRACT = (
    "dynamic_crop_v2_forward_backward_clip_adamw_peak_memory_runner_v2"
)
SMOKE_CACHE_LIMIT_BYTES = 512 * 1024**2
SMOKE_MEMORY_LIMIT_CONTRACT = (
    "mlx_limit_is_guidance_measured_peak_must_independently_fit_budget_v1"
)
SMOKE_RUNNER_LOG_SCHEMA_VERSION = 2
SMOKE_RUNNER_LOG_ANALYSIS_BASIS = "dynamic_crop_v2_mlx_smoke_process_log_v2"
SMOKE_RUNNER_LOG_PREFIX = "dynamic-crop-v2-mlx-smoke-log-"
SMOKE_SUITE_SCHEMA_VERSION = 1
SMOKE_SUITE_ANALYSIS_BASIS = "dynamic_crop_v2_exact_four_mlx_smoke_suite_v1"
SMOKE_SUITE_PREFIX = "dynamic-crop-v2-mlx-smoke-suite-"
SMOKE_WORKER_EVENT_PREFIX = "STAGEEDIT_MLX_SMOKE_EVENT="
SMOKE_SUPERVISOR_ATTESTATION_CONTRACT = (
    "fresh_popen_child_nonce_and_process_receipt_v1"
)
RUNTIME_DISTRIBUTION_VERIFICATION_CONTRACT = (
    "verify_record_sha256_for_non_bytecode_files_bind_all_installed_bytes_v1"
)

BANG_BANG = "IVE-BANG-BANG"
BLACKHOLE = "IVE-BLACKHOLE"
REBEL_HEART = "IVE-REBEL-HEART"
SELECTION_SONG_IDS = (BANG_BANG, BLACKHOLE, REBEL_HEART)
QUARANTINED_SONG_IDS = ("IVE-XOXZ", "IVE-ATTITUDE")
EXPECTED_REFERENCE_COUNT_PER_SONG = 5
EXPECTED_STYLE_ID_BY_SONG = {
    BANG_BANG: "bang-bang-human-reference-10source-v2",
    BLACKHOLE: "blackhole-human-reference-10source-v2",
    REBEL_HEART: "rebel-heart-human-reference-10source-v3-current-code",
}
EXPECTED_MLX_DEVICE = {
    "device_name": "Apple M4 Pro",
    "physical_memory_bytes": 25_769_803_776,
    "max_recommended_working_set_size": 19_069_665_280,
    "max_buffer_length": 14_302_248_960,
    "architecture": "applegpu_g16s",
    "mlx_version": "0.32.1",
}

PROBE_TRAINING_SCHEDULE: Mapping[str, Any] = {
    "seed": 20260824,
    "requested_epochs": 30,
    "early_stopping_patience": 24,
    "early_stopping_patience_unit": "finetune_optimizer_steps",
    "batch_size": 2,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-3,
    "gradient_clip_norm": 1.0,
    "cut_pretraining_steps": 500,
    "cut_pretraining_peak_learning_rate": 3.0e-4,
    "cut_pretraining_window_length": 32,
    "cut_pretraining_warmup_steps": 50,
    "source_pretraining_steps": 500,
    "source_pretraining_peak_learning_rate": 3.0e-4,
    "source_pretraining_window_length": 32,
    "source_pretraining_warmup_steps": 50,
    "finetune_warmup_steps": 8,
    "minimum_learning_rate_ratio": 0.10,
    "minimum_finetune_optimizer_steps": 120,
    "minimum_checkpoint_selection_finetune_steps": 60,
    "finetune_steps_planned": 120,
    "schedule_contract": (
        "independent_warmup_cosine_cut_source_and_joint_with_step_floor_v3"
    ),
}


class DynamicCropCapacityProbeError(ValueError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicCropCapacityProbeError(f"{label} must be an object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise DynamicCropCapacityProbeError(f"{label} keys are not exact")


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicCropCapacityProbeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise DynamicCropCapacityProbeError(f"{label} must be finite and non-negative")
    return result


def _regular_absolute(path: Path, label: str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise DynamicCropCapacityProbeError(f"{label} path must be absolute")
    if raw.is_symlink():
        raise DynamicCropCapacityProbeError(f"{label} must not be a symlink")
    try:
        mode = raw.stat().st_mode
    except OSError as exc:
        raise DynamicCropCapacityProbeError(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise DynamicCropCapacityProbeError(f"{label} must be a regular file")
    return raw.resolve()


def _pointer(path: Path, label: str) -> dict[str, str]:
    resolved = _regular_absolute(path, label)
    return {"path": str(resolved), "file_sha256": _file_sha256(resolved)}


def _verify_pointer(value: object, label: str) -> Path:
    pointer = _mapping(value, label)
    _exact_keys(pointer, {"path", "file_sha256"}, label)
    # Never even stat/open a path whose spelling identifies a quarantined song.
    _reject_quarantined_text(pointer["path"], f"{label} path")
    path = _regular_absolute(Path(str(pointer["path"])), label)
    if not _is_sha256(pointer["file_sha256"]):
        raise DynamicCropCapacityProbeError(f"{label} hash is invalid")
    if _file_sha256(path) != pointer["file_sha256"]:
        raise DynamicCropCapacityProbeError(f"{label} hash mismatch")
    return path


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DynamicCropCapacityProbeError(
            f"{label} is unreadable: {exc}"
        ) from exc
    return _mapping(value, label)


def _dynamic_crop_artifact_content_sha256(
    payload: Mapping[str, Any],
) -> str:
    normalized = dict(payload)
    provenance = _mapping(
        normalized.get("provenance"), "dynamic crop artifact provenance"
    )
    provenance["artifact_content_sha256"] = None
    normalized["provenance"] = provenance
    return _sha256(normalized)


def _reject_quarantined_text(value: object, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise DynamicCropCapacityProbeError(f"{label} cannot be blank")
    compact = "".join(character for character in text.upper() if character.isalnum())
    for song in QUARANTINED_SONG_IDS:
        song_compact = "".join(character for character in song if character.isalnum())
        title_compact = song_compact.removeprefix("IVE")
        if song_compact in compact or title_compact in compact:
            raise DynamicCropCapacityProbeError(f"{label} references a quarantined song")
    return text


def _addressed_payload(path: Path, *, prefix: str, label: str) -> dict[str, Any]:
    resolved = _regular_absolute(path, label)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DynamicCropCapacityProbeError(f"{label} is unreadable: {exc}") from exc
    root = _mapping(payload, label)
    digest = root.get("artifact_content_sha256")
    if not _is_sha256(digest):
        raise DynamicCropCapacityProbeError(f"{label} content hash is invalid")
    unsigned = dict(root)
    unsigned["artifact_content_sha256"] = None
    if _sha256(unsigned) != digest:
        raise DynamicCropCapacityProbeError(f"{label} content hash mismatch")
    if resolved.name != f"{prefix}{digest}.json":
        raise DynamicCropCapacityProbeError(f"{label} filename is not content-addressed")
    return root


def _atomic_exclusive_write(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = path.parent.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / path.name
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {final}")
    temporary = destination / f".{final.name}.tmp-{secrets.token_hex(8)}"
    encoded = _canonical_bytes(payload) + b"\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, final)
        except OSError as link_error:
            # ExFAT does not support hard links. macOS renameatx_np with
            # RENAME_EXCL preserves both atomic publication and no-overwrite.
            libc = ctypes.CDLL(None, use_errno=True)
            rename_exclusive = getattr(libc, "renameatx_np", None)
            renamed = False
            if rename_exclusive is not None:
                rename_exclusive.argtypes = (
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_uint,
                )
                rename_exclusive.restype = ctypes.c_int
                at_fdcwd = -2
                rename_excl = 0x00000004
                if rename_exclusive(
                    at_fdcwd,
                    os.fsencode(temporary),
                    at_fdcwd,
                    os.fsencode(final),
                    rename_excl,
                ) == 0:
                    renamed = True
                else:
                    error_number = ctypes.get_errno()
                    if error_number == errno.EEXIST:
                        raise FileExistsError(
                            f"refusing to overwrite immutable artifact: {final}"
                        )
                    if error_number not in {
                        errno.EINVAL,
                        errno.ENOSYS,
                        errno.ENOTSUP,
                        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
                    }:
                        raise OSError(
                            error_number, os.strerror(error_number), final
                        )
            if not renamed:
                if link_error.errno not in {
                    errno.EINVAL,
                    errno.ENOSYS,
                    errno.ENOTSUP,
                    getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
                }:
                    raise link_error
                # Last-resort ExFAT publication. O_EXCL retains the immutable
                # no-overwrite guarantee. A crash can expose a partial file,
                # but every reader verifies its content hash and therefore
                # rejects it rather than authorizing a release selection.
                final_descriptor = os.open(
                    final, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                with os.fdopen(final_descriptor, "wb", closefd=True) as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
        directory_fd = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return final


@dataclass(frozen=True, slots=True)
class SelectionObjective:
    base_weight: float = 1.0
    dynamic_value_weight: float = 1.0
    dynamic_time_weight: float = 0.5
    dynamic_presence_weight: float = 0.35
    dynamic_motion_weight: float = 0.15
    contract: str = "verified_component_fold_mean_v2"

    def __post_init__(self) -> None:
        weights = tuple(
            _finite_nonnegative(value, "selection objective weight")
            for value in (
                self.base_weight,
                self.dynamic_value_weight,
                self.dynamic_time_weight,
                self.dynamic_presence_weight,
                self.dynamic_motion_weight,
            )
        )
        if not any(value > 0.0 for value in weights):
            raise DynamicCropCapacityProbeError("selection weights are all zero")
        if self.contract != "verified_component_fold_mean_v2":
            raise DynamicCropCapacityProbeError("selection objective is unsupported")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def score(self, metrics: "_FoldMetrics") -> float:
        return (
            self.base_weight * metrics.base_loss
            + self.dynamic_value_weight * metrics.dynamic_value_loss
            + self.dynamic_time_weight * metrics.dynamic_time_loss
            + self.dynamic_presence_weight * metrics.dynamic_presence_loss
            + self.dynamic_motion_weight * metrics.dynamic_motion_loss
        )


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    minimum_fold_selection_loss: float = 1.0e-9
    maximum_fold_selection_loss: float = 100.0
    maximum_fold_variance: float = 25.0
    maximum_paired_regression: float = 0.10
    equivalence_tolerance: float = 1.0e-6
    contract: str = "absolute_worst_variance_paired_regression_then_small_v1"

    def __post_init__(self) -> None:
        for name in (
            "minimum_fold_selection_loss",
            "maximum_fold_selection_loss",
            "maximum_fold_variance",
            "maximum_paired_regression",
            "equivalence_tolerance",
        ):
            _finite_nonnegative(getattr(self, name), name)
        if self.contract != "absolute_worst_variance_paired_regression_then_small_v1":
            raise DynamicCropCapacityProbeError("selection policy is unsupported")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DynamicCropCapacityCandidate:
    candidate_name: str
    model_config: StageEditTransformerConfig
    dynamic_crop_config: DynamicCropHeadV2Config
    base_loss_contract: StageEditLossContract
    dynamic_crop_loss_contract: DynamicCropV2LossContract

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_name",
            _reject_quarantined_text(self.candidate_name, "candidate name"),
        )
        if self.model_config.token_schema_fingerprint != DEFAULT_STAGE_EDIT_SCHEMA.fingerprint:
            raise DynamicCropCapacityProbeError("candidate token schema is not current")
        if self.dynamic_crop_config != DynamicCropHeadV2Config():
            raise DynamicCropCapacityProbeError(
                "non-default DynamicCropHeadV2Config is forbidden until release consumes it"
            )
        authority = StageEditTransitionAuthority(
            mode=RENDERER_OWNED_TRANSITION_AUTHORITY
        )
        if (
            self.model_config.transition_authority != authority.mode
            or self.base_loss_contract.transition_authority != authority.mode
            or self.base_loss_contract.transition_weight != 0.0
        ):
            raise DynamicCropCapacityProbeError(
                "capacity candidates require renderer-owned transition authority"
            )
        if asdict(self.dynamic_crop_loss_contract.base) != asdict(self.base_loss_contract):
            raise DynamicCropCapacityProbeError("base loss contracts differ")

    def contract_dict(self) -> dict[str, Any]:
        model = self.model_config.to_dict()
        dynamic_config = self.dynamic_crop_config.to_dict()
        base_loss = asdict(self.base_loss_contract)
        dynamic_loss = asdict(self.dynamic_crop_loss_contract)
        return {
            "model_config": model,
            "model_config_sha256": _sha256(model),
            "dynamic_crop_config": dynamic_config,
            "dynamic_crop_config_sha256": _sha256(dynamic_config),
            "base_loss_contract": base_loss,
            "base_loss_contract_sha256": _sha256(base_loss),
            "dynamic_crop_loss_contract": dynamic_loss,
            "dynamic_crop_loss_contract_sha256": _sha256(dynamic_loss),
        }

    @property
    def candidate_id(self) -> str:
        return _sha256(self.contract_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "candidate_id": self.candidate_id,
            "contract": self.contract_dict(),
        }

    def release_model_training_contract(self) -> dict[str, Any]:
        contract = self.contract_dict()
        authority = StageEditTransitionAuthority(
            mode=RENDERER_OWNED_TRANSITION_AUTHORITY
        )
        return {
            "contract_version": 2,
            "model_config": contract["model_config"],
            "model_config_sha256": contract["model_config_sha256"],
            "base_loss_contract": contract["base_loss_contract"],
            "base_loss_contract_sha256": contract["base_loss_contract_sha256"],
            "dynamic_crop_loss_contract": contract["dynamic_crop_loss_contract"],
            "dynamic_crop_loss_contract_sha256": contract[
                "dynamic_crop_loss_contract_sha256"
            ],
            "source_objective": contract["base_loss_contract"]["source_objective"],
            "transition_authority_contract": authority.to_dict(),
            "transition_authority_contract_sha256": authority.fingerprint,
        }

    @classmethod
    def from_dict(cls, value: object) -> "DynamicCropCapacityCandidate":
        payload = _mapping(value, "candidate")
        _exact_keys(payload, {"candidate_name", "candidate_id", "contract"}, "candidate")
        contract = _mapping(payload["contract"], "candidate contract")
        base = _mapping(contract.get("base_loss_contract"), "base loss")
        dynamic = _mapping(contract.get("dynamic_crop_loss_contract"), "dynamic loss")
        dynamic_base = _mapping(dynamic.get("base"), "dynamic base loss")
        try:
            candidate = cls(
                candidate_name=str(payload["candidate_name"]),
                model_config=StageEditTransformerConfig.from_dict(
                    _mapping(contract.get("model_config"), "model config")
                ),
                dynamic_crop_config=DynamicCropHeadV2Config(
                    **_mapping(contract.get("dynamic_crop_config"), "dynamic config")
                ),
                base_loss_contract=StageEditLossContract(**base),
                dynamic_crop_loss_contract=DynamicCropV2LossContract(
                    **{**dynamic, "base": StageEditLossContract(**dynamic_base)}
                ),
            )
        except (TypeError, ValueError) as exc:
            raise DynamicCropCapacityProbeError(f"candidate is invalid: {exc}") from exc
        if candidate.to_dict() != payload:
            raise DynamicCropCapacityProbeError("candidate hashes do not match")
        return candidate


@dataclass(frozen=True, slots=True)
class DeviceMemoryBudget:
    device_name: str
    physical_memory_bytes: int
    max_recommended_working_set_size: int = 19_069_665_280
    max_buffer_length: int = 14_302_248_960
    architecture: str = "applegpu_g16s"
    mlx_version: str = "0.32.1"
    os_reserve_bytes: int = 4 * 1024**3
    maximum_training_fraction: float = 0.70
    batch_size: int = 2
    dtype_bytes: int = 4

    def __post_init__(self) -> None:
        if not str(self.device_name).strip():
            raise DynamicCropCapacityProbeError("device name cannot be blank")
        for name in (
            "physical_memory_bytes",
            "max_recommended_working_set_size",
            "max_buffer_length",
            "os_reserve_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DynamicCropCapacityProbeError(f"{name} must be a positive integer")
        if self.os_reserve_bytes >= self.physical_memory_bytes:
            raise DynamicCropCapacityProbeError("OS reserve exhausts physical memory")
        observed_device = {
            "device_name": self.device_name,
            "physical_memory_bytes": self.physical_memory_bytes,
            "max_recommended_working_set_size": self.max_recommended_working_set_size,
            "max_buffer_length": self.max_buffer_length,
            "architecture": self.architecture,
            "mlx_version": self.mlx_version,
        }
        if observed_device != EXPECTED_MLX_DEVICE:
            raise DynamicCropCapacityProbeError(
                "device budget differs from live pinned M4 Pro MLX evidence"
            )
        if self.batch_size != 2 or self.dtype_bytes != 4:
            raise DynamicCropCapacityProbeError("release memory proof requires batch=2 float32")
        if (
            isinstance(self.maximum_training_fraction, bool)
            or not isinstance(self.maximum_training_fraction, (int, float))
            or not 0.0 < float(self.maximum_training_fraction) <= 0.70
        ):
            raise DynamicCropCapacityProbeError("training fraction must be numeric and <= 0.70")

    @property
    def allowed_training_bytes(self) -> int:
        return int(
            min(
                self.physical_memory_bytes - self.os_reserve_bytes,
                self.max_recommended_working_set_size,
            )
            * float(self.maximum_training_fraction)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _memory_preflight(
    candidate: DynamicCropCapacityCandidate,
    budget: DeviceMemoryBudget,
) -> dict[str, Any]:
    config = candidate.model_config
    count = stage_edit_dynamic_crop_v2_parameter_count(config, candidate.dynamic_crop_config)
    parameter_bytes = count * 4
    trunk = (
        budget.batch_size
        * config.max_sequence_length
        * config.model_dimension
        * config.num_layers
        * 4
        * 12
    )
    raw_source = (
        budget.batch_size
        * config.max_sequence_length
        * config.max_sources
        * config.source_input_dimension
        * 4
    )
    relational_source = raw_source * 4
    source_embeddings = (
        budget.batch_size
        * config.max_sequence_length
        * config.max_sources
        * config.model_dimension
        * 4
    )
    subtotal = parameter_bytes * 4 + trunk + raw_source + 2 * relational_source + 2 * source_embeddings
    estimate = math.ceil(subtotal * 1.20)
    return {
        "contract": MEMORY_ESTIMATE_CONTRACT,
        "authoritative_for_selection": False,
        "real_mlx_smoke_required_separately": True,
        "parameter_count": count,
        "parameter_bytes": parameter_bytes,
        "trunk_activation_bytes": trunk,
        "raw_source_activation_bytes": raw_source,
        "relational_source_activation_bytes": relational_source,
        "source_embedding_activation_bytes": source_embeddings,
        "estimated_training_bytes": estimate,
        "allowed_training_bytes": budget.allowed_training_bytes,
        "passed_static_bound": estimate <= budget.allowed_training_bytes,
    }


@dataclass(frozen=True, slots=True, init=False)
class _FoldMetrics:
    base_loss: float
    dynamic_value_loss: float
    dynamic_time_loss: float
    dynamic_presence_loss: float
    dynamic_motion_loss: float

    @classmethod
    def _from_verified_components(cls, components: Mapping[str, Any]) -> "_FoldMetrics":
        value = object.__new__(cls)
        for field, key in (
            ("base_loss", "base_total"),
            ("dynamic_value_loss", "dynamic_crop_values"),
            ("dynamic_time_loss", "dynamic_crop_times"),
            ("dynamic_presence_loss", "dynamic_crop_presence"),
            ("dynamic_motion_loss", "dynamic_crop_motion"),
        ):
            object.__setattr__(value, field, _finite_nonnegative(components.get(key), key))
        return value

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True, init=False)
class FoldEvaluation:
    candidate_id: str
    validation_song_id: str
    training_song_ids: tuple[str, ...]
    validation_reference_ids: tuple[str, ...]
    training_reference_ids: tuple[str, ...]
    metrics: _FoldMetrics
    checkpoint_id: str
    checkpoint_weight_sha256: str
    report_file_sha256: str
    evidence_artifact_content_sha256: str

    @classmethod
    def _verified(cls, **kwargs: Any) -> "FoldEvaluation":
        value = object.__new__(cls)
        for name, item in kwargs.items():
            object.__setattr__(value, name, item)
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "validation_song_id": self.validation_song_id,
            "training_song_ids": list(self.training_song_ids),
            "validation_reference_ids": list(self.validation_reference_ids),
            "training_reference_ids": list(self.training_reference_ids),
            "metrics": self.metrics.to_dict(),
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_weight_sha256": self.checkpoint_weight_sha256,
            "report_file_sha256": self.report_file_sha256,
            "evidence_artifact_content_sha256": self.evidence_artifact_content_sha256,
        }


@dataclass(frozen=True, slots=True)
class _SelectionDataset:
    artifact_content_sha256: str
    reference_ids_by_song: Mapping[str, tuple[str, ...]]
    materialization_manifests_by_song: Mapping[str, Mapping[str, str]]


def _materialization_manifest_reference_ids(path: Path, song: str) -> tuple[str, ...]:
    resolved = _regular_absolute(path, f"{song} materialization manifest")
    try:
        payload = _mapping(
            json.loads(resolved.read_text(encoding="utf-8")),
            f"{song} materialization manifest",
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DynamicCropCapacityProbeError(
            f"{song} materialization manifest is unreadable: {exc}"
        ) from exc
    input_contract = _mapping(
        payload.get("input_contract"), f"{song} input contract"
    )
    inputs = input_contract.get("inputs")
    statistics = _mapping(payload.get("statistics"), f"{song} statistics")
    outputs = payload.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise DynamicCropCapacityProbeError(
            f"{song} materialization inputs/outputs must be arrays"
        )
    references = tuple(
        sorted(
            _reject_quarantined_text(
                _mapping(item, "materialization input").get("reference_id"),
                "materialization reference id",
            )
            for item in inputs
        )
    )
    if (
        input_contract.get("expected_reference_count")
        != EXPECTED_REFERENCE_COUNT_PER_SONG
        or statistics.get("reference_count") != EXPECTED_REFERENCE_COUNT_PER_SONG
        or len(inputs) != EXPECTED_REFERENCE_COUNT_PER_SONG
        or len(outputs) != EXPECTED_REFERENCE_COUNT_PER_SONG
        or len(set(references)) != EXPECTED_REFERENCE_COUNT_PER_SONG
        or input_contract.get("style_id") != EXPECTED_STYLE_ID_BY_SONG[song]
    ):
        raise DynamicCropCapacityProbeError(
            f"{song} materialization reference/style completeness failed"
        )
    output_reference_ids = {
        str(item.get("reference_id") or "").strip()
        for item in outputs
        if isinstance(item, Mapping) and item.get("reference_id") is not None
    }
    if output_reference_ids and output_reference_ids != set(references):
        raise DynamicCropCapacityProbeError(
            f"{song} materialization output references differ"
        )
    return references


def write_selection_dataset_manifest(
    destination: Path,
    *,
    manifest_paths_by_song: Mapping[str, Path],
    reference_ids_by_song: Mapping[str, Sequence[str]],
) -> Path:
    if set(manifest_paths_by_song) != set(SELECTION_SONG_IDS) or set(reference_ids_by_song) != set(SELECTION_SONG_IDS):
        raise DynamicCropCapacityProbeError("dataset must contain exactly three selection songs")
    songs: dict[str, Any] = {}
    seen: set[str] = set()
    for song in SELECTION_SONG_IDS:
        refs = tuple(sorted(_reject_quarantined_text(v, "reference id") for v in reference_ids_by_song[song]))
        if len(refs) != EXPECTED_REFERENCE_COUNT_PER_SONG or len(set(refs)) != len(refs):
            raise DynamicCropCapacityProbeError("each selection song requires exactly five references")
        if seen & set(refs):
            raise DynamicCropCapacityProbeError("reference ids overlap songs")
        manifest_references = _materialization_manifest_reference_ids(
            manifest_paths_by_song[song], song
        )
        if refs != manifest_references:
            raise DynamicCropCapacityProbeError(
                f"{song} caller references differ from bound materialization manifest"
            )
        seen.update(refs)
        songs[song] = {
            "reference_ids": list(refs),
            "source_manifest": _pointer(manifest_paths_by_song[song], f"{song} manifest"),
        }
    unsigned = {
        "schema_version": 1,
        "analysis_basis": DATASET_ANALYSIS_BASIS,
        "songs": songs,
        "quarantined_song_ids_opened": False,
        "artifact_content_sha256": None,
    }
    digest = _sha256(unsigned)
    return _atomic_exclusive_write(
        Path(destination).expanduser().resolve() / f"{DATASET_ARTIFACT_PREFIX}{digest}.json",
        {**unsigned, "artifact_content_sha256": digest},
    )


def _load_selection_dataset(path: Path) -> _SelectionDataset:
    root = _addressed_payload(path, prefix=DATASET_ARTIFACT_PREFIX, label="selection dataset")
    _exact_keys(
        root,
        {"schema_version", "analysis_basis", "songs", "quarantined_song_ids_opened", "artifact_content_sha256"},
        "selection dataset",
    )
    if root["schema_version"] != 1 or root["analysis_basis"] != DATASET_ANALYSIS_BASIS or root["quarantined_song_ids_opened"] is not False:
        raise DynamicCropCapacityProbeError("selection dataset contract mismatch")
    songs = _mapping(root["songs"], "dataset songs")
    if set(songs) != set(SELECTION_SONG_IDS):
        raise DynamicCropCapacityProbeError("dataset songs are not exact")
    references: dict[str, tuple[str, ...]] = {}
    manifest_pointers: dict[str, dict[str, str]] = {}
    globally_seen: set[str] = set()
    for song in SELECTION_SONG_IDS:
        record = _mapping(songs[song], "dataset song record")
        _exact_keys(record, {"reference_ids", "source_manifest"}, "dataset song record")
        refs = tuple(sorted(_reject_quarantined_text(v, "reference id") for v in record["reference_ids"]))
        if len(refs) != EXPECTED_REFERENCE_COUNT_PER_SONG or len(set(refs)) != len(refs):
            raise DynamicCropCapacityProbeError("dataset reference completeness failed")
        if globally_seen & set(refs):
            raise DynamicCropCapacityProbeError("dataset references overlap songs")
        globally_seen.update(refs)
        manifest_path = _verify_pointer(
            record["source_manifest"], f"{song} source manifest"
        )
        if refs != _materialization_manifest_reference_ids(manifest_path, song):
            raise DynamicCropCapacityProbeError(
                f"{song} stored references differ from materialization manifest"
            )
        references[song] = refs
        manifest_pointers[song] = dict(record["source_manifest"])
    return _SelectionDataset(
        str(root["artifact_content_sha256"]), references, manifest_pointers
    )


def write_fold_run_evidence(
    destination: Path,
    *,
    candidate_id: str,
    validation_song_id: str,
    dataset_manifest_path: Path,
    production_training_report_path: Path,
    selection_freeze_path: Path,
    checkpoint_metadata_path: Path,
    checkpoint_weights_path: Path,
) -> Path:
    dataset = _load_selection_dataset(dataset_manifest_path)
    if validation_song_id not in SELECTION_SONG_IDS or not _is_sha256(candidate_id):
        raise DynamicCropCapacityProbeError("fold identity is invalid")
    unsigned = {
        "schema_version": 1,
        "analysis_basis": FOLD_EVIDENCE_ANALYSIS_BASIS,
        "candidate_id": candidate_id,
        "validation_song_id": validation_song_id,
        "selection_dataset_content_sha256": dataset.artifact_content_sha256,
        "production_training_report": _pointer(production_training_report_path, "training report"),
        "selection_freeze": _pointer(selection_freeze_path, "selection freeze"),
        "checkpoint_metadata": _pointer(checkpoint_metadata_path, "checkpoint metadata"),
        "checkpoint_weights": _pointer(checkpoint_weights_path, "checkpoint weights"),
        "artifact_content_sha256": None,
    }
    digest = _sha256(unsigned)
    return _atomic_exclusive_write(
        Path(destination).expanduser().resolve() / f"{FOLD_EVIDENCE_PREFIX}{digest}.json",
        {**unsigned, "artifact_content_sha256": digest},
    )


def _expected_phase_learning_rate(
    phase_step: int,
    *,
    peak_learning_rate: float,
    total_steps: int,
    warmup_steps: int,
    minimum_learning_rate_ratio: float,
) -> float:
    """Mirror the pinned MLX schedule using one-indexed optimizer steps."""

    schedule_index = phase_step - 1
    if phase_step < 1 or schedule_index >= total_steps:
        raise DynamicCropCapacityProbeError(
            "phase learning-rate history step is outside the schedule"
        )
    if schedule_index < warmup_steps:
        start = peak_learning_rate * 0.1
        return start + (peak_learning_rate - start) * (
            schedule_index / warmup_steps
        )
    decay_steps = max(total_steps - warmup_steps, 1)
    decay_index = schedule_index - warmup_steps
    end = peak_learning_rate * minimum_learning_rate_ratio
    return end + 0.5 * (peak_learning_rate - end) * (
        1.0 + math.cos(math.pi * decay_index / decay_steps)
    )


def _validate_full_dynamic_validation_metrics(
    value: object,
    label: str,
) -> dict[str, Any]:
    metrics = _mapping(value, label)
    _exact_keys(
        metrics,
        {
            "schema_version",
            "target_contract",
            "model_contract",
            "loss_contract",
            "total_loss",
            "loss_components",
            "trajectory",
            "legacy_static_crop_head_used_for_trajectory",
            "production_activation_authorized",
        },
        label,
    )
    if (
        metrics.get("schema_version") != DYNAMIC_CROP_V2_SCHEMA_VERSION
        or metrics.get("model_contract") != DYNAMIC_CROP_V2_MODEL_CONTRACT
        or metrics.get("target_contract") != DYNAMIC_CROP_V2_TARGET_CONTRACT
        or metrics.get("loss_contract") != DYNAMIC_CROP_V2_LOSS_CONTRACT
        or metrics.get("legacy_static_crop_head_used_for_trajectory") is not False
        or metrics.get("production_activation_authorized") is not False
    ):
        raise DynamicCropCapacityProbeError(
            f"{label} Dynamic Crop v2 contract differs"
        )
    _finite_nonnegative(metrics.get("total_loss"), f"{label} total loss")
    loss_components = _mapping(
        metrics.get("loss_components"), f"{label} loss components"
    )
    _exact_keys(
        loss_components,
        {
            "base_total",
            "base_cut",
            "base_source_listwise",
            "base_source_pairwise",
            "base_duration",
            "base_transition",
            "base_crop",
            "dynamic_crop_values",
            "dynamic_crop_times",
            "dynamic_crop_presence",
            "dynamic_crop_motion",
            "dynamic_crop_total",
            "total",
        },
        f"{label} loss components",
    )
    for name, component in loss_components.items():
        _finite_nonnegative(component, f"{label} loss component {name}")
    trajectory = _mapping(metrics.get("trajectory"), f"{label} trajectory")
    _exact_keys(
        trajectory,
        {
            "track_label_count",
            "keyframe_label_count",
            "coordinates",
            "normalized_time_mae",
            "presence",
            "motion",
            "renderer_decode",
        },
        f"{label} trajectory",
    )
    coordinates = _mapping(
        trajectory.get("coordinates"), f"{label} trajectory coordinates"
    )
    if set(coordinates) != {"zoom", "center_x", "center_y", "roll_degrees"}:
        raise DynamicCropCapacityProbeError(
            f"{label} trajectory coordinate metrics are incomplete"
        )
    for name, coordinate in coordinates.items():
        _exact_keys(
            _mapping(coordinate, f"{label} {name} coordinate"),
            {"label_count", "mean_absolute_error", "p95_absolute_error"},
            f"{label} {name} coordinate",
        )
    _exact_keys(
        _mapping(trajectory.get("presence"), f"{label} presence"),
        {
            "label_count",
            "precision",
            "recall",
            "f1",
            "keyframe_count_exact_accuracy",
        },
        f"{label} presence",
    )
    _exact_keys(
        _mapping(trajectory.get("motion"), f"{label} motion"),
        {"label_count", "accuracy"},
        f"{label} motion",
    )
    _exact_keys(
        _mapping(
            trajectory.get("renderer_decode"), f"{label} renderer decode"
        ),
        {
            "attempt_count",
            "eligible_count",
            "eligible_rate",
            "minimum_safe_zoom_projection_count",
            "rejection_reasons",
        },
        f"{label} renderer decode",
    )
    return metrics


def _validate_source_pretraining_evidence(
    report: Mapping[str, Any],
    optimization: Mapping[str, Any],
    *,
    cut_steps: int,
    source_steps: int,
) -> None:
    source = _mapping(
        optimization.get("source_pretraining"), "source pretraining"
    )
    source_loss = _mapping(
        source.get("loss_contract"), "source-only loss contract"
    )
    if (
        source.get("enabled") is not True
        or source.get("partition") != "training_references_only"
        or source.get("confirmed_event_contract")
        != "target_cut_equals_one_and_selected_source_id_v1"
        or source.get("opening_source_selection_excluded") is not True
        or source.get("listwise_loss_enabled") is not True
        or source.get("eligible_all_pairwise_loss_enabled") is not True
        or source.get("other_head_losses_zero_weighted") is not True
        or source.get("checkpoint_selection_eligible") is not False
        or source.get("minimum_step_contract") != 500
        or source.get("validation_labels_consumed") is not False
        or source.get("heldout_labels_consumed") is not False
        or source_loss.get("source_objective") != "eligible_all_hybrid"
        or source_loss.get("cut_weight") != 0.0
        or source_loss.get("duration_weight") != 0.0
        or source_loss.get("transition_weight") != 0.0
        or source_loss.get("crop_weight") != 0.0
        or _finite_nonnegative(
            source_loss.get("source_listwise_weight"),
            "source-only listwise weight",
        )
        <= 0.0
        or _finite_nonnegative(
            source_loss.get("source_pairwise_weight"),
            "source-only pairwise weight",
        )
        <= 0.0
    ):
        raise DynamicCropCapacityProbeError(
            "source-only warmup loss/selection contract differs"
        )

    sampling = _mapping(source.get("sampling"), "source sampling evidence")
    split = _mapping(report.get("split"), "report split")
    training_references = set(split.get("training_reference_ids") or ())
    training_songs = set(split.get("training_song_ids") or ())
    reference_song_ids = _mapping(
        sampling.get("reference_song_ids"), "source reference-song ids"
    )
    group_counts = _mapping(
        sampling.get("per_reference_confirmed_group_counts"),
        "source confirmed group counts",
    )
    per_song_counts = _mapping(
        sampling.get("per_song_sampled_counts"),
        "source per-song sampled counts",
    )
    per_reference_counts = _mapping(
        sampling.get("per_reference_sampled_counts"),
        "source per-reference sampled counts",
    )
    if (
        sampling.get("contract")
        != "deterministic_song_reference_group_round_robin_v1"
        or set(reference_song_ids) != training_references
        or set(group_counts) != training_references
        or set(per_reference_counts) != training_references
        or set(per_song_counts) != training_songs
        or set(reference_song_ids.values()) != training_songs
    ):
        raise DynamicCropCapacityProbeError(
            "source hierarchical sampling identity differs"
        )
    for value, label in (
        (
            sampling.get("opening_source_excluded_count"),
            "opening source exclusion count",
        ),
        (
            sampling.get("unconfirmed_source_label_excluded_count"),
            "unconfirmed source exclusion count",
        ),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DynamicCropCapacityProbeError(f"{label} is invalid")
    confirmed_count = sampling.get("confirmed_source_event_count")
    if (
        isinstance(confirmed_count, bool)
        or not isinstance(confirmed_count, int)
        or confirmed_count < 1
        or any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            for count in group_counts.values()
        )
        or sum(group_counts.values()) != confirmed_count
        or any(song_id not in training_songs for song_id in reference_song_ids.values())
    ):
        raise DynamicCropCapacityProbeError(
            "source confirmed editorial group evidence differs"
        )
    for counts, label in (
        (per_song_counts, "source per-song sampled counts"),
        (per_reference_counts, "source per-reference sampled counts"),
    ):
        if any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            for count in counts.values()
        ) or sum(counts.values()) != source_steps:
            raise DynamicCropCapacityProbeError(f"{label} are invalid")
    if max(per_song_counts.values()) - min(per_song_counts.values()) > 1:
        raise DynamicCropCapacityProbeError(
            "source song sampling is not balanced"
        )
    for song_id in training_songs:
        reference_counts = [
            per_reference_counts[reference_id]
            for reference_id, observed_song_id in reference_song_ids.items()
            if observed_song_id == song_id
        ]
        if (
            not reference_counts
            or max(reference_counts) - min(reference_counts) > 1
            or sum(reference_counts) != per_song_counts[song_id]
        ):
            raise DynamicCropCapacityProbeError(
                "source reference sampling is not balanced within song"
            )

    history = report.get("source_pretraining_history")
    if not isinstance(history, list) or len(history) != source_steps:
        raise DynamicCropCapacityProbeError(
            "source pretraining history does not cover all 500 steps"
        )
    observed_song_counts = {song_id: 0 for song_id in training_songs}
    observed_reference_counts = {
        reference_id: 0 for reference_id in training_references
    }
    for phase_step, raw_record in enumerate(history, start=1):
        record = _mapping(raw_record, f"source pretraining step {phase_step}")
        song_id = record.get("song_id")
        reference_id = record.get("reference_id")
        if (
            record.get("source_pretrain_step") != phase_step
            or record.get("optimizer_step") != cut_steps + phase_step
            or record.get("checkpoint_selection_eligible") is not False
            or not isinstance(song_id, str)
            or song_id not in training_songs
            or not isinstance(reference_id, str)
            or reference_id not in training_references
            or reference_song_ids.get(reference_id) != song_id
            or not isinstance(record.get("group_id"), str)
            or not record.get("group_id")
            or isinstance(record.get("anchor_token_index"), bool)
            or not isinstance(record.get("anchor_token_index"), int)
            or record.get("anchor_token_index") < 1
        ):
            raise DynamicCropCapacityProbeError(
                "source pretraining phase-local history is incomplete"
            )
        observed_song_counts[song_id] += 1
        observed_reference_counts[reference_id] += 1
        actual_rate = _finite_nonnegative(
            record.get("actual_learning_rate"),
            f"source actual learning rate at step {phase_step}",
        )
        expected_rate = _expected_phase_learning_rate(
            phase_step,
            peak_learning_rate=float(
                PROBE_TRAINING_SCHEDULE[
                    "source_pretraining_peak_learning_rate"
                ]
            ),
            total_steps=source_steps,
            warmup_steps=int(
                PROBE_TRAINING_SCHEDULE["source_pretraining_warmup_steps"]
            ),
            minimum_learning_rate_ratio=float(
                PROBE_TRAINING_SCHEDULE["minimum_learning_rate_ratio"]
            ),
        )
        if not math.isclose(
            actual_rate, expected_rate, rel_tol=1.0e-5, abs_tol=1.0e-12
        ):
            raise DynamicCropCapacityProbeError(
                "source actual learning-rate history differs"
            )
    if (
        observed_song_counts != per_song_counts
        or observed_reference_counts != per_reference_counts
    ):
        raise DynamicCropCapacityProbeError(
            "source sampling counts differ from phase history"
        )


def _validate_schedule(report: Mapping[str, Any]) -> None:
    optimization = _mapping(report.get("optimization"), "optimization")
    schedule = _mapping(optimization.get("learning_rate_schedule"), "learning-rate schedule")
    observed = {
        "seed": optimization.get("seed"),
        "requested_epochs": optimization.get("requested_epochs"),
        "early_stopping_patience": optimization.get("early_stopping_patience"),
        "early_stopping_patience_unit": optimization.get(
            "early_stopping_patience_unit"
        ),
        "batch_size": optimization.get("batch_size"),
        "learning_rate": optimization.get("learning_rate"),
        "weight_decay": optimization.get("weight_decay"),
        "gradient_clip_norm": optimization.get("gradient_clip_norm"),
        "cut_pretraining_steps": schedule.get("cut_pretraining_steps"),
        "cut_pretraining_peak_learning_rate": schedule.get("cut_pretraining_peak_learning_rate"),
        "cut_pretraining_window_length": schedule.get("cut_pretraining_window_length"),
        "cut_pretraining_warmup_steps": schedule.get("cut_pretraining_warmup_steps"),
        "source_pretraining_steps": schedule.get("source_pretraining_steps"),
        "source_pretraining_peak_learning_rate": schedule.get(
            "source_pretraining_peak_learning_rate"
        ),
        "source_pretraining_window_length": schedule.get(
            "source_pretraining_window_length"
        ),
        "source_pretraining_warmup_steps": schedule.get(
            "source_pretraining_warmup_steps"
        ),
        "finetune_warmup_steps": schedule.get("finetune_warmup_steps"),
        "minimum_learning_rate_ratio": schedule.get("minimum_learning_rate_ratio"),
        "minimum_finetune_optimizer_steps": optimization.get(
            "minimum_finetune_optimizer_steps"
        ),
        "minimum_checkpoint_selection_finetune_steps": optimization.get(
            "minimum_checkpoint_selection_finetune_steps"
        ),
        "finetune_steps_planned": schedule.get("finetune_steps_planned"),
        "schedule_contract": schedule.get("contract"),
    }
    if observed != dict(PROBE_TRAINING_SCHEDULE):
        raise DynamicCropCapacityProbeError("fold training seed/schedule differs")

    cut_steps = int(PROBE_TRAINING_SCHEDULE["cut_pretraining_steps"])
    source_steps = int(PROBE_TRAINING_SCHEDULE["source_pretraining_steps"])
    finetune_steps = int(PROBE_TRAINING_SCHEDULE["finetune_steps_planned"])
    cut_pretraining = _mapping(
        optimization.get("cut_pretraining"), "cut pretraining"
    )
    if (
        cut_pretraining.get("enabled") is not True
        or cut_pretraining.get("partition") != "training_references_only"
        or cut_pretraining.get("balanced_anchor_positive_count_per_step") != 1
        or cut_pretraining.get("balanced_anchor_negative_count_per_step") != 1
        or cut_pretraining.get("other_head_losses_zero_weighted") is not True
        or cut_pretraining.get("source_head_losses_zero_weighted") is not True
        or cut_pretraining.get("checkpoint_selection_eligible") is not False
        or cut_pretraining.get("positive_weight") != 1.0
        or cut_pretraining.get("minimum_step_contract") != 500
        or cut_pretraining.get("validation_labels_consumed") is not False
        or cut_pretraining.get("heldout_labels_consumed") is not False
    ):
        raise DynamicCropCapacityProbeError(
            "cut-only pretraining could influence source/checkpoint selection"
        )
    _validate_source_pretraining_evidence(
        report,
        optimization,
        cut_steps=cut_steps,
        source_steps=source_steps,
    )
    epochs = int(PROBE_TRAINING_SCHEDULE["requested_epochs"])
    burn_in = int(
        PROBE_TRAINING_SCHEDULE[
            "minimum_checkpoint_selection_finetune_steps"
        ]
    )
    if finetune_steps % epochs:
        raise DynamicCropCapacityProbeError(
            "pinned finetune steps are not epoch divisible"
        )
    steps_per_epoch = finetune_steps // epochs
    if (
        optimization.get("completed_epochs") != epochs
        or optimization.get("cut_pretraining_optimizer_steps") != cut_steps
        or optimization.get("source_pretraining_optimizer_steps")
        != source_steps
        or optimization.get("multi_head_finetuning_optimizer_steps")
        != finetune_steps
        or optimization.get("source_head_update_steps")
        != source_steps + finetune_steps
        or optimization.get("optimizer_steps")
        != cut_steps + source_steps + finetune_steps
        or optimization.get("cut_pretraining_optimizer_initial_step") != 0
        or optimization.get("cut_pretraining_optimizer_state_final_step")
        != cut_steps
        or optimization.get("source_pretraining_optimizer_initial_step") != 0
        or optimization.get("source_pretraining_optimizer_state_final_step")
        != source_steps
        or optimization.get(
            "source_pretraining_optimizer_state_reset_after_cut_pretraining"
        )
        is not True
        or optimization.get("finetune_optimizer_initial_step") != 0
        or optimization.get("finetune_optimizer_state_final_step")
        != finetune_steps
        or optimization.get(
            "finetune_optimizer_state_reset_after_cut_pretraining"
        )
        is not True
        or optimization.get(
            "finetune_optimizer_state_reset_after_cut_and_source_pretraining"
        )
        is not True
        or optimization.get("checkpoint_writes_before_joint_finetuning") != 0
        or optimization.get("early_stopping_triggered") is not False
        or optimization.get("training_stop_reason")
        != "planned_finetune_steps_completed"
        or optimization.get("finetune_schedule_completion_fraction") != 1.0
    ):
        raise DynamicCropCapacityProbeError(
            "fold did not complete the exact 120-step multi-head schedule"
        )

    history = report.get("history")
    if not isinstance(history, list) or len(history) != epochs:
        raise DynamicCropCapacityProbeError(
            "fold training history does not cover every scheduled epoch"
        )
    for epoch, raw_record in enumerate(history, start=1):
        record = _mapping(raw_record, f"training history epoch {epoch}")
        first_step = (epoch - 1) * steps_per_epoch + 1
        last_step = epoch * steps_per_epoch
        expected_steps = list(range(first_step, last_step + 1))
        actual_steps = record.get("finetune_optimizer_steps")
        actual_rates = record.get("finetune_actual_learning_rates")
        if (
            record.get("epoch") != epoch
            or record.get("optimizer_step")
            != cut_steps + source_steps + last_step
            or record.get("finetune_optimizer_step_first") != first_step
            or record.get("finetune_optimizer_step_last") != last_step
            or actual_steps != expected_steps
            or not isinstance(actual_rates, list)
            or len(actual_rates) != steps_per_epoch
            or record.get("checkpoint_selection_eligible_after_burn_in")
            is not (last_step >= burn_in)
        ):
            raise DynamicCropCapacityProbeError(
                "fold phase-local optimizer-step history is incomplete"
            )
        for phase_step, raw_rate in zip(
            expected_steps, actual_rates, strict=True
        ):
            actual_rate = _finite_nonnegative(
                raw_rate, f"finetune actual learning rate at step {phase_step}"
            )
            expected_rate = _expected_phase_learning_rate(
                phase_step,
                peak_learning_rate=float(
                    PROBE_TRAINING_SCHEDULE["learning_rate"]
                ),
                total_steps=finetune_steps,
                warmup_steps=int(
                    PROBE_TRAINING_SCHEDULE["finetune_warmup_steps"]
                ),
                minimum_learning_rate_ratio=float(
                    PROBE_TRAINING_SCHEDULE["minimum_learning_rate_ratio"]
                ),
            )
            if not math.isclose(
                actual_rate, expected_rate, rel_tol=1.0e-5, abs_tol=1.0e-12
            ):
                raise DynamicCropCapacityProbeError(
                    "fold actual finetune learning-rate history differs"
                )
        dynamic_validation = _validate_full_dynamic_validation_metrics(
            record.get(
                "training_song_reference_validation_dynamic_crop_v2"
            ),
            f"training history epoch {epoch} dynamic validation",
        )
        if record.get(
            "training_song_reference_validation_total_loss"
        ) != dynamic_validation.get("total_loss"):
            raise DynamicCropCapacityProbeError(
                "checkpoint score history did not consume Dynamic Crop total loss"
            )

    best_epoch = optimization.get("best_epoch")
    best_finetune_step = optimization.get("best_finetune_optimizer_step")
    best_training_step = optimization.get("best_training_step")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or isinstance(best_finetune_step, bool)
        or not isinstance(best_finetune_step, int)
        or best_finetune_step < burn_in
        or best_finetune_step > finetune_steps
        or best_finetune_step != best_epoch * steps_per_epoch
        or best_training_step
        != cut_steps + source_steps + best_finetune_step
    ):
        raise DynamicCropCapacityProbeError(
            "selected checkpoint is outside the 60-step burn-in envelope"
        )
    best_record = _mapping(history[best_epoch - 1], "selected history record")
    if best_record.get("training_song_reference_validation_improved") is not True:
        raise DynamicCropCapacityProbeError(
            "selected checkpoint history record is not marked improved"
        )
    selected_metrics = _mapping(
        optimization.get("selected_checkpoint_validation_metrics"),
        "selected checkpoint validation metrics",
    )
    selected_dynamic = _validate_full_dynamic_validation_metrics(
        selected_metrics.get("dynamic_crop_v2"),
        "selected checkpoint dynamic validation",
    )
    if (
        selected_dynamic
        != best_record.get(
            "training_song_reference_validation_dynamic_crop_v2"
        )
        or selected_metrics.get("total_loss")
        != selected_dynamic.get("total_loss")
    ):
        raise DynamicCropCapacityProbeError(
            "selected dynamic validation metrics differ from checkpoint history"
        )
    best_selection = _mapping(
        optimization.get("best_validation_selection_evidence"),
        "best validation selection evidence",
    )
    if (
        best_selection.get("score")
        != best_record.get(
            "training_song_reference_validation_selection_score"
        )
        or best_selection.get("contract")
        != best_record.get(
            "training_song_reference_validation_selection_contract"
        )
        or optimization.get("best_validation_selection_score")
        != best_selection.get("score")
        or _mapping(
            best_selection.get("components"),
            "best validation selection components",
        ).get("total_loss")
        != selected_dynamic.get("total_loss")
    ):
        raise DynamicCropCapacityProbeError(
            "selected checkpoint score evidence differs from history"
        )


def _validate_effective_losses(report: Mapping[str, Any], candidate: DynamicCropCapacityCandidate) -> None:
    optimization = _mapping(report.get("optimization"), "optimization")
    observed_base = _mapping(optimization.get("effective_loss_contract"), "effective base loss")
    expected_base = asdict(candidate.base_loss_contract)
    applied = observed_base.get("cut_positive_weight")
    expected_base["cut_positive_weight"] = applied
    if observed_base != expected_base:
        raise DynamicCropCapacityProbeError("effective base loss/source objective differs")
    observed_dynamic = _mapping(optimization.get("effective_dynamic_crop_loss_contract"), "effective dynamic loss")
    expected_dynamic = asdict(candidate.dynamic_crop_loss_contract)
    expected_dynamic["base"]["cut_positive_weight"] = applied
    if observed_dynamic != expected_dynamic:
        raise DynamicCropCapacityProbeError("effective dynamic loss differs")


def _load_fold_evaluation(
    evidence_path: Path,
    *,
    candidate: DynamicCropCapacityCandidate,
    dataset: _SelectionDataset,
) -> FoldEvaluation:
    evidence = _addressed_payload(evidence_path, prefix=FOLD_EVIDENCE_PREFIX, label="fold run evidence")
    _exact_keys(
        evidence,
        {
            "schema_version", "analysis_basis", "candidate_id", "validation_song_id",
            "selection_dataset_content_sha256", "production_training_report",
            "selection_freeze", "checkpoint_metadata", "checkpoint_weights",
            "artifact_content_sha256",
        },
        "fold run evidence",
    )
    if (
        evidence["schema_version"] != 1
        or evidence["analysis_basis"] != FOLD_EVIDENCE_ANALYSIS_BASIS
        or evidence["candidate_id"] != candidate.candidate_id
        or evidence["selection_dataset_content_sha256"] != dataset.artifact_content_sha256
    ):
        raise DynamicCropCapacityProbeError("fold run identity mismatch")
    heldout_song = str(evidence["validation_song_id"])
    if heldout_song not in SELECTION_SONG_IDS:
        raise DynamicCropCapacityProbeError("fold validation song is invalid")
    training_songs = tuple(sorted(set(SELECTION_SONG_IDS) - {heldout_song}))

    report_path = _verify_pointer(evidence["production_training_report"], "training report")
    freeze_path = _verify_pointer(evidence["selection_freeze"], "selection freeze")
    metadata_path = _verify_pointer(evidence["checkpoint_metadata"], "checkpoint metadata")
    weights_path = _verify_pointer(evidence["checkpoint_weights"], "checkpoint weights")
    checkpoint_root = report_path.parent
    if (
        report_path.name != PRODUCTION_TRAINING_REPORT_FILENAME
        or freeze_path != checkpoint_root / STAGE_EDIT_SELECTION_FREEZE_FILENAME
        or metadata_path != checkpoint_root / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME
        or weights_path.parent != checkpoint_root
    ):
        raise DynamicCropCapacityProbeError("fold checkpoint paths are not canonical")
    try:
        report = _mapping(json.loads(report_path.read_text()), "training report")
        freeze = _mapping(json.loads(freeze_path.read_text()), "selection freeze")
        metadata = StageEditDynamicCropCheckpointMetadataV2.from_dict(
            _mapping(json.loads(metadata_path.read_text()), "checkpoint metadata")
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DynamicCropCapacityProbeError(f"fold checkpoint evidence is invalid: {exc}") from exc
    if (
        report.get("schema_version") != PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION
        or report.get("run_kind") != PRODUCTION_TRAINING_RUN_KIND
        or report.get("checkpoint_scope") != PRODUCTION_CHECKPOINT_SCOPE
        or report.get("production_ready") is not False
        or report.get("production_readiness_reason") != "activation_not_explicitly_authorized"
    ):
        raise DynamicCropCapacityProbeError("production report contract mismatch")
    authority = StageEditTransitionAuthority(
        mode=RENDERER_OWNED_TRANSITION_AUTHORITY
    )
    if (
        report.get("transition_authority_contract") != authority.to_dict()
        or report.get("transition_authority_contract_sha256")
        != authority.fingerprint
    ):
        raise DynamicCropCapacityProbeError(
            "production report transition authority differs"
        )
    model = _mapping(report.get("model"), "report model")
    if (
        model.get("config") != candidate.model_config.to_dict()
        or model.get("dynamic_crop_v2_enabled") is not True
        or metadata.base_model_config != candidate.model_config
        or metadata.dynamic_crop_config != DynamicCropHeadV2Config()
        or not metadata.weights_trained
        or metadata.training_steps < 1
        or weights_path.name != DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME
        or metadata.weight_filename != weights_path.name
        or metadata.weight_sha256 != _file_sha256(weights_path)
    ):
        raise DynamicCropCapacityProbeError("fold model/checkpoint contract differs")

    split = _mapping(report.get("split"), "report split")
    train_refs = tuple(sorted(split.get("training_reference_ids") or ()))
    internal_validation_refs = tuple(sorted(split.get("validation_reference_ids") or ()))
    heldout_refs = tuple(sorted(split.get("held_out_reference_ids") or ()))
    complete_training_refs = tuple(
        sorted(ref for song in training_songs for ref in dataset.reference_ids_by_song[song])
    )
    if (
        tuple(split.get("training_song_ids") or ()) != training_songs
        or split.get("held_out_song_ids") != [heldout_song]
        or split.get("held_out_used_for_model_selection") is not False
        or set(train_refs) & set(internal_validation_refs)
        or tuple(sorted((*train_refs, *internal_validation_refs))) != complete_training_refs
        or heldout_refs != dataset.reference_ids_by_song[heldout_song]
    ):
        raise DynamicCropCapacityProbeError("whole-song fold/reference completeness failed")

    optimization = _mapping(report.get("optimization"), "optimization")
    _validate_schedule(report)
    checkpoint = _mapping(report.get("checkpoint"), "report checkpoint")
    heldout = _mapping(report.get("held_out_evaluation"), "heldout evaluation")
    quality = _mapping(heldout.get("quality_gate"), "heldout quality gate")
    coverage = _mapping(report.get("structured_head_coverage_gate"), "structured coverage gate")
    if (
        checkpoint.get("checkpoint_id") != metadata.checkpoint_id
        or checkpoint.get("training_steps") != metadata.training_steps
        or checkpoint.get("weight_sha256") != metadata.weight_sha256
        or checkpoint.get("production_activation_authorized") is not False
        or heldout.get("checkpoint_id") != metadata.checkpoint_id
        or heldout.get("weight_sha256") != metadata.weight_sha256
        or heldout.get("training_song_ids") != list(training_songs)
        or heldout.get("held_out_song_ids") != [heldout_song]
        or heldout.get("held_out_reference_count") != EXPECTED_REFERENCE_COUNT_PER_SONG
        or heldout.get("cut_threshold_frozen_before_heldout_forward") is not True
        or heldout.get("passed") is not True
        or quality.get("passed") is not True
        or quality.get("thresholds") != ProductionQualityThresholds().to_dict()
        or coverage.get("passed") is not True
    ):
        raise DynamicCropCapacityProbeError("fold heldout/quality gate failed")
    if metadata.training_steps != optimization.get("best_training_step"):
        raise DynamicCropCapacityProbeError(
            "saved checkpoint training step differs from selection evidence"
        )

    report_freeze = _mapping(report.get("selection_freeze_evidence"), "freeze evidence")
    if (
        report_freeze.get("path") != str(freeze_path)
        or report_freeze.get("filename") != freeze_path.name
        or report_freeze.get("file_sha256") != _file_sha256(freeze_path)
        or report_freeze.get("content_sha256") != _sha256(freeze)
        or report_freeze.get("written_before_heldout_forward") is not True
        or freeze.get("schema_version") != 1
        or freeze.get("contract")
        != "stage_edit_model_config_checkpoint_and_threshold_freeze_v1"
        or freeze.get("model_config") != candidate.model_config.to_dict()
        or freeze.get("model_config_fingerprint_sha256") != _sha256(candidate.model_config.to_dict())
        or freeze.get("checkpoint_id") != metadata.checkpoint_id
        or freeze.get("checkpoint_weight_sha256") != metadata.weight_sha256
        or freeze.get("selected_training_step")
        != optimization.get("best_training_step")
        or freeze.get("selected_finetune_optimizer_step")
        != optimization.get("best_finetune_optimizer_step")
        or freeze.get("minimum_checkpoint_selection_finetune_steps")
        != PROBE_TRAINING_SCHEDULE[
            "minimum_checkpoint_selection_finetune_steps"
        ]
        or freeze.get("selected_after_checkpoint_burn_in") is not True
        or freeze.get("validation_selection")
        != optimization.get("best_validation_selection_evidence")
        or freeze.get("selected_validation_metrics")
        != optimization.get("selected_checkpoint_validation_metrics")
        or freeze.get("training_song_ids") != list(training_songs)
        or tuple(sorted(freeze.get("training_reference_ids") or ())) != train_refs
        or tuple(sorted(freeze.get("validation_reference_ids") or ())) != internal_validation_refs
        or freeze.get("held_out_song_ids_reserved_for_one_forward") != [heldout_song]
        or freeze.get("heldout_metrics_present") is not False
        or freeze.get("heldout_used_for_model_or_threshold_selection") is not False
        or freeze.get("style_or_attitude_configuration_mutable_after_freeze") is not False
    ):
        raise DynamicCropCapacityProbeError("selection freeze provenance failed")
    _validate_effective_losses(report, candidate)
    if optimization.get("unseen_song_used_for_model_selection") is not False:
        raise DynamicCropCapacityProbeError("heldout influenced model selection")
    dynamic_metrics = _mapping(heldout.get("dynamic_crop_v2_metrics"), "dynamic heldout metrics")
    if (
        dynamic_metrics.get("schema_version") != DYNAMIC_CROP_V2_SCHEMA_VERSION
        or dynamic_metrics.get("model_contract") != DYNAMIC_CROP_V2_MODEL_CONTRACT
        or dynamic_metrics.get("target_contract") != DYNAMIC_CROP_V2_TARGET_CONTRACT
        or dynamic_metrics.get("loss_contract") != DYNAMIC_CROP_V2_LOSS_CONTRACT
    ):
        raise DynamicCropCapacityProbeError("heldout evaluator contract differs")
    metrics = _FoldMetrics._from_verified_components(
        _mapping(dynamic_metrics.get("loss_components"), "heldout loss components")
    )
    return FoldEvaluation._verified(
        candidate_id=candidate.candidate_id,
        validation_song_id=heldout_song,
        training_song_ids=training_songs,
        validation_reference_ids=heldout_refs,
        training_reference_ids=complete_training_refs,
        metrics=metrics,
        checkpoint_id=metadata.checkpoint_id,
        checkpoint_weight_sha256=str(metadata.weight_sha256),
        report_file_sha256=_file_sha256(report_path),
        evidence_artifact_content_sha256=str(evidence["artifact_content_sha256"]),
    )


def _nullable_finite_nonnegative(value: object, label: str) -> float | None:
    if value is None:
        return None
    return _finite_nonnegative(value, label)


def _smoke_subprocess_receipt_core(log: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": log.get("candidate_id"),
        "command": log.get("command"),
        "supervisor_pid": log.get("supervisor_pid"),
        "worker_pid": log.get("worker_pid"),
        "returncode": log.get("returncode"),
        "stdout_sha256": hashlib.sha256(
            str(log.get("stdout", "")).encode("utf-8")
        ).hexdigest(),
        "stderr_sha256": hashlib.sha256(
            str(log.get("stderr", "")).encode("utf-8")
        ).hexdigest(),
        "run_nonce_sha256": log.get("run_nonce_sha256"),
        "started_at_unix_ns": log.get("started_at_unix_ns"),
        "ended_at_unix_ns": log.get("ended_at_unix_ns"),
        "suite_id": log.get("suite_id"),
        "suite_started_at_unix_ns": log.get("suite_started_at_unix_ns"),
        "suite_ended_at_unix_ns": log.get("suite_ended_at_unix_ns"),
    }


def _expected_smoke_source_paths() -> dict[str, Path]:
    module_root = Path(__file__).resolve().parent
    return {
        "runner": module_root / "stage_edit_dynamic_crop_mlx_smoke_cli.py",
        "capacity_probe": Path(__file__).resolve(),
        "dynamic_crop_head": module_root / "stage_edit_dynamic_crop_head_v2.py",
        "transformer": module_root / "stage_edit_transformer.py",
        "feature_snapshot_loader": module_root / "stage_edit_feature_materializer.py",
        "feature_snapshot_manifest_contract": module_root
        / "stage_edit_feature_snapshot_job.py",
        "reference_adapter": module_root / "stage_edit_reference_adapter.py",
        "dynamic_crop_training": module_root / "stage_edit_dynamic_crop_training.py",
        "production_training": module_root / "stage_edit_production_training.py",
        "dynamic_crop_manifest_validator": module_root / "reference_dynamic_crop_job.py",
        "dynamic_crop_supervision_loader": module_root
        / "reference_dynamic_crop_supervision.py",
    }


def _enforce_actual_smoke_source_paths(
    source_files: Mapping[str, Any], *, repo: Path
) -> None:
    for role, expected_path in _expected_smoke_source_paths().items():
        source = _mapping(source_files.get(role), f"smoke {role} source")
        expected_resolved = expected_path.resolve()
        if (
            Path(str(source.get("path"))).expanduser().resolve()
            != expected_resolved
            or source.get("repo_relative_path")
            != expected_resolved.relative_to(repo).as_posix()
        ):
            raise DynamicCropCapacityProbeError(
                f"release smoke {role} is not the actual module source"
            )


def _enforce_live_release_smoke_attestation(
    code: Mapping[str, Any], source_files: Mapping[str, Any]
) -> None:
    """Bind a release-passing record to this exact clean checkout."""

    module_root = Path(__file__).resolve().parent
    repo_result = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=module_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if repo_result.returncode:
        raise DynamicCropCapacityProbeError(
            "release smoke checkout repository is unavailable"
        )
    repo = Path(repo_result.stdout.decode("utf-8").strip()).resolve()
    _enforce_actual_smoke_source_paths(source_files, repo=repo)
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    diff = subprocess.run(
        ("git", "diff", "--binary", "HEAD", "--"),
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if head.returncode or status.returncode or diff.returncode:
        raise DynamicCropCapacityProbeError(
            "release smoke live git attestation failed"
        )
    current_head = head.stdout.decode("ascii").strip().lower()
    status_lines = status.stdout.decode("utf-8", errors="replace").splitlines()
    if (
        current_head != code["observed_git_commit"]
        or current_head != code["expected_git_commit"]
        or status.stdout
        or diff.stdout
        or code["repository_status_clean"] is not True
        or code["repository_status_entry_count"] != len(status_lines)
        or code["repository_status_porcelain_sha256"]
        != hashlib.sha256(status.stdout).hexdigest()
        or code["tracked_diff_clean"] is not True
        or code["tracked_diff_sha256"] != hashlib.sha256(diff.stdout).hexdigest()
    ):
        raise DynamicCropCapacityProbeError(
            "release smoke does not match the live clean full-repository checkout"
        )


def _enforce_live_runtime_module_attestation(
    runtime: Mapping[str, Any], budget: DeviceMemoryBudget, *, repo: Path
) -> None:
    modules = _mapping(
        runtime.get("imported_runtime_modules"),
        "release smoke imported runtime modules",
    )
    expected_imports = {
        "numpy": ("numpy", "numpy"),
        "mlx_core": ("mlx.core", "mlx"),
        "mlx_nn": ("mlx.nn", "mlx"),
        "mlx_optimizers": ("mlx.optimizers", "mlx"),
        "numpy_multiarray_native": (
            "numpy._core._multiarray_umath",
            "numpy",
        ),
        "mlx_nn_linear_impl": ("mlx.nn.layers.linear", "mlx"),
        "mlx_nn_transformer_impl": (
            "mlx.nn.layers.transformer",
            "mlx",
        ),
        "mlx_optimizer_impl": ("mlx.optimizers.optimizers", "mlx"),
    }
    for role, (import_name, distribution_name) in expected_imports.items():
        imported = importlib.import_module(import_name)
        actual_path = Path(str(imported.__file__)).resolve()
        record = _mapping(modules.get(role), f"release smoke {role} module")
        distribution = importlib.metadata.distribution(distribution_name)
        matching_records = sorted(
            str(package_path)
            for package_path in distribution.files or ()
            if Path(package_path.locate()).resolve() == actual_path
        )
        if (
            Path(str(record.get("path"))).resolve() != actual_path
            or record.get("file_sha256") != _file_sha256(actual_path)
            or record.get("distribution_name") != distribution_name
            or matching_records != [record.get("distribution_record_path")]
        ):
            raise DynamicCropCapacityProbeError(
                f"release smoke {role} does not match the actually imported module"
            )
    distributions = _mapping(
        runtime.get("runtime_distributions"),
        "release smoke runtime distributions",
    )
    if (
        distributions["numpy"]["version"]
        != importlib.metadata.version("numpy")
        or distributions["mlx"]["version"] != budget.mlx_version
        or distributions["mlx"]["version"]
        != importlib.metadata.version("mlx")
    ):
        raise DynamicCropCapacityProbeError(
            "release smoke runtime distribution versions differ live"
        )
    from .stage_edit_dynamic_crop_mlx_smoke_cli import (
        _compute_distribution_record_provenance,
    )

    for distribution_name in ("numpy", "mlx"):
        try:
            observed_distribution = _compute_distribution_record_provenance(
                distribution_name
            )
        except Exception as exc:
            raise DynamicCropCapacityProbeError(
                f"release smoke {distribution_name} distribution replay failed"
            ) from exc
        if distributions[distribution_name] != observed_distribution:
            raise DynamicCropCapacityProbeError(
                f"release smoke {distribution_name} installed bytes differ live"
            )
    expected_pythonpath: list[str] = [str((repo / "engine").resolve())]
    for distribution_name in ("numpy", "mlx"):
        root = str(
            Path(
                importlib.metadata.distribution(distribution_name).locate_file("")
            ).resolve()
        )
        if root not in expected_pythonpath:
            expected_pythonpath.append(root)
    if runtime.get("worker_pythonpath_entries") != expected_pythonpath:
        raise DynamicCropCapacityProbeError(
            "release smoke verified worker import roots differ live"
        )


def _validate_mlx_smoke_evidence_root(
    root: Mapping[str, Any],
    *,
    candidate: DynamicCropCapacityCandidate,
    budget: DeviceMemoryBudget,
    require_supervisor_attestation: bool = False,
) -> dict[str, Any]:
    """Validate one release-facing real-MLX smoke record.

    A smoke record can truthfully preserve an OOM, signal, or runtime failure,
    but it cannot authorize capacity selection unless code provenance is clean
    and the complete forward/backward/clip/AdamW step fits the pinned budget.
    """

    payload = dict(root)
    _exact_keys(
        payload,
        {
            "schema_version",
            "analysis_basis",
            "semantic_metrics_authorized",
            "candidate",
            "dataset_provenance",
            "code_provenance",
            "runtime_provenance",
            "hardware_provenance",
            "tensor_provenance",
            "execution",
            "supervisor_attestation",
            "runner_log",
            "artifact_content_sha256",
        },
        "MLX smoke evidence",
    )
    if (
        payload["schema_version"] != SMOKE_EVIDENCE_SCHEMA_VERSION
        or payload["analysis_basis"] != SMOKE_EVIDENCE_ANALYSIS_BASIS
        or payload["semantic_metrics_authorized"] is not False
        or payload["candidate"] != candidate.to_dict()
    ):
        raise DynamicCropCapacityProbeError("real MLX smoke evidence contract differs")

    dataset = _mapping(payload["dataset_provenance"], "smoke dataset provenance")
    _exact_keys(
        dataset,
        {
            "selection_song_ids",
            "quarantined_song_ids_opened",
            "reference_count",
            "reference_ids_by_song",
            "feature_manifests",
            "feature_snapshots",
            "materialization_manifests",
            "dynamic_crop_manifests",
            "reference_supervisions",
            "reference_supervision_count",
            "reference_supervision_aggregate_sha256",
            "dynamic_crop_artifacts",
            "dynamic_crop_artifact_count",
            "dynamic_crop_artifact_aggregate_sha256",
        },
        "smoke dataset provenance",
    )
    if (
        dataset["selection_song_ids"] != list(SELECTION_SONG_IDS)
        or dataset["quarantined_song_ids_opened"] is not False
        or dataset["reference_count"] != 15
    ):
        raise DynamicCropCapacityProbeError("smoke dataset is not exact BBR-15")
    references_by_song = _mapping(
        dataset["reference_ids_by_song"], "smoke reference ids by song"
    )
    manifests = _mapping(dataset["feature_manifests"], "smoke feature manifests")
    materialization_manifests = _mapping(
        dataset["materialization_manifests"],
        "smoke materialization manifests",
    )
    dynamic_manifests = _mapping(
        dataset["dynamic_crop_manifests"], "smoke dynamic crop manifests"
    )
    if any(
        set(value) != set(SELECTION_SONG_IDS)
        for value in (
            references_by_song,
            manifests,
            materialization_manifests,
            dynamic_manifests,
        )
    ):
        raise DynamicCropCapacityProbeError("smoke dataset song identities differ")
    expected_reference_pairs: set[tuple[str, str]] = set()
    for song in SELECTION_SONG_IDS:
        references = references_by_song[song]
        if (
            not isinstance(references, list)
            or len(references) != EXPECTED_REFERENCE_COUNT_PER_SONG
            or len(set(str(value) for value in references)) != len(references)
        ):
            raise DynamicCropCapacityProbeError(
                "smoke dataset needs five unique references per song"
            )
        for reference in references:
            normalized = _reject_quarantined_text(reference, "smoke reference id")
            expected_reference_pairs.add((song, normalized))
        manifest_path = _verify_pointer(
            manifests[song], f"{song} smoke feature manifest"
        )
        _reject_quarantined_text(manifest_path, "smoke feature manifest path")
        for family, pointer in (
            ("materialization", materialization_manifests[song]),
            ("dynamic crop", dynamic_manifests[song]),
        ):
            target_manifest_path = _verify_pointer(
                pointer, f"{song} smoke {family} manifest"
            )
            _reject_quarantined_text(
                target_manifest_path, f"smoke {family} manifest path"
            )

    snapshots = dataset["feature_snapshots"]
    if not isinstance(snapshots, list) or len(snapshots) != 15:
        raise DynamicCropCapacityProbeError("smoke evidence needs fifteen snapshots")
    observed_pairs: set[tuple[str, str]] = set()
    for index, raw in enumerate(snapshots):
        record = _mapping(raw, f"smoke snapshot {index}")
        _exact_keys(
            record,
            {
                "song_id",
                "reference_id",
                "token_count",
                "snapshot_content_sha256",
                "file",
            },
            f"smoke snapshot {index}",
        )
        song = str(record["song_id"])
        reference = _reject_quarantined_text(
            record["reference_id"], "smoke snapshot reference id"
        )
        if song not in SELECTION_SONG_IDS:
            raise DynamicCropCapacityProbeError("smoke snapshot song is not BBR")
        if (
            isinstance(record["token_count"], bool)
            or not isinstance(record["token_count"], int)
            or record["token_count"] < 1
            or not _is_sha256(record["snapshot_content_sha256"])
        ):
            raise DynamicCropCapacityProbeError("smoke snapshot metadata is invalid")
        snapshot_path = _verify_pointer(record["file"], "smoke feature snapshot")
        _reject_quarantined_text(snapshot_path, "smoke feature snapshot path")
        pair = (song, reference)
        if pair in observed_pairs:
            raise DynamicCropCapacityProbeError("duplicate smoke feature snapshot")
        observed_pairs.add(pair)
    if observed_pairs != expected_reference_pairs:
        raise DynamicCropCapacityProbeError("smoke snapshot/reference set differs")
    supervisions = dataset["reference_supervisions"]
    dynamic_artifacts = dataset["dynamic_crop_artifacts"]
    if (
        not isinstance(supervisions, list)
        or len(supervisions) != 15
        or not isinstance(dynamic_artifacts, list)
        or len(dynamic_artifacts) != 599
        or dataset["reference_supervision_count"] != 15
        or dataset["dynamic_crop_artifact_count"] != 599
        or not _is_sha256(dataset["reference_supervision_aggregate_sha256"])
        or not _is_sha256(dataset["dynamic_crop_artifact_aggregate_sha256"])
    ):
        raise DynamicCropCapacityProbeError(
            "smoke target aggregate provenance is incomplete"
        )

    def target_order(record: Mapping[str, Any]) -> tuple[int, str, int]:
        song = str(record.get("song_id"))
        if song not in SELECTION_SONG_IDS:
            raise DynamicCropCapacityProbeError("smoke target song is not BBR")
        shot = record.get("shot_index", -1)
        if isinstance(shot, bool) or not isinstance(shot, int):
            raise DynamicCropCapacityProbeError("smoke target shot index is invalid")
        return (SELECTION_SONG_IDS.index(song), str(record.get("reference_id")), shot)

    supervision_pairs: set[tuple[str, str]] = set()
    normalized_supervisions: list[dict[str, Any]] = []
    for index, raw in enumerate(supervisions):
        record = _mapping(raw, f"smoke reference supervision {index}")
        _exact_keys(
            record,
            {"song_id", "reference_id", "file"},
            f"smoke reference supervision {index}",
        )
        song = str(record["song_id"])
        if song not in SELECTION_SONG_IDS:
            raise DynamicCropCapacityProbeError("smoke supervision song is not BBR")
        reference = _reject_quarantined_text(
            record["reference_id"], "smoke supervision reference id"
        )
        path = _verify_pointer(record["file"], "smoke reference supervision")
        pair = (song, reference)
        if pair in supervision_pairs:
            raise DynamicCropCapacityProbeError("duplicate smoke supervision")
        supervision_pairs.add(pair)
        normalized_supervisions.append(
            {
                "song_id": song,
                "reference_id": reference,
                "file": {
                    "path": str(path),
                    "file_sha256": _file_sha256(path),
                },
            }
        )
    if (
        supervision_pairs != expected_reference_pairs
        or normalized_supervisions != supervisions
        or normalized_supervisions
        != sorted(normalized_supervisions, key=target_order)
        or _sha256(normalized_supervisions)
        != dataset["reference_supervision_aggregate_sha256"]
    ):
        raise DynamicCropCapacityProbeError(
            "smoke supervision provenance or aggregate differs"
        )

    manifest_supervisions: list[dict[str, Any]] = []
    manifest_artifacts: list[dict[str, Any]] = []
    for song in SELECTION_SONG_IDS:
        materialization_path = _verify_pointer(
            materialization_manifests[song],
            f"{song} smoke materialization manifest",
        )
        materialization = _read_json_object(
            materialization_path, f"{song} smoke materialization manifest"
        )
        outputs = materialization.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 5:
            raise DynamicCropCapacityProbeError(
                "smoke materialization manifest outputs differ"
            )
        for raw in outputs:
            output = _mapping(raw, "smoke materialization output")
            reference = _reject_quarantined_text(
                output.get("reference_id"), "smoke materialization reference id"
            )
            path_text = _reject_quarantined_text(
                output.get("supervision_path"), "smoke supervision manifest path"
            )
            manifest_supervisions.append(
                {
                    "song_id": song,
                    "reference_id": reference,
                    "file": {
                        "path": str(Path(path_text).expanduser().resolve()),
                        "file_sha256": output.get("supervision_file_sha256"),
                    },
                }
            )
        dynamic_manifest_path = _verify_pointer(
            dynamic_manifests[song], f"{song} smoke dynamic crop manifest"
        )
        dynamic_manifest = _read_json_object(
            dynamic_manifest_path, f"{song} smoke dynamic crop manifest"
        )
        manifest_records = dynamic_manifest.get("artifacts")
        if not isinstance(manifest_records, list):
            raise DynamicCropCapacityProbeError(
                "smoke dynamic crop manifest artifacts must be an array"
            )
        for raw in manifest_records:
            manifest_record = _mapping(raw, "smoke dynamic crop manifest record")
            reference = _reject_quarantined_text(
                manifest_record.get("reference_id"),
                "smoke dynamic crop manifest reference id",
            )
            shot_index = manifest_record.get("shot_index")
            if (
                isinstance(shot_index, bool)
                or not isinstance(shot_index, int)
                or shot_index < 0
            ):
                raise DynamicCropCapacityProbeError(
                    "smoke dynamic crop manifest shot index is invalid"
                )
            artifact_pointer = _mapping(
                manifest_record.get("artifact"),
                "smoke dynamic crop manifest artifact",
            )
            manifest_artifacts.append(
                {
                    "song_id": song,
                    "reference_id": reference,
                    "shot_index": shot_index,
                    "file": {
                        "path": str(
                            Path(
                                _reject_quarantined_text(
                                    artifact_pointer.get("path"),
                                    "smoke dynamic crop artifact path",
                                )
                            )
                            .expanduser()
                            .resolve()
                        ),
                        "file_sha256": artifact_pointer.get("file_sha256"),
                    },
                    "content_sha256": artifact_pointer.get("content_sha256"),
                }
            )
    manifest_supervisions.sort(key=target_order)
    manifest_artifacts.sort(key=target_order)
    if manifest_supervisions != normalized_supervisions:
        raise DynamicCropCapacityProbeError(
            "smoke supervisions differ from materialization manifests"
        )

    normalized_artifacts: list[dict[str, Any]] = []
    artifact_identities: set[tuple[str, str, int]] = set()
    artifact_paths: set[str] = set()
    for index, raw in enumerate(dynamic_artifacts):
        record = _mapping(raw, f"smoke dynamic crop artifact {index}")
        _exact_keys(
            record,
            {"song_id", "reference_id", "shot_index", "file", "content_sha256"},
            f"smoke dynamic crop artifact {index}",
        )
        song = str(record["song_id"])
        reference = _reject_quarantined_text(
            record["reference_id"], "smoke dynamic crop artifact reference id"
        )
        shot_index = record["shot_index"]
        if (
            song not in SELECTION_SONG_IDS
            or (song, reference) not in expected_reference_pairs
            or isinstance(shot_index, bool)
            or not isinstance(shot_index, int)
            or shot_index < 0
            or not _is_sha256(record["content_sha256"])
        ):
            raise DynamicCropCapacityProbeError(
                "smoke dynamic crop artifact identity is invalid"
            )
        path = _verify_pointer(record["file"], "smoke dynamic crop artifact")
        identity = (song, reference, shot_index)
        if identity in artifact_identities or str(path) in artifact_paths:
            raise DynamicCropCapacityProbeError(
                "duplicate smoke dynamic crop artifact identity or path"
            )
        artifact_identities.add(identity)
        artifact_paths.add(str(path))
        artifact_payload = _read_json_object(path, "smoke dynamic crop artifact")
        computed_content_hash = _dynamic_crop_artifact_content_sha256(
            artifact_payload
        )
        artifact_provenance = _mapping(
            artifact_payload.get("provenance"),
            "smoke dynamic crop artifact provenance",
        )
        if (
            computed_content_hash != record["content_sha256"]
            or artifact_provenance.get("artifact_content_sha256")
            != computed_content_hash
        ):
            raise DynamicCropCapacityProbeError(
                "smoke dynamic crop artifact content hash mismatch"
            )
        normalized_artifacts.append(
            {
                "song_id": song,
                "reference_id": reference,
                "shot_index": shot_index,
                "file": {
                    "path": str(path),
                    "file_sha256": _file_sha256(path),
                },
                "content_sha256": computed_content_hash,
            }
        )
    if (
        normalized_artifacts != dynamic_artifacts
        or normalized_artifacts != sorted(normalized_artifacts, key=target_order)
        or normalized_artifacts != manifest_artifacts
        or _sha256(normalized_artifacts)
        != dataset["dynamic_crop_artifact_aggregate_sha256"]
    ):
        raise DynamicCropCapacityProbeError(
            "smoke dynamic crop artifact inventory or aggregate differs"
        )

    code = _mapping(payload["code_provenance"], "smoke code provenance")
    _exact_keys(
        code,
        {
            "runner_contract",
            "expected_git_commit",
            "observed_git_commit",
            "tracked_diff_policy",
            "tracked_diff_clean",
            "tracked_diff_sha256",
            "repository_status_clean",
            "repository_status_entry_count",
            "repository_status_porcelain_sha256",
            "untracked_dependency_paths",
            "source_files_match_observed_commit",
            "release_trust_authorized",
            "source_files",
        },
        "smoke code provenance",
    )
    for name in ("expected_git_commit", "observed_git_commit"):
        value = code[name]
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise DynamicCropCapacityProbeError(f"smoke {name} is invalid")
    if (
        code["runner_contract"] != SMOKE_RUNNER_CONTRACT
        or code["tracked_diff_policy"]
        not in {"require_clean", "record_development"}
        or not isinstance(code["tracked_diff_clean"], bool)
        or not _is_sha256(code["tracked_diff_sha256"])
        or not isinstance(code["repository_status_clean"], bool)
        or isinstance(code["repository_status_entry_count"], bool)
        or not isinstance(code["repository_status_entry_count"], int)
        or code["repository_status_entry_count"] < 0
        or not _is_sha256(code["repository_status_porcelain_sha256"])
        or not isinstance(code["untracked_dependency_paths"], list)
        or not isinstance(code["source_files_match_observed_commit"], bool)
        or any(
            not isinstance(value, str) or not value.strip()
            for value in code["untracked_dependency_paths"]
        )
    ):
        raise DynamicCropCapacityProbeError("smoke code provenance is invalid")
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    if code["tracked_diff_clean"] is not (
        code["tracked_diff_sha256"] == empty_sha256
    ) or code["repository_status_clean"] is not (
        code["repository_status_entry_count"] == 0
        and code["repository_status_porcelain_sha256"] == empty_sha256
    ):
        raise DynamicCropCapacityProbeError(
            "smoke repository clean-state evidence is inconsistent"
        )
    source_files = _mapping(code["source_files"], "smoke source files")
    if set(source_files) != {
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
    }:
        raise DynamicCropCapacityProbeError("smoke runner source binding is incomplete")
    repo_root: Path | None = None
    all_sources_match_commit = True
    for role, raw_source in source_files.items():
        source = _mapping(raw_source, f"smoke {role} source")
        _exact_keys(
            source,
            {
                "path",
                "file_sha256",
                "repo_relative_path",
                "observed_commit_blob_sha256",
                "matches_observed_commit",
            },
            f"smoke {role} source",
        )
        source_path = _verify_pointer(
            {"path": source["path"], "file_sha256": source["file_sha256"]},
            f"smoke {role} source",
        )
        relative = Path(str(source["repo_relative_path"]))
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or (
                source["observed_commit_blob_sha256"] is not None
                and not _is_sha256(source["observed_commit_blob_sha256"])
            )
            or not isinstance(source["matches_observed_commit"], bool)
        ):
            raise DynamicCropCapacityProbeError(
                f"smoke {role} git source binding is invalid"
            )
        candidate_root = source_path
        for _ in relative.parts:
            candidate_root = candidate_root.parent
        if (candidate_root / relative).resolve() != source_path:
            raise DynamicCropCapacityProbeError(
                f"smoke {role} source path differs from repository path"
            )
        if repo_root is None:
            repo_root = candidate_root
        elif repo_root != candidate_root:
            raise DynamicCropCapacityProbeError(
                "smoke source files do not share one repository root"
            )
        result = subprocess.run(
            (
                "git",
                "show",
                f"{code['observed_git_commit']}:{relative.as_posix()}",
            ),
            cwd=candidate_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        blob_hash = (
            hashlib.sha256(result.stdout).hexdigest()
            if result.returncode == 0
            else None
        )
        matches_commit = (
            blob_hash == source["observed_commit_blob_sha256"]
            and blob_hash == source["file_sha256"]
        )
        if source["matches_observed_commit"] is not matches_commit:
            raise DynamicCropCapacityProbeError(
                f"smoke {role} observed commit binding differs"
            )
        all_sources_match_commit = all_sources_match_commit and matches_commit
    if code["source_files_match_observed_commit"] is not all_sources_match_commit:
        raise DynamicCropCapacityProbeError(
            "smoke aggregate source/commit binding differs"
        )
    expected_release_trust = (
        code["tracked_diff_policy"] == "require_clean"
        and code["expected_git_commit"] == code["observed_git_commit"]
        and code["tracked_diff_clean"] is True
        and code["repository_status_clean"] is True
        and code["repository_status_entry_count"] == 0
        and code["repository_status_porcelain_sha256"]
        == empty_sha256
        and code["untracked_dependency_paths"] == []
        and code["source_files_match_observed_commit"] is True
    )
    if code["release_trust_authorized"] is not expected_release_trust:
        raise DynamicCropCapacityProbeError("smoke release trust is inconsistent")
    if expected_release_trust:
        _enforce_live_release_smoke_attestation(code, source_files)

    runtime = _mapping(payload["runtime_provenance"], "smoke runtime provenance")
    _exact_keys(
        runtime,
        {
            "python_version",
            "python_executable",
            "platform",
            "mlx_version",
            "default_device",
            "process_isolation_contract",
            "worker_environment_contract",
            "worker_pythonpath_entries",
            "ambient_pythonpath_inherited",
            "python_no_user_site",
            "python_safe_path",
            "site_import_disabled",
            "installed_bytecode_cache_read_disabled",
            "imported_runtime_modules",
            "runtime_distributions",
        },
        "smoke runtime provenance",
    )
    if (
        any(
            not isinstance(runtime[name], str) or not runtime[name].strip()
            for name in (
                "python_version",
                "python_executable",
                "platform",
                "default_device",
            )
        )
        or runtime["mlx_version"] != budget.mlx_version
        or runtime["process_isolation_contract"]
        != SMOKE_PROCESS_ISOLATION_CONTRACT
        or runtime["worker_environment_contract"]
        != "safe_path_isolated_verified_import_roots_no_site_empty_pycache_nonce_v3"
        or runtime["ambient_pythonpath_inherited"] is not False
        or runtime["python_no_user_site"] is not True
        or runtime["python_safe_path"] is not True
        or runtime["site_import_disabled"] is not True
        or runtime["installed_bytecode_cache_read_disabled"] is not True
    ):
        raise DynamicCropCapacityProbeError("smoke runtime provenance differs")
    worker_pythonpath = runtime["worker_pythonpath_entries"]
    if (
        not isinstance(worker_pythonpath, list)
        or len(worker_pythonpath) < 2
        or any(
            not isinstance(entry, str)
            or not Path(entry).is_absolute()
            or Path(entry).is_symlink()
            or not Path(entry).is_dir()
            for entry in worker_pythonpath
        )
        or len(worker_pythonpath) != len(set(worker_pythonpath))
        or (repo_root is not None)
        and Path(str(worker_pythonpath[0])).resolve()
        != (repo_root / "engine").resolve()
    ):
        raise DynamicCropCapacityProbeError(
            "smoke worker PYTHONPATH isolation differs"
        )
    runtime_modules = _mapping(
        runtime["imported_runtime_modules"], "smoke imported runtime modules"
    )
    expected_runtime_module_names = {
        "numpy": "numpy",
        "mlx_core": "mlx.core",
        "mlx_nn": "mlx.nn",
        "mlx_optimizers": "mlx.optimizers",
        "numpy_multiarray_native": "numpy._core._multiarray_umath",
        "mlx_nn_linear_impl": "mlx.nn.layers.linear",
        "mlx_nn_transformer_impl": "mlx.nn.layers.transformer",
        "mlx_optimizer_impl": "mlx.optimizers.optimizers",
    }
    expected_runtime_distributions = {
        role: ("numpy" if role.startswith("numpy") else "mlx")
        for role in expected_runtime_module_names
    }
    if set(runtime_modules) != set(expected_runtime_module_names):
        raise DynamicCropCapacityProbeError(
            "smoke imported runtime module set differs"
        )
    for role, expected_module_name in expected_runtime_module_names.items():
        module = _mapping(runtime_modules[role], f"smoke {role} runtime module")
        _exact_keys(
            module,
            {
                "module_name",
                "version",
                "distribution_name",
                "distribution_record_path",
                "path",
                "file_sha256",
            },
            f"smoke {role} runtime module",
        )
        if (
            module["module_name"] != expected_module_name
            or module["distribution_name"]
            != expected_runtime_distributions[role]
            or not isinstance(module["distribution_record_path"], str)
            or not module["distribution_record_path"].strip()
            or Path(module["distribution_record_path"]).is_absolute()
            or ".." in Path(module["distribution_record_path"]).parts
            or (
                role.startswith("mlx_")
                and module["version"] != budget.mlx_version
            )
            or (
                role.startswith("numpy")
                and (
                    not isinstance(module["version"], str)
                    or not module["version"].strip()
                )
            )
        ):
            raise DynamicCropCapacityProbeError(
                f"smoke {role} runtime module identity differs"
            )
        _verify_pointer(
            {"path": module["path"], "file_sha256": module["file_sha256"]},
            f"smoke {role} runtime module",
        )
        module_path = Path(str(module["path"])).resolve()
        try:
            recorded_module_path = Path(
                importlib.metadata.distribution(
                    expected_runtime_distributions[role]
                ).locate_file(module["distribution_record_path"])
            ).resolve()
        except (importlib.metadata.PackageNotFoundError, TypeError) as exc:
            raise DynamicCropCapacityProbeError(
                f"smoke {role} distribution RECORD membership is unavailable"
            ) from exc
        if recorded_module_path != module_path:
            raise DynamicCropCapacityProbeError(
                f"smoke {role} module is not the corresponding distribution RECORD file"
            )
        if not any(
            module_path == Path(root).resolve()
            or Path(root).resolve() in module_path.parents
            for root in worker_pythonpath[1:]
        ):
            raise DynamicCropCapacityProbeError(
                f"smoke {role} runtime module is outside verified import roots"
            )
    distributions = _mapping(
        runtime["runtime_distributions"], "smoke runtime distributions"
    )
    if set(distributions) != {"numpy", "mlx"}:
        raise DynamicCropCapacityProbeError(
            "smoke runtime distribution set differs"
        )
    for name, raw_distribution in distributions.items():
        distribution = _mapping(
            raw_distribution, f"smoke {name} runtime distribution"
        )
        _exact_keys(
            distribution,
            {
                "distribution_name",
                "version",
                "record_count",
                "verified_record_hash_count",
                "mutable_bytecode_cache_record_count",
                "unhashed_record_count",
                "verification_contract",
                "record_identity_sha256",
                "actual_files_aggregate_sha256",
            },
            f"smoke {name} runtime distribution",
        )
        if (
            distribution["distribution_name"] != name
            or isinstance(distribution["record_count"], bool)
            or not isinstance(distribution["record_count"], int)
            or distribution["record_count"] < 1
            or isinstance(distribution["verified_record_hash_count"], bool)
            or not isinstance(distribution["verified_record_hash_count"], int)
            or distribution["verified_record_hash_count"] < 1
            or distribution["verified_record_hash_count"]
            > distribution["record_count"]
            or isinstance(
                distribution["mutable_bytecode_cache_record_count"], bool
            )
            or not isinstance(
                distribution["mutable_bytecode_cache_record_count"], int
            )
            or distribution["mutable_bytecode_cache_record_count"] < 0
            or isinstance(distribution["unhashed_record_count"], bool)
            or not isinstance(distribution["unhashed_record_count"], int)
            or distribution["unhashed_record_count"] < 0
            or distribution["verified_record_hash_count"]
            + distribution["mutable_bytecode_cache_record_count"]
            + distribution["unhashed_record_count"]
            != distribution["record_count"]
            or distribution["verification_contract"]
            != RUNTIME_DISTRIBUTION_VERIFICATION_CONTRACT
            or not _is_sha256(distribution["record_identity_sha256"])
            or not _is_sha256(distribution["actual_files_aggregate_sha256"])
            or (name == "mlx" and distribution["version"] != budget.mlx_version)
        ):
            raise DynamicCropCapacityProbeError(
                f"smoke {name} runtime distribution differs"
            )
    if any(
        runtime_modules[role]["version"] != distributions["numpy"]["version"]
        for role in ("numpy", "numpy_multiarray_native")
    ):
        raise DynamicCropCapacityProbeError(
            "smoke numpy module/distribution versions differ"
        )
    if expected_release_trust:
        assert repo_root is not None
        _enforce_live_runtime_module_attestation(
            runtime, budget, repo=repo_root
        )

    hardware = _mapping(payload["hardware_provenance"], "smoke hardware provenance")
    _exact_keys(
        hardware,
        {
            "device_name",
            "physical_memory_bytes",
            "max_recommended_working_set_size",
            "max_buffer_length",
            "architecture",
            "os_reserve_bytes",
            "allowed_training_bytes",
            "device_info_api",
        },
        "smoke hardware provenance",
    )
    if hardware != {
        "device_name": budget.device_name,
        "physical_memory_bytes": budget.physical_memory_bytes,
        "max_recommended_working_set_size": budget.max_recommended_working_set_size,
        "max_buffer_length": budget.max_buffer_length,
        "architecture": budget.architecture,
        "os_reserve_bytes": budget.os_reserve_bytes,
        "allowed_training_bytes": budget.allowed_training_bytes,
        "device_info_api": "mlx.core.device_info",
    }:
        raise DynamicCropCapacityProbeError("smoke hardware provenance differs")

    tensor = _mapping(payload["tensor_provenance"], "smoke tensor provenance")
    _exact_keys(
        tensor,
        {
            "packing_contract",
            "tiling_contract",
            "batch_size",
            "sequence_length",
            "max_sources",
            "actual_observed_source_max",
            "context_feature_dimension",
            "source_feature_dimension",
            "source_tensor_shape",
            "context_tensor_shape",
            "dynamic_values_shape",
            "dynamic_times_shape",
            "dynamic_presence_shape",
            "dtype",
            "lane_reference_ids",
            "lane_original_cycle_token_counts",
            "lane_expanded_token_counts",
            "lane_repeat_counts",
            "lane_truncated_prefix_counts",
            "lane_padding_counts",
            "base_label_mask_counts",
            "dynamic_label_mask_counts",
            "parameter_count",
            "input_tensor_sha256",
            "supervision_tensor_sha256",
            "real_input_materialized",
        },
        "smoke tensor provenance",
    )
    lane_references = _mapping(
        tensor["lane_reference_ids"], "smoke lane reference ids"
    )
    lane_counts = _mapping(
        tensor["lane_original_cycle_token_counts"],
        "smoke lane original cycle token counts",
    )
    lane_expanded = _mapping(
        tensor["lane_expanded_token_counts"], "smoke lane expanded token counts"
    )
    lane_repeats = _mapping(
        tensor["lane_repeat_counts"], "smoke lane repeat counts"
    )
    lane_prefix = _mapping(
        tensor["lane_truncated_prefix_counts"],
        "smoke lane truncated prefix counts",
    )
    lane_padding = _mapping(
        tensor["lane_padding_counts"], "smoke lane padding counts"
    )
    if any(
        set(value) != {"A", "B"}
        for value in (
            lane_references,
            lane_counts,
            lane_expanded,
            lane_repeats,
            lane_prefix,
            lane_padding,
        )
    ):
        raise DynamicCropCapacityProbeError("smoke lane contract differs")
    flattened_lane_references = [
        str(reference)
        for lane in ("A", "B")
        for reference in lane_references[lane]
    ]
    expected_references = {
        reference for _, reference in expected_reference_pairs
    }
    if (
        len(flattened_lane_references) != 15
        or set(flattened_lane_references) != expected_references
        or any(
            isinstance(lane_counts[lane], bool)
            or not isinstance(lane_counts[lane], int)
            or lane_counts[lane] < 1
            or lane_counts[lane] > 2048
            or lane_expanded[lane] != 2048
            or lane_repeats[lane] != 2048 // lane_counts[lane]
            or lane_prefix[lane] != 2048 % lane_counts[lane]
            or lane_padding[lane] != 0
            for lane in ("A", "B")
        )
        or tensor["packing_contract"] != SMOKE_INPUT_PACKING_CONTRACT
        or tensor["tiling_contract"] != SMOKE_TILING_CONTRACT
        or tensor["batch_size"] != 2
        or tensor["sequence_length"] != 2048
        or tensor["max_sources"] != 32
        or tensor["actual_observed_source_max"] != 10
        or candidate.model_config.max_sequence_length != 2048
        or candidate.model_config.max_sources != 32
        or tensor["context_feature_dimension"]
        != candidate.model_config.context_input_dimension
        or tensor["source_feature_dimension"] != 720
        or tensor["source_feature_dimension"]
        != candidate.model_config.source_input_dimension
        or tensor["source_tensor_shape"] != [2, 2048, 32, 720]
        or tensor["context_tensor_shape"]
        != [2, 2048, candidate.model_config.context_input_dimension]
        or tensor["dynamic_values_shape"] != [2, 2048, 8, 4]
        or tensor["dynamic_times_shape"] != [2, 2048, 8]
        or tensor["dynamic_presence_shape"] != [2, 2048, 8]
        or tensor["dtype"] != "float32"
        or tensor["parameter_count"]
        != stage_edit_dynamic_crop_v2_parameter_count(
            candidate.model_config, candidate.dynamic_crop_config
        )
        or not isinstance(tensor["real_input_materialized"], bool)
    ):
        raise DynamicCropCapacityProbeError("smoke tensor contract differs")
    for name in ("input_tensor_sha256", "supervision_tensor_sha256"):
        value = tensor[name]
        if value is not None and not _is_sha256(value):
            raise DynamicCropCapacityProbeError(f"smoke {name} is invalid")
    if tensor["real_input_materialized"] is not (
        tensor["input_tensor_sha256"] is not None
        and tensor["supervision_tensor_sha256"] is not None
    ):
        raise DynamicCropCapacityProbeError("smoke tensor materialization differs")
    for name, expected_keys in (
        (
            "base_label_mask_counts",
            {
                "cut",
                "source_listwise",
                "source_pairwise",
                "duration",
                "transition_metadata_only",
                "crop_coordinates",
            },
        ),
        (
            "dynamic_label_mask_counts",
            {"tracks", "keyframes", "presence", "motion"},
        ),
    ):
        counts = tensor[name]
        if tensor["real_input_materialized"]:
            counts = _mapping(counts, f"smoke {name}")
            if set(counts) != expected_keys or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in counts.values()
            ):
                raise DynamicCropCapacityProbeError(
                    f"smoke {name} is invalid"
                )
        elif counts is not None:
            raise DynamicCropCapacityProbeError(
                f"unmaterialized smoke {name} must be null"
            )

    execution = _mapping(payload["execution"], "smoke execution")
    _exact_keys(
        execution,
        {
            "model_allocated",
            "forward_completed",
            "loss_completed",
            "backward_completed",
            "gradients_finite",
            "gradient_global_norm_before_clip",
            "gradient_clip_norm",
            "gradient_clip_completed",
            "optimizer_step_completed",
            "optimizer_state_materialized",
            "parameter_tree",
            "gradient_tree",
            "post_update_parameter_tree",
            "updated_parameters_finite",
            "optimizer_state_tree",
            "optimizer_state_finite",
            "provenance_handshake",
            "optimizer_name",
            "optimizer_config",
            "loss_contract_sha256",
            "loss_value_for_execution_only",
            "previous_memory_limit_bytes",
            "applied_memory_limit_bytes",
            "previous_cache_limit_bytes",
            "applied_cache_limit_bytes",
            "memory_limit_contract",
            "pre_reset_active_memory_bytes",
            "pre_reset_cache_memory_bytes",
            "sync_stages",
            "final_active_memory_bytes",
            "final_cache_memory_bytes",
            "elapsed_seconds",
            "measured_peak_bytes",
            "allowed_training_bytes",
            "reserve_after_peak_bytes",
            "runtime_execution_passed",
            "release_trust_authorized",
            "passed",
            "failure_kind",
            "process_exit_code",
            "termination_signal",
        },
        "smoke execution",
    )
    progress_names = (
        "model_allocated",
        "forward_completed",
        "loss_completed",
        "backward_completed",
        "gradient_clip_completed",
        "optimizer_step_completed",
        "optimizer_state_materialized",
    )
    if any(not isinstance(execution[name], bool) for name in progress_names):
        raise DynamicCropCapacityProbeError("MLX progress flags must be booleans")
    handshake = _mapping(
        execution["provenance_handshake"], "smoke provenance handshake"
    )
    _exact_keys(
        handshake,
        {
            "candidate_id",
            "candidate_contract_sha256",
            "supervisor_dataset_provenance_sha256",
            "worker_dataset_provenance_sha256",
            "supervisor_code_provenance_sha256",
            "worker_code_provenance_sha256",
            "supervisor_runtime_provenance_sha256",
            "worker_runtime_provenance_sha256",
            "supervisor_hardware_provenance_sha256",
            "worker_hardware_provenance_sha256",
            "expected_git_commit",
            "worker_observed_git_commit",
            "matched",
        },
        "smoke provenance handshake",
    )
    expected_handshake_hashes = {
        "candidate_contract_sha256": _sha256(candidate.to_dict()),
        "supervisor_dataset_provenance_sha256": _sha256(dataset),
        "supervisor_code_provenance_sha256": _sha256(code),
        "supervisor_runtime_provenance_sha256": _sha256(runtime),
        "supervisor_hardware_provenance_sha256": _sha256(hardware),
    }
    for name, expected_value in expected_handshake_hashes.items():
        if handshake.get(name) != expected_value:
            raise DynamicCropCapacityProbeError(
                f"smoke {name} differs from supervisor evidence"
            )
    expected_handshake_match = (
        handshake.get("candidate_id") == candidate.candidate_id
        and handshake.get("worker_dataset_provenance_sha256")
        == expected_handshake_hashes["supervisor_dataset_provenance_sha256"]
        and handshake.get("worker_code_provenance_sha256")
        == expected_handshake_hashes["supervisor_code_provenance_sha256"]
        and handshake.get("worker_runtime_provenance_sha256")
        == expected_handshake_hashes["supervisor_runtime_provenance_sha256"]
        and handshake.get("worker_hardware_provenance_sha256")
        == expected_handshake_hashes["supervisor_hardware_provenance_sha256"]
        and handshake.get("expected_git_commit") == code["expected_git_commit"]
        and handshake.get("worker_observed_git_commit")
        == code["expected_git_commit"]
    )
    if (
        not isinstance(handshake.get("matched"), bool)
        or handshake["matched"] is not expected_handshake_match
    ):
        raise DynamicCropCapacityProbeError(
            "smoke worker provenance handshake is inconsistent"
        )
    progress = tuple(bool(execution[name]) for name in progress_names)
    if any(
        progress[index] and not progress[index - 1]
        for index in range(1, len(progress))
    ):
        raise DynamicCropCapacityProbeError("MLX progress flags are not monotonic")
    if execution["gradients_finite"] not in {True, False, None}:
        raise DynamicCropCapacityProbeError("smoke gradient finite flag is invalid")
    for name in ("updated_parameters_finite", "optimizer_state_finite"):
        if execution[name] not in {True, False, None}:
            raise DynamicCropCapacityProbeError(f"smoke {name} is invalid")
    gradient_norm = _nullable_finite_nonnegative(
        execution["gradient_global_norm_before_clip"], "smoke gradient norm"
    )
    loss_value = _nullable_finite_nonnegative(
        execution["loss_value_for_execution_only"], "smoke loss"
    )
    peak = execution["measured_peak_bytes"]
    if peak is not None and (
        isinstance(peak, bool)
        or not isinstance(peak, int)
        or peak <= 0
        or peak > budget.physical_memory_bytes
    ):
        raise DynamicCropCapacityProbeError("MLX peak memory is invalid")
    reserve = execution["reserve_after_peak_bytes"]
    expected_reserve = budget.allowed_training_bytes - peak if peak is not None else None
    if reserve != expected_reserve:
        raise DynamicCropCapacityProbeError("MLX reserve calculation differs")
    optimizer_config = _mapping(
        execution["optimizer_config"], "smoke optimizer config"
    )
    for name in (
        "previous_memory_limit_bytes",
        "applied_memory_limit_bytes",
        "previous_cache_limit_bytes",
        "applied_cache_limit_bytes",
    ):
        value = execution[name]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise DynamicCropCapacityProbeError(f"smoke {name} is invalid")
    limits_applied = execution["applied_memory_limit_bytes"] is not None
    if limits_applied is not (
        execution["previous_memory_limit_bytes"] is not None
        and execution["previous_cache_limit_bytes"] is not None
        and execution["applied_cache_limit_bytes"] is not None
    ):
        raise DynamicCropCapacityProbeError("smoke MLX limit provenance is partial")
    if limits_applied and (
        execution["applied_memory_limit_bytes"] != budget.allowed_training_bytes
        or execution["applied_cache_limit_bytes"] != SMOKE_CACHE_LIMIT_BYTES
    ):
        raise DynamicCropCapacityProbeError("smoke MLX limits differ from contract")
    for name in (
        "pre_reset_active_memory_bytes",
        "pre_reset_cache_memory_bytes",
        "final_active_memory_bytes",
        "final_cache_memory_bytes",
    ):
        value = execution[name]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise DynamicCropCapacityProbeError(f"smoke {name} is invalid")
    if limits_applied is not (
        execution["pre_reset_active_memory_bytes"] is not None
        and execution["pre_reset_cache_memory_bytes"] is not None
    ):
        raise DynamicCropCapacityProbeError("smoke pre-reset memory provenance differs")

    def validate_tree(name: str, *, required: bool) -> dict[str, Any] | None:
        raw = execution[name]
        if raw is None:
            if required:
                raise DynamicCropCapacityProbeError(f"smoke {name} is missing")
            return None
        if not required:
            raise DynamicCropCapacityProbeError(
                f"smoke {name} exists before its completed stage"
            )
        tree = _mapping(raw, f"smoke {name}")
        _exact_keys(
            tree,
            {
                "leaf_count",
                "element_count",
                "nbytes",
                "key_digest_sha256",
                "nonfinite_count",
            },
            f"smoke {name}",
        )
        if (
            any(
                isinstance(tree[field], bool)
                or not isinstance(tree[field], int)
                or tree[field] < 0
                for field in (
                    "leaf_count",
                    "element_count",
                    "nbytes",
                    "nonfinite_count",
                )
            )
            or tree["leaf_count"] < 1
            or tree["element_count"] < 1
            or tree["nbytes"] < 1
            or not _is_sha256(tree["key_digest_sha256"])
        ):
            raise DynamicCropCapacityProbeError(f"smoke {name} is invalid")
        return tree

    parameter_tree = validate_tree(
        "parameter_tree", required=execution["model_allocated"]
    )
    gradient_tree = validate_tree(
        "gradient_tree", required=execution["backward_completed"]
    )
    optimizer_tree = validate_tree(
        "optimizer_state_tree", required=execution["optimizer_step_completed"]
    )
    post_update_parameter_tree = validate_tree(
        "post_update_parameter_tree",
        required=execution["optimizer_step_completed"],
    )
    if (
        (parameter_tree is not None)
        and (
            parameter_tree["element_count"] != tensor["parameter_count"]
            or parameter_tree["nbytes"] != tensor["parameter_count"] * 4
            or parameter_tree["nonfinite_count"] != 0
        )
    ):
        raise DynamicCropCapacityProbeError("smoke parameter tree differs")
    if gradient_tree is not None and (
        gradient_tree["element_count"] != tensor["parameter_count"]
        or gradient_tree["nbytes"] != tensor["parameter_count"] * 4
    ):
        raise DynamicCropCapacityProbeError("smoke gradient tree differs")
    if execution["gradients_finite"] is True and (
        gradient_tree is None or gradient_tree["nonfinite_count"] != 0
    ):
        raise DynamicCropCapacityProbeError("smoke gradient finite proof differs")
    if execution["optimizer_state_materialized"] is not (
        optimizer_tree is not None and optimizer_tree["leaf_count"] > 0
    ):
        raise DynamicCropCapacityProbeError("smoke optimizer state proof differs")
    if post_update_parameter_tree is not None and (
        post_update_parameter_tree["element_count"] != tensor["parameter_count"]
        or post_update_parameter_tree["nbytes"] != tensor["parameter_count"] * 4
    ):
        raise DynamicCropCapacityProbeError(
            "smoke post-update parameter tree differs"
        )
    expected_updated_finite = (
        None
        if post_update_parameter_tree is None
        else post_update_parameter_tree["nonfinite_count"] == 0
    )
    if execution["updated_parameters_finite"] is not expected_updated_finite:
        raise DynamicCropCapacityProbeError(
            "smoke updated parameter finite proof differs"
        )
    expected_optimizer_finite = (
        None
        if optimizer_tree is None
        else optimizer_tree["nonfinite_count"] == 0
    )
    if execution["optimizer_state_finite"] is not expected_optimizer_finite:
        raise DynamicCropCapacityProbeError(
            "smoke optimizer state finite proof differs"
        )

    sync_stages = execution["sync_stages"]
    if not isinstance(sync_stages, list):
        raise DynamicCropCapacityProbeError("smoke sync stages must be an array")
    expected_stage_order = (
        "input_materialized",
        "model_allocated",
        "forward_backward_completed",
        "gradient_clip_completed",
        "optimizer_step_completed",
    )
    observed_stage_order: list[str] = []
    previous_elapsed = -1.0
    previous_peak = -1
    for index, raw_stage in enumerate(sync_stages):
        stage = _mapping(raw_stage, f"smoke sync stage {index}")
        _exact_keys(
            stage,
            {
                "stage",
                "synchronized",
                "active_memory_bytes",
                "cache_memory_bytes",
                "peak_memory_bytes",
                "elapsed_seconds",
            },
            f"smoke sync stage {index}",
        )
        elapsed = _finite_nonnegative(
            stage["elapsed_seconds"], "smoke stage elapsed seconds"
        )
        if (
            stage["synchronized"] is not True
            or any(
                isinstance(stage[name], bool)
                or not isinstance(stage[name], int)
                or stage[name] < 0
                for name in (
                    "active_memory_bytes",
                    "cache_memory_bytes",
                    "peak_memory_bytes",
                )
            )
            or elapsed < previous_elapsed
            or stage["peak_memory_bytes"] < previous_peak
        ):
            raise DynamicCropCapacityProbeError("smoke sync stage is invalid")
        previous_elapsed = elapsed
        previous_peak = stage["peak_memory_bytes"]
        observed_stage_order.append(str(stage["stage"]))
    if tuple(observed_stage_order) != expected_stage_order[: len(sync_stages)]:
        raise DynamicCropCapacityProbeError("smoke sync stage order differs")
    completed_stage_peak = (
        max(int(stage["peak_memory_bytes"]) for stage in sync_stages)
        if sync_stages
        else None
    )
    if completed_stage_peak is not None and (
        peak is None or peak < completed_stage_peak
    ):
        raise DynamicCropCapacityProbeError(
            "smoke measured peak is below a completed synchronized stage"
        )
    final_stage = sync_stages[-1] if sync_stages else None
    if (
        execution["final_active_memory_bytes"]
        != (final_stage["active_memory_bytes"] if final_stage else None)
        or execution["final_cache_memory_bytes"]
        != (final_stage["cache_memory_bytes"] if final_stage else None)
    ):
        raise DynamicCropCapacityProbeError("smoke final memory differs")
    elapsed_seconds = _nullable_finite_nonnegative(
        execution["elapsed_seconds"], "smoke elapsed seconds"
    )
    if elapsed_seconds != (
        float(final_stage["elapsed_seconds"]) if final_stage else None
    ):
        raise DynamicCropCapacityProbeError("smoke elapsed time differs")
    if (
        execution["optimizer_name"] != "mlx.optimizers.AdamW"
        or optimizer_config
        != {
            "learning_rate": PROBE_TRAINING_SCHEDULE["learning_rate"],
            "weight_decay": PROBE_TRAINING_SCHEDULE["weight_decay"],
        }
        or execution["gradient_clip_norm"]
        != PROBE_TRAINING_SCHEDULE["gradient_clip_norm"]
        or execution["loss_contract_sha256"]
        != candidate.contract_dict()["dynamic_crop_loss_contract_sha256"]
        or execution["allowed_training_bytes"] != budget.allowed_training_bytes
        or execution["memory_limit_contract"] != SMOKE_MEMORY_LIMIT_CONTRACT
        or not isinstance(execution["runtime_execution_passed"], bool)
        or not isinstance(execution["release_trust_authorized"], bool)
        or not isinstance(execution["passed"], bool)
    ):
        raise DynamicCropCapacityProbeError("smoke optimizer/loss contract differs")
    expected_runtime_pass = (
        all(progress)
        and handshake["matched"] is True
        and limits_applied
        and execution["pre_reset_active_memory_bytes"] == 0
        and tuple(observed_stage_order) == expected_stage_order
        and tensor["real_input_materialized"] is True
        and execution["gradients_finite"] is True
        and execution["updated_parameters_finite"] is True
        and execution["optimizer_state_finite"] is True
        and gradient_norm is not None
        and loss_value is not None
        and peak is not None
        and execution["process_exit_code"] == 0
        and execution["termination_signal"] is None
    )
    if execution["runtime_execution_passed"] is not expected_runtime_pass:
        raise DynamicCropCapacityProbeError("smoke runtime outcome is inconsistent")
    if expected_runtime_pass and (
        final_stage is None or peak != final_stage["peak_memory_bytes"]
    ):
        raise DynamicCropCapacityProbeError(
            "successful smoke peak differs from the final optimizer stage"
        )
    if execution["release_trust_authorized"] is not code[
        "release_trust_authorized"
    ]:
        raise DynamicCropCapacityProbeError("smoke trust outcome is inconsistent")
    expected_pass = (
        expected_runtime_pass
        and code["release_trust_authorized"] is True
        and peak is not None
        and peak <= budget.allowed_training_bytes
    )
    if execution["passed"] is not expected_pass:
        raise DynamicCropCapacityProbeError("smoke pass outcome is inconsistent")
    signal_number = execution["termination_signal"]
    exit_code = execution["process_exit_code"]
    if signal_number is not None and (
        isinstance(signal_number, bool)
        or not isinstance(signal_number, int)
        or signal_number <= 0
    ):
        raise DynamicCropCapacityProbeError("smoke termination signal is invalid")
    if exit_code is not None and (
        isinstance(exit_code, bool) or not isinstance(exit_code, int)
    ):
        raise DynamicCropCapacityProbeError("smoke process exit code is invalid")
    if expected_pass:
        expected_failure_kind = None
    elif expected_runtime_pass and peak is not None and peak > budget.allowed_training_bytes:
        expected_failure_kind = "budget_exceeded"
    elif expected_runtime_pass and not code["release_trust_authorized"]:
        expected_failure_kind = "provenance_not_release_trusted"
    elif signal_number is not None:
        expected_failure_kind = "process_signal"
    else:
        expected_failure_kind = execution["failure_kind"]
        if expected_failure_kind not in {
            "out_of_memory",
            "non_finite_loss_or_gradients",
            "runtime_error",
        }:
            raise DynamicCropCapacityProbeError("smoke failure kind is invalid")
    if execution["failure_kind"] != expected_failure_kind:
        raise DynamicCropCapacityProbeError("smoke failure classification differs")

    runner_log_path = _verify_pointer(
        payload["runner_log"], "MLX smoke runner log"
    )
    runner_log = _addressed_payload(
        runner_log_path,
        prefix=SMOKE_RUNNER_LOG_PREFIX,
        label="MLX smoke runner log",
    )
    _exact_keys(
        runner_log,
        {
            "schema_version",
            "analysis_basis",
            "candidate_id",
            "command",
            "supervisor_pid",
            "worker_pid",
            "returncode",
            "stdout",
            "worker_event_count",
            "stdout_event_sha256",
            "stderr",
            "run_nonce_sha256",
            "subprocess_receipt_sha256",
            "started_at_unix_ns",
            "ended_at_unix_ns",
            "suite_id",
            "suite_started_at_unix_ns",
            "suite_ended_at_unix_ns",
            "artifact_content_sha256",
        },
        "MLX smoke runner log",
    )
    command = runner_log["command"]
    returncode = runner_log["returncode"]
    stdout = runner_log["stdout"]
    if (
        runner_log["schema_version"] != SMOKE_RUNNER_LOG_SCHEMA_VERSION
        or runner_log["analysis_basis"] != SMOKE_RUNNER_LOG_ANALYSIS_BASIS
        or runner_log["candidate_id"] != candidate.candidate_id
        or not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) for value in command)
        or isinstance(returncode, bool)
        or not isinstance(returncode, int)
        or not isinstance(stdout, str)
        or not isinstance(runner_log["stderr"], str)
        or any(
            isinstance(runner_log[name], bool)
            or not isinstance(runner_log[name], int)
            or runner_log[name] <= 0
            for name in ("supervisor_pid", "worker_pid", "started_at_unix_ns")
        )
        or isinstance(runner_log["ended_at_unix_ns"], bool)
        or not isinstance(runner_log["ended_at_unix_ns"], int)
        or runner_log["ended_at_unix_ns"] < runner_log["started_at_unix_ns"]
        or not _is_sha256(runner_log["suite_id"])
        or any(
            isinstance(runner_log[name], bool)
            or not isinstance(runner_log[name], int)
            or runner_log[name] <= 0
            for name in (
                "suite_started_at_unix_ns",
                "suite_ended_at_unix_ns",
            )
        )
        or runner_log["suite_started_at_unix_ns"]
        > runner_log["started_at_unix_ns"]
        or runner_log["suite_ended_at_unix_ns"]
        < runner_log["ended_at_unix_ns"]
        or not _is_sha256(runner_log["run_nonce_sha256"])
        or not _is_sha256(runner_log["subprocess_receipt_sha256"])
        or runner_log["subprocess_receipt_sha256"]
        != _sha256(_smoke_subprocess_receipt_core(runner_log))
    ):
        raise DynamicCropCapacityProbeError("MLX smoke runner log contract differs")

    def command_values(flag: str) -> list[str]:
        values: list[str] = []
        for index, value in enumerate(command):
            if value == flag:
                if index + 1 >= len(command) or command[index + 1].startswith("--"):
                    raise DynamicCropCapacityProbeError(
                        f"MLX smoke command {flag} value is missing"
                    )
                values.append(command[index + 1])
        return values

    expected_feature_args = {
        f"{song}={manifests[song]['path']}" for song in SELECTION_SONG_IDS
    }
    expected_materialization_args = {
        f"{song}={materialization_manifests[song]['path']}"
        for song in SELECTION_SONG_IDS
    }
    expected_dynamic_args = {
        f"{song}={dynamic_manifests[song]['path']}"
        for song in SELECTION_SONG_IDS
    }
    expected_command_scalars = {
        "--candidate-id": candidate.candidate_id,
        "--expected-candidate-contract-sha256": _sha256(candidate.to_dict()),
        "--expected-dataset-provenance-sha256": _sha256(dataset),
        "--expected-code-provenance-sha256": _sha256(code),
        "--expected-runtime-provenance-sha256": _sha256(runtime),
        "--expected-hardware-provenance-sha256": _sha256(hardware),
        "--expected-git-commit": code["expected_git_commit"],
        "--tracked-diff-policy": code["tracked_diff_policy"],
        "--suite-id": runner_log["suite_id"],
    }
    if (
        len(command) < 6
        or Path(command[0]).expanduser().resolve()
        != Path(runtime["python_executable"]).expanduser().resolve()
        or command[1:5]
        != ["-P", "-S", "-m", "crosscut.stage_edit_dynamic_crop_mlx_smoke_cli"]
        or command.count("--worker") != 1
        or any(
            command_values(flag) != [expected_value]
            for flag, expected_value in expected_command_scalars.items()
        )
        or set(command_values("--feature-manifest")) != expected_feature_args
        or len(command_values("--feature-manifest")) != 3
        or set(command_values("--materialization-manifest"))
        != expected_materialization_args
        or len(command_values("--materialization-manifest")) != 3
        or set(command_values("--dynamic-crop-manifest")) != expected_dynamic_args
        or len(command_values("--dynamic-crop-manifest")) != 3
    ):
        raise DynamicCropCapacityProbeError(
            "MLX smoke runner command does not bind exact inputs and provenance"
        )
    try:
        from .stage_edit_dynamic_crop_mlx_smoke_cli import (
            _execution_from_process,
            _parse_worker_events,
        )
    except ImportError as exc:  # pragma: no cover - installation corruption.
        raise DynamicCropCapacityProbeError(
            f"MLX smoke replay implementation is unavailable: {exc}"
        ) from exc
    worker_events = _parse_worker_events(stdout)
    if (
        runner_log["worker_event_count"] != len(worker_events)
        or runner_log["stdout_event_sha256"] != _sha256(worker_events)
        or (execution["passed"] is True and not worker_events)
        or any(
            event.get("worker_pid") != runner_log["worker_pid"]
            or event.get("supervisor_run_nonce_sha256")
            != runner_log["run_nonce_sha256"]
            or event.get("supervisor_suite_id") != runner_log["suite_id"]
            for event in worker_events
        )
    ):
        raise DynamicCropCapacityProbeError(
            "MLX smoke runner stdout event digest differs"
        )
    input_events = [
        event
        for event in worker_events
        if event.get("event") == "input_and_limits_ready"
    ]
    provenance_events = [
        event
        for event in worker_events
        if event.get("event") == "provenance_verified"
    ]
    if handshake["matched"] is True and len(provenance_events) != 1:
        raise DynamicCropCapacityProbeError(
            "matched smoke provenance must come from exactly one worker event"
        )
    if provenance_events and (
        provenance_events[0].get("observed_runtime_modules_sha256")
        != _sha256(runtime["imported_runtime_modules"])
        or provenance_events[0].get(
            "observed_runtime_distributions_sha256"
        )
        != _sha256(runtime["runtime_distributions"])
        or provenance_events[0].get("worker_environment_contract")
        != runtime["worker_environment_contract"]
        or provenance_events[0].get("worker_pythonpath_entries")
        != runtime["worker_pythonpath_entries"]
    ):
        raise DynamicCropCapacityProbeError(
            "worker runtime module/environment event binding differs"
        )
    if tensor["real_input_materialized"]:
        if (
            len(input_events) != 1
            or not isinstance(input_events[0].get("tensor_provenance"), Mapping)
            or dict(input_events[0]["tensor_provenance"]) != tensor
        ):
            raise DynamicCropCapacityProbeError(
                "materialized smoke tensor is not derived from exactly one worker event"
            )
    elif len(input_events) > 1 or (
        input_events
        and (
            not isinstance(input_events[0].get("tensor_provenance"), Mapping)
            or dict(input_events[0]["tensor_provenance"]) != tensor
        )
    ):
        raise DynamicCropCapacityProbeError(
            "failed smoke tensor event provenance differs"
        )
    replay_tensor, replay_execution = _execution_from_process(
        candidate=candidate,
        budget=budget,
        code_provenance=code,
        provenance_expectations={
            "candidate_contract_sha256": _sha256(candidate.to_dict()),
            "dataset_provenance_sha256": _sha256(dataset),
            "code_provenance_sha256": _sha256(code),
            "runtime_provenance_sha256": _sha256(runtime),
            "hardware_provenance_sha256": _sha256(hardware),
            "git_commit": code["expected_git_commit"],
        },
        tensor_template=tensor,
        events=worker_events,
        returncode=returncode,
    )
    if replay_tensor != tensor or replay_execution != execution:
        raise DynamicCropCapacityProbeError(
            "MLX smoke evidence differs from replayed runner events"
        )
    raw_attestation = payload["supervisor_attestation"]
    if raw_attestation is None:
        if execution["passed"] is True or require_supervisor_attestation:
            raise DynamicCropCapacityProbeError(
                "selection-authoritative MLX smoke lacks a supervisor subprocess attestation"
            )
    else:
        attestation = _mapping(
            raw_attestation, "MLX smoke supervisor attestation"
        )
        _exact_keys(
            attestation,
            {
                "contract",
                "candidate_id",
                "supervisor_pid",
                "worker_pid",
                "run_nonce_sha256",
                "subprocess_receipt_sha256",
                "runner_log_artifact_content_sha256",
                "runner_source_file_sha256",
                "suite_id",
                "suite_started_at_unix_ns",
                "suite_ended_at_unix_ns",
            },
            "MLX smoke supervisor attestation",
        )
        runner_source = _mapping(
            source_files["runner"], "MLX smoke runner source"
        )
        if attestation != {
            "contract": SMOKE_SUPERVISOR_ATTESTATION_CONTRACT,
            "candidate_id": candidate.candidate_id,
            "supervisor_pid": runner_log["supervisor_pid"],
            "worker_pid": runner_log["worker_pid"],
            "run_nonce_sha256": runner_log["run_nonce_sha256"],
            "subprocess_receipt_sha256": runner_log[
                "subprocess_receipt_sha256"
            ],
            "runner_log_artifact_content_sha256": runner_log[
                "artifact_content_sha256"
            ],
            "runner_source_file_sha256": runner_source["file_sha256"],
            "suite_id": runner_log["suite_id"],
            "suite_started_at_unix_ns": runner_log[
                "suite_started_at_unix_ns"
            ],
            "suite_ended_at_unix_ns": runner_log[
                "suite_ended_at_unix_ns"
            ],
        }:
            raise DynamicCropCapacityProbeError(
                "MLX smoke supervisor attestation differs from process receipt"
            )
        if require_supervisor_attestation:
            actual_repo = Path(__file__).resolve().parents[2]
            _enforce_actual_smoke_source_paths(
                source_files, repo=actual_repo
            )
    return payload


def write_mlx_smoke_evidence(
    destination: Path,
    *,
    candidate: DynamicCropCapacityCandidate,
    budget: DeviceMemoryBudget,
    evidence: Mapping[str, Any],
) -> Path:
    """Publish a failed smoke record; passed records are supervisor-only.

    This public helper intentionally cannot turn caller-supplied dictionaries
    into release-authorizing evidence, even when the caller also fabricates a
    syntactically coherent process log.
    """

    execution = evidence.get("execution")
    if isinstance(execution, Mapping) and execution.get("passed") is True:
        raise DynamicCropCapacityProbeError(
            "passed MLX smoke evidence can only be published by the fresh-process supervisor"
        )
    if evidence.get("supervisor_attestation") is not None:
        raise DynamicCropCapacityProbeError(
            "public smoke writer cannot publish selection-authoritative subprocess attestation"
        )
    if "artifact_content_sha256" in evidence:
        raise DynamicCropCapacityProbeError(
            "caller must not supply the smoke artifact content hash"
        )
    unsigned = {
        "schema_version": SMOKE_EVIDENCE_SCHEMA_VERSION,
        "analysis_basis": SMOKE_EVIDENCE_ANALYSIS_BASIS,
        **dict(evidence),
        "artifact_content_sha256": None,
    }
    digest = _sha256(unsigned)
    payload = {**unsigned, "artifact_content_sha256": digest}
    _validate_mlx_smoke_evidence_root(
        payload, candidate=candidate, budget=budget
    )
    return _atomic_exclusive_write(
        Path(destination).expanduser().resolve()
        / f"{SMOKE_EVIDENCE_PREFIX}{digest}.json",
        payload,
    )


def _publish_supervised_mlx_smoke_evidence(
    destination: Path,
    *,
    candidate: DynamicCropCapacityCandidate,
    budget: DeviceMemoryBudget,
    evidence: Mapping[str, Any],
    runner_log_path: Path,
    run_nonce: str,
) -> Path:
    """Consume a live child receipt from the exact supervisor call frame."""

    caller = inspect.currentframe().f_back  # type: ignore[union-attr]
    expected_runner = (
        Path(__file__).resolve().parent
        / "stage_edit_dynamic_crop_mlx_smoke_cli.py"
    ).resolve()
    if (
        caller is None
        or Path(caller.f_code.co_filename).resolve() != expected_runner
        or caller.f_code.co_name != "_supervisor_main"
    ):
        raise DynamicCropCapacityProbeError(
            "supervised MLX smoke publication requires the real supervisor call frame"
        )
    if (
        not isinstance(run_nonce, str)
        or len(run_nonce) != 64
        or any(character not in "0123456789abcdef" for character in run_nonce)
    ):
        raise DynamicCropCapacityProbeError("supervisor run nonce is invalid")
    log_path = _regular_absolute(runner_log_path, "MLX smoke runner log")
    log = _addressed_payload(
        log_path,
        prefix=SMOKE_RUNNER_LOG_PREFIX,
        label="MLX smoke runner log",
    )
    nonce_sha256 = hashlib.sha256(run_nonce.encode("ascii")).hexdigest()
    runner_pointer = _mapping(evidence.get("runner_log"), "MLX smoke runner log")
    if (
        log.get("candidate_id") != candidate.candidate_id
        or log.get("run_nonce_sha256") != nonce_sha256
        or runner_pointer
        != {"path": str(log_path), "file_sha256": _file_sha256(log_path)}
    ):
        raise DynamicCropCapacityProbeError(
            "supervisor child receipt differs from the bound runner log"
        )

    if (
        "artifact_content_sha256" in evidence
        or "supervisor_attestation" in evidence
    ):
        raise DynamicCropCapacityProbeError(
            "caller must not supply smoke content hash or supervisor attestation"
        )
    source_files = _mapping(
        _mapping(evidence.get("code_provenance"), "smoke code provenance").get(
            "source_files"
        ),
        "smoke source files",
    )
    runner_source = _mapping(source_files.get("runner"), "smoke runner source")
    attestation = {
        "contract": SMOKE_SUPERVISOR_ATTESTATION_CONTRACT,
        "candidate_id": candidate.candidate_id,
        "supervisor_pid": log.get("supervisor_pid"),
        "worker_pid": log.get("worker_pid"),
        "run_nonce_sha256": nonce_sha256,
        "subprocess_receipt_sha256": log.get("subprocess_receipt_sha256"),
        "runner_log_artifact_content_sha256": log.get(
            "artifact_content_sha256"
        ),
        "runner_source_file_sha256": runner_source.get("file_sha256"),
        "suite_id": log.get("suite_id"),
        "suite_started_at_unix_ns": log.get("suite_started_at_unix_ns"),
        "suite_ended_at_unix_ns": log.get("suite_ended_at_unix_ns"),
    }
    unsigned = {
        "schema_version": SMOKE_EVIDENCE_SCHEMA_VERSION,
        "analysis_basis": SMOKE_EVIDENCE_ANALYSIS_BASIS,
        **dict(evidence),
        "supervisor_attestation": attestation,
        "artifact_content_sha256": None,
    }
    digest = _sha256(unsigned)
    payload = {**unsigned, "artifact_content_sha256": digest}
    _validate_mlx_smoke_evidence_root(
        payload, candidate=candidate, budget=budget
    )
    return _atomic_exclusive_write(
        Path(destination).expanduser().resolve()
        / f"{SMOKE_EVIDENCE_PREFIX}{digest}.json",
        payload,
    )


def _load_mlx_smoke_evidence(
    path: Path,
    *,
    candidate: DynamicCropCapacityCandidate,
    budget: DeviceMemoryBudget,
) -> dict[str, Any]:
    root = _addressed_payload(
        path, prefix=SMOKE_EVIDENCE_PREFIX, label="MLX smoke evidence"
    )
    return _validate_mlx_smoke_evidence_root(
        root,
        candidate=candidate,
        budget=budget,
        require_supervisor_attestation=True,
    )


def build_dynamic_crop_capacity_selection(
    *,
    candidates: Sequence[DynamicCropCapacityCandidate],
    fold_run_evidence_paths: Sequence[Path],
    selection_dataset_manifest_path: Path,
    mlx_smoke_evidence_paths_by_candidate: Mapping[str, Path],
    device_memory_budget: DeviceMemoryBudget,
    baseline_candidate_id: str,
    selection_objective: SelectionObjective = SelectionObjective(),
    selection_policy: SelectionPolicy = SelectionPolicy(),
) -> dict[str, Any]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if not candidates or len(candidate_by_id) != len(candidates) or len({c.candidate_name for c in candidates}) != len(candidates):
        raise DynamicCropCapacityProbeError("candidate identities/names must be nonempty and unique")
    _enforce_release_capacity_candidate_set(
        candidate_by_id, baseline_candidate_id=baseline_candidate_id
    )
    if baseline_candidate_id not in candidate_by_id:
        raise DynamicCropCapacityProbeError("baseline candidate is absent")
    if set(mlx_smoke_evidence_paths_by_candidate) != set(candidate_by_id):
        raise DynamicCropCapacityProbeError("every candidate needs real MLX smoke evidence")
    dataset_path = _regular_absolute(selection_dataset_manifest_path, "selection dataset")
    dataset = _load_selection_dataset(dataset_path)

    static: dict[str, Any] = {}
    smoke: dict[str, Any] = {}
    smoke_pointers: dict[str, Any] = {}
    for candidate_id, candidate in candidate_by_id.items():
        static[candidate_id] = _memory_preflight(candidate, device_memory_budget)
        smoke_path = mlx_smoke_evidence_paths_by_candidate[candidate_id]
        smoke[candidate_id] = _load_mlx_smoke_evidence(
            smoke_path, candidate=candidate, budget=device_memory_budget
        )
        smoke_materialization_manifests = _mapping(
            smoke[candidate_id]["dataset_provenance"][
                "materialization_manifests"
            ],
            "smoke materialization manifests",
        )
        if smoke_materialization_manifests != dict(
            dataset.materialization_manifests_by_song
        ):
            raise DynamicCropCapacityProbeError(
                "smoke materialization manifests differ from selection dataset"
            )
        smoke_pointers[candidate_id] = _pointer(smoke_path, "MLX smoke evidence")

    suite_envelopes = {
        _sha256(
            {
                "suite_id": smoke[candidate_id]["supervisor_attestation"][
                    "suite_id"
                ],
                "supervisor_pid": smoke[candidate_id][
                    "supervisor_attestation"
                ]["supervisor_pid"],
                "suite_started_at_unix_ns": smoke[candidate_id][
                    "supervisor_attestation"
                ]["suite_started_at_unix_ns"],
                "suite_ended_at_unix_ns": smoke[candidate_id][
                    "supervisor_attestation"
                ]["suite_ended_at_unix_ns"],
            }
        )
        for candidate_id in candidate_by_id
    }
    if len(suite_envelopes) != 1:
        raise DynamicCropCapacityProbeError(
            "all four MLX smoke records must come from one supervisor suite"
        )
    smoke_suite_attestation = smoke[next(iter(candidate_by_id))][
        "supervisor_attestation"
    ]

    memory_eligible_candidate_ids = {
        candidate_id
        for candidate_id in candidate_by_id
        if static[candidate_id]["passed_static_bound"] is True
        and smoke[candidate_id]["execution"]["passed"] is True
    }
    if baseline_candidate_id not in memory_eligible_candidate_ids:
        raise DynamicCropCapacityProbeError(
            "baseline candidate must pass static and real MLX memory gates"
        )
    if len(fold_run_evidence_paths) != len(memory_eligible_candidate_ids) * 3:
        raise DynamicCropCapacityProbeError(
            "every memory-eligible candidate needs exactly three fold reports"
        )
    evaluations: dict[tuple[str, str], FoldEvaluation] = {}
    fold_pointers: list[dict[str, str]] = []
    for path in fold_run_evidence_paths:
        raw = _addressed_payload(path, prefix=FOLD_EVIDENCE_PREFIX, label="fold run evidence")
        candidate_id = str(raw.get("candidate_id") or "")
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise DynamicCropCapacityProbeError("fold names unknown candidate")
        if candidate_id not in memory_eligible_candidate_ids:
            raise DynamicCropCapacityProbeError(
                "memory-ineligible candidate must not consume fold training"
            )
        evaluation = _load_fold_evaluation(path, candidate=candidate, dataset=dataset)
        key = (candidate_id, evaluation.validation_song_id)
        if key in evaluations:
            raise DynamicCropCapacityProbeError("duplicate candidate/song fold")
        evaluations[key] = evaluation
        fold_pointers.append(_pointer(path, "fold run evidence"))
    expected_pairs = {
        (candidate_id, song)
        for candidate_id in memory_eligible_candidate_ids
        for song in SELECTION_SONG_IDS
    }
    if set(evaluations) != expected_pairs:
        raise DynamicCropCapacityProbeError("rotating whole-song folds are incomplete")

    scores: dict[str, list[float]] = {
        candidate_id: [
            selection_objective.score(evaluations[(candidate_id, song)].metrics)
            for song in SELECTION_SONG_IDS
        ]
        for candidate_id in memory_eligible_candidate_ids
    }
    baseline_scores = scores[baseline_candidate_id]
    ranking: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for candidate_id, candidate in candidate_by_id.items():
        reasons: list[str] = []
        if not static[candidate_id]["passed_static_bound"]:
            reasons.append("static_capacity_bound_failed")
        if smoke[candidate_id]["execution"]["passed"] is not True:
            reasons.append("real_mlx_smoke_failed")
        if candidate_id in memory_eligible_candidate_ids:
            values = scores[candidate_id]
            mean: float | None = math.fsum(values) / len(values)
            variance: float | None = (
                math.fsum((value - mean) ** 2 for value in values) / len(values)
            )
            paired: list[float] | None = [
                value - baseline
                for value, baseline in zip(values, baseline_scores, strict=True)
            ]
            if any(
                value > selection_policy.maximum_fold_selection_loss
                for value in values
            ):
                reasons.append("absolute_fold_loss_gate_failed")
            if any(
                value < selection_policy.minimum_fold_selection_loss
                for value in values
            ):
                reasons.append("implausible_zero_fold_loss_gate_failed")
            if variance > selection_policy.maximum_fold_variance:
                reasons.append("fold_variance_gate_failed")
            if candidate_id != baseline_candidate_id and any(
                delta > selection_policy.maximum_paired_regression
                for delta in paired
            ):
                reasons.append("paired_fold_regression_gate_failed")
            fold_losses: dict[str, float] | None = dict(
                zip(SELECTION_SONG_IDS, values, strict=True)
            )
            paired_losses: dict[str, float] | None = dict(
                zip(SELECTION_SONG_IDS, paired, strict=True)
            )
            worst: float | None = max(values)
        else:
            mean = None
            variance = None
            fold_losses = None
            paired_losses = None
            worst = None
        row = {
            "candidate_id": candidate_id,
            "candidate_name": candidate.candidate_name,
            "mean_selection_loss": mean,
            "worst_fold_selection_loss": worst,
            "fold_selection_loss_variance": variance,
            "fold_selection_losses": fold_losses,
            "paired_regressions_vs_baseline": paired_losses,
            "parameter_count": static[candidate_id]["parameter_count"],
            "eligible": not reasons,
            "rejection_reasons": reasons,
        }
        ranking.append(row)
        if not reasons:
            eligible.append(row)
    if not eligible:
        raise DynamicCropCapacityProbeError("no candidate passes fold gates")
    best_mean = min(row["mean_selection_loss"] for row in eligible)
    equivalent = [row for row in eligible if row["mean_selection_loss"] <= best_mean + selection_policy.equivalence_tolerance]
    selected_row = min(equivalent, key=lambda row: (row["parameter_count"], row["candidate_id"]))
    ranking.sort(
        key=lambda row: (
            not row["eligible"],
            0 if row in equivalent else 1,
            (
                row["parameter_count"]
                if row in equivalent
                else (
                    row["mean_selection_loss"]
                    if row["mean_selection_loss"] is not None
                    else math.inf
                )
            ),
            row["candidate_id"],
        )
    )
    selected = candidate_by_id[str(selected_row["candidate_id"])]
    identity = {
        "dataset": dataset.artifact_content_sha256,
        "candidate_ids": sorted(candidate_by_id),
        "fold_evidence": sorted(value.evidence_artifact_content_sha256 for value in evaluations.values()),
        "smoke_evidence": {key: smoke[key]["artifact_content_sha256"] for key in sorted(smoke)},
        "objective": selection_objective.to_dict(),
        "policy": selection_policy.to_dict(),
        "baseline": baseline_candidate_id,
        "selected": selected.candidate_id,
    }
    selection_identity = _sha256(identity)
    unsigned: dict[str, Any] = {
        "schema_version": CAPACITY_PROBE_SCHEMA_VERSION,
        "analysis_basis": CAPACITY_PROBE_ANALYSIS_BASIS,
        "training_launched_by_probe": False,
        "quarantined_song_ids_opened": False,
        "selection_dataset_manifest": _pointer(dataset_path, "selection dataset"),
        "selection_dataset_content_sha256": dataset.artifact_content_sha256,
        "selection_protocol": {
            "song_ids": list(SELECTION_SONG_IDS),
            "fold_count_per_candidate": 3,
            "fold_training_candidate_scope": "static_and_real_mlx_smoke_pass_only",
            "memory_eligible_candidate_ids": sorted(
                memory_eligible_candidate_ids
            ),
            "smoke_suite_id": smoke_suite_attestation["suite_id"],
            "smoke_suite_candidate_ids": sorted(candidate_by_id),
            "fold_unit": "whole_song",
            "heldout_used_for_model_or_threshold_selection": False,
            "xoxz_development_evaluation_consumed": False,
            "attitude_final_evaluation_consumed": False,
        },
        "token_schema_fingerprint": DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
        "device_memory_budget": device_memory_budget.to_dict(),
        "static_capacity_preflights": {key: static[key] for key in sorted(static)},
        "real_mlx_smoke_evidence": {key: smoke_pointers[key] for key in sorted(smoke_pointers)},
        "fold_run_evidence": sorted(fold_pointers, key=lambda item: item["path"]),
        "candidates": [candidate_by_id[key].to_dict() for key in sorted(candidate_by_id)],
        "fold_evaluations": [evaluations[key].to_dict() for key in sorted(evaluations)],
        "selection_objective": selection_objective.to_dict(),
        "selection_policy": selection_policy.to_dict(),
        "baseline_candidate_id": baseline_candidate_id,
        "ranking": ranking,
        "selected_candidate_id": selected.candidate_id,
        "selection_identity_sha256": selection_identity,
        "release_model_training_contract": selected.release_model_training_contract(),
        "release_model_training_contract_origin": {
            "capacity_probe_schema_version": CAPACITY_PROBE_SCHEMA_VERSION,
            "selection_identity_sha256": selection_identity,
            "selected_candidate_id": selected.candidate_id,
        },
        "artifact_content_sha256": None,
    }
    digest = _sha256(unsigned)
    return {**unsigned, "artifact_content_sha256": digest}


def _enforce_release_capacity_candidate_set(
    candidate_by_id: Mapping[str, DynamicCropCapacityCandidate],
    *,
    baseline_candidate_id: str,
) -> None:
    """Prevent a release selection from silently omitting competitors."""

    defaults = default_dynamic_crop_v2_probe_candidates()
    expected = {candidate.candidate_id: candidate for candidate in defaults}
    if set(candidate_by_id) != set(expected) or any(
        candidate_by_id[candidate_id].to_dict()
        != expected[candidate_id].to_dict()
        for candidate_id in expected
    ):
        raise DynamicCropCapacityProbeError(
            "release capacity selection requires the exact four pinned candidates"
        )
    baseline_candidates = tuple(
        candidate
        for candidate in defaults
        if candidate.candidate_name == "compact-balanced"
    )
    if len(baseline_candidates) != 1:
        raise DynamicCropCapacityProbeError(
            "pinned compact-balanced baseline definition is invalid"
        )
    pinned_baseline = baseline_candidates[0].candidate_id
    if baseline_candidate_id != pinned_baseline:
        raise DynamicCropCapacityProbeError(
            "release capacity selection baseline must be compact-balanced"
        )


def write_dynamic_crop_capacity_selection(destination: Path, **kwargs: Any) -> Path:
    payload = build_dynamic_crop_capacity_selection(**kwargs)
    digest = payload["artifact_content_sha256"]
    return _atomic_exclusive_write(
        Path(destination).expanduser().resolve() / f"{SELECTION_ARTIFACT_PREFIX}{digest}.json",
        payload,
    )


def load_dynamic_crop_capacity_selection(path: Path) -> dict[str, Any]:
    root = _addressed_payload(path, prefix=SELECTION_ARTIFACT_PREFIX, label="capacity selection")
    if root.get("schema_version") != CAPACITY_PROBE_SCHEMA_VERSION or root.get("analysis_basis") != CAPACITY_PROBE_ANALYSIS_BASIS:
        raise DynamicCropCapacityProbeError("capacity selection schema mismatch")
    candidates = tuple(DynamicCropCapacityCandidate.from_dict(value) for value in root.get("candidates", ()))
    rebuilt = build_dynamic_crop_capacity_selection(
        candidates=candidates,
        fold_run_evidence_paths=tuple(Path(item["path"]) for item in root.get("fold_run_evidence", ())),
        selection_dataset_manifest_path=Path(root["selection_dataset_manifest"]["path"]),
        mlx_smoke_evidence_paths_by_candidate={
            key: Path(value["path"]) for key, value in root.get("real_mlx_smoke_evidence", {}).items()
        },
        device_memory_budget=DeviceMemoryBudget(**_mapping(root.get("device_memory_budget"), "device budget")),
        baseline_candidate_id=str(root.get("baseline_candidate_id") or ""),
        selection_objective=SelectionObjective(**_mapping(root.get("selection_objective"), "selection objective")),
        selection_policy=SelectionPolicy(**_mapping(root.get("selection_policy"), "selection policy")),
    )
    if rebuilt != root:
        raise DynamicCropCapacityProbeError("capacity selection deterministic replay failed")
    return root


def default_dynamic_crop_v2_probe_candidates() -> tuple[DynamicCropCapacityCandidate, ...]:
    authority = StageEditTransitionAuthority(
        mode=RENDERER_OWNED_TRANSITION_AUTHORITY
    )
    base = StageEditLossContract(
        transition_weight=0.0,
        transition_authority=authority.mode,
    )
    configurations = (
        ("compact-balanced", 512, 8, 10, 2048, {}),
        ("intermediate-balanced", 640, 10, 12, 2560, {}),
        ("production-balanced", 768, 12, 14, 3072, {}),
        (
            "production-trajectory-emphasis",
            768,
            12,
            14,
            3072,
            {"value_weight": 1.15, "time_weight": 0.65, "presence_weight": 0.40, "motion_weight": 0.25},
        ),
    )
    return tuple(
        DynamicCropCapacityCandidate(
            candidate_name=name,
            model_config=StageEditTransformerConfig.from_schema(
                DEFAULT_STAGE_EDIT_SCHEMA,
                model_dimension=dimension,
                num_heads=heads,
                num_layers=layers,
                feed_forward_dimension=feed_forward,
                max_sequence_length=2048,
                max_sources=32,
                dropout=0.1,
                transition_authority=authority.mode,
            ),
            dynamic_crop_config=DynamicCropHeadV2Config(),
            base_loss_contract=base,
            dynamic_crop_loss_contract=DynamicCropV2LossContract(base=base, **loss_kwargs),
        )
        for name, dimension, heads, layers, feed_forward, loss_kwargs in configurations
    )


__all__ = [
    "BANG_BANG", "BLACKHOLE", "REBEL_HEART", "SELECTION_SONG_IDS",
    "SMOKE_EVIDENCE_ANALYSIS_BASIS", "SMOKE_EVIDENCE_PREFIX",
    "SMOKE_INPUT_PACKING_CONTRACT", "SMOKE_MEMORY_LIMIT_CONTRACT",
    "SMOKE_PROCESS_ISOLATION_CONTRACT", "SMOKE_RUNNER_CONTRACT",
    "SMOKE_TILING_CONTRACT",
    "DeviceMemoryBudget", "DynamicCropCapacityCandidate",
    "DynamicCropCapacityProbeError", "FoldEvaluation", "SelectionObjective",
    "SelectionPolicy", "build_dynamic_crop_capacity_selection",
    "default_dynamic_crop_v2_probe_candidates", "load_dynamic_crop_capacity_selection",
    "write_dynamic_crop_capacity_selection", "write_fold_run_evidence",
    "write_mlx_smoke_evidence",
    "write_selection_dataset_manifest",
]
