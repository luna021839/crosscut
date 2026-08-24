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
import json
import math
import os
from pathlib import Path
import secrets
import stat
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
MEMORY_ESTIMATE_CONTRACT = "non_authoritative_float32_source720_bound_v2"
SELECTION_ARTIFACT_PREFIX = "dynamic-crop-v2-capacity-selection-"
DATASET_ARTIFACT_PREFIX = "dynamic-crop-v2-selection-dataset-"
FOLD_EVIDENCE_PREFIX = "dynamic-crop-v2-fold-run-"
SMOKE_EVIDENCE_PREFIX = "dynamic-crop-v2-mlx-smoke-"

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
    "early_stopping_patience": 6,
    "batch_size": 2,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-3,
    "gradient_clip_norm": 1.0,
    "cut_pretraining_steps": 500,
    "cut_pretraining_peak_learning_rate": 3.0e-4,
    "cut_pretraining_window_length": 32,
    "cut_pretraining_warmup_steps": 50,
    "finetune_warmup_steps": 8,
    "minimum_learning_rate_ratio": 0.10,
    "schedule_contract": "independent_warmup_cosine_pretrain_and_finetune_v1",
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
    path = _regular_absolute(Path(str(pointer["path"])), label)
    if not _is_sha256(pointer["file_sha256"]):
        raise DynamicCropCapacityProbeError(f"{label} hash is invalid")
    if _file_sha256(path) != pointer["file_sha256"]:
        raise DynamicCropCapacityProbeError(f"{label} hash mismatch")
    return path


def _reject_quarantined_text(value: object, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise DynamicCropCapacityProbeError(f"{label} cannot be blank")
    compact = "".join(character for character in text.upper() if character.isalnum())
    for song in QUARANTINED_SONG_IDS:
        song_compact = "".join(character for character in song if character.isalnum())
        if song_compact in compact:
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
    return _SelectionDataset(str(root["artifact_content_sha256"]), references)


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


def _validate_schedule(report: Mapping[str, Any]) -> None:
    optimization = _mapping(report.get("optimization"), "optimization")
    schedule = _mapping(optimization.get("learning_rate_schedule"), "learning-rate schedule")
    observed = {
        "seed": optimization.get("seed"),
        "requested_epochs": optimization.get("requested_epochs"),
        "early_stopping_patience": optimization.get("early_stopping_patience"),
        "batch_size": optimization.get("batch_size"),
        "learning_rate": optimization.get("learning_rate"),
        "weight_decay": optimization.get("weight_decay"),
        "gradient_clip_norm": optimization.get("gradient_clip_norm"),
        "cut_pretraining_steps": schedule.get("cut_pretraining_steps"),
        "cut_pretraining_peak_learning_rate": schedule.get("cut_pretraining_peak_learning_rate"),
        "cut_pretraining_window_length": schedule.get("cut_pretraining_window_length"),
        "cut_pretraining_warmup_steps": schedule.get("cut_pretraining_warmup_steps"),
        "finetune_warmup_steps": schedule.get("finetune_warmup_steps"),
        "minimum_learning_rate_ratio": schedule.get("minimum_learning_rate_ratio"),
        "schedule_contract": schedule.get("contract"),
    }
    if observed != dict(PROBE_TRAINING_SCHEDULE):
        raise DynamicCropCapacityProbeError("fold training seed/schedule differs")


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

    checkpoint = _mapping(report.get("checkpoint"), "report checkpoint")
    heldout = _mapping(report.get("held_out_evaluation"), "heldout evaluation")
    quality = _mapping(heldout.get("quality_gate"), "heldout quality gate")
    coverage = _mapping(report.get("structured_head_coverage_gate"), "structured coverage gate")
    if (
        checkpoint.get("checkpoint_id") != metadata.checkpoint_id
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

    report_freeze = _mapping(report.get("selection_freeze_evidence"), "freeze evidence")
    if (
        report_freeze.get("path") != str(freeze_path)
        or report_freeze.get("file_sha256") != _file_sha256(freeze_path)
        or report_freeze.get("written_before_heldout_forward") is not True
        or freeze.get("model_config") != candidate.model_config.to_dict()
        or freeze.get("model_config_fingerprint_sha256") != _sha256(candidate.model_config.to_dict())
        or freeze.get("checkpoint_id") != metadata.checkpoint_id
        or freeze.get("checkpoint_weight_sha256") != metadata.weight_sha256
        or freeze.get("training_song_ids") != list(training_songs)
        or tuple(sorted(freeze.get("training_reference_ids") or ())) != train_refs
        or tuple(sorted(freeze.get("validation_reference_ids") or ())) != internal_validation_refs
        or freeze.get("held_out_song_ids_reserved_for_one_forward") != [heldout_song]
        or freeze.get("heldout_metrics_present") is not False
        or freeze.get("heldout_used_for_model_or_threshold_selection") is not False
        or freeze.get("style_or_attitude_configuration_mutable_after_freeze") is not False
    ):
        raise DynamicCropCapacityProbeError("selection freeze provenance failed")
    _validate_schedule(report)
    _validate_effective_losses(report, candidate)
    if _mapping(report.get("optimization"), "optimization").get("unseen_song_used_for_model_selection") is not False:
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


def _load_mlx_smoke_evidence(
    path: Path,
    *,
    candidate: DynamicCropCapacityCandidate,
    budget: DeviceMemoryBudget,
) -> dict[str, Any]:
    root = _addressed_payload(path, prefix=SMOKE_EVIDENCE_PREFIX, label="MLX smoke evidence")
    _exact_keys(
        root,
        {
            "schema_version", "analysis_basis", "candidate_id", "model_config_sha256",
            "device_name", "physical_memory_bytes", "max_recommended_working_set_size",
            "max_buffer_length", "architecture", "mlx_version", "os_reserve_bytes", "batch_size",
            "dtype_bytes", "sequence_length", "max_sources", "model_allocated",
            "forward_completed", "backward_completed", "optimizer_step_completed",
            "optimizer_state_materialized", "measured_peak_bytes", "passed",
            "failure_kind", "process_exit_code", "termination_signal",
            "runner_log_sha256",
            "artifact_content_sha256",
        },
        "MLX smoke evidence",
    )
    peak = root.get("measured_peak_bytes")
    if peak is not None and (
        isinstance(peak, bool)
        or not isinstance(peak, int)
        or peak <= 0
        or peak > budget.physical_memory_bytes
    ):
        raise DynamicCropCapacityProbeError("MLX peak memory is invalid")
    progress_names = (
        "model_allocated", "forward_completed", "backward_completed",
        "optimizer_step_completed", "optimizer_state_materialized",
    )
    if any(not isinstance(root[name], bool) for name in progress_names):
        raise DynamicCropCapacityProbeError("MLX progress flags must be booleans")
    progress = tuple(bool(root[name]) for name in progress_names)
    if any(progress[index] and not progress[index - 1] for index in range(1, len(progress))):
        raise DynamicCropCapacityProbeError("MLX progress flags are not monotonic")
    passed = root.get("passed")
    if not isinstance(passed, bool) or not _is_sha256(root.get("runner_log_sha256")):
        raise DynamicCropCapacityProbeError("MLX outcome provenance is invalid")
    if (
        root["schema_version"] != 1
        or root["analysis_basis"] != SMOKE_EVIDENCE_ANALYSIS_BASIS
        or root["candidate_id"] != candidate.candidate_id
        or root["model_config_sha256"] != _sha256(candidate.model_config.to_dict())
        or root["device_name"] != budget.device_name
        or root["physical_memory_bytes"] != budget.physical_memory_bytes
        or root["max_recommended_working_set_size"]
        != budget.max_recommended_working_set_size
        or root["max_buffer_length"] != budget.max_buffer_length
        or root["architecture"] != budget.architecture
        or root["mlx_version"] != budget.mlx_version
        or root["os_reserve_bytes"] != budget.os_reserve_bytes
        or root["batch_size"] != 2
        or root["dtype_bytes"] != 4
        or root["sequence_length"] != candidate.model_config.max_sequence_length
        or root["max_sources"] != candidate.model_config.max_sources
    ):
        raise DynamicCropCapacityProbeError("real MLX smoke evidence contract differs")
    if passed:
        if (
            not all(progress)
            or peak is None
            or peak > budget.allowed_training_bytes
            or root.get("failure_kind") is not None
            or root.get("process_exit_code") != 0
            or root.get("termination_signal") is not None
        ):
            raise DynamicCropCapacityProbeError("passing MLX smoke evidence is inconsistent")
    else:
        failure_kind = root.get("failure_kind")
        exit_code = root.get("process_exit_code")
        signal_number = root.get("termination_signal")
        if (
            failure_kind not in {
                "out_of_memory", "budget_exceeded", "process_signal", "runtime_error"
            }
            or (exit_code is not None and (
                isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code == 0
            ))
            or (signal_number is not None and (
                isinstance(signal_number, bool)
                or not isinstance(signal_number, int)
                or signal_number <= 0
            ))
            or (failure_kind == "process_signal" and signal_number is None)
            or (failure_kind == "budget_exceeded" and (
                peak is None or peak <= budget.allowed_training_bytes
            ))
        ):
            raise DynamicCropCapacityProbeError("failed MLX smoke evidence is inconsistent")
    return root


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
        smoke_pointers[candidate_id] = _pointer(smoke_path, "MLX smoke evidence")

    if len(fold_run_evidence_paths) != len(candidate_by_id) * 3:
        raise DynamicCropCapacityProbeError("every candidate needs exactly three fold reports")
    evaluations: dict[tuple[str, str], FoldEvaluation] = {}
    fold_pointers: list[dict[str, str]] = []
    for path in fold_run_evidence_paths:
        raw = _addressed_payload(path, prefix=FOLD_EVIDENCE_PREFIX, label="fold run evidence")
        candidate_id = str(raw.get("candidate_id") or "")
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise DynamicCropCapacityProbeError("fold names unknown candidate")
        evaluation = _load_fold_evaluation(path, candidate=candidate, dataset=dataset)
        key = (candidate_id, evaluation.validation_song_id)
        if key in evaluations:
            raise DynamicCropCapacityProbeError("duplicate candidate/song fold")
        evaluations[key] = evaluation
        fold_pointers.append(_pointer(path, "fold run evidence"))
    expected_pairs = {(candidate_id, song) for candidate_id in candidate_by_id for song in SELECTION_SONG_IDS}
    if set(evaluations) != expected_pairs:
        raise DynamicCropCapacityProbeError("rotating whole-song folds are incomplete")

    scores = {
        candidate_id: [
            selection_objective.score(evaluations[(candidate_id, song)].metrics)
            for song in SELECTION_SONG_IDS
        ]
        for candidate_id in candidate_by_id
    }
    baseline_scores = scores[baseline_candidate_id]
    ranking: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for candidate_id, candidate in candidate_by_id.items():
        values = scores[candidate_id]
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
        paired = [value - baseline for value, baseline in zip(values, baseline_scores, strict=True)]
        reasons: list[str] = []
        if not static[candidate_id]["passed_static_bound"]:
            reasons.append("static_capacity_bound_failed")
        if smoke[candidate_id]["passed"] is not True:
            reasons.append("real_mlx_smoke_failed")
        if any(value > selection_policy.maximum_fold_selection_loss for value in values):
            reasons.append("absolute_fold_loss_gate_failed")
        if any(value < selection_policy.minimum_fold_selection_loss for value in values):
            reasons.append("implausible_zero_fold_loss_gate_failed")
        if variance > selection_policy.maximum_fold_variance:
            reasons.append("fold_variance_gate_failed")
        if candidate_id != baseline_candidate_id and any(
            delta > selection_policy.maximum_paired_regression for delta in paired
        ):
            reasons.append("paired_fold_regression_gate_failed")
        row = {
            "candidate_id": candidate_id,
            "candidate_name": candidate.candidate_name,
            "mean_selection_loss": mean,
            "worst_fold_selection_loss": max(values),
            "fold_selection_loss_variance": variance,
            "fold_selection_losses": dict(zip(SELECTION_SONG_IDS, values, strict=True)),
            "paired_regressions_vs_baseline": dict(zip(SELECTION_SONG_IDS, paired, strict=True)),
            "parameter_count": static[candidate_id]["parameter_count"],
            "eligible": not reasons,
            "rejection_reasons": reasons,
        }
        ranking.append(row)
        if not reasons:
            eligible.append(row)
    if smoke[baseline_candidate_id]["passed"] is not True:
        raise DynamicCropCapacityProbeError("baseline candidate real MLX smoke failed")
    if not eligible:
        raise DynamicCropCapacityProbeError("no candidate passes fold gates")
    best_mean = min(row["mean_selection_loss"] for row in eligible)
    equivalent = [row for row in eligible if row["mean_selection_loss"] <= best_mean + selection_policy.equivalence_tolerance]
    selected_row = min(equivalent, key=lambda row: (row["parameter_count"], row["candidate_id"]))
    ranking.sort(
        key=lambda row: (
            not row["eligible"],
            0 if row in equivalent else 1,
            row["parameter_count"] if row in equivalent else row["mean_selection_loss"],
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
    "DeviceMemoryBudget", "DynamicCropCapacityCandidate",
    "DynamicCropCapacityProbeError", "FoldEvaluation", "SelectionObjective",
    "SelectionPolicy", "build_dynamic_crop_capacity_selection",
    "default_dynamic_crop_v2_probe_candidates", "load_dynamic_crop_capacity_selection",
    "write_dynamic_crop_capacity_selection", "write_fold_run_evidence",
    "write_selection_dataset_manifest",
]
