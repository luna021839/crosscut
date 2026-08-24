"""Fail-closed learned reranking for retained visual timeline candidates.

This is deliberately a first-stage reranker, not a timeline decoder.  The
visual optimiser still creates the same deterministic top-32 paths and the
existing decoded-proxy review plus professional reference gate keep final
authority.  Learned inference only changes the order in which those already
safe paths are sent to the expensive eight-proxy stage.

Activation is explicit and strict: a caller must name both the checkpoint
directory and its exact colocated production training report, and the public
``stage_edit_readiness_report`` contract must accept that checkpoint.  Sparse
or merely loadable weights are an error, never a silent fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .section_director import SectionPlan
from .stage_edit_feature_materializer import (
    CausalEditDecision,
    SourceFeatureObservation,
    StageEditFeatureFrame,
    StageEditFeatureProvenance,
    StageEditFeatureSnapshot,
    materialize_stage_edit_feature_snapshot,
    source_observation_from_timeline_candidate,
)
from .stage_edit_transformer import (
    PRODUCTION_TRAINING_REPORT_FILENAME,
    StageEditCheckpointMetadata,
    StageEditReadinessReport,
    StageEditTokenSchema,
    load_stage_edit_checkpoint,
    mask_stage_edit_recent_history_for_semimarkov,
    stage_edit_readiness_report,
    tensorize_stage_edit_batch,
)
from .timeline import TimelineCandidate
from .visual_score import (
    VisualTimelineCandidateRecord,
    VisualTimelineCandidateSet,
)


STAGE_EDIT_CANDIDATE_RERANK_SCHEMA_VERSION = 1
STAGE_EDIT_CANDIDATE_RERANK_BASIS = (
    "production_stage_edit_transformer_retained_top32_rerank_v1"
)
STAGE_EDIT_RERANK_FEATURE_CONTRACT = (
    "inference_features_with_recent_edit_slice_scrubbed_before_model_v2"
)
STAGE_EDIT_RERANK_MICROBATCH_SIZE = 4


class StageEditCandidateRerankError(RuntimeError):
    """Base error for explicitly requested learned inference."""


class StageEditCandidateRerankNotReadyError(StageEditCandidateRerankError):
    """The requested checkpoint does not satisfy production readiness."""


class StageEditCandidateRerankInputError(StageEditCandidateRerankError):
    """The retained paths cannot be represented by the checkpoint contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise StageEditCandidateRerankInputError(f"{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class StageEditCandidateRerankWeights:
    """Inference weights mirroring the structured training-head balance."""

    cut: float = 1.0
    source: float = 1.0
    duration: float = 0.5
    transition: float = 0.5
    crop: float = 0.25

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in asdict(self).values())
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("StageEdit rerank weights must be finite and non-negative")
        if not any(value > 0.0 for value in values):
            raise ValueError("at least one StageEdit rerank weight must be positive")

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class StageEditCandidateComponentScore:
    candidate_index: int
    original_review_rank: int
    rerank_rank: int
    rank_change: int
    total_score: float
    cut_log_probability: float
    source_log_probability: float
    duration_log_probability: float
    transition_log_probability: float
    crop_log_probability: float
    token_count: int
    boundary_count: int
    source_event_count: int
    duration_event_count: int
    transition_event_count: int
    crop_event_count: int
    snapshot_sha256: str

    def __post_init__(self) -> None:
        if self.candidate_index < 0 or self.original_review_rank < 0:
            raise ValueError("candidate indices and ranks must be non-negative")
        if self.rerank_rank < 0 or self.token_count < 1 or self.boundary_count < 0:
            raise ValueError("StageEdit rerank score counts are invalid")
        if any(
            value < 0
            for value in (
                self.source_event_count,
                self.duration_event_count,
                self.transition_event_count,
                self.crop_event_count,
            )
        ):
            raise ValueError("StageEdit rerank event counts must be non-negative")
        for value in (
            self.total_score,
            self.cut_log_probability,
            self.source_log_probability,
            self.duration_log_probability,
            self.transition_log_probability,
            self.crop_log_probability,
        ):
            if not math.isfinite(value):
                raise ValueError("StageEdit rerank component scores must be finite")
        if len(self.snapshot_sha256) != 64:
            raise ValueError("StageEdit snapshot identity must be SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StageEditCandidateRerankResult:
    ranked_candidate_indices: tuple[int, ...]
    component_scores: tuple[StageEditCandidateComponentScore, ...]
    checkpoint_directory: str
    checkpoint_id: str
    training_steps: int
    checkpoint_bundle_sha256: str
    metadata_sha256: str
    production_training_report_path: str
    production_training_report_sha256: str
    weights_path: str
    weights_sha256: str
    token_schema_fingerprint: str
    feature_snapshot_set_sha256: str
    forward_microbatch_size: int
    readiness: StageEditReadinessReport
    weights: StageEditCandidateRerankWeights

    def __post_init__(self) -> None:
        indices = tuple(self.ranked_candidate_indices)
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("StageEdit rerank indices must be non-empty and unique")
        if len(indices) > 32:
            raise ValueError("StageEdit reranker accepts at most 32 candidates")
        by_index = {item.candidate_index: item for item in self.component_scores}
        if set(indices) != set(by_index):
            raise ValueError("StageEdit component scores do not cover reranked paths")
        if tuple(by_index[index].rerank_rank for index in indices) != tuple(
            range(len(indices))
        ):
            raise ValueError("StageEdit rerank ranks are inconsistent")
        for digest in (
            self.checkpoint_bundle_sha256,
            self.metadata_sha256,
            self.production_training_report_sha256,
            self.weights_sha256,
            self.token_schema_fingerprint,
            self.feature_snapshot_set_sha256,
        ):
            if len(digest) != 64:
                raise ValueError("StageEdit rerank identities must be SHA-256")
        if not (
            self.readiness.architecture_available
            and self.readiness.learned_inference_ready
            and self.readiness.status == "production_checkpoint_ready"
        ):
            raise ValueError("StageEdit rerank result requires production readiness")
        if not 1 <= self.forward_microbatch_size <= 32:
            raise ValueError("StageEdit forward microbatch size is invalid")

    @property
    def scores_by_candidate_index(
        self,
    ) -> Mapping[int, StageEditCandidateComponentScore]:
        return {item.candidate_index: item for item in self.component_scores}

    def ordered_records(
        self,
        candidate_set: VisualTimelineCandidateSet,
    ) -> tuple[VisualTimelineCandidateRecord, ...]:
        by_index = {item.candidate_index: item for item in candidate_set.candidates}
        if set(by_index) != set(self.ranked_candidate_indices):
            raise StageEditCandidateRerankInputError(
                "rerank result does not match the retained candidate set"
            )
        return tuple(by_index[index] for index in self.ranked_candidate_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE_EDIT_CANDIDATE_RERANK_SCHEMA_VERSION,
            "analysis_basis": STAGE_EDIT_CANDIDATE_RERANK_BASIS,
            "activation": "explicit_production_checkpoint_and_exact_report",
            "final_authority": (
                "existing_eight_decoded_proxy_review_and_professional_gate"
            ),
            "creates_new_timeline_paths": False,
            "ranked_candidate_indices": list(self.ranked_candidate_indices),
            "rank_change_semantics": "positive_is_promoted_from_review_rank",
            "checkpoint": {
                "directory": self.checkpoint_directory,
                "checkpoint_id": self.checkpoint_id,
                "training_steps": self.training_steps,
                "bundle_sha256": self.checkpoint_bundle_sha256,
                "metadata_sha256": self.metadata_sha256,
                "production_training_report_path": (
                    self.production_training_report_path
                ),
                "production_training_report_sha256": (
                    self.production_training_report_sha256
                ),
                "weights_path": self.weights_path,
                "weights_sha256": self.weights_sha256,
                "token_schema_fingerprint": self.token_schema_fingerprint,
            },
            "feature_snapshot_set_sha256": self.feature_snapshot_set_sha256,
            "forward_microbatch_size": self.forward_microbatch_size,
            "feature_contract": STAGE_EDIT_RERANK_FEATURE_CONTRACT,
            "weights": self.weights.to_dict(),
            "readiness": self.readiness.to_dict(),
            "component_scores": [item.to_dict() for item in self.component_scores],
        }


def _metadata_from_checkpoint(
    checkpoint_directory: Path,
) -> tuple[Any, Path, Path]:
    legacy_path = checkpoint_directory / "metadata.json"
    dynamic_path = checkpoint_directory / "dynamic-crop-v2-metadata.json"
    available = tuple(path for path in (legacy_path, dynamic_path) if path.is_file())
    if not available:
        raise StageEditCandidateRerankNotReadyError(
            "StageEdit checkpoint metadata does not exist"
        )
    if len(available) != 1:
        raise StageEditCandidateRerankNotReadyError(
            "StageEdit checkpoint contains ambiguous model formats"
        )
    metadata_path = available[0]
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("metadata root must be an object")
        if metadata_path == dynamic_path:
            from .stage_edit_dynamic_crop_head_v2 import (
                StageEditDynamicCropCheckpointMetadataV2,
            )

            metadata = StageEditDynamicCropCheckpointMetadataV2.from_dict(payload)
        else:
            metadata = StageEditCheckpointMetadata.from_dict(payload)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise StageEditCandidateRerankNotReadyError(
            f"StageEdit checkpoint metadata is invalid: {exc}"
        ) from exc
    weight_path = checkpoint_directory / metadata.weight_filename
    if not weight_path.is_file():
        raise StageEditCandidateRerankNotReadyError(
            "StageEdit checkpoint weight file does not exist"
        )
    return metadata, metadata_path, weight_path


def _strict_ready_checkpoint(
    checkpoint_directory: Path,
    production_training_report: Path,
) -> tuple[
    Any,
    Any,
    StageEditReadinessReport,
    dict[str, str],
]:
    checkpoint = Path(checkpoint_directory).expanduser().resolve()
    requested_report = Path(production_training_report).expanduser().resolve()
    expected_report = (
        checkpoint / PRODUCTION_TRAINING_REPORT_FILENAME
    ).resolve()
    if requested_report != expected_report:
        raise StageEditCandidateRerankNotReadyError(
            "requested StageEdit production training report is not the exact "
            "colocated checkpoint report"
        )
    if not requested_report.is_file():
        raise StageEditCandidateRerankNotReadyError(
            "requested StageEdit production training report does not exist"
        )
    metadata, metadata_path, weight_path = _metadata_from_checkpoint(checkpoint)
    before = {
        "metadata": _sha256_file(metadata_path),
        "report": _sha256_file(requested_report),
        "weights": _sha256_file(weight_path),
    }
    readiness = stage_edit_readiness_report(checkpoint)
    if not (
        readiness.architecture_available
        and readiness.learned_inference_ready
        and readiness.status == "production_checkpoint_ready"
        and readiness.checkpoint_path == str(checkpoint)
        and readiness.training_steps == metadata.training_steps
        and readiness.training_steps > 0
    ):
        raise StageEditCandidateRerankNotReadyError(
            "StageEdit production checkpoint rejected: "
            f"{readiness.status}: {readiness.reason}"
        )
    try:
        if hasattr(metadata, "dynamic_crop_config"):
            from .stage_edit_dynamic_crop_head_v2 import (
                load_stage_edit_dynamic_crop_checkpoint_v2,
            )

            model, loaded_metadata = load_stage_edit_dynamic_crop_checkpoint_v2(
                checkpoint,
                expected_schema=metadata.token_schema,
                require_trained=True,
            )
        else:
            model, loaded_metadata = load_stage_edit_checkpoint(
                checkpoint,
                expected_schema=metadata.token_schema,
                require_trained=True,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise StageEditCandidateRerankNotReadyError(
            f"StageEdit production checkpoint could not be loaded: {exc}"
        ) from exc
    after = {
        "metadata": _sha256_file(metadata_path),
        "report": _sha256_file(requested_report),
        "weights": _sha256_file(weight_path),
    }
    if before != after:
        raise StageEditCandidateRerankNotReadyError(
            "StageEdit checkpoint evidence changed while it was being loaded"
        )
    if loaded_metadata != metadata or metadata.weight_sha256 != after["weights"]:
        raise StageEditCandidateRerankNotReadyError(
            "StageEdit loaded checkpoint identity does not match verified evidence"
        )
    if hasattr(model, "eval"):
        model.eval()
    return model, metadata, readiness, before


def _observation_payload(observation: SourceFeatureObservation) -> dict[str, Any]:
    return {
        "source_id": observation.source_id,
        "features": observation.observed_features,
        "available": observation.available,
        "hard_gate_passed": observation.hard_gate_passed,
    }


def _candidate_feature_key(candidate: TimelineCandidate) -> str:
    observation = source_observation_from_timeline_candidate(candidate)
    return json.dumps(
        _observation_payload(observation),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _source_observations_by_segment(
    candidate_set: VisualTimelineCandidateSet,
) -> tuple[tuple[SourceFeatureObservation, ...], ...]:
    records = tuple(candidate_set.candidates)
    if not records or len(records) > 32:
        raise StageEditCandidateRerankInputError(
            "StageEdit reranker requires between one and 32 retained candidates"
        )
    segment_count = len(records[0].timeline_result.candidates)
    source_ids = tuple(str(index) for index in range(len(candidate_set.sources)))
    observations: list[tuple[SourceFeatureObservation, ...]] = []
    for segment_index in range(segment_count):
        by_source: dict[str, list[TimelineCandidate]] = {}
        for record in records:
            candidate = record.timeline_result.candidates[segment_index]
            by_source.setdefault(candidate.source_id, []).append(candidate)
        segment_observations: list[SourceFeatureObservation] = []
        for source_id in source_ids:
            candidates = by_source.get(source_id, ())
            if not candidates:
                segment_observations.append(
                    SourceFeatureObservation(
                        source_id=source_id,
                        available=False,
                        hard_gate_passed=False,
                    )
                )
                continue
            # A source/segment may be referenced by several retained paths.
            # Pick only by inference-safe automatic features, never by path
            # rank, optimiser score, reviewer result, or selected label.
            chosen = min(candidates, key=_candidate_feature_key)
            segment_observations.append(
                source_observation_from_timeline_candidate(chosen)
            )
        observations.append(tuple(segment_observations))
    return tuple(observations)


def _automatic_crop_target(
    candidate: TimelineCandidate,
) -> tuple[float, float, float, float]:
    bbox = (
        candidate.entry_focus_bbox_xywh
        or candidate.focus_bbox_xywh
        or candidate.entry_subject_bbox_xywh
        or candidate.subject_bbox_xywh
    )
    if bbox is None:
        return (1.0, 0.5, 0.5, 0.0)
    x, y, width, height = (float(value) for value in bbox)
    # Conservative virtual camera proposal from automatic geometry only.  It
    # is a scoring target, not an applied crop; downstream crop/affine gates
    # retain authority over the render plan.
    zoom = min(1.18, max(1.0, 0.35 / max(height, 1e-6)))
    return (
        zoom,
        min(1.0, max(0.0, x + width / 2.0)),
        min(1.0, max(0.0, y + height / 2.0)),
        0.0,
    )


def _causal_decision(
    candidate: TimelineCandidate,
) -> CausalEditDecision:
    zoom, center_x, center_y, roll = _automatic_crop_target(candidate)
    return CausalEditDecision(
        cut_time_seconds=candidate.start_seconds,
        source_id=candidate.source_id,
        face_scale=(
            candidate.entry_focus_bbox_xywh[3]
            if candidate.entry_focus_bbox_xywh is not None
            else candidate.focus_bbox_xywh[3]
            if candidate.focus_bbox_xywh is not None
            else None
        ),
        face_center_x=center_x,
        face_center_y=center_y,
        zoom=min(1.0, zoom / 1.18),
        roll_degrees_normalized=roll,
        transition_strength=0.0,
    )


def materialize_stage_edit_rerank_snapshots(
    candidate_set: VisualTimelineCandidateSet,
    *,
    schema: StageEditTokenSchema,
    rhythm: Any | None = None,
    section_plan: SectionPlan | None = None,
    style_features: Mapping[str, Any] | None = None,
) -> tuple[StageEditFeatureSnapshot, ...]:
    """Materialize label-free candidate sequences for train/serve parity.

    Each token sees the same automatic source-local observations.  A candidate
    path enters only through decisions strictly earlier than the current token;
    current/future source choices, review ranks, optimiser scores, professional
    gate labels, and reference attribution are not accepted by this API.
    """

    if not isinstance(candidate_set, VisualTimelineCandidateSet):
        raise TypeError("candidate_set must be a VisualTimelineCandidateSet")
    common_observations = _source_observations_by_segment(candidate_set)
    safe_feature_identity = _canonical_sha256(
        {
            "source_count": len(candidate_set.sources),
            "segments": [
                [_observation_payload(item) for item in segment]
                for segment in common_observations
            ],
        }
    )
    upstream_sha256: list[tuple[str, str]] = [
        ("timeline_candidate", safe_feature_identity)
    ]
    if rhythm is not None:
        rhythm_to_dict = getattr(rhythm, "to_dict", None)
        if not callable(rhythm_to_dict):
            raise StageEditCandidateRerankInputError(
                "StageEdit rhythm input needs deterministic to_dict provenance"
            )
        upstream_sha256.append(
            ("rhythm_analysis", _canonical_sha256(rhythm_to_dict()))
        )
    if section_plan is not None:
        upstream_sha256.append(
            ("section_plan", _canonical_sha256(section_plan.to_dict()))
        )
    if style_features is not None:
        upstream_sha256.append(
            ("style_profile", _canonical_sha256(dict(style_features)))
        )
    provenance = StageEditFeatureProvenance(
        upstream_sha256=tuple(upstream_sha256),
        contracts=(STAGE_EDIT_RERANK_FEATURE_CONTRACT,),
    )
    snapshots: list[StageEditFeatureSnapshot] = []
    for record in candidate_set.candidates:
        path = record.timeline_result.candidates
        history: list[CausalEditDecision] = []
        frames: list[StageEditFeatureFrame] = []
        previous_source: str | None = None
        for position, candidate in enumerate(path):
            frames.append(
                StageEditFeatureFrame(
                    master_time_seconds=candidate.start_seconds,
                    source_observations=common_observations[position],
                    causal_history=tuple(history),
                )
            )
            if previous_source is None or candidate.source_id != previous_source:
                # The decision is appended only after the frame at the same
                # timestamp was materialized, enforcing strict causality.
                history.append(_causal_decision(candidate))
            previous_source = candidate.source_id
        snapshots.append(
            materialize_stage_edit_feature_snapshot(
                f"visual-candidate-{record.candidate_index:02d}",
                tuple(frames),
                schema=schema,
                rhythm=rhythm,
                section_plan=section_plan,
                style_features=style_features,
                performance_start_seconds=path[0].start_seconds,
                performance_end_seconds=path[-1].end_seconds,
                provenance=provenance,
            )
        )
    return tuple(snapshots)


def _log_softmax(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or values.size == 0:
        raise StageEditCandidateRerankInputError(
            "StageEdit categorical logits must be a non-empty vector"
        )
    maximum = float(np.max(values))
    shifted = values - maximum
    normalizer = maximum + math.log(float(np.exp(shifted).sum()))
    return values - normalizer


def _categorical_log_probability(values: np.ndarray, target: int) -> float:
    if not 0 <= target < values.size:
        raise StageEditCandidateRerankInputError(
            "StageEdit categorical target is outside the model head"
        )
    return _finite(_log_softmax(values)[target], "categorical log probability")


def _run_duration(path: Sequence[TimelineCandidate], position: int) -> float:
    source_id = path[position].source_id
    end = path[position].end_seconds
    for later in path[position + 1 :]:
        if later.source_id != source_id:
            break
        end = later.end_seconds
    return end - path[position].start_seconds


def _candidate_component_values(
    *,
    record: VisualTimelineCandidateRecord,
    batch: Any,
    output: Any,
    batch_index: int,
    schema: StageEditTokenSchema,
    weights: StageEditCandidateRerankWeights,
) -> dict[str, Any]:
    path = record.timeline_result.candidates
    token_count = len(path)
    boundary_count = max(0, token_count - 1)
    cut_sum = 0.0
    source_sum = 0.0
    duration_sum = 0.0
    transition_sum = 0.0
    crop_sum = 0.0
    source_events = 0
    duration_events = 0
    transition_events = 0
    crop_events = 0
    hard_cut_index = (
        schema.transition_labels.index("hard_cut")
        if "hard_cut" in schema.transition_labels
        else None
    )
    if hard_cut_index is None:
        raise StageEditCandidateRerankInputError(
            "production StageEdit schema does not contain hard_cut"
        )
    source_logits = np.asarray(output.source_selection_logits)
    effective_mask = np.asarray(output.effective_source_mask, dtype=bool)
    cut_logits = np.asarray(output.cut_timing_logits)
    duration_logits = np.asarray(output.duration_bucket_logits)
    transition_logits = np.asarray(output.transition_logits)
    crops = np.asarray(output.crop)
    for position, candidate in enumerate(path):
        changed = position == 0 or candidate.source_id != path[position - 1].source_id
        if position > 0:
            logit = _finite(cut_logits[batch_index, position], "cut logit")
            cut_sum += (
                -float(np.logaddexp(0.0, -logit))
                if changed
                else -float(np.logaddexp(0.0, logit))
            )
        if not changed:
            continue
        slots = batch.source_ids[batch_index][position]
        try:
            source_slot = slots.index(candidate.source_id)
        except ValueError as exc:
            raise StageEditCandidateRerankInputError(
                "chosen source is absent from StageEdit candidate slots"
            ) from exc
        if not effective_mask[batch_index, position, source_slot]:
            raise StageEditCandidateRerankInputError(
                "chosen source is unavailable or hard-gated in StageEdit input"
            )
        valid_slots = effective_mask[batch_index, position]
        source_sum += _categorical_log_probability(
            source_logits[batch_index, position, valid_slots],
            int(np.flatnonzero(valid_slots).tolist().index(source_slot)),
        )
        source_events += 1
        duration_sum += _categorical_log_probability(
            duration_logits[batch_index, position],
            schema.duration_bucket(_run_duration(path, position)),
        )
        duration_events += 1
        if position > 0:
            transition_sum += _categorical_log_probability(
                transition_logits[batch_index, position],
                hard_cut_index,
            )
            transition_events += 1
        target_crop = np.asarray(_automatic_crop_target(candidate), dtype=np.float64)
        predicted_crop = np.asarray(crops[batch_index, position], dtype=np.float64)
        # The crop head is a regressor.  Treat it as the mean of a fixed-
        # variance diagonal Gaussian and record the log-kernel.  Fixed sigmas
        # make this deterministic and avoid claiming learned uncertainty.
        sigmas = np.asarray((0.12, 0.20, 0.20, 12.0), dtype=np.float64)
        normalized_residual = (target_crop - predicted_crop) / sigmas
        crop_sum += -0.5 * float(
            (
                np.square(normalized_residual)
                + np.log(2.0 * math.pi * np.square(sigmas))
            ).sum()
        )
        crop_events += 1
    denominator = float(token_count)
    components = {
        "cut_log_probability": cut_sum / denominator,
        "source_log_probability": source_sum / denominator,
        "duration_log_probability": duration_sum / denominator,
        "transition_log_probability": transition_sum / denominator,
        "crop_log_probability": crop_sum / denominator,
    }
    total = (
        weights.cut * components["cut_log_probability"]
        + weights.source * components["source_log_probability"]
        + weights.duration * components["duration_log_probability"]
        + weights.transition * components["transition_log_probability"]
        + weights.crop * components["crop_log_probability"]
    )
    return {
        **components,
        "total_score": _finite(total, "StageEdit candidate score"),
        "token_count": token_count,
        "boundary_count": boundary_count,
        "source_event_count": source_events,
        "duration_event_count": duration_events,
        "transition_event_count": transition_events,
        "crop_event_count": crop_events,
    }


def rerank_visual_timeline_candidates_with_stage_edit(
    candidate_set: VisualTimelineCandidateSet,
    *,
    checkpoint_directory: Path,
    production_training_report: Path,
    rhythm: Any | None = None,
    section_plan: SectionPlan | None = None,
    style_features: Mapping[str, Any] | None = None,
    weights: StageEditCandidateRerankWeights = StageEditCandidateRerankWeights(),
) -> StageEditCandidateRerankResult:
    """Rerank retained top-32 candidates with an actual production MLX forward."""

    if not isinstance(candidate_set, VisualTimelineCandidateSet):
        raise TypeError("candidate_set must be a VisualTimelineCandidateSet")
    model, metadata, readiness, evidence_sha = _strict_ready_checkpoint(
        checkpoint_directory,
        production_training_report,
    )
    if len(candidate_set.sources) > metadata.model_config.max_sources:
        raise StageEditCandidateRerankInputError(
            "retained candidate source count exceeds the production checkpoint"
        )
    if metadata.token_schema.crop_feature_names != (
        "zoom",
        "center_x",
        "center_y",
        "roll_degrees",
    ):
        raise StageEditCandidateRerankInputError(
            "production StageEdit crop head semantics are unsupported"
        )
    snapshots = materialize_stage_edit_rerank_snapshots(
        candidate_set,
        schema=metadata.token_schema,
        rhythm=rhythm,
        section_plan=section_plan,
        style_features=style_features,
    )
    snapshot_set_sha256 = _canonical_sha256(
        [
            {
                "candidate_index": record.candidate_index,
                "snapshot_sha256": snapshot.snapshot_sha256,
            }
            for record, snapshot in zip(
                candidate_set.candidates, snapshots, strict=True
            )
        ]
    )
    raw_scores: list[tuple[VisualTimelineCandidateRecord, dict[str, Any], str]] = []
    try:
        import mlx.core as mx

        paired = tuple(zip(candidate_set.candidates, snapshots, strict=True))
        for start in range(0, len(paired), STAGE_EDIT_RERANK_MICROBATCH_SIZE):
            microbatch = paired[start : start + STAGE_EDIT_RERANK_MICROBATCH_SIZE]
            batch = tensorize_stage_edit_batch(
                tuple(
                    mask_stage_edit_recent_history_for_semimarkov(
                        snapshot.sequence, metadata.token_schema
                    )
                    for _, snapshot in microbatch
                ),
                metadata.token_schema,
                metadata.model_config,
            )
            output = model(batch)
            mx.eval(
                output.cut_timing_logits,
                output.source_selection_logits,
                output.duration_bucket_logits,
                output.transition_logits,
                output.crop,
                output.effective_source_mask,
            )
            for batch_index, (record, snapshot) in enumerate(microbatch):
                raw_scores.append(
                    (
                        record,
                        _candidate_component_values(
                            record=record,
                            batch=batch,
                            output=output,
                            batch_index=batch_index,
                            schema=metadata.token_schema,
                            weights=weights,
                        ),
                        snapshot.snapshot_sha256,
                    )
                )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise StageEditCandidateRerankError(
            f"StageEdit production forward failed: {exc}"
        ) from exc
    ranked = tuple(
        sorted(
            raw_scores,
            key=lambda item: (
                -float(item[1]["total_score"]),
                item[0].review_rank,
                item[0].candidate_index,
            ),
        )
    )
    rank_by_index = {
        record.candidate_index: rank
        for rank, (record, _, _) in enumerate(ranked)
    }
    component_scores = tuple(
        StageEditCandidateComponentScore(
            candidate_index=record.candidate_index,
            original_review_rank=record.review_rank,
            rerank_rank=rank_by_index[record.candidate_index],
            rank_change=record.review_rank - rank_by_index[record.candidate_index],
            snapshot_sha256=snapshot_sha256,
            **values,
        )
        for record, values, snapshot_sha256 in raw_scores
    )
    checkpoint = Path(checkpoint_directory).expanduser().resolve()
    report = Path(production_training_report).expanduser().resolve()
    weight_path = (checkpoint / metadata.weight_filename).resolve()
    bundle_sha256 = _canonical_sha256(
        {
            "metadata_sha256": evidence_sha["metadata"],
            "production_training_report_sha256": evidence_sha["report"],
            "weights_sha256": evidence_sha["weights"],
        }
    )
    return StageEditCandidateRerankResult(
        ranked_candidate_indices=tuple(
            record.candidate_index for record, _, _ in ranked
        ),
        component_scores=component_scores,
        checkpoint_directory=str(checkpoint),
        checkpoint_id=metadata.checkpoint_id,
        training_steps=metadata.training_steps,
        checkpoint_bundle_sha256=bundle_sha256,
        metadata_sha256=evidence_sha["metadata"],
        production_training_report_path=str(report),
        production_training_report_sha256=evidence_sha["report"],
        weights_path=str(weight_path),
        weights_sha256=evidence_sha["weights"],
        token_schema_fingerprint=metadata.token_schema.fingerprint,
        feature_snapshot_set_sha256=snapshot_set_sha256,
        forward_microbatch_size=STAGE_EDIT_RERANK_MICROBATCH_SIZE,
        readiness=readiness,
        weights=weights,
    )


__all__ = [
    "STAGE_EDIT_CANDIDATE_RERANK_BASIS",
    "StageEditCandidateComponentScore",
    "StageEditCandidateRerankError",
    "StageEditCandidateRerankInputError",
    "StageEditCandidateRerankNotReadyError",
    "StageEditCandidateRerankResult",
    "StageEditCandidateRerankWeights",
    "materialize_stage_edit_rerank_snapshots",
    "rerank_visual_timeline_candidates_with_stage_edit",
]
