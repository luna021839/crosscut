"""Training, song-disjoint validation, and sealed artifacts for contextual v2.

The decoder, registry, and promotion bridge intentionally do not import this
module yet.  It is the standalone production core: exact frozen A/id3 fitting,
leave-one-song-out authorization, immutable checkpoint/report schemas, and the
mandatory post-affine beam-level QA gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from .stage_edit_continuity_authority import (
    CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2,
)
from .stage_edit_source_pair_contextual_v2 import (
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_AUTHORITY,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BACKTRACK_ATTEMPTS,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BOUNDED_SCORE_CONTRACT,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FILENAME,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FORMAT_VERSION,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CORPUS_FILENAME,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CROSS_VALIDATION_CONTRACT,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_CONTRACT,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_SCHEMA_SHA256,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_HANDCRAFTED_BASELINE_FRACTION,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_L2_PENALTY,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_LEARNING_RATE,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_MARGIN,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_OBJECTIVE_CONTRACT,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_CONTRACT,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_FILENAME,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_SCHEMA_VERSION,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RUNTIME_EVIDENCE_CONTRACT,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SIGNED_L1_RADIUS,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_TEMPERATURE,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_TOLERANCE,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_TRAINING_STEPS,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_UNIFORM_BASELINE_FRACTION,
    StageEditContextualSourcePairScorerV2,
    StageEditContextualSourcePairV2ArtifactError,
    StageEditContextualSourcePairV2Error,
    StageEditContextualTrainingCandidateV2,
    StageEditContextualTrainingConfigV2,
    StageEditContextualTrainingCorpusV2,
    StageEditContextualTrainingGroupV2,
    _canonical_json_bytes,
    _canonical_sha256,
    _finite,
    _integer,
    _name,
    _sha256,
    _strict_keys,
    materialize_stage_edit_contextual_pair_features_v2,
    new_stage_edit_contextual_training_corpus_v2,
    stage_edit_contextual_pair_baselines_v2,
    stage_edit_contextual_raw_to_decoder_score_v2,
)
from .stage_edit_source_pair_training import (
    STAGE_EDIT_SOURCE_PAIR_ACTIVATION_GATE_CONTRACT,
    STAGE_EDIT_SOURCE_PAIR_HANDCRAFTED_BASELINE_CONTRACT,
    StageEditSourcePairActivationGate,
    StageEditSourcePairRankingMetrics,
    stage_edit_source_pair_no_worse_activation_gate,
)


STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BEAM_QA_CONTRACT = (
    "affine_absolute_switch_cost_cadence_occupancy_highlight_runtime_qa_v1"
)
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RUNTIME_EQUIVALENCE_CONTRACT = (
    "shared_feature_raw_logit_and_affine_decoder_score_exact_table_v1"
)
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BEAM_QA_CASE_CONTRACT = (
    "full_decode_song_disjoint_measured_case_cadence_occupancy_highlight_v1"
)
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BEAM_QA_THRESHOLDS: Mapping[str, float | int] = {
    "minimum_song_disjoint_case_count": 2,
    "switches_per_minute_hard_minimum": 12.5,
    "switches_per_minute_target_minimum": 13.5,
    "switches_per_minute_target_maximum": 30.0,
    "switch_ratio_minimum": 0.34,
    "switch_ratio_maximum": 0.64,
    "visual_segment_median_seconds_maximum": 2.2,
    "source_run_median_seconds_minimum": 2.5,
    "source_run_median_seconds_maximum": 5.5,
    "longest_source_run_seconds_maximum": 7.2,
    "distinct_source_count_minimum": 8,
    "effective_source_count_minimum": 8.0,
    "dominant_source_ratio_target_maximum": 0.28,
    "dominant_source_ratio_hard_maximum": 0.45,
    "quality_score_p10_minimum": 0.36,
    "quality_score_median_span_maximum": 0.22,
    "boundary_match_score_p10_minimum": 58.0,
    "weighted_proxy_score_minimum": 60.0,
    "highlight_window_count_minimum": 1,
    "highlight_switch_count_minimum": 1,
    "closeup_motion_opportunity_count_minimum": 1,
    "closeup_motion_switch_count_minimum": 1,
    "geometry_issue_count_maximum": 0,
    "sync_issue_count_maximum": 0,
}
_PRODUCTION_BUNDLE_ACTIVATION_SEAL = object()
_EVALUATION_BUNDLE_ACTIVATION_SEAL = object()


def _project_signed_l1(values: np.ndarray, radius: float) -> np.ndarray:
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise StageEditContextualSourcePairV2Error(
            "signed L1 projection input is invalid"
        )
    if float(np.sum(np.abs(values))) <= radius:
        return values.copy()
    ordered = np.sort(np.abs(values))[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, len(values) + 1, dtype=np.float64)
    support = ordered * indices > cumulative - radius
    if not np.any(support):
        raise StageEditContextualSourcePairV2Error(
            "signed L1 projection found no support"
        )
    rho = int(np.flatnonzero(support)[-1])
    threshold = (cumulative[rho] - radius) / float(rho + 1)
    projected = np.sign(values) * np.maximum(
        np.abs(values) - threshold,
        0.0,
    )
    # The analytical projection is on the closed L1 ball, but summing hundreds
    # of float64 coordinates can land a few ulps above ``radius``.  The scorer
    # deliberately validates the hard radius without a permissive epsilon, so
    # make the projection itself satisfy that persisted contract.  Correct the
    # largest coordinate deterministically, then step it once toward zero if
    # the rounded aggregate still exceeds the boundary.
    projected_l1 = math.fsum(abs(float(value)) for value in projected)
    if projected_l1 > radius:
        largest_index = int(np.argmax(np.abs(projected)))
        largest = float(projected[largest_index])
        corrected_magnitude = max(
            0.0,
            abs(largest) - (projected_l1 - radius),
        )
        projected[largest_index] = math.copysign(
            corrected_magnitude,
            largest,
        )
        while (
            projected[largest_index] != 0.0
            and math.fsum(abs(float(value)) for value in projected) > radius
        ):
            projected[largest_index] = np.nextafter(
                projected[largest_index],
                0.0,
            )
    if math.fsum(abs(float(value)) for value in projected) > radius:
        raise StageEditContextualSourcePairV2Error(
            "signed L1 projection could not satisfy its hard radius"
        )
    return projected


def stage_edit_contextual_group_balance_weights_v2(
    groups: Sequence[StageEditContextualTrainingGroupV2],
) -> tuple[tuple[str, float], ...]:
    """Return exact equal song, then reference, then group weights."""

    values = tuple(
        sorted(groups, key=lambda value: (value.song_id, value.reference_id, value.group_id))
    )
    if not values or any(
        not isinstance(value, StageEditContextualTrainingGroupV2)
        for value in values
    ):
        raise StageEditContextualSourcePairV2Error(
            "listwise objective needs typed contextual groups"
        )
    group_ids = tuple(value.group_id for value in values)
    if len(group_ids) != len(set(group_ids)):
        raise StageEditContextualSourcePairV2Error(
            "listwise objective group ids must be unique"
        )
    song_ids = tuple(sorted({value.song_id for value in values}))
    weights: list[tuple[str, float]] = []
    for song_id in song_ids:
        song_groups = tuple(value for value in values if value.song_id == song_id)
        reference_ids = tuple(sorted({value.reference_id for value in song_groups}))
        for reference_id in reference_ids:
            reference_groups = tuple(
                value
                for value in song_groups
                if value.reference_id == reference_id
            )
            weight = 1.0 / len(song_ids) / len(reference_ids) / len(reference_groups)
            weights.extend((value.group_id, weight) for value in reference_groups)
    result = tuple(weights)
    if tuple(value for value, _weight in result) != group_ids:
        raise StageEditContextualSourcePairV2Error(
            "listwise hierarchy changed canonical group order"
        )
    if not math.isclose(
        math.fsum(weight for _group_id, weight in result),
        1.0,
        abs_tol=1.0e-12,
    ):
        raise StageEditContextualSourcePairV2Error(
            "listwise hierarchy weights do not sum to one"
        )
    return result


def _group_arrays(
    group: StageEditContextualTrainingGroupV2,
) -> tuple[np.ndarray, np.ndarray, int]:
    matrix = np.asarray(
        [candidate.features.values for candidate in group.candidates],
        dtype=np.float64,
    )
    baselines = np.asarray(
        [candidate.blended_baseline for candidate in group.candidates],
        dtype=np.float64,
    )
    chosen_index = tuple(
        candidate.source_id for candidate in group.candidates
    ).index(group.chosen_incoming_source_id)
    return matrix, baselines, chosen_index


def _loss_and_gradient(
    weights: np.ndarray,
    groups: Sequence[StageEditContextualTrainingGroupV2],
) -> tuple[float, np.ndarray]:
    group_values = tuple(
        sorted(groups, key=lambda value: (value.song_id, value.reference_id, value.group_id))
    )
    balances = dict(stage_edit_contextual_group_balance_weights_v2(group_values))
    total_loss = 0.0
    gradient = np.zeros_like(weights)
    for group in group_values:
        matrix, baselines, chosen_index = _group_arrays(group)
        logits = baselines + np.einsum(
            "cf,f->c", matrix, weights, optimize=False
        )
        # Temperature is exactly 1.0 and margin is exactly zero.  Keep those
        # values out of the arithmetic rather than introducing a second route.
        shifted = logits - float(np.max(logits))
        exponential = np.exp(shifted)
        probabilities = exponential / float(np.sum(exponential))
        group_loss = -float(shifted[chosen_index]) + math.log(
            float(np.sum(exponential))
        )
        direction = probabilities.copy()
        direction[chosen_index] -= 1.0
        group_gradient = np.einsum(
            "c,cf->f", direction, matrix, optimize=False
        )
        group_weight = balances[group.group_id]
        total_loss += group_weight * group_loss
        gradient += group_weight * group_gradient
    squared = float(np.einsum("f,f->", weights, weights, optimize=False))
    total_loss += 0.5 * STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_L2_PENALTY * squared
    gradient += STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_L2_PENALTY * weights
    if not math.isfinite(total_loss) or not np.all(np.isfinite(gradient)):
        raise StageEditContextualSourcePairV2Error(
            "contextual listwise objective became non-finite"
        )
    return total_loss, gradient


def stage_edit_contextual_listwise_loss_v2(
    scorer: StageEditContextualSourcePairScorerV2,
    groups: Sequence[StageEditContextualTrainingGroupV2],
) -> float:
    if not isinstance(scorer, StageEditContextualSourcePairScorerV2):
        raise StageEditContextualSourcePairV2Error(
            "listwise loss needs contextual v2 scorer"
        )
    loss, _gradient = _loss_and_gradient(
        np.asarray(scorer.weights, dtype=np.float64), groups
    )
    return loss


def _runtime_equivalence_payload(
    scorer: StageEditContextualSourcePairScorerV2,
    groups: Sequence[StageEditContextualTrainingGroupV2],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    weights = np.asarray(scorer.weights, dtype=np.float64)
    for group in sorted(
        groups, key=lambda value: (value.song_id, value.reference_id, value.group_id)
    ):
        for candidate in group.candidates:
            live_features = materialize_stage_edit_contextual_pair_features_v2(
                group.runtime_frame,
                group.outgoing_source_id,
                candidate.source_id,
            )
            live_uniform, live_handcrafted, live_blended = (
                stage_edit_contextual_pair_baselines_v2(
                    group.runtime_frame,
                    group.outgoing_source_id,
                    candidate.source_id,
                )
            )
            if (
                live_features != candidate.features
                or live_uniform != candidate.uniform_baseline
                or live_handcrafted != candidate.handcrafted_baseline
                or live_blended != candidate.blended_baseline
            ):
                raise StageEditContextualSourcePairV2Error(
                    "live runtime rematerialization differs from training corpus"
                )
            raw_runtime, blended, residual_runtime = scorer.raw_logit_from_materialized(
                live_features,
                uniform_baseline=live_uniform,
                handcrafted_baseline=live_handcrafted,
            )
            residual_training = float(
                np.einsum(
                    "f,f->",
                    np.asarray(candidate.features.values, dtype=np.float64),
                    weights,
                    optimize=False,
                )
            )
            raw_training = candidate.blended_baseline + residual_training
            if not math.isclose(raw_runtime, raw_training, abs_tol=1.0e-12):
                raise StageEditContextualSourcePairV2Error(
                    "train/runtime raw logit paths are not equivalent"
                )
            rows.append(
                {
                    "group_id": group.group_id,
                    "source_id": candidate.source_id,
                    "pair_feature_sha256": candidate.features.content_sha256,
                    "live_pair_feature_sha256": live_features.content_sha256,
                    "runtime_evidence_sha256": group.runtime_frame.runtime_evidence_sha256,
                    "uniform_baseline": candidate.uniform_baseline,
                    "handcrafted_baseline": candidate.handcrafted_baseline,
                    "blended_baseline": blended,
                    "signed_residual": residual_runtime,
                    "training_raw_logit": raw_training,
                    "runtime_raw_logit": raw_runtime,
                    "runtime_decoder_score": stage_edit_contextual_raw_to_decoder_score_v2(
                        raw_runtime
                    ),
                }
            )
    return {
        "contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RUNTIME_EQUIVALENCE_CONTRACT,
        "scorer_fingerprint": scorer.fingerprint,
        "feature_schema_sha256": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_SCHEMA_SHA256,
        "einsum_optimize": False,
        "exact_row_count": len(rows),
        "rows": rows,
    }


def _contextual_scorer_from_dict(
    payload: Mapping[str, Any],
) -> StageEditContextualSourcePairScorerV2:
    expected = {
        "authority",
        "contract",
        "token_schema_fingerprint",
        "feature_contract",
        "feature_schema_sha256",
        "feature_names",
        "weights",
        "signed_l1_radius",
        "handcrafted_baseline_fraction",
        "uniform_baseline_fraction",
        "bounded_score_contract",
    }
    _strict_keys(payload, expected, "contextual scorer")
    constants = {
        "feature_contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_CONTRACT,
        "feature_schema_sha256": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_SCHEMA_SHA256,
        "signed_l1_radius": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SIGNED_L1_RADIUS,
        "handcrafted_baseline_fraction": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_HANDCRAFTED_BASELINE_FRACTION,
        "uniform_baseline_fraction": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_UNIFORM_BASELINE_FRACTION,
        "bounded_score_contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BOUNDED_SCORE_CONTRACT,
    }
    if any(payload.get(field) != value for field, value in constants.items()):
        raise StageEditContextualSourcePairV2Error(
            "contextual scorer frozen constants differ"
        )
    if not isinstance(payload.get("feature_names"), list) or not isinstance(
        payload.get("weights"), list
    ):
        raise StageEditContextualSourcePairV2Error(
            "contextual scorer names and weights must be arrays"
        )
    return StageEditContextualSourcePairScorerV2(
        authority=payload["authority"],
        contract=payload["contract"],
        token_schema_fingerprint=payload["token_schema_fingerprint"],
        feature_names=tuple(payload["feature_names"]),
        weights=tuple(payload["weights"]),
    )


def _ranking_metrics_from_dict(
    payload: Mapping[str, Any],
) -> StageEditSourcePairRankingMetrics:
    expected = {
        "scorer_name",
        "reference_count",
        "group_count",
        "reference_balanced_top1_accuracy",
        "reference_balanced_pairwise_accuracy",
        "reference_balanced_mean_reciprocal_rank",
        "reference_balanced_mean_chosen_margin",
    }
    _strict_keys(payload, expected, "ranking metrics")
    return StageEditSourcePairRankingMetrics(
        scorer_name=payload["scorer_name"],
        reference_count=_integer(
            payload["reference_count"], "reference_count", minimum=1
        ),
        group_count=_integer(payload["group_count"], "group_count", minimum=1),
        reference_balanced_top1_accuracy=payload[
            "reference_balanced_top1_accuracy"
        ],
        reference_balanced_pairwise_accuracy=payload[
            "reference_balanced_pairwise_accuracy"
        ],
        reference_balanced_mean_reciprocal_rank=payload[
            "reference_balanced_mean_reciprocal_rank"
        ],
        reference_balanced_mean_chosen_margin=payload[
            "reference_balanced_mean_chosen_margin"
        ],
    )


def _activation_gate_from_dict(
    payload: Mapping[str, Any],
) -> StageEditSourcePairActivationGate:
    _strict_keys(payload, {"contract", "authorized", "reasons"}, "activation gate")
    if not isinstance(payload.get("reasons"), list):
        raise StageEditContextualSourcePairV2Error(
            "activation gate reasons must be an array"
        )
    return StageEditSourcePairActivationGate(
        contract=payload["contract"],
        authorized=payload["authorized"],
        reasons=tuple(payload["reasons"]),
    )


@dataclass(frozen=True, slots=True)
class StageEditContextualTrainingResultV2:
    scorer: StageEditContextualSourcePairScorerV2
    config: StageEditContextualTrainingConfigV2
    corpus_sha256: str
    source_data_id: str
    feature_snapshot_bindings_sha256: str
    clipped_observed_value_count: int
    training_song_ids: tuple[str, ...]
    training_reference_ids: tuple[str, ...]
    training_group_ids: tuple[str, ...]
    initial_loss: float
    final_loss: float
    completed_steps: int
    final_learning_rate: float
    runtime_equivalence_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.scorer, StageEditContextualSourcePairScorerV2):
            raise StageEditContextualSourcePairV2Error("training result scorer is invalid")
        if not isinstance(self.config, StageEditContextualTrainingConfigV2):
            raise StageEditContextualSourcePairV2Error("training result config is invalid")
        object.__setattr__(self, "corpus_sha256", _sha256(self.corpus_sha256, "corpus_sha256"))
        object.__setattr__(
            self, "source_data_id", _name(self.source_data_id, "source_data_id")
        )
        object.__setattr__(
            self,
            "feature_snapshot_bindings_sha256",
            _sha256(
                self.feature_snapshot_bindings_sha256,
                "feature_snapshot_bindings_sha256",
            ),
        )
        _integer(
            self.clipped_observed_value_count,
            "clipped_observed_value_count",
        )
        for field in (
            "training_song_ids",
            "training_reference_ids",
            "training_group_ids",
        ):
            values = tuple(_name(value, field) for value in getattr(self, field))
            if not values or values != tuple(sorted(set(values))):
                raise StageEditContextualSourcePairV2Error(
                    f"{field} must be non-empty, unique, and sorted"
                )
            object.__setattr__(self, field, values)
        initial = _finite(self.initial_loss, "initial_loss")
        final = _finite(self.final_loss, "final_loss")
        if final > initial + 1.0e-10:
            raise StageEditContextualSourcePairV2Error(
                "contextual training objective regressed"
            )
        completed = _integer(self.completed_steps, "completed_steps")
        if completed > self.config.steps:
            raise StageEditContextualSourcePairV2Error(
                "completed steps exceed frozen step budget"
            )
        rate = _finite(self.final_learning_rate, "final_learning_rate")
        if not 0.0 < rate <= self.config.learning_rate:
            raise StageEditContextualSourcePairV2Error(
                "final learning rate is outside the frozen backtracking bounds"
            )
        object.__setattr__(self, "initial_loss", initial)
        object.__setattr__(self, "final_loss", final)
        object.__setattr__(self, "final_learning_rate", rate)
        object.__setattr__(
            self,
            "runtime_equivalence_sha256",
            _sha256(
                self.runtime_equivalence_sha256,
                "runtime_equivalence_sha256",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "corpus_sha256": self.corpus_sha256,
            "source_data_id": self.source_data_id,
            "feature_snapshot_bindings_sha256": self.feature_snapshot_bindings_sha256,
            "clipped_observed_value_count": self.clipped_observed_value_count,
            "training_song_ids": list(self.training_song_ids),
            "training_reference_ids": list(self.training_reference_ids),
            "training_group_ids": list(self.training_group_ids),
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "completed_steps": self.completed_steps,
            "final_learning_rate": self.final_learning_rate,
            "scorer": self.scorer.to_dict(),
            "scorer_fingerprint": self.scorer.fingerprint,
            "runtime_equivalence_sha256": self.runtime_equivalence_sha256,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "StageEditContextualTrainingResultV2":
        expected = {
            "config",
            "corpus_sha256",
            "source_data_id",
            "feature_snapshot_bindings_sha256",
            "clipped_observed_value_count",
            "training_song_ids",
            "training_reference_ids",
            "training_group_ids",
            "initial_loss",
            "final_loss",
            "completed_steps",
            "final_learning_rate",
            "scorer",
            "scorer_fingerprint",
            "runtime_equivalence_sha256",
        }
        _strict_keys(payload, expected, "contextual training result")
        if not isinstance(payload.get("config"), Mapping) or dict(
            payload["config"]
        ) != StageEditContextualTrainingConfigV2().to_dict():
            raise StageEditContextualSourcePairV2Error(
                "contextual training result config differs from frozen config"
            )
        if not isinstance(payload.get("scorer"), Mapping):
            raise StageEditContextualSourcePairV2Error(
                "contextual training result scorer must be an object"
            )
        for field in (
            "training_song_ids",
            "training_reference_ids",
            "training_group_ids",
        ):
            if not isinstance(payload.get(field), list):
                raise StageEditContextualSourcePairV2Error(
                    f"contextual training result {field} must be an array"
                )
        scorer = _contextual_scorer_from_dict(payload["scorer"])
        if payload["scorer_fingerprint"] != scorer.fingerprint:
            raise StageEditContextualSourcePairV2Error(
                "contextual training result scorer fingerprint mismatch"
            )
        return cls(
            scorer=scorer,
            config=StageEditContextualTrainingConfigV2(),
            corpus_sha256=payload["corpus_sha256"],
            source_data_id=payload["source_data_id"],
            feature_snapshot_bindings_sha256=payload[
                "feature_snapshot_bindings_sha256"
            ],
            clipped_observed_value_count=_integer(
                payload["clipped_observed_value_count"],
                "clipped_observed_value_count",
            ),
            training_song_ids=tuple(payload["training_song_ids"]),
            training_reference_ids=tuple(payload["training_reference_ids"]),
            training_group_ids=tuple(payload["training_group_ids"]),
            initial_loss=payload["initial_loss"],
            final_loss=payload["final_loss"],
            completed_steps=_integer(
                payload["completed_steps"], "completed_steps"
            ),
            final_learning_rate=payload["final_learning_rate"],
            runtime_equivalence_sha256=payload["runtime_equivalence_sha256"],
        )


def train_stage_edit_contextual_source_pair_v2(
    corpus: StageEditContextualTrainingCorpusV2,
    *,
    config: StageEditContextualTrainingConfigV2 = StageEditContextualTrainingConfigV2(),
) -> StageEditContextualTrainingResultV2:
    """Fit exact frozen A/id3 projected-gradient listwise residual weights."""

    if not isinstance(corpus, StageEditContextualTrainingCorpusV2):
        raise StageEditContextualSourcePairV2Error(
            "contextual training needs typed materialized corpus"
        )
    if not isinstance(config, StageEditContextualTrainingConfigV2):
        raise StageEditContextualSourcePairV2Error("contextual training config is invalid")
    weights = np.zeros(
        len(STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES),
        dtype=np.float64,
    )
    initial_loss, _gradient = _loss_and_gradient(weights, corpus.groups)
    current_loss = initial_loss
    current_rate = config.learning_rate
    completed_steps = 0
    for _step in range(config.steps):
        current_loss, gradient = _loss_and_gradient(weights, corpus.groups)
        accepted = False
        trial_rate = current_rate
        for _attempt in range(config.backtrack_attempts):
            proposed = _project_signed_l1(
                weights - trial_rate * gradient,
                config.signed_l1_radius,
            )
            proposed_loss, _proposed_gradient = _loss_and_gradient(
                proposed, corpus.groups
            )
            if proposed_loss <= current_loss + config.tolerance:
                weights = proposed
                current_loss = proposed_loss
                current_rate = min(
                    config.learning_rate,
                    trial_rate * 1.02,
                )
                completed_steps += 1
                accepted = True
                break
            trial_rate *= 0.5
        if not accepted:
            break
    scorer = StageEditContextualSourcePairScorerV2(
        token_schema_fingerprint=corpus.token_schema_fingerprint,
        weights=tuple(float(value) for value in weights),
    )
    final_loss = stage_edit_contextual_listwise_loss_v2(scorer, corpus.groups)
    equivalence = _runtime_equivalence_payload(scorer, corpus.groups)
    return StageEditContextualTrainingResultV2(
        scorer=scorer,
        config=config,
        corpus_sha256=corpus.content_sha256,
        source_data_id=corpus.source_data_id,
        feature_snapshot_bindings_sha256=corpus.feature_snapshot_bindings_sha256,
        clipped_observed_value_count=corpus.clipped_observed_value_count,
        training_song_ids=corpus.song_ids,
        training_reference_ids=corpus.reference_ids,
        training_group_ids=tuple(sorted(corpus.group_ids)),
        initial_loss=initial_loss,
        final_loss=final_loss,
        completed_steps=completed_steps,
        final_learning_rate=current_rate,
        runtime_equivalence_sha256=_canonical_sha256(equivalence),
    )


def evaluate_stage_edit_contextual_source_pair_ranking_v2(
    groups: Sequence[StageEditContextualTrainingGroupV2],
    *,
    scorer: StageEditContextualSourcePairScorerV2 | None,
    scorer_name: str,
    baseline: str | None = None,
) -> StageEditSourcePairRankingMetrics:
    values = tuple(groups)
    if not values:
        raise StageEditContextualSourcePairV2Error("ranking evaluation needs groups")
    if (scorer is None) == (baseline is None):
        raise StageEditContextualSourcePairV2Error(
            "ranking evaluation needs exactly one scorer mode"
        )
    if baseline not in (None, "uniform", "handcrafted"):
        raise StageEditContextualSourcePairV2Error("unsupported ranking baseline")
    top1: dict[str, list[float]] = {}
    pairwise: dict[str, list[float]] = {}
    reciprocal: dict[str, list[float]] = {}
    margins: dict[str, list[float]] = {}
    tolerance = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_TOLERANCE
    for group in values:
        score_by_source: dict[str, float] = {}
        for candidate in group.candidates:
            if scorer is not None:
                score, _blended, _residual = scorer.raw_logit_from_materialized(
                    candidate.features,
                    uniform_baseline=candidate.uniform_baseline,
                    handcrafted_baseline=candidate.handcrafted_baseline,
                )
            elif baseline == "uniform":
                score = candidate.uniform_baseline
            else:
                score = candidate.handcrafted_baseline
            score_by_source[candidate.source_id] = _finite(
                score, f"{scorer_name} candidate score"
            )
        chosen = score_by_source[group.chosen_incoming_source_id]
        alternatives = tuple(
            value
            for source_id, value in score_by_source.items()
            if source_id != group.chosen_incoming_source_id
        )
        best = max(score_by_source.values())
        tied_best = sum(
            abs(value - best) <= tolerance for value in score_by_source.values()
        )
        top_credit = 1.0 / tied_best if abs(chosen - best) <= tolerance else 0.0
        better = sum(value > chosen + tolerance for value in alternatives)
        tied = sum(abs(value - chosen) <= tolerance for value in alternatives)
        pair_credit = float(
            np.mean(
                [
                    1.0
                    if chosen > value + tolerance
                    else 0.5
                    if abs(chosen - value) <= tolerance
                    else 0.0
                    for value in alternatives
                ]
            )
        )
        reference_id = group.reference_id
        top1.setdefault(reference_id, []).append(top_credit)
        pairwise.setdefault(reference_id, []).append(pair_credit)
        reciprocal.setdefault(reference_id, []).append(
            1.0 / (1.0 + better + 0.5 * tied)
        )
        margins.setdefault(reference_id, []).append(chosen - max(alternatives))

    def balanced(by_reference: Mapping[str, Sequence[float]]) -> float:
        return float(
            np.mean(
                [
                    float(np.mean(tuple(by_reference[reference_id])))
                    for reference_id in sorted(by_reference)
                ]
            )
        )

    return StageEditSourcePairRankingMetrics(
        scorer_name=scorer_name,
        reference_count=len(top1),
        group_count=len(values),
        reference_balanced_top1_accuracy=balanced(top1),
        reference_balanced_pairwise_accuracy=balanced(pairwise),
        reference_balanced_mean_reciprocal_rank=balanced(reciprocal),
        reference_balanced_mean_chosen_margin=balanced(margins),
    )


@dataclass(frozen=True, slots=True)
class StageEditContextualCrossValidationFoldV2:
    fold_index: int
    heldout_song_id: str
    training_song_ids: tuple[str, ...]
    training_reference_ids: tuple[str, ...]
    heldout_reference_ids: tuple[str, ...]
    training_group_ids: tuple[str, ...]
    heldout_group_ids: tuple[str, ...]
    training_corpus_sha256: str
    training_feature_snapshot_bindings_sha256: str
    training: StageEditContextualTrainingResultV2
    learned_metrics: StageEditSourcePairRankingMetrics
    uniform_metrics: StageEditSourcePairRankingMetrics
    handcrafted_metrics: StageEditSourcePairRankingMetrics
    activation_gate: StageEditSourcePairActivationGate

    def __post_init__(self) -> None:
        _integer(self.fold_index, "fold_index")
        object.__setattr__(self, "heldout_song_id", _name(self.heldout_song_id, "heldout_song_id"))
        for field in (
            "training_song_ids",
            "training_reference_ids",
            "heldout_reference_ids",
            "training_group_ids",
            "heldout_group_ids",
        ):
            values = tuple(_name(value, field) for value in getattr(self, field))
            if not values or values != tuple(sorted(set(values))):
                raise StageEditContextualSourcePairV2Error(f"{field} must be canonical")
            object.__setattr__(self, field, values)
        if self.heldout_song_id in self.training_song_ids:
            raise StageEditContextualSourcePairV2Error("held-out song leaked into training")
        if set(self.training_reference_ids).intersection(self.heldout_reference_ids):
            raise StageEditContextualSourcePairV2Error("held-out reference leaked into training")
        if set(self.training_group_ids).intersection(self.heldout_group_ids):
            raise StageEditContextualSourcePairV2Error("held-out group leaked into training")
        if not isinstance(self.training, StageEditContextualTrainingResultV2):
            raise StageEditContextualSourcePairV2Error("fold training result is invalid")
        object.__setattr__(
            self,
            "training_corpus_sha256",
            _sha256(self.training_corpus_sha256, "training_corpus_sha256"),
        )
        object.__setattr__(
            self,
            "training_feature_snapshot_bindings_sha256",
            _sha256(
                self.training_feature_snapshot_bindings_sha256,
                "training_feature_snapshot_bindings_sha256",
            ),
        )
        if self.training.training_song_ids != self.training_song_ids:
            raise StageEditContextualSourcePairV2Error(
                "fold embedded training songs differ from declared split"
            )
        if self.training.training_reference_ids != self.training_reference_ids:
            raise StageEditContextualSourcePairV2Error(
                "fold embedded training references differ from declared split"
            )
        if self.training.training_group_ids != self.training_group_ids:
            raise StageEditContextualSourcePairV2Error(
                "fold embedded training groups differ from declared split"
            )
        if self.training.corpus_sha256 != self.training_corpus_sha256:
            raise StageEditContextualSourcePairV2Error(
                "fold embedded training corpus differs from canonical subset"
            )
        if (
            self.training.feature_snapshot_bindings_sha256
            != self.training_feature_snapshot_bindings_sha256
        ):
            raise StageEditContextualSourcePairV2Error(
                "fold embedded snapshot bindings differ from canonical subset"
            )
        expected_identity = (len(self.heldout_reference_ids), len(self.heldout_group_ids))
        for metrics in (self.learned_metrics, self.uniform_metrics, self.handcrafted_metrics):
            if not isinstance(metrics, StageEditSourcePairRankingMetrics) or (
                metrics.reference_count,
                metrics.group_count,
            ) != expected_identity:
                raise StageEditContextualSourcePairV2Error("fold metric identity mismatch")
        expected_gate = stage_edit_source_pair_no_worse_activation_gate(
            self.learned_metrics,
            self.uniform_metrics,
            self.handcrafted_metrics,
            tolerance=STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_TOLERANCE,
        )
        if self.activation_gate != expected_gate:
            raise StageEditContextualSourcePairV2Error("fold activation gate mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "heldout_song_id": self.heldout_song_id,
            "training_song_ids": list(self.training_song_ids),
            "training_reference_ids": list(self.training_reference_ids),
            "heldout_reference_ids": list(self.heldout_reference_ids),
            "training_group_ids": list(self.training_group_ids),
            "heldout_group_ids": list(self.heldout_group_ids),
            "training_corpus_sha256": self.training_corpus_sha256,
            "training_feature_snapshot_bindings_sha256": (
                self.training_feature_snapshot_bindings_sha256
            ),
            "training": self.training.to_dict(),
            "learned": self.learned_metrics.to_dict(),
            "uniform": self.uniform_metrics.to_dict(),
            "handcrafted": self.handcrafted_metrics.to_dict(),
            "activation_gate": self.activation_gate.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "StageEditContextualCrossValidationFoldV2":
        expected = {
            "fold_index",
            "heldout_song_id",
            "training_song_ids",
            "training_reference_ids",
            "heldout_reference_ids",
            "training_group_ids",
            "heldout_group_ids",
            "training_corpus_sha256",
            "training_feature_snapshot_bindings_sha256",
            "training",
            "learned",
            "uniform",
            "handcrafted",
            "activation_gate",
        }
        _strict_keys(payload, expected, "contextual cross-validation fold")
        for field in (
            "training_song_ids",
            "training_reference_ids",
            "heldout_reference_ids",
            "training_group_ids",
            "heldout_group_ids",
        ):
            if not isinstance(payload.get(field), list):
                raise StageEditContextualSourcePairV2Error(
                    f"cross-validation fold {field} must be an array"
                )
        for field in (
            "training",
            "learned",
            "uniform",
            "handcrafted",
            "activation_gate",
        ):
            if not isinstance(payload.get(field), Mapping):
                raise StageEditContextualSourcePairV2Error(
                    f"cross-validation fold {field} must be an object"
                )
        return cls(
            fold_index=_integer(payload["fold_index"], "fold_index"),
            heldout_song_id=payload["heldout_song_id"],
            training_song_ids=tuple(payload["training_song_ids"]),
            training_reference_ids=tuple(payload["training_reference_ids"]),
            heldout_reference_ids=tuple(payload["heldout_reference_ids"]),
            training_group_ids=tuple(payload["training_group_ids"]),
            heldout_group_ids=tuple(payload["heldout_group_ids"]),
            training_corpus_sha256=payload["training_corpus_sha256"],
            training_feature_snapshot_bindings_sha256=payload[
                "training_feature_snapshot_bindings_sha256"
            ],
            training=StageEditContextualTrainingResultV2.from_dict(
                payload["training"]
            ),
            learned_metrics=_ranking_metrics_from_dict(payload["learned"]),
            uniform_metrics=_ranking_metrics_from_dict(payload["uniform"]),
            handcrafted_metrics=_ranking_metrics_from_dict(
                payload["handcrafted"]
            ),
            activation_gate=_activation_gate_from_dict(
                payload["activation_gate"]
            ),
        )


def _aggregate_metrics(
    folds: Sequence[StageEditContextualCrossValidationFoldV2],
    field: str,
    scorer_name: str,
) -> StageEditSourcePairRankingMetrics:
    metrics = tuple(getattr(fold, field) for fold in folds)
    reference_count = sum(value.reference_count for value in metrics)
    group_count = sum(value.group_count for value in metrics)

    def weighted(metric_name: str) -> float:
        return math.fsum(
            value.reference_count * getattr(value, metric_name)
            for value in metrics
        ) / reference_count

    return StageEditSourcePairRankingMetrics(
        scorer_name=scorer_name,
        reference_count=reference_count,
        group_count=group_count,
        reference_balanced_top1_accuracy=weighted("reference_balanced_top1_accuracy"),
        reference_balanced_pairwise_accuracy=weighted("reference_balanced_pairwise_accuracy"),
        reference_balanced_mean_reciprocal_rank=weighted("reference_balanced_mean_reciprocal_rank"),
        reference_balanced_mean_chosen_margin=weighted("reference_balanced_mean_chosen_margin"),
    )


@dataclass(frozen=True, slots=True)
class StageEditContextualCrossValidationResultV2:
    corpus_sha256: str
    source_data_id: str
    feature_snapshot_bindings_sha256: str
    clipped_observed_value_count: int
    token_schema_fingerprint: str
    song_ids: tuple[str, ...]
    reference_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    folds: tuple[StageEditContextualCrossValidationFoldV2, ...]
    learned_metrics: StageEditSourcePairRankingMetrics
    uniform_metrics: StageEditSourcePairRankingMetrics
    handcrafted_metrics: StageEditSourcePairRankingMetrics
    aggregate_gate: StageEditSourcePairActivationGate
    validation_sha256: str
    contract: str = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CROSS_VALIDATION_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CROSS_VALIDATION_CONTRACT:
            raise StageEditContextualSourcePairV2Error("cross-validation contract mismatch")
        _sha256(self.corpus_sha256, "corpus_sha256")
        object.__setattr__(
            self, "source_data_id", _name(self.source_data_id, "source_data_id")
        )
        _sha256(
            self.feature_snapshot_bindings_sha256,
            "feature_snapshot_bindings_sha256",
        )
        _integer(
            self.clipped_observed_value_count,
            "clipped_observed_value_count",
        )
        _sha256(self.token_schema_fingerprint, "token_schema_fingerprint")
        for field in ("song_ids", "reference_ids", "group_ids"):
            values = tuple(getattr(self, field))
            if not values or values != tuple(sorted(set(values))):
                raise StageEditContextualSourcePairV2Error(f"{field} must be canonical")
            object.__setattr__(self, field, values)
        folds = tuple(self.folds)
        if any(
            not isinstance(value, StageEditContextualCrossValidationFoldV2)
            for value in folds
        ):
            raise StageEditContextualSourcePairV2Error(
                "cross-validation folds must be typed"
            )
        if tuple(value.heldout_song_id for value in folds) != self.song_ids:
            raise StageEditContextualSourcePairV2Error("folds do not hold out every song canonically")
        song_universe = set(self.song_ids)
        reference_universe = set(self.reference_ids)
        group_universe = set(self.group_ids)
        for fold_index, fold in enumerate(folds):
            if fold.fold_index != fold_index:
                raise StageEditContextualSourcePairV2Error(
                    "fold indices must be canonical and contiguous"
                )
            if set(fold.training_song_ids) != song_universe - {
                fold.heldout_song_id
            }:
                raise StageEditContextualSourcePairV2Error(
                    "fold training songs are not the exact held-out complement"
                )
            if (
                set(fold.training_reference_ids)
                | set(fold.heldout_reference_ids)
                != reference_universe
                or set(fold.training_reference_ids)
                & set(fold.heldout_reference_ids)
            ):
                raise StageEditContextualSourcePairV2Error(
                    "fold references are not an exact universe partition"
                )
            if (
                set(fold.training_group_ids) | set(fold.heldout_group_ids)
                != group_universe
                or set(fold.training_group_ids) & set(fold.heldout_group_ids)
            ):
                raise StageEditContextualSourcePairV2Error(
                    "fold groups are not an exact universe partition"
                )
            if (
                fold.training.scorer.token_schema_fingerprint
                != self.token_schema_fingerprint
            ):
                raise StageEditContextualSourcePairV2Error(
                    "fold training scorer token schema differs from validation"
                )
        heldout_references = tuple(
            reference_id
            for fold in folds
            for reference_id in fold.heldout_reference_ids
        )
        if tuple(sorted(heldout_references)) != self.reference_ids:
            raise StageEditContextualSourcePairV2Error(
                "folds do not cover every reference exactly once"
            )
        heldout_groups = tuple(group_id for fold in folds for group_id in fold.heldout_group_ids)
        if tuple(sorted(heldout_groups)) != self.group_ids:
            raise StageEditContextualSourcePairV2Error("folds do not cover every group exactly once")
        expected_metrics = (
            _aggregate_metrics(folds, "learned_metrics", "contextual_v2:song_disjoint_loso"),
            _aggregate_metrics(folds, "uniform_metrics", "uniform:song_disjoint_loso"),
            _aggregate_metrics(
                folds,
                "handcrafted_metrics",
                f"{STAGE_EDIT_SOURCE_PAIR_HANDCRAFTED_BASELINE_CONTRACT}:song_disjoint_loso",
            ),
        )
        if expected_metrics != (
            self.learned_metrics,
            self.uniform_metrics,
            self.handcrafted_metrics,
        ):
            raise StageEditContextualSourcePairV2Error("aggregate metrics do not round-trip")
        expected_gate = stage_edit_source_pair_no_worse_activation_gate(
            self.learned_metrics,
            self.uniform_metrics,
            self.handcrafted_metrics,
            tolerance=STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_TOLERANCE,
        )
        if self.aggregate_gate != expected_gate:
            raise StageEditContextualSourcePairV2Error("aggregate activation gate mismatch")
        object.__setattr__(self, "folds", folds)
        digest = _sha256(self.validation_sha256, "validation_sha256")
        object.__setattr__(self, "validation_sha256", digest)
        if digest != _canonical_sha256(self.unsigned_dict()):
            raise StageEditContextualSourcePairV2Error("validation digest does not match contents")

    @property
    def authorized(self) -> bool:
        return self.aggregate_gate.authorized and all(
            fold.activation_gate.authorized for fold in self.folds
        )

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "corpus_sha256": self.corpus_sha256,
            "source_data_id": self.source_data_id,
            "feature_snapshot_bindings_sha256": self.feature_snapshot_bindings_sha256,
            "clipped_observed_value_count": self.clipped_observed_value_count,
            "token_schema_fingerprint": self.token_schema_fingerprint,
            "song_ids": list(self.song_ids),
            "reference_ids": list(self.reference_ids),
            "group_ids": list(self.group_ids),
            "folds": [value.to_dict() for value in self.folds],
            "aggregate": {
                "learned": self.learned_metrics.to_dict(),
                "uniform": self.uniform_metrics.to_dict(),
                "handcrafted": self.handcrafted_metrics.to_dict(),
                "activation_gate": self.aggregate_gate.to_dict(),
            },
            "all_fold_and_aggregate_gate_passed": self.authorized,
            "activation_gate_contract": STAGE_EDIT_SOURCE_PAIR_ACTIVATION_GATE_CONTRACT,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "validation_sha256": self.validation_sha256}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "StageEditContextualCrossValidationResultV2":
        expected = {
            "contract",
            "corpus_sha256",
            "source_data_id",
            "feature_snapshot_bindings_sha256",
            "clipped_observed_value_count",
            "token_schema_fingerprint",
            "song_ids",
            "reference_ids",
            "group_ids",
            "folds",
            "aggregate",
            "all_fold_and_aggregate_gate_passed",
            "activation_gate_contract",
            "validation_sha256",
        }
        _strict_keys(payload, expected, "contextual cross-validation result")
        for field in ("song_ids", "reference_ids", "group_ids", "folds"):
            if not isinstance(payload.get(field), list):
                raise StageEditContextualSourcePairV2Error(
                    f"cross-validation result {field} must be an array"
                )
        if any(not isinstance(value, Mapping) for value in payload["folds"]):
            raise StageEditContextualSourcePairV2Error(
                "cross-validation folds must be objects"
            )
        aggregate = payload.get("aggregate")
        if not isinstance(aggregate, Mapping):
            raise StageEditContextualSourcePairV2Error(
                "cross-validation aggregate must be an object"
            )
        _strict_keys(
            aggregate,
            {"learned", "uniform", "handcrafted", "activation_gate"},
            "cross-validation aggregate",
        )
        if any(not isinstance(value, Mapping) for value in aggregate.values()):
            raise StageEditContextualSourcePairV2Error(
                "cross-validation aggregate entries must be objects"
            )
        if (
            payload["activation_gate_contract"]
            != STAGE_EDIT_SOURCE_PAIR_ACTIVATION_GATE_CONTRACT
        ):
            raise StageEditContextualSourcePairV2Error(
                "cross-validation activation gate contract mismatch"
            )
        result = cls(
            contract=payload["contract"],
            corpus_sha256=payload["corpus_sha256"],
            source_data_id=payload["source_data_id"],
            feature_snapshot_bindings_sha256=payload[
                "feature_snapshot_bindings_sha256"
            ],
            clipped_observed_value_count=_integer(
                payload["clipped_observed_value_count"],
                "clipped_observed_value_count",
            ),
            token_schema_fingerprint=payload["token_schema_fingerprint"],
            song_ids=tuple(payload["song_ids"]),
            reference_ids=tuple(payload["reference_ids"]),
            group_ids=tuple(payload["group_ids"]),
            folds=tuple(
                StageEditContextualCrossValidationFoldV2.from_dict(value)
                for value in payload["folds"]
            ),
            learned_metrics=_ranking_metrics_from_dict(aggregate["learned"]),
            uniform_metrics=_ranking_metrics_from_dict(aggregate["uniform"]),
            handcrafted_metrics=_ranking_metrics_from_dict(
                aggregate["handcrafted"]
            ),
            aggregate_gate=_activation_gate_from_dict(
                aggregate["activation_gate"]
            ),
            validation_sha256=payload["validation_sha256"],
        )
        if payload["all_fold_and_aggregate_gate_passed"] is not result.authorized:
            raise StageEditContextualSourcePairV2Error(
                "cross-validation authorization does not match typed evidence"
            )
        return result


def cross_validate_stage_edit_contextual_source_pair_v2(
    corpus: StageEditContextualTrainingCorpusV2,
    *,
    config: StageEditContextualTrainingConfigV2 = StageEditContextualTrainingConfigV2(),
) -> StageEditContextualCrossValidationResultV2:
    if not isinstance(corpus, StageEditContextualTrainingCorpusV2):
        raise StageEditContextualSourcePairV2Error("cross-validation corpus is invalid")
    if len(corpus.song_ids) < 2:
        raise StageEditContextualSourcePairV2Error("LOSO needs at least two songs")
    groups_by_song = {
        song_id: tuple(value for value in corpus.groups if value.song_id == song_id)
        for song_id in corpus.song_ids
    }
    folds: list[StageEditContextualCrossValidationFoldV2] = []
    for fold_index, heldout_song_id in enumerate(corpus.song_ids):
        training_groups = tuple(
            value for value in corpus.groups if value.song_id != heldout_song_id
        )
        heldout_groups = groups_by_song[heldout_song_id]
        subset = new_stage_edit_contextual_training_corpus_v2(
            training_groups,
            token_schema_fingerprint=corpus.token_schema_fingerprint,
            source_data_id=corpus.source_data_id,
        )
        training = train_stage_edit_contextual_source_pair_v2(subset, config=config)
        learned = evaluate_stage_edit_contextual_source_pair_ranking_v2(
            heldout_groups,
            scorer=training.scorer,
            scorer_name=f"contextual_v2:heldout:{heldout_song_id}",
        )
        uniform = evaluate_stage_edit_contextual_source_pair_ranking_v2(
            heldout_groups,
            scorer=None,
            scorer_name=f"uniform:heldout:{heldout_song_id}",
            baseline="uniform",
        )
        handcrafted = evaluate_stage_edit_contextual_source_pair_ranking_v2(
            heldout_groups,
            scorer=None,
            scorer_name=f"{STAGE_EDIT_SOURCE_PAIR_HANDCRAFTED_BASELINE_CONTRACT}:heldout:{heldout_song_id}",
            baseline="handcrafted",
        )
        gate = stage_edit_source_pair_no_worse_activation_gate(
            learned,
            uniform,
            handcrafted,
            tolerance=config.tolerance,
        )
        folds.append(
            StageEditContextualCrossValidationFoldV2(
                fold_index=fold_index,
                heldout_song_id=heldout_song_id,
                training_song_ids=tuple(sorted({value.song_id for value in training_groups})),
                training_reference_ids=tuple(sorted({value.reference_id for value in training_groups})),
                heldout_reference_ids=tuple(sorted({value.reference_id for value in heldout_groups})),
                training_group_ids=tuple(sorted(value.group_id for value in training_groups)),
                heldout_group_ids=tuple(sorted(value.group_id for value in heldout_groups)),
                training_corpus_sha256=subset.content_sha256,
                training_feature_snapshot_bindings_sha256=(
                    subset.feature_snapshot_bindings_sha256
                ),
                training=training,
                learned_metrics=learned,
                uniform_metrics=uniform,
                handcrafted_metrics=handcrafted,
                activation_gate=gate,
            )
        )
    fold_values = tuple(folds)
    learned_aggregate = _aggregate_metrics(
        fold_values, "learned_metrics", "contextual_v2:song_disjoint_loso"
    )
    uniform_aggregate = _aggregate_metrics(
        fold_values, "uniform_metrics", "uniform:song_disjoint_loso"
    )
    handcrafted_aggregate = _aggregate_metrics(
        fold_values,
        "handcrafted_metrics",
        f"{STAGE_EDIT_SOURCE_PAIR_HANDCRAFTED_BASELINE_CONTRACT}:song_disjoint_loso",
    )
    aggregate_gate = stage_edit_source_pair_no_worse_activation_gate(
        learned_aggregate,
        uniform_aggregate,
        handcrafted_aggregate,
        tolerance=config.tolerance,
    )
    unsigned = {
        "contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CROSS_VALIDATION_CONTRACT,
        "corpus_sha256": corpus.content_sha256,
        "source_data_id": corpus.source_data_id,
        "feature_snapshot_bindings_sha256": corpus.feature_snapshot_bindings_sha256,
        "clipped_observed_value_count": corpus.clipped_observed_value_count,
        "token_schema_fingerprint": corpus.token_schema_fingerprint,
        "song_ids": list(corpus.song_ids),
        "reference_ids": list(corpus.reference_ids),
        "group_ids": list(tuple(sorted(corpus.group_ids))),
        "folds": [value.to_dict() for value in fold_values],
        "aggregate": {
            "learned": learned_aggregate.to_dict(),
            "uniform": uniform_aggregate.to_dict(),
            "handcrafted": handcrafted_aggregate.to_dict(),
            "activation_gate": aggregate_gate.to_dict(),
        },
        "all_fold_and_aggregate_gate_passed": aggregate_gate.authorized
        and all(value.activation_gate.authorized for value in fold_values),
        "activation_gate_contract": STAGE_EDIT_SOURCE_PAIR_ACTIVATION_GATE_CONTRACT,
    }
    return StageEditContextualCrossValidationResultV2(
        corpus_sha256=corpus.content_sha256,
        source_data_id=corpus.source_data_id,
        feature_snapshot_bindings_sha256=corpus.feature_snapshot_bindings_sha256,
        clipped_observed_value_count=corpus.clipped_observed_value_count,
        token_schema_fingerprint=corpus.token_schema_fingerprint,
        song_ids=corpus.song_ids,
        reference_ids=corpus.reference_ids,
        group_ids=tuple(sorted(corpus.group_ids)),
        folds=fold_values,
        learned_metrics=learned_aggregate,
        uniform_metrics=uniform_aggregate,
        handcrafted_metrics=handcrafted_aggregate,
        aggregate_gate=aggregate_gate,
        validation_sha256=_canonical_sha256(unsigned),
    )


@dataclass(frozen=True, slots=True)
class StageEditContextualSourcePairCheckpointV2:
    format_version: int
    authority: str
    contract: str
    checkpoint_id: str
    created_at: str
    token_schema_fingerprint: str
    feature_contract: str
    feature_schema_sha256: str
    feature_names: tuple[str, ...]
    objective_contract: str
    training_config: Mapping[str, Any]
    corpus_sha256: str
    source_data_id: str
    feature_snapshot_bindings_sha256: str
    clipped_observed_value_count: int
    training_song_ids: tuple[str, ...]
    training_reference_ids: tuple[str, ...]
    training_group_ids: tuple[str, ...]
    runtime_equivalence_sha256: str
    scorer_fingerprint: str
    weights: tuple[float, ...]
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if self.format_version != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FORMAT_VERSION:
            raise StageEditContextualSourcePairV2Error("checkpoint format mismatch")
        if self.authority != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_AUTHORITY:
            raise StageEditContextualSourcePairV2Error("checkpoint authority mismatch")
        if self.contract != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT:
            raise StageEditContextualSourcePairV2Error("checkpoint scorer contract mismatch")
        object.__setattr__(self, "checkpoint_id", _name(self.checkpoint_id, "checkpoint_id"))
        created = _name(self.created_at, "created_at")
        try:
            timestamp = datetime.fromisoformat(created)
        except ValueError as exc:
            raise StageEditContextualSourcePairV2Error("created_at must be ISO-8601") from exc
        if timestamp.tzinfo is None:
            raise StageEditContextualSourcePairV2Error("created_at must include timezone")
        object.__setattr__(self, "created_at", created)
        _sha256(self.token_schema_fingerprint, "token_schema_fingerprint")
        if self.feature_contract != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_CONTRACT:
            raise StageEditContextualSourcePairV2Error("checkpoint feature contract mismatch")
        if self.feature_schema_sha256 != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_SCHEMA_SHA256:
            raise StageEditContextualSourcePairV2Error("checkpoint feature schema digest mismatch")
        if tuple(self.feature_names) != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES:
            raise StageEditContextualSourcePairV2Error("checkpoint feature names mismatch")
        if self.objective_contract != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_OBJECTIVE_CONTRACT:
            raise StageEditContextualSourcePairV2Error("checkpoint objective contract mismatch")
        config = dict(self.training_config)
        if config != StageEditContextualTrainingConfigV2().to_dict():
            raise StageEditContextualSourcePairV2Error("checkpoint training config mismatch")
        object.__setattr__(self, "training_config", config)
        _sha256(self.corpus_sha256, "corpus_sha256")
        object.__setattr__(
            self, "source_data_id", _name(self.source_data_id, "source_data_id")
        )
        _sha256(
            self.feature_snapshot_bindings_sha256,
            "feature_snapshot_bindings_sha256",
        )
        _integer(
            self.clipped_observed_value_count,
            "clipped_observed_value_count",
        )
        for field in ("training_song_ids", "training_reference_ids", "training_group_ids"):
            values = tuple(getattr(self, field))
            if not values or values != tuple(sorted(set(values))):
                raise StageEditContextualSourcePairV2Error(f"checkpoint {field} is not canonical")
            object.__setattr__(self, field, values)
        _sha256(self.runtime_equivalence_sha256, "runtime_equivalence_sha256")
        scorer = StageEditContextualSourcePairScorerV2(
            token_schema_fingerprint=self.token_schema_fingerprint,
            weights=tuple(self.weights),
            feature_names=tuple(self.feature_names),
            contract=self.contract,
            authority=self.authority,
        )
        object.__setattr__(self, "weights", scorer.weights)
        if self.scorer_fingerprint != scorer.fingerprint:
            raise StageEditContextualSourcePairV2Error("checkpoint scorer fingerprint mismatch")
        digest = _sha256(self.checkpoint_sha256, "checkpoint_sha256")
        object.__setattr__(self, "checkpoint_sha256", digest)
        if digest != _canonical_sha256(self.unsigned_dict()):
            raise StageEditContextualSourcePairV2Error("checkpoint digest does not match contents")

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "authority": self.authority,
            "contract": self.contract,
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at,
            "token_schema_fingerprint": self.token_schema_fingerprint,
            "feature_contract": self.feature_contract,
            "feature_schema_sha256": self.feature_schema_sha256,
            "feature_names": list(self.feature_names),
            "objective_contract": self.objective_contract,
            "training_config": dict(self.training_config),
            "corpus_sha256": self.corpus_sha256,
            "source_data_id": self.source_data_id,
            "feature_snapshot_bindings_sha256": self.feature_snapshot_bindings_sha256,
            "clipped_observed_value_count": self.clipped_observed_value_count,
            "training_song_ids": list(self.training_song_ids),
            "training_reference_ids": list(self.training_reference_ids),
            "training_group_ids": list(self.training_group_ids),
            "runtime_equivalence_sha256": self.runtime_equivalence_sha256,
            "scorer_fingerprint": self.scorer_fingerprint,
            "weights": list(self.weights),
            "bounded_score_contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BOUNDED_SCORE_CONTRACT,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "checkpoint_sha256": self.checkpoint_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageEditContextualSourcePairCheckpointV2":
        expected = {
            "format_version",
            "authority",
            "contract",
            "checkpoint_id",
            "created_at",
            "token_schema_fingerprint",
            "feature_contract",
            "feature_schema_sha256",
            "feature_names",
            "objective_contract",
            "training_config",
            "corpus_sha256",
            "source_data_id",
            "feature_snapshot_bindings_sha256",
            "clipped_observed_value_count",
            "training_song_ids",
            "training_reference_ids",
            "training_group_ids",
            "runtime_equivalence_sha256",
            "scorer_fingerprint",
            "weights",
            "bounded_score_contract",
            "checkpoint_sha256",
        }
        _strict_keys(payload, expected, "contextual v2 checkpoint")
        if payload.get("bounded_score_contract") != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BOUNDED_SCORE_CONTRACT:
            raise StageEditContextualSourcePairV2Error("checkpoint bounded-score contract mismatch")
        if not isinstance(payload.get("training_config"), Mapping):
            raise StageEditContextualSourcePairV2Error("checkpoint training config must be an object")
        return cls(
            format_version=_integer(payload["format_version"], "format_version", minimum=1),
            authority=payload["authority"],
            contract=payload["contract"],
            checkpoint_id=payload["checkpoint_id"],
            created_at=payload["created_at"],
            token_schema_fingerprint=payload["token_schema_fingerprint"],
            feature_contract=payload["feature_contract"],
            feature_schema_sha256=payload["feature_schema_sha256"],
            feature_names=tuple(payload["feature_names"]),
            objective_contract=payload["objective_contract"],
            training_config=dict(payload["training_config"]),
            corpus_sha256=payload["corpus_sha256"],
            source_data_id=payload["source_data_id"],
            feature_snapshot_bindings_sha256=payload[
                "feature_snapshot_bindings_sha256"
            ],
            clipped_observed_value_count=_integer(
                payload["clipped_observed_value_count"],
                "clipped_observed_value_count",
            ),
            training_song_ids=tuple(payload["training_song_ids"]),
            training_reference_ids=tuple(payload["training_reference_ids"]),
            training_group_ids=tuple(payload["training_group_ids"]),
            runtime_equivalence_sha256=payload["runtime_equivalence_sha256"],
            scorer_fingerprint=payload["scorer_fingerprint"],
            weights=tuple(payload["weights"]),
            checkpoint_sha256=payload["checkpoint_sha256"],
        )


def new_stage_edit_contextual_source_pair_checkpoint_v2(
    checkpoint_id: str,
    result: StageEditContextualTrainingResultV2,
    *,
    created_at: str | None = None,
) -> StageEditContextualSourcePairCheckpointV2:
    if not isinstance(result, StageEditContextualTrainingResultV2):
        raise StageEditContextualSourcePairV2Error("checkpoint training result is invalid")
    unsigned = {
        "format_version": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FORMAT_VERSION,
        "authority": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_AUTHORITY,
        "contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT,
        "checkpoint_id": _name(checkpoint_id, "checkpoint_id"),
        "created_at": datetime.now(UTC).isoformat() if created_at is None else _name(created_at, "created_at"),
        "token_schema_fingerprint": result.scorer.token_schema_fingerprint,
        "feature_contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_CONTRACT,
        "feature_schema_sha256": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_SCHEMA_SHA256,
        "feature_names": list(result.scorer.feature_names),
        "objective_contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_OBJECTIVE_CONTRACT,
        "training_config": result.config.to_dict(),
        "corpus_sha256": result.corpus_sha256,
        "source_data_id": result.source_data_id,
        "feature_snapshot_bindings_sha256": result.feature_snapshot_bindings_sha256,
        "clipped_observed_value_count": result.clipped_observed_value_count,
        "training_song_ids": list(result.training_song_ids),
        "training_reference_ids": list(result.training_reference_ids),
        "training_group_ids": list(result.training_group_ids),
        "runtime_equivalence_sha256": result.runtime_equivalence_sha256,
        "scorer_fingerprint": result.scorer.fingerprint,
        "weights": list(result.scorer.weights),
        "bounded_score_contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BOUNDED_SCORE_CONTRACT,
    }
    return StageEditContextualSourcePairCheckpointV2(
        format_version=unsigned["format_version"],
        authority=unsigned["authority"],
        contract=unsigned["contract"],
        checkpoint_id=unsigned["checkpoint_id"],
        created_at=unsigned["created_at"],
        token_schema_fingerprint=unsigned["token_schema_fingerprint"],
        feature_contract=unsigned["feature_contract"],
        feature_schema_sha256=unsigned["feature_schema_sha256"],
        feature_names=tuple(unsigned["feature_names"]),
        objective_contract=unsigned["objective_contract"],
        training_config=unsigned["training_config"],
        corpus_sha256=unsigned["corpus_sha256"],
        source_data_id=unsigned["source_data_id"],
        feature_snapshot_bindings_sha256=unsigned[
            "feature_snapshot_bindings_sha256"
        ],
        clipped_observed_value_count=unsigned["clipped_observed_value_count"],
        training_song_ids=tuple(unsigned["training_song_ids"]),
        training_reference_ids=tuple(unsigned["training_reference_ids"]),
        training_group_ids=tuple(unsigned["training_group_ids"]),
        runtime_equivalence_sha256=unsigned["runtime_equivalence_sha256"],
        scorer_fingerprint=unsigned["scorer_fingerprint"],
        weights=tuple(unsigned["weights"]),
        checkpoint_sha256=_canonical_sha256(unsigned),
    )


@dataclass(frozen=True, slots=True)
class StageEditContextualBeamQACaseV2:
    """One full-decode, report-bound song case with measured QA inputs."""

    case_id: str
    song_id: str
    request_sha256: str
    decode_report_sha256: str
    full_decode: bool
    duration_seconds: float
    switch_count: int
    eligible_switch_opportunity_count: int
    visual_segment_median_seconds: float
    source_run_median_seconds: float
    longest_source_run_seconds: float
    distinct_source_count: int
    effective_source_count: float
    dominant_source_ratio: float
    quality_score_p10: float
    quality_score_median_span: float
    boundary_match_score_p10: float
    weighted_proxy_score: float
    highlight_window_count: int
    highlight_switch_opportunity_count: int
    highlight_switch_count: int
    verified_highlight_switch_count: int
    closeup_motion_opportunity_count: int
    closeup_motion_switch_count: int
    verified_closeup_motion_switch_count: int
    geometry_issue_count: int
    sync_issue_count: int
    case_sha256: str | None = None
    contract: str = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BEAM_QA_CASE_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BEAM_QA_CASE_CONTRACT:
            raise StageEditContextualSourcePairV2Error("beam QA case contract mismatch")
        object.__setattr__(self, "case_id", _name(self.case_id, "case_id"))
        object.__setattr__(self, "song_id", _name(self.song_id, "song_id"))
        for field in ("request_sha256", "decode_report_sha256"):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        if not isinstance(self.full_decode, bool):
            raise StageEditContextualSourcePairV2Error("full_decode must be boolean")
        duration = _finite(self.duration_seconds, "duration_seconds")
        if duration <= 0.0:
            raise StageEditContextualSourcePairV2Error(
                "beam QA duration must be positive"
            )
        object.__setattr__(self, "duration_seconds", duration)
        for field in (
            "switch_count",
            "eligible_switch_opportunity_count",
            "distinct_source_count",
            "highlight_window_count",
            "highlight_switch_opportunity_count",
            "highlight_switch_count",
            "verified_highlight_switch_count",
            "closeup_motion_opportunity_count",
            "closeup_motion_switch_count",
            "verified_closeup_motion_switch_count",
            "geometry_issue_count",
            "sync_issue_count",
        ):
            _integer(getattr(self, field), field)
        if (
            self.switch_count > self.eligible_switch_opportunity_count
            or self.highlight_switch_count > self.highlight_switch_opportunity_count
            or self.verified_highlight_switch_count > self.highlight_switch_count
            or self.closeup_motion_switch_count > self.closeup_motion_opportunity_count
            or self.verified_closeup_motion_switch_count
            > self.closeup_motion_switch_count
            or self.highlight_switch_count > self.switch_count
            or self.closeup_motion_switch_count > self.switch_count
        ):
            raise StageEditContextualSourcePairV2Error(
                "beam QA measured counts are internally inconsistent"
            )
        if self.eligible_switch_opportunity_count < 1:
            raise StageEditContextualSourcePairV2Error(
                "beam QA needs measured switch opportunities"
            )
        nonnegative_fields = (
            "visual_segment_median_seconds",
            "source_run_median_seconds",
            "longest_source_run_seconds",
            "quality_score_median_span",
            "boundary_match_score_p10",
            "weighted_proxy_score",
        )
        for field in nonnegative_fields:
            value = _finite(getattr(self, field), field)
            if value < 0.0:
                raise StageEditContextualSourcePairV2Error(
                    f"{field} must be non-negative"
                )
            object.__setattr__(self, field, value)
        if self.longest_source_run_seconds < self.source_run_median_seconds:
            raise StageEditContextualSourcePairV2Error(
                "longest source run cannot be shorter than the median"
            )
        effective = _finite(self.effective_source_count, "effective_source_count")
        if not 0.0 <= effective <= float(self.distinct_source_count):
            raise StageEditContextualSourcePairV2Error(
                "effective source count must lie within distinct source count"
            )
        object.__setattr__(self, "effective_source_count", effective)
        for field in ("dominant_source_ratio", "quality_score_p10"):
            value = _finite(getattr(self, field), field)
            if not 0.0 <= value <= 1.0:
                raise StageEditContextualSourcePairV2Error(
                    f"{field} must lie in [0, 1]"
                )
            object.__setattr__(self, field, value)
        for field in ("boundary_match_score_p10", "weighted_proxy_score"):
            if getattr(self, field) > 100.0:
                raise StageEditContextualSourcePairV2Error(
                    f"{field} must lie in [0, 100]"
                )
        expected = _canonical_sha256(self.unsigned_dict())
        if self.case_sha256 is None:
            object.__setattr__(self, "case_sha256", expected)
        else:
            digest = _sha256(self.case_sha256, "case_sha256")
            object.__setattr__(self, "case_sha256", digest)
            if digest != expected:
                raise StageEditContextualSourcePairV2Error(
                    "beam QA case digest does not match measured contents"
                )

    @property
    def switches_per_minute(self) -> float:
        return 60.0 * self.switch_count / self.duration_seconds

    @property
    def switch_ratio(self) -> float:
        return self.switch_count / self.eligible_switch_opportunity_count

    @property
    def gates(self) -> dict[str, bool]:
        threshold = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BEAM_QA_THRESHOLDS
        return {
            "full_decode": self.full_decode,
            "switches_per_minute_hard": self.switches_per_minute
            >= float(threshold["switches_per_minute_hard_minimum"]),
            "switches_per_minute_target": float(
                threshold["switches_per_minute_target_minimum"]
            )
            <= self.switches_per_minute
            <= float(threshold["switches_per_minute_target_maximum"]),
            "switch_ratio": float(threshold["switch_ratio_minimum"])
            <= self.switch_ratio
            <= float(threshold["switch_ratio_maximum"]),
            "visual_segment_median": self.visual_segment_median_seconds
            <= float(threshold["visual_segment_median_seconds_maximum"]),
            "source_run_median": float(
                threshold["source_run_median_seconds_minimum"]
            )
            <= self.source_run_median_seconds
            <= float(threshold["source_run_median_seconds_maximum"]),
            "longest_source_run": self.longest_source_run_seconds
            <= float(threshold["longest_source_run_seconds_maximum"]),
            "distinct_sources": self.distinct_source_count
            >= int(threshold["distinct_source_count_minimum"]),
            "effective_sources": self.effective_source_count
            >= float(threshold["effective_source_count_minimum"]),
            "dominant_source_hard": self.dominant_source_ratio
            <= float(threshold["dominant_source_ratio_hard_maximum"]),
            "dominant_source_target": self.dominant_source_ratio
            <= float(threshold["dominant_source_ratio_target_maximum"]),
            "quality_p10": self.quality_score_p10
            >= float(threshold["quality_score_p10_minimum"]),
            "quality_score_median_span": self.quality_score_median_span
            <= float(threshold["quality_score_median_span_maximum"]),
            "boundary_match_p10": self.boundary_match_score_p10
            >= float(threshold["boundary_match_score_p10_minimum"]),
            "weighted_proxy": self.weighted_proxy_score
            >= float(threshold["weighted_proxy_score_minimum"]),
            "highlight_evidence": self.highlight_window_count
            >= int(threshold["highlight_window_count_minimum"])
            and self.highlight_switch_count
            >= int(threshold["highlight_switch_count_minimum"])
            and self.verified_highlight_switch_count
            == self.highlight_switch_count,
            "closeup_motion_evidence": self.closeup_motion_opportunity_count
            >= int(threshold["closeup_motion_opportunity_count_minimum"])
            and self.closeup_motion_switch_count
            >= int(threshold["closeup_motion_switch_count_minimum"])
            and self.verified_closeup_motion_switch_count
            == self.closeup_motion_switch_count,
            "geometry_issues_zero": self.geometry_issue_count
            <= int(threshold["geometry_issue_count_maximum"]),
            "sync_issues_zero": self.sync_issue_count
            <= int(threshold["sync_issue_count_maximum"]),
        }

    @property
    def hard_safe(self) -> bool:
        gates = self.gates
        target_only = {"switches_per_minute_target", "dominant_source_target"}
        return all(value for name, value in gates.items() if name not in target_only)

    @property
    def authorized(self) -> bool:
        return self.hard_safe and all(self.gates.values())

    @property
    def target_profile_met(self) -> bool:
        return self.gates["switches_per_minute_target"] and self.gates[
            "dominant_source_target"
        ]

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "case_id": self.case_id,
            "song_id": self.song_id,
            "request_sha256": self.request_sha256,
            "decode_report_sha256": self.decode_report_sha256,
            "full_decode": self.full_decode,
            "duration_seconds": self.duration_seconds,
            "switch_count": self.switch_count,
            "eligible_switch_opportunity_count": self.eligible_switch_opportunity_count,
            "switches_per_minute": self.switches_per_minute,
            "switch_ratio": self.switch_ratio,
            "visual_segment_median_seconds": self.visual_segment_median_seconds,
            "source_run_median_seconds": self.source_run_median_seconds,
            "longest_source_run_seconds": self.longest_source_run_seconds,
            "distinct_source_count": self.distinct_source_count,
            "effective_source_count": self.effective_source_count,
            "dominant_source_ratio": self.dominant_source_ratio,
            "quality_score_p10": self.quality_score_p10,
            "quality_score_median_span": self.quality_score_median_span,
            "boundary_match_score_p10": self.boundary_match_score_p10,
            "weighted_proxy_score": self.weighted_proxy_score,
            "highlight_window_count": self.highlight_window_count,
            "highlight_switch_opportunity_count": self.highlight_switch_opportunity_count,
            "highlight_switch_count": self.highlight_switch_count,
            "verified_highlight_switch_count": self.verified_highlight_switch_count,
            "closeup_motion_opportunity_count": self.closeup_motion_opportunity_count,
            "closeup_motion_switch_count": self.closeup_motion_switch_count,
            "verified_closeup_motion_switch_count": self.verified_closeup_motion_switch_count,
            "geometry_issue_count": self.geometry_issue_count,
            "sync_issue_count": self.sync_issue_count,
            "gates": self.gates,
            "hard_safe": self.hard_safe,
            "target_profile_met": self.target_profile_met,
            "authorized": self.authorized,
        }

    def to_dict(self) -> dict[str, Any]:
        assert self.case_sha256 is not None
        return {**self.unsigned_dict(), "case_sha256": self.case_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageEditContextualBeamQACaseV2":
        measured_fields = {
            "contract",
            "case_id",
            "song_id",
            "request_sha256",
            "decode_report_sha256",
            "full_decode",
            "duration_seconds",
            "switch_count",
            "eligible_switch_opportunity_count",
            "switches_per_minute",
            "switch_ratio",
            "visual_segment_median_seconds",
            "source_run_median_seconds",
            "longest_source_run_seconds",
            "distinct_source_count",
            "effective_source_count",
            "dominant_source_ratio",
            "quality_score_p10",
            "quality_score_median_span",
            "boundary_match_score_p10",
            "weighted_proxy_score",
            "highlight_window_count",
            "highlight_switch_opportunity_count",
            "highlight_switch_count",
            "verified_highlight_switch_count",
            "closeup_motion_opportunity_count",
            "closeup_motion_switch_count",
            "verified_closeup_motion_switch_count",
            "geometry_issue_count",
            "sync_issue_count",
            "gates",
            "hard_safe",
            "target_profile_met",
            "authorized",
            "case_sha256",
        }
        _strict_keys(payload, measured_fields, "beam QA measured case")
        result = cls(
            contract=payload["contract"],
            case_id=payload["case_id"],
            song_id=payload["song_id"],
            request_sha256=payload["request_sha256"],
            decode_report_sha256=payload["decode_report_sha256"],
            full_decode=payload["full_decode"],
            duration_seconds=payload["duration_seconds"],
            switch_count=_integer(payload["switch_count"], "switch_count"),
            eligible_switch_opportunity_count=_integer(
                payload["eligible_switch_opportunity_count"],
                "eligible_switch_opportunity_count",
            ),
            visual_segment_median_seconds=payload["visual_segment_median_seconds"],
            source_run_median_seconds=payload["source_run_median_seconds"],
            longest_source_run_seconds=payload["longest_source_run_seconds"],
            distinct_source_count=_integer(
                payload["distinct_source_count"], "distinct_source_count"
            ),
            effective_source_count=payload["effective_source_count"],
            dominant_source_ratio=payload["dominant_source_ratio"],
            quality_score_p10=payload["quality_score_p10"],
            quality_score_median_span=payload["quality_score_median_span"],
            boundary_match_score_p10=payload["boundary_match_score_p10"],
            weighted_proxy_score=payload["weighted_proxy_score"],
            highlight_window_count=_integer(
                payload["highlight_window_count"], "highlight_window_count"
            ),
            highlight_switch_opportunity_count=_integer(
                payload["highlight_switch_opportunity_count"],
                "highlight_switch_opportunity_count",
            ),
            highlight_switch_count=_integer(
                payload["highlight_switch_count"], "highlight_switch_count"
            ),
            verified_highlight_switch_count=_integer(
                payload["verified_highlight_switch_count"],
                "verified_highlight_switch_count",
            ),
            closeup_motion_opportunity_count=_integer(
                payload["closeup_motion_opportunity_count"],
                "closeup_motion_opportunity_count",
            ),
            closeup_motion_switch_count=_integer(
                payload["closeup_motion_switch_count"],
                "closeup_motion_switch_count",
            ),
            verified_closeup_motion_switch_count=_integer(
                payload["verified_closeup_motion_switch_count"],
                "verified_closeup_motion_switch_count",
            ),
            geometry_issue_count=_integer(
                payload["geometry_issue_count"], "geometry_issue_count"
            ),
            sync_issue_count=_integer(payload["sync_issue_count"], "sync_issue_count"),
            case_sha256=payload["case_sha256"],
        )
        if (
            payload["switches_per_minute"] != result.switches_per_minute
            or payload["switch_ratio"] != result.switch_ratio
            or payload["gates"] != result.gates
            or payload["hard_safe"] is not result.hard_safe
            or payload["target_profile_met"] is not result.target_profile_met
            or payload["authorized"] is not result.authorized
        ):
            raise StageEditContextualSourcePairV2Error(
                "beam QA derived case fields do not match measured inputs"
            )
        return result


@dataclass(frozen=True, slots=True)
class StageEditContextualBeamQAEvidenceV2:
    checkpoint_sha256: str
    evaluation_corpus_sha256: str
    cases: tuple[StageEditContextualBeamQACaseV2, ...]
    qa_report_sha256: str | None = None
    contract: str = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BEAM_QA_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BEAM_QA_CONTRACT:
            raise StageEditContextualSourcePairV2Error("beam QA contract mismatch")
        for field in ("checkpoint_sha256", "evaluation_corpus_sha256"):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        cases = tuple(
            sorted(self.cases, key=lambda value: (value.song_id, value.case_id))
        )
        minimum_cases = int(
            STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BEAM_QA_THRESHOLDS[
                "minimum_song_disjoint_case_count"
            ]
        )
        if len(cases) < minimum_cases or any(
            not isinstance(value, StageEditContextualBeamQACaseV2)
            for value in cases
        ):
            raise StageEditContextualSourcePairV2Error(
                "beam QA needs at least two typed full-decode cases"
            )
        for field, values in (
            ("case ids", tuple(value.case_id for value in cases)),
            ("song ids", tuple(value.song_id for value in cases)),
            ("request hashes", tuple(value.request_sha256 for value in cases)),
            (
                "decode report hashes",
                tuple(value.decode_report_sha256 for value in cases),
            ),
        ):
            if len(values) != len(set(values)):
                raise StageEditContextualSourcePairV2Error(
                    f"beam QA {field} must be song-disjoint and unique"
                )
        object.__setattr__(self, "cases", cases)
        expected = _canonical_sha256(self.unsigned_dict())
        if self.qa_report_sha256 is None:
            object.__setattr__(self, "qa_report_sha256", expected)
        else:
            digest = _sha256(self.qa_report_sha256, "qa_report_sha256")
            object.__setattr__(self, "qa_report_sha256", digest)
            if digest != expected:
                raise StageEditContextualSourcePairV2Error(
                    "beam QA report digest does not match measured cases"
                )

    @property
    def authorized(self) -> bool:
        return all(value.authorized for value in self.cases)

    @property
    def all_target_profiles_met(self) -> bool:
        return all(value.target_profile_met for value in self.cases)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "checkpoint_sha256": self.checkpoint_sha256,
            "evaluation_corpus_sha256": self.evaluation_corpus_sha256,
            "thresholds": dict(STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BEAM_QA_THRESHOLDS),
            "cases": [value.to_dict() for value in self.cases],
            "song_disjoint_case_count": len(self.cases),
            "all_hard_gates_safe": all(value.hard_safe for value in self.cases),
            "production_authorized": self.authorized,
            "all_target_profiles_met": self.all_target_profiles_met,
            "blind_weight_times_four_compensation_permitted": False,
        }

    def to_dict(self) -> dict[str, Any]:
        assert self.qa_report_sha256 is not None
        return {**self.unsigned_dict(), "qa_report_sha256": self.qa_report_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageEditContextualBeamQAEvidenceV2":
        expected = {
            "contract",
            "checkpoint_sha256",
            "evaluation_corpus_sha256",
            "thresholds",
            "cases",
            "song_disjoint_case_count",
            "all_hard_gates_safe",
            "production_authorized",
            "all_target_profiles_met",
            "blind_weight_times_four_compensation_permitted",
            "qa_report_sha256",
        }
        _strict_keys(payload, expected, "beam QA measured report")
        if payload["thresholds"] != dict(
            STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BEAM_QA_THRESHOLDS
        ) or payload["blind_weight_times_four_compensation_permitted"] is not False:
            raise StageEditContextualSourcePairV2Error(
                "beam QA report thresholds or compensation policy mismatch"
            )
        raw_cases = payload["cases"]
        if not isinstance(raw_cases, list) or any(
            not isinstance(value, Mapping) for value in raw_cases
        ):
            raise StageEditContextualSourcePairV2Error(
                "beam QA report cases must be objects"
            )
        result = cls(
            contract=payload["contract"],
            checkpoint_sha256=payload["checkpoint_sha256"],
            evaluation_corpus_sha256=payload["evaluation_corpus_sha256"],
            cases=tuple(
                StageEditContextualBeamQACaseV2.from_dict(value)
                for value in raw_cases
            ),
            qa_report_sha256=payload["qa_report_sha256"],
        )
        if (
            payload["song_disjoint_case_count"] != len(result.cases)
            or payload["all_hard_gates_safe"]
            is not all(value.hard_safe for value in result.cases)
            or payload["production_authorized"] is not result.authorized
            or payload["all_target_profiles_met"]
            is not result.all_target_profiles_met
        ):
            raise StageEditContextualSourcePairV2Error(
                "beam QA aggregate gates do not match measured cases"
            )
        return result


def new_stage_edit_contextual_beam_qa_evidence_v2(
    *,
    checkpoint_sha256: str,
    evaluation_corpus_sha256: str,
    cases: Sequence[StageEditContextualBeamQACaseV2],
) -> StageEditContextualBeamQAEvidenceV2:
    return StageEditContextualBeamQAEvidenceV2(
        checkpoint_sha256=checkpoint_sha256,
        evaluation_corpus_sha256=evaluation_corpus_sha256,
        cases=tuple(cases),
    )


def _beam_qa_evidence_from_verified_envelopes_v2(
    *,
    checkpoint: StageEditContextualSourcePairCheckpointV2,
    evaluation_corpus_sha256: str,
    verified_cases: Sequence[object],
) -> StageEditContextualBeamQAEvidenceV2:
    # The adapter is intentionally a leaf that imports this DTO module.  Keep
    # this import lazy so the production activation boundary can demand its
    # process-local verification seal without creating a module import cycle.
    from .stage_edit_contextual_beam_qa_adapter_v2 import (
        require_verified_stage_edit_contextual_beam_qa_case_v2,
    )

    if isinstance(verified_cases, (str, bytes)) or not isinstance(
        verified_cases, Sequence
    ):
        raise StageEditContextualSourcePairV2Error(
            "beam QA activation cases must be a sequence of verified envelopes"
        )
    if not isinstance(checkpoint, StageEditContextualSourcePairCheckpointV2):
        raise StageEditContextualSourcePairV2Error(
            "beam QA activation checkpoint must be typed"
        )
    cases = tuple(
        require_verified_stage_edit_contextual_beam_qa_case_v2(
            value,
            expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
            expected_corpus_sha256=checkpoint.corpus_sha256,
            expected_scorer_fingerprint=checkpoint.scorer_fingerprint,
            expected_runtime_equivalence_sha256=(
                checkpoint.runtime_equivalence_sha256
            ),
            expected_token_schema_fingerprint=(
                checkpoint.token_schema_fingerprint
            ),
        )
        for value in verified_cases
    )
    return new_stage_edit_contextual_beam_qa_evidence_v2(
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        evaluation_corpus_sha256=evaluation_corpus_sha256,
        cases=cases,
    )


@dataclass(frozen=True, slots=True)
class StageEditContextualSourcePairProductionReportV2:
    schema_version: int
    contract: str
    report_id: str
    created_at: str
    authority: str
    checkpoint: StageEditContextualSourcePairCheckpointV2
    cross_validation: StageEditContextualCrossValidationResultV2
    final_full_corpus_training: StageEditContextualTrainingResultV2
    beam_level_qa: StageEditContextualBeamQAEvidenceV2
    report_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_SCHEMA_VERSION:
            raise StageEditContextualSourcePairV2Error("production report schema mismatch")
        if self.contract != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_CONTRACT:
            raise StageEditContextualSourcePairV2Error("production report contract mismatch")
        if self.authority != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_AUTHORITY:
            raise StageEditContextualSourcePairV2Error("production report authority mismatch")
        object.__setattr__(self, "report_id", _name(self.report_id, "report_id"))
        created = _name(self.created_at, "created_at")
        try:
            timestamp = datetime.fromisoformat(created)
        except ValueError as exc:
            raise StageEditContextualSourcePairV2Error("report created_at must be ISO-8601") from exc
        if timestamp.tzinfo is None:
            raise StageEditContextualSourcePairV2Error("report created_at must include timezone")
        object.__setattr__(self, "created_at", created)
        if not isinstance(self.checkpoint, StageEditContextualSourcePairCheckpointV2):
            raise StageEditContextualSourcePairV2Error("production report checkpoint is invalid")
        if not isinstance(
            self.cross_validation, StageEditContextualCrossValidationResultV2
        ):
            raise StageEditContextualSourcePairV2Error(
                "production report LOSO evidence must be typed"
            )
        cross_validation = self.cross_validation
        if (
            not cross_validation.authorized
            or cross_validation.corpus_sha256 != self.checkpoint.corpus_sha256
            or cross_validation.source_data_id != self.checkpoint.source_data_id
            or cross_validation.feature_snapshot_bindings_sha256
            != self.checkpoint.feature_snapshot_bindings_sha256
            or cross_validation.clipped_observed_value_count
            != self.checkpoint.clipped_observed_value_count
            or cross_validation.token_schema_fingerprint
            != self.checkpoint.token_schema_fingerprint
        ):
            raise StageEditContextualSourcePairV2Error("production report LOSO evidence is invalid")
        if not isinstance(
            self.final_full_corpus_training, StageEditContextualTrainingResultV2
        ):
            raise StageEditContextualSourcePairV2Error(
                "production report final training must be typed"
            )
        final_training = self.final_full_corpus_training
        if (
            final_training.corpus_sha256 != self.checkpoint.corpus_sha256
            or final_training.source_data_id != self.checkpoint.source_data_id
            or final_training.feature_snapshot_bindings_sha256
            != self.checkpoint.feature_snapshot_bindings_sha256
            or final_training.clipped_observed_value_count
            != self.checkpoint.clipped_observed_value_count
            or final_training.scorer.fingerprint
            != self.checkpoint.scorer_fingerprint
            or final_training.runtime_equivalence_sha256
            != self.checkpoint.runtime_equivalence_sha256
            or final_training.training_song_ids
            != self.checkpoint.training_song_ids
            or final_training.training_reference_ids
            != self.checkpoint.training_reference_ids
            or final_training.training_group_ids
            != self.checkpoint.training_group_ids
            or final_training.scorer.token_schema_fingerprint
            != self.checkpoint.token_schema_fingerprint
        ):
            raise StageEditContextualSourcePairV2Error("production report final training lineage mismatch")
        if not isinstance(self.beam_level_qa, StageEditContextualBeamQAEvidenceV2):
            raise StageEditContextualSourcePairV2Error("production report beam QA is invalid")
        if self.beam_level_qa.checkpoint_sha256 != self.checkpoint.checkpoint_sha256:
            raise StageEditContextualSourcePairV2Error("beam QA is not bound to checkpoint")
        if not self.beam_level_qa.authorized:
            raise StageEditContextualSourcePairV2Error(
                "production report requires cadence, occupancy, and highlight beam QA"
            )
        digest = _sha256(self.report_sha256, "report_sha256")
        object.__setattr__(self, "report_sha256", digest)
        if digest != _canonical_sha256(self.unsigned_dict()):
            raise StageEditContextualSourcePairV2Error("production report digest mismatch")

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "report_id": self.report_id,
            "created_at": self.created_at,
            "authority": self.authority,
            "checkpoint": self.checkpoint.to_dict(),
            "cross_validation": self.cross_validation.to_dict(),
            "final_full_corpus_training": self.final_full_corpus_training.to_dict(),
            "beam_level_qa": self.beam_level_qa.to_dict(),
            "causal_leakage_contract": {
                "runtime_evidence_contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RUNTIME_EVIDENCE_CONTRACT,
                "recent_edit_features_consumed": False,
                "model_logits_consumed": False,
                "reference_match_input_evidence_consumed": False,
                "current_token_music_section_style_consumed": True,
                "current_token_source_candidate_and_embedding_evidence_consumed": True,
            },
            "production_authorization": {
                "authorized": True,
                "all_fold_and_aggregate_loso_gate_passed": True,
                "full_corpus_fit_after_loso_gate": True,
                "beam_level_cadence_occupancy_highlight_qa_passed": True,
                "absolute_switch_cost_changed_by_affine_transform": True,
                "blind_continuity_weight_times_four_compensation_permitted": False,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "report_sha256": self.report_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageEditContextualSourcePairProductionReportV2":
        expected = {
            "schema_version",
            "contract",
            "report_id",
            "created_at",
            "authority",
            "checkpoint",
            "cross_validation",
            "final_full_corpus_training",
            "beam_level_qa",
            "causal_leakage_contract",
            "production_authorization",
            "report_sha256",
        }
        _strict_keys(payload, expected, "contextual v2 production report")
        leakage = payload.get("causal_leakage_contract")
        authorization = payload.get("production_authorization")
        if not isinstance(leakage, Mapping) or dict(leakage) != {
            "runtime_evidence_contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RUNTIME_EVIDENCE_CONTRACT,
            "recent_edit_features_consumed": False,
            "model_logits_consumed": False,
            "reference_match_input_evidence_consumed": False,
            "current_token_music_section_style_consumed": True,
            "current_token_source_candidate_and_embedding_evidence_consumed": True,
        }:
            raise StageEditContextualSourcePairV2Error("production report leakage contract mismatch")
        if not isinstance(authorization, Mapping) or dict(authorization) != {
            "authorized": True,
            "all_fold_and_aggregate_loso_gate_passed": True,
            "full_corpus_fit_after_loso_gate": True,
            "beam_level_cadence_occupancy_highlight_qa_passed": True,
            "absolute_switch_cost_changed_by_affine_transform": True,
            "blind_continuity_weight_times_four_compensation_permitted": False,
        }:
            raise StageEditContextualSourcePairV2Error("production report authorization mismatch")
        for field in ("checkpoint", "cross_validation", "final_full_corpus_training", "beam_level_qa"):
            if not isinstance(payload.get(field), Mapping):
                raise StageEditContextualSourcePairV2Error(f"production report {field} must be an object")
        return cls(
            schema_version=_integer(payload["schema_version"], "schema_version", minimum=1),
            contract=payload["contract"],
            report_id=payload["report_id"],
            created_at=payload["created_at"],
            authority=payload["authority"],
            checkpoint=StageEditContextualSourcePairCheckpointV2.from_dict(payload["checkpoint"]),
            cross_validation=StageEditContextualCrossValidationResultV2.from_dict(
                payload["cross_validation"]
            ),
            final_full_corpus_training=StageEditContextualTrainingResultV2.from_dict(
                payload["final_full_corpus_training"]
            ),
            beam_level_qa=StageEditContextualBeamQAEvidenceV2.from_dict(payload["beam_level_qa"]),
            report_sha256=payload["report_sha256"],
        )


def new_stage_edit_contextual_source_pair_production_report_v2(
    *,
    report_id: str,
    corpus: StageEditContextualTrainingCorpusV2,
    cross_validation: StageEditContextualCrossValidationResultV2,
    final_training: StageEditContextualTrainingResultV2,
    checkpoint: StageEditContextualSourcePairCheckpointV2,
    beam_qa_evaluation_corpus_sha256: str,
    verified_beam_qa_cases: Sequence[object],
    created_at: str | None = None,
) -> StageEditContextualSourcePairProductionReportV2:
    if not isinstance(corpus, StageEditContextualTrainingCorpusV2):
        raise StageEditContextualSourcePairV2Error(
            "production report requires the typed full training corpus"
        )
    if not isinstance(cross_validation, StageEditContextualCrossValidationResultV2) or not cross_validation.authorized:
        raise StageEditContextualSourcePairV2Error("denied LOSO result cannot form production report")
    if not isinstance(final_training, StageEditContextualTrainingResultV2):
        raise StageEditContextualSourcePairV2Error("final training result is invalid")
    if not isinstance(checkpoint, StageEditContextualSourcePairCheckpointV2):
        raise StageEditContextualSourcePairV2Error("checkpoint is invalid")
    if final_training.corpus_sha256 != cross_validation.corpus_sha256:
        raise StageEditContextualSourcePairV2Error("full-corpus fit is not bound to LOSO corpus")
    replayed_cross_validation = cross_validate_stage_edit_contextual_source_pair_v2(
        corpus, config=final_training.config
    )
    if replayed_cross_validation != cross_validation:
        raise StageEditContextualSourcePairV2Error(
            "LOSO evidence differs from deterministic full-corpus replay"
        )
    replayed_final_training = train_stage_edit_contextual_source_pair_v2(
        corpus, config=final_training.config
    )
    if replayed_final_training != final_training:
        raise StageEditContextualSourcePairV2Error(
            "final training differs from deterministic full-corpus replay"
        )
    beam_level_qa = _beam_qa_evidence_from_verified_envelopes_v2(
        checkpoint=checkpoint,
        evaluation_corpus_sha256=_sha256(
            beam_qa_evaluation_corpus_sha256,
            "beam_qa_evaluation_corpus_sha256",
        ),
        verified_cases=verified_beam_qa_cases,
    )
    unsigned = {
        "schema_version": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_SCHEMA_VERSION,
        "contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_CONTRACT,
        "report_id": _name(report_id, "report_id"),
        "created_at": datetime.now(UTC).isoformat() if created_at is None else _name(created_at, "created_at"),
        "authority": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_AUTHORITY,
        "checkpoint": checkpoint.to_dict(),
        "cross_validation": cross_validation.to_dict(),
        "final_full_corpus_training": final_training.to_dict(),
        "beam_level_qa": beam_level_qa.to_dict(),
        "causal_leakage_contract": {
            "runtime_evidence_contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RUNTIME_EVIDENCE_CONTRACT,
            "recent_edit_features_consumed": False,
            "model_logits_consumed": False,
            "reference_match_input_evidence_consumed": False,
            "current_token_music_section_style_consumed": True,
            "current_token_source_candidate_and_embedding_evidence_consumed": True,
        },
        "production_authorization": {
            "authorized": True,
            "all_fold_and_aggregate_loso_gate_passed": True,
            "full_corpus_fit_after_loso_gate": True,
            "beam_level_cadence_occupancy_highlight_qa_passed": True,
            "absolute_switch_cost_changed_by_affine_transform": True,
            "blind_continuity_weight_times_four_compensation_permitted": False,
        },
    }
    return StageEditContextualSourcePairProductionReportV2(
        schema_version=unsigned["schema_version"],
        contract=unsigned["contract"],
        report_id=unsigned["report_id"],
        created_at=unsigned["created_at"],
        authority=unsigned["authority"],
        checkpoint=checkpoint,
        cross_validation=cross_validation,
        final_full_corpus_training=final_training,
        beam_level_qa=beam_level_qa,
        report_sha256=_canonical_sha256(unsigned),
    )


def _absolute_lexical_path(value: os.PathLike[str] | str, field: str) -> Path:
    """Return an absolute normalized spelling without resolving any symlink."""

    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise StageEditContextualSourcePairV2ArtifactError(
            f"{field} must be a filesystem path"
        ) from exc
    if not isinstance(raw, str) or not raw or not os.path.isabs(raw):
        raise StageEditContextualSourcePairV2ArtifactError(
            f"{field} must be an absolute lexical path"
        )
    # Never expand '~', consult cwd, call realpath/resolve, or silently accept
    # traversal spellings.  The caller must name the exact artifact directory.
    lexical_parts = tuple(part for part in raw.split(os.sep) if part)
    if any(part in {".", ".."} for part in lexical_parts):
        raise StageEditContextualSourcePairV2ArtifactError(
            f"{field} cannot contain dot traversal components"
        )
    normalized = os.path.normpath(raw)
    if not os.path.isabs(normalized):
        raise StageEditContextualSourcePairV2ArtifactError(
            f"{field} lost its absolute lexical identity"
        )
    return Path(normalized)


def _open_verified_lexical_directory_fd(
    value: os.PathLike[str] | str,
    *,
    field: str,
    create: bool,
) -> tuple[Path, int]:
    """Pin an absolute directory by walking from root with openat/no-follow."""

    directory = _absolute_lexical_path(value, field)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        current_fd = os.open(directory.anchor, directory_flags)
    except OSError as exc:
        raise StageEditContextualSourcePairV2ArtifactError(
            f"{field} root cannot be opened safely: {exc}"
        ) from exc
    try:
        for component in directory.parts[1:]:
            try:
                status = os.stat(
                    component, dir_fd=current_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                if not create:
                    raise StageEditContextualSourcePairV2ArtifactError(
                        f"{field} does not exist at component: {component}"
                    ) from None
                try:
                    os.mkdir(component, 0o700, dir_fd=current_fd)
                    status = os.stat(
                        component, dir_fd=current_fd, follow_symlinks=False
                    )
                except OSError as exc:
                    raise StageEditContextualSourcePairV2ArtifactError(
                        f"{field} cannot be created safely: {exc}"
                    ) from exc
            except OSError as exc:
                raise StageEditContextualSourcePairV2ArtifactError(
                    f"{field} cannot be inspected safely: {exc}"
                ) from exc
            if stat.S_ISLNK(status.st_mode):
                raise StageEditContextualSourcePairV2ArtifactError(
                    f"{field} contains a symlink component: {component}"
                )
            if not stat.S_ISDIR(status.st_mode):
                raise StageEditContextualSourcePairV2ArtifactError(
                    f"{field} component is not a directory: {component}"
                )
            try:
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            except OSError as exc:
                raise StageEditContextualSourcePairV2ArtifactError(
                    f"{field} component cannot be opened safely: {exc}"
                ) from exc
            opened = os.fstat(next_fd)
            if (opened.st_dev, opened.st_ino) != (status.st_dev, status.st_ino):
                os.close(next_fd)
                raise StageEditContextualSourcePairV2ArtifactError(
                    f"{field} component identity changed while opening"
                )
            os.close(current_fd)
            current_fd = next_fd
        return directory, current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _verified_lexical_directory(
    value: os.PathLike[str] | str,
    *,
    field: str,
    create: bool,
) -> Path:
    directory, descriptor = _open_verified_lexical_directory_fd(
        value, field=field, create=create
    )
    os.close(descriptor)
    return directory


def _stable_file_identity(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


_STAGE_EDIT_CONTEXTUAL_PUBLISH_JOURNAL_CONTRACT = (
    "contextual_v2_immutable_artifact_publish_journal_v1"
)
_HARDLINK_UNSUPPORTED_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "ENOSYS", None),
        getattr(errno, "EINVAL", None),
        getattr(errno, "EPERM", None),
    )
    if value is not None
)
_DIRECTORY_FSYNC_UNSUPPORTED_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "EINVAL", None),
    )
    if value is not None
)


def _publish_journal_name(target_name: str) -> str:
    if (
        not target_name
        or target_name in {".", ".."}
        or os.sep in target_name
        or (os.altsep is not None and os.altsep in target_name)
    ):
        raise StageEditContextualSourcePairV2ArtifactError(
            "publish journal target name is invalid"
        )
    return f".{target_name}.publish-journal-v1.json"


def _fsync_directory_fd(directory_fd: int) -> None:
    try:
        os.fsync(directory_fd)
    except OSError as exc:
        # macOS ExFAT reports EINVAL/ENOTSUP for directory fsync even though
        # regular-file fsync is supported.  All other failures stay fatal.
        if exc.errno not in _DIRECTORY_FSYNC_UNSUPPORTED_ERRNOS:
            raise


def _write_all(descriptor: int, encoded: bytes) -> None:
    offset = 0
    while offset < len(encoded):
        written = os.write(descriptor, encoded[offset:])
        if written <= 0:
            raise OSError(errno.EIO, "short artifact write")
        offset += written


def _read_regular_file_at(
    directory_fd: int,
    name: str,
    *,
    field: str,
) -> tuple[bytes, os.stat_result]:
    if (
        not name
        or name in {".", ".."}
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
    ):
        raise StageEditContextualSourcePairV2Error(f"{field} name is invalid")
    before_path = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise StageEditContextualSourcePairV2Error(
            f"{field} must be a regular non-symlink file"
        )
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before_fd = os.fstat(descriptor)
        if not stat.S_ISREG(before_fd.st_mode) or (
            before_fd.st_dev,
            before_fd.st_ino,
        ) != (before_path.st_dev, before_path.st_ino):
            raise StageEditContextualSourcePairV2Error(
                f"{field} identity changed before read"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        encoded = b"".join(chunks)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        _stable_file_identity(before_path) != _stable_file_identity(before_fd)
        or _stable_file_identity(before_fd) != _stable_file_identity(after_fd)
        or _stable_file_identity(after_fd) != _stable_file_identity(after_path)
        or len(encoded) != before_fd.st_size
    ):
        raise StageEditContextualSourcePairV2Error(
            f"{field} changed during stable byte read"
        )
    return encoded, after_fd


def _journal_payload(
    *,
    target_name: str,
    expected_file_sha256: str,
    temporary_name: str,
    temporary_status: os.stat_result,
    phase: str,
    final_status: os.stat_result | None,
) -> dict[str, Any]:
    if phase not in {"prepared", "copying"}:
        raise StageEditContextualSourcePairV2ArtifactError(
            "publish journal phase is invalid"
        )
    return {
        "contract": _STAGE_EDIT_CONTEXTUAL_PUBLISH_JOURNAL_CONTRACT,
        "target_name": target_name,
        "expected_file_sha256": expected_file_sha256,
        "temporary_name": temporary_name,
        "temporary_device": temporary_status.st_dev,
        "temporary_inode": temporary_status.st_ino,
        "temporary_size": temporary_status.st_size,
        "phase": phase,
        "final_device": None if final_status is None else final_status.st_dev,
        "final_inode": None if final_status is None else final_status.st_ino,
    }


def _parse_publish_journal(
    payload: Mapping[str, Any], *, target_name: str
) -> dict[str, Any]:
    expected = {
        "contract",
        "target_name",
        "expected_file_sha256",
        "temporary_name",
        "temporary_device",
        "temporary_inode",
        "temporary_size",
        "phase",
        "final_device",
        "final_inode",
    }
    _strict_keys(payload, expected, "artifact publish journal")
    if (
        payload.get("contract") != _STAGE_EDIT_CONTEXTUAL_PUBLISH_JOURNAL_CONTRACT
        or payload.get("target_name") != target_name
    ):
        raise StageEditContextualSourcePairV2ArtifactError(
            "artifact publish journal identity mismatch"
        )
    temporary_name = payload.get("temporary_name")
    expected_prefix = f".{target_name}.tmp-"
    if (
        not isinstance(temporary_name, str)
        or not temporary_name.startswith(expected_prefix)
        or os.sep in temporary_name
        or (os.altsep is not None and os.altsep in temporary_name)
    ):
        raise StageEditContextualSourcePairV2ArtifactError(
            "artifact publish journal temporary identity is invalid"
        )
    phase = payload.get("phase")
    if phase not in {"prepared", "copying"}:
        raise StageEditContextualSourcePairV2ArtifactError(
            "artifact publish journal phase is invalid"
        )
    for field in ("temporary_device", "temporary_inode", "temporary_size"):
        _integer(payload.get(field), field)
    if phase == "prepared":
        if payload.get("final_device") is not None or payload.get("final_inode") is not None:
            raise StageEditContextualSourcePairV2ArtifactError(
                "prepared publish journal cannot claim a final inode"
            )
    else:
        _integer(payload.get("final_device"), "final_device")
        _integer(payload.get("final_inode"), "final_inode")
    _sha256(payload.get("expected_file_sha256"), "expected_file_sha256")
    return dict(payload)


def _read_publish_journal(
    directory_fd: int, target_name: str
) -> dict[str, Any] | None:
    journal_name = _publish_journal_name(target_name)
    try:
        encoded, _status = _read_regular_file_at(
            directory_fd, journal_name, field="artifact publish journal"
        )
    except FileNotFoundError:
        return None
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StageEditContextualSourcePairV2ArtifactError(
            f"artifact publish journal is malformed: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise StageEditContextualSourcePairV2ArtifactError(
            "artifact publish journal root must be an object"
        )
    try:
        return _parse_publish_journal(payload, target_name=target_name)
    except StageEditContextualSourcePairV2ArtifactError:
        raise
    except StageEditContextualSourcePairV2Error as exc:
        raise StageEditContextualSourcePairV2ArtifactError(str(exc)) from exc


def _assert_no_active_publish_journal(
    directory_fd: int, target_name: str
) -> None:
    journal_name = _publish_journal_name(target_name)
    try:
        os.stat(journal_name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise StageEditContextualSourcePairV2ArtifactError(
        f"artifact has an active publish journal: {target_name}"
    )


def _remove_if_exact_identity(
    directory_fd: int,
    name: str,
    *,
    device: int,
    inode: int,
) -> bool:
    try:
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise StageEditContextualSourcePairV2ArtifactError(
            "owned publish path changed to a non-regular file"
        )
    if (observed.st_dev, observed.st_ino) != (device, inode):
        raise StageEditContextualSourcePairV2ArtifactError(
            "owned publish path inode changed; refusing cleanup"
        )
    os.unlink(name, dir_fd=directory_fd)
    return True


def _recover_matching_publish_journal(
    directory_fd: int,
    *,
    target_name: str,
    expected_file_sha256: str,
) -> bool:
    """Recover a prior matching transaction; return true if final is complete."""

    journal = _read_publish_journal(directory_fd, target_name)
    if journal is None:
        return False
    if journal["expected_file_sha256"] != expected_file_sha256:
        raise StageEditContextualSourcePairV2ArtifactError(
            "active publish journal belongs to different artifact bytes"
        )
    temporary_name = journal["temporary_name"]
    temporary_identity = (
        journal["temporary_device"],
        journal["temporary_inode"],
    )
    temporary_exists = False
    try:
        temporary_bytes, temporary_status = _read_regular_file_at(
            directory_fd, temporary_name, field="publish temporary"
        )
        temporary_exists = True
    except FileNotFoundError:
        temporary_bytes = b""
        temporary_status = None
    if temporary_exists and (
        (temporary_status.st_dev, temporary_status.st_ino) != temporary_identity
        or temporary_status.st_size != journal["temporary_size"]
        or hashlib.sha256(temporary_bytes).hexdigest() != expected_file_sha256
    ):
        raise StageEditContextualSourcePairV2ArtifactError(
            "publish temporary no longer matches journal identity"
        )

    final_exists = False
    try:
        final_bytes, final_status = _read_regular_file_at(
            directory_fd, target_name, field="publish final"
        )
        final_exists = True
    except FileNotFoundError:
        final_bytes = b""
        final_status = None
    if final_exists and hashlib.sha256(final_bytes).hexdigest() == expected_file_sha256:
        if temporary_exists:
            _remove_if_exact_identity(
                directory_fd,
                temporary_name,
                device=temporary_identity[0],
                inode=temporary_identity[1],
            )
        os.unlink(_publish_journal_name(target_name), dir_fd=directory_fd)
        _fsync_directory_fd(directory_fd)
        return True
    if final_exists:
        if journal["phase"] != "copying" or (
            final_status.st_dev,
            final_status.st_ino,
        ) != (journal["final_device"], journal["final_inode"]):
            raise StageEditContextualSourcePairV2ArtifactError(
                "existing final is not the journal-owned partial artifact"
            )
        _remove_if_exact_identity(
            directory_fd,
            target_name,
            device=journal["final_device"],
            inode=journal["final_inode"],
        )
    if not temporary_exists:
        # A prior owner-safe cleanup may have removed both the partial final
        # and its temporary before crashing on journal unlink.  With no target
        # path left to overwrite, clearing the matching lock and restarting is
        # safe and makes the transaction retryable.
        os.unlink(_publish_journal_name(target_name), dir_fd=directory_fd)
        _fsync_directory_fd(directory_fd)
        return False
    _remove_if_exact_identity(
        directory_fd,
        temporary_name,
        device=temporary_identity[0],
        inode=temporary_identity[1],
    )
    os.unlink(_publish_journal_name(target_name), dir_fd=directory_fd)
    _fsync_directory_fd(directory_fd)
    return False


def _copy_temporary_to_final(
    temporary_descriptor: int, final_descriptor: int, expected_size: int
) -> None:
    os.lseek(temporary_descriptor, 0, os.SEEK_SET)
    os.lseek(final_descriptor, 0, os.SEEK_SET)
    copied = 0
    while copied < expected_size:
        chunk = os.read(
            temporary_descriptor, min(1024 * 1024, expected_size - copied)
        )
        if not chunk:
            raise OSError(errno.EIO, "publish temporary ended before expected size")
        _write_all(final_descriptor, chunk)
        copied += len(chunk)
    if os.read(temporary_descriptor, 1):
        raise OSError(errno.EIO, "publish temporary grew during copy")


def _publish_copy_with_journal(
    *,
    directory_fd: int,
    target_name: str,
    temporary_name: str,
    temporary_descriptor: int,
    temporary_status: os.stat_result,
    expected_file_sha256: str,
) -> None:
    journal_name = _publish_journal_name(target_name)
    prepared = _journal_payload(
        target_name=target_name,
        expected_file_sha256=expected_file_sha256,
        temporary_name=temporary_name,
        temporary_status=temporary_status,
        phase="prepared",
        final_status=None,
    )
    journal_descriptor = os.open(
        journal_name,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    journal_created = True
    final_descriptor: int | None = None
    final_status: os.stat_result | None = None
    final_created = False
    committed = False
    partial_cleanup_safe = True
    final_verified_complete = False
    try:
        _write_all(journal_descriptor, _canonical_json_bytes(prepared))
        os.fsync(journal_descriptor)
        _fsync_directory_fd(directory_fd)
        try:
            final_descriptor = os.open(
                target_name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite immutable artifact: {target_name}"
            ) from exc
        final_created = True
        # ExFAT reports a transient inode for a newly created zero-length file
        # and assigns its stable inode on the first allocation.  Allocate the
        # complete length before sealing ownership into the journal; the copy
        # then overwrites from offset zero and cannot grow past this identity.
        os.ftruncate(final_descriptor, temporary_status.st_size)
        os.fsync(final_descriptor)
        final_status = os.fstat(final_descriptor)
        final_path_status = os.stat(
            target_name, dir_fd=directory_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(final_status.st_mode)
            or (final_path_status.st_dev, final_path_status.st_ino)
            != (final_status.st_dev, final_status.st_ino)
            or final_status.st_size != temporary_status.st_size
        ):
            raise StageEditContextualSourcePairV2ArtifactError(
                "journal publish could not establish stable final identity"
            )
        copying = _journal_payload(
            target_name=target_name,
            expected_file_sha256=expected_file_sha256,
            temporary_name=temporary_name,
            temporary_status=temporary_status,
            phase="copying",
            final_status=final_status,
        )
        os.lseek(journal_descriptor, 0, os.SEEK_SET)
        os.ftruncate(journal_descriptor, 0)
        _write_all(journal_descriptor, _canonical_json_bytes(copying))
        os.fsync(journal_descriptor)
        _copy_temporary_to_final(
            temporary_descriptor, final_descriptor, temporary_status.st_size
        )
        os.fsync(final_descriptor)
        os.lseek(final_descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(final_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        final_bytes = b"".join(chunks)
        after_fd = os.fstat(final_descriptor)
        after_path = os.stat(
            target_name, dir_fd=directory_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(after_fd.st_mode)
            or (after_fd.st_dev, after_fd.st_ino)
            != (final_status.st_dev, final_status.st_ino)
            or (after_path.st_dev, after_path.st_ino)
            != (final_status.st_dev, final_status.st_ino)
            or after_fd.st_size != temporary_status.st_size
            or len(final_bytes) != temporary_status.st_size
            or hashlib.sha256(final_bytes).hexdigest() != expected_file_sha256
        ):
            raise StageEditContextualSourcePairV2ArtifactError(
                "journal publish final failed same-fd digest/inode verification"
            )
        final_verified_complete = True
        os.unlink(temporary_name, dir_fd=directory_fd)
        os.unlink(journal_name, dir_fd=directory_fd)
        journal_created = False
        _fsync_directory_fd(directory_fd)
        committed = True
    finally:
        if final_descriptor is not None:
            os.close(final_descriptor)
        os.close(journal_descriptor)
        if (
            not committed
            and final_created
            and final_status is not None
            and not (final_verified_complete and journal_created)
        ):
            try:
                _remove_if_exact_identity(
                    directory_fd,
                    target_name,
                    device=final_status.st_dev,
                    inode=final_status.st_ino,
                )
            except (OSError, StageEditContextualSourcePairV2ArtifactError):
                # Preserve the journal if ownership-safe cleanup cannot finish;
                # every loader will then fail closed.
                partial_cleanup_safe = False
        if (
            not committed
            and journal_created
            and partial_cleanup_safe
            and not final_verified_complete
        ):
            try:
                os.unlink(journal_name, dir_fd=directory_fd)
                journal_created = False
            except FileNotFoundError:
                journal_created = False
            except OSError:
                pass
        if not committed and not journal_created and partial_cleanup_safe:
            _fsync_directory_fd(directory_fd)


def _write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> Path:
    encoded = _canonical_json_bytes(payload, pretty=True)
    expected_file_sha256 = hashlib.sha256(encoded).hexdigest()
    artifact = _absolute_lexical_path(path, "artifact path")
    _parent, directory_fd = _open_verified_lexical_directory_fd(
        artifact.parent,
        field="artifact parent directory",
        create=True,
    )
    temporary_name = (
        f".{artifact.name}.tmp-{os.getpid()}-{secrets.token_hex(12)}"
    )
    temporary_descriptor: int | None = None
    temporary_status: os.stat_result | None = None
    temporary_created = False
    published = False
    hardlink_final_status: os.stat_result | None = None
    try:
        if _recover_matching_publish_journal(
            directory_fd,
            target_name=artifact.name,
            expected_file_sha256=expected_file_sha256,
        ):
            return artifact
        try:
            temporary_descriptor = os.open(
                temporary_name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError as exc:
            raise StageEditContextualSourcePairV2ArtifactError(
                "immutable artifact temporary-name collision"
            ) from exc
        temporary_created = True
        _write_all(temporary_descriptor, encoded)
        os.fsync(temporary_descriptor)
        temporary_status = os.fstat(temporary_descriptor)
        if (
            not stat.S_ISREG(temporary_status.st_mode)
            or temporary_status.st_size != len(encoded)
        ):
            raise StageEditContextualSourcePairV2ArtifactError(
                "immutable artifact temporary did not complete"
            )
        try:
            os.link(
                temporary_name,
                artifact.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite immutable artifact: {artifact}"
            ) from exc
        except OSError as exc:
            if exc.errno not in _HARDLINK_UNSUPPORTED_ERRNOS:
                raise
            _publish_copy_with_journal(
                directory_fd=directory_fd,
                target_name=artifact.name,
                temporary_name=temporary_name,
                temporary_descriptor=temporary_descriptor,
                temporary_status=temporary_status,
                expected_file_sha256=expected_file_sha256,
            )
            temporary_created = False
            published = True
        else:
            final_status = os.stat(
                artifact.name, dir_fd=directory_fd, follow_symlinks=False
            )
            hardlink_final_status = final_status
            os.lseek(temporary_descriptor, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(temporary_descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after_temp = os.fstat(temporary_descriptor)
            if (
                (final_status.st_dev, final_status.st_ino)
                != (temporary_status.st_dev, temporary_status.st_ino)
                or not stat.S_ISREG(after_temp.st_mode)
                or (after_temp.st_dev, after_temp.st_ino, after_temp.st_size)
                != (
                    temporary_status.st_dev,
                    temporary_status.st_ino,
                    temporary_status.st_size,
                )
                or hashlib.sha256(b"".join(chunks)).hexdigest()
                != expected_file_sha256
            ):
                try:
                    _remove_if_exact_identity(
                        directory_fd,
                        artifact.name,
                        device=final_status.st_dev,
                        inode=final_status.st_ino,
                    )
                finally:
                    raise StageEditContextualSourcePairV2ArtifactError(
                        "hard-link publish failed same-fd digest/inode verification"
                    )
            os.unlink(temporary_name, dir_fd=directory_fd)
            temporary_created = False
            _fsync_directory_fd(directory_fd)
            published = True
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if not published and hardlink_final_status is not None:
            try:
                _remove_if_exact_identity(
                    directory_fd,
                    artifact.name,
                    device=hardlink_final_status.st_dev,
                    inode=hardlink_final_status.st_ino,
                )
            except (OSError, StageEditContextualSourcePairV2ArtifactError):
                pass
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        if not published:
            try:
                _fsync_directory_fd(directory_fd)
            except OSError:
                pass
        os.close(directory_fd)
    return artifact


def _load_json_regular_file_with_sha256(
    path: Path,
) -> tuple[Mapping[str, Any], str]:
    artifact = _absolute_lexical_path(path, "artifact path")
    parent, directory_fd = _open_verified_lexical_directory_fd(
        artifact.parent,
        field="artifact parent directory",
        create=False,
    )
    try:
        try:
            _assert_no_active_publish_journal(directory_fd, artifact.name)
            before_path = os.lstat(artifact.name, dir_fd=directory_fd)
            if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(
                before_path.st_mode
            ):
                raise StageEditContextualSourcePairV2Error(
                    "artifact must be a regular non-symlink file"
                )
            descriptor = os.open(
                artifact.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            with os.fdopen(descriptor, "rb", closefd=True) as handle:
                before_fd = os.fstat(handle.fileno())
                if not stat.S_ISREG(before_fd.st_mode) or (
                    before_fd.st_dev,
                    before_fd.st_ino,
                ) != (before_path.st_dev, before_path.st_ino):
                    raise StageEditContextualSourcePairV2Error(
                        "artifact identity changed before read"
                    )
                encoded = handle.read()
                after_fd = os.fstat(handle.fileno())
            after_path = os.lstat(artifact.name, dir_fd=directory_fd)
            _assert_no_active_publish_journal(directory_fd, artifact.name)
            if (
                _stable_file_identity(before_path)
                != _stable_file_identity(before_fd)
                or _stable_file_identity(before_fd)
                != _stable_file_identity(after_fd)
                or _stable_file_identity(after_fd)
                != _stable_file_identity(after_path)
                or len(encoded) != before_fd.st_size
            ):
                raise StageEditContextualSourcePairV2Error(
                    "artifact changed during stable byte read"
                )
            payload = json.loads(encoded.decode("utf-8"))
            file_sha256 = hashlib.sha256(encoded).hexdigest()
        finally:
            os.close(directory_fd)
        if not isinstance(payload, Mapping):
            raise StageEditContextualSourcePairV2Error(
                "artifact root must be an object"
            )
        return payload, file_sha256
    except StageEditContextualSourcePairV2ArtifactError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        StageEditContextualSourcePairV2Error,
    ) as exc:
        raise StageEditContextualSourcePairV2ArtifactError(
            f"contextual v2 artifact is invalid: {exc}"
        ) from exc


def _load_json_regular_file(path: Path) -> Mapping[str, Any]:
    payload, _file_sha256 = _load_json_regular_file_with_sha256(path)
    return payload


def save_stage_edit_contextual_training_corpus_v2(
    directory: Path,
    corpus: StageEditContextualTrainingCorpusV2,
) -> Path:
    if not isinstance(corpus, StageEditContextualTrainingCorpusV2):
        raise StageEditContextualSourcePairV2Error(
            "contextual training corpus is invalid"
        )
    destination = _verified_lexical_directory(
        directory, field="training corpus directory", create=True
    )
    return _write_json_no_replace(
        destination / STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CORPUS_FILENAME,
        corpus.to_dict(),
    )


def _load_stage_edit_contextual_training_corpus_v2(
    directory: Path,
    *,
    expected_corpus_sha256: str | None = None,
    expected_corpus_file_sha256: str | None = None,
    expected_token_schema_fingerprint: str | None = None,
) -> tuple[StageEditContextualTrainingCorpusV2, str]:
    destination = _verified_lexical_directory(
        directory, field="training corpus directory", create=False
    )
    path = destination / STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CORPUS_FILENAME
    try:
        payload, file_sha256 = _load_json_regular_file_with_sha256(path)
        corpus = StageEditContextualTrainingCorpusV2.from_dict(payload)
        if expected_corpus_sha256 is not None and corpus.content_sha256 != _sha256(
            expected_corpus_sha256, "expected_corpus_sha256"
        ):
            raise StageEditContextualSourcePairV2Error(
                "expected corpus digest does not match artifact"
            )
        if (
            expected_corpus_file_sha256 is not None
            and file_sha256
            != _sha256(
                expected_corpus_file_sha256,
                "expected_corpus_file_sha256",
            )
        ):
            raise StageEditContextualSourcePairV2Error(
                "expected corpus file-byte digest does not match artifact"
            )
        if (
            expected_token_schema_fingerprint is not None
            and corpus.token_schema_fingerprint
            != _sha256(
                expected_token_schema_fingerprint,
                "expected_token_schema_fingerprint",
            )
        ):
            raise StageEditContextualSourcePairV2Error(
                "expected token schema fingerprint does not match corpus"
            )
    except StageEditContextualSourcePairV2ArtifactError:
        raise
    except (TypeError, ValueError, StageEditContextualSourcePairV2Error) as exc:
        raise StageEditContextualSourcePairV2ArtifactError(
            f"contextual v2 training corpus is invalid: {exc}"
        ) from exc
    return corpus, file_sha256


def inspect_stage_edit_contextual_training_corpus_v2(
    directory: Path,
    *,
    expected_corpus_sha256: str | None = None,
    expected_corpus_file_sha256: str | None = None,
    expected_token_schema_fingerprint: str | None = None,
) -> StageEditContextualTrainingCorpusV2:
    """Offline-only deep corpus inspection; never authorizes a scorer."""

    corpus, _file_sha256 = _load_stage_edit_contextual_training_corpus_v2(
        directory,
        expected_corpus_sha256=expected_corpus_sha256,
        expected_corpus_file_sha256=expected_corpus_file_sha256,
        expected_token_schema_fingerprint=expected_token_schema_fingerprint,
    )
    return corpus


def load_stage_edit_contextual_training_corpus_v2(
    directory: Path,
    *,
    expected_corpus_sha256: str,
    expected_corpus_file_sha256: str,
    expected_token_schema_fingerprint: str,
) -> tuple[StageEditContextualTrainingCorpusV2, str]:
    """Deep-load an exactly pinned corpus plus its observed file-byte SHA.

    This remains an offline data API.  Only the three-artifact production
    bundle loader can authorize a runtime scorer.
    """

    return _load_stage_edit_contextual_training_corpus_v2(
        directory,
        expected_corpus_sha256=_sha256(
            expected_corpus_sha256, "expected_corpus_sha256"
        ),
        expected_corpus_file_sha256=_sha256(
            expected_corpus_file_sha256,
            "expected_corpus_file_sha256",
        ),
        expected_token_schema_fingerprint=_sha256(
            expected_token_schema_fingerprint,
            "expected_token_schema_fingerprint",
        ),
    )


def save_stage_edit_contextual_source_pair_checkpoint_v2(
    directory: Path,
    checkpoint: StageEditContextualSourcePairCheckpointV2,
) -> Path:
    if not isinstance(checkpoint, StageEditContextualSourcePairCheckpointV2):
        raise StageEditContextualSourcePairV2Error("checkpoint is invalid")
    destination = _verified_lexical_directory(
        directory, field="checkpoint directory", create=True
    )
    return _write_json_no_replace(
        destination / STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FILENAME,
        checkpoint.to_dict(),
    )


def _load_stage_edit_contextual_source_pair_checkpoint_v2(
    directory: Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_file_sha256: str | None = None,
    expected_corpus_sha256: str | None = None,
    expected_token_schema_fingerprint: str | None = None,
) -> tuple[StageEditContextualSourcePairCheckpointV2, str]:
    destination = _verified_lexical_directory(
        directory, field="checkpoint directory", create=False
    )
    path = destination / STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FILENAME
    try:
        payload, file_sha256 = _load_json_regular_file_with_sha256(path)
        checkpoint = StageEditContextualSourcePairCheckpointV2.from_dict(
            payload
        )
        for expected, observed, field in (
            (expected_checkpoint_sha256, checkpoint.checkpoint_sha256, "expected_checkpoint_sha256"),
            (expected_corpus_sha256, checkpoint.corpus_sha256, "expected_corpus_sha256"),
            (expected_token_schema_fingerprint, checkpoint.token_schema_fingerprint, "expected_token_schema_fingerprint"),
        ):
            if expected is not None and observed != _sha256(expected, field):
                raise StageEditContextualSourcePairV2Error(f"{field} does not match artifact")
        if (
            expected_checkpoint_file_sha256 is not None
            and file_sha256
            != _sha256(
                expected_checkpoint_file_sha256,
                "expected_checkpoint_file_sha256",
            )
        ):
            raise StageEditContextualSourcePairV2Error(
                "expected checkpoint file-byte digest does not match artifact"
            )
    except StageEditContextualSourcePairV2ArtifactError:
        raise
    except (TypeError, ValueError, StageEditContextualSourcePairV2Error) as exc:
        raise StageEditContextualSourcePairV2ArtifactError(
            f"contextual v2 checkpoint is invalid: {exc}"
        ) from exc
    return checkpoint, file_sha256


def inspect_stage_edit_contextual_source_pair_checkpoint_v2(
    directory: Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_file_sha256: str | None = None,
    expected_corpus_sha256: str | None = None,
    expected_token_schema_fingerprint: str | None = None,
) -> StageEditContextualSourcePairCheckpointV2:
    """Offline-only artifact inspection; never creates an activation scorer."""

    checkpoint, _file_sha256 = _load_stage_edit_contextual_source_pair_checkpoint_v2(
        directory,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_checkpoint_file_sha256=expected_checkpoint_file_sha256,
        expected_corpus_sha256=expected_corpus_sha256,
        expected_token_schema_fingerprint=expected_token_schema_fingerprint,
    )
    return checkpoint


@dataclass(frozen=True, slots=True)
class StageEditContextualSourcePairEvaluationBundleV2:
    """Sealed scorer identity usable only for offline Beam-QA bootstrapping."""

    directory: Path
    corpus: StageEditContextualTrainingCorpusV2
    checkpoint: StageEditContextualSourcePairCheckpointV2
    scorer: StageEditContextualSourcePairScorerV2
    corpus_file_sha256: str
    checkpoint_file_sha256: str
    authority: str = CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2
    _activation_seal: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._activation_seal is not _EVALUATION_BUNDLE_ACTIVATION_SEAL:
            raise StageEditContextualSourcePairV2ArtifactError(
                "contextual-v2 evaluation bundle must be issued by its pinned loader"
            )
        if self.authority != (
            CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2
        ):
            raise StageEditContextualSourcePairV2ArtifactError(
                "contextual-v2 evaluation authority mismatch"
            )
        directory = _verified_lexical_directory(
            self.directory,
            field="contextual-v2 evaluation bundle directory",
            create=False,
        )
        object.__setattr__(self, "directory", directory)
        object.__setattr__(
            self,
            "corpus_file_sha256",
            _sha256(self.corpus_file_sha256, "corpus_file_sha256"),
        )
        object.__setattr__(
            self,
            "checkpoint_file_sha256",
            _sha256(self.checkpoint_file_sha256, "checkpoint_file_sha256"),
        )
        if not isinstance(self.corpus, StageEditContextualTrainingCorpusV2):
            raise StageEditContextualSourcePairV2ArtifactError(
                "contextual-v2 evaluation corpus is invalid"
            )
        if not isinstance(
            self.checkpoint, StageEditContextualSourcePairCheckpointV2
        ):
            raise StageEditContextualSourcePairV2ArtifactError(
                "contextual-v2 evaluation checkpoint is invalid"
            )
        if not isinstance(self.scorer, StageEditContextualSourcePairScorerV2):
            raise StageEditContextualSourcePairV2ArtifactError(
                "contextual-v2 evaluation scorer is invalid"
            )
        if (
            self.corpus.content_sha256 != self.checkpoint.corpus_sha256
            or self.corpus.token_schema_fingerprint
            != self.checkpoint.token_schema_fingerprint
            or self.corpus.source_data_id != self.checkpoint.source_data_id
            or self.corpus.feature_snapshot_bindings_sha256
            != self.checkpoint.feature_snapshot_bindings_sha256
            or self.corpus.clipped_observed_value_count
            != self.checkpoint.clipped_observed_value_count
            or self.corpus.song_ids != self.checkpoint.training_song_ids
            or self.corpus.reference_ids
            != self.checkpoint.training_reference_ids
            or tuple(sorted(self.corpus.group_ids))
            != self.checkpoint.training_group_ids
            or self.scorer.fingerprint != self.checkpoint.scorer_fingerprint
            or self.scorer.token_schema_fingerprint
            != self.checkpoint.token_schema_fingerprint
            or self.scorer.weights != self.checkpoint.weights
        ):
            raise StageEditContextualSourcePairV2ArtifactError(
                "contextual-v2 evaluation corpus/checkpoint/scorer lineage changed"
            )

    def loader_kwargs(self) -> dict[str, str]:
        return {
            "expected_checkpoint_sha256": self.checkpoint.checkpoint_sha256,
            "expected_checkpoint_file_sha256": self.checkpoint_file_sha256,
            "expected_corpus_sha256": self.corpus.content_sha256,
            "expected_corpus_file_sha256": self.corpus_file_sha256,
            "expected_token_schema_fingerprint": (
                self.checkpoint.token_schema_fingerprint
            ),
            "expected_scorer_fingerprint": self.checkpoint.scorer_fingerprint,
            "expected_runtime_equivalence_sha256": (
                self.checkpoint.runtime_equivalence_sha256
            ),
        }


def is_stage_edit_contextual_source_pair_evaluation_bundle_v2(
    value: object,
) -> bool:
    return bool(
        isinstance(value, StageEditContextualSourcePairEvaluationBundleV2)
        and value._activation_seal is _EVALUATION_BUNDLE_ACTIVATION_SEAL
    )


def require_stage_edit_contextual_source_pair_evaluation_bundle_v2(
    value: object,
) -> StageEditContextualSourcePairEvaluationBundleV2:
    if not is_stage_edit_contextual_source_pair_evaluation_bundle_v2(value):
        raise StageEditContextualSourcePairV2ArtifactError(
            "Beam-QA evaluation requires a loader-issued contextual-v2 bundle"
        )
    assert isinstance(value, StageEditContextualSourcePairEvaluationBundleV2)
    return value


def load_stage_edit_contextual_source_pair_evaluation_bundle_v2(
    directory: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_checkpoint_file_sha256: str,
    expected_corpus_sha256: str,
    expected_corpus_file_sha256: str,
    expected_token_schema_fingerprint: str,
    expected_scorer_fingerprint: str,
    expected_runtime_equivalence_sha256: str,
) -> StageEditContextualSourcePairEvaluationBundleV2:
    """Issue an offline-only scorer after exact corpus/checkpoint replay.

    This loader deliberately does not read or accept a production report.  Its
    sealed result is consumed only by the dedicated Beam-QA evaluation decoder;
    registry discovery and the normal app bridge never accept this authority.
    """

    destination = _verified_lexical_directory(
        directory,
        field="contextual-v2 evaluation bundle directory",
        create=False,
    )
    corpus, corpus_file_sha256 = _load_stage_edit_contextual_training_corpus_v2(
        destination,
        expected_corpus_sha256=_sha256(
            expected_corpus_sha256, "expected_corpus_sha256"
        ),
        expected_corpus_file_sha256=_sha256(
            expected_corpus_file_sha256,
            "expected_corpus_file_sha256",
        ),
        expected_token_schema_fingerprint=_sha256(
            expected_token_schema_fingerprint,
            "expected_token_schema_fingerprint",
        ),
    )
    checkpoint, checkpoint_file_sha256 = (
        _load_stage_edit_contextual_source_pair_checkpoint_v2(
            destination,
            expected_checkpoint_sha256=_sha256(
                expected_checkpoint_sha256,
                "expected_checkpoint_sha256",
            ),
            expected_checkpoint_file_sha256=_sha256(
                expected_checkpoint_file_sha256,
                "expected_checkpoint_file_sha256",
            ),
            expected_corpus_sha256=corpus.content_sha256,
            expected_token_schema_fingerprint=(
                corpus.token_schema_fingerprint
            ),
        )
    )
    expected_scorer = _sha256(
        expected_scorer_fingerprint,
        "expected_scorer_fingerprint",
    )
    expected_equivalence = _sha256(
        expected_runtime_equivalence_sha256,
        "expected_runtime_equivalence_sha256",
    )
    if (
        checkpoint.scorer_fingerprint != expected_scorer
        or checkpoint.runtime_equivalence_sha256 != expected_equivalence
    ):
        raise StageEditContextualSourcePairV2ArtifactError(
            "contextual-v2 evaluation scorer/runtime-equivalence pin changed"
        )
    try:
        replayed_training = train_stage_edit_contextual_source_pair_v2(
            corpus,
            config=StageEditContextualTrainingConfigV2(),
        )
    except StageEditContextualSourcePairV2Error as exc:
        raise StageEditContextualSourcePairV2ArtifactError(
            f"contextual-v2 evaluation runtime replay failed: {exc}"
        ) from exc
    scorer = StageEditContextualSourcePairScorerV2(
        token_schema_fingerprint=checkpoint.token_schema_fingerprint,
        weights=checkpoint.weights,
        feature_names=checkpoint.feature_names,
        contract=checkpoint.contract,
        authority=checkpoint.authority,
    )
    if (
        replayed_training.corpus_sha256 != checkpoint.corpus_sha256
        or replayed_training.source_data_id != checkpoint.source_data_id
        or replayed_training.feature_snapshot_bindings_sha256
        != checkpoint.feature_snapshot_bindings_sha256
        or replayed_training.clipped_observed_value_count
        != checkpoint.clipped_observed_value_count
        or replayed_training.training_song_ids != checkpoint.training_song_ids
        or replayed_training.training_reference_ids
        != checkpoint.training_reference_ids
        or replayed_training.training_group_ids
        != checkpoint.training_group_ids
        or replayed_training.config.to_dict() != dict(checkpoint.training_config)
        or replayed_training.scorer.to_dict() != scorer.to_dict()
        or replayed_training.scorer.fingerprint
        != checkpoint.scorer_fingerprint
        or replayed_training.runtime_equivalence_sha256
        != checkpoint.runtime_equivalence_sha256
    ):
        raise StageEditContextualSourcePairV2ArtifactError(
            "contextual-v2 evaluation checkpoint differs from deterministic runtime replay"
        )
    return StageEditContextualSourcePairEvaluationBundleV2(
        directory=destination,
        corpus=corpus,
        checkpoint=checkpoint,
        scorer=scorer,
        corpus_file_sha256=corpus_file_sha256,
        checkpoint_file_sha256=checkpoint_file_sha256,
        _activation_seal=_EVALUATION_BUNDLE_ACTIVATION_SEAL,
    )


def save_stage_edit_contextual_source_pair_report_v2(
    directory: Path,
    report: StageEditContextualSourcePairProductionReportV2,
) -> Path:
    if not isinstance(report, StageEditContextualSourcePairProductionReportV2):
        raise StageEditContextualSourcePairV2Error("production report is invalid")
    destination = _verified_lexical_directory(
        directory, field="production report directory", create=True
    )
    return _write_json_no_replace(
        destination / STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_FILENAME,
        report.to_dict(),
    )


def _load_stage_edit_contextual_source_pair_report_v2(
    directory: Path,
    *,
    expected_report_sha256: str | None = None,
    expected_report_file_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> tuple[StageEditContextualSourcePairProductionReportV2, str]:
    destination = _verified_lexical_directory(
        directory, field="production report directory", create=False
    )
    path = destination / STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_FILENAME
    try:
        payload, file_sha256 = _load_json_regular_file_with_sha256(path)
        report = StageEditContextualSourcePairProductionReportV2.from_dict(
            payload
        )
        if expected_report_sha256 is not None and report.report_sha256 != _sha256(
            expected_report_sha256, "expected_report_sha256"
        ):
            raise StageEditContextualSourcePairV2Error("expected report digest does not match")
        if expected_checkpoint_sha256 is not None and report.checkpoint.checkpoint_sha256 != _sha256(
            expected_checkpoint_sha256, "expected_checkpoint_sha256"
        ):
            raise StageEditContextualSourcePairV2Error("expected checkpoint digest does not match report")
        if (
            expected_report_file_sha256 is not None
            and file_sha256
            != _sha256(
                expected_report_file_sha256,
                "expected_report_file_sha256",
            )
        ):
            raise StageEditContextualSourcePairV2Error(
                "expected report file-byte digest does not match artifact"
            )
    except StageEditContextualSourcePairV2ArtifactError:
        raise
    except (TypeError, ValueError, StageEditContextualSourcePairV2Error) as exc:
        raise StageEditContextualSourcePairV2ArtifactError(
            f"contextual v2 production report is invalid: {exc}"
        ) from exc
    return report, file_sha256


def inspect_stage_edit_contextual_source_pair_report_v2(
    directory: Path,
    *,
    expected_report_sha256: str | None = None,
    expected_report_file_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> StageEditContextualSourcePairProductionReportV2:
    """Offline-only report inspection; not an activation boundary."""

    report, _file_sha256 = _load_stage_edit_contextual_source_pair_report_v2(
        directory,
        expected_report_sha256=expected_report_sha256,
        expected_report_file_sha256=expected_report_file_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    return report


@dataclass(frozen=True, slots=True)
class StageEditContextualSourcePairProductionBundleV2:
    """Exact sibling corpus/checkpoint/report set returned by activation."""

    directory: Path
    corpus: StageEditContextualTrainingCorpusV2
    checkpoint: StageEditContextualSourcePairCheckpointV2
    report: StageEditContextualSourcePairProductionReportV2
    scorer: StageEditContextualSourcePairScorerV2
    corpus_file_sha256: str
    checkpoint_file_sha256: str
    report_file_sha256: str
    _activation_seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._activation_seal is not _PRODUCTION_BUNDLE_ACTIVATION_SEAL:
            raise StageEditContextualSourcePairV2ArtifactError(
                "production bundle must be issued by the activation loader"
            )
        directory = _verified_lexical_directory(
            self.directory, field="production bundle directory", create=False
        )
        object.__setattr__(self, "directory", directory)
        object.__setattr__(
            self,
            "corpus_file_sha256",
            _sha256(self.corpus_file_sha256, "corpus_file_sha256"),
        )
        object.__setattr__(
            self,
            "checkpoint_file_sha256",
            _sha256(self.checkpoint_file_sha256, "checkpoint_file_sha256"),
        )
        object.__setattr__(
            self,
            "report_file_sha256",
            _sha256(self.report_file_sha256, "report_file_sha256"),
        )
        if not isinstance(self.corpus, StageEditContextualTrainingCorpusV2):
            raise StageEditContextualSourcePairV2ArtifactError(
                "production bundle training corpus is invalid"
            )
        if not isinstance(self.checkpoint, StageEditContextualSourcePairCheckpointV2):
            raise StageEditContextualSourcePairV2ArtifactError(
                "production bundle checkpoint is invalid"
            )
        if not isinstance(
            self.report, StageEditContextualSourcePairProductionReportV2
        ):
            raise StageEditContextualSourcePairV2ArtifactError(
                "production bundle report is invalid"
            )
        if self.report.checkpoint != self.checkpoint:
            raise StageEditContextualSourcePairV2ArtifactError(
                "production bundle sibling checkpoint differs from report"
            )
        if (
            self.corpus.content_sha256 != self.checkpoint.corpus_sha256
            or self.corpus.token_schema_fingerprint
            != self.checkpoint.token_schema_fingerprint
            or self.corpus.source_data_id != self.checkpoint.source_data_id
            or self.corpus.feature_snapshot_bindings_sha256
            != self.checkpoint.feature_snapshot_bindings_sha256
            or self.corpus.clipped_observed_value_count
            != self.checkpoint.clipped_observed_value_count
            or self.corpus.song_ids != self.checkpoint.training_song_ids
            or self.corpus.reference_ids != self.checkpoint.training_reference_ids
            or tuple(sorted(self.corpus.group_ids))
            != self.checkpoint.training_group_ids
        ):
            raise StageEditContextualSourcePairV2ArtifactError(
                "production bundle corpus differs from checkpoint lineage"
            )
        if (
            self.report.beam_level_qa.checkpoint_sha256
            != self.checkpoint.checkpoint_sha256
        ):
            raise StageEditContextualSourcePairV2ArtifactError(
                "production bundle QA is not bound to sibling checkpoint"
            )
        if not isinstance(self.scorer, StageEditContextualSourcePairScorerV2):
            raise StageEditContextualSourcePairV2ArtifactError(
                "production bundle scorer is invalid"
            )
        if (
            self.scorer.fingerprint != self.checkpoint.scorer_fingerprint
            or self.scorer.token_schema_fingerprint
            != self.checkpoint.token_schema_fingerprint
            or self.scorer.weights != self.checkpoint.weights
        ):
            raise StageEditContextualSourcePairV2ArtifactError(
                "production bundle scorer differs from sealed checkpoint"
            )


def load_stage_edit_contextual_source_pair_production_bundle_v2(
    directory: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_checkpoint_file_sha256: str,
    expected_report_sha256: str,
    expected_report_file_sha256: str,
    expected_beam_qa_report_sha256: str,
    expected_corpus_sha256: str,
    expected_corpus_file_sha256: str,
    expected_token_schema_fingerprint: str,
    verified_beam_qa_cases: Sequence[object],
) -> StageEditContextualSourcePairProductionBundleV2:
    """Activate only an exact, replayed, independently pinned sibling bundle."""

    destination = _verified_lexical_directory(
        directory, field="production bundle directory", create=False
    )
    corpus, corpus_file_sha256 = _load_stage_edit_contextual_training_corpus_v2(
        destination,
        expected_corpus_sha256=_sha256(
            expected_corpus_sha256, "expected_corpus_sha256"
        ),
        expected_corpus_file_sha256=_sha256(
            expected_corpus_file_sha256,
            "expected_corpus_file_sha256",
        ),
        expected_token_schema_fingerprint=_sha256(
            expected_token_schema_fingerprint,
            "expected_token_schema_fingerprint",
        ),
    )
    checkpoint, checkpoint_file_sha256 = _load_stage_edit_contextual_source_pair_checkpoint_v2(
        destination,
        expected_checkpoint_sha256=_sha256(
            expected_checkpoint_sha256, "expected_checkpoint_sha256"
        ),
        expected_checkpoint_file_sha256=_sha256(
            expected_checkpoint_file_sha256,
            "expected_checkpoint_file_sha256",
        ),
        expected_corpus_sha256=_sha256(
            expected_corpus_sha256, "expected_corpus_sha256"
        ),
        expected_token_schema_fingerprint=_sha256(
            expected_token_schema_fingerprint,
            "expected_token_schema_fingerprint",
        ),
    )
    report, report_file_sha256 = _load_stage_edit_contextual_source_pair_report_v2(
        destination,
        expected_report_sha256=_sha256(
            expected_report_sha256, "expected_report_sha256"
        ),
        expected_report_file_sha256=_sha256(
            expected_report_file_sha256,
            "expected_report_file_sha256",
        ),
        expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
    )
    if report.checkpoint != checkpoint:
        raise StageEditContextualSourcePairV2ArtifactError(
            "production report embedded checkpoint differs from sibling checkpoint"
        )
    expected_qa = _sha256(
        expected_beam_qa_report_sha256,
        "expected_beam_qa_report_sha256",
    )
    if report.beam_level_qa.qa_report_sha256 != expected_qa:
        raise StageEditContextualSourcePairV2ArtifactError(
            "expected beam QA digest does not match production report"
        )
    try:
        replayed_cross_validation = cross_validate_stage_edit_contextual_source_pair_v2(
            corpus,
            config=report.final_full_corpus_training.config,
        )
        replayed_final_training = train_stage_edit_contextual_source_pair_v2(
            corpus,
            config=report.final_full_corpus_training.config,
        )
    except StageEditContextualSourcePairV2Error as exc:
        raise StageEditContextualSourcePairV2ArtifactError(
            f"production bundle deterministic corpus replay failed: {exc}"
        ) from exc
    if replayed_cross_validation != report.cross_validation:
        raise StageEditContextualSourcePairV2ArtifactError(
            "production report LOSO differs from trusted corpus replay"
        )
    if replayed_final_training != report.final_full_corpus_training:
        raise StageEditContextualSourcePairV2ArtifactError(
            "production report final fit differs from trusted corpus replay"
        )
    try:
        replayed_beam_qa = _beam_qa_evidence_from_verified_envelopes_v2(
            checkpoint=checkpoint,
            evaluation_corpus_sha256=report.beam_level_qa.evaluation_corpus_sha256,
            verified_cases=verified_beam_qa_cases,
        )
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise StageEditContextualSourcePairV2ArtifactError(
            f"production bundle Beam-QA reverification failed: {exc}"
        ) from exc
    if replayed_beam_qa != report.beam_level_qa:
        raise StageEditContextualSourcePairV2ArtifactError(
            "persisted Beam-QA differs from reverified activation envelopes"
        )
    scorer = StageEditContextualSourcePairScorerV2(
        token_schema_fingerprint=checkpoint.token_schema_fingerprint,
        weights=checkpoint.weights,
        feature_names=checkpoint.feature_names,
        contract=checkpoint.contract,
        authority=checkpoint.authority,
    )
    return StageEditContextualSourcePairProductionBundleV2(
        directory=destination,
        corpus=corpus,
        checkpoint=checkpoint,
        report=report,
        scorer=scorer,
        corpus_file_sha256=corpus_file_sha256,
        checkpoint_file_sha256=checkpoint_file_sha256,
        report_file_sha256=report_file_sha256,
        _activation_seal=_PRODUCTION_BUNDLE_ACTIVATION_SEAL,
    )


__all__ = [
    "STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BEAM_QA_CASE_CONTRACT",
    "STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BEAM_QA_CONTRACT",
    "STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BEAM_QA_THRESHOLDS",
    "STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RUNTIME_EQUIVALENCE_CONTRACT",
    "StageEditContextualBeamQACaseV2",
    "StageEditContextualBeamQAEvidenceV2",
    "StageEditContextualCrossValidationFoldV2",
    "StageEditContextualCrossValidationResultV2",
    "StageEditContextualSourcePairCheckpointV2",
    "StageEditContextualSourcePairEvaluationBundleV2",
    "StageEditContextualSourcePairProductionBundleV2",
    "StageEditContextualSourcePairProductionReportV2",
    "StageEditContextualTrainingResultV2",
    "cross_validate_stage_edit_contextual_source_pair_v2",
    "evaluate_stage_edit_contextual_source_pair_ranking_v2",
    "inspect_stage_edit_contextual_training_corpus_v2",
    "inspect_stage_edit_contextual_source_pair_checkpoint_v2",
    "inspect_stage_edit_contextual_source_pair_report_v2",
    "is_stage_edit_contextual_source_pair_evaluation_bundle_v2",
    "load_stage_edit_contextual_source_pair_evaluation_bundle_v2",
    "load_stage_edit_contextual_source_pair_production_bundle_v2",
    "load_stage_edit_contextual_training_corpus_v2",
    "new_stage_edit_contextual_beam_qa_evidence_v2",
    "new_stage_edit_contextual_source_pair_checkpoint_v2",
    "new_stage_edit_contextual_source_pair_production_report_v2",
    "require_stage_edit_contextual_source_pair_evaluation_bundle_v2",
    "save_stage_edit_contextual_source_pair_checkpoint_v2",
    "save_stage_edit_contextual_source_pair_report_v2",
    "save_stage_edit_contextual_training_corpus_v2",
    "stage_edit_contextual_group_balance_weights_v2",
    "stage_edit_contextual_listwise_loss_v2",
    "train_stage_edit_contextual_source_pair_v2",
]
