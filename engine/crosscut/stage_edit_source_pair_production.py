"""Song-disjoint production gate for the learned source-pair scorer.

The reference bridge materializes causal, same-token source-pair groups once.
This module then performs deterministic leave-one-song-out cross-validation:
every fold fits only non-held-out songs and ranks only the held-out song.  A
full-corpus production checkpoint is permitted only when the learned scorer is
no worse than both fixed baselines on every fold and on the reference-balanced
aggregate.  The in-sample fit gate is deliberately absent from this module and
therefore cannot authorize production.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from .stage_edit_reference_adapter import ReferenceTrainingCorpus
from .stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME,
    StageEditSourcePairCheckpoint,
    StageEditSourcePairScorer,
    StageEditSourcePairTrainingGroup,
    StageEditSourcePairTrainingResult,
    load_stage_edit_source_pair_checkpoint,
    new_stage_edit_source_pair_checkpoint,
    save_stage_edit_source_pair_checkpoint,
    train_stage_edit_source_pair_scorer,
)
from .stage_edit_source_pair_training import (
    STAGE_EDIT_SOURCE_PAIR_HANDCRAFTED_BASELINE_CONTRACT,
    StageEditSourcePairActivationGate,
    StageEditSourcePairRankingMetrics,
    StageEditSourcePairTrainingCorpus,
    StageEditSourcePairTrainingError,
    evaluate_stage_edit_source_pair_ranking,
    materialize_stage_edit_source_pair_training_corpus,
    stage_edit_source_pair_no_worse_activation_gate,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    StageEditTokenSchema,
)


STAGE_EDIT_SOURCE_PAIR_PRODUCTION_SCHEMA_VERSION = 1
STAGE_EDIT_SOURCE_PAIR_CROSS_VALIDATION_CONTRACT = (
    "song_disjoint_leave_one_song_out_source_pair_gate_v1"
)
STAGE_EDIT_SOURCE_PAIR_PRODUCTION_REPORT_CONTRACT = (
    "tamper_bound_source_pair_checkpoint_song_disjoint_report_v1"
)
STAGE_EDIT_SOURCE_PAIR_PRODUCTION_REPORT_FILENAME = (
    "source-pair-production-report.json"
)


class StageEditSourcePairProductionError(StageEditSourcePairTrainingError):
    """Source-pair production validation or publication failed closed."""


class StageEditSourcePairCrossValidationError(StageEditSourcePairProductionError):
    """Song-disjoint folds are missing, overlapping, or contaminated."""


class StageEditSourcePairProductionGateError(StageEditSourcePairProductionError):
    """Held-out ranking failed one or more production requirements."""


def _name(value: object, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise StageEditSourcePairCrossValidationError(f"{field} cannot be blank")
    return result


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise StageEditSourcePairCrossValidationError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StageEditSourcePairCrossValidationError(
            f"{field} must be numeric"
        ) from exc
    if not math.isfinite(result):
        raise StageEditSourcePairCrossValidationError(f"{field} must be finite")
    return result


def _canonical_json_bytes(payload: object, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if pretty else None,
            sort_keys=True,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        + ("\n" if pretty else "")
    ).encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _training_dict(result: StageEditSourcePairTrainingResult) -> dict[str, Any]:
    return {
        "training_steps": result.training_steps,
        "reference_count": result.reference_count,
        "group_count": result.group_count,
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "margin": result.margin,
        "corpus_sha256": result.corpus_sha256,
        "scorer": result.scorer.to_dict(),
    }


@dataclass(frozen=True, slots=True)
class StageEditSourcePairCrossValidationFold:
    fold_index: int
    heldout_song_id: str
    training_song_ids: tuple[str, ...]
    training_reference_ids: tuple[str, ...]
    heldout_reference_ids: tuple[str, ...]
    training_group_ids: tuple[str, ...]
    heldout_group_ids: tuple[str, ...]
    training: StageEditSourcePairTrainingResult
    learned_metrics: StageEditSourcePairRankingMetrics
    uniform_metrics: StageEditSourcePairRankingMetrics
    handcrafted_metrics: StageEditSourcePairRankingMetrics
    activation_gate: StageEditSourcePairActivationGate

    def __post_init__(self) -> None:
        if (
            isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or self.fold_index < 0
        ):
            raise StageEditSourcePairCrossValidationError(
                "fold_index must be a non-negative integer"
            )
        object.__setattr__(
            self,
            "heldout_song_id",
            _name(self.heldout_song_id, "heldout_song_id"),
        )
        for field in (
            "training_song_ids",
            "training_reference_ids",
            "heldout_reference_ids",
            "training_group_ids",
            "heldout_group_ids",
        ):
            values = tuple(_name(value, field) for value in getattr(self, field))
            if not values or len(values) != len(set(values)):
                raise StageEditSourcePairCrossValidationError(
                    f"{field} must be non-empty and unique"
                )
            if values != tuple(sorted(values)):
                raise StageEditSourcePairCrossValidationError(
                    f"{field} must be sorted deterministically"
                )
            object.__setattr__(self, field, values)
        if self.heldout_song_id in self.training_song_ids:
            raise StageEditSourcePairCrossValidationError(
                "held-out song leaked into fold training songs"
            )
        if set(self.training_reference_ids).intersection(
            self.heldout_reference_ids
        ):
            raise StageEditSourcePairCrossValidationError(
                "training and held-out references overlap"
            )
        if set(self.training_group_ids).intersection(self.heldout_group_ids):
            raise StageEditSourcePairCrossValidationError(
                "training and held-out groups overlap"
            )
        if not isinstance(self.training, StageEditSourcePairTrainingResult):
            raise StageEditSourcePairCrossValidationError(
                "fold training result is invalid"
            )
        if self.training.reference_count != len(self.training_reference_ids):
            raise StageEditSourcePairCrossValidationError(
                "fold training reference count mismatch"
            )
        if self.training.group_count != len(self.training_group_ids):
            raise StageEditSourcePairCrossValidationError(
                "fold training group count mismatch"
            )
        heldout_identity = (
            len(self.heldout_reference_ids),
            len(self.heldout_group_ids),
        )
        for metrics in (
            self.learned_metrics,
            self.uniform_metrics,
            self.handcrafted_metrics,
        ):
            if not isinstance(metrics, StageEditSourcePairRankingMetrics):
                raise StageEditSourcePairCrossValidationError(
                    "fold ranking metrics are invalid"
                )
            if (metrics.reference_count, metrics.group_count) != heldout_identity:
                raise StageEditSourcePairCrossValidationError(
                    "fold metrics do not describe the exact held-out groups"
                )
        if not isinstance(self.activation_gate, StageEditSourcePairActivationGate):
            raise StageEditSourcePairCrossValidationError(
                "fold activation gate is invalid"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "heldout_song_id": self.heldout_song_id,
            "training_song_ids": list(self.training_song_ids),
            "training_reference_ids": list(self.training_reference_ids),
            "heldout_reference_ids": list(self.heldout_reference_ids),
            "training_group_ids": list(self.training_group_ids),
            "heldout_group_ids": list(self.heldout_group_ids),
            "training": _training_dict(self.training),
            "heldout_ranking": {
                "learned": self.learned_metrics.to_dict(),
                "uniform": self.uniform_metrics.to_dict(),
                "handcrafted": self.handcrafted_metrics.to_dict(),
            },
            "activation_gate": self.activation_gate.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class StageEditSourcePairCrossValidationResult:
    corpus: StageEditSourcePairTrainingCorpus
    folds: tuple[StageEditSourcePairCrossValidationFold, ...]
    learned_metrics: StageEditSourcePairRankingMetrics
    uniform_metrics: StageEditSourcePairRankingMetrics
    handcrafted_metrics: StageEditSourcePairRankingMetrics
    aggregate_gate: StageEditSourcePairActivationGate
    learning_rate: float
    activation_tolerance: float
    contract: str = STAGE_EDIT_SOURCE_PAIR_CROSS_VALIDATION_CONTRACT
    schema_version: int = STAGE_EDIT_SOURCE_PAIR_PRODUCTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.corpus, StageEditSourcePairTrainingCorpus):
            raise StageEditSourcePairCrossValidationError(
                "cross-validation corpus is invalid"
            )
        if self.contract != STAGE_EDIT_SOURCE_PAIR_CROSS_VALIDATION_CONTRACT:
            raise StageEditSourcePairCrossValidationError(
                "cross-validation contract mismatch"
            )
        if self.schema_version != STAGE_EDIT_SOURCE_PAIR_PRODUCTION_SCHEMA_VERSION:
            raise StageEditSourcePairCrossValidationError(
                "cross-validation schema mismatch"
            )
        folds = tuple(self.folds)
        if not folds or any(
            not isinstance(fold, StageEditSourcePairCrossValidationFold)
            for fold in folds
        ):
            raise StageEditSourcePairCrossValidationError(
                "cross-validation needs typed folds"
            )
        if tuple(fold.fold_index for fold in folds) != tuple(range(len(folds))):
            raise StageEditSourcePairCrossValidationError(
                "cross-validation fold indices are not canonical"
            )
        if tuple(fold.heldout_song_id for fold in folds) != self.corpus.song_ids:
            raise StageEditSourcePairCrossValidationError(
                "every source-pair song must be held out exactly once"
            )
        if len(folds) != self.corpus.song_count:
            raise StageEditSourcePairCrossValidationError(
                "cross-validation fold/song count mismatch"
            )
        learning_rate = _finite(self.learning_rate, "learning_rate")
        if learning_rate <= 0.0:
            raise StageEditSourcePairCrossValidationError(
                "learning_rate must be positive"
            )
        tolerance = _finite(self.activation_tolerance, "activation_tolerance")
        if tolerance < 0.0:
            raise StageEditSourcePairCrossValidationError(
                "activation_tolerance must be non-negative"
            )
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "activation_tolerance", tolerance)
        song_by_group = dict(self.corpus.group_song_ids)
        expected_group_ids = {group.group_id for group in self.corpus.groups}
        heldout_group_ids = [
            group_id for fold in folds for group_id in fold.heldout_group_ids
        ]
        if (
            len(heldout_group_ids) != len(set(heldout_group_ids))
            or set(heldout_group_ids) != expected_group_ids
        ):
            raise StageEditSourcePairCrossValidationError(
                "held-out folds do not cover every group exactly once"
            )
        heldout_reference_ids = [
            reference_id
            for fold in folds
            for reference_id in fold.heldout_reference_ids
        ]
        if (
            len(heldout_reference_ids) != len(set(heldout_reference_ids))
            or set(heldout_reference_ids) != set(self.corpus.reference_ids)
        ):
            raise StageEditSourcePairCrossValidationError(
                "held-out folds do not cover every reference exactly once"
            )
        group_by_id = {group.group_id: group for group in self.corpus.groups}
        for fold in folds:
            expected_training_songs = tuple(
                song_id
                for song_id in self.corpus.song_ids
                if song_id != fold.heldout_song_id
            )
            if fold.training_song_ids != expected_training_songs:
                raise StageEditSourcePairCrossValidationError(
                    "fold did not train on every non-held-out song"
                )
            expected_heldout_groups = tuple(
                sorted(
                    group_id
                    for group_id, song_id in song_by_group.items()
                    if song_id == fold.heldout_song_id
                )
            )
            expected_training_groups = tuple(
                sorted(expected_group_ids - set(expected_heldout_groups))
            )
            if fold.heldout_group_ids != expected_heldout_groups:
                raise StageEditSourcePairCrossValidationError(
                    "fold held-out group set differs from its exact song"
                )
            if fold.training_group_ids != expected_training_groups:
                raise StageEditSourcePairCrossValidationError(
                    "fold training group set omits or includes the wrong song"
                )
            expected_heldout_references = tuple(
                sorted(
                    {
                        group_by_id[group_id].reference_id
                        for group_id in expected_heldout_groups
                    }
                )
            )
            expected_training_references = tuple(
                sorted(
                    {
                        group_by_id[group_id].reference_id
                        for group_id in expected_training_groups
                    }
                )
            )
            if fold.heldout_reference_ids != expected_heldout_references:
                raise StageEditSourcePairCrossValidationError(
                    "fold held-out reference set differs from its exact song"
                )
            if fold.training_reference_ids != expected_training_references:
                raise StageEditSourcePairCrossValidationError(
                    "fold training reference set is incomplete or contaminated"
                )
            expected_fold_gate = stage_edit_source_pair_no_worse_activation_gate(
                fold.learned_metrics,
                fold.uniform_metrics,
                fold.handcrafted_metrics,
                tolerance=tolerance,
            )
            if fold.activation_gate != expected_fold_gate:
                raise StageEditSourcePairCrossValidationError(
                    "fold activation gate does not match its held-out metrics"
                )
        aggregate_identity = (self.corpus.reference_count, self.corpus.group_count)
        for metrics in (
            self.learned_metrics,
            self.uniform_metrics,
            self.handcrafted_metrics,
        ):
            if (metrics.reference_count, metrics.group_count) != aggregate_identity:
                raise StageEditSourcePairCrossValidationError(
                    "aggregate metrics do not cover the full held-out corpus"
                )
        if not isinstance(self.aggregate_gate, StageEditSourcePairActivationGate):
            raise StageEditSourcePairCrossValidationError(
                "aggregate activation gate is invalid"
            )
        expected_aggregate_gate = stage_edit_source_pair_no_worse_activation_gate(
            self.learned_metrics,
            self.uniform_metrics,
            self.handcrafted_metrics,
            tolerance=tolerance,
        )
        if self.aggregate_gate != expected_aggregate_gate:
            raise StageEditSourcePairCrossValidationError(
                "aggregate activation gate does not match held-out metrics"
            )
        object.__setattr__(self, "folds", folds)

    @property
    def authorized(self) -> bool:
        return self.aggregate_gate.authorized and all(
            fold.activation_gate.authorized for fold in self.folds
        )

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons = [
            f"overall:{reason}" for reason in self.aggregate_gate.reasons
        ]
        for fold in self.folds:
            reasons.extend(
                f"heldout_song={fold.heldout_song_id}:{reason}"
                for reason in fold.activation_gate.reasons
            )
        return tuple(reasons)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "protocol": "song_disjoint_leave_one_song_out",
            "configuration": {
                "learning_rate": self.learning_rate,
                "activation_tolerance": self.activation_tolerance,
            },
            "source_corpus": self.corpus.to_dict(),
            "folds": [fold.to_dict() for fold in self.folds],
            "reference_balanced_heldout_aggregate": {
                "learned": self.learned_metrics.to_dict(),
                "uniform": self.uniform_metrics.to_dict(),
                "handcrafted": self.handcrafted_metrics.to_dict(),
            },
            "aggregate_activation_gate": self.aggregate_gate.to_dict(),
            "production_authorized": self.authorized,
            "production_denial_reasons": list(self.reasons),
            "leakage_contract": {
                "materialized_once_before_folds": True,
                "training_and_heldout_songs_disjoint": True,
                "training_and_heldout_references_disjoint": True,
                "heldout_groups_used_for_training": False,
                "every_song_held_out_once": True,
                "every_reference_aggregated_once": True,
                "in_sample_fit_gate_used_for_authorization": False,
            },
        }

    @property
    def validation_sha256(self) -> str:
        return _canonical_sha256(self.unsigned_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.unsigned_dict(),
            "validation_sha256": self.validation_sha256,
        }


def _reference_weighted_aggregate(
    folds: Sequence[StageEditSourcePairCrossValidationFold],
    field: str,
    *,
    scorer_name: str,
) -> StageEditSourcePairRankingMetrics:
    metrics = tuple(getattr(fold, field) for fold in folds)
    reference_count = sum(value.reference_count for value in metrics)
    group_count = sum(value.group_count for value in metrics)
    if reference_count < 1 or group_count < 1:
        raise StageEditSourcePairCrossValidationError(
            "held-out aggregate cannot be empty"
        )

    def weighted(metric_name: str) -> float:
        return math.fsum(
            value.reference_count * getattr(value, metric_name)
            for value in metrics
        ) / reference_count

    return StageEditSourcePairRankingMetrics(
        scorer_name=scorer_name,
        reference_count=reference_count,
        group_count=group_count,
        reference_balanced_top1_accuracy=weighted(
            "reference_balanced_top1_accuracy"
        ),
        reference_balanced_pairwise_accuracy=weighted(
            "reference_balanced_pairwise_accuracy"
        ),
        reference_balanced_mean_reciprocal_rank=weighted(
            "reference_balanced_mean_reciprocal_rank"
        ),
        reference_balanced_mean_chosen_margin=weighted(
            "reference_balanced_mean_chosen_margin"
        ),
    )


def _validated_song_groups(
    corpus: StageEditSourcePairTrainingCorpus,
) -> dict[str, tuple[StageEditSourcePairTrainingGroup, ...]]:
    if not isinstance(corpus, StageEditSourcePairTrainingCorpus):
        raise StageEditSourcePairCrossValidationError(
            "cross-validation requires a materialized source-pair corpus"
        )
    if corpus.song_count < 2:
        raise StageEditSourcePairCrossValidationError(
            "source-pair production cross-validation needs at least two songs"
        )
    song_by_group = dict(corpus.group_song_ids)
    if len(song_by_group) != len(corpus.group_song_ids):
        raise StageEditSourcePairCrossValidationError(
            "duplicate group/song identities are not allowed"
        )
    by_song: dict[str, list[StageEditSourcePairTrainingGroup]] = {
        song_id: [] for song_id in corpus.song_ids
    }
    reference_songs: dict[str, set[str]] = {}
    for group in corpus.groups:
        if group.group_id not in song_by_group:
            raise StageEditSourcePairCrossValidationError(
                "group is missing its canonical song identity"
            )
        song_id = _name(song_by_group[group.group_id], "group song_id")
        if song_id not in by_song:
            raise StageEditSourcePairCrossValidationError(
                "group names a song outside the canonical corpus song set"
            )
        by_song[song_id].append(group)
        reference_songs.setdefault(group.reference_id, set()).add(song_id)
    empty_songs = tuple(song for song in corpus.song_ids if not by_song[song])
    if empty_songs:
        raise StageEditSourcePairCrossValidationError(
            "leave-one-song-out fold is empty: " + ", ".join(empty_songs)
        )
    leaked_references = tuple(
        sorted(
            reference_id
            for reference_id, songs in reference_songs.items()
            if len(songs) != 1
        )
    )
    if leaked_references:
        raise StageEditSourcePairCrossValidationError(
            "references span multiple songs: " + ", ".join(leaked_references)
        )
    return {
        song_id: tuple(
            sorted(
                by_song[song_id],
                key=lambda group: (group.reference_id, group.group_id),
            )
        )
        for song_id in corpus.song_ids
    }


def cross_validate_stage_edit_source_pair_materialized_corpus(
    corpus: StageEditSourcePairTrainingCorpus,
    *,
    steps: int = 250,
    learning_rate: float = 0.25,
    margin: float = 0.05,
    tolerance: float = 1.0e-12,
) -> StageEditSourcePairCrossValidationResult:
    """Fit non-held-out songs and evaluate each exact held-out song once."""

    groups_by_song = _validated_song_groups(corpus)
    all_group_ids = {group.group_id for group in corpus.groups}
    folds: list[StageEditSourcePairCrossValidationFold] = []
    for fold_index, heldout_song_id in enumerate(corpus.song_ids):
        training_song_ids = tuple(
            song_id for song_id in corpus.song_ids if song_id != heldout_song_id
        )
        training_groups = tuple(
            group
            for song_id in training_song_ids
            for group in groups_by_song[song_id]
        )
        heldout_groups = groups_by_song[heldout_song_id]
        if not training_groups or not heldout_groups:
            raise StageEditSourcePairCrossValidationError(
                f"fold for {heldout_song_id} has empty training or held-out groups"
            )
        training_group_ids = tuple(sorted(group.group_id for group in training_groups))
        heldout_group_ids = tuple(sorted(group.group_id for group in heldout_groups))
        if set(training_group_ids).intersection(heldout_group_ids):
            raise StageEditSourcePairCrossValidationError(
                f"fold for {heldout_song_id} has overlapping groups"
            )
        if set(training_group_ids).union(heldout_group_ids) != all_group_ids:
            raise StageEditSourcePairCrossValidationError(
                f"fold for {heldout_song_id} does not partition all groups"
            )
        training_reference_ids = tuple(
            sorted({group.reference_id for group in training_groups})
        )
        heldout_reference_ids = tuple(
            sorted({group.reference_id for group in heldout_groups})
        )
        if set(training_reference_ids).intersection(heldout_reference_ids):
            raise StageEditSourcePairCrossValidationError(
                f"fold for {heldout_song_id} leaks held-out references into training"
            )

        training = train_stage_edit_source_pair_scorer(
            training_groups,
            steps=steps,
            learning_rate=learning_rate,
            margin=margin,
        )
        learned = evaluate_stage_edit_source_pair_ranking(
            heldout_groups,
            scorer=training.scorer,
            scorer_name=f"learned:heldout:{heldout_song_id}",
        )
        uniform = evaluate_stage_edit_source_pair_ranking(
            heldout_groups,
            scorer=StageEditSourcePairScorer.uniform(),
            scorer_name=f"uniform:heldout:{heldout_song_id}",
        )
        handcrafted = evaluate_stage_edit_source_pair_ranking(
            heldout_groups,
            scorer=None,
            scorer_name=(
                f"{STAGE_EDIT_SOURCE_PAIR_HANDCRAFTED_BASELINE_CONTRACT}:"
                f"heldout:{heldout_song_id}"
            ),
            handcrafted_baseline=True,
        )
        fold_gate = stage_edit_source_pair_no_worse_activation_gate(
            learned,
            uniform,
            handcrafted,
            tolerance=tolerance,
        )
        folds.append(
            StageEditSourcePairCrossValidationFold(
                fold_index=fold_index,
                heldout_song_id=heldout_song_id,
                training_song_ids=training_song_ids,
                training_reference_ids=training_reference_ids,
                heldout_reference_ids=heldout_reference_ids,
                training_group_ids=training_group_ids,
                heldout_group_ids=heldout_group_ids,
                training=training,
                learned_metrics=learned,
                uniform_metrics=uniform,
                handcrafted_metrics=handcrafted,
                activation_gate=fold_gate,
            )
        )

    fold_values = tuple(folds)
    learned_aggregate = _reference_weighted_aggregate(
        fold_values,
        "learned_metrics",
        scorer_name="learned:song_disjoint_leave_one_song_out",
    )
    uniform_aggregate = _reference_weighted_aggregate(
        fold_values,
        "uniform_metrics",
        scorer_name="uniform:song_disjoint_leave_one_song_out",
    )
    handcrafted_aggregate = _reference_weighted_aggregate(
        fold_values,
        "handcrafted_metrics",
        scorer_name=(
            f"{STAGE_EDIT_SOURCE_PAIR_HANDCRAFTED_BASELINE_CONTRACT}:"
            "song_disjoint_leave_one_song_out"
        ),
    )
    aggregate_gate = stage_edit_source_pair_no_worse_activation_gate(
        learned_aggregate,
        uniform_aggregate,
        handcrafted_aggregate,
        tolerance=tolerance,
    )
    return StageEditSourcePairCrossValidationResult(
        corpus=corpus,
        folds=fold_values,
        learned_metrics=learned_aggregate,
        uniform_metrics=uniform_aggregate,
        handcrafted_metrics=handcrafted_aggregate,
        aggregate_gate=aggregate_gate,
        learning_rate=learning_rate,
        activation_tolerance=tolerance,
    )


def cross_validate_stage_edit_source_pair_reference_corpus(
    corpus: ReferenceTrainingCorpus,
    *,
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
    steps: int = 250,
    learning_rate: float = 0.25,
    margin: float = 0.05,
    tolerance: float = 1.0e-12,
) -> StageEditSourcePairCrossValidationResult:
    """Materialize once, then run song-disjoint leave-one-song-out folds."""

    materialized = materialize_stage_edit_source_pair_training_corpus(
        corpus, schema=schema
    )
    return cross_validate_stage_edit_source_pair_materialized_corpus(
        materialized,
        steps=steps,
        learning_rate=learning_rate,
        margin=margin,
        tolerance=tolerance,
    )


@dataclass(frozen=True, slots=True)
class StageEditSourcePairProductionReport:
    cross_validation: StageEditSourcePairCrossValidationResult
    final_training: StageEditSourcePairTrainingResult
    checkpoint: StageEditSourcePairCheckpoint
    report_sha256: str
    contract: str = STAGE_EDIT_SOURCE_PAIR_PRODUCTION_REPORT_CONTRACT
    schema_version: int = STAGE_EDIT_SOURCE_PAIR_PRODUCTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.cross_validation.authorized:
            raise StageEditSourcePairProductionGateError(
                "denied cross-validation cannot form a production report"
            )
        if self.contract != STAGE_EDIT_SOURCE_PAIR_PRODUCTION_REPORT_CONTRACT:
            raise StageEditSourcePairProductionError(
                "production report contract mismatch"
            )
        if self.schema_version != STAGE_EDIT_SOURCE_PAIR_PRODUCTION_SCHEMA_VERSION:
            raise StageEditSourcePairProductionError(
                "production report schema mismatch"
            )
        corpus = self.cross_validation.corpus
        if self.final_training.reference_count != corpus.reference_count:
            raise StageEditSourcePairProductionError(
                "final training did not cover every reference"
            )
        if self.final_training.group_count != corpus.group_count:
            raise StageEditSourcePairProductionError(
                "final training did not cover every group"
            )
        if self.checkpoint.corpus_sha256 != self.final_training.corpus_sha256:
            raise StageEditSourcePairProductionError(
                "checkpoint is not bound to final training corpus"
            )
        if self.checkpoint.scorer != self.final_training.scorer:
            raise StageEditSourcePairProductionError(
                "checkpoint scorer differs from final full-corpus scorer"
            )
        if (
            not isinstance(self.report_sha256, str)
            or len(self.report_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.report_sha256)
        ):
            raise StageEditSourcePairProductionError(
                "production report digest must be lowercase SHA-256"
            )
        if self.report_sha256 != _canonical_sha256(self.unsigned_dict()):
            raise StageEditSourcePairProductionError(
                "production report digest does not match its contents"
            )

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "cross_validation": self.cross_validation.to_dict(),
            "final_full_corpus_training": _training_dict(self.final_training),
            "checkpoint": self.checkpoint.to_dict(),
            "production_authorization": {
                "authorized": True,
                "all_fold_gates_authorized": all(
                    fold.activation_gate.authorized
                    for fold in self.cross_validation.folds
                ),
                "overall_gate_authorized": (
                    self.cross_validation.aggregate_gate.authorized
                ),
                "final_scorer_trained_after_cross_validation_gate": True,
                "in_sample_fit_gate_used_for_authorization": False,
                "materialization_pass_count": 1,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "report_sha256": self.report_sha256}


@dataclass(frozen=True, slots=True)
class StageEditSourcePairProductionResult:
    report: StageEditSourcePairProductionReport
    checkpoint_path: Path
    report_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.report, StageEditSourcePairProductionReport):
            raise StageEditSourcePairProductionError(
                "production result report is invalid"
            )
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(self, "report_path", Path(self.report_path))


def _new_production_report(
    cross_validation: StageEditSourcePairCrossValidationResult,
    final_training: StageEditSourcePairTrainingResult,
    checkpoint: StageEditSourcePairCheckpoint,
) -> StageEditSourcePairProductionReport:
    unsigned = {
        "schema_version": STAGE_EDIT_SOURCE_PAIR_PRODUCTION_SCHEMA_VERSION,
        "contract": STAGE_EDIT_SOURCE_PAIR_PRODUCTION_REPORT_CONTRACT,
        "cross_validation": cross_validation.to_dict(),
        "final_full_corpus_training": _training_dict(final_training),
        "checkpoint": checkpoint.to_dict(),
        "production_authorization": {
            "authorized": True,
            "all_fold_gates_authorized": all(
                fold.activation_gate.authorized
                for fold in cross_validation.folds
            ),
            "overall_gate_authorized": cross_validation.aggregate_gate.authorized,
            "final_scorer_trained_after_cross_validation_gate": True,
            "in_sample_fit_gate_used_for_authorization": False,
            "materialization_pass_count": 1,
        },
    }
    return StageEditSourcePairProductionReport(
        cross_validation=cross_validation,
        final_training=final_training,
        checkpoint=checkpoint,
        report_sha256=_canonical_sha256(unsigned),
    )


def _preflight_output_paths(directory: Path) -> tuple[Path, Path, Path]:
    destination = Path(directory).expanduser().resolve()
    checkpoint_path = destination / STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME
    report_path = destination / STAGE_EDIT_SOURCE_PAIR_PRODUCTION_REPORT_FILENAME
    for path in (checkpoint_path, report_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                "refusing to overwrite immutable source-pair production "
                f"artifact: {path}"
            )
    return destination, checkpoint_path, report_path


def _save_report_no_replace(
    path: Path,
    report: StageEditSourcePairProductionReport,
) -> Path:
    encoded = _canonical_json_bytes(report.to_dict(), pretty=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite immutable source-pair production report: {path}"
        ) from exc
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return path.resolve()


def produce_stage_edit_source_pair_checkpoint(
    corpus: ReferenceTrainingCorpus,
    directory: Path,
    *,
    checkpoint_id: str,
    created_at: str,
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
    steps: int = 250,
    learning_rate: float = 0.25,
    margin: float = 0.05,
    tolerance: float = 1.0e-12,
) -> StageEditSourcePairProductionResult:
    """Cross-validate, fit the full corpus, and publish immutable artifacts."""

    destination, expected_checkpoint_path, report_path = _preflight_output_paths(
        directory
    )
    # Exactly one bridge pass. All folds and final training reuse these groups.
    materialized = materialize_stage_edit_source_pair_training_corpus(
        corpus, schema=schema
    )
    cross_validation = cross_validate_stage_edit_source_pair_materialized_corpus(
        materialized,
        steps=steps,
        learning_rate=learning_rate,
        margin=margin,
        tolerance=tolerance,
    )
    if not cross_validation.authorized:
        reason = "; ".join(cross_validation.reasons) or "unknown held-out denial"
        raise StageEditSourcePairProductionGateError(
            "source-pair production cross-validation denied: " + reason
        )

    # This fit happens only after every out-of-song gate passed. No in-sample
    # evaluation result participates in production authorization.
    final_training = train_stage_edit_source_pair_scorer(
        materialized.groups,
        steps=steps,
        learning_rate=learning_rate,
        margin=margin,
    )
    checkpoint = new_stage_edit_source_pair_checkpoint(
        _name(checkpoint_id, "checkpoint_id"),
        final_training,
        created_at=_name(created_at, "created_at"),
    )
    report = _new_production_report(
        cross_validation,
        final_training,
        checkpoint,
    )

    checkpoint_path = save_stage_edit_source_pair_checkpoint(
        destination, checkpoint
    )
    if checkpoint_path != expected_checkpoint_path.resolve():
        raise StageEditSourcePairProductionError(
            "checkpoint publisher returned an unexpected destination"
        )
    _loaded_scorer, loaded_checkpoint = load_stage_edit_source_pair_checkpoint(
        destination,
        expected_corpus_sha256=checkpoint.corpus_sha256,
        expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
    )
    if loaded_checkpoint != checkpoint:
        raise StageEditSourcePairProductionError(
            "published checkpoint does not round-trip exactly"
        )
    saved_report_path = _save_report_no_replace(report_path, report)
    return StageEditSourcePairProductionResult(
        report=report,
        checkpoint_path=checkpoint_path,
        report_path=saved_report_path,
    )


__all__ = [
    "STAGE_EDIT_SOURCE_PAIR_CROSS_VALIDATION_CONTRACT",
    "STAGE_EDIT_SOURCE_PAIR_PRODUCTION_REPORT_CONTRACT",
    "STAGE_EDIT_SOURCE_PAIR_PRODUCTION_REPORT_FILENAME",
    "STAGE_EDIT_SOURCE_PAIR_PRODUCTION_SCHEMA_VERSION",
    "StageEditSourcePairCrossValidationError",
    "StageEditSourcePairCrossValidationFold",
    "StageEditSourcePairCrossValidationResult",
    "StageEditSourcePairProductionError",
    "StageEditSourcePairProductionGateError",
    "StageEditSourcePairProductionReport",
    "StageEditSourcePairProductionResult",
    "cross_validate_stage_edit_source_pair_materialized_corpus",
    "cross_validate_stage_edit_source_pair_reference_corpus",
    "produce_stage_edit_source_pair_checkpoint",
]
