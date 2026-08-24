"""Learned additive continuity penalties for outgoing/incoming source pairs.

The global StageEdit Transformer deliberately has no access to a beam's
previously selected source.  This module provides the small, label-free core
needed to score that state-dependent pair later in the semi-Markov decoder.
It consumes only observations already carried by ``StageEditSourceContinuity``
and never imports the decoder module, avoiding a future circular dependency.

Every feature is a bounded discontinuity in ``[0, 1]``.  Scorer weights are a
non-negative simplex, so the additive score is a finite penalty in ``[-1, 0]``.
Training uses a deterministic projected-gradient rank objective averaged over
alternatives, then groups within each reference, then references equally.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence
import uuid

import numpy as np


STAGE_EDIT_SOURCE_PAIR_FEATURE_SCHEMA_VERSION = 1
STAGE_EDIT_SOURCE_PAIR_SCORER_FORMAT_VERSION = 1
STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT = (
    "label_free_additive_nonnegative_continuity_pair_penalty_"
    "reference_balanced_rank_v1"
)
STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME = "source-pair-scorer.json"

STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES = (
    "face_scale_log_delta",
    "face_center_distance",
    "body_center_distance_fallback",
    "pose_motion_delta",
    "detected_person_count_delta",
    "formation_person_count_delta",
    "formation_pose_coverage_delta",
    "camera_motion_distance",
    "camera_zoom_velocity_delta",
    "camera_roll_delta",
    "exposure_delta",
    "color_temperature_delta",
    "sharpness_delta",
    "missing_evidence_fraction",
)

_OBSERVATION_FIELD_NAMES = (
    "face_scale",
    "face_center_x",
    "face_center_y",
    "body_center_x",
    "body_center_y",
    "pose_motion",
    "camera_motion_x",
    "camera_motion_y",
    "camera_zoom_velocity",
    "camera_roll_normalized",
    "sharpness",
    "exposure",
    "color_temperature_normalized",
    "detected_person_count_normalized",
    "formation_person_count_normalized",
    "formation_pose_coverage",
)


class StageEditSourcePairScorerError(RuntimeError):
    """Base error for source-pair materialization, training, or loading."""


class StageEditSourcePairInputError(StageEditSourcePairScorerError, ValueError):
    """Raised when pair observations or supervision violate the contract."""


class StageEditSourcePairCheckpointError(StageEditSourcePairScorerError):
    """Raised when a checkpoint is missing, malformed, or tampered with."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise StageEditSourcePairInputError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StageEditSourcePairInputError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise StageEditSourcePairInputError(f"{name} must be finite")
    return number


def _name(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise StageEditSourcePairInputError(f"{name} cannot be blank")
    return result


def _strict_integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StageEditSourcePairInputError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise StageEditSourcePairInputError(f"{name} must be a lowercase SHA-256")
    return value


def _strict_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise StageEditSourcePairInputError(
            f"{name} keys are not exact; missing={missing}, extra={extra}"
        )


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _observation_values(observation: Any, name: str) -> dict[str, float | None]:
    if isinstance(observation, Mapping):
        getter = observation.get
    elif any(hasattr(observation, field) for field in _OBSERVATION_FIELD_NAMES):
        getter = lambda field: getattr(observation, field, None)
    else:
        raise StageEditSourcePairInputError(
            f"{name} must expose StageEdit source-continuity observations"
        )
    output: dict[str, float | None] = {}
    for field in _OBSERVATION_FIELD_NAMES:
        value = getter(field)
        output[field] = None if value is None else _finite(value, f"{name}.{field}")
    return output


def _bounded_delta(left: float, right: float, scale: float) -> float:
    return min(1.0, abs(left - right) / scale)


def _bounded_distance(
    left_x: float,
    left_y: float,
    right_x: float,
    right_y: float,
    scale: float,
) -> float:
    return min(1.0, math.hypot(right_x - left_x, right_y - left_y) / scale)


@dataclass(frozen=True, slots=True)
class StageEditSourcePairFeatureVector:
    """One bounded, label-free outgoing/incoming discontinuity vector."""

    values: tuple[float, ...]
    schema_version: int = STAGE_EDIT_SOURCE_PAIR_FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_EDIT_SOURCE_PAIR_FEATURE_SCHEMA_VERSION:
            raise StageEditSourcePairInputError(
                "unsupported source-pair feature schema version"
            )
        values = tuple(
            _finite(value, f"source_pair_features[{index}]")
            for index, value in enumerate(self.values)
        )
        if len(values) != len(STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES):
            raise StageEditSourcePairInputError(
                "source-pair feature vector has the wrong dimension"
            )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise StageEditSourcePairInputError(
                "source-pair feature values must lie in [0, 1]"
            )
        object.__setattr__(self, "values", values)

    @property
    def by_name(self) -> dict[str, float]:
        return dict(zip(STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES, self.values, strict=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_names": list(STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES),
            "values": list(self.values),
        }


def materialize_stage_edit_source_pair_features(
    outgoing: Any,
    incoming: Any,
) -> StageEditSourcePairFeatureVector:
    """Materialize bounded pair deltas from current continuity observations."""

    left = _observation_values(outgoing, "outgoing")
    right = _observation_values(incoming, "incoming")

    face_scale_observed = (
        left["face_scale"] is not None and right["face_scale"] is not None
    )
    if face_scale_observed:
        left_scale = max(abs(float(left["face_scale"])), 1.0e-6)
        right_scale = max(abs(float(right["face_scale"])), 1.0e-6)
        face_scale = min(
            1.0,
            abs(math.log(right_scale / left_scale)) / math.log(1.50),
        )
    else:
        face_scale = 0.0

    face_center_fields = (
        left["face_center_x"],
        left["face_center_y"],
        right["face_center_x"],
        right["face_center_y"],
    )
    face_center_observed = all(value is not None for value in face_center_fields)
    body_center_fields = (
        left["body_center_x"],
        left["body_center_y"],
        right["body_center_x"],
        right["body_center_y"],
    )
    body_center_observed = all(value is not None for value in body_center_fields)
    face_center = (
        _bounded_distance(
            *(float(value) for value in face_center_fields),
            scale=0.22,
        )
        if face_center_observed
        else 0.0
    )
    body_center_fallback = (
        _bounded_distance(
            *(float(value) for value in body_center_fields),
            scale=0.28,
        )
        if not face_center_observed and body_center_observed
        else 0.0
    )

    def scalar_delta(field: str, scale: float) -> tuple[float, bool]:
        observed = left[field] is not None and right[field] is not None
        return (
            _bounded_delta(float(left[field]), float(right[field]), scale)
            if observed
            else 0.0,
            observed,
        )

    pose_motion, pose_observed = scalar_delta("pose_motion", 0.40)
    detected_people, detected_observed = scalar_delta(
        "detected_person_count_normalized", 1.0
    )
    formation_people, formation_people_observed = scalar_delta(
        "formation_person_count_normalized", 1.0
    )
    formation_coverage, formation_coverage_observed = scalar_delta(
        "formation_pose_coverage", 1.0
    )

    camera_motion_fields = (
        left["camera_motion_x"],
        left["camera_motion_y"],
        right["camera_motion_x"],
        right["camera_motion_y"],
    )
    camera_motion_observed = all(
        value is not None for value in camera_motion_fields
    )
    camera_motion = (
        _bounded_distance(
            *(float(value) for value in camera_motion_fields),
            scale=0.75,
        )
        if camera_motion_observed
        else 0.0
    )
    zoom, zoom_observed = scalar_delta("camera_zoom_velocity", 0.35)
    roll, roll_observed = scalar_delta("camera_roll_normalized", 0.25)
    exposure, exposure_observed = scalar_delta("exposure", 0.25)
    color, color_observed = scalar_delta("color_temperature_normalized", 0.30)
    sharpness, sharpness_observed = scalar_delta("sharpness", 0.35)
    # Missingness is defined over twelve independent observation families, not
    # over the thirteen delta features above. Face and body centers share one
    # mutually exclusive geometry opportunity because body center is the
    # explicit fallback for a missing face center. Named slots make accidental
    # duplication (for example counting roll twice) impossible to hide in the
    # denominator.
    observed_evidence = {
        "face_scale": face_scale_observed,
        "subject_center": face_center_observed or body_center_observed,
        "pose_motion": pose_observed,
        "detected_people": detected_observed,
        "formation_people": formation_people_observed,
        "formation_coverage": formation_coverage_observed,
        "camera_motion": camera_motion_observed,
        "camera_zoom": zoom_observed,
        "camera_roll": roll_observed,
        "exposure": exposure_observed,
        "color_temperature": color_observed,
        "sharpness": sharpness_observed,
    }
    missing_fraction = 1.0 - sum(observed_evidence.values()) / len(
        observed_evidence
    )

    return StageEditSourcePairFeatureVector(
        (
            face_scale,
            face_center,
            body_center_fallback,
            pose_motion,
            detected_people,
            formation_people,
            formation_coverage,
            camera_motion,
            zoom,
            roll,
            exposure,
            color,
            sharpness,
            missing_fraction,
        )
    )


@dataclass(frozen=True, slots=True)
class StageEditSourcePairCandidate:
    source_id: str
    incoming: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _name(self.source_id, "source_id"))
        _observation_values(self.incoming, f"candidate[{self.source_id}]")


@dataclass(frozen=True, slots=True)
class StageEditSourcePairTrainingGroup:
    """One reference cut with a chosen incoming source and alternatives."""

    group_id: str
    reference_id: str
    outgoing_source_id: str
    outgoing: Any
    candidates: tuple[StageEditSourcePairCandidate, ...]
    chosen_incoming_source_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _name(self.group_id, "group_id"))
        object.__setattr__(
            self, "reference_id", _name(self.reference_id, "reference_id")
        )
        object.__setattr__(
            self,
            "outgoing_source_id",
            _name(self.outgoing_source_id, "outgoing_source_id"),
        )
        chosen = _name(
            self.chosen_incoming_source_id, "chosen_incoming_source_id"
        )
        object.__setattr__(self, "chosen_incoming_source_id", chosen)
        _observation_values(self.outgoing, "outgoing")
        candidates = tuple(self.candidates)
        if len(candidates) < 2 or any(
            not isinstance(candidate, StageEditSourcePairCandidate)
            for candidate in candidates
        ):
            raise StageEditSourcePairInputError(
                "source-pair group needs at least two typed candidates"
            )
        source_ids = tuple(candidate.source_id for candidate in candidates)
        if len(set(source_ids)) != len(source_ids):
            raise StageEditSourcePairInputError(
                "source-pair candidate source ids must be unique"
            )
        if chosen not in source_ids:
            raise StageEditSourcePairInputError(
                "chosen incoming source is absent from candidates"
            )
        for candidate in candidates:
            materialize_stage_edit_source_pair_features(
                self.outgoing, candidate.incoming
            )
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True, slots=True)
class StageEditSourcePairCandidateScore:
    source_id: str
    score: float
    penalty: float
    features: StageEditSourcePairFeatureVector

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _name(self.source_id, "source_id"))
        score = _finite(self.score, "score")
        penalty = _finite(self.penalty, "penalty")
        if not -1.0 <= score <= 0.0 or not 0.0 <= penalty <= 1.0:
            raise StageEditSourcePairInputError(
                "source-pair score/penalty lies outside its bounded contract"
            )
        if not math.isclose(score, -penalty, abs_tol=1.0e-12):
            raise StageEditSourcePairInputError(
                "source-pair score must be the negative additive penalty"
            )
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "penalty", penalty)


@dataclass(frozen=True, slots=True)
class StageEditSourcePairScorer:
    """A bounded additive penalty with non-negative simplex weights."""

    weights: tuple[float, ...]
    feature_names: tuple[str, ...] = STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES
    contract: str = STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT

    def __post_init__(self) -> None:
        if tuple(self.feature_names) != STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES:
            raise StageEditSourcePairInputError(
                "source-pair scorer feature schema is unsupported"
            )
        if self.contract != STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT:
            raise StageEditSourcePairInputError(
                "source-pair scorer contract is unsupported"
            )
        weights = tuple(
            _finite(value, f"weights[{index}]")
            for index, value in enumerate(self.weights)
        )
        if len(weights) != len(STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES):
            raise StageEditSourcePairInputError(
                "source-pair scorer weights have the wrong dimension"
            )
        if any(value < 0.0 for value in weights):
            raise StageEditSourcePairInputError(
                "source-pair continuity penalty weights must be non-negative"
            )
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise StageEditSourcePairInputError(
                "source-pair continuity penalty weights must sum to one"
            )
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    @classmethod
    def uniform(cls) -> "StageEditSourcePairScorer":
        count = len(STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES)
        return cls((1.0 / count,) * count)

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
        }

    def score(self, outgoing: Any, incoming: Any) -> float:
        features = materialize_stage_edit_source_pair_features(outgoing, incoming)
        penalty = float(np.dot(self.weights, features.values))
        return -min(1.0, max(0.0, penalty))

    def score_candidates(
        self,
        outgoing: Any,
        candidates: Sequence[StageEditSourcePairCandidate],
    ) -> tuple[StageEditSourcePairCandidateScore, ...]:
        values = tuple(candidates)
        if not values:
            raise StageEditSourcePairInputError("at least one candidate is required")
        if any(
            not isinstance(candidate, StageEditSourcePairCandidate)
            for candidate in values
        ):
            raise StageEditSourcePairInputError(
                "candidates must use StageEditSourcePairCandidate"
            )
        if len({candidate.source_id for candidate in values}) != len(values):
            raise StageEditSourcePairInputError("candidate source ids must be unique")
        scores: list[StageEditSourcePairCandidateScore] = []
        for candidate in sorted(values, key=lambda item: item.source_id):
            features = materialize_stage_edit_source_pair_features(
                outgoing, candidate.incoming
            )
            penalty = min(
                1.0, max(0.0, float(np.dot(self.weights, features.values)))
            )
            scores.append(
                StageEditSourcePairCandidateScore(
                    source_id=candidate.source_id,
                    score=-penalty,
                    penalty=penalty,
                    features=features,
                )
            )
        return tuple(scores)

    def rank_candidates(
        self,
        outgoing: Any,
        candidates: Sequence[StageEditSourcePairCandidate],
    ) -> tuple[StageEditSourcePairCandidateScore, ...]:
        return tuple(
            sorted(
                self.score_candidates(outgoing, candidates),
                key=lambda item: (-item.score, item.source_id),
            )
        )


def _validated_groups(
    groups: Sequence[StageEditSourcePairTrainingGroup],
) -> tuple[StageEditSourcePairTrainingGroup, ...]:
    values = tuple(groups)
    if not values or any(
        not isinstance(group, StageEditSourcePairTrainingGroup) for group in values
    ):
        raise StageEditSourcePairInputError(
            "training needs at least one typed source-pair group"
        )
    identities = tuple((group.reference_id, group.group_id) for group in values)
    if len(set(identities)) != len(identities):
        raise StageEditSourcePairInputError(
            "source-pair group ids must be unique within each reference"
        )
    return tuple(sorted(values, key=lambda group: (group.reference_id, group.group_id)))


def _materialized_group(
    group: StageEditSourcePairTrainingGroup,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    candidates = {
        candidate.source_id: np.asarray(
            materialize_stage_edit_source_pair_features(
                group.outgoing, candidate.incoming
            ).values,
            dtype=np.float64,
        )
        for candidate in sorted(group.candidates, key=lambda item: item.source_id)
    }
    chosen = candidates[group.chosen_incoming_source_id]
    alternatives = tuple(
        candidates[source_id]
        for source_id in sorted(candidates)
        if source_id != group.chosen_incoming_source_id
    )
    return chosen, alternatives


def _reference_balanced_loss_and_gradient(
    weights: np.ndarray,
    groups: Sequence[StageEditSourcePairTrainingGroup],
    *,
    margin: float,
) -> tuple[float, np.ndarray]:
    by_reference: dict[str, list[StageEditSourcePairTrainingGroup]] = {}
    for group in _validated_groups(groups):
        by_reference.setdefault(group.reference_id, []).append(group)

    reference_losses: list[float] = []
    reference_gradients: list[np.ndarray] = []
    for reference_id in sorted(by_reference):
        group_losses: list[float] = []
        group_gradients: list[np.ndarray] = []
        for group in by_reference[reference_id]:
            chosen, alternatives = _materialized_group(group)
            losses: list[float] = []
            gradients: list[np.ndarray] = []
            for alternative in alternatives:
                difference = chosen - alternative
                logit = margin + float(np.dot(weights, difference))
                losses.append(float(np.logaddexp(0.0, logit)))
                if logit >= 0.0:
                    sigmoid = 1.0 / (1.0 + math.exp(-logit))
                else:
                    exponential = math.exp(logit)
                    sigmoid = exponential / (1.0 + exponential)
                gradients.append(sigmoid * difference)
            group_losses.append(float(np.mean(losses)))
            group_gradients.append(np.mean(gradients, axis=0))
        reference_losses.append(float(np.mean(group_losses)))
        reference_gradients.append(np.mean(group_gradients, axis=0))
    loss = float(np.mean(reference_losses))
    gradient = np.mean(reference_gradients, axis=0)
    if not math.isfinite(loss) or not np.all(np.isfinite(gradient)):
        raise StageEditSourcePairInputError(
            "source-pair ranking objective became non-finite"
        )
    return loss, gradient


def stage_edit_source_pair_reference_balanced_loss(
    scorer: StageEditSourcePairScorer,
    groups: Sequence[StageEditSourcePairTrainingGroup],
    *,
    margin: float = 0.05,
) -> float:
    margin_value = _finite(margin, "margin")
    if margin_value < 0.0:
        raise StageEditSourcePairInputError("margin must be non-negative")
    loss, _gradient = _reference_balanced_loss_and_gradient(
        np.asarray(scorer.weights, dtype=np.float64),
        groups,
        margin=margin_value,
    )
    return loss


def _project_to_simplex(values: np.ndarray) -> np.ndarray:
    """Project finite values onto the non-negative unit simplex."""

    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise StageEditSourcePairInputError("simplex projection input is invalid")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, len(values) + 1, dtype=np.float64)
    positive = ordered - cumulative / indices > 0.0
    if not np.any(positive):  # Mathematically unreachable for finite input.
        raise StageEditSourcePairInputError("simplex projection has no support")
    rho = int(np.flatnonzero(positive)[-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(values - threshold, 0.0)
    projected /= projected.sum()
    return projected


def _training_corpus_sha256(
    groups: Sequence[StageEditSourcePairTrainingGroup],
) -> str:
    payload_groups: list[dict[str, Any]] = []
    for group in _validated_groups(groups):
        candidates = []
        for candidate in sorted(group.candidates, key=lambda item: item.source_id):
            candidates.append(
                {
                    "source_id": candidate.source_id,
                    "features": list(
                        materialize_stage_edit_source_pair_features(
                            group.outgoing, candidate.incoming
                        ).values
                    ),
                }
            )
        payload_groups.append(
            {
                "group_id": group.group_id,
                "reference_id": group.reference_id,
                "outgoing_source_id": group.outgoing_source_id,
                "chosen_incoming_source_id": group.chosen_incoming_source_id,
                "candidates": candidates,
            }
        )
    return _canonical_sha256(
        {
            "contract": STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
            "feature_names": list(STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES),
            "groups": payload_groups,
        }
    )


@dataclass(frozen=True, slots=True)
class StageEditSourcePairTrainingResult:
    scorer: StageEditSourcePairScorer
    training_steps: int
    reference_count: int
    group_count: int
    initial_loss: float
    final_loss: float
    margin: float
    corpus_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.scorer, StageEditSourcePairScorer):
            raise StageEditSourcePairInputError("training result scorer is invalid")
        _strict_integer(self.training_steps, "training_steps", minimum=1)
        _strict_integer(self.reference_count, "reference_count", minimum=1)
        _strict_integer(self.group_count, "group_count", minimum=1)
        initial = _finite(self.initial_loss, "initial_loss")
        final = _finite(self.final_loss, "final_loss")
        if final > initial + 1.0e-10:
            raise StageEditSourcePairInputError(
                "source-pair training result regressed its objective"
            )
        margin = _finite(self.margin, "margin")
        if margin < 0.0:
            raise StageEditSourcePairInputError("margin must be non-negative")
        object.__setattr__(self, "initial_loss", initial)
        object.__setattr__(self, "final_loss", final)
        object.__setattr__(self, "margin", margin)
        object.__setattr__(
            self, "corpus_sha256", _sha256(self.corpus_sha256, "corpus_sha256")
        )


def train_stage_edit_source_pair_scorer(
    groups: Sequence[StageEditSourcePairTrainingGroup],
    *,
    steps: int = 250,
    learning_rate: float = 0.25,
    margin: float = 0.05,
) -> StageEditSourcePairTrainingResult:
    """Fit deterministic non-negative weights with reference-balanced ranking."""

    values = _validated_groups(groups)
    step_count = _strict_integer(steps, "steps", minimum=1)
    rate = _finite(learning_rate, "learning_rate")
    margin_value = _finite(margin, "margin")
    if rate <= 0.0:
        raise StageEditSourcePairInputError("learning_rate must be positive")
    if margin_value < 0.0:
        raise StageEditSourcePairInputError("margin must be non-negative")

    weights = np.full(
        len(STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES),
        1.0 / len(STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES),
        dtype=np.float64,
    )
    initial_loss, _ = _reference_balanced_loss_and_gradient(
        weights, values, margin=margin_value
    )
    current_loss = initial_loss
    current_rate = rate
    for _step in range(step_count):
        current_loss, gradient = _reference_balanced_loss_and_gradient(
            weights, values, margin=margin_value
        )
        accepted = False
        trial_rate = current_rate
        for _attempt in range(24):
            proposed = _project_to_simplex(weights - trial_rate * gradient)
            proposed_loss, _ = _reference_balanced_loss_and_gradient(
                proposed, values, margin=margin_value
            )
            if proposed_loss <= current_loss + 1.0e-12:
                weights = proposed
                current_loss = proposed_loss
                current_rate = min(rate, trial_rate * 1.02)
                accepted = True
                break
            trial_rate *= 0.5
        if not accepted:
            break

    scorer = StageEditSourcePairScorer(tuple(float(value) for value in weights))
    final_loss = stage_edit_source_pair_reference_balanced_loss(
        scorer, values, margin=margin_value
    )
    return StageEditSourcePairTrainingResult(
        scorer=scorer,
        training_steps=step_count,
        reference_count=len({group.reference_id for group in values}),
        group_count=len(values),
        initial_loss=initial_loss,
        final_loss=final_loss,
        margin=margin_value,
        corpus_sha256=_training_corpus_sha256(values),
    )


@dataclass(frozen=True, slots=True)
class StageEditSourcePairCheckpoint:
    format_version: int
    contract: str
    checkpoint_id: str
    created_at: str
    training_steps: int
    reference_count: int
    group_count: int
    corpus_sha256: str
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if self.format_version != STAGE_EDIT_SOURCE_PAIR_SCORER_FORMAT_VERSION:
            raise StageEditSourcePairInputError(
                "unsupported source-pair checkpoint format"
            )
        if self.contract != STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT:
            raise StageEditSourcePairInputError(
                "unsupported source-pair checkpoint contract"
            )
        object.__setattr__(
            self, "checkpoint_id", _name(self.checkpoint_id, "checkpoint_id")
        )
        created = _name(self.created_at, "created_at")
        try:
            timestamp = datetime.fromisoformat(created)
        except ValueError as exc:
            raise StageEditSourcePairInputError(
                "created_at must be ISO-8601"
            ) from exc
        if timestamp.tzinfo is None:
            raise StageEditSourcePairInputError(
                "created_at must include a timezone"
            )
        object.__setattr__(self, "created_at", created)
        _strict_integer(self.training_steps, "training_steps", minimum=1)
        _strict_integer(self.reference_count, "reference_count", minimum=1)
        _strict_integer(self.group_count, "group_count", minimum=1)
        object.__setattr__(
            self, "corpus_sha256", _sha256(self.corpus_sha256, "corpus_sha256")
        )
        scorer = StageEditSourcePairScorer(
            weights=tuple(self.weights),
            feature_names=tuple(self.feature_names),
            contract=self.contract,
        )
        object.__setattr__(self, "weights", scorer.weights)
        object.__setattr__(self, "feature_names", scorer.feature_names)
        digest = _sha256(self.checkpoint_sha256, "checkpoint_sha256")
        object.__setattr__(self, "checkpoint_sha256", digest)
        if digest != _canonical_sha256(self.unsigned_dict()):
            raise StageEditSourcePairInputError(
                "source-pair checkpoint digest does not match its contents"
            )

    @property
    def scorer(self) -> StageEditSourcePairScorer:
        return StageEditSourcePairScorer(
            weights=self.weights,
            feature_names=self.feature_names,
            contract=self.contract,
        )

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "contract": self.contract,
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at,
            "training_steps": self.training_steps,
            "reference_count": self.reference_count,
            "group_count": self.group_count,
            "corpus_sha256": self.corpus_sha256,
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.unsigned_dict(),
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "StageEditSourcePairCheckpoint":
        expected = {
            "format_version",
            "contract",
            "checkpoint_id",
            "created_at",
            "training_steps",
            "reference_count",
            "group_count",
            "corpus_sha256",
            "feature_names",
            "weights",
            "checkpoint_sha256",
        }
        _strict_keys(payload, expected, "source-pair checkpoint")
        return cls(
            format_version=_strict_integer(
                payload["format_version"], "format_version", minimum=1
            ),
            contract=payload["contract"],
            checkpoint_id=payload["checkpoint_id"],
            created_at=payload["created_at"],
            training_steps=_strict_integer(
                payload["training_steps"], "training_steps", minimum=1
            ),
            reference_count=_strict_integer(
                payload["reference_count"], "reference_count", minimum=1
            ),
            group_count=_strict_integer(
                payload["group_count"], "group_count", minimum=1
            ),
            corpus_sha256=payload["corpus_sha256"],
            feature_names=tuple(payload["feature_names"]),
            weights=tuple(payload["weights"]),
            checkpoint_sha256=payload["checkpoint_sha256"],
        )


def new_stage_edit_source_pair_checkpoint(
    checkpoint_id: str,
    result: StageEditSourcePairTrainingResult,
    *,
    created_at: str | None = None,
) -> StageEditSourcePairCheckpoint:
    if not isinstance(result, StageEditSourcePairTrainingResult):
        raise StageEditSourcePairInputError("training result is invalid")
    unsigned = {
        "format_version": STAGE_EDIT_SOURCE_PAIR_SCORER_FORMAT_VERSION,
        "contract": STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
        "checkpoint_id": _name(checkpoint_id, "checkpoint_id"),
        "created_at": (
            datetime.now(UTC).isoformat()
            if created_at is None
            else _name(created_at, "created_at")
        ),
        "training_steps": result.training_steps,
        "reference_count": result.reference_count,
        "group_count": result.group_count,
        "corpus_sha256": result.corpus_sha256,
        "feature_names": list(result.scorer.feature_names),
        "weights": list(result.scorer.weights),
    }
    return StageEditSourcePairCheckpoint(
        format_version=unsigned["format_version"],
        contract=unsigned["contract"],
        checkpoint_id=unsigned["checkpoint_id"],
        created_at=unsigned["created_at"],
        training_steps=unsigned["training_steps"],
        reference_count=unsigned["reference_count"],
        group_count=unsigned["group_count"],
        corpus_sha256=unsigned["corpus_sha256"],
        feature_names=tuple(unsigned["feature_names"]),
        weights=tuple(unsigned["weights"]),
        checkpoint_sha256=_canonical_sha256(unsigned),
    )


def _checkpoint_json_bytes(checkpoint: StageEditSourcePairCheckpoint) -> bytes:
    return (
        json.dumps(
            checkpoint.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _darwin_rename_no_replace(temporary: Path, final: Path) -> bool:
    """Try Darwin's atomic no-replace rename, returning false if unsupported."""

    if sys.platform != "darwin":
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    rename_exclusive = getattr(libc, "renamex_np", None)
    if rename_exclusive is None:
        return False
    rename_exclusive.argtypes = (
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename_exclusive.restype = ctypes.c_int
    if rename_exclusive(
        os.fsencode(temporary),
        os.fsencode(final),
        0x00000004,  # RENAME_EXCL
    ) == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            f"refusing to overwrite immutable checkpoint: {final}"
        )
    unsupported = {
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTSUP,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
    }
    if error_number in unsupported:
        return False
    raise OSError(error_number, os.strerror(error_number), final)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {
                errno.EINVAL,
                errno.ENOTSUP,
                getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
            }:
                raise
    finally:
        os.close(descriptor)


def _publish_checkpoint_no_replace(final: Path, encoded: bytes) -> None:
    """Publish one immutable checkpoint without a check-then-replace race."""

    if final.exists() or final.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite immutable checkpoint: {final}"
        )
    temporary = final.with_name(f".{final.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            # Same-directory hard-link publication is atomic and cannot replace
            # an existing regular file, directory, or symlink.
            os.link(temporary, final, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite immutable checkpoint: {final}"
            ) from exc
        except OSError as link_error:
            unsupported_link = {
                errno.EINVAL,
                errno.ENOSYS,
                errno.ENOTSUP,
                errno.EPERM,
                getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
            }
            if link_error.errno not in unsupported_link:
                raise
            if not _darwin_rename_no_replace(temporary, final):
                # Some filesystems (notably ExFAT) expose neither hard links
                # nor exclusive rename. O_EXCL still guarantees immutability;
                # an interrupted partial file is fail-closed by the digest
                # validation in the loader.
                try:
                    final_descriptor = os.open(
                        final,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                    )
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"refusing to overwrite immutable checkpoint: {final}"
                    ) from exc
                with os.fdopen(final_descriptor, "wb", closefd=True) as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
        _fsync_directory(final.parent)
    finally:
        temporary.unlink(missing_ok=True)


def save_stage_edit_source_pair_checkpoint(
    directory: Path,
    checkpoint: StageEditSourcePairCheckpoint,
) -> Path:
    if not isinstance(checkpoint, StageEditSourcePairCheckpoint):
        raise StageEditSourcePairInputError("checkpoint is invalid")
    destination = Path(directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME
    _publish_checkpoint_no_replace(path, _checkpoint_json_bytes(checkpoint))
    return path.resolve()


def load_stage_edit_source_pair_checkpoint(
    directory: Path,
    *,
    expected_corpus_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> tuple[StageEditSourcePairScorer, StageEditSourcePairCheckpoint]:
    path = Path(directory) / STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME
    if path.is_symlink():
        raise StageEditSourcePairCheckpointError(
            "source-pair checkpoint must be a regular non-symlink file"
        )
    if not path.is_file():
        raise StageEditSourcePairCheckpointError(
            "source-pair checkpoint is missing"
        )
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=True) as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise StageEditSourcePairInputError(
                    "source-pair checkpoint must be a regular file"
                )
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise StageEditSourcePairInputError(
                "source-pair checkpoint root must be an object"
            )
        checkpoint = StageEditSourcePairCheckpoint.from_dict(payload)
        if expected_corpus_sha256 is not None and checkpoint.corpus_sha256 != _sha256(
            expected_corpus_sha256, "expected_corpus_sha256"
        ):
            raise StageEditSourcePairInputError(
                "source-pair checkpoint corpus digest does not match"
            )
        if (
            expected_checkpoint_sha256 is not None
            and checkpoint.checkpoint_sha256
            != _sha256(
                expected_checkpoint_sha256, "expected_checkpoint_sha256"
            )
        ):
            raise StageEditSourcePairInputError(
                "source-pair checkpoint digest does not match expected digest"
            )
    except (
        OSError,
        json.JSONDecodeError,
        StageEditSourcePairInputError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        raise StageEditSourcePairCheckpointError(
            f"source-pair checkpoint is invalid: {exc}"
        ) from exc
    return checkpoint.scorer, checkpoint


__all__ = [
    "STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME",
    "STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES",
    "STAGE_EDIT_SOURCE_PAIR_FEATURE_SCHEMA_VERSION",
    "STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT",
    "STAGE_EDIT_SOURCE_PAIR_SCORER_FORMAT_VERSION",
    "StageEditSourcePairCandidate",
    "StageEditSourcePairCandidateScore",
    "StageEditSourcePairCheckpoint",
    "StageEditSourcePairCheckpointError",
    "StageEditSourcePairFeatureVector",
    "StageEditSourcePairInputError",
    "StageEditSourcePairScorer",
    "StageEditSourcePairScorerError",
    "StageEditSourcePairTrainingGroup",
    "StageEditSourcePairTrainingResult",
    "load_stage_edit_source_pair_checkpoint",
    "materialize_stage_edit_source_pair_features",
    "new_stage_edit_source_pair_checkpoint",
    "save_stage_edit_source_pair_checkpoint",
    "stage_edit_source_pair_reference_balanced_loss",
    "train_stage_edit_source_pair_scorer",
]
