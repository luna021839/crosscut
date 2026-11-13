"""MLX learned reviewer with fail-closed production evidence.

The model estimates editorial quality and localized issue causes from decoded
proxy evidence.  It is not a replacement for technical validation: any
decoded technical error is an unconditional production failure regardless of
the learned score.  This module is offline and is not imported by the
production pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

try:  # Optional so corpus/readiness inspection remains portable.
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError as exc:  # pragma: no cover - non-Apple environments.
    mx = None
    nn = None
    optim = None
    _MLX_IMPORT_ERROR: ImportError | None = exc
else:
    _MLX_IMPORT_ERROR = None


LEARNED_REVIEWER_SCHEMA_VERSION = 2
LEARNED_REVIEWER_CORPUS_VERSION = 1
LEARNED_REVIEWER_CHECKPOINT_VERSION = 1
LEARNED_REVIEWER_REPORT_VERSION = 1
LEARNED_REVIEWER_REPORT_FILENAME = "training-report.json"
LEARNED_REVIEWER_RUN_KIND = "stage_edit_learned_reviewer_production_training"
LEARNED_REVIEWER_CHECKPOINT_SCOPE = "production_learned_reviewer"

HUMAN_REFERENCE = "human_reference_positive"
SYNTHETIC_SHIFTED = "synthetic_shifted_negative"
SYNTHETIC_WORSENED = "synthetic_worsened_negative"
SYNTHETIC_TECHNICAL = "synthetic_technical_negative"
CANDIDATE = "candidate"
EXAMPLE_KINDS = frozenset(
    {
        HUMAN_REFERENCE,
        SYNTHETIC_SHIFTED,
        SYNTHETIC_WORSENED,
        SYNTHETIC_TECHNICAL,
        CANDIDATE,
    }
)

MINIMUM_PRODUCTION_SONG_COUNT = 3
MINIMUM_PRODUCTION_TRAIN_SONG_COUNT = 2
MINIMUM_PRODUCTION_HELDOUT_SONG_COUNT = 1
MINIMUM_PRODUCTION_HUMAN_REFERENCE_COUNT = 3
MINIMUM_PRODUCTION_DENSE_EXAMPLE_COUNT = 6
MINIMUM_PRODUCTION_PAIRWISE_COUNT = 6
MINIMUM_PRODUCTION_LISTWISE_GROUP_COUNT = 3
MINIMUM_HELDOUT_HUMAN_WORSENED_AUROC = 0.80
MINIMUM_HELDOUT_MEAN_RANK_MARGIN = 0.10
MAXIMUM_HELDOUT_BRIER_CALIBRATION_ERROR = 0.20

LEARNED_REVIEWER_FEATURE_MATERIALIZER_CONTRACT = {
    "contract_version": 1,
    "source_schema": "decoded_output_proxy_domain_aware_quality_review_v3",
    "normalization_domain": "all_observed_features_in_zero_to_one",
    "boundary_mapping": {
        "visual_boundary_score": "weighted_score_div_100",
        "visual_discontinuity": "one_minus_weighted_score_div_100",
        "face_match_confidence": "one_minus_mean_face_scale_position_when_comparable",
        "face_scale_mismatch": "face_scale_discontinuity_or_missing",
        "face_center_mismatch": "face_position_discontinuity_or_missing",
        "silhouette_mismatch": "max_scale_position_when_available",
        "pose_mismatch": "missing_not_available_from_proxy",
        "motion_discontinuity": "motion_direction_discontinuity",
        "camera_motion_mismatch": "motion_direction_discontinuity",
        "color_delta": "color_shock",
        "exposure_delta": "exposure_discontinuity",
        "sharpness_normalized": "mean_boundary_frame_sharpness",
        "technical_black_frame": "temporal_black_frame_offsets_nonempty",
        "technical_duplicate_frame": "temporal_duplicate_or_freeze_detected",
        "technical_decode_error": "temporal_or_frame_metrics_missing",
        "technical_clock_error": "technical_source_clock_reason_present",
    },
    "timeline_mapping": {
        "cadence_score": "triangular_distance_from_15_5_switches_per_minute",
        "source_switch_rate_normalized": "switches_per_minute_div_30_clamped",
        "source_entropy": "duration_share_entropy_normalized_by_used_sources",
        "maximum_source_share": "maximum_duration_share",
        "unique_source_fraction": "used_sources_div_declared_sources",
        "median_shot_duration_normalized": "median_duration_div_8_clamped",
        "short_shot_fraction": "fraction_duration_below_1_second",
        "long_shot_fraction": "fraction_duration_above_7_2_seconds",
        "section_energy_alignment": "missing_not_available_from_proxy",
        "section_cut_density_alignment": "missing_not_available_from_proxy",
        "aba_rate": "aba_triplets_div_possible_triplets",
        "recent_cut_history_density": "switches_per_minute_div_30_clamped",
    },
    "technical_missing_policy": "missing_technical_evidence_is_hard_failure",
}
LEARNED_REVIEWER_FEATURE_MATERIALIZER_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        LEARNED_REVIEWER_FEATURE_MATERIALIZER_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def _production_requirements() -> dict[str, int]:
    return {
        "song_count": MINIMUM_PRODUCTION_SONG_COUNT,
        "training_song_count": MINIMUM_PRODUCTION_TRAIN_SONG_COUNT,
        "heldout_song_count": MINIMUM_PRODUCTION_HELDOUT_SONG_COUNT,
        "human_reference_count": MINIMUM_PRODUCTION_HUMAN_REFERENCE_COUNT,
        "human_reference_song_count": MINIMUM_PRODUCTION_SONG_COUNT,
        "synthetic_shifted_count": MINIMUM_PRODUCTION_SONG_COUNT,
        "synthetic_shifted_song_count": MINIMUM_PRODUCTION_SONG_COUNT,
        "synthetic_worsened_count": MINIMUM_PRODUCTION_SONG_COUNT,
        "synthetic_worsened_song_count": MINIMUM_PRODUCTION_SONG_COUNT,
        "synthetic_technical_count": MINIMUM_PRODUCTION_SONG_COUNT,
        "synthetic_technical_song_count": MINIMUM_PRODUCTION_SONG_COUNT,
        "dense_example_count": MINIMUM_PRODUCTION_DENSE_EXAMPLE_COUNT,
        "dense_example_song_count": MINIMUM_PRODUCTION_SONG_COUNT,
        "pairwise_preference_count": MINIMUM_PRODUCTION_PAIRWISE_COUNT,
        "listwise_group_count": MINIMUM_PRODUCTION_LISTWISE_GROUP_COUNT,
        "ranked_candidate_count": MINIMUM_PRODUCTION_SONG_COUNT,
        "ranked_candidate_song_count": MINIMUM_PRODUCTION_SONG_COUNT,
    }


class LearnedReviewerError(RuntimeError):
    """Base learned-reviewer failure."""


class LearnedReviewerInputError(LearnedReviewerError, ValueError):
    """Malformed corpus, proxy evidence, or training input."""


class LearnedReviewerNotReadyError(LearnedReviewerError):
    """Production checkpoint evidence is unavailable or rejected."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise LearnedReviewerInputError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise LearnedReviewerInputError(f"{name} must be finite")
    return number


def _probability(value: Any, name: str) -> float:
    number = _finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise LearnedReviewerInputError(f"{name} must lie in 0...1")
    return number


def _name(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise LearnedReviewerInputError(f"{field_name} must be a string")
    result = value.strip()
    if not result:
        raise LearnedReviewerInputError(f"{field_name} cannot be blank")
    return result


def _names(values: Sequence[Any], field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise LearnedReviewerInputError(f"{field_name} must be an array")
    result = tuple(_name(value, field_name) for value in values)
    if not allow_empty and not result:
        raise LearnedReviewerInputError(f"{field_name} cannot be empty")
    if len(set(result)) != len(result):
        raise LearnedReviewerInputError(f"{field_name} must contain unique names")
    return result


def _optional_features(values: Sequence[float | None], name: str) -> tuple[float | None, ...]:
    output: list[float | None] = []
    for index, value in enumerate(values):
        if value is None:
            output.append(None)
            continue
        number = _finite(value, f"{name}[{index}]")
        if not -1.0 <= number <= 1.0:
            raise LearnedReviewerInputError(
                f"{name}[{index}] must be normalized to -1...1"
            )
        output.append(number)
    return tuple(output)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def _strict_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise LearnedReviewerInputError(
            f"{name} keys differ; missing={missing}, extra={extra}"
        )


@dataclass(frozen=True, slots=True)
class LearnedReviewerFeatureSchema:
    schema_version: int
    feature_materializer_contract_sha256: str
    boundary_feature_names: tuple[str, ...]
    timeline_feature_names: tuple[str, ...]
    cause_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != LEARNED_REVIEWER_SCHEMA_VERSION
        ):
            raise LearnedReviewerInputError("unsupported learned-reviewer schema")
        contract_sha = self.feature_materializer_contract_sha256
        if (
            not isinstance(contract_sha, str)
            or len(contract_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in contract_sha
            )
        ):
            raise LearnedReviewerInputError(
                "feature materializer contract must be a lowercase SHA-256"
            )
        object.__setattr__(
            self,
            "boundary_feature_names",
            _names(self.boundary_feature_names, "boundary_feature_names"),
        )
        object.__setattr__(
            self,
            "timeline_feature_names",
            _names(self.timeline_feature_names, "timeline_feature_names"),
        )
        object.__setattr__(
            self,
            "cause_labels",
            _names(self.cause_labels, "cause_labels"),
        )
        if "technical_integrity" not in self.cause_labels:
            raise LearnedReviewerInputError(
                "cause labels must include technical_integrity"
            )
        if not any(name.startswith("technical_") for name in self.boundary_feature_names):
            raise LearnedReviewerInputError(
                "boundary schema needs explicit technical features"
            )

    @property
    def boundary_dimension(self) -> int:
        return len(self.boundary_feature_names)

    @property
    def encoded_boundary_dimension(self) -> int:
        return 2 * self.boundary_dimension

    @property
    def timeline_dimension(self) -> int:
        return len(self.timeline_feature_names)

    @property
    def encoded_timeline_dimension(self) -> int:
        return 2 * self.timeline_dimension

    @property
    def technical_feature_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, name in enumerate(self.boundary_feature_names)
            if name.startswith("technical_")
        )

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LearnedReviewerFeatureSchema":
        _strict_keys(
            payload,
            {
                "schema_version",
                "feature_materializer_contract_sha256",
                "boundary_feature_names",
                "timeline_feature_names",
                "cause_labels",
            },
            "learned-reviewer feature schema",
        )
        return cls(
            schema_version=_strict_int(
                payload["schema_version"], "schema_version", minimum=1
            ),
            feature_materializer_contract_sha256=payload[
                "feature_materializer_contract_sha256"
            ],
            boundary_feature_names=tuple(payload["boundary_feature_names"]),
            timeline_feature_names=tuple(payload["timeline_feature_names"]),
            cause_labels=tuple(payload["cause_labels"]),
        )


DEFAULT_LEARNED_REVIEWER_SCHEMA = LearnedReviewerFeatureSchema(
    schema_version=LEARNED_REVIEWER_SCHEMA_VERSION,
    feature_materializer_contract_sha256=(
        LEARNED_REVIEWER_FEATURE_MATERIALIZER_CONTRACT_SHA256
    ),
    boundary_feature_names=(
        "visual_boundary_score",
        "visual_discontinuity",
        "face_match_confidence",
        "face_scale_mismatch",
        "face_center_mismatch",
        "silhouette_mismatch",
        "pose_mismatch",
        "motion_discontinuity",
        "camera_motion_mismatch",
        "color_delta",
        "exposure_delta",
        "sharpness_normalized",
        "technical_black_frame",
        "technical_duplicate_frame",
        "technical_decode_error",
        "technical_clock_error",
    ),
    timeline_feature_names=(
        "cadence_score",
        "source_switch_rate_normalized",
        "source_entropy",
        "maximum_source_share",
        "unique_source_fraction",
        "median_shot_duration_normalized",
        "short_shot_fraction",
        "long_shot_fraction",
        "section_energy_alignment",
        "section_cut_density_alignment",
        "aba_rate",
        "recent_cut_history_density",
    ),
    cause_labels=(
        "visual_discontinuity",
        "face_mismatch",
        "silhouette_pose_mismatch",
        "motion_mismatch",
        "color_mismatch",
        "cadence_source_distribution",
        "technical_integrity",
    ),
)


@dataclass(frozen=True, slots=True)
class LearnedReviewerBoundary:
    boundary_id: str
    at_seconds: float
    features: tuple[float | None, ...]
    issue_target: float | None = None
    cause_targets: tuple[float | None, ...] | None = None
    technical_error_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary_id", _name(self.boundary_id, "boundary_id"))
        at_seconds = _finite(self.at_seconds, "boundary at_seconds")
        if at_seconds < 0.0:
            raise LearnedReviewerInputError("boundary at_seconds cannot be negative")
        object.__setattr__(self, "at_seconds", at_seconds)
        object.__setattr__(
            self,
            "features",
            _optional_features(self.features, "boundary features"),
        )
        if self.issue_target is not None:
            object.__setattr__(
                self,
                "issue_target",
                _probability(self.issue_target, "boundary issue_target"),
            )
        if self.cause_targets is not None:
            targets = tuple(
                None
                if value is None
                else _probability(value, f"cause_targets[{index}]")
                for index, value in enumerate(self.cause_targets)
            )
            object.__setattr__(self, "cause_targets", targets)
        object.__setattr__(
            self,
            "technical_error_codes",
            tuple(
                sorted(
                    _names(
                        self.technical_error_codes,
                        "technical_error_codes",
                        allow_empty=True,
                    )
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "at_seconds": self.at_seconds,
            "features": list(self.features),
            "issue_target": self.issue_target,
            "cause_targets": (
                list(self.cause_targets) if self.cause_targets is not None else None
            ),
            "technical_error_codes": list(self.technical_error_codes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LearnedReviewerBoundary":
        _strict_keys(
            payload,
            {
                "boundary_id",
                "at_seconds",
                "features",
                "issue_target",
                "cause_targets",
                "technical_error_codes",
            },
            "learned-reviewer boundary",
        )
        causes = payload["cause_targets"]
        return cls(
            boundary_id=payload["boundary_id"],
            at_seconds=payload["at_seconds"],
            features=tuple(payload["features"]),
            issue_target=payload["issue_target"],
            cause_targets=tuple(causes) if causes is not None else None,
            technical_error_codes=tuple(payload["technical_error_codes"]),
        )


@dataclass(frozen=True, slots=True)
class LearnedReviewerExample:
    example_id: str
    song_id: str
    reference_id: str
    example_kind: str
    boundaries: tuple[LearnedReviewerBoundary, ...]
    timeline_features: tuple[float | None, ...]
    quality_target: float | None
    parent_human_example_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("example_id", "song_id", "reference_id"):
            object.__setattr__(self, name, _name(getattr(self, name), name))
        kind = str(self.example_kind).strip()
        if kind not in EXAMPLE_KINDS:
            raise LearnedReviewerInputError("unknown learned-reviewer example_kind")
        object.__setattr__(self, "example_kind", kind)
        boundaries = tuple(self.boundaries)
        if not boundaries or any(
            not isinstance(value, LearnedReviewerBoundary) for value in boundaries
        ):
            raise LearnedReviewerInputError("reviewer example needs boundaries")
        if any(
            right.at_seconds <= left.at_seconds
            for left, right in zip(boundaries, boundaries[1:])
        ):
            raise LearnedReviewerInputError("boundary times must increase")
        ids = tuple(value.boundary_id for value in boundaries)
        if len(set(ids)) != len(ids):
            raise LearnedReviewerInputError("boundary ids must be unique per example")
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(
            self,
            "timeline_features",
            _optional_features(self.timeline_features, "timeline features"),
        )
        if self.quality_target is not None:
            object.__setattr__(
                self,
                "quality_target",
                _probability(self.quality_target, "quality_target"),
            )
        parent = (
            _name(self.parent_human_example_id, "parent_human_example_id")
            if self.parent_human_example_id is not None
            else None
        )
        if kind in {SYNTHETIC_SHIFTED, SYNTHETIC_WORSENED, SYNTHETIC_TECHNICAL}:
            if parent is None or self.quality_target is None:
                raise LearnedReviewerInputError(
                    "synthetic negatives need parent human id and quality target"
                )
        elif parent is not None:
            raise LearnedReviewerInputError(
                "only synthetic negatives may name a parent human example"
            )
        if kind == HUMAN_REFERENCE and (
            self.quality_target is None or self.quality_target < 0.5
        ):
            raise LearnedReviewerInputError(
                "human reference positives require quality_target >= 0.5"
            )
        object.__setattr__(self, "parent_human_example_id", parent)

    @property
    def dense_boundary_supervision(self) -> bool:
        return all(
            boundary.issue_target is not None
            and boundary.cause_targets is not None
            and all(value is not None for value in boundary.cause_targets)
            for boundary in self.boundaries
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "song_id": self.song_id,
            "reference_id": self.reference_id,
            "example_kind": self.example_kind,
            "boundaries": [value.to_dict() for value in self.boundaries],
            "timeline_features": list(self.timeline_features),
            "quality_target": self.quality_target,
            "parent_human_example_id": self.parent_human_example_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LearnedReviewerExample":
        _strict_keys(
            payload,
            {
                "example_id",
                "song_id",
                "reference_id",
                "example_kind",
                "boundaries",
                "timeline_features",
                "quality_target",
                "parent_human_example_id",
            },
            "learned-reviewer example",
        )
        return cls(
            example_id=payload["example_id"],
            song_id=payload["song_id"],
            reference_id=payload["reference_id"],
            example_kind=payload["example_kind"],
            boundaries=tuple(
                LearnedReviewerBoundary.from_dict(value)
                for value in payload["boundaries"]
            ),
            timeline_features=tuple(payload["timeline_features"]),
            quality_target=payload["quality_target"],
            parent_human_example_id=payload["parent_human_example_id"],
        )


@dataclass(frozen=True, slots=True)
class LearnedReviewerPairwisePreference:
    preference_id: str
    group_id: str
    preferred_example_id: str
    rejected_example_id: str
    margin: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "preference_id",
            "group_id",
            "preferred_example_id",
            "rejected_example_id",
        ):
            object.__setattr__(self, name, _name(getattr(self, name), name))
        if self.preferred_example_id == self.rejected_example_id:
            raise LearnedReviewerInputError("pairwise preference needs two examples")
        margin = _finite(self.margin, "pairwise margin")
        if margin < 0.0 or margin > 2.0:
            raise LearnedReviewerInputError("pairwise margin is outside 0...2")
        object.__setattr__(self, "margin", margin)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LearnedReviewerPairwisePreference":
        _strict_keys(
            payload,
            {
                "preference_id",
                "group_id",
                "preferred_example_id",
                "rejected_example_id",
                "margin",
            },
            "pairwise preference",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class LearnedReviewerListwiseGroup:
    group_id: str
    ordered_example_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _name(self.group_id, "group_id"))
        ordered = _names(self.ordered_example_ids, "ordered_example_ids")
        if len(ordered) < 2:
            raise LearnedReviewerInputError("listwise group needs at least two items")
        object.__setattr__(self, "ordered_example_ids", ordered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "ordered_example_ids": list(self.ordered_example_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LearnedReviewerListwiseGroup":
        _strict_keys(
            payload,
            {"group_id", "ordered_example_ids"},
            "listwise group",
        )
        return cls(payload["group_id"], tuple(payload["ordered_example_ids"]))


@dataclass(frozen=True, slots=True)
class LearnedReviewerCorpus:
    corpus_version: int
    corpus_id: str
    feature_schema: LearnedReviewerFeatureSchema
    examples: tuple[LearnedReviewerExample, ...]
    pairwise_preferences: tuple[LearnedReviewerPairwisePreference, ...]
    listwise_groups: tuple[LearnedReviewerListwiseGroup, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.corpus_version, bool)
            or not isinstance(self.corpus_version, int)
            or self.corpus_version != LEARNED_REVIEWER_CORPUS_VERSION
        ):
            raise LearnedReviewerInputError("unsupported reviewer corpus version")
        object.__setattr__(self, "corpus_id", _name(self.corpus_id, "corpus_id"))
        if not isinstance(self.feature_schema, LearnedReviewerFeatureSchema):
            raise LearnedReviewerInputError("corpus feature_schema is invalid")
        examples = tuple(sorted(self.examples, key=lambda value: value.example_id))
        if not examples:
            raise LearnedReviewerInputError("reviewer corpus cannot be empty")
        if any(not isinstance(value, LearnedReviewerExample) for value in examples):
            raise LearnedReviewerInputError("corpus examples have invalid types")
        ids = tuple(value.example_id for value in examples)
        if len(set(ids)) != len(ids):
            raise LearnedReviewerInputError("corpus example ids must be unique")
        by_id = {value.example_id: value for value in examples}
        for example in examples:
            if len(example.timeline_features) != self.feature_schema.timeline_dimension:
                raise LearnedReviewerInputError("timeline feature dimension mismatch")
            for boundary in example.boundaries:
                if len(boundary.features) != self.feature_schema.boundary_dimension:
                    raise LearnedReviewerInputError("boundary feature dimension mismatch")
                if boundary.cause_targets is not None and len(
                    boundary.cause_targets
                ) != len(self.feature_schema.cause_labels):
                    raise LearnedReviewerInputError("cause target dimension mismatch")
        preferences = tuple(
            sorted(self.pairwise_preferences, key=lambda value: value.preference_id)
        )
        pref_ids = tuple(value.preference_id for value in preferences)
        if len(set(pref_ids)) != len(pref_ids):
            raise LearnedReviewerInputError("pairwise preference ids must be unique")
        for preference in preferences:
            if (
                preference.preferred_example_id not in by_id
                or preference.rejected_example_id not in by_id
            ):
                raise LearnedReviewerInputError("pairwise preference references unknown example")
            preferred = by_id[preference.preferred_example_id]
            rejected = by_id[preference.rejected_example_id]
            if preferred.song_id != rejected.song_id:
                raise LearnedReviewerInputError(
                    "pairwise preference cannot cross song-level split groups"
                )
        listwise = tuple(sorted(self.listwise_groups, key=lambda value: value.group_id))
        group_ids = tuple(value.group_id for value in listwise)
        if len(set(group_ids)) != len(group_ids):
            raise LearnedReviewerInputError("listwise group ids must be unique")
        for group in listwise:
            if any(value not in by_id for value in group.ordered_example_ids):
                raise LearnedReviewerInputError("listwise group references unknown example")
            songs = {by_id[value].song_id for value in group.ordered_example_ids}
            if len(songs) != 1:
                raise LearnedReviewerInputError(
                    "listwise group cannot cross song-level split groups"
                )
        for example in examples:
            if example.example_kind in {
                SYNTHETIC_SHIFTED,
                SYNTHETIC_WORSENED,
                SYNTHETIC_TECHNICAL,
            }:
                parent = by_id.get(example.parent_human_example_id or "")
                if parent is None or parent.example_kind != HUMAN_REFERENCE:
                    raise LearnedReviewerInputError(
                        "synthetic parent must be a corpus human reference"
                    )
                if parent.song_id != example.song_id:
                    raise LearnedReviewerInputError(
                        "synthetic parent must belong to the same song"
                    )
                if parent.quality_target is None or example.quality_target is None or not (
                    parent.quality_target > example.quality_target
                ):
                    raise LearnedReviewerInputError(
                        "synthetic negative quality must be below its human parent"
                    )
            if example.example_kind == SYNTHETIC_TECHNICAL and not _example_has_technical_error(
                example, self.feature_schema
            ):
                raise LearnedReviewerInputError(
                    "synthetic technical negative needs explicit technical evidence"
                )
        ranked_ids = {
            value.preferred_example_id for value in preferences
        } | {value.rejected_example_id for value in preferences} | {
            item for group in listwise for item in group.ordered_example_ids
        }
        for example in examples:
            if example.example_kind == CANDIDATE and example.example_id not in ranked_ids:
                raise LearnedReviewerInputError(
                    "candidate examples need pairwise or listwise supervision"
                )
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "pairwise_preferences", preferences)
        object.__setattr__(self, "listwise_groups", listwise)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @property
    def song_ids(self) -> tuple[str, ...]:
        return tuple(sorted({value.song_id for value in self.examples}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_version": self.corpus_version,
            "corpus_id": self.corpus_id,
            "feature_schema": self.feature_schema.to_dict(),
            "examples": [value.to_dict() for value in self.examples],
            "pairwise_preferences": [
                value.to_dict() for value in self.pairwise_preferences
            ],
            "listwise_groups": [value.to_dict() for value in self.listwise_groups],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LearnedReviewerCorpus":
        _strict_keys(
            payload,
            {
                "corpus_version",
                "corpus_id",
                "feature_schema",
                "examples",
                "pairwise_preferences",
                "listwise_groups",
            },
            "learned-reviewer corpus",
        )
        return cls(
            corpus_version=_strict_int(
                payload["corpus_version"], "corpus_version", minimum=1
            ),
            corpus_id=payload["corpus_id"],
            feature_schema=LearnedReviewerFeatureSchema.from_dict(
                payload["feature_schema"]
            ),
            examples=tuple(
                LearnedReviewerExample.from_dict(value) for value in payload["examples"]
            ),
            pairwise_preferences=tuple(
                LearnedReviewerPairwisePreference.from_dict(value)
                for value in payload["pairwise_preferences"]
            ),
            listwise_groups=tuple(
                LearnedReviewerListwiseGroup.from_dict(value)
                for value in payload["listwise_groups"]
            ),
        )


def _example_has_technical_error(
    example: LearnedReviewerExample,
    schema: LearnedReviewerFeatureSchema,
) -> bool:
    return any(_technical_codes(boundary, schema) for boundary in example.boundaries)


@dataclass(frozen=True, slots=True)
class LearnedReviewerSplit:
    train_example_ids: tuple[str, ...]
    heldout_example_ids: tuple[str, ...]
    training_song_ids: tuple[str, ...]
    heldout_song_ids: tuple[str, ...]
    split_level: str = "song"

    def __post_init__(self) -> None:
        if self.split_level != "song":
            raise LearnedReviewerInputError("reviewer split must be song-level")
        if set(self.train_example_ids).intersection(self.heldout_example_ids):
            raise LearnedReviewerInputError("train/heldout examples overlap")
        if set(self.training_song_ids).intersection(self.heldout_song_ids):
            raise LearnedReviewerInputError("train/heldout songs overlap")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_learned_reviewer_corpus(
    corpus: LearnedReviewerCorpus,
    *,
    heldout_fraction: float = 0.25,
    seed: int = 20260824,
) -> LearnedReviewerSplit:
    fraction = _finite(heldout_fraction, "heldout_fraction")
    if not 0.0 < fraction < 1.0:
        raise LearnedReviewerInputError("heldout_fraction must lie in (0, 1)")
    songs = tuple(
        sorted(
            corpus.song_ids,
            key=lambda song: hashlib.sha256(f"{seed}:{song}".encode()).hexdigest(),
        )
    )
    if len(songs) < 2:
        heldout_songs: tuple[str, ...] = ()
        training_songs = songs
    else:
        heldout_count = max(1, int(round(len(songs) * fraction)))
        heldout_count = min(heldout_count, len(songs) - 1)
        heldout_songs = tuple(sorted(songs[:heldout_count]))
        training_songs = tuple(sorted(songs[heldout_count:]))
    heldout_set = set(heldout_songs)
    train_ids = tuple(
        value.example_id for value in corpus.examples if value.song_id not in heldout_set
    )
    heldout_ids = tuple(
        value.example_id for value in corpus.examples if value.song_id in heldout_set
    )
    return LearnedReviewerSplit(
        train_example_ids=train_ids,
        heldout_example_ids=heldout_ids,
        training_song_ids=training_songs,
        heldout_song_ids=heldout_songs,
    )


@dataclass(frozen=True, slots=True)
class LearnedReviewerConfig:
    feature_schema_fingerprint: str
    boundary_input_dimension: int
    timeline_input_dimension: int
    cause_count: int
    max_boundaries: int = 128
    model_dimension: int = 32
    num_layers: int = 2
    num_heads: int = 4
    feed_forward_dimension: int = 64
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.feature_schema_fingerprint, str)
            or len(self.feature_schema_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.feature_schema_fingerprint
            )
        ):
            raise LearnedReviewerInputError("feature schema fingerprint must be SHA-256")
        for name in (
            "boundary_input_dimension",
            "timeline_input_dimension",
            "cause_count",
            "max_boundaries",
            "model_dimension",
            "num_layers",
            "num_heads",
            "feed_forward_dimension",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise LearnedReviewerInputError(f"{name} must be a positive integer")
        if self.model_dimension % self.num_heads:
            raise LearnedReviewerInputError("model_dimension must divide num_heads")
        dropout = _finite(self.dropout, "dropout")
        if not 0.0 <= dropout < 1.0:
            raise LearnedReviewerInputError("dropout must lie in [0, 1)")
        object.__setattr__(self, "dropout", dropout)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LearnedReviewerConfig":
        _strict_keys(
            payload,
            {
                "feature_schema_fingerprint",
                "boundary_input_dimension",
                "timeline_input_dimension",
                "cause_count",
                "max_boundaries",
                "model_dimension",
                "num_layers",
                "num_heads",
                "feed_forward_dimension",
                "dropout",
            },
            "learned-reviewer config",
        )
        return cls(**payload)


def default_learned_reviewer_config(
    schema: LearnedReviewerFeatureSchema = DEFAULT_LEARNED_REVIEWER_SCHEMA,
) -> LearnedReviewerConfig:
    return LearnedReviewerConfig(
        feature_schema_fingerprint=schema.fingerprint,
        boundary_input_dimension=schema.encoded_boundary_dimension,
        timeline_input_dimension=schema.encoded_timeline_dimension,
        cause_count=len(schema.cause_labels),
    )


@dataclass(frozen=True, slots=True)
class LearnedReviewerTensorBatch:
    boundary_features: Any
    boundary_observed_mask: Any
    sequence_mask: Any
    timeline_features: Any
    timeline_observed_mask: Any
    technical_boundary_mask: Any
    technical_example_mask: Any
    example_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LearnedReviewerSupervision:
    quality_targets: Any
    quality_mask: Any
    boundary_issue_targets: Any
    boundary_issue_mask: Any
    cause_targets: Any
    cause_mask: Any
    pairwise_preferred_indices: Any
    pairwise_rejected_indices: Any
    pairwise_margins: Any
    listwise_preferred_indices: Any
    listwise_rejected_indices: Any


def _require_mlx() -> None:
    if mx is None or nn is None or optim is None:
        raise RuntimeError(
            "MLX is required for learned reviewer model operations"
        ) from _MLX_IMPORT_ERROR


def tensorize_learned_reviewer_corpus(
    corpus: LearnedReviewerCorpus,
    example_ids: Sequence[str],
    config: LearnedReviewerConfig,
    *,
    device: Any = None,
) -> tuple[LearnedReviewerTensorBatch, LearnedReviewerSupervision]:
    _require_mlx()
    schema = corpus.feature_schema
    if config.feature_schema_fingerprint != schema.fingerprint:
        raise LearnedReviewerInputError("model config and corpus schema differ")
    if (
        config.boundary_input_dimension != schema.encoded_boundary_dimension
        or config.timeline_input_dimension != schema.encoded_timeline_dimension
        or config.cause_count != len(schema.cause_labels)
    ):
        raise LearnedReviewerInputError("model input dimensions differ from schema")
    requested = tuple(example_ids)
    if not requested or len(set(requested)) != len(requested):
        raise LearnedReviewerInputError("tensorization example ids must be unique")
    by_id = {value.example_id: value for value in corpus.examples}
    if any(value not in by_id for value in requested):
        raise LearnedReviewerInputError("tensorization references unknown example")
    examples = tuple(by_id[value] for value in requested)
    maximum = max(len(value.boundaries) for value in examples)
    if maximum > config.max_boundaries:
        raise LearnedReviewerInputError("example exceeds max_boundaries")
    batch_size = len(examples)
    boundary_values = np.zeros(
        (batch_size, maximum, schema.boundary_dimension), dtype=np.float32
    )
    boundary_observed = np.zeros_like(boundary_values, dtype=bool)
    sequence_mask = np.zeros((batch_size, maximum), dtype=bool)
    timeline_values = np.zeros(
        (batch_size, schema.timeline_dimension), dtype=np.float32
    )
    timeline_observed = np.zeros_like(timeline_values, dtype=bool)
    technical_boundary = np.zeros((batch_size, maximum), dtype=bool)
    quality_targets = np.zeros((batch_size,), dtype=np.float32)
    quality_mask = np.zeros((batch_size,), dtype=bool)
    issue_targets = np.zeros((batch_size, maximum), dtype=np.float32)
    issue_mask = np.zeros((batch_size, maximum), dtype=bool)
    cause_targets = np.zeros(
        (batch_size, maximum, len(schema.cause_labels)), dtype=np.float32
    )
    cause_mask = np.zeros_like(cause_targets, dtype=bool)
    for batch_index, example in enumerate(examples):
        sequence_mask[batch_index, : len(example.boundaries)] = True
        for feature_index, value in enumerate(example.timeline_features):
            if value is not None:
                timeline_values[batch_index, feature_index] = value
                timeline_observed[batch_index, feature_index] = True
        if example.quality_target is not None:
            quality_targets[batch_index] = example.quality_target
            quality_mask[batch_index] = True
        for boundary_index, boundary in enumerate(example.boundaries):
            for feature_index, value in enumerate(boundary.features):
                if value is not None:
                    boundary_values[batch_index, boundary_index, feature_index] = value
                    boundary_observed[batch_index, boundary_index, feature_index] = True
            technical_boundary[batch_index, boundary_index] = bool(
                _technical_codes(boundary, schema)
            )
            if boundary.issue_target is not None:
                issue_targets[batch_index, boundary_index] = boundary.issue_target
                issue_mask[batch_index, boundary_index] = True
            if boundary.cause_targets is not None:
                for cause_index, value in enumerate(boundary.cause_targets):
                    if value is not None:
                        cause_targets[batch_index, boundary_index, cause_index] = value
                        cause_mask[batch_index, boundary_index, cause_index] = True
    encoded_boundary = np.concatenate(
        (boundary_values, boundary_observed.astype(np.float32)), axis=-1
    )
    encoded_timeline = np.concatenate(
        (timeline_values, timeline_observed.astype(np.float32)), axis=-1
    )
    local_index = {example_id: index for index, example_id in enumerate(requested)}
    pairwise = tuple(
        value
        for value in corpus.pairwise_preferences
        if value.preferred_example_id in local_index and value.rejected_example_id in local_index
    )
    listwise_pairs: list[tuple[int, int]] = []
    for group in corpus.listwise_groups:
        if all(value in local_index for value in group.ordered_example_ids):
            for left in range(len(group.ordered_example_ids)):
                for right in range(left + 1, len(group.ordered_example_ids)):
                    listwise_pairs.append(
                        (
                            local_index[group.ordered_example_ids[left]],
                            local_index[group.ordered_example_ids[right]],
                        )
                    )
    active_device = device if device is not None else mx.default_device()
    with mx.stream(active_device):
        batch = LearnedReviewerTensorBatch(
            boundary_features=mx.array(encoded_boundary),
            boundary_observed_mask=mx.array(boundary_observed),
            sequence_mask=mx.array(sequence_mask),
            timeline_features=mx.array(encoded_timeline),
            timeline_observed_mask=mx.array(timeline_observed),
            technical_boundary_mask=mx.array(technical_boundary),
            technical_example_mask=mx.array(np.any(technical_boundary, axis=1)),
            example_ids=requested,
        )
        supervision = LearnedReviewerSupervision(
            quality_targets=mx.array(quality_targets),
            quality_mask=mx.array(quality_mask),
            boundary_issue_targets=mx.array(issue_targets),
            boundary_issue_mask=mx.array(issue_mask),
            cause_targets=mx.array(cause_targets),
            cause_mask=mx.array(cause_mask),
            pairwise_preferred_indices=mx.array(
                np.asarray(
                    [local_index[value.preferred_example_id] for value in pairwise],
                    dtype=np.int32,
                )
            ),
            pairwise_rejected_indices=mx.array(
                np.asarray(
                    [local_index[value.rejected_example_id] for value in pairwise],
                    dtype=np.int32,
                )
            ),
            pairwise_margins=mx.array(
                np.asarray([value.margin for value in pairwise], dtype=np.float32)
            ),
            listwise_preferred_indices=mx.array(
                np.asarray([value[0] for value in listwise_pairs], dtype=np.int32)
            ),
            listwise_rejected_indices=mx.array(
                np.asarray([value[1] for value in listwise_pairs], dtype=np.int32)
            ),
        )
    return batch, supervision


@dataclass(frozen=True, slots=True)
class LearnedReviewerModelOutput:
    global_quality_logits: Any
    boundary_issue_logits: Any
    boundary_cause_logits: Any


_ModuleBase = nn.Module if nn is not None else object


class StageEditLearnedReviewer(_ModuleBase):
    """Bidirectional boundary encoder with global and localized heads."""

    def __init__(self, config: LearnedReviewerConfig):
        _require_mlx()
        super().__init__()
        self.config = config
        dim = config.model_dimension
        self.boundary_projection = nn.Linear(config.boundary_input_dimension, dim)
        self.timeline_projection = nn.Linear(config.timeline_input_dimension, dim)
        self.position_embedding = nn.Embedding(config.max_boundaries, dim)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = nn.TransformerEncoder(
            num_layers=config.num_layers,
            dims=dim,
            num_heads=config.num_heads,
            mlp_dims=config.feed_forward_dimension,
            dropout=config.dropout,
            norm_first=True,
        )
        self.global_fusion = nn.Linear(2 * dim, dim)
        self.global_quality_head = nn.Linear(dim, 1)
        self.boundary_fusion = nn.Linear(2 * dim, dim)
        self.boundary_issue_head = nn.Linear(dim, 1)
        self.boundary_cause_head = nn.Linear(dim, config.cause_count)

    def __call__(self, batch: LearnedReviewerTensorBatch) -> LearnedReviewerModelOutput:
        length = batch.boundary_features.shape[1]
        if length > self.config.max_boundaries:
            raise LearnedReviewerInputError("batch exceeds reviewer max_boundaries")
        hidden = nn.relu(self.boundary_projection(batch.boundary_features))
        hidden = hidden + self.position_embedding(mx.arange(length))[None, :, :]
        hidden = self.dropout(hidden)
        padding = mx.where(
            batch.sequence_mask[:, None, None, :],
            mx.array(0.0, dtype=hidden.dtype),
            mx.array(-1.0e9, dtype=hidden.dtype),
        )
        hidden = self.encoder(hidden, padding)
        sequence_weights = batch.sequence_mask.astype(hidden.dtype)[..., None]
        hidden = hidden * sequence_weights
        pooled = hidden.sum(axis=1) / mx.maximum(sequence_weights.sum(axis=1), 1.0)
        timeline = nn.relu(self.timeline_projection(batch.timeline_features))
        global_hidden = nn.relu(
            self.global_fusion(mx.concatenate((pooled, timeline), axis=-1))
        )
        quality = self.global_quality_head(global_hidden).squeeze(-1)
        broadcast_global = mx.broadcast_to(
            global_hidden[:, None, :], hidden.shape
        )
        boundary_hidden = nn.relu(
            self.boundary_fusion(
                mx.concatenate((hidden, broadcast_global), axis=-1)
            )
        )
        issue = self.boundary_issue_head(boundary_hidden).squeeze(-1)
        causes = self.boundary_cause_head(boundary_hidden)
        issue = mx.where(batch.sequence_mask, issue, mx.zeros_like(issue))
        causes = mx.where(
            batch.sequence_mask[..., None], causes, mx.zeros_like(causes)
        )
        return LearnedReviewerModelOutput(quality, issue, causes)


@dataclass(frozen=True, slots=True)
class LearnedReviewerLossContract:
    pairwise_margin_weight: float = 1.0
    listwise_ranking_weight: float = 0.5
    boundary_issue_weight: float = 0.75
    cause_bce_weight: float = 1.0
    calibration_weight: float = 0.75

    def __post_init__(self) -> None:
        values = tuple(_finite(value, name) for name, value in asdict(self).items())
        if any(value < 0.0 for value in values) or not any(value > 0.0 for value in values):
            raise LearnedReviewerInputError("loss weights must be non-negative and nonzero")


def _masked_mean(values: Any, mask: Any) -> Any:
    weights = mask.astype(values.dtype)
    safe_values = mx.where(mask, values, mx.zeros_like(values))
    return safe_values.sum() / mx.maximum(weights.sum(), 1.0)


def _ranking_loss(
    logits: Any,
    preferred_indices: Any,
    rejected_indices: Any,
    margins: Any | None,
) -> Any:
    if int(preferred_indices.shape[0]) == 0:
        return mx.array(0.0, dtype=logits.dtype)
    preferred = logits[preferred_indices]
    rejected = logits[rejected_indices]
    required_margin = (
        margins
        if margins is not None
        else mx.zeros_like(preferred)
    )
    return nn.softplus(required_margin - preferred + rejected).mean()


def learned_reviewer_loss_from_output(
    output: LearnedReviewerModelOutput,
    supervision: LearnedReviewerSupervision,
    contract: LearnedReviewerLossContract = LearnedReviewerLossContract(),
) -> tuple[Any, dict[str, Any]]:
    _require_mlx()
    probabilities = mx.sigmoid(output.global_quality_logits)
    calibration_values = (probabilities - supervision.quality_targets) ** 2
    calibration = _masked_mean(calibration_values, supervision.quality_mask)
    issue_values = nn.losses.binary_cross_entropy(
        output.boundary_issue_logits,
        supervision.boundary_issue_targets,
        with_logits=True,
        reduction="none",
    )
    issue = _masked_mean(issue_values, supervision.boundary_issue_mask)
    cause_values = nn.losses.binary_cross_entropy(
        output.boundary_cause_logits,
        supervision.cause_targets,
        with_logits=True,
        reduction="none",
    )
    cause = _masked_mean(cause_values, supervision.cause_mask)
    pairwise = _ranking_loss(
        output.global_quality_logits,
        supervision.pairwise_preferred_indices,
        supervision.pairwise_rejected_indices,
        supervision.pairwise_margins,
    )
    listwise = _ranking_loss(
        output.global_quality_logits,
        supervision.listwise_preferred_indices,
        supervision.listwise_rejected_indices,
        None,
    )
    total = (
        contract.pairwise_margin_weight * pairwise
        + contract.listwise_ranking_weight * listwise
        + contract.boundary_issue_weight * issue
        + contract.cause_bce_weight * cause
        + contract.calibration_weight * calibration
    )
    return total, {
        "total": total,
        "pairwise_ranking": pairwise,
        "listwise_ranking": listwise,
        "boundary_issue_bce": issue,
        "cause_bce": cause,
        "calibration_brier": calibration,
    }


def learned_reviewer_loss(
    model: StageEditLearnedReviewer,
    batch: LearnedReviewerTensorBatch,
    supervision: LearnedReviewerSupervision,
    contract: LearnedReviewerLossContract = LearnedReviewerLossContract(),
) -> tuple[Any, dict[str, Any]]:
    return learned_reviewer_loss_from_output(model(batch), supervision, contract)


@dataclass(frozen=True, slots=True)
class LearnedReviewerProductionDataGate:
    passed: bool
    observed: Mapping[str, int]
    requirements: Mapping[str, int]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "observed": dict(self.observed),
            "requirements": dict(self.requirements),
            "reasons": list(self.reasons),
        }


def learned_reviewer_production_data_gate(
    corpus: LearnedReviewerCorpus,
    split: LearnedReviewerSplit,
) -> LearnedReviewerProductionDataGate:
    """Require dense, multi-song supervision before production can be claimed."""

    ranked_ids = {
        preference.preferred_example_id
        for preference in corpus.pairwise_preferences
    } | {
        preference.rejected_example_id
        for preference in corpus.pairwise_preferences
    } | {
        example_id
        for group in corpus.listwise_groups
        for example_id in group.ordered_example_ids
    }
    counts = {
        kind: sum(example.example_kind == kind for example in corpus.examples)
        for kind in EXAMPLE_KINDS
    }
    songs_by_kind = {
        kind: {
            example.song_id
            for example in corpus.examples
            if example.example_kind == kind
        }
        for kind in EXAMPLE_KINDS
    }
    dense_songs = {
        example.song_id
        for example in corpus.examples
        if example.dense_boundary_supervision
    }
    ranked_candidate_songs = {
        example.song_id
        for example in corpus.examples
        if example.example_kind == CANDIDATE and example.example_id in ranked_ids
    }
    observed = {
        "song_count": len(corpus.song_ids),
        "training_song_count": len(split.training_song_ids),
        "heldout_song_count": len(split.heldout_song_ids),
        "human_reference_count": counts[HUMAN_REFERENCE],
        "human_reference_song_count": len(songs_by_kind[HUMAN_REFERENCE]),
        "synthetic_shifted_count": counts[SYNTHETIC_SHIFTED],
        "synthetic_shifted_song_count": len(songs_by_kind[SYNTHETIC_SHIFTED]),
        "synthetic_worsened_count": counts[SYNTHETIC_WORSENED],
        "synthetic_worsened_song_count": len(songs_by_kind[SYNTHETIC_WORSENED]),
        "synthetic_technical_count": counts[SYNTHETIC_TECHNICAL],
        "synthetic_technical_song_count": len(songs_by_kind[SYNTHETIC_TECHNICAL]),
        "dense_example_count": sum(
            example.dense_boundary_supervision for example in corpus.examples
        ),
        "dense_example_song_count": len(dense_songs),
        "pairwise_preference_count": len(corpus.pairwise_preferences),
        "listwise_group_count": len(corpus.listwise_groups),
        "ranked_candidate_count": sum(
            example.example_kind == CANDIDATE
            and example.example_id in ranked_ids
            for example in corpus.examples
        ),
        "ranked_candidate_song_count": len(ranked_candidate_songs),
    }
    requirements = _production_requirements()
    reasons = tuple(
        f"{name}={observed[name]} is below required {minimum}"
        for name, minimum in requirements.items()
        if observed[name] < minimum
    )
    return LearnedReviewerProductionDataGate(
        passed=not reasons,
        observed=observed,
        requirements=requirements,
        reasons=reasons,
    )


def _tensorize_learned_reviewer_examples(
    examples: Sequence[LearnedReviewerExample],
    schema: LearnedReviewerFeatureSchema,
    config: LearnedReviewerConfig,
    *,
    device: Any = None,
) -> LearnedReviewerTensorBatch:
    """Tensorize inference-only examples without inventing supervision."""

    _require_mlx()
    selected = tuple(examples)
    if not selected:
        raise LearnedReviewerInputError("reviewer inference needs an example")
    if len({example.example_id for example in selected}) != len(selected):
        raise LearnedReviewerInputError("reviewer inference ids must be unique")
    if config.feature_schema_fingerprint != schema.fingerprint:
        raise LearnedReviewerInputError("model config and inference schema differ")
    if (
        config.boundary_input_dimension != schema.encoded_boundary_dimension
        or config.timeline_input_dimension != schema.encoded_timeline_dimension
        or config.cause_count != len(schema.cause_labels)
    ):
        raise LearnedReviewerInputError("model input dimensions differ from schema")
    maximum = max(len(example.boundaries) for example in selected)
    if maximum > config.max_boundaries:
        raise LearnedReviewerInputError("example exceeds max_boundaries")
    batch_size = len(selected)
    boundary_values = np.zeros(
        (batch_size, maximum, schema.boundary_dimension), dtype=np.float32
    )
    boundary_observed = np.zeros_like(boundary_values, dtype=bool)
    sequence_mask = np.zeros((batch_size, maximum), dtype=bool)
    timeline_values = np.zeros(
        (batch_size, schema.timeline_dimension), dtype=np.float32
    )
    timeline_observed = np.zeros_like(timeline_values, dtype=bool)
    technical_boundary = np.zeros((batch_size, maximum), dtype=bool)
    for batch_index, example in enumerate(selected):
        if len(example.timeline_features) != schema.timeline_dimension:
            raise LearnedReviewerInputError("timeline feature dimension mismatch")
        sequence_mask[batch_index, : len(example.boundaries)] = True
        for feature_index, value in enumerate(example.timeline_features):
            if value is not None:
                timeline_values[batch_index, feature_index] = value
                timeline_observed[batch_index, feature_index] = True
        for boundary_index, boundary in enumerate(example.boundaries):
            if len(boundary.features) != schema.boundary_dimension:
                raise LearnedReviewerInputError("boundary feature dimension mismatch")
            for feature_index, value in enumerate(boundary.features):
                if value is not None:
                    boundary_values[batch_index, boundary_index, feature_index] = value
                    boundary_observed[
                        batch_index, boundary_index, feature_index
                    ] = True
            technical_boundary[batch_index, boundary_index] = bool(
                _technical_codes(boundary, schema)
            )
    encoded_boundary = np.concatenate(
        (boundary_values, boundary_observed.astype(np.float32)), axis=-1
    )
    encoded_timeline = np.concatenate(
        (timeline_values, timeline_observed.astype(np.float32)), axis=-1
    )
    active_device = device if device is not None else mx.default_device()
    with mx.stream(active_device):
        return LearnedReviewerTensorBatch(
            boundary_features=mx.array(encoded_boundary),
            boundary_observed_mask=mx.array(boundary_observed),
            sequence_mask=mx.array(sequence_mask),
            timeline_features=mx.array(encoded_timeline),
            timeline_observed_mask=mx.array(timeline_observed),
            technical_boundary_mask=mx.array(technical_boundary),
            technical_example_mask=mx.array(np.any(technical_boundary, axis=1)),
            example_ids=tuple(example.example_id for example in selected),
        )


@dataclass(frozen=True, slots=True)
class LearnedReviewerBoundaryPrediction:
    boundary_id: str
    at_seconds: float
    issue_probability: float
    cause_probabilities: Mapping[str, float]
    technical_error_codes: tuple[str, ...]
    technical_hard_failure: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "at_seconds": self.at_seconds,
            "issue_probability": self.issue_probability,
            "cause_probabilities": dict(self.cause_probabilities),
            "technical_error_codes": list(self.technical_error_codes),
            "technical_hard_failure": self.technical_hard_failure,
        }


@dataclass(frozen=True, slots=True)
class LearnedReviewerPrediction:
    example_id: str
    learned_quality_probability: float
    learned_quality_score: float
    model_recommendation_passed: bool
    technical_zero_tolerance_passed: bool
    review_passed: bool
    boundary_predictions: tuple[LearnedReviewerBoundaryPrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "learned_quality_probability": self.learned_quality_probability,
            "learned_quality_score": self.learned_quality_score,
            "model_recommendation_passed": self.model_recommendation_passed,
            "technical_zero_tolerance_passed": (
                self.technical_zero_tolerance_passed
            ),
            "review_passed": self.review_passed,
            "boundary_predictions": [
                prediction.to_dict() for prediction in self.boundary_predictions
            ],
        }


def _technical_codes(
    boundary: LearnedReviewerBoundary,
    schema: LearnedReviewerFeatureSchema,
) -> tuple[str, ...]:
    derived = {
        schema.boundary_feature_names[index]
        for index in schema.technical_feature_indices
        if boundary.features[index] is not None
        and float(boundary.features[index]) >= 0.5
    }
    missing = {
        f"missing_{schema.boundary_feature_names[index]}"
        for index in schema.technical_feature_indices
        if boundary.features[index] is None
    }
    return tuple(
        sorted(set(boundary.technical_error_codes) | derived | missing)
    )


def predict_learned_reviewer(
    model: StageEditLearnedReviewer,
    examples: Sequence[LearnedReviewerExample],
    schema: LearnedReviewerFeatureSchema = DEFAULT_LEARNED_REVIEWER_SCHEMA,
    *,
    quality_threshold: float = 0.5,
    device: Any = None,
) -> tuple[LearnedReviewerPrediction, ...]:
    """Run learned heads while preserving an independent technical hard gate.

    ``review_passed`` is only this reviewer's recommendation.  Existing
    deterministic professional/cadence gates remain separate consumers and
    can only further reject a candidate.
    """

    threshold = _probability(quality_threshold, "quality_threshold")
    selected = tuple(examples)
    batch = _tensorize_learned_reviewer_examples(
        selected, schema, model.config, device=device
    )
    if hasattr(model, "eval"):
        model.eval()
    output = model(batch)
    quality = mx.sigmoid(output.global_quality_logits)
    issues = mx.sigmoid(output.boundary_issue_logits)
    causes = mx.sigmoid(output.boundary_cause_logits)
    mx.eval(quality, issues, causes)
    quality_values = np.asarray(quality, dtype=np.float64)
    issue_values = np.asarray(issues, dtype=np.float64)
    cause_values = np.asarray(causes, dtype=np.float64)
    if not (
        np.isfinite(quality_values).all()
        and np.isfinite(issue_values).all()
        and np.isfinite(cause_values).all()
    ):
        raise LearnedReviewerInputError("learned reviewer produced non-finite output")
    predictions: list[LearnedReviewerPrediction] = []
    for example_index, example in enumerate(selected):
        boundaries: list[LearnedReviewerBoundaryPrediction] = []
        technical_passed = True
        for boundary_index, boundary in enumerate(example.boundaries):
            codes = _technical_codes(boundary, schema)
            technical_passed = technical_passed and not codes
            boundaries.append(
                LearnedReviewerBoundaryPrediction(
                    boundary_id=boundary.boundary_id,
                    at_seconds=boundary.at_seconds,
                    issue_probability=float(issue_values[example_index, boundary_index]),
                    cause_probabilities={
                        label: float(
                            cause_values[example_index, boundary_index, cause_index]
                        )
                        for cause_index, label in enumerate(schema.cause_labels)
                    },
                    technical_error_codes=codes,
                    technical_hard_failure=bool(codes),
                )
            )
        probability = float(quality_values[example_index])
        model_passed = probability >= threshold
        predictions.append(
            LearnedReviewerPrediction(
                example_id=example.example_id,
                learned_quality_probability=probability,
                learned_quality_score=100.0 * probability,
                model_recommendation_passed=model_passed,
                technical_zero_tolerance_passed=technical_passed,
                review_passed=model_passed and technical_passed,
                boundary_predictions=tuple(boundaries),
            )
        )
    return tuple(predictions)


def _binary_auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = 0.0
    comparisons = 0
    for positive_score in positive:
        for negative_score in negative:
            comparisons += 1
            wins += float(positive_score > negative_score)
            wins += 0.5 * float(positive_score == negative_score)
    return wins / comparisons


@dataclass(frozen=True, slots=True)
class LearnedReviewerEvaluation:
    example_count: int
    human_count: int
    worsened_count: int
    rank_pair_count: int
    quality_label_count: int
    human_vs_worsened_auc: float | None
    mean_rank_margin: float | None
    brier_calibration_error: float | None
    technical_zero_tolerance_policy_verified: bool
    passed: bool
    loss_components: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_learned_reviewer(
    model: StageEditLearnedReviewer,
    corpus: LearnedReviewerCorpus,
    example_ids: Sequence[str],
    *,
    contract: LearnedReviewerLossContract = LearnedReviewerLossContract(),
    device: Any = None,
) -> LearnedReviewerEvaluation:
    selected_ids = tuple(example_ids)
    batch, supervision = tensorize_learned_reviewer_corpus(
        corpus, selected_ids, model.config, device=device
    )
    if hasattr(model, "eval"):
        model.eval()
    output = model(batch)
    _, components = learned_reviewer_loss_from_output(
        output, supervision, contract
    )
    probabilities = mx.sigmoid(output.global_quality_logits)
    mx.eval(probabilities, components)
    scores = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise LearnedReviewerInputError("reviewer evaluation produced non-finite scores")
    by_id = {example.example_id: example for example in corpus.examples}
    examples = tuple(by_id[example_id] for example_id in selected_ids)
    local_index = {example_id: index for index, example_id in enumerate(selected_ids)}
    positives = [
        float(scores[index])
        for index, example in enumerate(examples)
        if example.example_kind == HUMAN_REFERENCE
    ]
    negatives = [
        float(scores[index])
        for index, example in enumerate(examples)
        if example.example_kind == SYNTHETIC_WORSENED
    ]
    rank_pairs: list[tuple[int, int]] = [
        (
            local_index[preference.preferred_example_id],
            local_index[preference.rejected_example_id],
        )
        for preference in corpus.pairwise_preferences
        if preference.preferred_example_id in local_index
        and preference.rejected_example_id in local_index
    ]
    for group in corpus.listwise_groups:
        if all(example_id in local_index for example_id in group.ordered_example_ids):
            for left in range(len(group.ordered_example_ids)):
                for right in range(left + 1, len(group.ordered_example_ids)):
                    rank_pairs.append(
                        (
                            local_index[group.ordered_example_ids[left]],
                            local_index[group.ordered_example_ids[right]],
                        )
                    )
    margins = [float(scores[left] - scores[right]) for left, right in rank_pairs]
    quality_pairs = [
        (float(scores[index]), float(example.quality_target))
        for index, example in enumerate(examples)
        if example.quality_target is not None
    ]
    brier = (
        sum((score - target) ** 2 for score, target in quality_pairs)
        / len(quality_pairs)
        if quality_pairs
        else None
    )
    auc = _binary_auc(positives, negatives)
    mean_margin = sum(margins) / len(margins) if margins else None
    technical_examples = [
        example
        for example in examples
        if example.example_kind == SYNTHETIC_TECHNICAL
    ]
    technical_predictions = (
        predict_learned_reviewer(
            model,
            technical_examples,
            corpus.feature_schema,
            quality_threshold=0.0,
            device=device,
        )
        if technical_examples
        else ()
    )
    technical_verified = bool(technical_examples) and all(
        _example_has_technical_error(example, corpus.feature_schema)
        and not prediction.technical_zero_tolerance_passed
        and not prediction.review_passed
        for example, prediction in zip(
            technical_examples, technical_predictions, strict=True
        )
    )
    passed = bool(
        auc is not None
        and auc >= MINIMUM_HELDOUT_HUMAN_WORSENED_AUROC
        and mean_margin is not None
        and mean_margin >= MINIMUM_HELDOUT_MEAN_RANK_MARGIN
        and brier is not None
        and brier <= MAXIMUM_HELDOUT_BRIER_CALIBRATION_ERROR
        and technical_verified
    )
    scalar_components = {
        name: float(np.asarray(value)) for name, value in components.items()
    }
    if not all(math.isfinite(value) for value in scalar_components.values()):
        raise LearnedReviewerInputError("reviewer evaluation loss is non-finite")
    return LearnedReviewerEvaluation(
        example_count=len(examples),
        human_count=len(positives),
        worsened_count=len(negatives),
        rank_pair_count=len(rank_pairs),
        quality_label_count=len(quality_pairs),
        human_vs_worsened_auc=auc,
        mean_rank_margin=mean_margin,
        brier_calibration_error=brier,
        technical_zero_tolerance_policy_verified=technical_verified,
        passed=passed,
        loss_components=scalar_components,
    )


def mlx_runtime_version() -> str | None:
    if mx is None:
        return None
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "installed-version-unknown"


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LearnedReviewerInputError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise LearnedReviewerInputError(f"{name} must be a boolean")
    return value


def _sha256(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise LearnedReviewerInputError(f"{name} must be a string")
    result = value
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise LearnedReviewerInputError(f"{name} must be a lowercase SHA-256")
    return result


@dataclass(frozen=True, slots=True)
class LearnedReviewerCheckpointMetadata:
    format_version: int
    checkpoint_id: str
    created_at: str
    training_steps: int
    weights_trained: bool
    corpus_id: str
    corpus_sha256: str
    feature_schema: LearnedReviewerFeatureSchema
    model_config: LearnedReviewerConfig
    weight_filename: str = "model.safetensors"
    weight_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.format_version != LEARNED_REVIEWER_CHECKPOINT_VERSION:
            raise LearnedReviewerInputError("unsupported reviewer checkpoint format")
        object.__setattr__(self, "checkpoint_id", _name(self.checkpoint_id, "checkpoint_id"))
        created = _name(self.created_at, "created_at")
        try:
            parsed = datetime.fromisoformat(created)
        except ValueError as exc:
            raise LearnedReviewerInputError("created_at must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise LearnedReviewerInputError("created_at must include a timezone")
        object.__setattr__(self, "created_at", created)
        steps = _strict_int(self.training_steps, "training_steps")
        trained = _strict_bool(self.weights_trained, "weights_trained")
        if trained and steps < 1:
            raise LearnedReviewerInputError("trained weights require a positive step")
        object.__setattr__(self, "training_steps", steps)
        object.__setattr__(self, "weights_trained", trained)
        object.__setattr__(self, "corpus_id", _name(self.corpus_id, "corpus_id"))
        object.__setattr__(
            self,
            "corpus_sha256",
            _sha256(self.corpus_sha256, "corpus_sha256"),
        )
        if not isinstance(self.feature_schema, LearnedReviewerFeatureSchema):
            raise LearnedReviewerInputError("checkpoint feature_schema is invalid")
        if not isinstance(self.model_config, LearnedReviewerConfig):
            raise LearnedReviewerInputError("checkpoint model_config is invalid")
        if self.model_config.feature_schema_fingerprint != self.feature_schema.fingerprint:
            raise LearnedReviewerInputError("checkpoint config and feature schema differ")
        filename = _name(self.weight_filename, "weight_filename")
        if Path(filename).name != filename:
            raise LearnedReviewerInputError("weight_filename cannot contain a path")
        if Path(filename).suffix != ".safetensors":
            raise LearnedReviewerInputError("reviewer weights must use safetensors")
        object.__setattr__(self, "weight_filename", filename)
        object.__setattr__(
            self,
            "weight_sha256",
            _sha256(self.weight_sha256, "weight_sha256", optional=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at,
            "training_steps": self.training_steps,
            "weights_trained": self.weights_trained,
            "corpus_id": self.corpus_id,
            "corpus_sha256": self.corpus_sha256,
            "feature_schema": self.feature_schema.to_dict(),
            "model_config": self.model_config.to_dict(),
            "weight_filename": self.weight_filename,
            "weight_sha256": self.weight_sha256,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "LearnedReviewerCheckpointMetadata":
        _strict_keys(
            payload,
            {
                "format_version",
                "checkpoint_id",
                "created_at",
                "training_steps",
                "weights_trained",
                "corpus_id",
                "corpus_sha256",
                "feature_schema",
                "model_config",
                "weight_filename",
                "weight_sha256",
            },
            "learned-reviewer checkpoint metadata",
        )
        return cls(
            format_version=_strict_int(
                payload["format_version"], "format_version", minimum=1
            ),
            checkpoint_id=payload["checkpoint_id"],
            created_at=payload["created_at"],
            training_steps=_strict_int(payload["training_steps"], "training_steps"),
            weights_trained=_strict_bool(
                payload["weights_trained"], "weights_trained"
            ),
            corpus_id=payload["corpus_id"],
            corpus_sha256=payload["corpus_sha256"],
            feature_schema=LearnedReviewerFeatureSchema.from_dict(
                payload["feature_schema"]
            ),
            model_config=LearnedReviewerConfig.from_dict(payload["model_config"]),
            weight_filename=payload["weight_filename"],
            weight_sha256=payload["weight_sha256"],
        )


def new_learned_reviewer_checkpoint_metadata(
    checkpoint_id: str,
    corpus: LearnedReviewerCorpus,
    config: LearnedReviewerConfig,
    *,
    training_steps: int,
    weights_trained: bool,
) -> LearnedReviewerCheckpointMetadata:
    return LearnedReviewerCheckpointMetadata(
        format_version=LEARNED_REVIEWER_CHECKPOINT_VERSION,
        checkpoint_id=checkpoint_id,
        created_at=datetime.now(UTC).isoformat(),
        training_steps=training_steps,
        weights_trained=weights_trained,
        corpus_id=corpus.corpus_id,
        corpus_sha256=corpus.sha256,
        feature_schema=corpus.feature_schema,
        model_config=config,
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination.resolve()


def save_learned_reviewer_checkpoint(
    directory: Path,
    model: StageEditLearnedReviewer,
    metadata: LearnedReviewerCheckpointMetadata,
) -> LearnedReviewerCheckpointMetadata:
    _require_mlx()
    if model.config != metadata.model_config:
        raise LearnedReviewerInputError("model config does not match metadata")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    weight_path = destination / metadata.weight_filename
    temporary = weight_path.with_name(
        f".{weight_path.stem}.{uuid.uuid4().hex}.tmp.safetensors"
    )
    try:
        model.save_weights(str(temporary))
        temporary.replace(weight_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    saved = replace(metadata, weight_sha256=_sha256_file(weight_path))
    _atomic_write_json(destination / "metadata.json", saved.to_dict())
    return saved


def _read_checkpoint_metadata(
    directory: Path,
) -> tuple[LearnedReviewerCheckpointMetadata, Path, Path]:
    source = Path(directory)
    metadata_path = source / "metadata.json"
    if not metadata_path.is_file():
        raise LearnedReviewerNotReadyError("reviewer checkpoint metadata is missing")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise LearnedReviewerInputError("checkpoint metadata root must be an object")
        metadata = LearnedReviewerCheckpointMetadata.from_dict(payload)
    except (
        OSError,
        json.JSONDecodeError,
        LearnedReviewerInputError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise LearnedReviewerNotReadyError(
            f"reviewer checkpoint metadata is invalid: {exc}"
        ) from exc
    weight_path = source / metadata.weight_filename
    if not weight_path.is_file():
        raise LearnedReviewerNotReadyError("reviewer checkpoint weights are missing")
    return metadata, metadata_path, weight_path


def load_learned_reviewer_checkpoint(
    directory: Path,
    *,
    expected_schema: LearnedReviewerFeatureSchema | None = None,
    require_trained: bool = True,
) -> tuple[StageEditLearnedReviewer, LearnedReviewerCheckpointMetadata]:
    _require_mlx()
    metadata, metadata_path, weight_path = _read_checkpoint_metadata(Path(directory))
    if expected_schema is not None and (
        expected_schema.fingerprint != metadata.feature_schema.fingerprint
    ):
        raise LearnedReviewerNotReadyError("reviewer feature schema does not match")
    if require_trained and not (
        metadata.weights_trained and metadata.training_steps > 0
    ):
        raise LearnedReviewerNotReadyError("reviewer checkpoint is not trained")
    before = {
        "metadata": _sha256_file(metadata_path),
        "weights": _sha256_file(weight_path),
    }
    if metadata.weight_sha256 != before["weights"]:
        raise LearnedReviewerNotReadyError("reviewer checkpoint weight digest is invalid")
    model = StageEditLearnedReviewer(metadata.model_config)
    try:
        model.load_weights(str(weight_path), strict=True)
        mx.eval(model.parameters())
    except (OSError, RuntimeError, ValueError) as exc:
        raise LearnedReviewerNotReadyError(
            f"reviewer checkpoint weights cannot be loaded: {exc}"
        ) from exc
    after = {
        "metadata": _sha256_file(metadata_path),
        "weights": _sha256_file(weight_path),
    }
    if before != after:
        raise LearnedReviewerNotReadyError(
            "reviewer checkpoint changed while it was being loaded"
        )
    if hasattr(model, "eval"):
        model.eval()
    return model, metadata


@dataclass(frozen=True, slots=True)
class LearnedReviewerReadinessReport:
    architecture_available: bool
    learned_inference_ready: bool
    status: str
    reason: str
    mlx_version: str | None
    checkpoint_path: str | None
    checkpoint_id: str | None
    training_steps: int
    metadata_sha256: str | None = None
    training_report_sha256: str | None = None
    weight_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_report_metric(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LearnedReviewerInputError(f"{name} must be numeric")
    return _finite(value, name)


def _production_report_errors(
    payload: Any,
    *,
    metadata: LearnedReviewerCheckpointMetadata,
    metadata_sha256: str,
    weight_sha256: str,
) -> tuple[str, ...]:
    if not isinstance(payload, Mapping):
        return ("production training report root must be an object",)
    errors: list[str] = []
    expected_outer = {
        "schema_version",
        "run_kind",
        "checkpoint_scope",
        "production_ready",
        "production_activation_authorized",
        "corpus",
        "checkpoint",
        "production_data_gate",
        "split",
        "held_out_evaluation",
        "training",
    }
    if set(payload) != expected_outer:
        errors.append("production training report keys are not exact")
    if (
        isinstance(payload.get("schema_version"), bool)
        or not isinstance(payload.get("schema_version"), int)
        or payload.get("schema_version") != LEARNED_REVIEWER_REPORT_VERSION
    ):
        errors.append("production training report schema_version is invalid")
    if payload.get("run_kind") != LEARNED_REVIEWER_RUN_KIND:
        errors.append(f"run_kind must be {LEARNED_REVIEWER_RUN_KIND!r}")
    if payload.get("checkpoint_scope") != LEARNED_REVIEWER_CHECKPOINT_SCOPE:
        errors.append(
            f"checkpoint_scope must be {LEARNED_REVIEWER_CHECKPOINT_SCOPE!r}"
        )
    if payload.get("production_ready") is not True:
        errors.append("production_ready must be explicitly true")
    if payload.get("production_activation_authorized") is not True:
        errors.append("production activation was not explicitly authorized")

    corpus = payload.get("corpus")
    expected_corpus_keys = {
        "corpus_id",
        "corpus_sha256",
        "feature_schema_fingerprint",
    }
    if not isinstance(corpus, Mapping) or set(corpus) != expected_corpus_keys:
        errors.append("corpus evidence is missing or malformed")
    else:
        if corpus.get("corpus_id") != metadata.corpus_id:
            errors.append("report corpus_id does not match checkpoint")
        if corpus.get("corpus_sha256") != metadata.corpus_sha256:
            errors.append("report corpus digest does not match checkpoint")
        if (
            corpus.get("feature_schema_fingerprint")
            != metadata.feature_schema.fingerprint
        ):
            errors.append("report feature schema does not match checkpoint")

    checkpoint = payload.get("checkpoint")
    expected_checkpoint_keys = {
        "checkpoint_id",
        "training_steps",
        "weight_sha256",
        "metadata_sha256",
    }
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != expected_checkpoint_keys:
        errors.append("checkpoint evidence is missing or malformed")
    else:
        if checkpoint.get("checkpoint_id") != metadata.checkpoint_id:
            errors.append("report checkpoint_id does not match metadata")
        report_steps = checkpoint.get("training_steps")
        if (
            isinstance(report_steps, bool)
            or not isinstance(report_steps, int)
            or report_steps != metadata.training_steps
        ):
            errors.append("report training_steps do not match metadata")
        if checkpoint.get("weight_sha256") != weight_sha256:
            errors.append("report weight digest does not match checkpoint")
        if checkpoint.get("metadata_sha256") != metadata_sha256:
            errors.append("report metadata digest does not match checkpoint")

    gate = payload.get("production_data_gate")
    expected_gate_keys = {"passed", "observed", "requirements", "reasons"}
    requirements = _production_requirements()
    if not isinstance(gate, Mapping) or set(gate) != expected_gate_keys:
        errors.append("production data gate is missing or malformed")
    else:
        if gate.get("passed") is not True:
            errors.append("production data gate did not pass")
        observed = gate.get("observed")
        report_requirements = gate.get("requirements")
        if not isinstance(observed, Mapping) or set(observed) != set(requirements):
            errors.append("production data observations are malformed")
        else:
            for name, minimum in requirements.items():
                value = observed.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    errors.append(f"production data {name} must be at least {minimum}")
        valid_report_requirements = bool(
            isinstance(report_requirements, Mapping)
            and set(report_requirements) == set(requirements)
            and all(
                not isinstance(report_requirements[name], bool)
                and isinstance(report_requirements[name], int)
                and report_requirements[name] == minimum
                for name, minimum in requirements.items()
            )
        )
        if not valid_report_requirements:
            errors.append("production data requirements do not match code contract")
        if gate.get("reasons") != []:
            errors.append("passed production data gate cannot contain reasons")

    split = payload.get("split")
    expected_split_keys = {
        "split_level",
        "training_song_ids",
        "heldout_song_ids",
    }
    valid_training_songs = False
    valid_heldout_songs = False
    if not isinstance(split, Mapping) or set(split) != expected_split_keys:
        errors.append("song-level split evidence is missing or malformed")
        training_songs: Any = None
        heldout_songs: Any = None
    else:
        if split.get("split_level") != "song":
            errors.append("reviewer split must be song-level")
        training_songs = split.get("training_song_ids")
        heldout_songs = split.get("heldout_song_ids")
        valid_training_songs = bool(
            isinstance(training_songs, list)
            and len(training_songs) >= MINIMUM_PRODUCTION_TRAIN_SONG_COUNT
            and all(isinstance(song, str) and song.strip() for song in training_songs)
            and len(set(training_songs)) == len(training_songs)
        )
        valid_heldout_songs = bool(
            isinstance(heldout_songs, list)
            and len(heldout_songs) >= MINIMUM_PRODUCTION_HELDOUT_SONG_COUNT
            and all(isinstance(song, str) and song.strip() for song in heldout_songs)
            and len(set(heldout_songs)) == len(heldout_songs)
        )
        if not valid_training_songs:
            errors.append("reviewer split needs at least two training songs")
        if not valid_heldout_songs:
            errors.append("reviewer split needs at least one held-out song")
        if valid_training_songs and valid_heldout_songs and not set(
            training_songs
        ).isdisjoint(heldout_songs):
            errors.append("training and held-out songs must be disjoint")

    heldout = payload.get("held_out_evaluation")
    expected_heldout_keys = {
        "passed",
        "checkpoint_id",
        "weight_sha256",
        "corpus_sha256",
        "split_level",
        "training_song_ids",
        "heldout_song_ids",
        "human_count",
        "worsened_count",
        "rank_pair_count",
        "quality_label_count",
        "human_vs_worsened_auc",
        "mean_rank_margin",
        "brier_calibration_error",
        "technical_zero_tolerance_policy_verified",
        "thresholds",
        "loss_components",
    }
    if not isinstance(heldout, Mapping) or set(heldout) != expected_heldout_keys:
        errors.append("held-out evaluation is missing or malformed")
    else:
        if heldout.get("passed") is not True:
            errors.append("held-out evaluation did not pass")
        if heldout.get("checkpoint_id") != metadata.checkpoint_id:
            errors.append("held-out checkpoint_id does not match metadata")
        if heldout.get("weight_sha256") != weight_sha256:
            errors.append("held-out weight digest does not match checkpoint")
        if heldout.get("corpus_sha256") != metadata.corpus_sha256:
            errors.append("held-out corpus digest does not match checkpoint")
        if heldout.get("split_level") != "song":
            errors.append("held-out evaluation must be song-level")
        if isinstance(split, Mapping):
            if heldout.get("training_song_ids") != split.get("training_song_ids"):
                errors.append("held-out training song ids do not match split")
            if heldout.get("heldout_song_ids") != split.get("heldout_song_ids"):
                errors.append("held-out song ids do not match split")
        for count_name in (
            "human_count",
            "worsened_count",
            "rank_pair_count",
            "quality_label_count",
        ):
            value = heldout.get(count_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                errors.append(f"held-out {count_name} must be positive")
        try:
            auc = _finite_report_metric(
                heldout.get("human_vs_worsened_auc"),
                "human_vs_worsened_auc",
            )
            margin = _finite_report_metric(
                heldout.get("mean_rank_margin"), "mean_rank_margin"
            )
            brier = _finite_report_metric(
                heldout.get("brier_calibration_error"),
                "brier_calibration_error",
            )
            if auc < MINIMUM_HELDOUT_HUMAN_WORSENED_AUROC:
                errors.append("held-out human-vs-worsened AUROC is below threshold")
            if margin < MINIMUM_HELDOUT_MEAN_RANK_MARGIN:
                errors.append("held-out mean rank margin is below threshold")
            if brier > MAXIMUM_HELDOUT_BRIER_CALIBRATION_ERROR:
                errors.append("held-out Brier calibration error exceeds threshold")
        except LearnedReviewerInputError as exc:
            errors.append(str(exc))
        if heldout.get("technical_zero_tolerance_policy_verified") is not True:
            errors.append("technical zero-tolerance policy was not verified")
        expected_thresholds = {
            "minimum_human_vs_worsened_auc": (
                MINIMUM_HELDOUT_HUMAN_WORSENED_AUROC
            ),
            "minimum_mean_rank_margin": MINIMUM_HELDOUT_MEAN_RANK_MARGIN,
            "maximum_brier_calibration_error": (
                MAXIMUM_HELDOUT_BRIER_CALIBRATION_ERROR
            ),
        }
        if heldout.get("thresholds") != expected_thresholds:
            errors.append("held-out thresholds do not match code contract")
        losses = heldout.get("loss_components")
        if not isinstance(losses, Mapping) or not losses:
            errors.append("held-out loss components are missing")
        else:
            try:
                for name, value in losses.items():
                    _finite_report_metric(value, f"held-out loss {name}")
            except LearnedReviewerInputError as exc:
                errors.append(str(exc))

    training = payload.get("training")
    expected_training_keys = {
        "epochs_completed",
        "best_epoch",
        "best_training_step",
        "seed",
        "loss_contract",
        "history",
    }
    if not isinstance(training, Mapping) or set(training) != expected_training_keys:
        errors.append("training evidence is missing or malformed")
    else:
        for name in ("epochs_completed", "best_epoch", "best_training_step"):
            value = training.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                errors.append(f"training {name} must be positive")
        if training.get("best_training_step") != metadata.training_steps:
            errors.append("best training step does not match checkpoint")
        if isinstance(training.get("seed"), bool) or not isinstance(
            training.get("seed"), int
        ):
            errors.append("training seed must be an integer")
        if not isinstance(training.get("loss_contract"), Mapping):
            errors.append("training loss contract is missing")
        if not isinstance(training.get("history"), list) or not training.get("history"):
            errors.append("training history is missing")
    return tuple(errors)


def learned_reviewer_readiness_report(
    checkpoint_directory: Path | None = None,
) -> LearnedReviewerReadinessReport:
    version = mlx_runtime_version()
    if version is None:
        return LearnedReviewerReadinessReport(
            architecture_available=False,
            learned_inference_ready=False,
            status="optional_dependency_missing",
            reason="MLX is not installed; corpus inspection only is available",
            mlx_version=None,
            checkpoint_path=None,
            checkpoint_id=None,
            training_steps=0,
        )
    if checkpoint_directory is None:
        return LearnedReviewerReadinessReport(
            architecture_available=True,
            learned_inference_ready=False,
            status="architecture_only_no_checkpoint",
            reason="reviewer model exists but no checkpoint was supplied",
            mlx_version=version,
            checkpoint_path=None,
            checkpoint_id=None,
            training_steps=0,
        )
    source = Path(checkpoint_directory).expanduser().resolve()
    try:
        metadata, metadata_path, weight_path = _read_checkpoint_metadata(source)
    except LearnedReviewerNotReadyError as exc:
        return LearnedReviewerReadinessReport(
            architecture_available=True,
            learned_inference_ready=False,
            status="checkpoint_invalid",
            reason=str(exc),
            mlx_version=version,
            checkpoint_path=str(source),
            checkpoint_id=None,
            training_steps=0,
        )
    metadata_sha256 = _sha256_file(metadata_path)
    weight_sha256 = _sha256_file(weight_path)
    common = {
        "architecture_available": True,
        "mlx_version": version,
        "checkpoint_path": str(source),
        "checkpoint_id": metadata.checkpoint_id,
        "training_steps": metadata.training_steps,
        "metadata_sha256": metadata_sha256,
        "weight_sha256": weight_sha256,
    }
    if not (
        metadata.weights_trained
        and metadata.training_steps > 0
        and metadata.weight_sha256 == weight_sha256
    ):
        return LearnedReviewerReadinessReport(
            learned_inference_ready=False,
            status="untrained_checkpoint_only",
            reason="checkpoint is not declared and verified as trained",
            training_report_sha256=None,
            **common,
        )
    report_path = source / LEARNED_REVIEWER_REPORT_FILENAME
    if not report_path.is_file():
        return LearnedReviewerReadinessReport(
            learned_inference_ready=False,
            status="production_report_missing",
            reason="colocated learned-reviewer production report is missing",
            training_report_sha256=None,
            **common,
        )
    before = {
        "metadata": metadata_sha256,
        "weights": weight_sha256,
        "report": _sha256_file(report_path),
    }
    try:
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return LearnedReviewerReadinessReport(
            learned_inference_ready=False,
            status="production_evidence_invalid",
            reason=f"production report cannot be read: {exc}",
            training_report_sha256=before["report"],
            **common,
        )
    try:
        errors = _production_report_errors(
            report_payload,
            metadata=metadata,
            metadata_sha256=metadata_sha256,
            weight_sha256=weight_sha256,
        )
    except (KeyError, TypeError, ValueError, LearnedReviewerInputError) as exc:
        errors = (f"production report validation failed closed: {exc}",)
    if errors:
        return LearnedReviewerReadinessReport(
            learned_inference_ready=False,
            status="production_evidence_invalid",
            reason="; ".join(errors),
            training_report_sha256=before["report"],
            **common,
        )
    try:
        _, loaded_metadata = load_learned_reviewer_checkpoint(
            source,
            expected_schema=metadata.feature_schema,
            require_trained=True,
        )
    except (LearnedReviewerNotReadyError, RuntimeError) as exc:
        return LearnedReviewerReadinessReport(
            learned_inference_ready=False,
            status="checkpoint_load_failed",
            reason=str(exc),
            training_report_sha256=before["report"],
            **common,
        )
    after = {
        "metadata": _sha256_file(metadata_path),
        "weights": _sha256_file(weight_path),
        "report": _sha256_file(report_path),
    }
    if before != after or loaded_metadata != metadata:
        return LearnedReviewerReadinessReport(
            learned_inference_ready=False,
            status="checkpoint_evidence_changed",
            reason="checkpoint evidence changed while readiness was verified",
            training_report_sha256=after["report"],
            **common,
        )
    return LearnedReviewerReadinessReport(
        learned_inference_ready=True,
        status="production_checkpoint_ready",
        reason=(
            "trained MLX reviewer passed multi-song data, held-out ranking, "
            "calibration, technical-policy, and authorization gates"
        ),
        training_report_sha256=before["report"],
        **common,
    )


def load_production_learned_reviewer(
    checkpoint_directory: Path,
    production_training_report: Path,
) -> tuple[
    StageEditLearnedReviewer,
    LearnedReviewerCheckpointMetadata,
    LearnedReviewerReadinessReport,
    Mapping[str, str],
]:
    """Strictly bind the loaded model to one immutable evidence snapshot."""

    checkpoint = Path(checkpoint_directory).expanduser().resolve()
    requested_report = Path(production_training_report).expanduser().resolve()
    expected_report = (checkpoint / LEARNED_REVIEWER_REPORT_FILENAME).resolve()
    if requested_report != expected_report:
        raise LearnedReviewerNotReadyError(
            "requested reviewer report is not the exact colocated checkpoint report"
        )
    if not requested_report.is_file():
        raise LearnedReviewerNotReadyError("requested reviewer report does not exist")
    metadata, metadata_path, weight_path = _read_checkpoint_metadata(checkpoint)
    before = {
        "metadata_sha256": _sha256_file(metadata_path),
        "training_report_sha256": _sha256_file(requested_report),
        "weight_sha256": _sha256_file(weight_path),
    }
    readiness = learned_reviewer_readiness_report(checkpoint)
    if not (
        readiness.architecture_available
        and readiness.learned_inference_ready
        and readiness.status == "production_checkpoint_ready"
        and readiness.checkpoint_path == str(checkpoint)
        and readiness.checkpoint_id == metadata.checkpoint_id
        and readiness.training_steps == metadata.training_steps
        and readiness.metadata_sha256 == before["metadata_sha256"]
        and readiness.training_report_sha256
        == before["training_report_sha256"]
        and readiness.weight_sha256 == before["weight_sha256"]
    ):
        raise LearnedReviewerNotReadyError(
            "reviewer production checkpoint rejected: "
            f"{readiness.status}: {readiness.reason}"
        )
    model, loaded_metadata = load_learned_reviewer_checkpoint(
        checkpoint,
        expected_schema=metadata.feature_schema,
        require_trained=True,
    )
    after = {
        "metadata_sha256": _sha256_file(metadata_path),
        "training_report_sha256": _sha256_file(requested_report),
        "weight_sha256": _sha256_file(weight_path),
    }
    if before != after:
        raise LearnedReviewerNotReadyError(
            "reviewer checkpoint evidence changed while it was being loaded"
        )
    if loaded_metadata != metadata or metadata.weight_sha256 != after["weight_sha256"]:
        raise LearnedReviewerNotReadyError(
            "loaded reviewer identity does not match verified evidence"
        )
    evidence = dict(before)
    evidence["bundle_sha256"] = _canonical_sha256(before)
    if hasattr(model, "eval"):
        model.eval()
    return model, metadata, readiness, evidence


@dataclass(frozen=True, slots=True)
class LearnedReviewerTrainingResult:
    checkpoint_directory: Path
    report_path: Path
    best_epoch: int
    best_training_step: int
    report: Mapping[str, Any]
    readiness: LearnedReviewerReadinessReport


def _empty_heldout_evaluation(
    loss_components: Mapping[str, float],
) -> LearnedReviewerEvaluation:
    return LearnedReviewerEvaluation(
        example_count=0,
        human_count=0,
        worsened_count=0,
        rank_pair_count=0,
        quality_label_count=0,
        human_vs_worsened_auc=None,
        mean_rank_margin=None,
        brier_calibration_error=None,
        technical_zero_tolerance_policy_verified=False,
        passed=False,
        loss_components=dict(loss_components),
    )


def train_learned_reviewer(
    corpus: LearnedReviewerCorpus,
    *,
    checkpoint_directory: Path,
    model_config: LearnedReviewerConfig | None = None,
    epochs: int = 40,
    early_stopping_patience: int = 8,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-3,
    gradient_clip_norm: float = 1.0,
    heldout_fraction: float = 0.25,
    seed: int = 20260824,
    authorize_production: bool = False,
    device: Any = None,
    loss_contract: LearnedReviewerLossContract = LearnedReviewerLossContract(),
) -> LearnedReviewerTrainingResult:
    """Train, reload, evaluate, and emit a fail-closed evidence bundle."""

    _require_mlx()
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise LearnedReviewerInputError("epochs must be a positive integer")
    if (
        isinstance(early_stopping_patience, bool)
        or not isinstance(early_stopping_patience, int)
        or early_stopping_patience < 1
    ):
        raise LearnedReviewerInputError(
            "early_stopping_patience must be a positive integer"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise LearnedReviewerInputError("seed must be an integer")
    authorization = _strict_bool(
        authorize_production, "authorize_production"
    )
    for value, name in (
        (learning_rate, "learning_rate"),
        (weight_decay, "weight_decay"),
        (gradient_clip_norm, "gradient_clip_norm"),
    ):
        number = _finite(value, name)
        if number < 0.0:
            raise LearnedReviewerInputError(f"{name} cannot be negative")
    config = model_config or default_learned_reviewer_config(corpus.feature_schema)
    if config.feature_schema_fingerprint != corpus.feature_schema.fingerprint:
        raise LearnedReviewerInputError("training model config and corpus differ")
    if max(len(example.boundaries) for example in corpus.examples) > config.max_boundaries:
        raise LearnedReviewerInputError("corpus exceeds reviewer max_boundaries")
    split = split_learned_reviewer_corpus(
        corpus, heldout_fraction=heldout_fraction, seed=seed
    )
    data_gate = learned_reviewer_production_data_gate(corpus, split)
    active_device = device if device is not None else mx.default_device()
    checkpoint_root = Path(checkpoint_directory).expanduser().resolve()
    report_path = checkpoint_root / LEARNED_REVIEWER_REPORT_FILENAME
    with mx.stream(active_device):
        mx.random.seed(seed)
        model = StageEditLearnedReviewer(config)
        train_batch, train_supervision = tensorize_learned_reviewer_corpus(
            corpus, split.train_example_ids, config, device=active_device
        )
        optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        def loss_function(trained_model: StageEditLearnedReviewer) -> Any:
            return learned_reviewer_loss(
                trained_model,
                train_batch,
                train_supervision,
                loss_contract,
            )[0]

        loss_and_grad = nn.value_and_grad(model, loss_function)
        clipping_supported = hasattr(optim, "clip_grad_norm")
        best_validation_loss = math.inf
        best_epoch = 0
        best_step = 0
        epochs_without_improvement = 0
        history: list[dict[str, Any]] = []
        optimizer_steps = 0
        checkpoint_id = (
            f"learned-reviewer-{corpus.sha256[:16]}-{seed}"
        )
        saved_metadata: LearnedReviewerCheckpointMetadata | None = None
        for epoch in range(1, epochs + 1):
            model.train()
            loss, gradients = loss_and_grad(model)
            if clipping_supported:
                gradients, gradient_norm = optim.clip_grad_norm(
                    gradients, gradient_clip_norm
                )
            else:  # pragma: no cover - pinned MLX supports clipping.
                gradient_norm = mx.array(0.0)
            optimizer.update(model, gradients)
            optimizer_steps += 1
            mx.eval(loss, gradient_norm, model.parameters(), optimizer.state)
            training_evaluation = evaluate_learned_reviewer(
                model,
                corpus,
                split.train_example_ids,
                contract=loss_contract,
                device=active_device,
            )
            if split.heldout_example_ids:
                validation_evaluation = evaluate_learned_reviewer(
                    model,
                    corpus,
                    split.heldout_example_ids,
                    contract=loss_contract,
                    device=active_device,
                )
            else:
                validation_evaluation = training_evaluation
            validation_loss = validation_evaluation.loss_components["total"]
            loss_value = float(np.asarray(loss))
            gradient_value = float(np.asarray(gradient_norm))
            if not all(
                math.isfinite(value)
                for value in (loss_value, validation_loss, gradient_value)
            ):
                raise LearnedReviewerInputError(
                    "training produced non-finite loss or gradients"
                )
            improved = validation_loss < best_validation_loss - 1.0e-8
            history.append(
                {
                    "epoch": epoch,
                    "optimizer_step": optimizer_steps,
                    "train_total_loss": training_evaluation.loss_components["total"],
                    "validation_total_loss": validation_loss,
                    "gradient_norm_before_clip": gradient_value,
                    "validation_improved": improved,
                }
            )
            if improved:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_step = optimizer_steps
                epochs_without_improvement = 0
                metadata = new_learned_reviewer_checkpoint_metadata(
                    checkpoint_id,
                    corpus,
                    config,
                    training_steps=best_step,
                    weights_trained=True,
                )
                saved_metadata = save_learned_reviewer_checkpoint(
                    checkpoint_root, model, metadata
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    break
    if saved_metadata is None or best_epoch < 1 or best_step < 1:
        raise LearnedReviewerInputError("training never produced a finite checkpoint")
    reloaded_model, reloaded_metadata = load_learned_reviewer_checkpoint(
        checkpoint_root,
        expected_schema=corpus.feature_schema,
        require_trained=True,
    )
    if reloaded_metadata != saved_metadata:
        raise LearnedReviewerNotReadyError(
            "reloaded best checkpoint does not match saved metadata"
        )
    training_evaluation = evaluate_learned_reviewer(
        reloaded_model,
        corpus,
        split.train_example_ids,
        contract=loss_contract,
        device=active_device,
    )
    heldout_evaluation = (
        evaluate_learned_reviewer(
            reloaded_model,
            corpus,
            split.heldout_example_ids,
            contract=loss_contract,
            device=active_device,
        )
        if split.heldout_example_ids
        else _empty_heldout_evaluation(training_evaluation.loss_components)
    )
    metadata_path = checkpoint_root / "metadata.json"
    weight_path = checkpoint_root / reloaded_metadata.weight_filename
    metadata_sha256 = _sha256_file(metadata_path)
    weight_sha256 = _sha256_file(weight_path)
    production_ready = bool(
        authorization and data_gate.passed and heldout_evaluation.passed
    )
    heldout_payload = {
        "passed": heldout_evaluation.passed,
        "checkpoint_id": reloaded_metadata.checkpoint_id,
        "weight_sha256": weight_sha256,
        "corpus_sha256": corpus.sha256,
        "split_level": split.split_level,
        "training_song_ids": list(split.training_song_ids),
        "heldout_song_ids": list(split.heldout_song_ids),
        "human_count": heldout_evaluation.human_count,
        "worsened_count": heldout_evaluation.worsened_count,
        "rank_pair_count": heldout_evaluation.rank_pair_count,
        "quality_label_count": heldout_evaluation.quality_label_count,
        "human_vs_worsened_auc": heldout_evaluation.human_vs_worsened_auc,
        "mean_rank_margin": heldout_evaluation.mean_rank_margin,
        "brier_calibration_error": (
            heldout_evaluation.brier_calibration_error
        ),
        "technical_zero_tolerance_policy_verified": (
            heldout_evaluation.technical_zero_tolerance_policy_verified
        ),
        "thresholds": {
            "minimum_human_vs_worsened_auc": (
                MINIMUM_HELDOUT_HUMAN_WORSENED_AUROC
            ),
            "minimum_mean_rank_margin": MINIMUM_HELDOUT_MEAN_RANK_MARGIN,
            "maximum_brier_calibration_error": (
                MAXIMUM_HELDOUT_BRIER_CALIBRATION_ERROR
            ),
        },
        "loss_components": dict(heldout_evaluation.loss_components),
    }
    report: dict[str, Any] = {
        "schema_version": LEARNED_REVIEWER_REPORT_VERSION,
        "run_kind": LEARNED_REVIEWER_RUN_KIND,
        "checkpoint_scope": LEARNED_REVIEWER_CHECKPOINT_SCOPE,
        "production_ready": production_ready,
        "production_activation_authorized": authorization,
        "corpus": {
            "corpus_id": corpus.corpus_id,
            "corpus_sha256": corpus.sha256,
            "feature_schema_fingerprint": corpus.feature_schema.fingerprint,
        },
        "checkpoint": {
            "checkpoint_id": reloaded_metadata.checkpoint_id,
            "training_steps": reloaded_metadata.training_steps,
            "weight_sha256": weight_sha256,
            "metadata_sha256": metadata_sha256,
        },
        "production_data_gate": data_gate.to_dict(),
        "split": {
            "split_level": split.split_level,
            "training_song_ids": list(split.training_song_ids),
            "heldout_song_ids": list(split.heldout_song_ids),
        },
        "held_out_evaluation": heldout_payload,
        "training": {
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_training_step": best_step,
            "seed": seed,
            "loss_contract": asdict(loss_contract),
            "history": history,
        },
    }
    _atomic_write_json(report_path, report)
    readiness = learned_reviewer_readiness_report(checkpoint_root)
    if production_ready and not readiness.learned_inference_ready:
        raise LearnedReviewerNotReadyError(
            "trainer emitted production_ready evidence that strict readiness rejected: "
            f"{readiness.reason}"
        )
    return LearnedReviewerTrainingResult(
        checkpoint_directory=checkpoint_root,
        report_path=report_path.resolve(),
        best_epoch=best_epoch,
        best_training_step=best_step,
        report=report,
        readiness=readiness,
    )
