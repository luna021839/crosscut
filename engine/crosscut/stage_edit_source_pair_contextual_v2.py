"""Production-v2 causal contextual source-pair scorer core.

This module is deliberately independent from the semi-Markov decoder, model
logits, and reference-match evidence.  It freezes the successful contextual
feature design from the offline diagnostic as a shared train/serve
materializer, then fits the exact song/reference/group-balanced listwise model
selected by the diagnostic sweep.

Only current-token music, section, style, and source observations are accepted.
Recent edit history must already be zero-filled and missing-masked by the
global StageEdit contract.  The runtime evidence object contains no availability
gate, selected-source label, model output, or reference attribution field.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from .stage_edit_continuity_authority import (
    PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
)
from .stage_edit_reference_adapter import ReferenceTrainingCorpus
from .stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES,
    materialize_stage_edit_source_pair_features,
)
from .stage_edit_source_pair_training import (
    STAGE_EDIT_SOURCE_PAIR_HANDCRAFTED_BASELINE_CONTRACT,
    StageEditSourcePairActivationGate,
    StageEditSourcePairObservation,
    StageEditSourcePairRankingMetrics,
    StageEditSourcePairTrainingError,
    _handcrafted_pair_score,
    materialize_stage_edit_source_pair_training_corpus,
    stage_edit_source_pair_no_worse_activation_gate,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_RECENT_HISTORY_MASK_CONTRACT,
    SourceCandidateToken,
    StageEditSequence,
    StageEditToken,
    StageEditTokenSchema,
    mask_stage_edit_recent_history_for_semimarkov,
    stage_edit_recent_history_mask_contract,
)


STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_AUTHORITY = (
    PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
)
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCHEMA_VERSION = 2
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FORMAT_VERSION = 2
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_SCHEMA_VERSION = 2
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FILENAME = (
    "source-pair-contextual-scorer-v2.json"
)
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_FILENAME = (
    "source-pair-contextual-production-report-v2.json"
)
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CORPUS_FILENAME = (
    "source-pair-contextual-training-corpus-v2.json"
)
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_CONTRACT = (
    "current_token_causal_music_section_style_source_pair_embedding_id3_v2"
)
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RUNTIME_EVIDENCE_CONTRACT = (
    "masked_current_token_context_and_canonical_source_evidence_v2"
)
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT = (
    "song_reference_group_balanced_listwise_signed_l1_contextual_residual_v2"
)
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_OBJECTIVE_CONTRACT = (
    "architecture_A_listwise_softmax_ce_zero_margin_temperature_one_v2"
)
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BOUNDED_SCORE_CONTRACT = (
    "raw_minus_1p5_div_4_global_affine_rank_preserving_to_minus1_zero_v1"
)
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CROSS_VALIDATION_CONTRACT = (
    "exact_leave_one_song_out_all_fold_and_aggregate_no_worse_gate_v2"
)
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_REPORT_CONTRACT = (
    "tamper_bound_contextual_source_pair_checkpoint_and_loso_report_v2"
)

STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_HANDCRAFTED_BASELINE_FRACTION = 0.30
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_UNIFORM_BASELINE_FRACTION = 0.70
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SIGNED_L1_RADIUS = 1.5
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_L2_PENALTY = 0.05
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_TRAINING_STEPS = 160
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_LEARNING_RATE = 0.5
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BACKTRACK_ATTEMPTS = 24
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_TEMPERATURE = 1.0
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_MARGIN = 0.0
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_TOLERANCE = 1.0e-12
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RAW_MINIMUM = -2.5
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RAW_MAXIMUM = 1.5

_SCALAR_SOURCE_FEATURE_NAMES = (
    "sync_confidence",
    "sync_residual_normalized",
    "shot_scale",
    "face_scale",
    "face_center_x",
    "face_center_y",
    "body_center_x",
    "body_center_y",
    "pose_motion",
    "camera_motion_x",
    "camera_motion_y",
    "camera_zoom_velocity",
    "camera_roll_degrees_normalized",
    "sharpness",
    "exposure",
    "color_temperature_normalized",
    "occlusion_probability",
    "visual_match_confidence",
    "pose_model_confidence",
    "detected_person_count_normalized",
    "formation_person_count_normalized",
    "formation_pose_coverage",
)

_CONTEXT_FEATURE_NAMES = (
    "music.downbeat_probability",
    "music.onset_strength",
    "music.rms_energy",
    "music.spectral_flux",
    "music.vocal_probability",
    "music.phrase_progress",
    "section.intro_probability",
    "section.verse_probability",
    "section.prechorus_probability",
    "section.chorus_probability",
    "section.bridge_probability",
    "section.outro_probability",
    "section.section_progress",
    "section.section_energy_delta",
    "style.target_cut_density",
    "style.matchcut_strength",
    "style.motion_transition_strength",
    "style.closeup_preference",
    "style.source_diversity_preference",
    "style.color_continuity_preference",
)

_INTERACTION_PAIR_FEATURE_NAMES = (
    "face_scale_log_delta",
    "face_center_distance",
    "body_center_distance_fallback",
    "pose_motion_delta",
    "detected_person_count_delta",
    "formation_person_count_delta",
    "formation_pose_coverage_delta",
    "camera_motion_distance",
    "camera_zoom_velocity_delta",
    "exposure_delta",
    "color_temperature_delta",
)

_INTERACTION_SOURCE_FEATURE_NAMES = (
    "shot_scale",
    "face_scale",
    "pose_motion",
    "camera_zoom_velocity",
    "detected_person_count_normalized",
    "formation_person_count_normalized",
    "formation_pose_coverage",
    "occlusion_probability",
)

_EMBEDDING_PREFIXES = (
    "pose_model_embedding_",
    "formation_layout_embedding_",
    "formation_pose_person_",
)

_PAIR_OBSERVATION_TO_SOURCE_NAME = {
    "face_scale": "face_scale",
    "face_center_x": "face_center_x",
    "face_center_y": "face_center_y",
    "body_center_x": "body_center_x",
    "body_center_y": "body_center_y",
    "pose_motion": "pose_motion",
    "camera_motion_x": "camera_motion_x",
    "camera_motion_y": "camera_motion_y",
    "camera_zoom_velocity": "camera_zoom_velocity",
    "camera_roll_normalized": "camera_roll_degrees_normalized",
    "sharpness": "sharpness",
    "exposure": "exposure",
    "color_temperature_normalized": "color_temperature_normalized",
    "detected_person_count_normalized": "detected_person_count_normalized",
    "formation_person_count_normalized": "formation_person_count_normalized",
    "formation_pose_coverage": "formation_pose_coverage",
}


class StageEditContextualSourcePairV2Error(StageEditSourcePairTrainingError):
    """The v2 causal evidence, fit, checkpoint, or report is invalid."""


class StageEditContextualSourcePairV2ArtifactError(
    StageEditContextualSourcePairV2Error
):
    """A v2 checkpoint or report is missing, malformed, or tampered with."""


def _name(value: object, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise StageEditContextualSourcePairV2Error(f"{field} cannot be blank")
    return result


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise StageEditContextualSourcePairV2Error(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StageEditContextualSourcePairV2Error(
            f"{field} must be numeric"
        ) from exc
    if not math.isfinite(result):
        raise StageEditContextualSourcePairV2Error(f"{field} must be finite")
    return result


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StageEditContextualSourcePairV2Error(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise StageEditContextualSourcePairV2Error(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _strict_keys(payload: Mapping[str, Any], expected: set[str], field: str) -> None:
    if set(payload) != expected:
        raise StageEditContextualSourcePairV2Error(
            f"{field} keys are not exact; "
            f"missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )


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


def _bounded_with_clip(value: object, field: str) -> tuple[float, bool]:
    raw = _finite(value, field)
    bounded = min(1.0, max(-1.0, raw))
    return bounded, bounded != raw


def _exact_sorted_names(values: Sequence[object], field: str) -> tuple[str, ...]:
    names = tuple(_name(value, field) for value in values)
    if not names or len(names) != len(set(names)):
        raise StageEditContextualSourcePairV2Error(
            f"{field} must be non-empty and unique"
        )
    # Source ordering is never semantic.  Canonicalize here so both the hash and
    # score-by-id behavior are invariant to pipeline or candidate enumeration.
    return tuple(sorted(names))


def _contextual_feature_names() -> tuple[str, ...]:
    names: list[str] = [
        f"pair.{name}" for name in STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES
    ]
    for source_name in _SCALAR_SOURCE_FEATURE_NAMES:
        names.extend(
            (
                f"incoming.{source_name}",
                f"incoming_missing.{source_name}",
                f"outgoing.{source_name}",
                f"outgoing_missing.{source_name}",
                f"directed.{source_name}",
                f"directed_missing.{source_name}",
                f"product.{source_name}",
            )
        )
    for prefix in _EMBEDDING_PREFIXES:
        names.extend(
            (
                f"embedding.{prefix}incoming_norm",
                f"embedding.{prefix}outgoing_norm",
                f"embedding.{prefix}cosine",
                f"embedding.{prefix}distance",
                f"embedding.{prefix}missing_fraction",
            )
        )
    for side in ("incoming", "outgoing"):
        for shot_size in ("wide", "medium", "close"):
            names.append(f"shot_size.{side}.{shot_size}")
    interaction_core = tuple(
        sorted(
            (
                *(f"pair.{name}" for name in _INTERACTION_PAIR_FEATURE_NAMES),
                *(
                    value
                    for source_name in _INTERACTION_SOURCE_FEATURE_NAMES
                    for value in (
                        f"incoming.{source_name}",
                        f"directed.{source_name}",
                    )
                ),
            )
        )
    )
    for context_name in _CONTEXT_FEATURE_NAMES:
        for core_name in interaction_core:
            names.append(f"interaction.{context_name}*{core_name}")
    result = tuple(names)
    if len(result) != 729 or len(result) != len(set(result)):
        raise RuntimeError("frozen contextual id3 feature schema is invalid")
    if any("recent" in name or "logit" in name for name in result):
        raise RuntimeError("frozen contextual id3 schema contains prohibited inputs")
    return result


STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES = _contextual_feature_names()
STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_SCHEMA_SHA256 = _canonical_sha256(
    {
        "schema_version": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCHEMA_VERSION,
        "contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_CONTRACT,
        "feature_names": list(STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES),
    }
)


@dataclass(frozen=True, slots=True)
class StageEditContextualSourceEvidenceV2:
    source_id: str
    derivative_family_id: int
    values: tuple[float, ...]
    missing_mask: tuple[bool, ...]
    source_feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _name(self.source_id, "source_id"))
        _integer(self.derivative_family_id, "derivative_family_id")
        names = tuple(_name(value, "source_feature_name") for value in self.source_feature_names)
        if not names or len(names) != len(set(names)):
            raise StageEditContextualSourcePairV2Error(
                "source feature names must be non-empty and unique"
            )
        values = tuple(
            _finite(value, f"source values[{index}]")
            for index, value in enumerate(self.values)
        )
        missing = tuple(self.missing_mask)
        if len(values) != len(names) or len(missing) != len(names):
            raise StageEditContextualSourcePairV2Error(
                "source evidence dimensions differ from names"
            )
        if any(not isinstance(value, (bool, np.bool_)) for value in missing):
            raise StageEditContextualSourcePairV2Error(
                "source missing mask must be boolean"
            )
        if any(value != 0.0 for value, is_missing in zip(values, missing) if is_missing):
            raise StageEditContextualSourcePairV2Error(
                "missing source evidence must be zero-filled"
            )
        object.__setattr__(self, "source_feature_names", names)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "missing_mask", tuple(bool(value) for value in missing))

    @property
    def by_name(self) -> dict[str, float]:
        return dict(zip(self.source_feature_names, self.values, strict=True))

    @property
    def missing_by_name(self) -> dict[str, bool]:
        return dict(zip(self.source_feature_names, self.missing_mask, strict=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "derivative_family_id": self.derivative_family_id,
            "values": list(self.values),
            "missing_mask": list(self.missing_mask),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        source_feature_names: Sequence[str],
    ) -> "StageEditContextualSourceEvidenceV2":
        if not isinstance(payload, Mapping):
            raise StageEditContextualSourcePairV2Error(
                "source evidence must be an object"
            )
        _strict_keys(
            payload,
            {"source_id", "derivative_family_id", "values", "missing_mask"},
            "source evidence",
        )
        if not isinstance(payload.get("values"), list) or not isinstance(
            payload.get("missing_mask"), list
        ):
            raise StageEditContextualSourcePairV2Error(
                "source evidence values and missing mask must be arrays"
            )
        return cls(
            source_id=payload["source_id"],
            derivative_family_id=_integer(
                payload["derivative_family_id"], "derivative_family_id"
            ),
            values=tuple(payload["values"]),
            missing_mask=tuple(payload["missing_mask"]),
            source_feature_names=tuple(source_feature_names),
        )


@dataclass(frozen=True, slots=True)
class StageEditContextualRuntimeFrameV2:
    master_time_seconds: float
    token_schema_fingerprint: str
    source_feature_names: tuple[str, ...]
    context_feature_names: tuple[str, ...]
    context_values: tuple[float, ...]
    context_missing_mask: tuple[bool, ...]
    sources: tuple[StageEditContextualSourceEvidenceV2, ...]
    clipped_observed_value_count: int
    recent_history_mask_contract_sha256: str
    runtime_evidence_sha256: str
    contract: str = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RUNTIME_EVIDENCE_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RUNTIME_EVIDENCE_CONTRACT:
            raise StageEditContextualSourcePairV2Error(
                "runtime evidence contract mismatch"
            )
        object.__setattr__(
            self,
            "master_time_seconds",
            _finite(self.master_time_seconds, "master_time_seconds"),
        )
        object.__setattr__(
            self,
            "token_schema_fingerprint",
            _sha256(self.token_schema_fingerprint, "token_schema_fingerprint"),
        )
        object.__setattr__(
            self,
            "recent_history_mask_contract_sha256",
            _sha256(
                self.recent_history_mask_contract_sha256,
                "recent_history_mask_contract_sha256",
            ),
        )
        source_names = tuple(self.source_feature_names)
        context_names = tuple(self.context_feature_names)
        if context_names != _CONTEXT_FEATURE_NAMES:
            raise StageEditContextualSourcePairV2Error(
                "runtime context feature schema is not frozen id3"
            )
        if not source_names or len(source_names) != len(set(source_names)):
            raise StageEditContextualSourcePairV2Error(
                "runtime source feature schema is invalid"
            )
        context_values = tuple(
            _finite(value, f"context values[{index}]")
            for index, value in enumerate(self.context_values)
        )
        context_missing = tuple(self.context_missing_mask)
        if len(context_values) != len(context_names) or len(context_missing) != len(
            context_names
        ):
            raise StageEditContextualSourcePairV2Error(
                "runtime context dimensions differ from id3"
            )
        if any(not isinstance(value, (bool, np.bool_)) for value in context_missing):
            raise StageEditContextualSourcePairV2Error(
                "runtime context missing mask must be boolean"
            )
        if any(
            value != 0.0
            for value, is_missing in zip(context_values, context_missing)
            if is_missing
        ):
            raise StageEditContextualSourcePairV2Error(
                "missing context values must be zero-filled"
            )
        sources = tuple(self.sources)
        if not sources or any(
            not isinstance(value, StageEditContextualSourceEvidenceV2)
            for value in sources
        ):
            raise StageEditContextualSourcePairV2Error(
                "runtime frame needs typed source evidence"
            )
        source_ids = tuple(value.source_id for value in sources)
        if source_ids != tuple(sorted(set(source_ids))):
            raise StageEditContextualSourcePairV2Error(
                "runtime source ids must be unique and sorted"
            )
        if any(value.source_feature_names != source_names for value in sources):
            raise StageEditContextualSourcePairV2Error(
                "runtime source evidence schema changed within frame"
            )
        clipped_count = _integer(
            self.clipped_observed_value_count,
            "clipped_observed_value_count",
        )
        expected_clipped_count = sum(
            int(not -1.0 <= value <= 1.0)
            for value, is_missing in zip(
                context_values, context_missing, strict=True
            )
            if not is_missing
        ) + sum(
            int(not -1.0 <= value <= 1.0)
            for source in sources
            for value, is_missing in zip(
                source.values, source.missing_mask, strict=True
            )
            if not is_missing
        )
        if clipped_count != expected_clipped_count:
            raise StageEditContextualSourcePairV2Error(
                "runtime clipping audit does not match observed evidence"
            )
        object.__setattr__(self, "source_feature_names", source_names)
        object.__setattr__(self, "context_feature_names", context_names)
        object.__setattr__(self, "context_values", context_values)
        object.__setattr__(
            self,
            "context_missing_mask",
            tuple(bool(value) for value in context_missing),
        )
        object.__setattr__(self, "sources", sources)
        digest = _sha256(self.runtime_evidence_sha256, "runtime_evidence_sha256")
        object.__setattr__(self, "runtime_evidence_sha256", digest)
        if digest != _canonical_sha256(self.unsigned_dict()):
            raise StageEditContextualSourcePairV2Error(
                "runtime evidence digest does not match contents"
            )

    @property
    def source_by_id(self) -> dict[str, StageEditContextualSourceEvidenceV2]:
        return {value.source_id: value for value in self.sources}

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "master_time_seconds": self.master_time_seconds,
            "token_schema_fingerprint": self.token_schema_fingerprint,
            "source_feature_names": list(self.source_feature_names),
            "context_feature_names": list(self.context_feature_names),
            "context_values": list(self.context_values),
            "context_missing_mask": list(self.context_missing_mask),
            "sources": [value.to_dict() for value in self.sources],
            "clipped_observed_value_count": self.clipped_observed_value_count,
            "recent_history_mask_contract_sha256": (
                self.recent_history_mask_contract_sha256
            ),
            "recent_edit_features_consumed": False,
            "model_logits_consumed": False,
            "reference_match_input_evidence_consumed": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.unsigned_dict(),
            "runtime_evidence_sha256": self.runtime_evidence_sha256,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "StageEditContextualRuntimeFrameV2":
        expected = {
            "contract",
            "master_time_seconds",
            "token_schema_fingerprint",
            "source_feature_names",
            "context_feature_names",
            "context_values",
            "context_missing_mask",
            "sources",
            "clipped_observed_value_count",
            "recent_history_mask_contract_sha256",
            "recent_edit_features_consumed",
            "model_logits_consumed",
            "reference_match_input_evidence_consumed",
            "runtime_evidence_sha256",
        }
        if not isinstance(payload, Mapping):
            raise StageEditContextualSourcePairV2Error(
                "runtime frame must be an object"
            )
        _strict_keys(payload, expected, "runtime frame")
        for field in (
            "source_feature_names",
            "context_feature_names",
            "context_values",
            "context_missing_mask",
            "sources",
        ):
            if not isinstance(payload.get(field), list):
                raise StageEditContextualSourcePairV2Error(
                    f"runtime frame {field} must be an array"
                )
        for field in (
            "recent_edit_features_consumed",
            "model_logits_consumed",
            "reference_match_input_evidence_consumed",
        ):
            if payload.get(field) is not False:
                raise StageEditContextualSourcePairV2Error(
                    f"runtime frame prohibited-input audit {field} must be false"
                )
        source_names = tuple(payload["source_feature_names"])
        if any(not isinstance(value, Mapping) for value in payload["sources"]):
            raise StageEditContextualSourcePairV2Error(
                "runtime frame sources must be objects"
            )
        return cls(
            contract=payload["contract"],
            master_time_seconds=payload["master_time_seconds"],
            token_schema_fingerprint=payload["token_schema_fingerprint"],
            source_feature_names=source_names,
            context_feature_names=tuple(payload["context_feature_names"]),
            context_values=tuple(payload["context_values"]),
            context_missing_mask=tuple(payload["context_missing_mask"]),
            sources=tuple(
                StageEditContextualSourceEvidenceV2.from_dict(
                    value, source_feature_names=source_names
                )
                for value in payload["sources"]
            ),
            clipped_observed_value_count=_integer(
                payload["clipped_observed_value_count"],
                "clipped_observed_value_count",
            ),
            recent_history_mask_contract_sha256=payload[
                "recent_history_mask_contract_sha256"
            ],
            runtime_evidence_sha256=payload["runtime_evidence_sha256"],
        )

    def audit_identity_dict(self) -> dict[str, Any]:
        """Return public-safe lineage without raw context or embeddings."""

        return {
            "contract": self.contract,
            "master_time_seconds": self.master_time_seconds,
            "token_schema_fingerprint": self.token_schema_fingerprint,
            "runtime_evidence_sha256": self.runtime_evidence_sha256,
            "recent_history_mask_contract_sha256": (
                self.recent_history_mask_contract_sha256
            ),
            "clipped_observed_value_count": self.clipped_observed_value_count,
            "sources": [
                {
                    "source_id": source.source_id,
                    "derivative_family_id": source.derivative_family_id,
                }
                for source in self.sources
            ],
        }


def _masked_context_from_token(
    token: StageEditToken,
    schema: StageEditTokenSchema,
) -> tuple[tuple[float, ...], tuple[bool, ...], int]:
    dimensions = (
        len(schema.music_feature_names),
        len(schema.section_feature_names),
        len(schema.recent_edit_feature_names),
        len(schema.style_feature_names),
    )
    if (
        len(token.music_features),
        len(token.section_features),
        len(token.recent_edit_features),
        len(token.style_features),
    ) != dimensions:
        raise StageEditContextualSourcePairV2Error(
            "runtime token dimensions differ from schema"
        )
    if len(token.missing_mask) != schema.raw_context_dimension:
        raise StageEditContextualSourcePairV2Error(
            "runtime token needs the explicit full context missing mask"
        )
    if any(not isinstance(value, (bool, np.bool_)) for value in token.missing_mask):
        raise StageEditContextualSourcePairV2Error(
            "runtime token context missing mask must be boolean"
        )
    recent_start = dimensions[0] + dimensions[1]
    recent_stop = recent_start + dimensions[2]
    if any(value != 0.0 for value in token.recent_edit_features) or not all(
        bool(value) for value in token.missing_mask[recent_start:recent_stop]
    ):
        raise StageEditContextualSourcePairV2Error(
            "runtime token recent history is not fully zero-filled and missing-masked"
        )
    by_name: dict[str, tuple[float, bool]] = {}
    clipped = 0

    def add(
        namespace: str,
        names: Sequence[str],
        values: Sequence[float],
        offset: int,
    ) -> None:
        nonlocal clipped
        for index, (name, raw) in enumerate(zip(names, values, strict=True)):
            key = f"{namespace}.{name}"
            is_missing = bool(token.missing_mask[offset + index])
            if is_missing:
                by_name[key] = (0.0, True)
            else:
                observed = _finite(raw, key)
                did_clip = not -1.0 <= observed <= 1.0
                by_name[key] = (observed, False)
                clipped += int(did_clip)

    add("music", schema.music_feature_names, token.music_features, 0)
    section_start = dimensions[0]
    add(
        "section",
        schema.section_feature_names,
        token.section_features,
        section_start,
    )
    style_start = recent_stop
    add("style", schema.style_feature_names, token.style_features, style_start)
    missing_names = tuple(sorted(set(_CONTEXT_FEATURE_NAMES) - set(by_name)))
    if missing_names:
        raise StageEditContextualSourcePairV2Error(
            "token schema lacks frozen id3 context features: " + ", ".join(missing_names)
        )
    values = tuple(by_name[name][0] for name in _CONTEXT_FEATURE_NAMES)
    missing = tuple(by_name[name][1] for name in _CONTEXT_FEATURE_NAMES)
    return values, missing, clipped


def materialize_stage_edit_contextual_runtime_frame_v2(
    token: StageEditToken,
    *,
    canonical_source_ids: Sequence[str],
    derivative_family_id_by_source: Mapping[str, int],
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
) -> StageEditContextualRuntimeFrameV2:
    """Sanitize one already-history-masked current token for train or serve."""

    if not isinstance(token, StageEditToken):
        raise StageEditContextualSourcePairV2Error(
            "runtime frame requires StageEditToken"
        )
    if not isinstance(schema, StageEditTokenSchema):
        raise StageEditContextualSourcePairV2Error(
            "runtime frame schema is invalid"
        )
    source_ids = _exact_sorted_names(canonical_source_ids, "canonical_source_ids")
    family_map = dict(derivative_family_id_by_source)
    if set(family_map) != set(source_ids):
        raise StageEditContextualSourcePairV2Error(
            "derivative family mapping must cover canonical sources exactly"
        )
    for source_id in source_ids:
        _integer(family_map[source_id], f"derivative family for {source_id}")
    context_values, context_missing, clipped = _masked_context_from_token(
        token, schema
    )
    candidates: dict[str, SourceCandidateToken] = {}
    for candidate in token.source_candidates:
        if not isinstance(candidate, SourceCandidateToken):
            raise StageEditContextualSourcePairV2Error(
                "runtime token source candidate is invalid"
            )
        source_id = _name(candidate.source_id, "candidate source_id")
        if source_id not in source_ids:
            raise StageEditContextualSourcePairV2Error(
                "runtime token contains a source outside the canonical source set"
            )
        if source_id in candidates:
            raise StageEditContextualSourcePairV2Error(
                "runtime token source ids must be unique"
            )
        candidates[source_id] = candidate
    source_feature_names = tuple(schema.source_feature_names)
    sources: list[StageEditContextualSourceEvidenceV2] = []
    for source_id in source_ids:
        candidate = candidates.get(source_id)
        if candidate is None:
            values = (0.0,) * len(source_feature_names)
            missing = (True,) * len(source_feature_names)
        else:
            if len(candidate.features) != len(source_feature_names):
                raise StageEditContextualSourcePairV2Error(
                    f"source {source_id} feature dimension differs from schema"
                )
            if candidate.missing_mask and len(candidate.missing_mask) != len(
                source_feature_names
            ):
                raise StageEditContextualSourcePairV2Error(
                    f"source {source_id} missing-mask dimension differs from schema"
                )
            missing = (
                tuple(candidate.missing_mask)
                if candidate.missing_mask
                else (False,) * len(source_feature_names)
            )
            if any(not isinstance(value, (bool, np.bool_)) for value in missing):
                raise StageEditContextualSourcePairV2Error(
                    f"source {source_id} missing mask must be boolean"
                )
            materialized: list[float] = []
            for index, (name, raw) in enumerate(
                zip(source_feature_names, candidate.features, strict=True)
            ):
                if bool(missing[index]):
                    materialized.append(0.0)
                else:
                    observed = _finite(raw, f"source {source_id}.{name}")
                    did_clip = not -1.0 <= observed <= 1.0
                    materialized.append(observed)
                    clipped += int(did_clip)
            values = tuple(materialized)
        sources.append(
            StageEditContextualSourceEvidenceV2(
                source_id=source_id,
                derivative_family_id=family_map[source_id],
                values=tuple(values),
                missing_mask=tuple(bool(value) for value in missing),
                source_feature_names=source_feature_names,
            )
        )
    mask_sha256 = stage_edit_recent_history_mask_contract(schema)["contract_sha256"]
    unsigned = {
        "contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RUNTIME_EVIDENCE_CONTRACT,
        "master_time_seconds": _finite(
            token.master_time_seconds, "master_time_seconds"
        ),
        "token_schema_fingerprint": schema.fingerprint,
        "source_feature_names": list(source_feature_names),
        "context_feature_names": list(_CONTEXT_FEATURE_NAMES),
        "context_values": list(context_values),
        "context_missing_mask": list(context_missing),
        "sources": [value.to_dict() for value in sources],
        "clipped_observed_value_count": clipped,
        "recent_history_mask_contract_sha256": mask_sha256,
        "recent_edit_features_consumed": False,
        "model_logits_consumed": False,
        "reference_match_input_evidence_consumed": False,
    }
    return StageEditContextualRuntimeFrameV2(
        master_time_seconds=unsigned["master_time_seconds"],
        token_schema_fingerprint=schema.fingerprint,
        source_feature_names=source_feature_names,
        context_feature_names=_CONTEXT_FEATURE_NAMES,
        context_values=context_values,
        context_missing_mask=context_missing,
        sources=tuple(sources),
        clipped_observed_value_count=clipped,
        recent_history_mask_contract_sha256=mask_sha256,
        runtime_evidence_sha256=_canonical_sha256(unsigned),
    )


def _source_observation(
    source: StageEditContextualSourceEvidenceV2,
) -> StageEditSourcePairObservation:
    values = source.by_name
    missing = source.missing_by_name
    missing_schema = tuple(
        sorted(set(_PAIR_OBSERVATION_TO_SOURCE_NAME.values()) - set(values))
    )
    if missing_schema:
        raise StageEditContextualSourcePairV2Error(
            "source schema lacks v1 pair observations: " + ", ".join(missing_schema)
        )
    return StageEditSourcePairObservation(
        **{
            field: None if missing[name] else values[name]
            for field, name in _PAIR_OBSERVATION_TO_SOURCE_NAME.items()
        }
    )


def _embedding_summary(
    prefix: str,
    outgoing: StageEditContextualSourceEvidenceV2,
    incoming: StageEditContextualSourceEvidenceV2,
) -> tuple[float, ...]:
    left_values = {
        name: _bounded_with_clip(value, f"outgoing embedding {name}")[0]
        for name, value in outgoing.by_name.items()
    }
    right_values = {
        name: _bounded_with_clip(value, f"incoming embedding {name}")[0]
        for name, value in incoming.by_name.items()
    }
    left_missing = outgoing.missing_by_name
    right_missing = incoming.missing_by_name
    names = tuple(sorted(name for name in left_values if name.startswith(prefix)))
    if not names or any(name not in right_values for name in names):
        raise StageEditContextualSourcePairV2Error(
            f"source schema lacks embedding block {prefix}"
        )
    observed = tuple(
        name
        for name in names
        if not left_missing[name] and not right_missing[name]
    )
    if not observed:
        return (0.0, 0.0, 0.0, 0.0, 1.0)
    left = np.asarray([left_values[name] for name in observed], dtype=np.float64)
    right = np.asarray([right_values[name] for name in observed], dtype=np.float64)
    left_norm_raw = float(np.linalg.norm(left))
    right_norm_raw = float(np.linalg.norm(right))
    normalizer = math.sqrt(len(observed))
    denominator = left_norm_raw * right_norm_raw
    # Keep this byte-for-byte aligned with the frozen diagnostic id3 builder.
    cosine = (
        float(np.dot(left, right)) / denominator
        if denominator > 1.0e-12
        else 0.0
    )
    distance = float(np.linalg.norm(right - left) / (2.0 * normalizer))
    return (
        min(1.0, right_norm_raw / normalizer),
        min(1.0, left_norm_raw / normalizer),
        min(1.0, max(-1.0, cosine)),
        min(1.0, distance),
        1.0 - len(observed) / len(names),
    )


def _shot_size_basis(value: float) -> tuple[float, float, float]:
    normalized = min(1.0, max(0.0, value))
    return (
        max(0.0, 1.0 - 2.0 * normalized),
        max(0.0, 1.0 - abs(2.0 * normalized - 1.0)),
        max(0.0, 2.0 * normalized - 1.0),
    )


@dataclass(frozen=True, slots=True)
class StageEditContextualPairFeatureVectorV2:
    outgoing_source_id: str
    incoming_source_id: str
    runtime_evidence_sha256: str
    values: tuple[float, ...]
    feature_names: tuple[str, ...] = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES
    feature_schema_sha256: str = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_SCHEMA_SHA256
    content_sha256: str = ""
    contract: str = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_CONTRACT:
            raise StageEditContextualSourcePairV2Error("pair feature contract mismatch")
        object.__setattr__(
            self,
            "outgoing_source_id",
            _name(self.outgoing_source_id, "outgoing_source_id"),
        )
        object.__setattr__(
            self,
            "incoming_source_id",
            _name(self.incoming_source_id, "incoming_source_id"),
        )
        if self.outgoing_source_id == self.incoming_source_id:
            raise StageEditContextualSourcePairV2Error(
                "contextual pair requires different exact sources"
            )
        object.__setattr__(
            self,
            "runtime_evidence_sha256",
            _sha256(self.runtime_evidence_sha256, "runtime_evidence_sha256"),
        )
        names = tuple(self.feature_names)
        if names != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES:
            raise StageEditContextualSourcePairV2Error(
                "pair feature names differ from frozen id3"
            )
        if self.feature_schema_sha256 != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_SCHEMA_SHA256:
            raise StageEditContextualSourcePairV2Error(
                "pair feature schema digest differs from frozen id3"
            )
        values = tuple(
            _finite(value, f"pair feature values[{index}]")
            for index, value in enumerate(self.values)
        )
        if len(values) != len(names) or any(not -1.0 <= value <= 1.0 for value in values):
            raise StageEditContextualSourcePairV2Error(
                "pair features have the wrong dimension or bounds"
            )
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "values", values)
        digest = _sha256(self.content_sha256, "content_sha256")
        object.__setattr__(self, "content_sha256", digest)
        if digest != _canonical_sha256(self.unsigned_dict()):
            raise StageEditContextualSourcePairV2Error(
                "pair feature digest does not match contents"
            )

    @property
    def by_name(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.values, strict=True))

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "outgoing_source_id": self.outgoing_source_id,
            "incoming_source_id": self.incoming_source_id,
            "runtime_evidence_sha256": self.runtime_evidence_sha256,
            "feature_names": list(self.feature_names),
            "feature_schema_sha256": self.feature_schema_sha256,
            "values": list(self.values),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "StageEditContextualPairFeatureVectorV2":
        expected = {
            "contract",
            "outgoing_source_id",
            "incoming_source_id",
            "runtime_evidence_sha256",
            "feature_names",
            "feature_schema_sha256",
            "values",
            "content_sha256",
        }
        if not isinstance(payload, Mapping):
            raise StageEditContextualSourcePairV2Error(
                "pair feature vector must be an object"
            )
        _strict_keys(payload, expected, "pair feature vector")
        if not isinstance(payload.get("feature_names"), list) or not isinstance(
            payload.get("values"), list
        ):
            raise StageEditContextualSourcePairV2Error(
                "pair feature names and values must be arrays"
            )
        return cls(
            contract=payload["contract"],
            outgoing_source_id=payload["outgoing_source_id"],
            incoming_source_id=payload["incoming_source_id"],
            runtime_evidence_sha256=payload["runtime_evidence_sha256"],
            feature_names=tuple(payload["feature_names"]),
            feature_schema_sha256=payload["feature_schema_sha256"],
            values=tuple(payload["values"]),
            content_sha256=payload["content_sha256"],
        )


def materialize_stage_edit_contextual_pair_features_v2(
    frame: StageEditContextualRuntimeFrameV2,
    outgoing_source_id: str,
    incoming_source_id: str,
) -> StageEditContextualPairFeatureVectorV2:
    """Build the exact frozen id3 vector from one sanitized runtime frame."""

    if not isinstance(frame, StageEditContextualRuntimeFrameV2):
        raise StageEditContextualSourcePairV2Error(
            "pair materialization requires a typed runtime frame"
        )
    outgoing_id = _name(outgoing_source_id, "outgoing_source_id")
    incoming_id = _name(incoming_source_id, "incoming_source_id")
    sources = frame.source_by_id
    if outgoing_id not in sources or incoming_id not in sources:
        raise StageEditContextualSourcePairV2Error(
            "pair source is absent from the canonical runtime frame"
        )
    outgoing = sources[outgoing_id]
    incoming = sources[incoming_id]
    if outgoing_id == incoming_id:
        raise StageEditContextualSourcePairV2Error(
            "pair materialization requires different exact sources"
        )
    left_observation = _source_observation(outgoing)
    right_observation = _source_observation(incoming)
    pair = materialize_stage_edit_source_pair_features(
        left_observation, right_observation
    ).by_name
    # The frozen diagnostic has two intentional views: the legacy v1 pair
    # materializer consumes raw finite observations, while all contextual
    # scalar/embedding/shot/interaction inputs consume per-source bounded
    # values.  Preserve both instead of clipping before the nonlinear v1 pair.
    left = {
        name: _bounded_with_clip(value, f"outgoing source {name}")[0]
        for name, value in outgoing.by_name.items()
    }
    right = {
        name: _bounded_with_clip(value, f"incoming source {name}")[0]
        for name, value in incoming.by_name.items()
    }
    left_missing = outgoing.missing_by_name
    right_missing = incoming.missing_by_name
    context = {
        name: _bounded_with_clip(value, f"runtime context {name}")[0]
        for name, value in zip(
            frame.context_feature_names, frame.context_values, strict=True
        )
    }
    items: list[tuple[str, float]] = [
        (f"pair.{name}", pair[name]) for name in STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES
    ]
    for source_name in _SCALAR_SOURCE_FEATURE_NAMES:
        if source_name not in left or source_name not in right:
            raise StageEditContextualSourcePairV2Error(
                f"source schema lacks scalar evidence {source_name}"
            )
        left_value = left[source_name]
        right_value = right[source_name]
        jointly_missing = left_missing[source_name] or right_missing[source_name]
        items.extend(
            (
                (f"incoming.{source_name}", right_value),
                (f"incoming_missing.{source_name}", float(right_missing[source_name])),
                (f"outgoing.{source_name}", left_value),
                (f"outgoing_missing.{source_name}", float(left_missing[source_name])),
                (
                    f"directed.{source_name}",
                    0.0
                    if jointly_missing
                    else min(1.0, max(-1.0, right_value - left_value)),
                ),
                (f"directed_missing.{source_name}", float(jointly_missing)),
                (
                    f"product.{source_name}",
                    0.0
                    if jointly_missing
                    else min(1.0, max(-1.0, right_value * left_value)),
                ),
            )
        )
    for prefix in _EMBEDDING_PREFIXES:
        for name, value in zip(
            (
                f"embedding.{prefix}incoming_norm",
                f"embedding.{prefix}outgoing_norm",
                f"embedding.{prefix}cosine",
                f"embedding.{prefix}distance",
                f"embedding.{prefix}missing_fraction",
            ),
            _embedding_summary(prefix, outgoing, incoming),
            strict=True,
        ):
            items.append((name, value))
    for side, shot_scale in (("incoming", right["shot_scale"]), ("outgoing", left["shot_scale"])):
        for size, value in zip(
            ("wide", "medium", "close"),
            _shot_size_basis(shot_scale),
            strict=True,
        ):
            items.append((f"shot_size.{side}.{size}", value))
    interaction_core: dict[str, float] = {
        f"pair.{name}": pair[name] for name in _INTERACTION_PAIR_FEATURE_NAMES
    }
    for source_name in _INTERACTION_SOURCE_FEATURE_NAMES:
        interaction_core[f"incoming.{source_name}"] = right[source_name]
        interaction_core[f"directed.{source_name}"] = (
            0.0
            if left_missing[source_name] or right_missing[source_name]
            else min(1.0, max(-1.0, right[source_name] - left[source_name]))
        )
    for context_name in _CONTEXT_FEATURE_NAMES:
        for core_name in sorted(interaction_core):
            items.append(
                (
                    f"interaction.{context_name}*{core_name}",
                    context[context_name] * interaction_core[core_name],
                )
            )
    names = tuple(name for name, _value in items)
    # Preserve the diagnostic's nonlinear pair construction on raw finite
    # observations, then apply the production bound to the final id3 vector.
    # Clipping source inputs first changes deltas (for example zoom 1.2->1.1)
    # and therefore is not train/serve equivalent to the frozen exploration.
    values = tuple(
        _bounded_with_clip(value, f"pair feature {name}")[0]
        for name, value in items
    )
    if names != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES:
        raise StageEditContextualSourcePairV2Error(
            "shared pair materializer drifted from frozen id3 schema"
        )
    unsigned = {
        "contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_CONTRACT,
        "outgoing_source_id": outgoing_id,
        "incoming_source_id": incoming_id,
        "runtime_evidence_sha256": frame.runtime_evidence_sha256,
        "feature_names": list(names),
        "feature_schema_sha256": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_SCHEMA_SHA256,
        "values": list(values),
    }
    return StageEditContextualPairFeatureVectorV2(
        outgoing_source_id=outgoing_id,
        incoming_source_id=incoming_id,
        runtime_evidence_sha256=frame.runtime_evidence_sha256,
        values=values,
        content_sha256=_canonical_sha256(unsigned),
    )


def stage_edit_contextual_pair_baselines_v2(
    frame: StageEditContextualRuntimeFrameV2,
    outgoing_source_id: str,
    incoming_source_id: str,
) -> tuple[float, float, float]:
    """Return uniform, handcrafted, and frozen 30/70 blended raw baselines."""

    features = materialize_stage_edit_contextual_pair_features_v2(
        frame, outgoing_source_id, incoming_source_id
    )
    pair_values = tuple(
        features.by_name[f"pair.{name}"]
        for name in STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES
    )
    uniform = -float(np.mean(np.asarray(pair_values, dtype=np.float64)))
    outgoing = _source_observation(frame.source_by_id[features.outgoing_source_id])
    incoming = _source_observation(frame.source_by_id[features.incoming_source_id])
    handcrafted = _handcrafted_pair_score(outgoing, incoming)
    blended = (
        STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_HANDCRAFTED_BASELINE_FRACTION
        * handcrafted
        + STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_UNIFORM_BASELINE_FRACTION
        * uniform
    )
    for name, value in (
        ("uniform baseline", uniform),
        ("handcrafted baseline", handcrafted),
        ("blended baseline", blended),
    ):
        if not -1.0 <= value <= 0.0:
            raise StageEditContextualSourcePairV2Error(f"{name} is outside [-1, 0]")
    return uniform, handcrafted, blended


def stage_edit_contextual_raw_to_decoder_score_v2(raw_logit: float) -> float:
    """Map the theoretical raw range [-2.5, 1.5] affinely to [-1, 0]."""

    raw = _finite(raw_logit, "raw_logit")
    if not (
        STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RAW_MINIMUM
        <= raw
        <= STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RAW_MAXIMUM
    ):
        raise StageEditContextualSourcePairV2Error(
            "contextual raw logit exceeds its theoretical L1-derived bounds"
        )
    score = (
        raw - STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RAW_MAXIMUM
    ) / (
        STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RAW_MAXIMUM
        - STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RAW_MINIMUM
    )
    if not -1.0 - 1.0e-12 <= score <= 1.0e-12:
        raise StageEditContextualSourcePairV2Error(
            "contextual decoder score exceeds [-1, 0]"
        )
    return min(0.0, max(-1.0, score))


assert stage_edit_contextual_raw_to_decoder_score_v2(
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RAW_MINIMUM
) == -1.0
assert stage_edit_contextual_raw_to_decoder_score_v2(
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RAW_MAXIMUM
) == 0.0


@dataclass(frozen=True, slots=True)
class StageEditContextualBoundaryScoreV2:
    outgoing_source_id: str
    incoming_source_id: str
    applied: bool
    neutral_reason: str | None
    uniform_baseline: float | None
    handcrafted_baseline: float | None
    blended_baseline: float | None
    signed_residual: float | None
    raw_logit: float
    decoder_score: float
    runtime_evidence_sha256: str
    pair_feature_sha256: str | None
    scorer_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "outgoing_source_id", _name(self.outgoing_source_id, "outgoing_source_id"))
        object.__setattr__(self, "incoming_source_id", _name(self.incoming_source_id, "incoming_source_id"))
        if not isinstance(self.applied, bool):
            raise StageEditContextualSourcePairV2Error("applied must be boolean")
        _sha256(self.runtime_evidence_sha256, "runtime_evidence_sha256")
        _sha256(self.scorer_fingerprint, "scorer_fingerprint")
        raw = _finite(self.raw_logit, "raw_logit")
        decoder = _finite(self.decoder_score, "decoder_score")
        if not -1.0 <= decoder <= 0.0:
            raise StageEditContextualSourcePairV2Error("decoder score must lie in [-1, 0]")
        if self.applied:
            if self.neutral_reason is not None or self.pair_feature_sha256 is None:
                raise StageEditContextualSourcePairV2Error("applied score lineage is incomplete")
            _sha256(self.pair_feature_sha256, "pair_feature_sha256")
            values = (
                self.uniform_baseline,
                self.handcrafted_baseline,
                self.blended_baseline,
                self.signed_residual,
            )
            if any(value is None for value in values):
                raise StageEditContextualSourcePairV2Error("applied score components are missing")
            finite_values = tuple(_finite(value, "score component") for value in values)
            uniform, handcrafted, blended, residual = finite_values
            if not math.isclose(
                blended,
                0.30 * handcrafted + 0.70 * uniform,
                abs_tol=1.0e-12,
            ) or not math.isclose(raw, blended + residual, abs_tol=1.0e-12):
                raise StageEditContextualSourcePairV2Error("applied score arithmetic is inconsistent")
            if not math.isclose(
                decoder,
                stage_edit_contextual_raw_to_decoder_score_v2(raw),
                abs_tol=1.0e-12,
            ):
                raise StageEditContextualSourcePairV2Error("bounded score transform is inconsistent")
        else:
            if self.neutral_reason not in {"same_exact_source", "same_derivative_family"}:
                raise StageEditContextualSourcePairV2Error("neutral score reason is unsupported")
            if any(
                value is not None
                for value in (
                    self.uniform_baseline,
                    self.handcrafted_baseline,
                    self.blended_baseline,
                    self.signed_residual,
                    self.pair_feature_sha256,
                )
            ) or raw != 0.0 or decoder != 0.0:
                raise StageEditContextualSourcePairV2Error("neutral boundary must bypass scoring exactly")
        object.__setattr__(self, "raw_logit", raw)
        object.__setattr__(self, "decoder_score", decoder)

    def to_dict(self) -> dict[str, Any]:
        """Return scalar score lineage only; never expose runtime tensors."""

        return {
            "outgoing_source_id": self.outgoing_source_id,
            "incoming_source_id": self.incoming_source_id,
            "applied": self.applied,
            "neutral_reason": self.neutral_reason,
            "uniform_baseline": self.uniform_baseline,
            "handcrafted_baseline": self.handcrafted_baseline,
            "blended_baseline": self.blended_baseline,
            "signed_residual": self.signed_residual,
            "raw_logit": self.raw_logit,
            "decoder_score": self.decoder_score,
            "runtime_evidence_sha256": self.runtime_evidence_sha256,
            "pair_feature_sha256": self.pair_feature_sha256,
            "scorer_fingerprint": self.scorer_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class StageEditContextualSourcePairScorerV2:
    token_schema_fingerprint: str
    weights: tuple[float, ...]
    feature_names: tuple[str, ...] = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES
    contract: str = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT
    authority: str = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_AUTHORITY

    def __post_init__(self) -> None:
        if self.authority != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_AUTHORITY:
            raise StageEditContextualSourcePairV2Error("contextual scorer authority mismatch")
        if self.contract != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT:
            raise StageEditContextualSourcePairV2Error("contextual scorer contract mismatch")
        object.__setattr__(
            self,
            "token_schema_fingerprint",
            _sha256(self.token_schema_fingerprint, "token_schema_fingerprint"),
        )
        if tuple(self.feature_names) != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES:
            raise StageEditContextualSourcePairV2Error("contextual scorer feature schema mismatch")
        weights = tuple(
            _finite(value, f"weights[{index}]")
            for index, value in enumerate(self.weights)
        )
        if len(weights) != len(self.feature_names):
            raise StageEditContextualSourcePairV2Error("contextual scorer weight dimension mismatch")
        if math.fsum(abs(value) for value in weights) > (
            STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SIGNED_L1_RADIUS
        ):
            raise StageEditContextualSourcePairV2Error("contextual scorer exceeds signed L1 radius")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "contract": self.contract,
            "token_schema_fingerprint": self.token_schema_fingerprint,
            "feature_contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_CONTRACT,
            "feature_schema_sha256": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_SCHEMA_SHA256,
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
            "signed_l1_radius": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SIGNED_L1_RADIUS,
            "handcrafted_baseline_fraction": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_HANDCRAFTED_BASELINE_FRACTION,
            "uniform_baseline_fraction": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_UNIFORM_BASELINE_FRACTION,
            "bounded_score_contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BOUNDED_SCORE_CONTRACT,
        }

    def raw_logit_from_materialized(
        self,
        features: StageEditContextualPairFeatureVectorV2,
        *,
        uniform_baseline: float,
        handcrafted_baseline: float,
    ) -> tuple[float, float, float]:
        if not isinstance(features, StageEditContextualPairFeatureVectorV2):
            raise StageEditContextualSourcePairV2Error("scoring needs typed contextual pair features")
        uniform = _finite(uniform_baseline, "uniform_baseline")
        handcrafted = _finite(handcrafted_baseline, "handcrafted_baseline")
        if not -1.0 <= uniform <= 0.0 or not -1.0 <= handcrafted <= 0.0:
            raise StageEditContextualSourcePairV2Error("baseline score is outside [-1, 0]")
        blended = 0.30 * handcrafted + 0.70 * uniform
        # Use the exact optimize=False reduction selected by the frozen A/id3
        # exploration.  A one-row listwise matrix and this vector reduction are
        # bit-identical, which makes the runtime-equivalence digest meaningful.
        residual = float(
            np.einsum(
                "f,f->",
                np.asarray(features.values, dtype=np.float64),
                np.asarray(self.weights, dtype=np.float64),
                optimize=False,
            )
        )
        raw = blended + residual
        stage_edit_contextual_raw_to_decoder_score_v2(raw)
        return raw, blended, residual

    def score_boundary(
        self,
        frame: StageEditContextualRuntimeFrameV2,
        outgoing_source_id: str,
        incoming_source_id: str,
    ) -> StageEditContextualBoundaryScoreV2:
        if not isinstance(frame, StageEditContextualRuntimeFrameV2):
            raise StageEditContextualSourcePairV2Error("scorer needs typed runtime frame")
        if frame.token_schema_fingerprint != self.token_schema_fingerprint:
            raise StageEditContextualSourcePairV2Error("scorer/runtime token schema mismatch")
        outgoing_id = _name(outgoing_source_id, "outgoing_source_id")
        incoming_id = _name(incoming_source_id, "incoming_source_id")
        sources = frame.source_by_id
        if outgoing_id not in sources or incoming_id not in sources:
            raise StageEditContextualSourcePairV2Error("score source is absent from runtime frame")
        neutral_reason: str | None = None
        if outgoing_id == incoming_id:
            neutral_reason = "same_exact_source"
        elif sources[outgoing_id].derivative_family_id == sources[incoming_id].derivative_family_id:
            neutral_reason = "same_derivative_family"
        if neutral_reason is not None:
            return StageEditContextualBoundaryScoreV2(
                outgoing_source_id=outgoing_id,
                incoming_source_id=incoming_id,
                applied=False,
                neutral_reason=neutral_reason,
                uniform_baseline=None,
                handcrafted_baseline=None,
                blended_baseline=None,
                signed_residual=None,
                raw_logit=0.0,
                decoder_score=0.0,
                runtime_evidence_sha256=frame.runtime_evidence_sha256,
                pair_feature_sha256=None,
                scorer_fingerprint=self.fingerprint,
            )
        features = materialize_stage_edit_contextual_pair_features_v2(
            frame, outgoing_id, incoming_id
        )
        uniform, handcrafted, _expected_blended = stage_edit_contextual_pair_baselines_v2(
            frame, outgoing_id, incoming_id
        )
        raw, blended, residual = self.raw_logit_from_materialized(
            features,
            uniform_baseline=uniform,
            handcrafted_baseline=handcrafted,
        )
        return StageEditContextualBoundaryScoreV2(
            outgoing_source_id=outgoing_id,
            incoming_source_id=incoming_id,
            applied=True,
            neutral_reason=None,
            uniform_baseline=uniform,
            handcrafted_baseline=handcrafted,
            blended_baseline=blended,
            signed_residual=residual,
            raw_logit=raw,
            decoder_score=stage_edit_contextual_raw_to_decoder_score_v2(raw),
            runtime_evidence_sha256=frame.runtime_evidence_sha256,
            pair_feature_sha256=features.content_sha256,
            scorer_fingerprint=self.fingerprint,
        )


@dataclass(frozen=True, slots=True)
class StageEditContextualTrainingConfigV2:
    architecture: str = "A"
    feature_design: str = "id3"
    steps: int = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_TRAINING_STEPS
    learning_rate: float = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_LEARNING_RATE
    backtrack_attempts: int = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_BACKTRACK_ATTEMPTS
    temperature: float = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_TEMPERATURE
    margin: float = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_MARGIN
    handcrafted_baseline_fraction: float = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_HANDCRAFTED_BASELINE_FRACTION
    signed_l1_radius: float = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SIGNED_L1_RADIUS
    l2_penalty: float = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_L2_PENALTY
    tolerance: float = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_TOLERANCE
    zero_initialization: bool = True
    einsum_optimize: bool = False

    def __post_init__(self) -> None:
        expected = {
            "architecture": "A",
            "feature_design": "id3",
            "steps": 160,
            "learning_rate": 0.5,
            "backtrack_attempts": 24,
            "temperature": 1.0,
            "margin": 0.0,
            "handcrafted_baseline_fraction": 0.30,
            "signed_l1_radius": 1.5,
            "l2_penalty": 0.05,
            "tolerance": 1.0e-12,
            "zero_initialization": True,
            "einsum_optimize": False,
        }
        observed = {field: getattr(self, field) for field in expected}
        if observed != expected:
            raise StageEditContextualSourcePairV2Error(
                "v2 training config differs from the exact frozen A/id3 exploration"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "feature_design": self.feature_design,
            "objective_contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_OBJECTIVE_CONTRACT,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "backtrack_attempts": self.backtrack_attempts,
            "accepted_learning_rate_policy": "min_0p5_trial_times_1p02",
            "rejected_learning_rate_policy": "trial_times_0p5",
            "temperature": self.temperature,
            "margin": self.margin,
            "handcrafted_baseline_fraction": self.handcrafted_baseline_fraction,
            "uniform_baseline_fraction": 0.70,
            "signed_l1_radius": self.signed_l1_radius,
            "l2_penalty": self.l2_penalty,
            "l2_loss_form": "0p5_times_l2_times_sum_squared_weights",
            "tolerance": self.tolerance,
            "zero_initialization": self.zero_initialization,
            "einsum_optimize": self.einsum_optimize,
            "balance_order": ["song", "reference", "group"],
        }


@dataclass(frozen=True, slots=True)
class StageEditContextualTrainingCandidateV2:
    source_id: str
    derivative_family_id: int
    features: StageEditContextualPairFeatureVectorV2
    uniform_baseline: float
    handcrafted_baseline: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _name(self.source_id, "source_id"))
        _integer(self.derivative_family_id, "derivative_family_id")
        if not isinstance(self.features, StageEditContextualPairFeatureVectorV2):
            raise StageEditContextualSourcePairV2Error("training candidate features are invalid")
        if self.features.incoming_source_id != self.source_id:
            raise StageEditContextualSourcePairV2Error("training candidate/source identity mismatch")
        for field in ("uniform_baseline", "handcrafted_baseline"):
            value = _finite(getattr(self, field), field)
            if not -1.0 <= value <= 0.0:
                raise StageEditContextualSourcePairV2Error(f"{field} is outside [-1, 0]")
            object.__setattr__(self, field, value)

    @property
    def blended_baseline(self) -> float:
        return 0.30 * self.handcrafted_baseline + 0.70 * self.uniform_baseline

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "derivative_family_id": self.derivative_family_id,
            "features": self.features.to_dict(),
            "uniform_baseline": self.uniform_baseline,
            "handcrafted_baseline": self.handcrafted_baseline,
            "blended_baseline": self.blended_baseline,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "StageEditContextualTrainingCandidateV2":
        expected = {
            "source_id",
            "derivative_family_id",
            "features",
            "uniform_baseline",
            "handcrafted_baseline",
            "blended_baseline",
        }
        if not isinstance(payload, Mapping):
            raise StageEditContextualSourcePairV2Error(
                "training candidate must be an object"
            )
        _strict_keys(payload, expected, "training candidate")
        if not isinstance(payload.get("features"), Mapping):
            raise StageEditContextualSourcePairV2Error(
                "training candidate features must be an object"
            )
        result = cls(
            source_id=payload["source_id"],
            derivative_family_id=_integer(
                payload["derivative_family_id"], "derivative_family_id"
            ),
            features=StageEditContextualPairFeatureVectorV2.from_dict(
                payload["features"]
            ),
            uniform_baseline=payload["uniform_baseline"],
            handcrafted_baseline=payload["handcrafted_baseline"],
        )
        if payload["blended_baseline"] != result.blended_baseline:
            raise StageEditContextualSourcePairV2Error(
                "training candidate blended baseline does not round-trip"
            )
        return result


@dataclass(frozen=True, slots=True)
class StageEditContextualTrainingGroupV2:
    group_id: str
    song_id: str
    reference_id: str
    token_index: int
    outgoing_source_id: str
    outgoing_derivative_family_id: int
    chosen_incoming_source_id: str
    candidates: tuple[StageEditContextualTrainingCandidateV2, ...]
    runtime_evidence_sha256: str
    feature_snapshot_sha256: str
    clipped_observed_value_count: int
    runtime_frame: StageEditContextualRuntimeFrameV2

    def __post_init__(self) -> None:
        for field in ("group_id", "song_id", "reference_id", "outgoing_source_id", "chosen_incoming_source_id"):
            object.__setattr__(self, field, _name(getattr(self, field), field))
        _integer(self.token_index, "token_index")
        _integer(self.outgoing_derivative_family_id, "outgoing_derivative_family_id")
        _sha256(self.runtime_evidence_sha256, "runtime_evidence_sha256")
        _sha256(self.feature_snapshot_sha256, "feature_snapshot_sha256")
        _integer(
            self.clipped_observed_value_count,
            "clipped_observed_value_count",
        )
        if not isinstance(self.runtime_frame, StageEditContextualRuntimeFrameV2):
            raise StageEditContextualSourcePairV2Error(
                "listwise group runtime frame is invalid"
            )
        if self.runtime_frame.runtime_evidence_sha256 != self.runtime_evidence_sha256:
            raise StageEditContextualSourcePairV2Error(
                "listwise group runtime frame digest mismatch"
            )
        if (
            self.runtime_frame.clipped_observed_value_count
            != self.clipped_observed_value_count
        ):
            raise StageEditContextualSourcePairV2Error(
                "listwise group clipping audit differs from runtime frame"
            )
        candidates = tuple(self.candidates)
        if len(candidates) < 2 or any(
            not isinstance(value, StageEditContextualTrainingCandidateV2)
            for value in candidates
        ):
            raise StageEditContextualSourcePairV2Error("listwise group needs at least two typed candidates")
        source_ids = tuple(value.source_id for value in candidates)
        if source_ids != tuple(sorted(set(source_ids))):
            raise StageEditContextualSourcePairV2Error("listwise candidate ids must be unique and sorted")
        if self.chosen_incoming_source_id not in source_ids:
            raise StageEditContextualSourcePairV2Error("chosen incoming source is absent from listwise group")
        if self.outgoing_source_id in source_ids:
            raise StageEditContextualSourcePairV2Error("outgoing exact source cannot be a switch candidate")
        if any(value.derivative_family_id == self.outgoing_derivative_family_id for value in candidates):
            raise StageEditContextualSourcePairV2Error(
                "same derivative-family candidates must be removed before v2 training"
            )
        sources = self.runtime_frame.source_by_id
        if self.outgoing_source_id not in sources or (
            sources[self.outgoing_source_id].derivative_family_id
            != self.outgoing_derivative_family_id
        ):
            raise StageEditContextualSourcePairV2Error(
                "listwise outgoing derivative family differs from runtime frame"
            )
        if any(
            candidate.source_id not in sources
            or sources[candidate.source_id].derivative_family_id
            != candidate.derivative_family_id
            for candidate in candidates
        ):
            raise StageEditContextualSourcePairV2Error(
                "listwise candidate derivative family differs from runtime frame"
            )
        if any(value.features.runtime_evidence_sha256 != self.runtime_evidence_sha256 for value in candidates):
            raise StageEditContextualSourcePairV2Error("listwise candidates do not share exact runtime evidence")
        for candidate in candidates:
            rematerialized = materialize_stage_edit_contextual_pair_features_v2(
                self.runtime_frame,
                self.outgoing_source_id,
                candidate.source_id,
            )
            if rematerialized != candidate.features:
                raise StageEditContextualSourcePairV2Error(
                    "listwise candidate differs from live runtime rematerialization"
                )
            uniform, handcrafted, _blended = stage_edit_contextual_pair_baselines_v2(
                self.runtime_frame,
                self.outgoing_source_id,
                candidate.source_id,
            )
            if (
                uniform != candidate.uniform_baseline
                or handcrafted != candidate.handcrafted_baseline
            ):
                raise StageEditContextualSourcePairV2Error(
                    "listwise baselines differ from live runtime rematerialization"
                )
        object.__setattr__(self, "candidates", candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "song_id": self.song_id,
            "reference_id": self.reference_id,
            "token_index": self.token_index,
            "outgoing_source_id": self.outgoing_source_id,
            "outgoing_derivative_family_id": self.outgoing_derivative_family_id,
            "chosen_incoming_source_id": self.chosen_incoming_source_id,
            "candidates": [value.to_dict() for value in self.candidates],
            "runtime_evidence_sha256": self.runtime_evidence_sha256,
            "feature_snapshot_sha256": self.feature_snapshot_sha256,
            "clipped_observed_value_count": self.clipped_observed_value_count,
            "runtime_frame": self.runtime_frame.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "StageEditContextualTrainingGroupV2":
        expected = {
            "group_id",
            "song_id",
            "reference_id",
            "token_index",
            "outgoing_source_id",
            "outgoing_derivative_family_id",
            "chosen_incoming_source_id",
            "candidates",
            "runtime_evidence_sha256",
            "feature_snapshot_sha256",
            "clipped_observed_value_count",
            "runtime_frame",
        }
        if not isinstance(payload, Mapping):
            raise StageEditContextualSourcePairV2Error(
                "training group must be an object"
            )
        _strict_keys(payload, expected, "training group")
        if not isinstance(payload.get("candidates"), list) or any(
            not isinstance(value, Mapping) for value in payload.get("candidates", ())
        ):
            raise StageEditContextualSourcePairV2Error(
                "training group candidates must be objects"
            )
        if not isinstance(payload.get("runtime_frame"), Mapping):
            raise StageEditContextualSourcePairV2Error(
                "training group runtime frame must be an object"
            )
        return cls(
            group_id=payload["group_id"],
            song_id=payload["song_id"],
            reference_id=payload["reference_id"],
            token_index=_integer(payload["token_index"], "token_index"),
            outgoing_source_id=payload["outgoing_source_id"],
            outgoing_derivative_family_id=_integer(
                payload["outgoing_derivative_family_id"],
                "outgoing_derivative_family_id",
            ),
            chosen_incoming_source_id=payload["chosen_incoming_source_id"],
            candidates=tuple(
                StageEditContextualTrainingCandidateV2.from_dict(value)
                for value in payload["candidates"]
            ),
            runtime_evidence_sha256=payload["runtime_evidence_sha256"],
            feature_snapshot_sha256=payload["feature_snapshot_sha256"],
            clipped_observed_value_count=_integer(
                payload["clipped_observed_value_count"],
                "clipped_observed_value_count",
            ),
            runtime_frame=StageEditContextualRuntimeFrameV2.from_dict(
                payload["runtime_frame"]
            ),
        )


def materialize_stage_edit_contextual_training_group_v2(
    *,
    group_id: str,
    song_id: str,
    reference_id: str,
    token_index: int,
    frame: StageEditContextualRuntimeFrameV2,
    outgoing_source_id: str,
    incoming_source_ids: Sequence[str],
    chosen_incoming_source_id: str,
    feature_snapshot_sha256: str,
) -> StageEditContextualTrainingGroupV2:
    incoming_ids = _exact_sorted_names(incoming_source_ids, "incoming_source_ids")
    outgoing_id = _name(outgoing_source_id, "outgoing_source_id")
    sources = frame.source_by_id
    if outgoing_id not in sources or any(value not in sources for value in incoming_ids):
        raise StageEditContextualSourcePairV2Error("training group source is absent from runtime frame")
    candidates: list[StageEditContextualTrainingCandidateV2] = []
    for source_id in incoming_ids:
        features = materialize_stage_edit_contextual_pair_features_v2(
            frame, outgoing_id, source_id
        )
        uniform, handcrafted, _blended = stage_edit_contextual_pair_baselines_v2(
            frame, outgoing_id, source_id
        )
        candidates.append(
            StageEditContextualTrainingCandidateV2(
                source_id=source_id,
                derivative_family_id=sources[source_id].derivative_family_id,
                features=features,
                uniform_baseline=uniform,
                handcrafted_baseline=handcrafted,
            )
        )
    return StageEditContextualTrainingGroupV2(
        group_id=group_id,
        song_id=song_id,
        reference_id=reference_id,
        token_index=token_index,
        outgoing_source_id=outgoing_id,
        outgoing_derivative_family_id=sources[outgoing_id].derivative_family_id,
        chosen_incoming_source_id=chosen_incoming_source_id,
        candidates=tuple(candidates),
        runtime_evidence_sha256=frame.runtime_evidence_sha256,
        feature_snapshot_sha256=feature_snapshot_sha256,
        clipped_observed_value_count=frame.clipped_observed_value_count,
        runtime_frame=frame,
    )


@dataclass(frozen=True, slots=True)
class StageEditContextualTrainingCorpusV2:
    groups: tuple[StageEditContextualTrainingGroupV2, ...]
    token_schema_fingerprint: str
    source_data_id: str
    content_sha256: str
    feature_snapshot_bindings_sha256: str

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        if not groups or any(not isinstance(value, StageEditContextualTrainingGroupV2) for value in groups):
            raise StageEditContextualSourcePairV2Error("contextual corpus needs typed groups")
        ordered = tuple(sorted(groups, key=lambda value: (value.song_id, value.reference_id, value.group_id)))
        if groups != ordered or len({value.group_id for value in groups}) != len(groups):
            raise StageEditContextualSourcePairV2Error("contextual corpus groups must be unique and canonical")
        reference_song: dict[str, str] = {}
        snapshot_song: dict[str, str] = {}
        for group in groups:
            previous = reference_song.setdefault(group.reference_id, group.song_id)
            if previous != group.song_id:
                raise StageEditContextualSourcePairV2Error("one reference cannot span multiple songs")
            snapshot_song_id = snapshot_song.setdefault(
                group.feature_snapshot_sha256, group.song_id
            )
            if snapshot_song_id != group.song_id:
                raise StageEditContextualSourcePairV2Error(
                    "one feature snapshot digest cannot be reused across songs"
                )
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "token_schema_fingerprint", _sha256(self.token_schema_fingerprint, "token_schema_fingerprint"))
        if any(
            group.runtime_frame.token_schema_fingerprint
            != self.token_schema_fingerprint
            for group in groups
        ):
            raise StageEditContextualSourcePairV2Error(
                "contextual corpus token schema differs from a group runtime frame"
            )
        object.__setattr__(self, "source_data_id", _name(self.source_data_id, "source_data_id"))
        bindings_digest = _sha256(
            self.feature_snapshot_bindings_sha256,
            "feature_snapshot_bindings_sha256",
        )
        expected_bindings_digest = _canonical_sha256(
            [
                {
                    "reference_id": group.reference_id,
                    "feature_snapshot_sha256": group.feature_snapshot_sha256,
                }
                for group in groups
            ]
        )
        if bindings_digest != expected_bindings_digest:
            raise StageEditContextualSourcePairV2Error(
                "contextual corpus snapshot bindings digest does not match groups"
            )
        object.__setattr__(
            self, "feature_snapshot_bindings_sha256", bindings_digest
        )
        digest = _sha256(self.content_sha256, "content_sha256")
        object.__setattr__(self, "content_sha256", digest)
        if digest != _canonical_sha256(self.unsigned_dict()):
            raise StageEditContextualSourcePairV2Error("contextual corpus digest does not match contents")

    @property
    def song_ids(self) -> tuple[str, ...]:
        return tuple(sorted({value.song_id for value in self.groups}))

    @property
    def reference_ids(self) -> tuple[str, ...]:
        return tuple(sorted({value.reference_id for value in self.groups}))

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(value.group_id for value in self.groups)

    @property
    def clipped_observed_value_count(self) -> int:
        return sum(value.clipped_observed_value_count for value in self.groups)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCHEMA_VERSION,
            "feature_contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_CONTRACT,
            "feature_schema_sha256": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_SCHEMA_SHA256,
            "token_schema_fingerprint": self.token_schema_fingerprint,
            "source_data_id": self.source_data_id,
            "feature_snapshot_bindings_sha256": self.feature_snapshot_bindings_sha256,
            "clipped_observed_value_count": self.clipped_observed_value_count,
            "groups": [value.to_dict() for value in self.groups],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "StageEditContextualTrainingCorpusV2":
        expected = {
            "schema_version",
            "feature_contract",
            "feature_schema_sha256",
            "token_schema_fingerprint",
            "source_data_id",
            "feature_snapshot_bindings_sha256",
            "clipped_observed_value_count",
            "groups",
            "content_sha256",
        }
        if not isinstance(payload, Mapping):
            raise StageEditContextualSourcePairV2Error(
                "contextual training corpus must be an object"
            )
        _strict_keys(payload, expected, "contextual training corpus")
        if (
            payload.get("schema_version")
            != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCHEMA_VERSION
            or payload.get("feature_contract")
            != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_CONTRACT
            or payload.get("feature_schema_sha256")
            != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_SCHEMA_SHA256
        ):
            raise StageEditContextualSourcePairV2Error(
                "contextual training corpus schema contract mismatch"
            )
        if not isinstance(payload.get("groups"), list) or any(
            not isinstance(value, Mapping) for value in payload.get("groups", ())
        ):
            raise StageEditContextualSourcePairV2Error(
                "contextual training corpus groups must be objects"
            )
        result = cls(
            groups=tuple(
                StageEditContextualTrainingGroupV2.from_dict(value)
                for value in payload["groups"]
            ),
            token_schema_fingerprint=payload["token_schema_fingerprint"],
            source_data_id=payload["source_data_id"],
            feature_snapshot_bindings_sha256=payload[
                "feature_snapshot_bindings_sha256"
            ],
            content_sha256=payload["content_sha256"],
        )
        if payload["clipped_observed_value_count"] != result.clipped_observed_value_count:
            raise StageEditContextualSourcePairV2Error(
                "contextual corpus clipping audit does not round-trip"
            )
        return result


def new_stage_edit_contextual_training_corpus_v2(
    groups: Sequence[StageEditContextualTrainingGroupV2],
    *,
    token_schema_fingerprint: str,
    source_data_id: str,
) -> StageEditContextualTrainingCorpusV2:
    ordered = tuple(sorted(groups, key=lambda value: (value.song_id, value.reference_id, value.group_id)))
    bindings = [
        {
            "reference_id": group.reference_id,
            "feature_snapshot_sha256": group.feature_snapshot_sha256,
        }
        for group in ordered
    ]
    bindings_sha256 = _canonical_sha256(bindings)
    unsigned = {
        "schema_version": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCHEMA_VERSION,
        "feature_contract": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_CONTRACT,
        "feature_schema_sha256": STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_SCHEMA_SHA256,
        "token_schema_fingerprint": token_schema_fingerprint,
        "source_data_id": source_data_id,
        "feature_snapshot_bindings_sha256": bindings_sha256,
        "clipped_observed_value_count": sum(
            value.clipped_observed_value_count for value in ordered
        ),
        "groups": [value.to_dict() for value in ordered],
    }
    return StageEditContextualTrainingCorpusV2(
        groups=ordered,
        token_schema_fingerprint=token_schema_fingerprint,
        source_data_id=source_data_id,
        content_sha256=_canonical_sha256(unsigned),
        feature_snapshot_bindings_sha256=bindings_sha256,
    )


def materialize_stage_edit_contextual_training_corpus_v2(
    corpus: ReferenceTrainingCorpus,
    *,
    derivative_family_id_by_source: Mapping[str, int],
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
) -> StageEditContextualTrainingCorpusV2:
    """Materialize v1 switch labels through the exact shared v2 runtime path."""

    if not isinstance(corpus, ReferenceTrainingCorpus):
        raise StageEditContextualSourcePairV2Error("reference corpus is invalid")
    pair_corpus = materialize_stage_edit_source_pair_training_corpus(corpus, schema=schema)
    examples = {value.reference_id: value for value in corpus.examples}
    if len(examples) != len(corpus.examples):
        raise StageEditContextualSourcePairV2Error("reference ids must be unique")
    masked_sequences: dict[str, StageEditSequence] = {}
    for reference_id, example in examples.items():
        if example.feature_snapshot_sha256 is None:
            raise StageEditContextualSourcePairV2Error(
                f"reference {reference_id} lacks a content-hashed feature snapshot"
            )
        _sha256(example.feature_snapshot_sha256, f"reference {reference_id} feature snapshot")
        masked_sequences[reference_id] = mask_stage_edit_recent_history_for_semimarkov(
            example.sequence, schema
        )
    song_by_group = dict(pair_corpus.group_song_ids)
    groups: list[StageEditContextualTrainingGroupV2] = []
    for pair_group in pair_corpus.groups:
        prefix, separator, suffix = pair_group.group_id.rpartition(":source-pair:")
        if not separator or prefix != pair_group.reference_id or not suffix.isdigit():
            raise StageEditContextualSourcePairV2Error("pair group token identity is not canonical")
        token_index = int(suffix)
        token = masked_sequences[pair_group.reference_id].tokens[token_index]
        token_source_ids = tuple(sorted(candidate.source_id for candidate in token.source_candidates))
        frame = materialize_stage_edit_contextual_runtime_frame_v2(
            token,
            canonical_source_ids=token_source_ids,
            derivative_family_id_by_source={
                source_id: derivative_family_id_by_source[source_id]
                for source_id in token_source_ids
                if source_id in derivative_family_id_by_source
            },
            schema=schema,
        )
        groups.append(
            materialize_stage_edit_contextual_training_group_v2(
                group_id=pair_group.group_id,
                song_id=song_by_group[pair_group.group_id],
                reference_id=pair_group.reference_id,
                token_index=token_index,
                frame=frame,
                outgoing_source_id=pair_group.outgoing_source_id,
                incoming_source_ids=tuple(candidate.source_id for candidate in pair_group.candidates),
                chosen_incoming_source_id=pair_group.chosen_incoming_source_id,
                feature_snapshot_sha256=examples[pair_group.reference_id].feature_snapshot_sha256,
            )
        )
    return new_stage_edit_contextual_training_corpus_v2(
        groups,
        token_schema_fingerprint=schema.fingerprint,
        source_data_id=pair_corpus.source_data_id,
    )
