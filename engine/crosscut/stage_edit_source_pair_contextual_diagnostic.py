"""Song-disjoint diagnostic for contextual source-pair residual scoring.

This module is intentionally isolated from the production scorer and decoder.
It tests whether causal context already present in ``StageEditToken`` can make
source-pair ranking generalize across songs without weakening the existing
uniform/handcrafted activation gate.

No model output or reference-match evidence is consumed.  The diagnostic uses
only same-token source inputs, music/section/style inputs, compact summaries of
the existing pose/formation embeddings, and the v1 pair deltas.  Recent-edit
features are excluded because the global Transformer contract masks them and a
reference timeline could otherwise leak human editorial history.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .stage_edit_reference_adapter import ReferenceTrainingCorpus
from .stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES,
    StageEditSourcePairCandidate,
    StageEditSourcePairTrainingGroup,
    materialize_stage_edit_source_pair_features,
)
from .stage_edit_source_pair_training import (
    STAGE_EDIT_SOURCE_PAIR_HANDCRAFTED_BASELINE_CONTRACT,
    StageEditSourcePairActivationGate,
    StageEditSourcePairRankingMetrics,
    StageEditSourcePairTrainingError,
    _handcrafted_pair_score,
    materialize_stage_edit_source_pair_training_corpus,
    stage_edit_source_pair_no_worse_activation_gate,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditToken,
    StageEditTokenSchema,
)


STAGE_EDIT_SOURCE_PAIR_CONTEXTUAL_DIAGNOSTIC_SCHEMA_VERSION = 1
STAGE_EDIT_SOURCE_PAIR_CONTEXTUAL_DIAGNOSTIC_CONTRACT = (
    "diagnostic_only_causal_contextual_signed_bounded_source_pair_residual_"
    "song_balanced_loso_v1"
)

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


class StageEditSourcePairContextualDiagnosticError(StageEditSourcePairTrainingError):
    """The diagnostic corpus, fit, or song-disjoint evaluation is invalid."""


def _name(value: object, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise StageEditSourcePairContextualDiagnosticError(
            f"{field} cannot be blank"
        )
    return result


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise StageEditSourcePairContextualDiagnosticError(
            f"{field} must be numeric"
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StageEditSourcePairContextualDiagnosticError(
            f"{field} must be numeric"
        ) from exc
    if not math.isfinite(result):
        raise StageEditSourcePairContextualDiagnosticError(
            f"{field} must be finite"
        )
    return result


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bounded(value: object, field: str) -> float:
    return min(1.0, max(-1.0, _finite(value, field)))


def _source_values(
    candidate: SourceCandidateToken,
    *,
    schema: StageEditTokenSchema,
    field: str,
) -> tuple[dict[str, float], dict[str, bool]]:
    if not isinstance(candidate, SourceCandidateToken):
        raise StageEditSourcePairContextualDiagnosticError(
            f"{field} must be a SourceCandidateToken"
        )
    if len(candidate.features) != len(schema.source_feature_names):
        raise StageEditSourcePairContextualDiagnosticError(
            f"{field} feature dimension differs from schema"
        )
    if candidate.missing_mask and len(candidate.missing_mask) != len(
        schema.source_feature_names
    ):
        raise StageEditSourcePairContextualDiagnosticError(
            f"{field} missing-mask dimension differs from schema"
        )
    missing = (
        tuple(candidate.missing_mask)
        if candidate.missing_mask
        else (False,) * len(schema.source_feature_names)
    )
    if any(not isinstance(value, (bool, np.bool_)) for value in missing):
        raise StageEditSourcePairContextualDiagnosticError(
            f"{field} missing mask must be boolean"
        )
    values = {
        name: (
            0.0
            if bool(missing[index])
            else _bounded(candidate.features[index], f"{field}.{name}")
        )
        for index, name in enumerate(schema.source_feature_names)
    }
    return values, {
        name: bool(missing[index])
        for index, name in enumerate(schema.source_feature_names)
    }


def _context_values(
    token: StageEditToken,
    *,
    schema: StageEditTokenSchema,
) -> dict[str, float]:
    if not isinstance(token, StageEditToken):
        raise StageEditSourcePairContextualDiagnosticError(
            "contextual pair token is invalid"
        )
    dimensions = (
        len(schema.music_feature_names),
        len(schema.section_feature_names),
        len(schema.recent_edit_feature_names),
        len(schema.style_feature_names),
    )
    observed_dimensions = (
        len(token.music_features),
        len(token.section_features),
        len(token.recent_edit_features),
        len(token.style_features),
    )
    if observed_dimensions != dimensions:
        raise StageEditSourcePairContextualDiagnosticError(
            "contextual pair token dimensions differ from schema"
        )
    if token.missing_mask and len(token.missing_mask) != schema.raw_context_dimension:
        raise StageEditSourcePairContextualDiagnosticError(
            "contextual pair token missing-mask dimension differs from schema"
        )
    missing = (
        tuple(token.missing_mask)
        if token.missing_mask
        else (False,) * schema.raw_context_dimension
    )
    music_offset = 0
    section_offset = dimensions[0]
    recent_offset = section_offset + dimensions[1]
    style_offset = recent_offset + dimensions[2]
    by_name: dict[str, float] = {}

    def add(
        namespace: str,
        names: Sequence[str],
        values: Sequence[float],
        offset: int,
    ) -> None:
        for index, (name, value) in enumerate(zip(names, values, strict=True)):
            key = f"{namespace}.{name}"
            by_name[key] = (
                0.0
                if bool(missing[offset + index])
                else _bounded(value, key)
            )

    add("music", schema.music_feature_names, token.music_features, music_offset)
    add(
        "section",
        schema.section_feature_names,
        token.section_features,
        section_offset,
    )
    # The recent-edit slice is deliberately not read into the result.
    add("style", schema.style_feature_names, token.style_features, style_offset)
    missing_names = sorted(set(_CONTEXT_FEATURE_NAMES) - set(by_name))
    if missing_names:
        raise StageEditSourcePairContextualDiagnosticError(
            "context schema lacks diagnostic features: " + ", ".join(missing_names)
        )
    return {name: by_name[name] for name in _CONTEXT_FEATURE_NAMES}


def _embedding_summary(
    prefix: str,
    outgoing_values: Mapping[str, float],
    outgoing_missing: Mapping[str, bool],
    incoming_values: Mapping[str, float],
    incoming_missing: Mapping[str, bool],
) -> tuple[tuple[str, float], ...]:
    names = tuple(sorted(name for name in outgoing_values if name.startswith(prefix)))
    if not names:
        raise StageEditSourcePairContextualDiagnosticError(
            f"source schema lacks embedding block {prefix}"
        )
    observed = tuple(
        name
        for name in names
        if not outgoing_missing[name] and not incoming_missing[name]
    )
    if not observed:
        return (
            (f"embedding.{prefix}incoming_norm", 0.0),
            (f"embedding.{prefix}outgoing_norm", 0.0),
            (f"embedding.{prefix}cosine", 0.0),
            (f"embedding.{prefix}distance", 0.0),
            (f"embedding.{prefix}missing_fraction", 1.0),
        )
    left = np.asarray([outgoing_values[name] for name in observed], dtype=np.float64)
    right = np.asarray([incoming_values[name] for name in observed], dtype=np.float64)
    left_norm = float(np.linalg.norm(left) / math.sqrt(len(observed)))
    right_norm = float(np.linalg.norm(right) / math.sqrt(len(observed)))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    cosine = float(np.dot(left, right) / denominator) if denominator > 1.0e-12 else 0.0
    distance = float(np.linalg.norm(right - left) / (2.0 * math.sqrt(len(observed))))
    return (
        (f"embedding.{prefix}incoming_norm", min(1.0, right_norm)),
        (f"embedding.{prefix}outgoing_norm", min(1.0, left_norm)),
        (f"embedding.{prefix}cosine", min(1.0, max(-1.0, cosine))),
        (f"embedding.{prefix}distance", min(1.0, distance)),
        (
            f"embedding.{prefix}missing_fraction",
            1.0 - len(observed) / len(names),
        ),
    )


def _shot_size_basis(value: float) -> tuple[float, float, float]:
    normalized = min(1.0, max(0.0, value))
    return (
        max(0.0, 1.0 - 2.0 * normalized),
        max(0.0, 1.0 - abs(2.0 * normalized - 1.0)),
        max(0.0, 2.0 * normalized - 1.0),
    )


def _contextual_feature_items(
    pair_group: StageEditSourcePairTrainingGroup,
    pair_candidate: StageEditSourcePairCandidate,
    token: StageEditToken,
    outgoing_candidate: SourceCandidateToken,
    incoming_candidate: SourceCandidateToken,
    *,
    schema: StageEditTokenSchema,
) -> tuple[tuple[str, float], ...]:
    pair = materialize_stage_edit_source_pair_features(
        pair_group.outgoing, pair_candidate.incoming
    ).by_name
    outgoing, outgoing_missing = _source_values(
        outgoing_candidate, schema=schema, field="outgoing"
    )
    incoming, incoming_missing = _source_values(
        incoming_candidate, schema=schema, field="incoming"
    )
    context = _context_values(token, schema=schema)
    items: list[tuple[str, float]] = [
        (f"pair.{name}", pair[name]) for name in STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES
    ]
    for source_name in _SCALAR_SOURCE_FEATURE_NAMES:
        if source_name not in outgoing or source_name not in incoming:
            raise StageEditSourcePairContextualDiagnosticError(
                f"source schema lacks scalar evidence {source_name}"
            )
        left = outgoing[source_name]
        right = incoming[source_name]
        jointly_missing = outgoing_missing[source_name] or incoming_missing[source_name]
        items.extend(
            (
                (f"incoming.{source_name}", right),
                (f"incoming_missing.{source_name}", float(incoming_missing[source_name])),
                (f"outgoing.{source_name}", left),
                (f"outgoing_missing.{source_name}", float(outgoing_missing[source_name])),
                (
                    f"directed.{source_name}",
                    0.0 if jointly_missing else min(1.0, max(-1.0, right - left)),
                ),
                (f"directed_missing.{source_name}", float(jointly_missing)),
                (
                    f"product.{source_name}",
                    0.0 if jointly_missing else min(1.0, max(-1.0, right * left)),
                ),
            )
        )
    for prefix in _EMBEDDING_PREFIXES:
        items.extend(
            _embedding_summary(
                prefix,
                outgoing,
                outgoing_missing,
                incoming,
                incoming_missing,
            )
        )
    for side, value in (
        ("incoming", incoming["shot_scale"]),
        ("outgoing", outgoing["shot_scale"]),
    ):
        wide, medium, close = _shot_size_basis(value)
        items.extend(
            (
                (f"shot_size.{side}.wide", wide),
                (f"shot_size.{side}.medium", medium),
                (f"shot_size.{side}.close", close),
            )
        )
    interaction_core: dict[str, float] = {
        f"pair.{name}": pair[name] for name in _INTERACTION_PAIR_FEATURE_NAMES
    }
    for source_name in _INTERACTION_SOURCE_FEATURE_NAMES:
        interaction_core[f"incoming.{source_name}"] = incoming[source_name]
        interaction_core[f"directed.{source_name}"] = (
            0.0
            if outgoing_missing[source_name] or incoming_missing[source_name]
            else min(
                1.0,
                max(-1.0, incoming[source_name] - outgoing[source_name]),
            )
        )
    for context_name in _CONTEXT_FEATURE_NAMES:
        for core_name in sorted(interaction_core):
            items.append(
                (
                    f"interaction.{context_name}*{core_name}",
                    context[context_name] * interaction_core[core_name],
                )
            )
    if len({name for name, _value in items}) != len(items):
        raise StageEditSourcePairContextualDiagnosticError(
            "contextual feature names are not unique"
        )
    return tuple(items)


@dataclass(frozen=True, slots=True)
class StageEditSourcePairContextualCandidate:
    source_id: str
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _name(self.source_id, "source_id"))
        values = tuple(
            _finite(value, f"contextual features[{index}]")
            for index, value in enumerate(self.features)
        )
        if not values or any(not -1.0 <= value <= 1.0 for value in values):
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual features must be non-empty and bounded in [-1, 1]"
            )
        object.__setattr__(self, "features", values)


@dataclass(frozen=True, slots=True)
class StageEditSourcePairContextualGroup:
    pair_group: StageEditSourcePairTrainingGroup
    song_id: str
    token_index: int
    feature_names: tuple[str, ...]
    candidates: tuple[StageEditSourcePairContextualCandidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pair_group, StageEditSourcePairTrainingGroup):
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual group needs its typed pair group"
            )
        object.__setattr__(self, "song_id", _name(self.song_id, "song_id"))
        if (
            isinstance(self.token_index, bool)
            or not isinstance(self.token_index, int)
            or self.token_index < 0
        ):
            raise StageEditSourcePairContextualDiagnosticError(
                "token_index must be a non-negative integer"
            )
        names = tuple(_name(value, "feature_name") for value in self.feature_names)
        if not names or len(names) != len(set(names)):
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual feature names must be non-empty and unique"
            )
        candidates = tuple(self.candidates)
        if len(candidates) != len(self.pair_group.candidates):
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual candidates do not cover the pair group"
            )
        expected_ids = tuple(
            candidate.source_id for candidate in self.pair_group.candidates
        )
        if tuple(candidate.source_id for candidate in candidates) != expected_ids:
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual candidate order differs from canonical pair order"
            )
        if any(len(candidate.features) != len(names) for candidate in candidates):
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual candidate feature dimension mismatch"
            )
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "candidates", candidates)

    @property
    def group_id(self) -> str:
        return self.pair_group.group_id

    @property
    def reference_id(self) -> str:
        return self.pair_group.reference_id

    @property
    def chosen_incoming_source_id(self) -> str:
        return self.pair_group.chosen_incoming_source_id


@dataclass(frozen=True, slots=True)
class StageEditSourcePairContextualCorpus:
    groups: tuple[StageEditSourcePairContextualGroup, ...]
    feature_names: tuple[str, ...]
    source_data_id: str
    token_schema_fingerprint: str
    content_sha256: str
    contract: str = STAGE_EDIT_SOURCE_PAIR_CONTEXTUAL_DIAGNOSTIC_CONTRACT
    schema_version: int = STAGE_EDIT_SOURCE_PAIR_CONTEXTUAL_DIAGNOSTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.contract != STAGE_EDIT_SOURCE_PAIR_CONTEXTUAL_DIAGNOSTIC_CONTRACT:
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual diagnostic contract mismatch"
            )
        if self.schema_version != STAGE_EDIT_SOURCE_PAIR_CONTEXTUAL_DIAGNOSTIC_SCHEMA_VERSION:
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual diagnostic schema mismatch"
            )
        groups = tuple(self.groups)
        if not groups or any(
            not isinstance(group, StageEditSourcePairContextualGroup)
            for group in groups
        ):
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual diagnostic corpus needs typed groups"
            )
        names = tuple(self.feature_names)
        if any(group.feature_names != names for group in groups):
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual corpus feature schema changed between groups"
            )
        if len(self.token_schema_fingerprint) != 64 or len(self.content_sha256) != 64:
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual corpus hashes must be SHA-256"
            )
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "feature_names", names)

    @property
    def song_ids(self) -> tuple[str, ...]:
        return tuple(sorted({group.song_id for group in self.groups}))

    @property
    def reference_ids(self) -> tuple[str, ...]:
        return tuple(sorted({group.reference_id for group in self.groups}))

    @property
    def production_activation_permitted(self) -> bool:
        return False


def materialize_stage_edit_source_pair_contextual_diagnostic_corpus(
    corpus: ReferenceTrainingCorpus,
    *,
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
) -> StageEditSourcePairContextualCorpus:
    """Materialize causal contextual candidates for the existing pair groups."""

    try:
        pair_corpus = materialize_stage_edit_source_pair_training_corpus(
            corpus, schema=schema
        )
    except StageEditSourcePairTrainingError as exc:
        raise StageEditSourcePairContextualDiagnosticError(
            f"base source-pair corpus is invalid: {exc}"
        ) from exc
    examples = {example.reference_id: example for example in corpus.examples}
    if len(examples) != len(corpus.examples):
        raise StageEditSourcePairContextualDiagnosticError(
            "reference ids must be unique"
        )
    song_by_group = dict(pair_corpus.group_song_ids)
    contextual_groups: list[StageEditSourcePairContextualGroup] = []
    canonical_feature_names: tuple[str, ...] | None = None
    for pair_group in pair_corpus.groups:
        example = examples.get(pair_group.reference_id)
        if example is None:
            raise StageEditSourcePairContextualDiagnosticError(
                "pair group lost its reference example"
            )
        prefix, separator, suffix = pair_group.group_id.rpartition(":source-pair:")
        if not separator or prefix != pair_group.reference_id or not suffix.isdigit():
            raise StageEditSourcePairContextualDiagnosticError(
                "pair group token identity is not canonical"
            )
        token_index = int(suffix)
        if token_index >= len(example.sequence.tokens):
            raise StageEditSourcePairContextualDiagnosticError(
                "pair group token index is outside its sequence"
            )
        token = example.sequence.tokens[token_index]
        candidate_by_id = {candidate.source_id: candidate for candidate in token.source_candidates}
        if len(candidate_by_id) != len(token.source_candidates):
            raise StageEditSourcePairContextualDiagnosticError(
                "token source ids must be unique"
            )
        outgoing_candidate = candidate_by_id.get(pair_group.outgoing_source_id)
        if outgoing_candidate is None:
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual group is missing its outgoing source token"
            )
        contextual_candidates: list[StageEditSourcePairContextualCandidate] = []
        for pair_candidate in pair_group.candidates:
            incoming_candidate = candidate_by_id.get(pair_candidate.source_id)
            if incoming_candidate is None:
                raise StageEditSourcePairContextualDiagnosticError(
                    "contextual group is missing an incoming source token"
                )
            items = _contextual_feature_items(
                pair_group,
                pair_candidate,
                token,
                outgoing_candidate,
                incoming_candidate,
                schema=schema,
            )
            names = tuple(name for name, _value in items)
            if canonical_feature_names is None:
                canonical_feature_names = names
            elif names != canonical_feature_names:
                raise StageEditSourcePairContextualDiagnosticError(
                    "contextual feature schema is not stable"
                )
            contextual_candidates.append(
                StageEditSourcePairContextualCandidate(
                    pair_candidate.source_id,
                    tuple(value for _name_value, value in items),
                )
            )
        assert canonical_feature_names is not None
        contextual_groups.append(
            StageEditSourcePairContextualGroup(
                pair_group=pair_group,
                song_id=song_by_group[pair_group.group_id],
                token_index=token_index,
                feature_names=canonical_feature_names,
                candidates=tuple(contextual_candidates),
            )
        )
    ordered = tuple(
        sorted(contextual_groups, key=lambda group: (group.reference_id, group.group_id))
    )
    assert canonical_feature_names is not None
    payload = {
        "contract": STAGE_EDIT_SOURCE_PAIR_CONTEXTUAL_DIAGNOSTIC_CONTRACT,
        "source_data_id": pair_corpus.source_data_id,
        "token_schema_fingerprint": schema.fingerprint,
        "feature_names": list(canonical_feature_names),
        "recent_edit_features_consumed": False,
        "model_logits_consumed": False,
        "groups": [
            {
                "group_id": group.group_id,
                "song_id": group.song_id,
                "reference_id": group.reference_id,
                "chosen_incoming_source_id": group.chosen_incoming_source_id,
                "candidates": [
                    {
                        "source_id": candidate.source_id,
                        "features": list(candidate.features),
                    }
                    for candidate in group.candidates
                ],
            }
            for group in ordered
        ],
    }
    return StageEditSourcePairContextualCorpus(
        groups=ordered,
        feature_names=canonical_feature_names,
        source_data_id=pair_corpus.source_data_id,
        token_schema_fingerprint=schema.fingerprint,
        content_sha256=_canonical_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class StageEditSourcePairContextualDiagnosticConfig:
    steps: int = 120
    learning_rate: float = 0.5
    margin: float = 0.05
    l2_penalty: float = 0.01
    l1_radius: float = 1.0
    handcrafted_baseline_fraction: float = 0.30
    activation_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise StageEditSourcePairContextualDiagnosticError(
                "steps must be a positive integer"
            )
        for field in (
            "learning_rate",
            "margin",
            "l2_penalty",
            "l1_radius",
            "handcrafted_baseline_fraction",
            "activation_tolerance",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        if self.learning_rate <= 0.0 or self.l1_radius <= 0.0:
            raise StageEditSourcePairContextualDiagnosticError(
                "learning rate and L1 radius must be positive"
            )
        if self.margin < 0.0 or self.l2_penalty < 0.0 or self.activation_tolerance < 0.0:
            raise StageEditSourcePairContextualDiagnosticError(
                "margin, L2 penalty, and tolerance must be non-negative"
            )
        if not 0.0 <= self.handcrafted_baseline_fraction <= 1.0:
            raise StageEditSourcePairContextualDiagnosticError(
                "handcrafted baseline fraction must lie in [0, 1]"
            )


def _baseline_scores(
    group: StageEditSourcePairContextualGroup,
) -> tuple[dict[str, float], dict[str, float]]:
    uniform: dict[str, float] = {}
    handcrafted: dict[str, float] = {}
    pair_candidate_by_id = {
        candidate.source_id: candidate for candidate in group.pair_group.candidates
    }
    for candidate in group.candidates:
        pair_candidate = pair_candidate_by_id[candidate.source_id]
        pair_features = materialize_stage_edit_source_pair_features(
            group.pair_group.outgoing,
            pair_candidate.incoming,
        )
        uniform[candidate.source_id] = -float(np.mean(pair_features.values))
        handcrafted[candidate.source_id] = _handcrafted_pair_score(
            group.pair_group.outgoing,
            pair_candidate.incoming,
        )
    return uniform, handcrafted


@dataclass(frozen=True, slots=True)
class StageEditSourcePairContextualResidualModel:
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    handcrafted_baseline_fraction: float
    l1_radius: float

    def __post_init__(self) -> None:
        names = tuple(self.feature_names)
        weights = tuple(
            _finite(value, f"weights[{index}]")
            for index, value in enumerate(self.weights)
        )
        if not names or len(weights) != len(names):
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual residual weight dimension mismatch"
            )
        radius = _finite(self.l1_radius, "l1_radius")
        if radius <= 0.0 or math.fsum(abs(value) for value in weights) > radius + 1.0e-8:
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual residual weights exceed their signed L1 bound"
            )
        fraction = _finite(
            self.handcrafted_baseline_fraction,
            "handcrafted_baseline_fraction",
        )
        if not 0.0 <= fraction <= 1.0:
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual residual baseline fraction is invalid"
            )
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "l1_radius", radius)
        object.__setattr__(self, "handcrafted_baseline_fraction", fraction)

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(
            {
                "contract": STAGE_EDIT_SOURCE_PAIR_CONTEXTUAL_DIAGNOSTIC_CONTRACT,
                "feature_names": list(self.feature_names),
                "weights": list(self.weights),
                "handcrafted_baseline_fraction": self.handcrafted_baseline_fraction,
                "l1_radius": self.l1_radius,
            }
        )

    def score(
        self,
        group: StageEditSourcePairContextualGroup,
        candidate: StageEditSourcePairContextualCandidate,
    ) -> float:
        if group.feature_names != self.feature_names:
            raise StageEditSourcePairContextualDiagnosticError(
                "model/group contextual feature schema mismatch"
            )
        uniform, handcrafted = _baseline_scores(group)
        if candidate.source_id not in uniform:
            raise StageEditSourcePairContextualDiagnosticError(
                "candidate does not belong to contextual group"
            )
        fraction = self.handcrafted_baseline_fraction
        baseline = (
            fraction * handcrafted[candidate.source_id]
            + (1.0 - fraction) * uniform[candidate.source_id]
        )
        return baseline + float(np.dot(self.weights, candidate.features))


@dataclass(frozen=True, slots=True)
class StageEditSourcePairContextualTrainingResult:
    model: StageEditSourcePairContextualResidualModel
    training_song_ids: tuple[str, ...]
    training_reference_ids: tuple[str, ...]
    training_group_ids: tuple[str, ...]
    initial_loss: float
    final_loss: float
    completed_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.model, StageEditSourcePairContextualResidualModel):
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual training model is invalid"
            )
        for field in (
            "training_song_ids",
            "training_reference_ids",
            "training_group_ids",
        ):
            values = tuple(_name(value, field) for value in getattr(self, field))
            if not values or values != tuple(sorted(set(values))):
                raise StageEditSourcePairContextualDiagnosticError(
                    f"{field} must be sorted and unique"
                )
            object.__setattr__(self, field, values)
        initial = _finite(self.initial_loss, "initial_loss")
        final = _finite(self.final_loss, "final_loss")
        if final > initial + 1.0e-9:
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual training objective regressed"
            )
        if (
            isinstance(self.completed_steps, bool)
            or not isinstance(self.completed_steps, int)
            or self.completed_steps < 0
        ):
            raise StageEditSourcePairContextualDiagnosticError(
                "completed_steps must be a non-negative integer"
            )
        object.__setattr__(self, "initial_loss", initial)
        object.__setattr__(self, "final_loss", final)


def _project_signed_l1(values: np.ndarray, radius: float) -> np.ndarray:
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise StageEditSourcePairContextualDiagnosticError(
            "signed L1 projection input is invalid"
        )
    if float(np.sum(np.abs(values))) <= radius:
        return values.copy()
    ordered = np.sort(np.abs(values))[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, len(values) + 1, dtype=np.float64)
    support = ordered * indices > cumulative - radius
    if not np.any(support):
        raise StageEditSourcePairContextualDiagnosticError(
            "signed L1 projection found no support"
        )
    rho = int(np.flatnonzero(support)[-1])
    threshold = (cumulative[rho] - radius) / float(rho + 1)
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _loss_and_gradient(
    weights: np.ndarray,
    differences: np.ndarray,
    baseline_differences: np.ndarray,
    sample_weights: np.ndarray,
    *,
    margin: float,
    l2_penalty: float,
) -> tuple[float, np.ndarray]:
    logits = margin - baseline_differences - np.dot(differences, weights)
    losses = np.logaddexp(0.0, logits)
    sigmoid = np.empty_like(logits)
    positive = logits >= 0.0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    sigmoid[~positive] = exponential / (1.0 + exponential)
    loss = float(
        np.dot(sample_weights, losses)
        + 0.5 * l2_penalty * np.dot(weights, weights)
    )
    gradient = -np.dot(differences.T, sample_weights * sigmoid) + l2_penalty * weights
    if not math.isfinite(loss) or not np.all(np.isfinite(gradient)):
        raise StageEditSourcePairContextualDiagnosticError(
            "contextual residual objective became non-finite"
        )
    return loss, gradient


def train_stage_edit_source_pair_contextual_residual(
    groups: Sequence[StageEditSourcePairContextualGroup],
    *,
    config: StageEditSourcePairContextualDiagnosticConfig = (
        StageEditSourcePairContextualDiagnosticConfig()
    ),
) -> StageEditSourcePairContextualTrainingResult:
    """Fit signed L1-bounded residuals with equal song/reference influence."""

    values = tuple(
        sorted(groups, key=lambda group: (group.song_id, group.reference_id, group.group_id))
    )
    if not values or any(
        not isinstance(group, StageEditSourcePairContextualGroup)
        for group in values
    ):
        raise StageEditSourcePairContextualDiagnosticError(
            "contextual training needs typed groups"
        )
    if not isinstance(config, StageEditSourcePairContextualDiagnosticConfig):
        raise StageEditSourcePairContextualDiagnosticError(
            "contextual training config is invalid"
        )
    feature_names = values[0].feature_names
    if any(group.feature_names != feature_names for group in values):
        raise StageEditSourcePairContextualDiagnosticError(
            "contextual training feature schema changed"
        )
    songs = tuple(sorted({group.song_id for group in values}))
    differences: list[np.ndarray] = []
    baseline_differences: list[float] = []
    sample_weights: list[float] = []
    fraction = config.handcrafted_baseline_fraction
    for song_id in songs:
        song_groups = tuple(group for group in values if group.song_id == song_id)
        references = tuple(sorted({group.reference_id for group in song_groups}))
        for reference_id in references:
            reference_groups = tuple(
                group for group in song_groups if group.reference_id == reference_id
            )
            for group in reference_groups:
                candidates = {candidate.source_id: candidate for candidate in group.candidates}
                chosen = candidates[group.chosen_incoming_source_id]
                uniform, handcrafted = _baseline_scores(group)
                chosen_baseline = (
                    fraction * handcrafted[chosen.source_id]
                    + (1.0 - fraction) * uniform[chosen.source_id]
                )
                alternatives = tuple(
                    candidate
                    for candidate in group.candidates
                    if candidate.source_id != chosen.source_id
                )
                for alternative in alternatives:
                    alternative_baseline = (
                        fraction * handcrafted[alternative.source_id]
                        + (1.0 - fraction) * uniform[alternative.source_id]
                    )
                    differences.append(
                        np.asarray(chosen.features, dtype=np.float64)
                        - np.asarray(alternative.features, dtype=np.float64)
                    )
                    baseline_differences.append(
                        chosen_baseline - alternative_baseline
                    )
                    sample_weights.append(
                        1.0
                        / len(songs)
                        / len(references)
                        / len(reference_groups)
                        / len(alternatives)
                    )
    matrix = np.stack(differences)
    offsets = np.asarray(baseline_differences, dtype=np.float64)
    balances = np.asarray(sample_weights, dtype=np.float64)
    if not math.isclose(float(np.sum(balances)), 1.0, abs_tol=1.0e-9):
        raise StageEditSourcePairContextualDiagnosticError(
            "song-balanced comparison weights do not sum to one"
        )
    weights = np.zeros(len(feature_names), dtype=np.float64)
    initial_loss, _ = _loss_and_gradient(
        weights,
        matrix,
        offsets,
        balances,
        margin=config.margin,
        l2_penalty=config.l2_penalty,
    )
    current_loss = initial_loss
    current_rate = config.learning_rate
    completed_steps = 0
    for _step in range(config.steps):
        current_loss, gradient = _loss_and_gradient(
            weights,
            matrix,
            offsets,
            balances,
            margin=config.margin,
            l2_penalty=config.l2_penalty,
        )
        accepted = False
        trial_rate = current_rate
        for _attempt in range(30):
            proposed = _project_signed_l1(
                weights - trial_rate * gradient,
                config.l1_radius,
            )
            proposed_loss, _ = _loss_and_gradient(
                proposed,
                matrix,
                offsets,
                balances,
                margin=config.margin,
                l2_penalty=config.l2_penalty,
            )
            if proposed_loss <= current_loss + 1.0e-12:
                weights = proposed
                current_loss = proposed_loss
                current_rate = min(config.learning_rate, trial_rate * 1.02)
                completed_steps += 1
                accepted = True
                break
            trial_rate *= 0.5
        if not accepted:
            break
    final_loss, _ = _loss_and_gradient(
        weights,
        matrix,
        offsets,
        balances,
        margin=config.margin,
        l2_penalty=config.l2_penalty,
    )
    return StageEditSourcePairContextualTrainingResult(
        model=StageEditSourcePairContextualResidualModel(
            feature_names=feature_names,
            weights=tuple(float(value) for value in weights),
            handcrafted_baseline_fraction=fraction,
            l1_radius=config.l1_radius,
        ),
        training_song_ids=songs,
        training_reference_ids=tuple(sorted({group.reference_id for group in values})),
        training_group_ids=tuple(sorted(group.group_id for group in values)),
        initial_loss=initial_loss,
        final_loss=final_loss,
        completed_steps=completed_steps,
    )


def evaluate_stage_edit_source_pair_contextual_ranking(
    groups: Sequence[StageEditSourcePairContextualGroup],
    *,
    model: StageEditSourcePairContextualResidualModel | None,
    scorer_name: str,
    baseline: str | None = None,
) -> StageEditSourcePairRankingMetrics:
    values = tuple(groups)
    if not values:
        raise StageEditSourcePairContextualDiagnosticError(
            "contextual ranking evaluation needs groups"
        )
    if (model is None) == (baseline is None):
        raise StageEditSourcePairContextualDiagnosticError(
            "contextual evaluation needs exactly one scorer mode"
        )
    if baseline not in (None, "uniform", "handcrafted"):
        raise StageEditSourcePairContextualDiagnosticError(
            "unsupported contextual ranking baseline"
        )
    top1: dict[str, list[float]] = {}
    pairwise: dict[str, list[float]] = {}
    reciprocal: dict[str, list[float]] = {}
    margins: dict[str, list[float]] = {}
    tolerance = 1.0e-12
    for group in values:
        uniform, handcrafted = _baseline_scores(group)
        score_by_source: dict[str, float] = {}
        for candidate in group.candidates:
            if model is not None:
                score = model.score(group, candidate)
            elif baseline == "uniform":
                score = uniform[candidate.source_id]
            else:
                score = handcrafted[candidate.source_id]
            score_by_source[candidate.source_id] = _finite(score, "candidate score")
        chosen = score_by_source[group.chosen_incoming_source_id]
        alternatives = tuple(
            score
            for source_id, score in score_by_source.items()
            if source_id != group.chosen_incoming_source_id
        )
        best = max(score_by_source.values())
        tied_best = sum(
            abs(score - best) <= tolerance for score in score_by_source.values()
        )
        top_credit = 1.0 / tied_best if abs(chosen - best) <= tolerance else 0.0
        better = sum(score > chosen + tolerance for score in alternatives)
        tied = sum(abs(score - chosen) <= tolerance for score in alternatives)
        pair_credit = float(
            np.mean(
                [
                    1.0
                    if chosen > score + tolerance
                    else 0.5
                    if abs(chosen - score) <= tolerance
                    else 0.0
                    for score in alternatives
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

    def balanced(values_by_reference: Mapping[str, Sequence[float]]) -> float:
        return float(
            np.mean(
                [
                    float(np.mean(tuple(values_by_reference[reference_id])))
                    for reference_id in sorted(values_by_reference)
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
class StageEditSourcePairContextualFold:
    heldout_song_id: str
    training: StageEditSourcePairContextualTrainingResult
    heldout_reference_ids: tuple[str, ...]
    heldout_group_ids: tuple[str, ...]
    learned_metrics: StageEditSourcePairRankingMetrics
    uniform_metrics: StageEditSourcePairRankingMetrics
    handcrafted_metrics: StageEditSourcePairRankingMetrics
    activation_gate: StageEditSourcePairActivationGate
    activation_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "heldout_song_id", _name(self.heldout_song_id, "heldout_song_id")
        )
        for field in ("heldout_reference_ids", "heldout_group_ids"):
            values = tuple(_name(value, field) for value in getattr(self, field))
            if not values or values != tuple(sorted(set(values))):
                raise StageEditSourcePairContextualDiagnosticError(
                    f"{field} must be sorted and unique"
                )
            object.__setattr__(self, field, values)
        if self.heldout_song_id in self.training.training_song_ids:
            raise StageEditSourcePairContextualDiagnosticError(
                "held-out song leaked into contextual training"
            )
        if set(self.heldout_reference_ids).intersection(
            self.training.training_reference_ids
        ):
            raise StageEditSourcePairContextualDiagnosticError(
                "held-out reference leaked into contextual training"
            )
        if set(self.heldout_group_ids).intersection(self.training.training_group_ids):
            raise StageEditSourcePairContextualDiagnosticError(
                "held-out group leaked into contextual training"
            )
        tolerance = _finite(self.activation_tolerance, "activation_tolerance")
        if tolerance < 0.0:
            raise StageEditSourcePairContextualDiagnosticError(
                "activation_tolerance must be non-negative"
            )
        object.__setattr__(self, "activation_tolerance", tolerance)
        identity = (len(self.heldout_reference_ids), len(self.heldout_group_ids))
        for metrics in (
            self.learned_metrics,
            self.uniform_metrics,
            self.handcrafted_metrics,
        ):
            if not isinstance(metrics, StageEditSourcePairRankingMetrics) or (
                metrics.reference_count,
                metrics.group_count,
            ) != identity:
                raise StageEditSourcePairContextualDiagnosticError(
                    "contextual fold metrics do not match held-out identities"
                )
        expected_gate = stage_edit_source_pair_no_worse_activation_gate(
            self.learned_metrics,
            self.uniform_metrics,
            self.handcrafted_metrics,
            tolerance=tolerance,
        )
        if self.activation_gate != expected_gate:
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual fold gate differs from the unchanged activation gate"
            )


def _aggregate_metrics(
    folds: Sequence[StageEditSourcePairContextualFold],
    field: str,
    scorer_name: str,
) -> StageEditSourcePairRankingMetrics:
    metrics = tuple(getattr(fold, field) for fold in folds)
    reference_count = sum(value.reference_count for value in metrics)
    group_count = sum(value.group_count for value in metrics)

    def weighted(name: str) -> float:
        return math.fsum(
            value.reference_count * getattr(value, name) for value in metrics
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


@dataclass(frozen=True, slots=True)
class StageEditSourcePairContextualCrossValidationResult:
    corpus: StageEditSourcePairContextualCorpus
    config: StageEditSourcePairContextualDiagnosticConfig
    folds: tuple[StageEditSourcePairContextualFold, ...]
    learned_metrics: StageEditSourcePairRankingMetrics
    uniform_metrics: StageEditSourcePairRankingMetrics
    handcrafted_metrics: StageEditSourcePairRankingMetrics
    aggregate_gate: StageEditSourcePairActivationGate

    def __post_init__(self) -> None:
        if not isinstance(self.corpus, StageEditSourcePairContextualCorpus) or not isinstance(
            self.config, StageEditSourcePairContextualDiagnosticConfig
        ):
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual cross-validation corpus/config is invalid"
            )
        folds = tuple(self.folds)
        if tuple(fold.heldout_song_id for fold in folds) != self.corpus.song_ids:
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual folds do not hold out every song canonically"
            )
        expected_group_ids = {group.group_id for group in self.corpus.groups}
        heldout_group_ids = [
            group_id for fold in folds for group_id in fold.heldout_group_ids
        ]
        if (
            len(heldout_group_ids) != len(set(heldout_group_ids))
            or set(heldout_group_ids) != expected_group_ids
        ):
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual held-out folds do not cover every group exactly once"
            )
        expected_aggregates = (
            _aggregate_metrics(
                folds,
                "learned_metrics",
                "contextual_signed_residual:song_disjoint_leave_one_song_out",
            ),
            _aggregate_metrics(
                folds,
                "uniform_metrics",
                "uniform:song_disjoint_leave_one_song_out",
            ),
            _aggregate_metrics(
                folds,
                "handcrafted_metrics",
                (
                    f"{STAGE_EDIT_SOURCE_PAIR_HANDCRAFTED_BASELINE_CONTRACT}:"
                    "song_disjoint_leave_one_song_out"
                ),
            ),
        )
        if expected_aggregates != (
            self.learned_metrics,
            self.uniform_metrics,
            self.handcrafted_metrics,
        ):
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual aggregate metrics do not round-trip from folds"
            )
        expected_gate = stage_edit_source_pair_no_worse_activation_gate(
            self.learned_metrics,
            self.uniform_metrics,
            self.handcrafted_metrics,
            tolerance=self.config.activation_tolerance,
        )
        if self.aggregate_gate != expected_gate:
            raise StageEditSourcePairContextualDiagnosticError(
                "contextual aggregate gate differs from the unchanged activation gate"
            )
        object.__setattr__(self, "folds", folds)

    @property
    def authorized(self) -> bool:
        return self.aggregate_gate.authorized and all(
            fold.activation_gate.authorized for fold in self.folds
        )

    @property
    def production_activation_permitted(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE_EDIT_SOURCE_PAIR_CONTEXTUAL_DIAGNOSTIC_SCHEMA_VERSION,
            "contract": STAGE_EDIT_SOURCE_PAIR_CONTEXTUAL_DIAGNOSTIC_CONTRACT,
            "diagnostic_only": True,
            "production_activation_permitted": False,
            "causal_evidence": {
                "pair_deltas_consumed": True,
                "absolute_outgoing_incoming_shot_evidence_consumed": True,
                "music_section_style_context_consumed": True,
                "source_candidate_evidence_consumed": True,
                "embedding_summaries_consumed": True,
                "recent_edit_features_consumed": False,
                "model_logits_consumed": False,
                "reference_match_input_evidence_consumed": False,
            },
            "corpus_sha256": self.corpus.content_sha256,
            "feature_count": len(self.corpus.feature_names),
            "song_ids": list(self.corpus.song_ids),
            "folds": [
                {
                    "heldout_song_id": fold.heldout_song_id,
                    "training_song_ids": list(fold.training.training_song_ids),
                    "heldout_reference_ids": list(fold.heldout_reference_ids),
                    "heldout_group_ids": list(fold.heldout_group_ids),
                    "model_fingerprint": fold.training.model.fingerprint,
                    "learned": fold.learned_metrics.to_dict(),
                    "uniform": fold.uniform_metrics.to_dict(),
                    "handcrafted": fold.handcrafted_metrics.to_dict(),
                    "activation_gate": fold.activation_gate.to_dict(),
                    "activation_tolerance": fold.activation_tolerance,
                }
                for fold in self.folds
            ],
            "aggregate": {
                "learned": self.learned_metrics.to_dict(),
                "uniform": self.uniform_metrics.to_dict(),
                "handcrafted": self.handcrafted_metrics.to_dict(),
                "activation_gate": self.aggregate_gate.to_dict(),
            },
            "all_fold_and_aggregate_gate_passed": self.authorized,
        }


def cross_validate_stage_edit_source_pair_contextual_diagnostic(
    corpus: StageEditSourcePairContextualCorpus,
    *,
    config: StageEditSourcePairContextualDiagnosticConfig = (
        StageEditSourcePairContextualDiagnosticConfig()
    ),
) -> StageEditSourcePairContextualCrossValidationResult:
    """Run exact leave-one-song-out evaluation without production authority."""

    if not isinstance(corpus, StageEditSourcePairContextualCorpus):
        raise StageEditSourcePairContextualDiagnosticError(
            "contextual LOSO needs a materialized contextual corpus"
        )
    if len(corpus.song_ids) < 2:
        raise StageEditSourcePairContextualDiagnosticError(
            "contextual LOSO needs at least two songs"
        )
    folds: list[StageEditSourcePairContextualFold] = []
    for heldout_song_id in corpus.song_ids:
        training_groups = tuple(
            group for group in corpus.groups if group.song_id != heldout_song_id
        )
        heldout_groups = tuple(
            group for group in corpus.groups if group.song_id == heldout_song_id
        )
        training = train_stage_edit_source_pair_contextual_residual(
            training_groups,
            config=config,
        )
        learned = evaluate_stage_edit_source_pair_contextual_ranking(
            heldout_groups,
            model=training.model,
            scorer_name=f"contextual_signed_residual:heldout:{heldout_song_id}",
        )
        uniform = evaluate_stage_edit_source_pair_contextual_ranking(
            heldout_groups,
            model=None,
            scorer_name=f"uniform:heldout:{heldout_song_id}",
            baseline="uniform",
        )
        handcrafted = evaluate_stage_edit_source_pair_contextual_ranking(
            heldout_groups,
            model=None,
            scorer_name=(
                f"{STAGE_EDIT_SOURCE_PAIR_HANDCRAFTED_BASELINE_CONTRACT}:"
                f"heldout:{heldout_song_id}"
            ),
            baseline="handcrafted",
        )
        folds.append(
            StageEditSourcePairContextualFold(
                heldout_song_id=heldout_song_id,
                training=training,
                heldout_reference_ids=tuple(
                    sorted({group.reference_id for group in heldout_groups})
                ),
                heldout_group_ids=tuple(
                    sorted(group.group_id for group in heldout_groups)
                ),
                learned_metrics=learned,
                uniform_metrics=uniform,
                handcrafted_metrics=handcrafted,
                activation_gate=stage_edit_source_pair_no_worse_activation_gate(
                    learned,
                    uniform,
                    handcrafted,
                    tolerance=config.activation_tolerance,
                ),
                activation_tolerance=config.activation_tolerance,
            )
        )
    fold_values = tuple(folds)
    learned_aggregate = _aggregate_metrics(
        fold_values,
        "learned_metrics",
        "contextual_signed_residual:song_disjoint_leave_one_song_out",
    )
    uniform_aggregate = _aggregate_metrics(
        fold_values,
        "uniform_metrics",
        "uniform:song_disjoint_leave_one_song_out",
    )
    handcrafted_aggregate = _aggregate_metrics(
        fold_values,
        "handcrafted_metrics",
        f"{STAGE_EDIT_SOURCE_PAIR_HANDCRAFTED_BASELINE_CONTRACT}:song_disjoint_leave_one_song_out",
    )
    return StageEditSourcePairContextualCrossValidationResult(
        corpus=corpus,
        config=config,
        folds=fold_values,
        learned_metrics=learned_aggregate,
        uniform_metrics=uniform_aggregate,
        handcrafted_metrics=handcrafted_aggregate,
        aggregate_gate=stage_edit_source_pair_no_worse_activation_gate(
            learned_aggregate,
            uniform_aggregate,
            handcrafted_aggregate,
            tolerance=config.activation_tolerance,
        ),
    )


__all__ = [
    "STAGE_EDIT_SOURCE_PAIR_CONTEXTUAL_DIAGNOSTIC_CONTRACT",
    "STAGE_EDIT_SOURCE_PAIR_CONTEXTUAL_DIAGNOSTIC_SCHEMA_VERSION",
    "StageEditSourcePairContextualCandidate",
    "StageEditSourcePairContextualCorpus",
    "StageEditSourcePairContextualCrossValidationResult",
    "StageEditSourcePairContextualDiagnosticConfig",
    "StageEditSourcePairContextualDiagnosticError",
    "StageEditSourcePairContextualFold",
    "StageEditSourcePairContextualGroup",
    "StageEditSourcePairContextualResidualModel",
    "StageEditSourcePairContextualTrainingResult",
    "cross_validate_stage_edit_source_pair_contextual_diagnostic",
    "evaluate_stage_edit_source_pair_contextual_ranking",
    "materialize_stage_edit_source_pair_contextual_diagnostic_corpus",
    "train_stage_edit_source_pair_contextual_residual",
]
