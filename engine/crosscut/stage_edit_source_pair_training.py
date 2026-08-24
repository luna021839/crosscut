"""Reference-corpus bridge for source-pair continuity scorer training.

This module converts leakage-safe :class:`ReferenceTrainingCorpus` examples
into the source-pair groups consumed by :mod:`stage_edit_source_pair_scorer`.
Only confirmed non-opening source switches are labels.  At each such switch,
the outgoing source and every incoming candidate are observed at the same exact
token, preserving the causal same-time continuity contract.

The module intentionally does not import the semi-Markov decoder.  Its fixed
handcrafted baseline is a local, versioned snapshot of the current continuity
formula so scorer training cannot introduce an import cycle.
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
    StageEditSourcePairCandidate,
    StageEditSourcePairInputError,
    StageEditSourcePairScorer,
    StageEditSourcePairTrainingGroup,
    StageEditSourcePairTrainingResult,
    train_stage_edit_source_pair_scorer,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditToken,
    StageEditTokenSchema,
)


STAGE_EDIT_SOURCE_PAIR_TRAINING_SCHEMA_VERSION = 1
STAGE_EDIT_SOURCE_PAIR_TRAINING_CONTRACT = (
    "confirmed_nonopening_same_token_source_pair_reference_training_v1"
)
STAGE_EDIT_SOURCE_PAIR_HANDCRAFTED_BASELINE_CONTRACT = (
    "semimarkov_boundary_continuity_handcrafted_snapshot_v1"
)
STAGE_EDIT_SOURCE_PAIR_ACTIVATION_GATE_CONTRACT = (
    "reference_balanced_top1_pairwise_mrr_no_worse_than_uniform_and_"
    "handcrafted_v1"
)


class StageEditSourcePairTrainingError(StageEditSourcePairInputError):
    """Reference corpus cannot produce trustworthy source-pair supervision."""


_PAIR_OBSERVATION_TO_SCHEMA_NAME = {
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


def _name(value: object, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise StageEditSourcePairTrainingError(f"{field} cannot be blank")
    return result


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise StageEditSourcePairTrainingError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StageEditSourcePairTrainingError(
            f"{field} must be numeric"
        ) from exc
    if not math.isfinite(result):
        raise StageEditSourcePairTrainingError(f"{field} must be finite")
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


@dataclass(frozen=True, slots=True)
class StageEditSourcePairObservation:
    """Schema-named, missing-mask-aware observations for one source/token."""

    face_scale: float | None = None
    face_center_x: float | None = None
    face_center_y: float | None = None
    body_center_x: float | None = None
    body_center_y: float | None = None
    pose_motion: float | None = None
    camera_motion_x: float | None = None
    camera_motion_y: float | None = None
    camera_zoom_velocity: float | None = None
    camera_roll_normalized: float | None = None
    sharpness: float | None = None
    exposure: float | None = None
    color_temperature_normalized: float | None = None
    detected_person_count_normalized: float | None = None
    formation_person_count_normalized: float | None = None
    formation_pose_coverage: float | None = None

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _finite(value, field))

    def to_dict(self) -> dict[str, float | None]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }


def stage_edit_source_pair_observation_from_candidate(
    candidate: SourceCandidateToken,
    *,
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
) -> StageEditSourcePairObservation:
    """Read only named, unmasked source features from one exact token."""

    if not isinstance(candidate, SourceCandidateToken):
        raise StageEditSourcePairTrainingError(
            "source-pair observation requires SourceCandidateToken"
        )
    names = tuple(schema.source_feature_names)
    if len(names) != len(set(names)):
        raise StageEditSourcePairTrainingError(
            "source feature schema names must be unique"
        )
    missing_schema_names = sorted(
        set(_PAIR_OBSERVATION_TO_SCHEMA_NAME.values()) - set(names)
    )
    if missing_schema_names:
        raise StageEditSourcePairTrainingError(
            "source feature schema lacks pair observations: "
            + ", ".join(missing_schema_names)
        )
    if len(candidate.features) != len(names):
        raise StageEditSourcePairTrainingError(
            f"source {candidate.source_id!r} feature length differs from schema"
        )
    if candidate.missing_mask and len(candidate.missing_mask) != len(names):
        raise StageEditSourcePairTrainingError(
            f"source {candidate.source_id!r} missing-mask length differs from schema"
        )
    missing = (
        tuple(candidate.missing_mask)
        if candidate.missing_mask
        else (False,) * len(names)
    )
    if any(not isinstance(value, (bool, np.bool_)) for value in missing):
        raise StageEditSourcePairTrainingError(
            f"source {candidate.source_id!r} missing mask must be boolean"
        )
    index_by_name = {name: index for index, name in enumerate(names)}
    values: dict[str, float | None] = {}
    for field, schema_name in _PAIR_OBSERVATION_TO_SCHEMA_NAME.items():
        index = index_by_name[schema_name]
        values[field] = (
            None
            if bool(missing[index])
            else _finite(
                candidate.features[index],
                f"source {candidate.source_id!r} feature {schema_name}",
            )
        )
    return StageEditSourcePairObservation(**values)


def _eligible_candidates_at_token(
    token: StageEditToken,
    *,
    schema: StageEditTokenSchema,
    reference_id: str,
    token_index: int,
) -> tuple[
    dict[str, SourceCandidateToken],
    dict[str, StageEditSourcePairObservation],
]:
    if not isinstance(token, StageEditToken):
        raise StageEditSourcePairTrainingError(
            f"reference {reference_id} token {token_index} is invalid"
        )
    by_id: dict[str, SourceCandidateToken] = {}
    eligible: dict[str, StageEditSourcePairObservation] = {}
    for candidate in token.source_candidates:
        if not isinstance(candidate, SourceCandidateToken):
            raise StageEditSourcePairTrainingError(
                f"reference {reference_id} token {token_index} has an invalid candidate"
            )
        source_id = _name(candidate.source_id, "candidate source_id")
        if source_id in by_id:
            raise StageEditSourcePairTrainingError(
                f"reference {reference_id} token {token_index} has duplicate source ids"
            )
        if not isinstance(candidate.available, bool) or not isinstance(
            candidate.hard_gate_passed, bool
        ):
            raise StageEditSourcePairTrainingError(
                f"reference {reference_id} token {token_index} source gates must be boolean"
            )
        observation = stage_edit_source_pair_observation_from_candidate(
            candidate, schema=schema
        )
        by_id[source_id] = candidate
        if candidate.available and candidate.hard_gate_passed:
            eligible[source_id] = observation
    return by_id, eligible


def _group_payload(
    group: StageEditSourcePairTrainingGroup,
    *,
    song_id: str,
) -> dict[str, Any]:
    outgoing = group.outgoing
    if not isinstance(outgoing, StageEditSourcePairObservation):
        raise StageEditSourcePairTrainingError(
            "source-pair corpus outgoing observation type changed"
        )
    candidates = []
    for candidate in sorted(group.candidates, key=lambda value: value.source_id):
        if not isinstance(candidate.incoming, StageEditSourcePairObservation):
            raise StageEditSourcePairTrainingError(
                "source-pair corpus incoming observation type changed"
            )
        candidates.append(
            {
                "source_id": candidate.source_id,
                "observation": candidate.incoming.to_dict(),
            }
        )
    return {
        "group_id": group.group_id,
        "song_id": song_id,
        "reference_id": group.reference_id,
        "outgoing_source_id": group.outgoing_source_id,
        "outgoing": outgoing.to_dict(),
        "chosen_incoming_source_id": group.chosen_incoming_source_id,
        "candidates": candidates,
    }


@dataclass(frozen=True, slots=True)
class StageEditSourcePairTrainingCorpus:
    groups: tuple[StageEditSourcePairTrainingGroup, ...]
    source_data_id: str
    token_schema_fingerprint: str
    group_song_ids: tuple[tuple[str, str], ...]
    song_ids: tuple[str, ...]
    reference_ids: tuple[str, ...]
    per_song_group_counts: tuple[tuple[str, int], ...]
    per_reference_group_counts: tuple[tuple[str, int], ...]
    input_reference_count: int
    opening_selection_excluded_count: int
    unconfirmed_selection_excluded_count: int
    content_sha256: str
    contract: str = STAGE_EDIT_SOURCE_PAIR_TRAINING_CONTRACT
    schema_version: int = STAGE_EDIT_SOURCE_PAIR_TRAINING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        if not groups or any(
            not isinstance(group, StageEditSourcePairTrainingGroup)
            for group in groups
        ):
            raise StageEditSourcePairTrainingError(
                "source-pair training corpus needs typed groups"
            )
        group_ids = tuple(group.group_id for group in groups)
        if len(group_ids) != len(set(group_ids)):
            raise StageEditSourcePairTrainingError(
                "source-pair training corpus group ids must be unique"
            )
        if self.contract != STAGE_EDIT_SOURCE_PAIR_TRAINING_CONTRACT:
            raise StageEditSourcePairTrainingError(
                "source-pair training corpus contract mismatch"
            )
        if self.schema_version != STAGE_EDIT_SOURCE_PAIR_TRAINING_SCHEMA_VERSION:
            raise StageEditSourcePairTrainingError(
                "source-pair training corpus schema version mismatch"
            )
        _name(self.source_data_id, "source_data_id")
        if len(self.token_schema_fingerprint) != 64:
            raise StageEditSourcePairTrainingError(
                "token schema fingerprint must be SHA-256"
            )
        if any(
            value not in "0123456789abcdef"
            for value in self.token_schema_fingerprint
        ):
            raise StageEditSourcePairTrainingError(
                "token schema fingerprint must be lowercase hexadecimal"
            )
        song_by_group = dict(self.group_song_ids)
        if len(song_by_group) != len(self.group_song_ids) or set(song_by_group) != set(
            group_ids
        ):
            raise StageEditSourcePairTrainingError(
                "group/song identities do not cover every training group exactly"
            )
        if tuple(sorted(set(song_by_group.values()))) != tuple(self.song_ids):
            raise StageEditSourcePairTrainingError("song ids/counts are inconsistent")
        if tuple(sorted({group.reference_id for group in groups})) != tuple(
            self.reference_ids
        ):
            raise StageEditSourcePairTrainingError(
                "reference ids/counts are inconsistent"
            )
        actual_per_song: dict[str, int] = {}
        actual_per_reference: dict[str, int] = {}
        for group in groups:
            song_id = _name(song_by_group[group.group_id], "group song_id")
            actual_per_song[song_id] = actual_per_song.get(song_id, 0) + 1
            actual_per_reference[group.reference_id] = (
                actual_per_reference.get(group.reference_id, 0) + 1
            )
        if tuple(sorted(actual_per_song.items())) != tuple(
            self.per_song_group_counts
        ):
            raise StageEditSourcePairTrainingError(
                "per-song source-pair group counts are inconsistent"
            )
        if tuple(sorted(actual_per_reference.items())) != tuple(
            self.per_reference_group_counts
        ):
            raise StageEditSourcePairTrainingError(
                "per-reference source-pair group counts are inconsistent"
            )
        for field in (
            "input_reference_count",
            "opening_selection_excluded_count",
            "unconfirmed_selection_excluded_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise StageEditSourcePairTrainingError(
                    f"{field} must be a non-negative integer"
                )
        if self.input_reference_count < len(self.reference_ids):
            raise StageEditSourcePairTrainingError(
                "input reference count is smaller than trained reference count"
            )
        if len(self.content_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.content_sha256
        ):
            raise StageEditSourcePairTrainingError(
                "source-pair corpus content hash must be lowercase SHA-256"
            )
        object.__setattr__(self, "groups", groups)

    @property
    def song_count(self) -> int:
        return len(self.song_ids)

    @property
    def reference_count(self) -> int:
        return len(self.reference_ids)

    @property
    def group_count(self) -> int:
        return len(self.groups)

    def to_dict(self) -> dict[str, Any]:
        song_by_group = dict(self.group_song_ids)
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "source_data_id": self.source_data_id,
            "token_schema_fingerprint": self.token_schema_fingerprint,
            "content_sha256": self.content_sha256,
            "song_count": self.song_count,
            "reference_count": self.reference_count,
            "input_reference_count": self.input_reference_count,
            "group_count": self.group_count,
            "song_ids": list(self.song_ids),
            "reference_ids": list(self.reference_ids),
            "per_song_group_counts": dict(self.per_song_group_counts),
            "per_reference_group_counts": dict(
                self.per_reference_group_counts
            ),
            "opening_selection_excluded_count": (
                self.opening_selection_excluded_count
            ),
            "unconfirmed_selection_excluded_count": (
                self.unconfirmed_selection_excluded_count
            ),
            "groups": [
                _group_payload(group, song_id=song_by_group[group.group_id])
                for group in self.groups
            ],
        }


def materialize_stage_edit_source_pair_training_corpus(
    corpus: ReferenceTrainingCorpus,
    *,
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
) -> StageEditSourcePairTrainingCorpus:
    """Build deterministic same-token groups from confirmed source switches."""

    if not isinstance(corpus, ReferenceTrainingCorpus):
        raise StageEditSourcePairTrainingError(
            "source-pair training requires ReferenceTrainingCorpus"
        )
    source_data_id = _name(corpus.data_id, "reference corpus data_id")
    examples = tuple(corpus.examples)
    if not examples:
        raise StageEditSourcePairTrainingError(
            "source-pair training requires reference examples"
        )
    references = tuple(_name(value.reference_id, "reference_id") for value in examples)
    if len(references) != len(set(references)):
        raise StageEditSourcePairTrainingError(
            "source-pair training reference ids must be unique"
        )

    groups: list[StageEditSourcePairTrainingGroup] = []
    group_song_ids: list[tuple[str, str]] = []
    opening_excluded = 0
    unconfirmed_excluded = 0
    per_song: dict[str, int] = {}
    per_reference: dict[str, int] = {}
    ordered_examples = sorted(
        examples,
        key=lambda value: (
            str(value.song_id or "").strip(),
            str(value.reference_id).strip(),
        ),
    )
    for example in ordered_examples:
        reference_id = _name(example.reference_id, "reference_id")
        song_id = _name(example.song_id, f"reference {reference_id} song_id")
        sequence = example.sequence
        targets = example.targets
        if sequence.sequence_id != targets.sequence_id:
            raise StageEditSourcePairTrainingError(
                f"reference {reference_id} sequence/target identity mismatch"
            )
        if len(sequence.tokens) != len(targets.targets):
            raise StageEditSourcePairTrainingError(
                f"reference {reference_id} sequence/target length mismatch"
            )
        if not sequence.tokens:
            raise StageEditSourcePairTrainingError(
                f"reference {reference_id} has no tokens"
            )

        prior_source_id: str | None = None
        for token_index, (token, target) in enumerate(
            zip(sequence.tokens, targets.targets, strict=True)
        ):
            selected_source_id = (
                _name(target.selected_source_id, "selected_source_id")
                if target.selected_source_id is not None
                else None
            )
            if target.cut is None:
                cut_value: float | None = None
            else:
                if isinstance(target.cut, bool) or not isinstance(
                    target.cut, (int, float, np.number)
                ):
                    raise StageEditSourcePairTrainingError(
                        f"reference {reference_id} cut target must be numeric 0/1"
                    )
                cut_value = _finite(target.cut, "cut target")
                if cut_value not in (0.0, 1.0):
                    raise StageEditSourcePairTrainingError(
                        f"reference {reference_id} cut target must equal 0 or 1"
                    )
            confirmed = (
                cut_value == 1.0
                and selected_source_id is not None
            )
            if token_index == 0 and selected_source_id is not None:
                _all, opening_eligible = _eligible_candidates_at_token(
                    token,
                    schema=schema,
                    reference_id=reference_id,
                    token_index=token_index,
                )
                if selected_source_id not in opening_eligible:
                    raise StageEditSourcePairTrainingError(
                        f"reference {reference_id} opening source is not eligible"
                    )
                prior_source_id = selected_source_id
                opening_excluded += 1
                continue
            if selected_source_id is None:
                continue
            if not confirmed:
                unconfirmed_excluded += 1
                continue
            if prior_source_id is None:
                raise StageEditSourcePairTrainingError(
                    f"reference {reference_id} confirmed switch has no prior source"
                )
            _all_candidates, eligible = _eligible_candidates_at_token(
                token,
                schema=schema,
                reference_id=reference_id,
                token_index=token_index,
            )
            outgoing = eligible.get(prior_source_id)
            if outgoing is None:
                raise StageEditSourcePairTrainingError(
                    f"reference {reference_id} switch is missing eligible prior/outgoing source"
                )
            if selected_source_id not in eligible:
                raise StageEditSourcePairTrainingError(
                    f"reference {reference_id} chosen source is not eligible"
                )
            if selected_source_id == prior_source_id:
                raise StageEditSourcePairTrainingError(
                    f"reference {reference_id} confirmed source event does not switch"
                )
            # The decoder rejects a consecutive reuse of the outgoing exact
            # source before continuity scoring.  Keeping that impossible
            # candidate in the rank group would make both training and the
            # activation metrics look artificially easy, so supervision must
            # contain only candidates that can actually win this switch.
            incoming_eligible = {
                source_id: observation
                for source_id, observation in eligible.items()
                if source_id != prior_source_id
            }
            if len(incoming_eligible) < 2:
                raise StageEditSourcePairTrainingError(
                    f"reference {reference_id} switch has fewer than two viable "
                    "incoming candidates"
                )
            group_id = f"{reference_id}:source-pair:{token_index:06d}"
            group = StageEditSourcePairTrainingGroup(
                group_id=group_id,
                reference_id=reference_id,
                outgoing_source_id=prior_source_id,
                outgoing=outgoing,
                candidates=tuple(
                    StageEditSourcePairCandidate(
                        source_id, incoming_eligible[source_id]
                    )
                    for source_id in sorted(incoming_eligible)
                ),
                chosen_incoming_source_id=selected_source_id,
            )
            groups.append(group)
            group_song_ids.append((group_id, song_id))
            per_song[song_id] = per_song.get(song_id, 0) + 1
            per_reference[reference_id] = per_reference.get(reference_id, 0) + 1
            prior_source_id = selected_source_id

    if not groups:
        raise StageEditSourcePairTrainingError(
            "reference corpus has no confirmed non-opening source-pair switches"
        )
    ordered = tuple(
        sorted(groups, key=lambda group: (group.reference_id, group.group_id))
    )
    song_by_group = dict(group_song_ids)
    ordered_group_song_ids = tuple(
        (group.group_id, song_by_group[group.group_id]) for group in ordered
    )
    payload = {
        "schema_version": STAGE_EDIT_SOURCE_PAIR_TRAINING_SCHEMA_VERSION,
        "contract": STAGE_EDIT_SOURCE_PAIR_TRAINING_CONTRACT,
        "source_data_id": source_data_id,
        "token_schema_fingerprint": schema.fingerprint,
        "opening_selection_excluded_count": opening_excluded,
        "unconfirmed_selection_excluded_count": unconfirmed_excluded,
        "groups": [
            _group_payload(group, song_id=song_by_group[group.group_id])
            for group in ordered
        ],
    }
    return StageEditSourcePairTrainingCorpus(
        groups=ordered,
        source_data_id=source_data_id,
        token_schema_fingerprint=schema.fingerprint,
        group_song_ids=ordered_group_song_ids,
        song_ids=tuple(sorted(per_song)),
        reference_ids=tuple(sorted(per_reference)),
        per_song_group_counts=tuple(sorted(per_song.items())),
        per_reference_group_counts=tuple(sorted(per_reference.items())),
        input_reference_count=len(examples),
        opening_selection_excluded_count=opening_excluded,
        unconfirmed_selection_excluded_count=unconfirmed_excluded,
        content_sha256=_canonical_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class StageEditSourcePairRankingMetrics:
    scorer_name: str
    reference_count: int
    group_count: int
    reference_balanced_top1_accuracy: float
    reference_balanced_pairwise_accuracy: float
    reference_balanced_mean_reciprocal_rank: float
    reference_balanced_mean_chosen_margin: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "scorer_name", _name(self.scorer_name, "scorer_name"))
        for field in ("reference_count", "group_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise StageEditSourcePairTrainingError(f"{field} must be positive")
        for field in (
            "reference_balanced_top1_accuracy",
            "reference_balanced_pairwise_accuracy",
            "reference_balanced_mean_reciprocal_rank",
        ):
            value = _finite(getattr(self, field), field)
            if not 0.0 <= value <= 1.0:
                raise StageEditSourcePairTrainingError(f"{field} must lie in [0, 1]")
            object.__setattr__(self, field, value)
        object.__setattr__(
            self,
            "reference_balanced_mean_chosen_margin",
            _finite(
                self.reference_balanced_mean_chosen_margin,
                "reference_balanced_mean_chosen_margin",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scorer_name": self.scorer_name,
            "reference_count": self.reference_count,
            "group_count": self.group_count,
            "reference_balanced_top1_accuracy": (
                self.reference_balanced_top1_accuracy
            ),
            "reference_balanced_pairwise_accuracy": (
                self.reference_balanced_pairwise_accuracy
            ),
            "reference_balanced_mean_reciprocal_rank": (
                self.reference_balanced_mean_reciprocal_rank
            ),
            "reference_balanced_mean_chosen_margin": (
                self.reference_balanced_mean_chosen_margin
            ),
        }


def _handcrafted_pair_score(
    outgoing: StageEditSourcePairObservation,
    incoming: StageEditSourcePairObservation,
) -> float:
    """Exact local snapshot of the decoder's current handcrafted score."""

    weighted: list[tuple[float, float]] = []
    if outgoing.face_scale is not None and incoming.face_scale is not None:
        left = max(abs(outgoing.face_scale), 1.0e-6)
        right = max(abs(incoming.face_scale), 1.0e-6)
        weighted.append(
            (0.28, min(1.0, abs(math.log(right / left)) / math.log(1.50)))
        )
    face = (
        outgoing.face_center_x,
        outgoing.face_center_y,
        incoming.face_center_x,
        incoming.face_center_y,
    )
    body = (
        outgoing.body_center_x,
        outgoing.body_center_y,
        incoming.body_center_x,
        incoming.body_center_y,
    )
    if all(value is not None for value in face):
        weighted.append(
            (
                0.28,
                min(
                    1.0,
                    math.hypot(
                        float(incoming.face_center_x) - float(outgoing.face_center_x),
                        float(incoming.face_center_y) - float(outgoing.face_center_y),
                    )
                    / 0.22,
                ),
            )
        )
    elif all(value is not None for value in body):
        weighted.append(
            (
                0.18,
                min(
                    1.0,
                    math.hypot(
                        float(incoming.body_center_x) - float(outgoing.body_center_x),
                        float(incoming.body_center_y) - float(outgoing.body_center_y),
                    )
                    / 0.28,
                ),
            )
        )

    def scalar(weight: float, field: str, scale: float) -> None:
        left = getattr(outgoing, field)
        right = getattr(incoming, field)
        if left is not None and right is not None:
            weighted.append((weight, min(1.0, abs(right - left) / scale)))

    motion = (
        outgoing.camera_motion_x,
        outgoing.camera_motion_y,
        incoming.camera_motion_x,
        incoming.camera_motion_y,
    )
    if all(value is not None for value in motion):
        weighted.append(
            (
                0.12,
                min(
                    1.0,
                    math.hypot(
                        float(incoming.camera_motion_x)
                        - float(outgoing.camera_motion_x),
                        float(incoming.camera_motion_y)
                        - float(outgoing.camera_motion_y),
                    )
                    / 0.75,
                ),
            )
        )
    scalar(0.06, "pose_motion", 0.40)
    scalar(0.05, "camera_zoom_velocity", 0.35)
    scalar(0.05, "camera_roll_normalized", 0.25)
    scalar(0.06, "exposure", 0.25)
    scalar(0.04, "color_temperature_normalized", 0.30)
    scalar(0.03, "sharpness", 0.35)
    observed_weight = sum(weight for weight, _value in weighted)
    observed_penalty = (
        sum(weight * value for weight, value in weighted) / observed_weight
        if observed_weight
        else 0.0
    )
    coverage = min(1.0, observed_weight / 0.97)
    penalty = coverage * observed_penalty + (1.0 - coverage) * 0.12
    return -min(1.0, max(0.0, penalty))


def _reference_balanced_mean(
    values: Mapping[str, Sequence[float]],
) -> float:
    return float(
        np.mean(
            [
                float(np.mean(tuple(values[reference_id])))
                for reference_id in sorted(values)
            ]
        )
    )


def evaluate_stage_edit_source_pair_ranking(
    groups: Sequence[StageEditSourcePairTrainingGroup],
    *,
    scorer: StageEditSourcePairScorer | None,
    scorer_name: str,
    handcrafted_baseline: bool = False,
) -> StageEditSourcePairRankingMetrics:
    """Evaluate rankings with equal weight per reference, not per event."""

    values = tuple(groups)
    if not values:
        raise StageEditSourcePairTrainingError("ranking evaluation needs groups")
    if handcrafted_baseline == (scorer is not None):
        raise StageEditSourcePairTrainingError(
            "ranking evaluation needs exactly one scorer mode"
        )
    top1: dict[str, list[float]] = {}
    pairwise: dict[str, list[float]] = {}
    reciprocal: dict[str, list[float]] = {}
    margins: dict[str, list[float]] = {}
    tolerance = 1.0e-12
    for group in values:
        candidates = tuple(sorted(group.candidates, key=lambda item: item.source_id))
        score_by_source: dict[str, float] = {}
        for candidate in candidates:
            if handcrafted_baseline:
                if not isinstance(
                    group.outgoing, StageEditSourcePairObservation
                ) or not isinstance(
                    candidate.incoming, StageEditSourcePairObservation
                ):
                    raise StageEditSourcePairTrainingError(
                        "handcrafted evaluation needs typed pair observations"
                    )
                score = _handcrafted_pair_score(
                    group.outgoing, candidate.incoming
                )
            else:
                assert scorer is not None
                score = scorer.score(group.outgoing, candidate.incoming)
            score_by_source[candidate.source_id] = _finite(
                score, f"{scorer_name} candidate score"
            )
        chosen = score_by_source[group.chosen_incoming_source_id]
        alternatives = tuple(
            score
            for source_id, score in score_by_source.items()
            if source_id != group.chosen_incoming_source_id
        )
        best = max(score_by_source.values())
        tied_best = sum(abs(score - best) <= tolerance for score in score_by_source.values())
        top_credit = 1.0 / tied_best if abs(chosen - best) <= tolerance else 0.0
        better = sum(score > chosen + tolerance for score in alternatives)
        tied = sum(abs(score - chosen) <= tolerance for score in alternatives)
        average_rank = 1.0 + better + 0.5 * tied
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
        reciprocal.setdefault(reference_id, []).append(1.0 / average_rank)
        margins.setdefault(reference_id, []).append(chosen - max(alternatives))
    return StageEditSourcePairRankingMetrics(
        scorer_name=scorer_name,
        reference_count=len(top1),
        group_count=len(values),
        reference_balanced_top1_accuracy=_reference_balanced_mean(top1),
        reference_balanced_pairwise_accuracy=_reference_balanced_mean(pairwise),
        reference_balanced_mean_reciprocal_rank=_reference_balanced_mean(reciprocal),
        reference_balanced_mean_chosen_margin=_reference_balanced_mean(margins),
    )


@dataclass(frozen=True, slots=True)
class StageEditSourcePairActivationGate:
    authorized: bool
    reasons: tuple[str, ...]
    contract: str = STAGE_EDIT_SOURCE_PAIR_ACTIVATION_GATE_CONTRACT

    def __post_init__(self) -> None:
        if not isinstance(self.authorized, bool):
            raise StageEditSourcePairTrainingError("activation authorization must be bool")
        if self.contract != STAGE_EDIT_SOURCE_PAIR_ACTIVATION_GATE_CONTRACT:
            raise StageEditSourcePairTrainingError("activation gate contract mismatch")
        reasons = tuple(_name(value, "activation reason") for value in self.reasons)
        if self.authorized == bool(reasons):
            raise StageEditSourcePairTrainingError(
                "authorized gate must have no reasons; rejected gate needs reasons"
            )
        object.__setattr__(self, "reasons", reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "authorized": self.authorized,
            "reasons": list(self.reasons),
        }


def stage_edit_source_pair_no_worse_activation_gate(
    learned: StageEditSourcePairRankingMetrics,
    uniform: StageEditSourcePairRankingMetrics,
    handcrafted: StageEditSourcePairRankingMetrics,
    *,
    tolerance: float = 1.0e-12,
) -> StageEditSourcePairActivationGate:
    """Authorize only when learned rankings are no worse on every metric."""

    tolerance_value = _finite(tolerance, "activation tolerance")
    if tolerance_value < 0.0:
        raise StageEditSourcePairTrainingError(
            "activation tolerance must be non-negative"
        )
    identities = {
        (value.reference_count, value.group_count)
        for value in (learned, uniform, handcrafted)
    }
    if len(identities) != 1:
        raise StageEditSourcePairTrainingError(
            "activation metrics must describe the same reference/group corpus"
        )
    reasons = []
    metrics = (
        "reference_balanced_top1_accuracy",
        "reference_balanced_pairwise_accuracy",
        "reference_balanced_mean_reciprocal_rank",
    )
    for metric in metrics:
        learned_value = getattr(learned, metric)
        required = max(getattr(uniform, metric), getattr(handcrafted, metric))
        if learned_value + tolerance_value < required:
            reasons.append(f"learned_{metric}_worse_than_best_baseline")
    return StageEditSourcePairActivationGate(
        authorized=not reasons,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class StageEditSourcePairReferenceFitResult:
    corpus: StageEditSourcePairTrainingCorpus
    training: StageEditSourcePairTrainingResult
    learned_metrics: StageEditSourcePairRankingMetrics
    uniform_metrics: StageEditSourcePairRankingMetrics
    handcrafted_metrics: StageEditSourcePairRankingMetrics
    activation_gate: StageEditSourcePairActivationGate

    @property
    def activation_authorized(self) -> bool:
        return self.activation_gate.authorized

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus.to_dict(),
            "training": {
                "training_steps": self.training.training_steps,
                "reference_count": self.training.reference_count,
                "group_count": self.training.group_count,
                "initial_loss": self.training.initial_loss,
                "final_loss": self.training.final_loss,
                "margin": self.training.margin,
                "corpus_sha256": self.training.corpus_sha256,
                "scorer": self.training.scorer.to_dict(),
            },
            "ranking": {
                "learned": self.learned_metrics.to_dict(),
                "uniform": self.uniform_metrics.to_dict(),
                "handcrafted": self.handcrafted_metrics.to_dict(),
            },
            "activation_gate": self.activation_gate.to_dict(),
        }


def fit_stage_edit_source_pair_scorer_from_reference_corpus(
    corpus: ReferenceTrainingCorpus,
    *,
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
    steps: int = 250,
    learning_rate: float = 0.25,
    margin: float = 0.05,
) -> StageEditSourcePairReferenceFitResult:
    """Materialize, fit, benchmark, and gate one deterministic pair scorer."""

    materialized = materialize_stage_edit_source_pair_training_corpus(
        corpus, schema=schema
    )
    training = train_stage_edit_source_pair_scorer(
        materialized.groups,
        steps=steps,
        learning_rate=learning_rate,
        margin=margin,
    )
    learned = evaluate_stage_edit_source_pair_ranking(
        materialized.groups,
        scorer=training.scorer,
        scorer_name="learned",
    )
    uniform = evaluate_stage_edit_source_pair_ranking(
        materialized.groups,
        scorer=StageEditSourcePairScorer.uniform(),
        scorer_name="uniform",
    )
    handcrafted = evaluate_stage_edit_source_pair_ranking(
        materialized.groups,
        scorer=None,
        scorer_name=STAGE_EDIT_SOURCE_PAIR_HANDCRAFTED_BASELINE_CONTRACT,
        handcrafted_baseline=True,
    )
    return StageEditSourcePairReferenceFitResult(
        corpus=materialized,
        training=training,
        learned_metrics=learned,
        uniform_metrics=uniform,
        handcrafted_metrics=handcrafted,
        activation_gate=stage_edit_source_pair_no_worse_activation_gate(
            learned, uniform, handcrafted
        ),
    )


__all__ = [
    "STAGE_EDIT_SOURCE_PAIR_ACTIVATION_GATE_CONTRACT",
    "STAGE_EDIT_SOURCE_PAIR_HANDCRAFTED_BASELINE_CONTRACT",
    "STAGE_EDIT_SOURCE_PAIR_TRAINING_CONTRACT",
    "STAGE_EDIT_SOURCE_PAIR_TRAINING_SCHEMA_VERSION",
    "StageEditSourcePairActivationGate",
    "StageEditSourcePairObservation",
    "StageEditSourcePairRankingMetrics",
    "StageEditSourcePairReferenceFitResult",
    "StageEditSourcePairTrainingCorpus",
    "StageEditSourcePairTrainingError",
    "evaluate_stage_edit_source_pair_ranking",
    "fit_stage_edit_source_pair_scorer_from_reference_corpus",
    "materialize_stage_edit_source_pair_training_corpus",
    "stage_edit_source_pair_no_worse_activation_gate",
    "stage_edit_source_pair_observation_from_candidate",
]
