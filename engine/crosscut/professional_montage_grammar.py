"""Fail-closed visual montage grammar learned from IVE human references.

This module measures *what the reference editors did* without inventing a
close-up quota.  Every labelled reference shot is joined to the exact
``cut_master_time`` token in an immutable StageEdit feature snapshot.  The
selected source's inference-safe ``shot_scale`` column and its missing bit are
then used to measure wide/medium/close framing by both honest shot counts and
observed duration, explicit non-wide duration, consecutive wide holds,
source-switch shot-size pairs, wide establishing anchors, and data-relative
short non-wide bursts.

The result is a multi-song profile whose lower floors are quantiles of the
reference-edit distribution.  It contains no hand-authored quality pass
thresholds and no reference pixels, embeddings, or attribution scores.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .stage_edit_feature_materializer import (
    FEATURE_MATERIALIZER_CONTRACT,
    FEATURE_MATERIALIZER_SCHEMA_VERSION,
    STAGE_EDIT_CAUSAL_SENSING_BINDING_CONTRACT,
    STAGE_EDIT_CAUSAL_SENSING_MERGE_CONTRACT,
    StageEditCausalSensingBinding,
    StageEditFeatureSnapshot,
    validate_production_causal_sensing_gate_payload,
)
from .stage_edit_feature_snapshot_job import (
    FEATURE_SNAPSHOT_JOB_ANALYSIS_BASIS,
    FEATURE_SNAPSHOT_JOB_INPUT_POLICY,
    FEATURE_SNAPSHOT_JOB_PROVENANCE_CONTRACTS,
    FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION,
)
from .stage_edit_transformer import DEFAULT_STAGE_EDIT_SCHEMA


PROFILE_SCHEMA_VERSION = 2
PROFILE_ANALYSIS_BASIS = "ive_human_reference_visual_montage_grammar_v2"
MODULE_VERSION = "2"
FEATURE_JOB_SCHEMA_VERSION = FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION
FEATURE_JOB_ANALYSIS_BASIS = FEATURE_SNAPSHOT_JOB_ANALYSIS_BASIS
MATERIALIZATION_SCHEMA_VERSION = 1
MATERIALIZATION_ANALYSIS_BASIS = "immutable_reference_training_materialization_v1"
REFERENCE_SUPERVISION_SCHEMA_VERSION = 3
REFERENCE_SUPERVISION_KIND = "reference_stage_mix_pairwise_listwise_supervision"
ROLE_SEMANTICS = "acoustic_editing_roles_not_ground_truth_lyrical_sections"
SHOT_SCALE_FEATURE_NAME = "shot_scale"
WIDE_MAX_EXCLUSIVE = 0.22
MEDIUM_MAX_EXCLUSIVE = 0.52
BOUNDARY_GRAMMAR_SCHEMA_VERSION = 1
BOUNDARY_GRAMMAR_ANALYSIS_BASIS = (
    "ive_professional_cross_source_boundary_grammar_v1"
)
HIGHLIGHT_BURST_EXEMPLAR_CUTS_PER_MINUTE = 60.0
APPLIED_TRANSFORM_SEGMENT_COUNT_BASIS = (
    "segment_with_exact_zoom_pan_or_roll_onset_or_active_value_change_v1"
)

ROLE_NAMES = (
    "intro",
    "verse_like",
    "build",
    "chorus_like",
    "bridge_like",
    "outro",
)
SHOT_SIZE_NAMES = ("wide", "medium", "close")
PAIR_NAMES = tuple(
    f"{outgoing}_to_{incoming}"
    for outgoing in SHOT_SIZE_NAMES
    for incoming in SHOT_SIZE_NAMES
)

_SHA256_LENGTH = 64
_FEATURE_MANIFEST_KEYS = {
    "schema_version",
    "analysis_basis",
    "song_id",
    "input_policy",
    "sampling",
    "source_catalog_sha256",
    "source_count",
    "source_content",
    "supervision_inputs",
    "feature_provenance",
    "causal_sensing_binding",
    "production_causal_sensing_gate",
    "snapshots",
    "statistics",
    "manifest_sha256",
}
_MATERIALIZATION_KEYS = {
    "schema_version",
    "analysis_basis",
    "created_at",
    "input_fingerprint_sha256",
    "input_contract",
    "outputs",
    "statistics",
    "immutable_inputs_modified",
}
_SUPERVISION_KEYS = {
    "schema_version",
    "dataset_kind",
    "style_id",
    "reference_id",
    "reference_path",
    "source_catalog",
    "base_reference_dataset_schema_version",
    "canonical_shot_schema",
    "dense_attribution_schema",
    "crop_supervision_schema",
    "label_policy",
    "attribution_analysis",
    "reference_evidence_provenance",
    "shot_labels",
    "detected_boundary_records",
    "detected_boundary_statistics",
    "pairwise_records",
    "listwise_records",
    "statistics",
}


class MontageGrammarEvidenceError(ValueError):
    """Raised when immutable visual-grammar evidence is inconsistent."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MontageGrammarEvidenceError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise MontageGrammarEvidenceError(f"{name} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise MontageGrammarEvidenceError(
            f"{name} schema mismatch; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MontageGrammarEvidenceError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise MontageGrammarEvidenceError(f"{name} must be finite")
    return number


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MontageGrammarEvidenceError(f"{name} must be an integer")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stable_file_bytes(path: Path, name: str) -> bytes:
    source = Path(path).expanduser()
    if not source.is_absolute():
        raise MontageGrammarEvidenceError(f"{name} path must be absolute")
    if source.is_symlink() or not source.is_file():
        raise MontageGrammarEvidenceError(f"{name} must be a regular file: {source}")
    before = source.stat()
    raw = source.read_bytes()
    after = source.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(raw) != before.st_size:
        raise MontageGrammarEvidenceError(f"{name} changed while it was read")
    return raw


def _read_json(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    raw = _stable_file_bytes(path, name)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MontageGrammarEvidenceError(f"could not decode {name}: {exc}") from exc
    return dict(_mapping(value, name)), raw


def _verified_json(
    path: Path,
    expected_sha256: object,
    name: str,
) -> tuple[dict[str, Any], bytes]:
    if not _is_sha256(expected_sha256):
        raise MontageGrammarEvidenceError(f"{name} expected SHA-256 is invalid")
    payload, raw = _read_json(path, name)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise MontageGrammarEvidenceError(f"{name} file SHA-256 mismatch")
    return payload, raw


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    clean = sorted(_finite(value, "distribution value") for value in values)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = fraction * (len(clean) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def _reference_distribution(
    values: Sequence[float],
    *,
    floor_quantile: float,
) -> dict[str, Any]:
    clean = tuple(_finite(value, "reference distribution value") for value in values)
    floor = _quantile(clean, floor_quantile)
    return {
        "count": len(clean),
        "minimum": min(clean) if clean else None,
        "p10": _quantile(clean, 0.10),
        "p25": _quantile(clean, 0.25),
        "median": median(clean) if clean else None,
        "p75": _quantile(clean, 0.75),
        "p90": _quantile(clean, 0.90),
        "maximum": max(clean) if clean else None,
        "reference_floor": {
            "quantile": floor_quantile,
            "value": floor,
            "basis": "lower_quantile_of_contributing_human_reference_edits",
        },
    }


def _event_distribution(values: Sequence[float]) -> dict[str, Any]:
    clean = tuple(_finite(value, "event distribution value") for value in values)
    return {
        "count": len(clean),
        "minimum": min(clean) if clean else None,
        "p10": _quantile(clean, 0.10),
        "p25": _quantile(clean, 0.25),
        "median": median(clean) if clean else None,
        "p75": _quantile(clean, 0.75),
        "p90": _quantile(clean, 0.90),
        "maximum": max(clean) if clean else None,
    }


@dataclass(frozen=True, slots=True)
class MontageGrammarConfig:
    """Evidence-coverage policy; no visual-style quota is configured here."""

    minimum_song_count: int = 3
    minimum_references_per_song: int = 3
    minimum_sources_per_song: int = 8
    early_anchor_shot_count: int = 2
    reference_floor_quantile: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "minimum_song_count",
            "minimum_references_per_song",
            "minimum_sources_per_song",
            "early_anchor_shot_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        quantile = _finite(
            self.reference_floor_quantile,
            "reference_floor_quantile",
        )
        if not 0.0 < quantile <= 0.25:
            raise ValueError("reference_floor_quantile must lie in (0, 0.25]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_song_count": self.minimum_song_count,
            "minimum_references_per_song": self.minimum_references_per_song,
            "minimum_sources_per_song": self.minimum_sources_per_song,
            "early_anchor_shot_count": self.early_anchor_shot_count,
            "reference_floor_quantile": self.reference_floor_quantile,
            "short_shot_basis": (
                "duration_at_or_below_each_reference_edits_observed_median"
            ),
            "burst_connectivity": (
                "contiguous_known_shots_with_cross_source_switches"
            ),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MontageGrammarConfig":
        _exact_keys(
            value,
            {
                "minimum_song_count",
                "minimum_references_per_song",
                "minimum_sources_per_song",
                "early_anchor_shot_count",
                "reference_floor_quantile",
                "short_shot_basis",
                "burst_connectivity",
            },
            "montage grammar configuration",
        )
        if value["short_shot_basis"] != (
            "duration_at_or_below_each_reference_edits_observed_median"
        ):
            raise MontageGrammarEvidenceError("unsupported short-shot basis")
        if value["burst_connectivity"] != (
            "contiguous_known_shots_with_cross_source_switches"
        ):
            raise MontageGrammarEvidenceError("unsupported burst connectivity")
        return cls(
            minimum_song_count=_integer(
                value["minimum_song_count"], "minimum_song_count"
            ),
            minimum_references_per_song=_integer(
                value["minimum_references_per_song"],
                "minimum_references_per_song",
            ),
            minimum_sources_per_song=_integer(
                value["minimum_sources_per_song"], "minimum_sources_per_song"
            ),
            early_anchor_shot_count=_integer(
                value["early_anchor_shot_count"], "early_anchor_shot_count"
            ),
            reference_floor_quantile=_finite(
                value["reference_floor_quantile"],
                "reference_floor_quantile",
            ),
        )


_SECTION_KINDS = ("general", "chorus", "highlight")


def _strict_seconds(
    values: Sequence[float],
    *,
    duration_seconds: float,
    name: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise MontageGrammarEvidenceError(f"{name} must be an array")
    seconds = tuple(_finite(item, f"{name} value") for item in values)
    if tuple(sorted(seconds)) != seconds or len(set(seconds)) != len(seconds):
        raise MontageGrammarEvidenceError(
            f"{name} must be strictly increasing and unique"
        )
    if any(not 0.0 < item < duration_seconds for item in seconds):
        raise MontageGrammarEvidenceError(
            f"{name} values must lie strictly inside the performance span"
        )
    return seconds


def _strict_transform_event_seconds(
    values: Sequence[float],
    *,
    duration_seconds: float,
    name: str,
) -> tuple[float, ...]:
    """Validate exact transform-onset clocks on the output timeline.

    Unlike cut clocks, an editorial transform may already be active on the
    first rendered frame.  The performance endpoint is still excluded because
    no output frame exists after it.  Multiple onsets may belong to one
    Dynamic Crop trajectory, so this validator intentionally does not collapse
    or cap events by transformed-segment count; the typed professional policy
    applies the bounded whole-plan density limits later.
    """

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise MontageGrammarEvidenceError(f"{name} must be an array")
    seconds = tuple(_finite(item, f"{name} value") for item in values)
    if tuple(sorted(seconds)) != seconds or len(set(seconds)) != len(seconds):
        raise MontageGrammarEvidenceError(
            f"{name} must be strictly increasing and unique"
        )
    if any(not 0.0 <= item < duration_seconds for item in seconds):
        raise MontageGrammarEvidenceError(
            f"{name} values must lie in [0, performance duration)"
        )
    return seconds


@dataclass(frozen=True, slots=True)
class MontageSectionWindow:
    """One non-overlapping acoustic/editing section on the output clock."""

    section_id: str
    kind: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        section_id = str(self.section_id).strip()
        if not section_id or any(character.isspace() for character in section_id):
            raise MontageGrammarEvidenceError(
                "section_id must be non-blank and contain no whitespace"
            )
        kind = str(self.kind).strip()
        if kind not in _SECTION_KINDS:
            raise MontageGrammarEvidenceError(
                f"section kind must be one of {_SECTION_KINDS}"
            )
        start = _finite(self.start_seconds, "section start_seconds")
        end = _finite(self.end_seconds, "section end_seconds")
        if start < 0.0 or end <= start:
            raise MontageGrammarEvidenceError("section span must be positive")
        object.__setattr__(self, "section_id", section_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "kind": self.kind,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MontageSectionWindow":
        _exact_keys(
            value,
            {"section_id", "kind", "start_seconds", "end_seconds"},
            "montage section window",
        )
        return cls(
            section_id=str(value["section_id"]),
            kind=str(value["kind"]),
            start_seconds=_finite(value["start_seconds"], "section start_seconds"),
            end_seconds=_finite(value["end_seconds"], "section end_seconds"),
        )


@dataclass(frozen=True, slots=True)
class IntroducedBoundaryContinuityEvidence:
    """Decoded review evidence for newly introduced cross-source boundaries."""

    boundary_count: int
    issue_count: int
    issue_fraction: float
    score_p10: float | None

    def __post_init__(self) -> None:
        boundary_count = _integer(self.boundary_count, "introduced boundary_count")
        issue_count = _integer(self.issue_count, "introduced issue_count")
        issue_fraction = _finite(
            self.issue_fraction,
            "introduced issue_fraction",
        )
        if boundary_count < 0 or not 0 <= issue_count <= boundary_count:
            raise MontageGrammarEvidenceError(
                "introduced boundary issue counts are inconsistent"
            )
        expected_fraction = (
            issue_count / boundary_count if boundary_count else 0.0
        )
        if not math.isclose(
            issue_fraction,
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise MontageGrammarEvidenceError(
                "introduced issue_fraction does not match the exact counts"
            )
        if boundary_count:
            if self.score_p10 is None:
                if issue_count != boundary_count:
                    raise MontageGrammarEvidenceError(
                        "unobserved introduced scores must mark every boundary as an issue"
                    )
                score_p10 = None
            else:
                score_p10 = _finite(self.score_p10, "introduced score_p10")
            if score_p10 is not None and not 0.0 <= score_p10 <= 100.0:
                raise MontageGrammarEvidenceError(
                    "introduced score_p10 must lie in [0, 100]"
                )
        else:
            if self.score_p10 is not None:
                raise MontageGrammarEvidenceError(
                    "introduced score_p10 must be null when no boundaries exist"
                )
            score_p10 = None
        object.__setattr__(self, "boundary_count", boundary_count)
        object.__setattr__(self, "issue_count", issue_count)
        object.__setattr__(self, "issue_fraction", issue_fraction)
        object.__setattr__(self, "score_p10", score_p10)

    @classmethod
    def from_counts(
        cls,
        *,
        boundary_count: int,
        issue_count: int,
        score_p10: float | None,
    ) -> "IntroducedBoundaryContinuityEvidence":
        fraction = issue_count / boundary_count if boundary_count else 0.0
        return cls(boundary_count, issue_count, fraction, score_p10)

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_count": self.boundary_count,
            "issue_count": self.issue_count,
            "issue_fraction": self.issue_fraction,
            "score_p10": self.score_p10,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "IntroducedBoundaryContinuityEvidence":
        _exact_keys(
            value,
            {"boundary_count", "issue_count", "issue_fraction", "score_p10"},
            "introduced-boundary continuity evidence",
        )
        return cls(
            boundary_count=_integer(value["boundary_count"], "boundary_count"),
            issue_count=_integer(value["issue_count"], "issue_count"),
            issue_fraction=_finite(value["issue_fraction"], "issue_fraction"),
            score_p10=(
                None
                if value["score_p10"] is None
                else _finite(value["score_p10"], "score_p10")
            ),
        )


@dataclass(frozen=True, slots=True)
class EditorialTransformEvidence:
    """Finite render-plan transform evidence with exact visible event clocks.

    ``applied_segment_count`` counts segments containing at least one visible
    zoom, pan, or roll onset/value change.  Merely attaching a constant learned
    framing track to every segment does not turn every shot into an editorial
    motion event.
    """

    segment_count: int
    applied_segment_count: int
    zoom_event_times_seconds: tuple[float, ...] = ()
    pan_event_times_seconds: tuple[float, ...] = ()
    roll_event_times_seconds: tuple[float, ...] = ()

    def validated(self, *, duration_seconds: float) -> "EditorialTransformEvidence":
        segment_count = _integer(self.segment_count, "transform segment_count")
        applied = _integer(
            self.applied_segment_count,
            "transform applied_segment_count",
        )
        if segment_count < 1 or not 0 <= applied <= segment_count:
            raise MontageGrammarEvidenceError(
                "transform segment counts are inconsistent"
            )
        zoom = _strict_transform_event_seconds(
            self.zoom_event_times_seconds,
            duration_seconds=duration_seconds,
            name="zoom event times",
        )
        pan = _strict_transform_event_seconds(
            self.pan_event_times_seconds,
            duration_seconds=duration_seconds,
            name="pan event times",
        )
        roll = _strict_transform_event_seconds(
            self.roll_event_times_seconds,
            duration_seconds=duration_seconds,
            name="roll event times",
        )
        if not applied and any((zoom, pan, roll)):
            raise MontageGrammarEvidenceError(
                "transform events require an applied transform segment"
            )
        if applied > len(zoom) + len(pan) + len(roll):
            raise MontageGrammarEvidenceError(
                "each applied transform segment requires at least one exact event"
            )
        return EditorialTransformEvidence(
            segment_count=segment_count,
            applied_segment_count=applied,
            zoom_event_times_seconds=zoom,
            pan_event_times_seconds=pan,
            roll_event_times_seconds=roll,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_count": self.segment_count,
            "applied_segment_count": self.applied_segment_count,
            "applied_segment_count_basis": (
                APPLIED_TRANSFORM_SEGMENT_COUNT_BASIS
            ),
            "zoom_event_times_seconds": list(self.zoom_event_times_seconds),
            "pan_event_times_seconds": list(self.pan_event_times_seconds),
            "roll_event_times_seconds": list(self.roll_event_times_seconds),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EditorialTransformEvidence":
        _exact_keys(
            value,
            {
                "segment_count",
                "applied_segment_count",
                "applied_segment_count_basis",
                "zoom_event_times_seconds",
                "pan_event_times_seconds",
                "roll_event_times_seconds",
            },
            "editorial transform evidence",
        )
        if value["applied_segment_count_basis"] != (
            APPLIED_TRANSFORM_SEGMENT_COUNT_BASIS
        ):
            raise MontageGrammarEvidenceError(
                "editorial transform applied-segment count basis mismatch"
            )
        return cls(
            segment_count=_integer(value["segment_count"], "segment_count"),
            applied_segment_count=_integer(
                value["applied_segment_count"],
                "applied_segment_count",
            ),
            zoom_event_times_seconds=tuple(
                _finite(item, "zoom event time")
                for item in _array(
                    value["zoom_event_times_seconds"],
                    "zoom event times",
                )
            ),
            pan_event_times_seconds=tuple(
                _finite(item, "pan event time")
                for item in _array(
                    value["pan_event_times_seconds"],
                    "pan event times",
                )
            ),
            roll_event_times_seconds=tuple(
                _finite(item, "roll event time")
                for item in _array(
                    value["roll_event_times_seconds"],
                    "roll event times",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ProfessionalBoundaryGrammarEvidence:
    """Identity-free clocks and decoded QA needed for boundary assessment."""

    performance_id: str
    duration_seconds: float
    source_switch_times_seconds: tuple[float, ...]
    native_internal_cut_times_seconds: tuple[float, ...]
    sections: tuple[MontageSectionWindow, ...]
    introduced_boundary_continuity: IntroducedBoundaryContinuityEvidence
    editorial_transforms: EditorialTransformEvidence

    def __post_init__(self) -> None:
        performance_id = str(self.performance_id).strip()
        if not performance_id or any(
            character.isspace() for character in performance_id
        ):
            raise MontageGrammarEvidenceError(
                "performance_id must be non-blank and contain no whitespace"
            )
        duration = _finite(self.duration_seconds, "performance duration_seconds")
        if duration <= 0.0:
            raise MontageGrammarEvidenceError(
                "performance duration_seconds must be positive"
            )
        switches = _strict_seconds(
            self.source_switch_times_seconds,
            duration_seconds=duration,
            name="source-switch times",
        )
        native = _strict_seconds(
            self.native_internal_cut_times_seconds,
            duration_seconds=duration,
            name="native/internal cut times",
        )
        if set(switches).intersection(native):
            raise MontageGrammarEvidenceError(
                "source switches and native/internal cuts must be disjoint"
            )
        sections = tuple(self.sections)
        if not all(isinstance(item, MontageSectionWindow) for item in sections):
            raise MontageGrammarEvidenceError(
                "sections must contain MontageSectionWindow values"
            )
        if tuple(sorted(sections, key=lambda item: item.start_seconds)) != sections:
            raise MontageGrammarEvidenceError("sections must be ordered by start")
        if not sections:
            raise MontageGrammarEvidenceError(
                "sections must partition the complete performance span"
            )
        if len({item.section_id for item in sections}) != len(sections):
            raise MontageGrammarEvidenceError("section_id values must be unique")
        if not math.isclose(
            sections[0].start_seconds,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise MontageGrammarEvidenceError(
                "sections must begin at the performance start"
            )
        for index, section in enumerate(sections):
            if section.end_seconds > duration + 1e-12:
                raise MontageGrammarEvidenceError(
                    "section extends beyond the performance span"
                )
            if index and not math.isclose(
                sections[index - 1].end_seconds,
                section.start_seconds,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise MontageGrammarEvidenceError(
                    "sections must be contiguous and non-overlapping"
                )
        if not math.isclose(
            sections[-1].end_seconds,
            duration,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise MontageGrammarEvidenceError(
                "sections must end at the performance end"
            )
        continuity = self.introduced_boundary_continuity
        if not isinstance(continuity, IntroducedBoundaryContinuityEvidence):
            raise MontageGrammarEvidenceError(
                "introduced_boundary_continuity has the wrong type"
            )
        if continuity.boundary_count != len(switches):
            raise MontageGrammarEvidenceError(
                "introduced boundary_count must equal the source-switch count"
            )
        transforms = self.editorial_transforms
        if not isinstance(transforms, EditorialTransformEvidence):
            raise MontageGrammarEvidenceError(
                "editorial_transforms has the wrong type"
            )
        transforms = transforms.validated(duration_seconds=duration)
        object.__setattr__(self, "performance_id", performance_id)
        object.__setattr__(self, "duration_seconds", duration)
        object.__setattr__(self, "source_switch_times_seconds", switches)
        object.__setattr__(self, "native_internal_cut_times_seconds", native)
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "editorial_transforms", transforms)

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BOUNDARY_GRAMMAR_SCHEMA_VERSION,
            "analysis_basis": BOUNDARY_GRAMMAR_ANALYSIS_BASIS,
            "performance_id": self.performance_id,
            "duration_seconds": self.duration_seconds,
            "source_switch_times_seconds": list(self.source_switch_times_seconds),
            "native_internal_cut_times_seconds": list(
                self.native_internal_cut_times_seconds
            ),
            "sections": [item.to_dict() for item in self.sections],
            "introduced_boundary_continuity": (
                self.introduced_boundary_continuity.to_dict()
            ),
            "editorial_transforms": self.editorial_transforms.to_dict(),
        }

    @property
    def evidence_sha256(self) -> str:
        return _canonical_sha256(self._unsigned_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self._unsigned_dict()
        payload["evidence_sha256"] = self.evidence_sha256
        return payload

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ProfessionalBoundaryGrammarEvidence":
        expected = {
            "schema_version",
            "analysis_basis",
            "performance_id",
            "duration_seconds",
            "source_switch_times_seconds",
            "native_internal_cut_times_seconds",
            "sections",
            "introduced_boundary_continuity",
            "editorial_transforms",
            "evidence_sha256",
        }
        _exact_keys(value, expected, "professional boundary grammar evidence")
        if value["schema_version"] != BOUNDARY_GRAMMAR_SCHEMA_VERSION:
            raise MontageGrammarEvidenceError(
                "boundary grammar evidence schema version mismatch"
            )
        if value["analysis_basis"] != BOUNDARY_GRAMMAR_ANALYSIS_BASIS:
            raise MontageGrammarEvidenceError(
                "boundary grammar evidence analysis basis mismatch"
            )
        claimed_hash = value["evidence_sha256"]
        if not _is_sha256(claimed_hash):
            raise MontageGrammarEvidenceError(
                "boundary grammar evidence SHA-256 is invalid"
            )
        unsigned = dict(value)
        unsigned.pop("evidence_sha256")
        if _canonical_sha256(unsigned) != claimed_hash:
            raise MontageGrammarEvidenceError(
                "boundary grammar evidence content hash mismatch"
            )
        evidence = cls(
            performance_id=str(value["performance_id"]),
            duration_seconds=_finite(value["duration_seconds"], "duration_seconds"),
            source_switch_times_seconds=tuple(
                _finite(item, "source-switch time")
                for item in _array(
                    value["source_switch_times_seconds"],
                    "source-switch times",
                )
            ),
            native_internal_cut_times_seconds=tuple(
                _finite(item, "native/internal cut time")
                for item in _array(
                    value["native_internal_cut_times_seconds"],
                    "native/internal cut times",
                )
            ),
            sections=tuple(
                MontageSectionWindow.from_mapping(
                    _mapping(item, "montage section window")
                )
                for item in _array(value["sections"], "sections")
            ),
            introduced_boundary_continuity=(
                IntroducedBoundaryContinuityEvidence.from_mapping(
                    _mapping(
                        value["introduced_boundary_continuity"],
                        "introduced-boundary continuity evidence",
                    )
                )
            ),
            editorial_transforms=EditorialTransformEvidence.from_mapping(
                _mapping(value["editorial_transforms"], "editorial transforms")
            ),
        )
        if evidence.evidence_sha256 != claimed_hash:
            raise MontageGrammarEvidenceError(
                "boundary grammar evidence failed strict round-trip"
            )
        return evidence


@dataclass(frozen=True, slots=True)
class ProfessionalBoundaryGrammarPolicy:
    """Finite professional floors; the 60/min exemplar is not a general floor."""

    minimum_whole_song_source_switches_per_minute: float = 12.0
    minimum_chorus_source_switches_per_minute: float = 18.0
    minimum_highlight_source_switches_per_minute: float = 24.0
    maximum_chorus_no_switch_gap_seconds: float = 4.0
    maximum_highlight_no_switch_gap_seconds: float = 2.75
    maximum_introduced_boundary_issue_fraction: float = 0.25
    minimum_introduced_boundary_score_p10: float = 70.0
    minimum_editorial_transform_segment_fraction: float = 0.08
    maximum_editorial_transform_segment_fraction: float = 0.35
    minimum_zoom_pan_events_per_segment: float = 0.05
    maximum_zoom_pan_events_per_segment: float = 0.50
    maximum_roll_events_per_applied_segment: float = 0.20

    def __post_init__(self) -> None:
        positive = (
            "minimum_whole_song_source_switches_per_minute",
            "minimum_chorus_source_switches_per_minute",
            "minimum_highlight_source_switches_per_minute",
            "maximum_chorus_no_switch_gap_seconds",
            "maximum_highlight_no_switch_gap_seconds",
            "minimum_introduced_boundary_score_p10",
        )
        for name in positive:
            if _finite(getattr(self, name), name) <= 0.0:
                raise MontageGrammarEvidenceError(f"{name} must be positive")
        unit = (
            "maximum_introduced_boundary_issue_fraction",
            "minimum_editorial_transform_segment_fraction",
            "maximum_editorial_transform_segment_fraction",
            "minimum_zoom_pan_events_per_segment",
            "maximum_zoom_pan_events_per_segment",
            "maximum_roll_events_per_applied_segment",
        )
        for name in unit:
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise MontageGrammarEvidenceError(f"{name} must lie in [0, 1]")
        if self.minimum_introduced_boundary_score_p10 > 100.0:
            raise MontageGrammarEvidenceError(
                "minimum_introduced_boundary_score_p10 must lie in (0, 100]"
            )
        if (
            self.minimum_editorial_transform_segment_fraction
            > self.maximum_editorial_transform_segment_fraction
            or self.minimum_zoom_pan_events_per_segment
            > self.maximum_zoom_pan_events_per_segment
        ):
            raise MontageGrammarEvidenceError(
                "minimum transform coverage cannot exceed its maximum"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        } | {
            "roll_allowed_section_kinds": ["chorus", "highlight"],
            "highlight_burst_exemplar": {
                "reference": "I_AM_51_to_57_seconds",
                "cuts_per_minute": HIGHLIGHT_BURST_EXEMPLAR_CUTS_PER_MINUTE,
                "used_as_general_section_minimum": False,
                "used_as_whole_song_minimum": False,
            },
        }


@dataclass(frozen=True, slots=True)
class SectionBoundaryAssessment:
    section_id: str
    kind: str
    duration_seconds: float
    source_switch_count: int
    source_switches_per_minute: float
    maximum_no_switch_gap_seconds: float
    minimum_required_source_switches_per_minute: float | None
    maximum_allowed_no_switch_gap_seconds: float | None
    passed: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (
                list(value) if name == "failure_reasons" else value
            )
            for name, value in (
                (field, getattr(self, field))
                for field in self.__dataclass_fields__
            )
        }


@dataclass(frozen=True, slots=True)
class ProfessionalBoundaryGrammarAssessment:
    passed: bool
    failure_reasons: tuple[str, ...]
    source_switch_count: int
    source_switches_per_minute: float
    native_internal_cut_count: int
    native_internal_cuts_per_minute: float
    total_visible_cut_count: int
    total_visible_cuts_per_minute: float
    introduced_boundary_issue_fraction: float
    introduced_boundary_score_p10: float | None
    editorial_transform_segment_fraction: float
    zoom_pan_events_per_segment: float
    roll_events_per_applied_segment: float
    section_assessments: tuple[SectionBoundaryAssessment, ...]
    evidence_sha256: str
    policy: ProfessionalBoundaryGrammarPolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BOUNDARY_GRAMMAR_SCHEMA_VERSION,
            "analysis_basis": BOUNDARY_GRAMMAR_ANALYSIS_BASIS,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "cadence": {
                "source_switch_count": self.source_switch_count,
                "source_switches_per_minute": self.source_switches_per_minute,
                "native_internal_cut_count": self.native_internal_cut_count,
                "native_internal_cuts_per_minute": (
                    self.native_internal_cuts_per_minute
                ),
                "total_visible_cut_count": self.total_visible_cut_count,
                "total_visible_cuts_per_minute": self.total_visible_cuts_per_minute,
                "native_cuts_satisfy_source_switch_policy": False,
            },
            "introduced_boundary_continuity": {
                "issue_fraction": self.introduced_boundary_issue_fraction,
                "score_p10": self.introduced_boundary_score_p10,
            },
            "editorial_transforms": {
                "applied_segment_fraction": (
                    self.editorial_transform_segment_fraction
                ),
                "applied_segment_fraction_basis": (
                    APPLIED_TRANSFORM_SEGMENT_COUNT_BASIS
                ),
                "zoom_pan_events_per_segment": self.zoom_pan_events_per_segment,
                "roll_events_per_applied_segment": (
                    self.roll_events_per_applied_segment
                ),
                "roll_allowed_section_kinds": ["chorus", "highlight"],
            },
            "sections": [item.to_dict() for item in self.section_assessments],
            "evidence_sha256": self.evidence_sha256,
            "policy": self.policy.to_dict(),
        }


def _section_switch_times(
    switch_times: Sequence[float],
    section: MontageSectionWindow,
) -> tuple[float, ...]:
    return tuple(
        item
        for item in switch_times
        if section.start_seconds <= item < section.end_seconds
    )


def _maximum_section_gap(
    switch_times: Sequence[float],
    section: MontageSectionWindow,
) -> float:
    clocks = (section.start_seconds, *switch_times, section.end_seconds)
    return max(right - left for left, right in zip(clocks, clocks[1:]))


def _section_kind_at(
    seconds: float,
    sections: Sequence[MontageSectionWindow],
) -> str:
    for section in sections:
        if section.start_seconds <= seconds < section.end_seconds:
            return section.kind
    return "general"


def assess_professional_boundary_grammar(
    evidence: ProfessionalBoundaryGrammarEvidence,
    *,
    policy: ProfessionalBoundaryGrammarPolicy | None = None,
) -> ProfessionalBoundaryGrammarAssessment:
    """Fail closed on sparse/rough source changes without counting native cuts.

    The I AM 51--57 second 60 cuts/min observation calibrates the existence of
    a highlight burst. It is intentionally neither the whole-song nor the
    ordinary-section floor.
    """

    if not isinstance(evidence, ProfessionalBoundaryGrammarEvidence):
        raise MontageGrammarEvidenceError(
            "boundary grammar evidence must be typed and validated"
        )
    settings = policy or ProfessionalBoundaryGrammarPolicy()
    if not isinstance(settings, ProfessionalBoundaryGrammarPolicy):
        raise MontageGrammarEvidenceError(
            "boundary grammar policy must be typed and validated"
        )
    duration = evidence.duration_seconds
    source_count = len(evidence.source_switch_times_seconds)
    native_count = len(evidence.native_internal_cut_times_seconds)
    source_rate = source_count / duration * 60.0
    native_rate = native_count / duration * 60.0
    total_count = source_count + native_count
    total_rate = total_count / duration * 60.0
    failures: list[str] = []
    if source_rate + 1e-12 < (
        settings.minimum_whole_song_source_switches_per_minute
    ):
        failures.append("whole_song_source_switch_density_below_minimum")

    section_results: list[SectionBoundaryAssessment] = []
    for section in evidence.sections:
        switch_times = _section_switch_times(
            evidence.source_switch_times_seconds,
            section,
        )
        section_duration = section.end_seconds - section.start_seconds
        section_rate = len(switch_times) / section_duration * 60.0
        maximum_gap = _maximum_section_gap(switch_times, section)
        minimum_rate: float | None = None
        maximum_gap_allowed: float | None = None
        section_failures: list[str] = []
        if section.kind == "chorus":
            minimum_rate = settings.minimum_chorus_source_switches_per_minute
            maximum_gap_allowed = settings.maximum_chorus_no_switch_gap_seconds
        elif section.kind == "highlight":
            minimum_rate = settings.minimum_highlight_source_switches_per_minute
            maximum_gap_allowed = settings.maximum_highlight_no_switch_gap_seconds
        if minimum_rate is not None and section_rate + 1e-12 < minimum_rate:
            section_failures.append(
                f"{section.kind}_source_switch_density_below_minimum"
            )
        if (
            maximum_gap_allowed is not None
            and maximum_gap > maximum_gap_allowed + 1e-12
        ):
            section_failures.append(
                f"{section.kind}_no_switch_gap_above_maximum"
            )
        failures.extend(section_failures)
        section_results.append(
            SectionBoundaryAssessment(
                section_id=section.section_id,
                kind=section.kind,
                duration_seconds=section_duration,
                source_switch_count=len(switch_times),
                source_switches_per_minute=section_rate,
                maximum_no_switch_gap_seconds=maximum_gap,
                minimum_required_source_switches_per_minute=minimum_rate,
                maximum_allowed_no_switch_gap_seconds=maximum_gap_allowed,
                passed=not section_failures,
                failure_reasons=tuple(section_failures),
            )
        )

    continuity = evidence.introduced_boundary_continuity
    if continuity.boundary_count and continuity.score_p10 is None:
        failures.append("introduced_boundary_score_unobserved")
    if continuity.issue_fraction > (
        settings.maximum_introduced_boundary_issue_fraction + 1e-12
    ):
        failures.append("introduced_boundary_issue_fraction_above_maximum")
    if (
        continuity.boundary_count
        and continuity.score_p10 is not None
        and continuity.score_p10 + 1e-12
        < settings.minimum_introduced_boundary_score_p10
    ):
        failures.append("introduced_boundary_score_p10_below_minimum")

    transforms = evidence.editorial_transforms
    applied_fraction = (
        transforms.applied_segment_count / transforms.segment_count
    )
    zoom_pan_rate = (
        len(transforms.zoom_event_times_seconds)
        + len(transforms.pan_event_times_seconds)
    ) / transforms.segment_count
    roll_rate = (
        len(transforms.roll_event_times_seconds)
        / transforms.applied_segment_count
        if transforms.applied_segment_count
        else 0.0
    )
    if applied_fraction + 1e-12 < (
        settings.minimum_editorial_transform_segment_fraction
    ):
        failures.append("editorial_transform_coverage_below_minimum")
    if applied_fraction > (
        settings.maximum_editorial_transform_segment_fraction + 1e-12
    ):
        failures.append("editorial_transform_coverage_above_maximum")
    if zoom_pan_rate + 1e-12 < settings.minimum_zoom_pan_events_per_segment:
        failures.append("zoom_pan_coverage_below_minimum")
    if zoom_pan_rate > settings.maximum_zoom_pan_events_per_segment + 1e-12:
        failures.append("zoom_pan_coverage_above_maximum")
    if roll_rate > settings.maximum_roll_events_per_applied_segment + 1e-12:
        failures.append("roll_fraction_above_maximum")
    if any(
        _section_kind_at(seconds, evidence.sections)
        not in {"chorus", "highlight"}
        for seconds in transforms.roll_event_times_seconds
    ):
        failures.append("roll_outside_allowed_section")

    unique_failures = tuple(dict.fromkeys(failures))
    return ProfessionalBoundaryGrammarAssessment(
        passed=not unique_failures,
        failure_reasons=unique_failures,
        source_switch_count=source_count,
        source_switches_per_minute=source_rate,
        native_internal_cut_count=native_count,
        native_internal_cuts_per_minute=native_rate,
        total_visible_cut_count=total_count,
        total_visible_cuts_per_minute=total_rate,
        introduced_boundary_issue_fraction=continuity.issue_fraction,
        introduced_boundary_score_p10=continuity.score_p10,
        editorial_transform_segment_fraction=applied_fraction,
        zoom_pan_events_per_segment=zoom_pan_rate,
        roll_events_per_applied_segment=roll_rate,
        section_assessments=tuple(section_results),
        evidence_sha256=evidence.evidence_sha256,
        policy=settings,
    )


@dataclass(frozen=True, slots=True)
class MontageSongInput:
    song_id: str
    feature_manifest_path: Path
    materialization_manifest_path: Path

    def __post_init__(self) -> None:
        song_id = str(self.song_id).strip()
        if not song_id or any(character.isspace() for character in song_id):
            raise ValueError("song_id must be non-blank and contain no whitespace")
        feature = Path(self.feature_manifest_path).expanduser().resolve()
        materialization = Path(
            self.materialization_manifest_path
        ).expanduser().resolve()
        if not feature.is_file() or not materialization.is_file():
            raise FileNotFoundError("montage grammar manifests must exist")
        object.__setattr__(self, "song_id", song_id)
        object.__setattr__(self, "feature_manifest_path", feature)
        object.__setattr__(
            self,
            "materialization_manifest_path",
            materialization,
        )


@dataclass(frozen=True, slots=True)
class _Shot:
    shot_index: int
    start: float
    end: float
    duration: float
    source_id: str
    role: str
    section_progress: float
    phrase_progress: float
    size: str | None
    scale_unknown_reason: str | None
    detected_person_count_normalized: float | None
    formation_person_count_normalized: float | None


def _shot_size(scale: float) -> str:
    if scale < WIDE_MAX_EXCLUSIVE:
        return "wide"
    if scale < MEDIUM_MAX_EXCLUSIVE:
        return "medium"
    return "close"


def _role_and_progress(token: Any) -> tuple[str, float, float]:
    section_values = tuple(token.section_features)
    offset = len(DEFAULT_STAGE_EDIT_SCHEMA.music_feature_names)
    section_missing = tuple(token.missing_mask)[
        offset : offset + len(DEFAULT_STAGE_EDIT_SCHEMA.section_feature_names)
    ]
    if len(section_values) != len(DEFAULT_STAGE_EDIT_SCHEMA.section_feature_names):
        raise MontageGrammarEvidenceError("StageEdit section feature width mismatch")
    if any(section_missing[index] for index in range(7)):
        raise MontageGrammarEvidenceError(
            "acoustic role/progress is missing at a labelled shot clock"
        )
    positives = [
        index for index, value in enumerate(section_values[:6]) if value > 0.0
    ]
    if len(positives) != 1:
        raise MontageGrammarEvidenceError(
            "labelled shot token has no unique acoustic role"
        )
    if any(value < 0.0 or value > 1.0 for value in section_values[:7]):
        raise MontageGrammarEvidenceError("section role/progress lies outside [0, 1]")
    phrase_index = DEFAULT_STAGE_EDIT_SCHEMA.music_feature_names.index(
        "phrase_progress"
    )
    if token.missing_mask[phrase_index]:
        raise MontageGrammarEvidenceError(
            "phrase progress is missing at a labelled shot clock"
        )
    phrase_progress = _finite(
        token.music_features[phrase_index],
        "phrase_progress",
    )
    if not 0.0 <= phrase_progress <= 1.0:
        raise MontageGrammarEvidenceError("phrase progress lies outside [0, 1]")
    return ROLE_NAMES[positives[0]], section_values[6], phrase_progress


def _selected_visual_observation(
    token: Any,
    source_id: str,
) -> tuple[str | None, str | None, float | None, float | None]:
    matches = tuple(
        candidate
        for candidate in token.source_candidates
        if candidate.source_id == source_id
    )
    if len(matches) != 1:
        raise MontageGrammarEvidenceError(
            f"selected source {source_id} is absent from exact StageEdit token"
        )
    candidate = matches[0]
    index = DEFAULT_STAGE_EDIT_SCHEMA.source_feature_names.index(
        SHOT_SCALE_FEATURE_NAME
    )
    value = _finite(candidate.features[index], "selected source shot_scale")
    missing = candidate.missing_mask[index]
    if missing and value != 0.0:
        raise MontageGrammarEvidenceError(
            "masked selected-source shot_scale must use the canonical zero fill"
        )
    if missing:
        size: str | None = None
        unknown_reason: str | None = "missing_mask"
    elif not candidate.available:
        size = None
        unknown_reason = "source_unavailable"
    elif not candidate.hard_gate_passed:
        size = None
        unknown_reason = "hard_gate_failed"
    else:
        if not 0.0 <= value <= 1.0:
            raise MontageGrammarEvidenceError(
                "selected source shot_scale is outside [0, 1]"
            )
        size = _shot_size(value)
        unknown_reason = None

    def optional_feature(name: str) -> float | None:
        feature_index = DEFAULT_STAGE_EDIT_SCHEMA.source_feature_names.index(name)
        feature_value = _finite(
            candidate.features[feature_index],
            f"selected source {name}",
        )
        if candidate.missing_mask[feature_index]:
            if feature_value != 0.0:
                raise MontageGrammarEvidenceError(
                    f"masked selected-source {name} must use canonical zero fill"
                )
            return None
        if not candidate.available or not candidate.hard_gate_passed:
            return None
        if not 0.0 <= feature_value <= 1.0:
            raise MontageGrammarEvidenceError(
                f"selected source {name} lies outside [0, 1]"
            )
        return feature_value

    detected_count = optional_feature("detected_person_count_normalized")
    formation_count = optional_feature("formation_person_count_normalized")
    return size, unknown_reason, detected_count, formation_count


def _burst_events(
    shots: Sequence[_Shot],
    *,
    maximum_shot_duration: float | None,
    close_only: bool,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: list[_Shot] = []

    def qualifies(shot: _Shot) -> bool:
        allowed = {"close"} if close_only else {"medium", "close"}
        return shot.size in allowed and (
            maximum_shot_duration is None
            or shot.duration <= maximum_shot_duration + 1e-12
        )

    def flush() -> None:
        nonlocal current
        source_switch_count = sum(
            left.source_id != right.source_id
            for left, right in zip(current, current[1:])
        )
        if len(current) >= 2 and source_switch_count >= 1:
            duration = sum(item.duration for item in current)
            events.append(
                {
                    "role": current[0].role,
                    "shot_count": len(current),
                    "duration_seconds": duration,
                    "source_switch_count": source_switch_count,
                    "source_switch_fraction": source_switch_count / (len(current) - 1),
                }
            )
        current = []

    for shot in shots:
        if not qualifies(shot):
            flush()
            continue
        if not current:
            current = [shot]
            continue
        previous = current[-1]
        connected = (
            previous.end == shot.start
            and previous.role == shot.role
        )
        if connected:
            current.append(shot)
        else:
            flush()
            current = [shot]
    flush()
    return events


def _consecutive_wide_runs(shots: Sequence[_Shot]) -> list[dict[str, Any]]:
    """Return maximal, contiguous known-wide runs within one acoustic role.

    A single long wide shot is deliberately a run: the production failure this
    evidence is meant to expose is excessive *time* spent wide, not merely a
    repeated count label.  Unknown shot scale, a clock gap, or a role change
    terminates a run and is never silently treated as wide.
    """

    events: list[dict[str, Any]] = []
    current: list[_Shot] = []

    def flush() -> None:
        nonlocal current
        if current:
            events.append(
                {
                    "role": current[0].role,
                    "shot_count": len(current),
                    "duration_seconds": sum(shot.duration for shot in current),
                }
            )
        current = []

    for shot in shots:
        if shot.size != "wide":
            flush()
            continue
        if not current:
            current = [shot]
            continue
        previous = current[-1]
        if previous.end == shot.start and previous.role == shot.role:
            current.append(shot)
        else:
            flush()
            current = [shot]
    flush()
    return events


def _empty_role_metric(role: str) -> dict[str, Any]:
    return {
        "role": role,
        "shot_count": 0,
        "role_duration_seconds": 0.0,
        "observed_shot_scale_count": 0,
        "unknown_shot_scale_count": 0,
        "observed_shot_scale_duration_seconds": 0.0,
        "unknown_shot_scale_duration_seconds": 0.0,
        "unknown_shot_scale_reasons": {
            "missing_mask": 0,
            "source_unavailable": 0,
            "hard_gate_failed": 0,
        },
        "shot_size_counts": {name: 0 for name in SHOT_SIZE_NAMES},
        "shot_size_fractions": {name: None for name in SHOT_SIZE_NAMES},
        "shot_size_duration_seconds": {
            name: 0.0 for name in SHOT_SIZE_NAMES
        },
        "shot_size_duration_fractions": {
            name: None for name in SHOT_SIZE_NAMES
        },
        "nonwide_duration_seconds": 0.0,
        "nonwide_duration_fraction": None,
        "consecutive_wide_run_count": 0,
        "consecutive_wide_run_shot_counts": [],
        "consecutive_wide_run_durations_seconds": [],
        "maximum_consecutive_wide_run_seconds": 0.0,
        "eligible_source_boundary_count": 0,
        "source_switch_count": 0,
        "source_switch_fraction": None,
        "observed_source_switch_pair_count": 0,
        "unknown_source_switch_pair_count": 0,
        "source_switch_pair_counts": {name: 0 for name in PAIR_NAMES},
        "source_switch_pair_fractions": {name: None for name in PAIR_NAMES},
        "wide_to_wide_source_switch_fraction": None,
        "source_switch_to_wide_fraction": None,
        "source_switch_from_wide_fraction": None,
        "source_switch_with_wide_fraction": None,
        "wide_shot_detected_person_count_observed_count": 0,
        "wide_shot_detected_person_count_normalized_values": [],
        "wide_shot_formation_person_count_observed_count": 0,
        "wide_shot_formation_person_count_normalized_values": [],
        "section_count": 0,
        "observed_entry_anchor_count": 0,
        "entry_wide_anchor_count": 0,
        "entry_wide_anchor_fraction": None,
        "observed_early_anchor_count": 0,
        "early_wide_anchor_count": 0,
        "early_wide_anchor_fraction": None,
        "phrase_count": 0,
        "observed_phrase_entry_anchor_count": 0,
        "phrase_entry_wide_anchor_count": 0,
        "phrase_entry_wide_anchor_fraction": None,
        "short_nonwide_burst_count": 0,
        "short_nonwide_burst_rate_per_minute": None,
        "short_nonwide_burst_shot_counts": [],
        "short_nonwide_burst_durations_seconds": [],
        "short_nonwide_burst_source_switch_fractions": [],
        "short_closeup_burst_count": 0,
        "short_closeup_burst_rate_per_minute": None,
        "short_closeup_burst_shot_counts": [],
        "short_closeup_burst_durations_seconds": [],
        "short_closeup_burst_source_switch_fractions": [],
        "nonwide_multi_source_burst_count": 0,
        "nonwide_multi_source_burst_rate_per_minute": None,
        "nonwide_multi_source_burst_shot_counts": [],
        "nonwide_multi_source_burst_durations_seconds": [],
        "nonwide_multi_source_burst_source_switch_fractions": [],
        "closeup_multi_source_burst_count": 0,
        "closeup_multi_source_burst_rate_per_minute": None,
        "closeup_multi_source_burst_shot_counts": [],
        "closeup_multi_source_burst_durations_seconds": [],
        "closeup_multi_source_burst_source_switch_fractions": [],
    }


def _measure_reference(
    *,
    song_id: str,
    reference_id: str,
    supervision: Mapping[str, Any],
    snapshot: StageEditFeatureSnapshot,
    source_ids: tuple[str, ...],
    early_anchor_shot_count: int,
) -> dict[str, Any]:
    raw_labels = _array(supervision.get("shot_labels"), "shot_labels")
    if not raw_labels:
        raise MontageGrammarEvidenceError("reference supervision has no shot labels")
    token_by_time = {
        token.master_time_seconds: token for token in snapshot.sequence.tokens
    }
    if len(token_by_time) != len(snapshot.sequence.tokens):
        raise MontageGrammarEvidenceError("snapshot token clock is not unique")
    shots: list[_Shot] = []
    previous_start = -math.inf
    seen_indices: set[int] = set()
    for label_position, raw_label in enumerate(raw_labels):
        label = _mapping(raw_label, f"shot label {label_position}")
        required = {
            "record_id",
            "style_id",
            "reference_id",
            "shot_index",
            "selected_source_id",
            "selected_source_time",
            "edit_labels",
            "attribution_confidence",
        }
        _exact_keys(label, required, f"shot label {label_position}")
        if label.get("reference_id") != reference_id:
            raise MontageGrammarEvidenceError("shot label reference id mismatch")
        shot_index = _integer(label.get("shot_index"), "shot_index")
        if shot_index < 0 or shot_index in seen_indices:
            raise MontageGrammarEvidenceError("shot indices are invalid or duplicate")
        seen_indices.add(shot_index)
        source_id = str(label.get("selected_source_id") or "").strip()
        if source_id not in source_ids:
            raise MontageGrammarEvidenceError("selected source is outside source catalog")
        edit_labels = _mapping(label.get("edit_labels"), "edit_labels")
        _exact_keys(
            edit_labels,
            {
                "cut",
                "source_transition",
                "boundary_event",
                "dense_attribution",
                "crop",
                "geometry_evidence",
            },
            "edit_labels",
        )
        cut = _mapping(edit_labels.get("cut"), "cut label")
        _exact_keys(
            cut,
            {
                "boundary_label",
                "nominal_cut_master_time",
                "cut_master_time",
                "end_master_time",
                "duration_seconds",
            },
            "cut label",
        )
        start = _finite(cut.get("cut_master_time"), "cut_master_time")
        nominal = _finite(
            cut.get("nominal_cut_master_time"),
            "nominal_cut_master_time",
        )
        end = _finite(cut.get("end_master_time"), "end_master_time")
        duration = _finite(cut.get("duration_seconds"), "duration_seconds")
        if start < 0.0 or start <= previous_start or end <= start:
            raise MontageGrammarEvidenceError("labelled shot clock is invalid")
        if duration <= 0.0 or not math.isclose(
            duration,
            end - start,
            abs_tol=1e-9,
        ):
            raise MontageGrammarEvidenceError("shot duration disagrees with its clock")
        if nominal < 0.0:
            raise MontageGrammarEvidenceError("nominal cut clock cannot be negative")
        if start not in token_by_time:
            raise MontageGrammarEvidenceError(
                f"{song_id}:{reference_id} has no exact StageEdit token at "
                f"cut_master_time={start!r}"
            )
        token = token_by_time[start]
        role, progress, phrase_progress = _role_and_progress(token)
        (
            size,
            unknown_reason,
            detected_person_count,
            formation_person_count,
        ) = _selected_visual_observation(token, source_id)
        shots.append(
            _Shot(
                shot_index=shot_index,
                start=start,
                end=end,
                duration=duration,
                source_id=source_id,
                role=role,
                section_progress=progress,
                phrase_progress=phrase_progress,
                size=size,
                scale_unknown_reason=unknown_reason,
                detected_person_count_normalized=detected_person_count,
                formation_person_count_normalized=formation_person_count,
            )
        )
        previous_start = start

    durations = tuple(shot.duration for shot in shots)
    duration_median = float(median(durations))
    nonwide_events = _burst_events(
        shots,
        maximum_shot_duration=duration_median,
        close_only=False,
    )
    close_events = _burst_events(
        shots,
        maximum_shot_duration=duration_median,
        close_only=True,
    )
    all_nonwide_events = _burst_events(
        shots,
        maximum_shot_duration=None,
        close_only=False,
    )
    all_close_events = _burst_events(
        shots,
        maximum_shot_duration=None,
        close_only=True,
    )
    consecutive_wide_runs = _consecutive_wide_runs(shots)
    role_metrics = {role: _empty_role_metric(role) for role in ROLE_NAMES}
    for shot in shots:
        metric = role_metrics[shot.role]
        metric["shot_count"] += 1
        metric["role_duration_seconds"] += shot.duration
        if shot.size is None:
            metric["unknown_shot_scale_count"] += 1
            metric["unknown_shot_scale_duration_seconds"] += shot.duration
            metric["unknown_shot_scale_reasons"][shot.scale_unknown_reason] += 1
        else:
            metric["observed_shot_scale_count"] += 1
            metric["observed_shot_scale_duration_seconds"] += shot.duration
            metric["shot_size_counts"][shot.size] += 1
            metric["shot_size_duration_seconds"][shot.size] += shot.duration
            if shot.size in {"medium", "close"}:
                metric["nonwide_duration_seconds"] += shot.duration
            if shot.size == "wide":
                if shot.detected_person_count_normalized is not None:
                    metric[
                        "wide_shot_detected_person_count_observed_count"
                    ] += 1
                    metric[
                        "wide_shot_detected_person_count_normalized_values"
                    ].append(shot.detected_person_count_normalized)
                if shot.formation_person_count_normalized is not None:
                    metric[
                        "wide_shot_formation_person_count_observed_count"
                    ] += 1
                    metric[
                        "wide_shot_formation_person_count_normalized_values"
                    ].append(shot.formation_person_count_normalized)

    for event in consecutive_wide_runs:
        metric = role_metrics[str(event["role"])]
        metric["consecutive_wide_run_count"] += 1
        metric["consecutive_wide_run_shot_counts"].append(event["shot_count"])
        metric["consecutive_wide_run_durations_seconds"].append(
            event["duration_seconds"]
        )
        metric["maximum_consecutive_wide_run_seconds"] = max(
            metric["maximum_consecutive_wide_run_seconds"],
            event["duration_seconds"],
        )

    for earlier, later in zip(shots, shots[1:]):
        if earlier.end != later.start:
            continue
        metric = role_metrics[later.role]
        metric["eligible_source_boundary_count"] += 1
        if earlier.source_id == later.source_id:
            continue
        metric["source_switch_count"] += 1
        if earlier.size is None or later.size is None:
            metric["unknown_source_switch_pair_count"] += 1
            continue
        pair = f"{earlier.size}_to_{later.size}"
        metric["observed_source_switch_pair_count"] += 1
        metric["source_switch_pair_counts"][pair] += 1

    # Section identity is recovered from the exact role token and the reset of
    # its serialized section-progress coordinate.  This does not guess lyrical
    # boundaries from timestamps.
    sections: list[list[_Shot]] = []
    for shot in shots:
        if not sections:
            sections.append([shot])
            continue
        previous = sections[-1][-1]
        new_section = (
            shot.role != previous.role
            or shot.section_progress + 1e-12 < previous.section_progress
        )
        if new_section:
            sections.append([shot])
        else:
            sections[-1].append(shot)
    for section in sections:
        metric = role_metrics[section[0].role]
        metric["section_count"] += 1
        entry = section[0]
        if entry.size is not None:
            metric["observed_entry_anchor_count"] += 1
            if entry.size == "wide":
                metric["entry_wide_anchor_count"] += 1
        early = section[:early_anchor_shot_count]
        observed_early = tuple(shot for shot in early if shot.size is not None)
        if observed_early:
            metric["observed_early_anchor_count"] += 1
            if any(shot.size == "wide" for shot in observed_early):
                metric["early_wide_anchor_count"] += 1

    # Phrase entries use only the serialized inference-time phrase-progress
    # reset.  No lyric labels or guessed wall-clock window is introduced.
    phrases: list[list[_Shot]] = []
    for shot in shots:
        if not phrases:
            phrases.append([shot])
            continue
        previous = phrases[-1][-1]
        if shot.phrase_progress + 1e-12 < previous.phrase_progress:
            phrases.append([shot])
        else:
            phrases[-1].append(shot)
    for phrase in phrases:
        entry = phrase[0]
        metric = role_metrics[entry.role]
        metric["phrase_count"] += 1
        if entry.size is not None:
            metric["observed_phrase_entry_anchor_count"] += 1
            if entry.size == "wide":
                metric["phrase_entry_wide_anchor_count"] += 1

    for event_name, events in (
        ("short_nonwide_burst", nonwide_events),
        ("short_closeup_burst", close_events),
        ("nonwide_multi_source_burst", all_nonwide_events),
        ("closeup_multi_source_burst", all_close_events),
    ):
        for event in events:
            metric = role_metrics[str(event["role"])]
            metric[f"{event_name}_count"] += 1
            metric[f"{event_name}_shot_counts"].append(event["shot_count"])
            metric[f"{event_name}_durations_seconds"].append(
                event["duration_seconds"]
            )
            metric[f"{event_name}_source_switch_fractions"].append(
                event["source_switch_fraction"]
            )

    for metric in role_metrics.values():
        observed = metric["observed_shot_scale_count"]
        if observed:
            metric["shot_size_fractions"] = {
                name: metric["shot_size_counts"][name] / observed
                for name in SHOT_SIZE_NAMES
            }
        observed_duration = metric["observed_shot_scale_duration_seconds"]
        if observed_duration > 0.0:
            metric["shot_size_duration_fractions"] = {
                name: metric["shot_size_duration_seconds"][name]
                / observed_duration
                for name in SHOT_SIZE_NAMES
            }
            # This is intentionally calculated from the two measured duration
            # buckets.  It is never synthesized as ``1 - wide``.
            metric["nonwide_duration_fraction"] = (
                metric["shot_size_duration_seconds"]["medium"]
                + metric["shot_size_duration_seconds"]["close"]
            ) / observed_duration
        boundaries = metric["eligible_source_boundary_count"]
        if boundaries:
            metric["source_switch_fraction"] = (
                metric["source_switch_count"] / boundaries
            )
        observed_pairs = metric["observed_source_switch_pair_count"]
        if observed_pairs:
            fractions = {
                name: metric["source_switch_pair_counts"][name] / observed_pairs
                for name in PAIR_NAMES
            }
            metric["source_switch_pair_fractions"] = fractions
            metric["wide_to_wide_source_switch_fraction"] = fractions[
                "wide_to_wide"
            ]
            metric["source_switch_to_wide_fraction"] = sum(
                fractions[f"{outgoing}_to_wide"] for outgoing in SHOT_SIZE_NAMES
            )
            metric["source_switch_from_wide_fraction"] = sum(
                fractions[f"wide_to_{incoming}"] for incoming in SHOT_SIZE_NAMES
            )
            metric["source_switch_with_wide_fraction"] = sum(
                fraction
                for pair, fraction in fractions.items()
                if pair.startswith("wide_to_") or pair.endswith("_to_wide")
            )
        entry_observed = metric["observed_entry_anchor_count"]
        if entry_observed:
            metric["entry_wide_anchor_fraction"] = (
                metric["entry_wide_anchor_count"] / entry_observed
            )
        early_observed = metric["observed_early_anchor_count"]
        if early_observed:
            metric["early_wide_anchor_fraction"] = (
                metric["early_wide_anchor_count"] / early_observed
            )
        phrase_observed = metric["observed_phrase_entry_anchor_count"]
        if phrase_observed:
            metric["phrase_entry_wide_anchor_fraction"] = (
                metric["phrase_entry_wide_anchor_count"] / phrase_observed
            )
        role_duration = metric["role_duration_seconds"]
        if role_duration > 0.0:
            for name in (
                "short_nonwide_burst",
                "short_closeup_burst",
                "nonwide_multi_source_burst",
                "closeup_multi_source_burst",
            ):
                metric[f"{name}_rate_per_minute"] = (
                    metric[f"{name}_count"] * 60.0 / role_duration
                )

    return {
        "reference_id": reference_id,
        "shot_count": len(shots),
        "known_neighbor_pair_count": max(0, len(shots) - 1),
        "contiguous_known_boundary_count": sum(
            earlier.end == later.start
            for earlier, later in zip(shots, shots[1:])
        ),
        "unknown_gap_between_known_shots_count": sum(
            earlier.end != later.start
            for earlier, later in zip(shots, shots[1:])
        ),
        "observed_shot_scale_count": sum(
            metric["observed_shot_scale_count"] for metric in role_metrics.values()
        ),
        "unknown_shot_scale_count": sum(
            metric["unknown_shot_scale_count"] for metric in role_metrics.values()
        ),
        "short_shot_reference_median_duration_seconds": duration_median,
        "by_acoustic_role": role_metrics,
    }


def _metric_values(
    references: Sequence[Mapping[str, Any]],
    role: str,
    getter: Any,
) -> list[float]:
    values: list[float] = []
    for reference in references:
        role_metric = _mapping(reference["by_acoustic_role"], "by_acoustic_role")[
            role
        ]
        value = getter(_mapping(role_metric, f"{role} metric"))
        if value is not None:
            values.append(_finite(value, f"{role} aggregate metric"))
    return values


def _aggregate_roles(
    songs: Sequence[Mapping[str, Any]],
    *,
    floor_quantile: float,
) -> dict[str, Any]:
    references = [
        _mapping(reference, "reference metrics")
        for song in songs
        for reference in _array(song.get("references"), "song references")
    ]
    result: dict[str, Any] = {}
    for role in ROLE_NAMES:
        contributors = [
            reference
            for reference in references
            if _mapping(reference["by_acoustic_role"], "by_acoustic_role")[role][
                "shot_count"
            ]
            > 0
        ]
        pooled_counts = {
            "shot_count": sum(
                reference["by_acoustic_role"][role]["shot_count"]
                for reference in contributors
            ),
            "observed_shot_scale_count": sum(
                reference["by_acoustic_role"][role]["observed_shot_scale_count"]
                for reference in contributors
            ),
            "unknown_shot_scale_count": sum(
                reference["by_acoustic_role"][role]["unknown_shot_scale_count"]
                for reference in contributors
            ),
            "consecutive_wide_run_count": sum(
                reference["by_acoustic_role"][role][
                    "consecutive_wide_run_count"
                ]
                for reference in contributors
            ),
            "source_switch_count": sum(
                reference["by_acoustic_role"][role]["source_switch_count"]
                for reference in contributors
            ),
            "observed_source_switch_pair_count": sum(
                reference["by_acoustic_role"][role][
                    "observed_source_switch_pair_count"
                ]
                for reference in contributors
            ),
            "section_count": sum(
                reference["by_acoustic_role"][role]["section_count"]
                for reference in contributors
            ),
            "phrase_count": sum(
                reference["by_acoustic_role"][role]["phrase_count"]
                for reference in contributors
            ),
            "short_nonwide_burst_count": sum(
                reference["by_acoustic_role"][role][
                    "short_nonwide_burst_count"
                ]
                for reference in contributors
            ),
            "short_closeup_burst_count": sum(
                reference["by_acoustic_role"][role]["short_closeup_burst_count"]
                for reference in contributors
            ),
            "nonwide_multi_source_burst_count": sum(
                reference["by_acoustic_role"][role][
                    "nonwide_multi_source_burst_count"
                ]
                for reference in contributors
            ),
            "closeup_multi_source_burst_count": sum(
                reference["by_acoustic_role"][role][
                    "closeup_multi_source_burst_count"
                ]
                for reference in contributors
            ),
        }
        pooled_durations_seconds = {
            "role": sum(
                reference["by_acoustic_role"][role]["role_duration_seconds"]
                for reference in contributors
            ),
            "observed_shot_scale": sum(
                reference["by_acoustic_role"][role][
                    "observed_shot_scale_duration_seconds"
                ]
                for reference in contributors
            ),
            "unknown_shot_scale": sum(
                reference["by_acoustic_role"][role][
                    "unknown_shot_scale_duration_seconds"
                ]
                for reference in contributors
            ),
            "shot_size": {
                size: sum(
                    reference["by_acoustic_role"][role][
                        "shot_size_duration_seconds"
                    ][size]
                    for reference in contributors
                )
                for size in SHOT_SIZE_NAMES
            },
            "nonwide": sum(
                reference["by_acoustic_role"][role]["nonwide_duration_seconds"]
                for reference in contributors
            ),
        }
        distributions: dict[str, Any] = {
            # Retained as a clearly count-derived audit distribution for
            # compatibility.  Production duration policies must use the
            # separate duration paths below.
            "shot_size_fractions": {
                size: _reference_distribution(
                    _metric_values(
                        contributors,
                        role,
                        lambda metric, size=size: metric["shot_size_fractions"][
                            size
                        ],
                    ),
                    floor_quantile=floor_quantile,
                )
                for size in SHOT_SIZE_NAMES
            },
            "shot_size_duration_fractions": {
                size: _reference_distribution(
                    _metric_values(
                        contributors,
                        role,
                        lambda metric, size=size: metric[
                            "shot_size_duration_fractions"
                        ][size],
                    ),
                    floor_quantile=floor_quantile,
                )
                for size in SHOT_SIZE_NAMES
            },
            "nonwide_duration_fraction": _reference_distribution(
                _metric_values(
                    contributors,
                    role,
                    lambda metric: metric["nonwide_duration_fraction"],
                ),
                floor_quantile=floor_quantile,
            ),
            "maximum_consecutive_wide_run_seconds": _reference_distribution(
                _metric_values(
                    contributors,
                    role,
                    lambda metric: metric[
                        "maximum_consecutive_wide_run_seconds"
                    ],
                ),
                floor_quantile=floor_quantile,
            ),
            "source_switch_fraction": _reference_distribution(
                _metric_values(
                    contributors,
                    role,
                    lambda metric: metric["source_switch_fraction"],
                ),
                floor_quantile=floor_quantile,
            ),
            "source_switch_pair_fractions": {
                pair: _reference_distribution(
                    _metric_values(
                        contributors,
                        role,
                        lambda metric, pair=pair: metric[
                            "source_switch_pair_fractions"
                        ][pair],
                    ),
                    floor_quantile=floor_quantile,
                )
                for pair in PAIR_NAMES
            },
            "wide_to_wide_source_switch_fraction": _reference_distribution(
                _metric_values(
                    contributors,
                    role,
                    lambda metric: metric[
                        "wide_to_wide_source_switch_fraction"
                    ],
                ),
                floor_quantile=floor_quantile,
            ),
            "source_switch_to_wide_fraction": _reference_distribution(
                _metric_values(
                    contributors,
                    role,
                    lambda metric: metric["source_switch_to_wide_fraction"],
                ),
                floor_quantile=floor_quantile,
            ),
            "source_switch_from_wide_fraction": _reference_distribution(
                _metric_values(
                    contributors,
                    role,
                    lambda metric: metric["source_switch_from_wide_fraction"],
                ),
                floor_quantile=floor_quantile,
            ),
            "source_switch_with_wide_fraction": _reference_distribution(
                _metric_values(
                    contributors,
                    role,
                    lambda metric: metric["source_switch_with_wide_fraction"],
                ),
                floor_quantile=floor_quantile,
            ),
            "entry_wide_anchor_fraction": _reference_distribution(
                _metric_values(
                    contributors,
                    role,
                    lambda metric: metric["entry_wide_anchor_fraction"],
                ),
                floor_quantile=floor_quantile,
            ),
            "early_wide_anchor_fraction": _reference_distribution(
                _metric_values(
                    contributors,
                    role,
                    lambda metric: metric["early_wide_anchor_fraction"],
                ),
                floor_quantile=floor_quantile,
            ),
            "phrase_entry_wide_anchor_fraction": _reference_distribution(
                _metric_values(
                    contributors,
                    role,
                    lambda metric: metric[
                        "phrase_entry_wide_anchor_fraction"
                    ],
                ),
                floor_quantile=floor_quantile,
            ),
            "short_nonwide_burst_rate_per_minute": _reference_distribution(
                _metric_values(
                    contributors,
                    role,
                    lambda metric: metric["short_nonwide_burst_rate_per_minute"],
                ),
                floor_quantile=floor_quantile,
            ),
            "short_closeup_burst_rate_per_minute": _reference_distribution(
                _metric_values(
                    contributors,
                    role,
                    lambda metric: metric["short_closeup_burst_rate_per_minute"],
                ),
                floor_quantile=floor_quantile,
            ),
            "nonwide_multi_source_burst_rate_per_minute": _reference_distribution(
                _metric_values(
                    contributors,
                    role,
                    lambda metric: metric[
                        "nonwide_multi_source_burst_rate_per_minute"
                    ],
                ),
                floor_quantile=floor_quantile,
            ),
            "closeup_multi_source_burst_rate_per_minute": _reference_distribution(
                _metric_values(
                    contributors,
                    role,
                    lambda metric: metric[
                        "closeup_multi_source_burst_rate_per_minute"
                    ],
                ),
                floor_quantile=floor_quantile,
            ),
        }
        nonwide_shot_counts = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "short_nonwide_burst_shot_counts"
            ]
        ]
        nonwide_durations = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "short_nonwide_burst_durations_seconds"
            ]
        ]
        close_shot_counts = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "short_closeup_burst_shot_counts"
            ]
        ]
        close_durations = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "short_closeup_burst_durations_seconds"
            ]
        ]
        short_nonwide_switch_fractions = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "short_nonwide_burst_source_switch_fractions"
            ]
        ]
        short_close_switch_fractions = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "short_closeup_burst_source_switch_fractions"
            ]
        ]
        wide_detected_counts = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "wide_shot_detected_person_count_normalized_values"
            ]
        ]
        wide_formation_counts = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "wide_shot_formation_person_count_normalized_values"
            ]
        ]
        multi_nonwide_shot_counts = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "nonwide_multi_source_burst_shot_counts"
            ]
        ]
        multi_nonwide_durations = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "nonwide_multi_source_burst_durations_seconds"
            ]
        ]
        multi_nonwide_switch_fractions = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "nonwide_multi_source_burst_source_switch_fractions"
            ]
        ]
        multi_close_shot_counts = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "closeup_multi_source_burst_shot_counts"
            ]
        ]
        multi_close_durations = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "closeup_multi_source_burst_durations_seconds"
            ]
        ]
        multi_close_switch_fractions = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "closeup_multi_source_burst_source_switch_fractions"
            ]
        ]
        wide_run_shot_counts = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "consecutive_wide_run_shot_counts"
            ]
        ]
        wide_run_durations = [
            float(value)
            for reference in contributors
            for value in reference["by_acoustic_role"][role][
                "consecutive_wide_run_durations_seconds"
            ]
        ]
        result[role] = {
            "contributing_reference_count": len(contributors),
            "pooled_counts": pooled_counts,
            "pooled_durations_seconds": pooled_durations_seconds,
            "reference_distributions": distributions,
            "burst_event_distributions": {
                "short_nonwide_shot_count": _event_distribution(
                    nonwide_shot_counts
                ),
                "short_nonwide_duration_seconds": _event_distribution(
                    nonwide_durations
                ),
                "short_nonwide_switch_fraction": _event_distribution(
                    short_nonwide_switch_fractions
                ),
                "short_closeup_shot_count": _event_distribution(close_shot_counts),
                "short_closeup_duration_seconds": _event_distribution(
                    close_durations
                ),
                "short_closeup_switch_fraction": _event_distribution(
                    short_close_switch_fractions
                ),
                "nonwide_multi_source_shot_count": _event_distribution(
                    multi_nonwide_shot_counts
                ),
                "nonwide_multi_source_duration_seconds": _event_distribution(
                    multi_nonwide_durations
                ),
                "nonwide_multi_source_switch_fraction": _event_distribution(
                    multi_nonwide_switch_fractions
                ),
                "closeup_multi_source_shot_count": _event_distribution(
                    multi_close_shot_counts
                ),
                "closeup_multi_source_duration_seconds": _event_distribution(
                    multi_close_durations
                ),
                "closeup_multi_source_switch_fraction": _event_distribution(
                    multi_close_switch_fractions
                ),
            },
            "wide_run_event_distributions": {
                "consecutive_wide_run_shot_count": _event_distribution(
                    wide_run_shot_counts
                ),
                "consecutive_wide_run_duration_seconds": _event_distribution(
                    wide_run_durations
                ),
                "semantics": (
                    "maximal_contiguous_known_wide_shots_within_one_"
                    "acoustic_role_single_wide_shot_is_a_run"
                ),
            },
            "wide_shot_person_count_evidence": {
                "detected_person_count_normalized": _event_distribution(
                    wide_detected_counts
                ),
                "formation_person_count_normalized": _event_distribution(
                    wide_formation_counts
                ),
                "semantics": (
                    "identity_free_observed_counts_only_no_group_formation_threshold"
                ),
            },
        }
    return result


def _load_materialization(
    item: MontageSongInput,
    config: MontageGrammarConfig,
) -> tuple[dict[str, Any], dict[str, tuple[Path, str]], bytes]:
    payload, raw = _read_json(
        item.materialization_manifest_path,
        f"{item.song_id} materialization manifest",
    )
    _exact_keys(payload, _MATERIALIZATION_KEYS, "materialization manifest")
    if payload.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION:
        raise MontageGrammarEvidenceError("materialization schema version mismatch")
    if payload.get("analysis_basis") != MATERIALIZATION_ANALYSIS_BASIS:
        raise MontageGrammarEvidenceError("materialization analysis basis mismatch")
    if payload.get("immutable_inputs_modified") is not False:
        raise MontageGrammarEvidenceError("materialization mutated immutable inputs")
    contract = _mapping(payload.get("input_contract"), "input contract")
    fingerprint = payload.get("input_fingerprint_sha256")
    if not _is_sha256(fingerprint) or _canonical_sha256(contract) != fingerprint:
        raise MontageGrammarEvidenceError("materialization fingerprint mismatch")
    expected_name = f"reference-training-materialization-{fingerprint[:16]}.json"
    if item.materialization_manifest_path.name != expected_name:
        raise MontageGrammarEvidenceError(
            "materialization manifest filename is not content-addressed"
        )
    if contract.get("analysis_basis") != MATERIALIZATION_ANALYSIS_BASIS:
        raise MontageGrammarEvidenceError("materialization contract basis mismatch")
    if contract.get("reference_supervision_schema_version") != (
        REFERENCE_SUPERVISION_SCHEMA_VERSION
    ):
        raise MontageGrammarEvidenceError("reference supervision schema mismatch")
    expected_references = _integer(
        contract.get("expected_reference_count"),
        "expected_reference_count",
    )
    expected_sources = _integer(
        contract.get("expected_source_count"),
        "expected_source_count",
    )
    if expected_references < config.minimum_references_per_song:
        raise MontageGrammarEvidenceError("not enough references for montage profile")
    if expected_sources < config.minimum_sources_per_song:
        raise MontageGrammarEvidenceError("not enough sources for montage profile")
    contract_references = {
        str(_mapping(value, "materialization input").get("reference_id") or "")
        for value in _array(contract.get("inputs"), "materialization inputs")
    }
    if len(contract_references) != expected_references or "" in contract_references:
        raise MontageGrammarEvidenceError("materialization input references mismatch")
    outputs: dict[str, tuple[Path, str]] = {}
    for raw_output in _array(payload.get("outputs"), "materialization outputs"):
        output = _mapping(raw_output, "materialization output")
        reference_id = str(output.get("reference_id") or "").strip()
        if not reference_id or reference_id in outputs:
            raise MontageGrammarEvidenceError("materialization output ids are invalid")
        supervision_path = Path(str(output.get("supervision_path") or "")).resolve()
        supervision_sha = output.get("supervision_file_sha256")
        _verified_json(
            supervision_path,
            supervision_sha,
            f"{item.song_id}:{reference_id} supervision",
        )
        outputs[reference_id] = (supervision_path, str(supervision_sha))
    if set(outputs) != contract_references or len(outputs) != expected_references:
        raise MontageGrammarEvidenceError(
            "materialization input/output reference sets mismatch"
        )
    statistics = _mapping(payload.get("statistics"), "materialization statistics")
    if statistics.get("reference_count") != expected_references:
        raise MontageGrammarEvidenceError("materialization statistics mismatch")
    return payload, outputs, raw


def _load_feature_manifest(
    item: MontageSongInput,
    config: MontageGrammarConfig,
    materialization_outputs: Mapping[str, tuple[Path, str]],
) -> tuple[dict[str, Any], bytes]:
    payload, raw = _read_json(
        item.feature_manifest_path,
        f"{item.song_id} feature manifest",
    )
    _exact_keys(payload, _FEATURE_MANIFEST_KEYS, "feature manifest")
    if payload.get("schema_version") != FEATURE_JOB_SCHEMA_VERSION:
        raise MontageGrammarEvidenceError("feature manifest schema version mismatch")
    if payload.get("analysis_basis") != FEATURE_JOB_ANALYSIS_BASIS:
        raise MontageGrammarEvidenceError("feature manifest analysis basis mismatch")
    if payload.get("song_id") != item.song_id:
        raise MontageGrammarEvidenceError("feature manifest song id mismatch")
    unsigned = deepcopy(payload)
    manifest_sha = unsigned.pop("manifest_sha256", None)
    if not _is_sha256(manifest_sha) or _canonical_sha256(unsigned) != manifest_sha:
        raise MontageGrammarEvidenceError("feature manifest content hash mismatch")
    expected_name = (
        f"stageedit-feature-snapshot-job-manifest-{manifest_sha[:16]}.json"
    )
    if item.feature_manifest_path.name != expected_name:
        raise MontageGrammarEvidenceError(
            "feature manifest filename is not content-addressed"
        )
    policy = _mapping(payload.get("input_policy"), "feature input policy")
    if dict(policy) != dict(FEATURE_SNAPSHOT_JOB_INPUT_POLICY):
        raise MontageGrammarEvidenceError("feature snapshot leakage policy mismatch")
    provenance = _mapping(payload.get("feature_provenance"), "feature provenance")
    if (
        provenance.get("producer") != FEATURE_MATERIALIZER_CONTRACT
        or set(_array(provenance.get("contracts"), "feature contracts"))
        != {
            *FEATURE_SNAPSHOT_JOB_PROVENANCE_CONTRACTS,
            STAGE_EDIT_CAUSAL_SENSING_BINDING_CONTRACT,
            STAGE_EDIT_CAUSAL_SENSING_MERGE_CONTRACT,
        }
    ):
        raise MontageGrammarEvidenceError("feature provenance contract mismatch")
    try:
        sensing_gate = validate_production_causal_sensing_gate_payload(
            _mapping(
                payload.get("production_causal_sensing_gate"),
                "production causal-sensing gate",
            ),
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
        )
    except (TypeError, ValueError) as exc:
        raise MontageGrammarEvidenceError(
            f"feature production causal-sensing gate is invalid: {exc}"
        ) from exc
    if not (
        sensing_gate["edit_mode"] == "group"
        and sensing_gate["target_member"] is None
        and sensing_gate["member_feature_mode"] == "group_no_target"
    ):
        raise MontageGrammarEvidenceError(
            "professional montage grammar requires group_no_target sensing"
        )
    try:
        manifest_binding = StageEditCausalSensingBinding.from_dict(
            _mapping(
                payload.get("causal_sensing_binding"),
                "causal-sensing binding",
            )
        )
    except (TypeError, ValueError) as exc:
        raise MontageGrammarEvidenceError(
            f"feature causal-sensing binding is invalid: {exc}"
        ) from exc
    if not (
        manifest_binding.shooting_grammar_content_sha256
        == sensing_gate["shooting_grammar_content_sha256"]
        and manifest_binding.member_state_graph_content_sha256
        == sensing_gate["member_state_graph_content_sha256"]
        and manifest_binding.ive_performer_state_content_sha256
        == sensing_gate["ive_performer_state_content_sha256"]
        and manifest_binding.target_member is None
    ):
        raise MontageGrammarEvidenceError(
            "feature causal-sensing binding/gate mismatch"
        )
    source_count = _integer(payload.get("source_count"), "source_count")
    if source_count < config.minimum_sources_per_song:
        raise MontageGrammarEvidenceError("feature manifest source count is too small")
    if len(_array(payload.get("source_content"), "source_content")) != source_count:
        raise MontageGrammarEvidenceError("feature source count mismatch")
    references = set(materialization_outputs)
    supervision_records: dict[str, tuple[Path, str]] = {}
    for raw_record in _array(payload.get("supervision_inputs"), "supervision inputs"):
        record = _mapping(raw_record, "supervision input")
        _exact_keys(record, {"reference_id", "path", "sha256"}, "supervision input")
        reference_id = str(record.get("reference_id") or "").strip()
        if not reference_id or reference_id in supervision_records:
            raise MontageGrammarEvidenceError("feature supervision ids are invalid")
        path = Path(str(record.get("path") or "")).resolve()
        digest = str(record.get("sha256") or "")
        if materialization_outputs.get(reference_id) != (path, digest):
            raise MontageGrammarEvidenceError(
                "feature/materialization supervision binding mismatch"
            )
        supervision_records[reference_id] = (path, digest)
    if set(supervision_records) != references:
        raise MontageGrammarEvidenceError("feature supervision reference set mismatch")
    raw_snapshot_records = _array(payload.get("snapshots"), "snapshot records")
    snapshot_ids = {
        str(_mapping(value, "snapshot record").get("reference_id") or "")
        for value in raw_snapshot_records
    }
    if (
        snapshot_ids != references
        or len(snapshot_ids) != len(references)
        or len(raw_snapshot_records) != len(references)
    ):
        raise MontageGrammarEvidenceError("feature snapshot reference set mismatch")
    statistics = _mapping(payload.get("statistics"), "feature statistics")
    if (
        statistics.get("reference_count") != len(references)
        or statistics.get("source_count") != source_count
    ):
        raise MontageGrammarEvidenceError("feature manifest statistics mismatch")
    return payload, raw


def _source_ids_and_catalog_hash(
    feature_manifest: Mapping[str, Any],
    supervisions: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, ...], str]:
    source_ids: list[str] = []
    for raw_source in _array(feature_manifest.get("source_content"), "source content"):
        source = _mapping(raw_source, "feature source")
        source_id = str(source.get("source_id") or "").strip()
        if not source_id or source_id in source_ids:
            raise MontageGrammarEvidenceError("feature source ids are invalid")
        if not _is_sha256(source.get("content_sha256")):
            raise MontageGrammarEvidenceError("feature source content hash is invalid")
        source_ids.append(source_id)
    canonical_ids = tuple(sorted(source_ids))
    if tuple(source_ids) != canonical_ids:
        raise MontageGrammarEvidenceError("feature source order is not canonical")
    catalog_hashes: set[str] = set()
    for reference_id, supervision in supervisions.items():
        catalog = _array(supervision.get("source_catalog"), "source catalog")
        ids = tuple(
            str(_mapping(item, "supervision source").get("video_id") or "").strip()
            for item in catalog
        )
        if len(set(ids)) != len(ids) or set(ids) != set(canonical_ids):
            raise MontageGrammarEvidenceError(
                f"{reference_id} supervision source set mismatch"
            )
        catalog_hashes.add(_canonical_sha256(catalog))
    declared = feature_manifest.get("source_catalog_sha256")
    if not _is_sha256(declared) or catalog_hashes != {declared}:
        raise MontageGrammarEvidenceError("source catalog content hash mismatch")
    return canonical_ids, str(declared)


def _build_song(
    item: MontageSongInput,
    config: MontageGrammarConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    materialization, outputs, materialization_raw = _load_materialization(
        item,
        config,
    )
    feature_manifest, feature_raw = _load_feature_manifest(
        item,
        config,
        outputs,
    )
    sensing_gate = validate_production_causal_sensing_gate_payload(
        _mapping(
            feature_manifest.get("production_causal_sensing_gate"),
            "production causal-sensing gate",
        ),
        schema=DEFAULT_STAGE_EDIT_SCHEMA,
    )
    contract = _mapping(materialization["input_contract"], "input contract")
    if feature_manifest.get("source_count") != contract.get(
        "expected_source_count"
    ):
        raise MontageGrammarEvidenceError(
            "materialization/feature source count mismatch"
        )
    if len(outputs) != contract.get("expected_reference_count"):
        raise MontageGrammarEvidenceError(
            "materialization/feature reference count mismatch"
        )
    supervisions: dict[str, dict[str, Any]] = {}
    supervision_raw: dict[str, bytes] = {}
    for reference_id, (path, digest) in outputs.items():
        payload, raw = _verified_json(
            path,
            digest,
            f"{item.song_id}:{reference_id} supervision",
        )
        _exact_keys(payload, _SUPERVISION_KEYS, "reference supervision")
        if payload.get("schema_version") != REFERENCE_SUPERVISION_SCHEMA_VERSION:
            raise MontageGrammarEvidenceError("supervision schema mismatch")
        if payload.get("dataset_kind") != REFERENCE_SUPERVISION_KIND:
            raise MontageGrammarEvidenceError("supervision dataset kind mismatch")
        if payload.get("reference_id") != reference_id:
            raise MontageGrammarEvidenceError("supervision reference id mismatch")
        supervisions[reference_id] = payload
        supervision_raw[reference_id] = raw
    source_ids, catalog_hash = _source_ids_and_catalog_hash(
        feature_manifest,
        supervisions,
    )
    snapshot_records = {
        str(_mapping(value, "snapshot record")["reference_id"]): _mapping(
            value,
            "snapshot record",
        )
        for value in _array(feature_manifest.get("snapshots"), "snapshot records")
    }
    reference_metrics: list[dict[str, Any]] = []
    reference_provenance: list[dict[str, Any]] = []
    for reference_id in sorted(supervisions):
        record = snapshot_records[reference_id]
        _exact_keys(
            record,
            {
                "reference_id",
                "path",
                "snapshot_sha256",
                "token_count",
                "source_count",
                "performance_start_seconds",
                "performance_end_seconds",
            },
            "snapshot record",
        )
        snapshot_path = Path(str(record.get("path") or "")).resolve()
        snapshot_payload, snapshot_raw = _read_json(
            snapshot_path,
            f"{item.song_id}:{reference_id} snapshot",
        )
        try:
            snapshot = StageEditFeatureSnapshot.from_dict(snapshot_payload)
        except (TypeError, ValueError) as exc:
            raise MontageGrammarEvidenceError(
                f"{item.song_id}:{reference_id} snapshot is invalid: {exc}"
            ) from exc
        if snapshot.snapshot_sha256 != record.get("snapshot_sha256"):
            raise MontageGrammarEvidenceError("snapshot internal hash mismatch")
        if snapshot.schema_version != FEATURE_MATERIALIZER_SCHEMA_VERSION:
            raise MontageGrammarEvidenceError("snapshot schema is not current")
        try:
            snapshot_gate = snapshot.require_production_causal_sensing(
                edit_mode="group",
                target_member=None,
                schema=DEFAULT_STAGE_EDIT_SCHEMA,
            )
        except (TypeError, ValueError) as exc:
            raise MontageGrammarEvidenceError(
                f"snapshot production causal sensing is invalid: {exc}"
            ) from exc
        if snapshot_gate != sensing_gate:
            raise MontageGrammarEvidenceError(
                "snapshot/manifest causal-sensing binding mismatch"
            )
        if len(snapshot.sequence.tokens) != record.get("token_count"):
            raise MontageGrammarEvidenceError("snapshot token count mismatch")
        expected_sequence = f"stageedit-features:{item.song_id}:{reference_id}"
        if snapshot.sequence.sequence_id != expected_sequence:
            raise MontageGrammarEvidenceError("snapshot sequence/reference mismatch")
        if record.get("source_count") != len(source_ids):
            raise MontageGrammarEvidenceError("snapshot source count mismatch")
        start = _finite(
            record.get("performance_start_seconds"),
            "performance_start_seconds",
        )
        end = _finite(
            record.get("performance_end_seconds"),
            "performance_end_seconds",
        )
        if end <= start:
            raise MontageGrammarEvidenceError("snapshot performance span is invalid")
        for token in snapshot.sequence.tokens:
            token_ids = tuple(candidate.source_id for candidate in token.source_candidates)
            if token_ids != source_ids:
                raise MontageGrammarEvidenceError("snapshot token source set mismatch")
        metric = _measure_reference(
            song_id=item.song_id,
            reference_id=reference_id,
            supervision=supervisions[reference_id],
            snapshot=snapshot,
            source_ids=source_ids,
            early_anchor_shot_count=config.early_anchor_shot_count,
        )
        reference_metrics.append(metric)
        supervision_path, supervision_sha = outputs[reference_id]
        reference_provenance.append(
            {
                "reference_id": reference_id,
                "supervision_path": os.fspath(supervision_path),
                "supervision_file_sha256": supervision_sha,
                "snapshot_path": os.fspath(snapshot_path),
                "snapshot_file_sha256": hashlib.sha256(snapshot_raw).hexdigest(),
                "snapshot_content_sha256": snapshot.snapshot_sha256,
                "snapshot_sequence_id": snapshot.sequence.sequence_id,
                "token_count": len(snapshot.sequence.tokens),
            }
        )
    song_payload = {
        "song_id": item.song_id,
        "reference_count": len(reference_metrics),
        "source_count": len(source_ids),
        "shot_count": sum(item["shot_count"] for item in reference_metrics),
        "known_neighbor_pair_count": sum(
            item["known_neighbor_pair_count"] for item in reference_metrics
        ),
        "contiguous_known_boundary_count": sum(
            item["contiguous_known_boundary_count"] for item in reference_metrics
        ),
        "unknown_gap_between_known_shots_count": sum(
            item["unknown_gap_between_known_shots_count"] for item in reference_metrics
        ),
        "observed_shot_scale_count": sum(
            item["observed_shot_scale_count"] for item in reference_metrics
        ),
        "unknown_shot_scale_count": sum(
            item["unknown_shot_scale_count"] for item in reference_metrics
        ),
        "references": reference_metrics,
    }
    song_payload["aggregate_by_acoustic_role"] = _aggregate_roles(
        (song_payload,),
        floor_quantile=config.reference_floor_quantile,
    )
    provenance = {
        "song_id": item.song_id,
        "feature_manifest_path": os.fspath(item.feature_manifest_path),
        "feature_manifest_file_sha256": hashlib.sha256(feature_raw).hexdigest(),
        "feature_manifest_content_sha256": feature_manifest["manifest_sha256"],
        "materialization_manifest_path": os.fspath(
            item.materialization_manifest_path
        ),
        "materialization_manifest_file_sha256": hashlib.sha256(
            materialization_raw
        ).hexdigest(),
        "materialization_input_fingerprint_sha256": materialization[
            "input_fingerprint_sha256"
        ],
        "source_catalog_sha256": catalog_hash,
        "source_ids": list(source_ids),
        "references": reference_provenance,
    }
    return song_payload, provenance


def _profile_content_sha256(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    provenance = dict(_mapping(payload.get("provenance"), "profile provenance"))
    provenance["profile_content_sha256"] = None
    payload["provenance"] = provenance
    return _canonical_sha256(payload)


def build_professional_montage_grammar_profile(
    songs: Sequence[MontageSongInput],
    *,
    artist: str = "IVE",
    config: MontageGrammarConfig | None = None,
) -> "ProfessionalMontageGrammarProfile":
    """Build a deterministic multi-song profile from immutable evidence."""

    settings = config or MontageGrammarConfig()
    if str(artist).strip().upper() != "IVE":
        raise ValueError("this visual montage grammar is deliberately IVE-only")
    inputs = tuple(sorted(songs, key=lambda item: item.song_id))
    if len(inputs) < settings.minimum_song_count:
        raise ValueError("not enough songs for a multi-song montage profile")
    if len({item.song_id for item in inputs}) != len(inputs):
        raise ValueError("montage profile song ids must be unique")
    song_payloads: list[dict[str, Any]] = []
    song_provenance: list[dict[str, Any]] = []
    for item in inputs:
        payload, provenance = _build_song(item, settings)
        song_payloads.append(payload)
        song_provenance.append(provenance)
    profile: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "analysis_basis": PROFILE_ANALYSIS_BASIS,
        "scope": {
            "artist": "IVE",
            "song_count": len(song_payloads),
            "reference_count": sum(
                song["reference_count"] for song in song_payloads
            ),
            "role_semantics": ROLE_SEMANTICS,
            "shot_scale_classification": {
                "wide": {"minimum_inclusive": 0.0, "maximum_exclusive": 0.22},
                "medium": {"minimum_inclusive": 0.22, "maximum_exclusive": 0.52},
                "close": {"minimum_inclusive": 0.52, "maximum_inclusive": 1.0},
                "missing_or_gate_failed": "unknown_never_coerced_to_wide",
            },
        },
        "configuration": settings.to_dict(),
        "provenance": {
            "content_hash_algorithm": (
                "sha256_of_canonical_json_with_profile_content_sha256_null"
            ),
            "profile_content_sha256": None,
            "generator": {
                "module": "crosscut.professional_montage_grammar",
                "module_version": MODULE_VERSION,
            },
            "token_schema_fingerprint": DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
            "shot_scale_source_feature_name": SHOT_SCALE_FEATURE_NAME,
            "shot_scale_source_feature_index": (
                DEFAULT_STAGE_EDIT_SCHEMA.source_feature_names.index(
                    SHOT_SCALE_FEATURE_NAME
                )
            ),
            "songs": song_provenance,
        },
        "songs": song_payloads,
        "aggregate_by_acoustic_role": _aggregate_roles(
            song_payloads,
            floor_quantile=settings.reference_floor_quantile,
        ),
        "evidence_limitations": {
            "wide_classification_is_shot_scale_only": True,
            "generic_wide_is_proven_group_formation": False,
            "person_count_evidence": (
                "identity_free_detected_and_formation_counts_when_unmasked"
            ),
            "person_count_threshold_used_to_assert_group_formation": False,
            "formation_identity_or_member_labels_present": False,
            "policy": (
                "retain_person_count_distributions_but_never_relabel_generic_wide_"
                "as_proven_formation"
            ),
            "zero_burst_count_proves_no_burst_in_reference_video": False,
            "burst_measurements_are_lower_bounds_when_supervision_has_unknown_gaps": True,
        },
        "distributional_policy": {
            "hand_authored_visual_quality_thresholds_present": False,
            "reference_floors_are_lower_distribution_quantiles": True,
            "reference_floor_quantile": settings.reference_floor_quantile,
            "zero_event_reference_edits_are_retained_in_rate_distributions": True,
            "missing_shot_scale_is_counted_as_unknown": True,
            "shot_size_count_fractions_are_count_only_audit_metrics": True,
            "shot_size_duration_fractions_use_observed_duration_only": True,
            "nonwide_duration_is_explicit_medium_plus_close_never_one_minus_wide": True,
            "consecutive_wide_runs_are_maximal_contiguous_known_wide_shots_within_role": True,
            "single_wide_shot_is_a_consecutive_wide_run": True,
            "reference_maximum_wide_run_is_zero_when_no_wide_is_observed": True,
            "source_switch_pairs_require_contiguous_known_outgoing_and_incoming_shots": True,
            "section_entries_use_exact_token_role_and_section_progress_reset": True,
            "phrase_entries_use_exact_token_phrase_progress_reset": True,
            "short_bursts_use_each_reference_edits_own_median_shot_duration": True,
            "short_bursts_require_multi_source_switch_connectivity": True,
            "general_nonwide_bursts_require_multiple_shots_and_at_least_one_source_switch": True,
        },
    }
    profile["provenance"]["profile_content_sha256"] = _profile_content_sha256(
        profile
    )
    return ProfessionalMontageGrammarProfile.from_mapping(profile)


def _assert_equivalent(expected: object, actual: object, name: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise MontageGrammarEvidenceError(f"{name} object does not match")
        for key in expected:
            _assert_equivalent(expected[key], actual[key], f"{name}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise MontageGrammarEvidenceError(f"{name} array does not match")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_equivalent(left, right, f"{name}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        left = _finite(expected, name)
        right = _finite(actual, name)
        if not math.isclose(left, right, abs_tol=1e-10):
            raise MontageGrammarEvidenceError(f"{name} numeric value does not match")
        return
    if expected != actual:
        raise MontageGrammarEvidenceError(f"{name} value does not match")


def _validate_reference_metric(reference: Mapping[str, Any]) -> None:
    _exact_keys(
        reference,
        {
            "reference_id",
            "shot_count",
            "known_neighbor_pair_count",
            "contiguous_known_boundary_count",
            "unknown_gap_between_known_shots_count",
            "observed_shot_scale_count",
            "unknown_shot_scale_count",
            "short_shot_reference_median_duration_seconds",
            "by_acoustic_role",
        },
        "reference metric",
    )
    if not str(reference.get("reference_id") or "").strip():
        raise MontageGrammarEvidenceError("reference metric id is blank")
    shot_count_declared = _integer(reference["shot_count"], "shot_count")
    neighbor_count = _integer(
        reference["known_neighbor_pair_count"],
        "known_neighbor_pair_count",
    )
    contiguous_count = _integer(
        reference["contiguous_known_boundary_count"],
        "contiguous_known_boundary_count",
    )
    gap_count = _integer(
        reference["unknown_gap_between_known_shots_count"],
        "unknown_gap_between_known_shots_count",
    )
    if (
        neighbor_count != max(0, shot_count_declared - 1)
        or contiguous_count + gap_count != neighbor_count
    ):
        raise MontageGrammarEvidenceError("reference known-gap coverage mismatch")
    roles = _mapping(reference.get("by_acoustic_role"), "by_acoustic_role")
    if set(roles) != set(ROLE_NAMES):
        raise MontageGrammarEvidenceError("reference acoustic roles mismatch")
    observed_total = 0
    unknown_total = 0
    shot_total = 0
    for role in ROLE_NAMES:
        metric = _mapping(roles[role], f"{role} metric")
        if set(metric) != set(_empty_role_metric(role)):
            raise MontageGrammarEvidenceError(f"{role} metric schema mismatch")
        if metric.get("role") != role:
            raise MontageGrammarEvidenceError("role metric identity mismatch")
        shot_count = _integer(metric["shot_count"], "role shot_count")
        observed = _integer(
            metric["observed_shot_scale_count"],
            "observed_shot_scale_count",
        )
        unknown = _integer(
            metric["unknown_shot_scale_count"],
            "unknown_shot_scale_count",
        )
        if min(shot_count, observed, unknown) < 0 or observed + unknown != shot_count:
            raise MontageGrammarEvidenceError("role shot-scale counts mismatch")
        size_counts = _mapping(metric["shot_size_counts"], "shot_size_counts")
        if set(size_counts) != set(SHOT_SIZE_NAMES):
            raise MontageGrammarEvidenceError("shot size count schema mismatch")
        if sum(_integer(size_counts[name], name) for name in SHOT_SIZE_NAMES) != observed:
            raise MontageGrammarEvidenceError("shot size counts mismatch observed count")
        reasons = _mapping(
            metric["unknown_shot_scale_reasons"],
            "unknown_shot_scale_reasons",
        )
        if set(reasons) != {
            "missing_mask",
            "source_unavailable",
            "hard_gate_failed",
        } or sum(_integer(value, "unknown reason count") for value in reasons.values()) != unknown:
            raise MontageGrammarEvidenceError("unknown shot-scale reasons mismatch")
        size_fractions = _mapping(
            metric["shot_size_fractions"],
            "shot_size_fractions",
        )
        if set(size_fractions) != set(SHOT_SIZE_NAMES):
            raise MontageGrammarEvidenceError("shot-size fraction schema mismatch")
        for size in SHOT_SIZE_NAMES:
            expected = size_counts[size] / observed if observed else None
            actual = size_fractions[size]
            if expected is None:
                if actual is not None:
                    raise MontageGrammarEvidenceError(
                        "shot-size fraction exists without observed scale"
                    )
            elif actual is None or not math.isclose(
                _finite(actual, "shot-size fraction"),
                expected,
                abs_tol=1e-12,
            ):
                raise MontageGrammarEvidenceError("shot-size fraction mismatch")

        role_duration = _finite(
            metric["role_duration_seconds"],
            "role_duration_seconds",
        )
        observed_duration = _finite(
            metric["observed_shot_scale_duration_seconds"],
            "observed_shot_scale_duration_seconds",
        )
        unknown_duration = _finite(
            metric["unknown_shot_scale_duration_seconds"],
            "unknown_shot_scale_duration_seconds",
        )
        if min(role_duration, observed_duration, unknown_duration) < 0.0:
            raise MontageGrammarEvidenceError(
                "shot-scale duration evidence cannot be negative"
            )
        if (shot_count > 0) != (role_duration > 0.0) or not math.isclose(
            observed_duration + unknown_duration,
            role_duration,
            abs_tol=1e-10,
        ):
            raise MontageGrammarEvidenceError(
                "role duration/shot-scale duration mismatch"
            )
        size_durations = _mapping(
            metric["shot_size_duration_seconds"],
            "shot_size_duration_seconds",
        )
        if set(size_durations) != set(SHOT_SIZE_NAMES):
            raise MontageGrammarEvidenceError(
                "shot-size duration schema mismatch"
            )
        typed_size_durations = {
            size: _finite(
                size_durations[size], f"{size} duration seconds"
            )
            for size in SHOT_SIZE_NAMES
        }
        if any(value < 0.0 for value in typed_size_durations.values()) or not (
            math.isclose(
                sum(typed_size_durations.values()),
                observed_duration,
                abs_tol=1e-10,
            )
        ):
            raise MontageGrammarEvidenceError(
                "shot-size durations mismatch observed duration"
            )
        duration_fractions = _mapping(
            metric["shot_size_duration_fractions"],
            "shot_size_duration_fractions",
        )
        if set(duration_fractions) != set(SHOT_SIZE_NAMES):
            raise MontageGrammarEvidenceError(
                "shot-size duration fraction schema mismatch"
            )
        for size in SHOT_SIZE_NAMES:
            expected = (
                typed_size_durations[size] / observed_duration
                if observed_duration > 0.0
                else None
            )
            actual = duration_fractions[size]
            if expected is None:
                if actual is not None:
                    raise MontageGrammarEvidenceError(
                        "shot-size duration fraction exists without observed duration"
                    )
            elif actual is None or not math.isclose(
                _finite(actual, "shot-size duration fraction"),
                expected,
                abs_tol=1e-12,
            ):
                raise MontageGrammarEvidenceError(
                    "shot-size duration fraction mismatch"
                )
        explicit_nonwide_duration = _finite(
            metric["nonwide_duration_seconds"],
            "nonwide_duration_seconds",
        )
        expected_nonwide_duration = (
            typed_size_durations["medium"] + typed_size_durations["close"]
        )
        if explicit_nonwide_duration < 0.0 or not math.isclose(
            explicit_nonwide_duration,
            expected_nonwide_duration,
            abs_tol=1e-10,
        ):
            raise MontageGrammarEvidenceError(
                "explicit nonwide duration is not medium plus close duration"
            )
        expected_nonwide_fraction = (
            explicit_nonwide_duration / observed_duration
            if observed_duration > 0.0
            else None
        )
        actual_nonwide_fraction = metric["nonwide_duration_fraction"]
        if expected_nonwide_fraction is None:
            if actual_nonwide_fraction is not None:
                raise MontageGrammarEvidenceError(
                    "nonwide duration fraction exists without observed duration"
                )
        elif actual_nonwide_fraction is None or not math.isclose(
            _finite(actual_nonwide_fraction, "nonwide_duration_fraction"),
            expected_nonwide_fraction,
            abs_tol=1e-12,
        ):
            raise MontageGrammarEvidenceError(
                "explicit nonwide duration fraction mismatch"
            )

        wide_run_count = _integer(
            metric["consecutive_wide_run_count"],
            "consecutive_wide_run_count",
        )
        wide_run_shot_counts = _array(
            metric["consecutive_wide_run_shot_counts"],
            "consecutive_wide_run_shot_counts",
        )
        wide_run_durations = _array(
            metric["consecutive_wide_run_durations_seconds"],
            "consecutive_wide_run_durations_seconds",
        )
        if (
            wide_run_count < 0
            or wide_run_count != len(wide_run_shot_counts)
            or wide_run_count != len(wide_run_durations)
        ):
            raise MontageGrammarEvidenceError("consecutive wide-run evidence mismatch")
        typed_run_shot_counts = tuple(
            _integer(value, "wide-run shot count")
            for value in wide_run_shot_counts
        )
        typed_run_durations = tuple(
            _finite(value, "wide-run duration") for value in wide_run_durations
        )
        if any(value < 1 for value in typed_run_shot_counts) or any(
            value <= 0.0 for value in typed_run_durations
        ):
            raise MontageGrammarEvidenceError("consecutive wide-run values are invalid")
        if sum(typed_run_shot_counts) != _integer(
            size_counts["wide"], "wide shot count"
        ) or not math.isclose(
            sum(typed_run_durations),
            typed_size_durations["wide"],
            abs_tol=1e-10,
        ):
            raise MontageGrammarEvidenceError(
                "consecutive wide runs do not partition observed wide shots"
            )
        maximum_wide_run = _finite(
            metric["maximum_consecutive_wide_run_seconds"],
            "maximum_consecutive_wide_run_seconds",
        )
        expected_maximum_wide_run = (
            max(typed_run_durations) if typed_run_durations else 0.0
        )
        if maximum_wide_run < 0.0 or not math.isclose(
            maximum_wide_run,
            expected_maximum_wide_run,
            abs_tol=1e-10,
        ):
            raise MontageGrammarEvidenceError(
                "maximum consecutive wide-run duration mismatch"
            )
        pair_counts = _mapping(
            metric["source_switch_pair_counts"],
            "source_switch_pair_counts",
        )
        if set(pair_counts) != set(PAIR_NAMES):
            raise MontageGrammarEvidenceError("source-switch pair schema mismatch")
        observed_pairs = _integer(
            metric["observed_source_switch_pair_count"],
            "observed_source_switch_pair_count",
        )
        if sum(_integer(pair_counts[name], name) for name in PAIR_NAMES) != observed_pairs:
            raise MontageGrammarEvidenceError("source-switch pair counts mismatch")
        switch_count = _integer(metric["source_switch_count"], "source_switch_count")
        unknown_pairs = _integer(
            metric["unknown_source_switch_pair_count"],
            "unknown_source_switch_pair_count",
        )
        if observed_pairs + unknown_pairs != switch_count:
            raise MontageGrammarEvidenceError("source-switch evidence counts mismatch")
        eligible = _integer(
            metric["eligible_source_boundary_count"],
            "eligible_source_boundary_count",
        )
        if not 0 <= switch_count <= eligible:
            raise MontageGrammarEvidenceError("source-switch counts exceed boundaries")
        expected_switch_fraction = switch_count / eligible if eligible else None
        actual_switch_fraction = metric["source_switch_fraction"]
        if expected_switch_fraction is None:
            if actual_switch_fraction is not None:
                raise MontageGrammarEvidenceError(
                    "source-switch fraction exists without boundaries"
                )
        elif actual_switch_fraction is None or not math.isclose(
            _finite(actual_switch_fraction, "source-switch fraction"),
            expected_switch_fraction,
            abs_tol=1e-12,
        ):
            raise MontageGrammarEvidenceError("source-switch fraction mismatch")
        pair_fractions = _mapping(
            metric["source_switch_pair_fractions"],
            "source_switch_pair_fractions",
        )
        if set(pair_fractions) != set(PAIR_NAMES):
            raise MontageGrammarEvidenceError("source-switch pair fraction schema mismatch")
        for pair in PAIR_NAMES:
            expected = pair_counts[pair] / observed_pairs if observed_pairs else None
            actual = pair_fractions[pair]
            if expected is None:
                if actual is not None:
                    raise MontageGrammarEvidenceError(
                        "source-switch pair fraction exists without observed pairs"
                    )
            elif actual is None or not math.isclose(
                _finite(actual, "source-switch pair fraction"),
                expected,
                abs_tol=1e-12,
            ):
                raise MontageGrammarEvidenceError(
                    "source-switch pair fraction mismatch"
                )
        derived_pair_fractions = {
            "wide_to_wide_source_switch_fraction": (
                pair_fractions["wide_to_wide"] if observed_pairs else None
            ),
            "source_switch_to_wide_fraction": (
                sum(
                    pair_fractions[f"{outgoing}_to_wide"]
                    for outgoing in SHOT_SIZE_NAMES
                )
                if observed_pairs
                else None
            ),
            "source_switch_from_wide_fraction": (
                sum(
                    pair_fractions[f"wide_to_{incoming}"]
                    for incoming in SHOT_SIZE_NAMES
                )
                if observed_pairs
                else None
            ),
            "source_switch_with_wide_fraction": (
                sum(
                    value
                    for pair, value in pair_fractions.items()
                    if pair.startswith("wide_to_") or pair.endswith("_to_wide")
                )
                if observed_pairs
                else None
            ),
        }
        for name, expected in derived_pair_fractions.items():
            actual = metric[name]
            if expected is None:
                if actual is not None:
                    raise MontageGrammarEvidenceError(
                        f"{name} exists without observed pairs"
                    )
            elif actual is None or not math.isclose(
                _finite(actual, name),
                expected,
                abs_tol=1e-12,
            ):
                raise MontageGrammarEvidenceError(f"{name} mismatch")

        section_count = _integer(metric["section_count"], "section_count")
        entry_observed = _integer(
            metric["observed_entry_anchor_count"],
            "observed_entry_anchor_count",
        )
        entry_wide = _integer(
            metric["entry_wide_anchor_count"],
            "entry_wide_anchor_count",
        )
        early_observed = _integer(
            metric["observed_early_anchor_count"],
            "observed_early_anchor_count",
        )
        early_wide = _integer(
            metric["early_wide_anchor_count"],
            "early_wide_anchor_count",
        )
        phrase_count = _integer(metric["phrase_count"], "phrase_count")
        phrase_observed = _integer(
            metric["observed_phrase_entry_anchor_count"],
            "observed_phrase_entry_anchor_count",
        )
        phrase_wide = _integer(
            metric["phrase_entry_wide_anchor_count"],
            "phrase_entry_wide_anchor_count",
        )
        if not (
            0 <= entry_wide <= entry_observed <= section_count
            and 0 <= early_wide <= early_observed <= section_count
            and 0 <= phrase_wide <= phrase_observed <= phrase_count
        ):
            raise MontageGrammarEvidenceError("wide anchor counts mismatch")
        for numerator, denominator, fraction_name in (
            (entry_wide, entry_observed, "entry_wide_anchor_fraction"),
            (early_wide, early_observed, "early_wide_anchor_fraction"),
            (
                phrase_wide,
                phrase_observed,
                "phrase_entry_wide_anchor_fraction",
            ),
        ):
            expected = numerator / denominator if denominator else None
            actual = metric[fraction_name]
            if expected is None:
                if actual is not None:
                    raise MontageGrammarEvidenceError(
                        f"{fraction_name} exists without observed anchors"
                    )
            elif actual is None or not math.isclose(
                _finite(actual, fraction_name),
                expected,
                abs_tol=1e-12,
            ):
                raise MontageGrammarEvidenceError(f"{fraction_name} mismatch")

        wide_count = _integer(size_counts["wide"], "wide shot count")
        for prefix in (
            "wide_shot_detected_person_count",
            "wide_shot_formation_person_count",
        ):
            count = _integer(
                metric[f"{prefix}_observed_count"],
                f"{prefix}_observed_count",
            )
            values = _array(
                metric[f"{prefix}_normalized_values"],
                f"{prefix}_normalized_values",
            )
            if count != len(values) or not 0 <= count <= wide_count:
                raise MontageGrammarEvidenceError(
                    f"{prefix} evidence count mismatch"
                )
            if any(
                not 0.0 <= _finite(value, f"{prefix} value") <= 1.0
                for value in values
            ):
                raise MontageGrammarEvidenceError(
                    f"{prefix} value lies outside [0, 1]"
                )

        for burst_name in (
            "short_nonwide_burst",
            "short_closeup_burst",
            "nonwide_multi_source_burst",
            "closeup_multi_source_burst",
        ):
            burst_count = _integer(
                metric[f"{burst_name}_count"],
                f"{burst_name}_count",
            )
            burst_shots = _array(
                metric[f"{burst_name}_shot_counts"],
                f"{burst_name}_shot_counts",
            )
            burst_durations = _array(
                metric[f"{burst_name}_durations_seconds"],
                f"{burst_name}_durations_seconds",
            )
            burst_switch_fractions = _array(
                metric[f"{burst_name}_source_switch_fractions"],
                f"{burst_name}_source_switch_fractions",
            )
            if burst_count != len(burst_shots) or burst_count != len(
                burst_durations
            ) or burst_count != len(burst_switch_fractions):
                raise MontageGrammarEvidenceError(f"{burst_name} evidence mismatch")
            if any(
                _integer(value, f"{burst_name} shot count") < 2
                for value in burst_shots
            ) or any(
                _finite(value, f"{burst_name} duration") <= 0.0
                for value in burst_durations
            ):
                raise MontageGrammarEvidenceError(f"{burst_name} values are invalid")
            if any(
                not 0.0 < _finite(value, f"{burst_name} switch fraction") <= 1.0
                for value in burst_switch_fractions
            ):
                raise MontageGrammarEvidenceError(
                    f"{burst_name} switch fraction is invalid"
                )
            expected_rate = (
                burst_count * 60.0 / role_duration if role_duration > 0.0 else None
            )
            actual_rate = metric[f"{burst_name}_rate_per_minute"]
            if expected_rate is None:
                if actual_rate is not None:
                    raise MontageGrammarEvidenceError(
                        f"{burst_name} rate exists without role duration"
                    )
            elif actual_rate is None or not math.isclose(
                _finite(actual_rate, f"{burst_name} rate"),
                expected_rate,
                abs_tol=1e-10,
            ):
                raise MontageGrammarEvidenceError(f"{burst_name} rate mismatch")
        shot_total += shot_count
        observed_total += observed
        unknown_total += unknown
    if shot_total != shot_count_declared:
        raise MontageGrammarEvidenceError("reference shot count mismatch")
    if observed_total != reference["observed_shot_scale_count"]:
        raise MontageGrammarEvidenceError("reference observed-scale count mismatch")
    if unknown_total != reference["unknown_shot_scale_count"]:
        raise MontageGrammarEvidenceError("reference unknown-scale count mismatch")


@dataclass(frozen=True, slots=True)
class ProfessionalMontageGrammarProfile:
    """Strictly validated, content-addressed montage grammar evidence."""

    _payload: Mapping[str, Any]
    source_path: Path | None = None
    file_sha256: str | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ProfessionalMontageGrammarProfile":
        payload = deepcopy(dict(value))
        cls._validate(payload)
        return cls(payload)

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        *,
        verify_input_hashes: bool = True,
    ) -> "ProfessionalMontageGrammarProfile":
        source = Path(path).expanduser().resolve()
        payload, raw = _read_json(source, "montage grammar profile")
        cls._validate(payload)
        result = cls(
            payload,
            source_path=source,
            file_sha256=hashlib.sha256(raw).hexdigest(),
        )
        if verify_input_hashes:
            provenance = _mapping(payload["provenance"], "profile provenance")
            song_inputs = tuple(
                MontageSongInput(
                    song_id=str(item["song_id"]),
                    feature_manifest_path=Path(str(item["feature_manifest_path"])),
                    materialization_manifest_path=Path(
                        str(item["materialization_manifest_path"])
                    ),
                )
                for item in _array(provenance.get("songs"), "provenance songs")
            )
            rebuilt = build_professional_montage_grammar_profile(
                song_inputs,
                artist=str(_mapping(payload["scope"], "scope")["artist"]),
                config=MontageGrammarConfig.from_mapping(
                    _mapping(payload["configuration"], "configuration")
                ),
            )
            _assert_equivalent(payload, rebuilt.to_dict(), "rebuilt profile")
        return result

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        _exact_keys(
            payload,
            {
                "schema_version",
                "analysis_basis",
                "scope",
                "configuration",
                "provenance",
                "songs",
                "aggregate_by_acoustic_role",
                "evidence_limitations",
                "distributional_policy",
            },
            "montage grammar profile",
        )
        if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
            raise MontageGrammarEvidenceError("profile schema version mismatch")
        if payload.get("analysis_basis") != PROFILE_ANALYSIS_BASIS:
            raise MontageGrammarEvidenceError("profile analysis basis mismatch")
        settings = MontageGrammarConfig.from_mapping(
            _mapping(payload["configuration"], "configuration")
        )
        scope = _mapping(payload["scope"], "scope")
        _exact_keys(
            scope,
            {
                "artist",
                "song_count",
                "reference_count",
                "role_semantics",
                "shot_scale_classification",
            },
            "scope",
        )
        if scope.get("artist") != "IVE" or scope.get("role_semantics") != ROLE_SEMANTICS:
            raise MontageGrammarEvidenceError("profile scope is not IVE acoustic roles")
        expected_classification = {
            "wide": {"minimum_inclusive": 0.0, "maximum_exclusive": 0.22},
            "medium": {"minimum_inclusive": 0.22, "maximum_exclusive": 0.52},
            "close": {"minimum_inclusive": 0.52, "maximum_inclusive": 1.0},
            "missing_or_gate_failed": "unknown_never_coerced_to_wide",
        }
        if scope.get("shot_scale_classification") != expected_classification:
            raise MontageGrammarEvidenceError("shot-scale classification mismatch")
        songs = _array(payload["songs"], "profile songs")
        if len(songs) < settings.minimum_song_count or scope.get("song_count") != len(songs):
            raise MontageGrammarEvidenceError("profile song count mismatch")
        song_ids: set[str] = set()
        reference_count = 0
        for raw_song in songs:
            song = _mapping(raw_song, "profile song")
            _exact_keys(
                song,
                {
                    "song_id",
                    "reference_count",
                    "source_count",
                    "shot_count",
                    "known_neighbor_pair_count",
                    "contiguous_known_boundary_count",
                    "unknown_gap_between_known_shots_count",
                    "observed_shot_scale_count",
                    "unknown_shot_scale_count",
                    "references",
                    "aggregate_by_acoustic_role",
                },
                "profile song",
            )
            song_id = str(song.get("song_id") or "")
            if not song_id or song_id in song_ids:
                raise MontageGrammarEvidenceError("profile song ids are invalid")
            song_ids.add(song_id)
            references = _array(song.get("references"), "song references")
            if len(references) != song.get("reference_count") or len(references) < (
                settings.minimum_references_per_song
            ):
                raise MontageGrammarEvidenceError("song reference count mismatch")
            if song.get("source_count", 0) < settings.minimum_sources_per_song:
                raise MontageGrammarEvidenceError("song source count is too small")
            for reference in references:
                _validate_reference_metric(_mapping(reference, "reference metric"))
            if sum(reference["shot_count"] for reference in references) != song["shot_count"]:
                raise MontageGrammarEvidenceError("song shot count mismatch")
            for count_name in (
                "known_neighbor_pair_count",
                "contiguous_known_boundary_count",
                "unknown_gap_between_known_shots_count",
            ):
                if sum(reference[count_name] for reference in references) != song[
                    count_name
                ]:
                    raise MontageGrammarEvidenceError(
                        f"song {count_name} mismatch"
                    )
            if sum(
                reference["observed_shot_scale_count"] for reference in references
            ) != song["observed_shot_scale_count"]:
                raise MontageGrammarEvidenceError("song observed-scale count mismatch")
            if sum(
                reference["unknown_shot_scale_count"] for reference in references
            ) != song["unknown_shot_scale_count"]:
                raise MontageGrammarEvidenceError("song unknown-scale count mismatch")
            expected_song_aggregate = _aggregate_roles(
                (song,),
                floor_quantile=settings.reference_floor_quantile,
            )
            _assert_equivalent(
                expected_song_aggregate,
                song["aggregate_by_acoustic_role"],
                f"{song_id} aggregate_by_acoustic_role",
            )
            reference_count += len(references)
        if scope.get("reference_count") != reference_count:
            raise MontageGrammarEvidenceError("scope reference count mismatch")
        provenance = _mapping(payload["provenance"], "profile provenance")
        _exact_keys(
            provenance,
            {
                "content_hash_algorithm",
                "profile_content_sha256",
                "generator",
                "token_schema_fingerprint",
                "shot_scale_source_feature_name",
                "shot_scale_source_feature_index",
                "songs",
            },
            "profile provenance",
        )
        if provenance.get("content_hash_algorithm") != (
            "sha256_of_canonical_json_with_profile_content_sha256_null"
        ):
            raise MontageGrammarEvidenceError("profile hash algorithm mismatch")
        if provenance.get("token_schema_fingerprint") != (
            DEFAULT_STAGE_EDIT_SCHEMA.fingerprint
        ):
            raise MontageGrammarEvidenceError("profile token schema mismatch")
        if provenance.get("shot_scale_source_feature_name") != SHOT_SCALE_FEATURE_NAME:
            raise MontageGrammarEvidenceError("profile shot-scale feature mismatch")
        if provenance.get("shot_scale_source_feature_index") != (
            DEFAULT_STAGE_EDIT_SCHEMA.source_feature_names.index(
                SHOT_SCALE_FEATURE_NAME
            )
        ):
            raise MontageGrammarEvidenceError("profile shot-scale index mismatch")
        profile_hash = provenance.get("profile_content_sha256")
        if not _is_sha256(profile_hash) or profile_hash != _profile_content_sha256(payload):
            raise MontageGrammarEvidenceError("profile content SHA-256 mismatch")
        provenance_songs = _array(provenance.get("songs"), "provenance songs")
        if [item.get("song_id") for item in provenance_songs] != [
            item.get("song_id") for item in songs
        ]:
            raise MontageGrammarEvidenceError("profile song provenance order mismatch")
        generator = _mapping(provenance.get("generator"), "profile generator")
        if dict(generator) != {
            "module": "crosscut.professional_montage_grammar",
            "module_version": MODULE_VERSION,
        }:
            raise MontageGrammarEvidenceError("profile generator mismatch")
        song_metrics_by_id = {str(item["song_id"]): item for item in songs}
        for raw_song_provenance in provenance_songs:
            song_provenance = _mapping(
                raw_song_provenance,
                "song provenance",
            )
            _exact_keys(
                song_provenance,
                {
                    "song_id",
                    "feature_manifest_path",
                    "feature_manifest_file_sha256",
                    "feature_manifest_content_sha256",
                    "materialization_manifest_path",
                    "materialization_manifest_file_sha256",
                    "materialization_input_fingerprint_sha256",
                    "source_catalog_sha256",
                    "source_ids",
                    "references",
                },
                "song provenance",
            )
            song_id = str(song_provenance["song_id"])
            song_metric = song_metrics_by_id[song_id]
            for path_name in (
                "feature_manifest_path",
                "materialization_manifest_path",
            ):
                if not Path(str(song_provenance[path_name])).is_absolute():
                    raise MontageGrammarEvidenceError(
                        f"{path_name} must be absolute"
                    )
            for hash_name in (
                "feature_manifest_file_sha256",
                "feature_manifest_content_sha256",
                "materialization_manifest_file_sha256",
                "materialization_input_fingerprint_sha256",
                "source_catalog_sha256",
            ):
                if not _is_sha256(song_provenance[hash_name]):
                    raise MontageGrammarEvidenceError(
                        f"{hash_name} is invalid"
                    )
            source_ids = _array(song_provenance["source_ids"], "source ids")
            if (
                len(source_ids) != song_metric["source_count"]
                or len(set(source_ids)) != len(source_ids)
                or source_ids != sorted(source_ids)
            ):
                raise MontageGrammarEvidenceError("song provenance sources mismatch")
            reference_provenance = _array(
                song_provenance["references"],
                "reference provenance",
            )
            metric_reference_ids = [
                reference["reference_id"] for reference in song_metric["references"]
            ]
            if [item.get("reference_id") for item in reference_provenance] != (
                metric_reference_ids
            ):
                raise MontageGrammarEvidenceError(
                    "reference provenance order mismatch"
                )
            for raw_reference_provenance in reference_provenance:
                reference = _mapping(
                    raw_reference_provenance,
                    "reference provenance",
                )
                _exact_keys(
                    reference,
                    {
                        "reference_id",
                        "supervision_path",
                        "supervision_file_sha256",
                        "snapshot_path",
                        "snapshot_file_sha256",
                        "snapshot_content_sha256",
                        "snapshot_sequence_id",
                        "token_count",
                    },
                    "reference provenance",
                )
                if not Path(str(reference["supervision_path"])).is_absolute() or not Path(
                    str(reference["snapshot_path"])
                ).is_absolute():
                    raise MontageGrammarEvidenceError(
                        "reference provenance paths must be absolute"
                    )
                for hash_name in (
                    "supervision_file_sha256",
                    "snapshot_file_sha256",
                    "snapshot_content_sha256",
                ):
                    if not _is_sha256(reference[hash_name]):
                        raise MontageGrammarEvidenceError(
                            f"reference {hash_name} is invalid"
                        )
                if _integer(reference["token_count"], "token_count") < 1:
                    raise MontageGrammarEvidenceError("token count must be positive")

        expected_limitations = {
            "wide_classification_is_shot_scale_only": True,
            "generic_wide_is_proven_group_formation": False,
            "person_count_evidence": (
                "identity_free_detected_and_formation_counts_when_unmasked"
            ),
            "person_count_threshold_used_to_assert_group_formation": False,
            "formation_identity_or_member_labels_present": False,
            "policy": (
                "retain_person_count_distributions_but_never_relabel_generic_wide_"
                "as_proven_formation"
            ),
            "zero_burst_count_proves_no_burst_in_reference_video": False,
            "burst_measurements_are_lower_bounds_when_supervision_has_unknown_gaps": True,
        }
        if payload.get("evidence_limitations") != expected_limitations:
            raise MontageGrammarEvidenceError("evidence limitations mismatch")
        expected_distributional_policy = {
            "hand_authored_visual_quality_thresholds_present": False,
            "reference_floors_are_lower_distribution_quantiles": True,
            "reference_floor_quantile": settings.reference_floor_quantile,
            "zero_event_reference_edits_are_retained_in_rate_distributions": True,
            "missing_shot_scale_is_counted_as_unknown": True,
            "shot_size_count_fractions_are_count_only_audit_metrics": True,
            "shot_size_duration_fractions_use_observed_duration_only": True,
            "nonwide_duration_is_explicit_medium_plus_close_never_one_minus_wide": True,
            "consecutive_wide_runs_are_maximal_contiguous_known_wide_shots_within_role": True,
            "single_wide_shot_is_a_consecutive_wide_run": True,
            "reference_maximum_wide_run_is_zero_when_no_wide_is_observed": True,
            "source_switch_pairs_require_contiguous_known_outgoing_and_incoming_shots": True,
            "section_entries_use_exact_token_role_and_section_progress_reset": True,
            "phrase_entries_use_exact_token_phrase_progress_reset": True,
            "short_bursts_use_each_reference_edits_own_median_shot_duration": True,
            "short_bursts_require_multi_source_switch_connectivity": True,
            "general_nonwide_bursts_require_multiple_shots_and_at_least_one_source_switch": True,
        }
        if payload.get("distributional_policy") != expected_distributional_policy:
            raise MontageGrammarEvidenceError("distributional policy mismatch")
        expected_aggregate = _aggregate_roles(
            songs,
            floor_quantile=settings.reference_floor_quantile,
        )
        _assert_equivalent(
            expected_aggregate,
            payload["aggregate_by_acoustic_role"],
            "aggregate_by_acoustic_role",
        )

    @property
    def profile_content_sha256(self) -> str:
        return str(self._payload["provenance"]["profile_content_sha256"])

    @property
    def song_count(self) -> int:
        return int(self._payload["scope"]["song_count"])

    @property
    def reference_count(self) -> int:
        return int(self._payload["scope"]["reference_count"])

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self._payload))

    def write_json(self, path: str | os.PathLike[str]) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        restored = ProfessionalMontageGrammarProfile.from_file(
            destination,
            verify_input_hashes=False,
        )
        if restored.profile_content_sha256 != self.profile_content_sha256:
            raise MontageGrammarEvidenceError("written profile failed strict round-trip")
        return destination


__all__ = [
    "APPLIED_TRANSFORM_SEGMENT_COUNT_BASIS",
    "BOUNDARY_GRAMMAR_ANALYSIS_BASIS",
    "BOUNDARY_GRAMMAR_SCHEMA_VERSION",
    "EditorialTransformEvidence",
    "HIGHLIGHT_BURST_EXEMPLAR_CUTS_PER_MINUTE",
    "IntroducedBoundaryContinuityEvidence",
    "MEDIUM_MAX_EXCLUSIVE",
    "MontageSectionWindow",
    "MontageGrammarConfig",
    "MontageGrammarEvidenceError",
    "MontageSongInput",
    "PROFILE_ANALYSIS_BASIS",
    "PROFILE_SCHEMA_VERSION",
    "ProfessionalBoundaryGrammarAssessment",
    "ProfessionalBoundaryGrammarEvidence",
    "ProfessionalBoundaryGrammarPolicy",
    "ProfessionalMontageGrammarProfile",
    "ROLE_NAMES",
    "SectionBoundaryAssessment",
    "SHOT_SIZE_NAMES",
    "WIDE_MAX_EXCLUSIVE",
    "assess_professional_boundary_grammar",
    "build_professional_montage_grammar_profile",
]
