"""Immutable, strictly causal per-source shooting-grammar evidence.

This module deliberately stops at an inference-safe scalar artifact.  Upstream
analyzers may measure framing, camera motion, composition, people, and native
cuts, but pixels, frames, descriptors, and identity embeddings are neither
accepted nor serialized here.  Per-source temporal filtering consumes only the
current observation and state produced by earlier observations.  An appended
future observation therefore cannot alter any existing record.

The StageEdit token schema is intentionally not imported.  The named adapter at
the bottom of this module is the reviewed hand-off contract for a later schema
integration without coupling this artifact to the current Transformer version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


SHOOTING_GRAMMAR_SCHEMA_VERSION = 1
SHOOTING_GRAMMAR_ARTIFACT_KIND = "causal_shooting_grammar"
SHOOTING_GRAMMAR_ANALYSIS_BASIS = (
    "per_source_strictly_past_current_causal_shooting_grammar_v1"
)
SHOOTING_GRAMMAR_PRODUCER = "crosscut.causal_shooting_grammar_v1"
SHOOTING_GRAMMAR_CONTENT_HASH_BASIS = (
    "sha256_canonical_json_with_provenance.artifact_content_sha256_null"
)
STAGE_EDIT_SHOOTING_GRAMMAR_ADAPTER = (
    "stage_edit_named_causal_shooting_grammar_features_v1"
)

# The order is part of the adapter contract.  Every value has exactly one bit
# in the same-position missing mask; ``True`` means unavailable and the stored
# numeric placeholder is then zero.
STAGE_EDIT_SHOOTING_GRAMMAR_FEATURE_NAMES = (
    "shot_closeup_probability",
    "shot_medium_probability",
    "shot_wide_probability",
    "camera_pan_velocity",
    "camera_tilt_velocity",
    "camera_zoom_in_probability",
    "camera_zoom_out_probability",
    "camera_static_probability",
    "camera_roll_degrees_normalized",
    "subject_headroom_normalized",
    "subject_lead_room_normalized",
    "detected_person_count_normalized",
    "formation_pose_coverage",
    "native_cut_proximity",
    "shooting_grammar_confidence",
)

STAGE_EDIT_PERSON_COUNT_NORMALIZER = 6
MAX_DETECTED_PERSON_COUNT = 64
MAX_FORMATION_PERSON_COUNT = 64

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROVENANCE_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_PROBABILITY_SUM_TOLERANCE = 1e-6
_STABLE_DECIMALS = 12
_SHOT_STATES = frozenset({"closeup", "medium", "wide", "unknown", "missing"})
_ZOOM_STATES = frozenset({"in", "out", "static", "unknown", "missing"})
_PAN_DIRECTIONS = frozenset({"left", "right", "static", "missing"})
_TILT_DIRECTIONS = frozenset({"up", "down", "static", "missing"})
_SIGNED_STAGE_EDIT_FEATURES = frozenset(
    {
        "camera_pan_velocity",
        "camera_tilt_velocity",
        "camera_roll_degrees_normalized",
    }
)

_STORAGE_POLICY: Mapping[str, bool] = {
    "raw_frames_persisted": False,
    "pixels_persisted": False,
    "visual_descriptors_persisted": False,
    "identity_or_appearance_embeddings_persisted": False,
    "scalar_features_and_state_only": True,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _optional_bounded(
    value: object | None,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    number = _finite(value, name)
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return number


def _optional_count(value: object | None, name: str, maximum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer or None")
    integer = int(value)
    if integer < 0 or integer > maximum:
        raise ValueError(f"{name} must be in [0, {maximum}]")
    return integer


def _stable_float(value: float) -> float:
    rounded = round(float(value), _STABLE_DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def _clean_source_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("shooting-grammar source_id must be a string")
    source_id = value.strip()
    if not source_id or len(source_id) > 160:
        raise ValueError("shooting-grammar source_id must be a bounded string")
    if any(ord(character) < 32 or ord(character) == 127 for character in source_id):
        raise ValueError("shooting-grammar source_id cannot contain control characters")
    return source_id


def _validated_probability_group(
    values: Sequence[object | None],
    name: str,
) -> tuple[float, float, float] | None:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three probabilities")
    present = tuple(value is not None for value in values)
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"{name} probabilities must be all present or all missing")
    normalized = tuple(
        _optional_bounded(value, f"{name}[{index}]", minimum=0.0, maximum=1.0)
        for index, value in enumerate(values)
    )
    assert all(value is not None for value in normalized)
    result = tuple(float(value) for value in normalized)
    if abs(sum(result) - 1.0) > _PROBABILITY_SUM_TOLERANCE:
        raise ValueError(f"{name} probabilities must sum to one")
    return (result[0], result[1], result[2])


def _probability_state(
    probabilities: tuple[float, float, float] | None,
    labels: tuple[str, str, str],
    *,
    tie_tolerance: float,
) -> str:
    if probabilities is None:
        return "missing"
    ranked = sorted(
        enumerate(probabilities),
        key=lambda item: (-item[1], item[0]),
    )
    if ranked[0][1] - ranked[1][1] <= tie_tolerance:
        return "unknown"
    return labels[ranked[0][0]]


def _pan_direction(value: float | None, threshold: float) -> str:
    if value is None:
        return "missing"
    if abs(value) <= threshold:
        return "static"
    return "left" if value < 0.0 else "right"


def _tilt_direction(value: float | None, threshold: float) -> str:
    if value is None:
        return "missing"
    if abs(value) <= threshold:
        return "static"
    # Match normalized screen coordinates: negative y is up, positive y down.
    return "up" if value < 0.0 else "down"


@dataclass(frozen=True, slots=True)
class CausalShootingGrammarSmoothing:
    """Deterministic causal filtering and classification parameters."""

    ema_alpha: float = 0.5
    maximum_smoothing_gap_seconds: float = 1.0
    direction_static_threshold: float = 0.025
    probability_tie_tolerance: float = 1e-9
    native_cut_proximity_window_seconds: float = 0.5

    def __post_init__(self) -> None:
        alpha = _finite(self.ema_alpha, "ema_alpha")
        gap = _finite(
            self.maximum_smoothing_gap_seconds,
            "maximum_smoothing_gap_seconds",
        )
        threshold = _finite(
            self.direction_static_threshold,
            "direction_static_threshold",
        )
        tie = _finite(self.probability_tie_tolerance, "probability_tie_tolerance")
        window = _finite(
            self.native_cut_proximity_window_seconds,
            "native_cut_proximity_window_seconds",
        )
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if gap <= 0.0:
            raise ValueError("maximum_smoothing_gap_seconds must be positive")
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("direction_static_threshold must be in [0, 1]")
        if tie < 0.0 or tie >= 0.5:
            raise ValueError("probability_tie_tolerance must be in [0, 0.5)")
        if window <= 0.0:
            raise ValueError("native_cut_proximity_window_seconds must be positive")
        threshold = 0.0 if threshold == 0.0 else threshold
        tie = 0.0 if tie == 0.0 else tie
        object.__setattr__(self, "ema_alpha", alpha)
        object.__setattr__(self, "maximum_smoothing_gap_seconds", gap)
        object.__setattr__(self, "direction_static_threshold", threshold)
        object.__setattr__(self, "probability_tie_tolerance", tie)
        object.__setattr__(self, "native_cut_proximity_window_seconds", window)

    def to_dict(self) -> dict[str, float]:
        return {
            "ema_alpha": self.ema_alpha,
            "maximum_smoothing_gap_seconds": self.maximum_smoothing_gap_seconds,
            "direction_static_threshold": self.direction_static_threshold,
            "probability_tie_tolerance": self.probability_tie_tolerance,
            "native_cut_proximity_window_seconds": (
                self.native_cut_proximity_window_seconds
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "CausalShootingGrammarSmoothing":
        expected = {
            "ema_alpha",
            "maximum_smoothing_gap_seconds",
            "direction_static_threshold",
            "probability_tie_tolerance",
            "native_cut_proximity_window_seconds",
        }
        if set(payload) != expected:
            raise ValueError("shooting-grammar smoothing schema does not match")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class ShootingGrammarObservation:
    """One analyzer observation available at a synchronized decision time.

    ``internal_native_cut_since_previous`` is an interval fact, not proximity
    to a future boundary.  ``True`` means at least one native cut was observed
    after the preceding sample and through the current sample; ``False`` means
    the complete interval was checked and contains none; ``None`` means the
    detector evidence for that interval is unavailable.
    """

    source_id: str
    master_time_seconds: float
    shot_closeup_probability: float | None = None
    shot_medium_probability: float | None = None
    shot_wide_probability: float | None = None
    camera_pan_velocity: float | None = None
    camera_tilt_velocity: float | None = None
    camera_zoom_in_probability: float | None = None
    camera_zoom_out_probability: float | None = None
    camera_static_probability: float | None = None
    camera_roll_degrees_normalized: float | None = None
    subject_headroom_normalized: float | None = None
    subject_lead_room_normalized: float | None = None
    detected_person_count: int | None = None
    formation_person_count: int | None = None
    internal_native_cut_since_previous: bool | None = None
    shooting_grammar_confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _clean_source_id(self.source_id))
        timestamp = _finite(self.master_time_seconds, "master_time_seconds")
        if timestamp < 0.0:
            raise ValueError("master_time_seconds must be non-negative")
        timestamp = 0.0 if timestamp == 0.0 else timestamp
        object.__setattr__(self, "master_time_seconds", timestamp)

        shot = _validated_probability_group(
            (
                self.shot_closeup_probability,
                self.shot_medium_probability,
                self.shot_wide_probability,
            ),
            "shot-scale",
        )
        zoom = _validated_probability_group(
            (
                self.camera_zoom_in_probability,
                self.camera_zoom_out_probability,
                self.camera_static_probability,
            ),
            "camera-zoom",
        )
        for field_name, value in zip(
            (
                "shot_closeup_probability",
                "shot_medium_probability",
                "shot_wide_probability",
            ),
            shot or (None, None, None),
            strict=True,
        ):
            object.__setattr__(self, field_name, value)
        for field_name, value in zip(
            (
                "camera_zoom_in_probability",
                "camera_zoom_out_probability",
                "camera_static_probability",
            ),
            zoom or (None, None, None),
            strict=True,
        ):
            object.__setattr__(self, field_name, value)

        for field_name in ("camera_pan_velocity", "camera_tilt_velocity"):
            object.__setattr__(
                self,
                field_name,
                _optional_bounded(
                    getattr(self, field_name),
                    field_name,
                    minimum=-1.0,
                    maximum=1.0,
                ),
            )
        object.__setattr__(
            self,
            "camera_roll_degrees_normalized",
            _optional_bounded(
                self.camera_roll_degrees_normalized,
                "camera_roll_degrees_normalized",
                minimum=-1.0,
                maximum=1.0,
            ),
        )
        for field_name in (
            "subject_headroom_normalized",
            "subject_lead_room_normalized",
            "shooting_grammar_confidence",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_bounded(
                    getattr(self, field_name),
                    field_name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        subject_count = _optional_count(
            self.detected_person_count,
            "detected_person_count",
            MAX_DETECTED_PERSON_COUNT,
        )
        formation_count = _optional_count(
            self.formation_person_count,
            "formation_person_count",
            MAX_FORMATION_PERSON_COUNT,
        )
        if (
            subject_count is not None
            and formation_count is not None
            and formation_count > subject_count
        ):
            raise ValueError(
                "formation_person_count cannot exceed detected_person_count"
            )
        object.__setattr__(self, "detected_person_count", subject_count)
        object.__setattr__(self, "formation_person_count", formation_count)
        if self.internal_native_cut_since_previous is not None and not isinstance(
            self.internal_native_cut_since_previous,
            bool,
        ):
            raise ValueError(
                "internal_native_cut_since_previous must be boolean or None"
            )


@dataclass(frozen=True, slots=True)
class ShootingGrammarProvenance:
    """Hash identities and analyzer contracts, without paths or media data."""

    upstream_sha256: tuple[tuple[str, str], ...]
    analyzer_fingerprint_sha256: str
    contracts: tuple[str, ...] = ()
    producer: str = SHOOTING_GRAMMAR_PRODUCER

    def __post_init__(self) -> None:
        normalized: list[tuple[str, str]] = []
        seen: set[str] = set()
        for raw_name, raw_digest in self.upstream_sha256:
            if not isinstance(raw_name, str) or not isinstance(raw_digest, str):
                raise ValueError(
                    "shooting-grammar provenance identities must be strings"
                )
            name = raw_name.strip()
            digest = raw_digest.strip()
            if _PROVENANCE_NAME.fullmatch(name) is None:
                raise ValueError("shooting-grammar provenance name is invalid")
            if name in seen:
                raise ValueError("shooting-grammar provenance names must be unique")
            if _SHA256.fullmatch(digest) is None:
                raise ValueError("shooting-grammar upstream identity must be SHA-256")
            normalized.append((name, digest))
            seen.add(name)
        if not normalized:
            raise ValueError("shooting-grammar provenance needs an upstream identity")
        if not isinstance(self.analyzer_fingerprint_sha256, str):
            raise ValueError("shooting-grammar analyzer fingerprint must be SHA-256")
        analyzer = self.analyzer_fingerprint_sha256.strip()
        if _SHA256.fullmatch(analyzer) is None:
            raise ValueError("shooting-grammar analyzer fingerprint must be SHA-256")
        if any(not isinstance(value, str) for value in self.contracts):
            raise ValueError("shooting-grammar provenance contracts must be strings")
        contracts = tuple(sorted({value.strip() for value in self.contracts}))
        if any(not value or len(value) > 200 for value in contracts):
            raise ValueError("shooting-grammar provenance contracts are invalid")
        if (
            not isinstance(self.producer, str)
            or self.producer.strip() != SHOOTING_GRAMMAR_PRODUCER
        ):
            raise ValueError("shooting-grammar producer contract is unsupported")
        object.__setattr__(self, "upstream_sha256", tuple(sorted(normalized)))
        object.__setattr__(self, "analyzer_fingerprint_sha256", analyzer)
        object.__setattr__(self, "contracts", contracts)
        object.__setattr__(self, "producer", SHOOTING_GRAMMAR_PRODUCER)

    def to_dict(self, *, artifact_content_sha256: str | None) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "upstream_sha256": dict(self.upstream_sha256),
            "analyzer_fingerprint_sha256": self.analyzer_fingerprint_sha256,
            "contracts": list(self.contracts),
            "content_hash_basis": SHOOTING_GRAMMAR_CONTENT_HASH_BASIS,
            "artifact_content_sha256": artifact_content_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> tuple["ShootingGrammarProvenance", str]:
        expected = {
            "producer",
            "upstream_sha256",
            "analyzer_fingerprint_sha256",
            "contracts",
            "content_hash_basis",
            "artifact_content_sha256",
        }
        if set(payload) != expected:
            raise ValueError("shooting-grammar provenance schema does not match")
        if payload.get("content_hash_basis") != SHOOTING_GRAMMAR_CONTENT_HASH_BASIS:
            raise ValueError("shooting-grammar content-hash basis is unsupported")
        upstream = payload.get("upstream_sha256")
        contracts = payload.get("contracts")
        digest = payload.get("artifact_content_sha256")
        if not isinstance(upstream, Mapping) or not isinstance(contracts, list):
            raise ValueError("shooting-grammar provenance values are invalid")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError("shooting-grammar artifact content hash is invalid")
        provenance = cls(
            upstream_sha256=tuple(
                (name, value) for name, value in upstream.items()
            ),
            analyzer_fingerprint_sha256=payload.get("analyzer_fingerprint_sha256"),
            contracts=tuple(contracts),
            producer=payload.get("producer"),
        )
        return provenance, digest


@dataclass(frozen=True, slots=True)
class CausalShootingGrammarRecord:
    """Smoothed scalar state for exactly one source and master timestamp."""

    source_id: str
    master_time_seconds: float
    shot_closeup_probability: float | None
    shot_medium_probability: float | None
    shot_wide_probability: float | None
    shot_scale_state: str
    camera_pan_velocity: float | None
    camera_pan_direction: str
    camera_tilt_velocity: float | None
    camera_tilt_direction: str
    camera_zoom_in_probability: float | None
    camera_zoom_out_probability: float | None
    camera_static_probability: float | None
    camera_zoom_state: str
    camera_roll_degrees_normalized: float | None
    subject_headroom_normalized: float | None
    subject_lead_room_normalized: float | None
    detected_person_count: int | None
    formation_person_count: int | None
    native_cut_proximity: float | None
    shooting_grammar_confidence: float | None
    missing_mask: tuple[bool, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _clean_source_id(self.source_id))
        timestamp = _finite(self.master_time_seconds, "master_time_seconds")
        if timestamp < 0.0:
            raise ValueError("master_time_seconds must be non-negative")
        timestamp = 0.0 if timestamp == 0.0 else timestamp
        object.__setattr__(self, "master_time_seconds", timestamp)
        shot = _validated_probability_group(
            (
                self.shot_closeup_probability,
                self.shot_medium_probability,
                self.shot_wide_probability,
            ),
            "smoothed shot-scale",
        )
        zoom = _validated_probability_group(
            (
                self.camera_zoom_in_probability,
                self.camera_zoom_out_probability,
                self.camera_static_probability,
            ),
            "smoothed camera-zoom",
        )
        for field_name, value in zip(
            (
                "shot_closeup_probability",
                "shot_medium_probability",
                "shot_wide_probability",
            ),
            shot or (None, None, None),
            strict=True,
        ):
            object.__setattr__(self, field_name, value)
        for field_name, value in zip(
            (
                "camera_zoom_in_probability",
                "camera_zoom_out_probability",
                "camera_static_probability",
            ),
            zoom or (None, None, None),
            strict=True,
        ):
            object.__setattr__(self, field_name, value)
        for field_name in ("camera_pan_velocity", "camera_tilt_velocity"):
            object.__setattr__(
                self,
                field_name,
                _optional_bounded(
                    getattr(self, field_name),
                    field_name,
                    minimum=-1.0,
                    maximum=1.0,
                ),
            )
        object.__setattr__(
            self,
            "camera_roll_degrees_normalized",
            _optional_bounded(
                self.camera_roll_degrees_normalized,
                "camera_roll_degrees_normalized",
                minimum=-1.0,
                maximum=1.0,
            ),
        )
        for field_name in (
            "subject_headroom_normalized",
            "subject_lead_room_normalized",
            "native_cut_proximity",
            "shooting_grammar_confidence",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_bounded(
                    getattr(self, field_name),
                    field_name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        subject_count = _optional_count(
            self.detected_person_count,
            "detected_person_count",
            MAX_DETECTED_PERSON_COUNT,
        )
        formation_count = _optional_count(
            self.formation_person_count,
            "formation_person_count",
            MAX_FORMATION_PERSON_COUNT,
        )
        if (
            subject_count is not None
            and formation_count is not None
            and formation_count > subject_count
        ):
            raise ValueError(
                "formation_person_count cannot exceed detected_person_count"
            )
        object.__setattr__(self, "detected_person_count", subject_count)
        object.__setattr__(self, "formation_person_count", formation_count)
        shot_state = str(self.shot_scale_state).strip()
        zoom_state = str(self.camera_zoom_state).strip()
        pan_direction = str(self.camera_pan_direction).strip()
        tilt_direction = str(self.camera_tilt_direction).strip()
        if shot_state not in _SHOT_STATES:
            raise ValueError("shot_scale_state is unsupported")
        if zoom_state not in _ZOOM_STATES:
            raise ValueError("camera_zoom_state is unsupported")
        if pan_direction not in _PAN_DIRECTIONS:
            raise ValueError("camera_pan_direction is unsupported")
        if tilt_direction not in _TILT_DIRECTIONS:
            raise ValueError("camera_tilt_direction is unsupported")
        if (shot is None) != (shot_state == "missing"):
            raise ValueError("shot-scale state and probability missingness disagree")
        if (zoom is None) != (zoom_state == "missing"):
            raise ValueError("camera-zoom state and probability missingness disagree")
        if (self.camera_pan_velocity is None) != (pan_direction == "missing"):
            raise ValueError("pan direction and velocity missingness disagree")
        if (self.camera_tilt_velocity is None) != (tilt_direction == "missing"):
            raise ValueError("tilt direction and velocity missingness disagree")
        object.__setattr__(self, "shot_scale_state", shot_state)
        object.__setattr__(self, "camera_zoom_state", zoom_state)
        object.__setattr__(self, "camera_pan_direction", pan_direction)
        object.__setattr__(self, "camera_tilt_direction", tilt_direction)
        _values, mask = _stage_edit_values_and_mask(self)
        object.__setattr__(self, "missing_mask", mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "master_time_seconds": self.master_time_seconds,
            "shot_closeup_probability": self.shot_closeup_probability,
            "shot_medium_probability": self.shot_medium_probability,
            "shot_wide_probability": self.shot_wide_probability,
            "shot_scale_state": self.shot_scale_state,
            "camera_pan_velocity": self.camera_pan_velocity,
            "camera_pan_direction": self.camera_pan_direction,
            "camera_tilt_velocity": self.camera_tilt_velocity,
            "camera_tilt_direction": self.camera_tilt_direction,
            "camera_zoom_in_probability": self.camera_zoom_in_probability,
            "camera_zoom_out_probability": self.camera_zoom_out_probability,
            "camera_static_probability": self.camera_static_probability,
            "camera_zoom_state": self.camera_zoom_state,
            "camera_roll_degrees_normalized": (
                self.camera_roll_degrees_normalized
            ),
            "subject_headroom_normalized": self.subject_headroom_normalized,
            "subject_lead_room_normalized": self.subject_lead_room_normalized,
            "detected_person_count": self.detected_person_count,
            "formation_person_count": self.formation_person_count,
            "native_cut_proximity": self.native_cut_proximity,
            "shooting_grammar_confidence": self.shooting_grammar_confidence,
            "missing_mask": dict(
                zip(
                    STAGE_EDIT_SHOOTING_GRAMMAR_FEATURE_NAMES,
                    self.missing_mask,
                    strict=True,
                )
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CausalShootingGrammarRecord":
        expected = {
            "source_id",
            "master_time_seconds",
            "shot_closeup_probability",
            "shot_medium_probability",
            "shot_wide_probability",
            "shot_scale_state",
            "camera_pan_velocity",
            "camera_pan_direction",
            "camera_tilt_velocity",
            "camera_tilt_direction",
            "camera_zoom_in_probability",
            "camera_zoom_out_probability",
            "camera_static_probability",
            "camera_zoom_state",
            "camera_roll_degrees_normalized",
            "subject_headroom_normalized",
            "subject_lead_room_normalized",
            "detected_person_count",
            "formation_person_count",
            "native_cut_proximity",
            "shooting_grammar_confidence",
            "missing_mask",
        }
        if set(payload) != expected:
            raise ValueError("shooting-grammar record schema does not match")
        raw_mask = payload.get("missing_mask")
        if not isinstance(raw_mask, Mapping) or set(raw_mask) != set(
            STAGE_EDIT_SHOOTING_GRAMMAR_FEATURE_NAMES
        ):
            raise ValueError("shooting-grammar missing-mask schema does not match")
        serialized_mask = tuple(
            raw_mask[name] for name in STAGE_EDIT_SHOOTING_GRAMMAR_FEATURE_NAMES
        )
        if any(not isinstance(value, bool) for value in serialized_mask):
            raise ValueError("shooting-grammar missing mask must contain booleans")
        record = cls(
            **{name: value for name, value in payload.items() if name != "missing_mask"}
        )
        if serialized_mask != record.missing_mask:
            raise ValueError("shooting-grammar missing mask disagrees with values")
        return record


def _stage_edit_values_and_mask(
    record: CausalShootingGrammarRecord,
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    formation_coverage: float | None
    if (
        record.detected_person_count is None
        or record.formation_person_count is None
    ):
        formation_coverage = None
    elif record.detected_person_count == 0:
        formation_coverage = 0.0
    else:
        formation_coverage = (
            record.formation_person_count / record.detected_person_count
        )
    detected_normalized = (
        min(STAGE_EDIT_PERSON_COUNT_NORMALIZER, record.detected_person_count)
        / STAGE_EDIT_PERSON_COUNT_NORMALIZER
        if record.detected_person_count is not None
        else None
    )
    optional_values: tuple[float | None, ...] = (
        record.shot_closeup_probability,
        record.shot_medium_probability,
        record.shot_wide_probability,
        record.camera_pan_velocity,
        record.camera_tilt_velocity,
        record.camera_zoom_in_probability,
        record.camera_zoom_out_probability,
        record.camera_static_probability,
        record.camera_roll_degrees_normalized,
        record.subject_headroom_normalized,
        record.subject_lead_room_normalized,
        detected_normalized,
        formation_coverage,
        record.native_cut_proximity,
        record.shooting_grammar_confidence,
    )
    missing = tuple(value is None for value in optional_values)
    values = tuple(0.0 if value is None else float(value) for value in optional_values)
    return values, missing


@dataclass(frozen=True, slots=True)
class StageEditNamedShootingGrammarFeatures:
    """Immutable numeric hand-off for a future StageEdit schema revision."""

    source_id: str
    master_time_seconds: float
    values: tuple[float, ...]
    missing_mask: tuple[bool, ...]
    feature_names: tuple[str, ...] = STAGE_EDIT_SHOOTING_GRAMMAR_FEATURE_NAMES
    adapter: str = STAGE_EDIT_SHOOTING_GRAMMAR_ADAPTER

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _clean_source_id(self.source_id))
        timestamp = _finite(self.master_time_seconds, "master_time_seconds")
        if timestamp < 0.0:
            raise ValueError("master_time_seconds must be non-negative")
        timestamp = 0.0 if timestamp == 0.0 else timestamp
        object.__setattr__(self, "master_time_seconds", timestamp)
        if self.feature_names != STAGE_EDIT_SHOOTING_GRAMMAR_FEATURE_NAMES:
            raise ValueError("shooting-grammar adapter feature names changed")
        if self.adapter != STAGE_EDIT_SHOOTING_GRAMMAR_ADAPTER:
            raise ValueError("shooting-grammar adapter identity changed")
        values = tuple(
            _finite(value, f"shooting-grammar adapter value {index}")
            for index, value in enumerate(self.values)
        )
        missing_mask = tuple(self.missing_mask)
        if len(values) != len(self.feature_names) or len(
            missing_mask
        ) != len(self.feature_names):
            raise ValueError("shooting-grammar adapter dimensions do not match")
        if any(not isinstance(value, bool) for value in missing_mask):
            raise ValueError("shooting-grammar adapter mask must be boolean")
        for name, value, missing in zip(
            self.feature_names,
            values,
            missing_mask,
            strict=True,
        ):
            if missing:
                continue
            minimum = -1.0 if name in _SIGNED_STAGE_EDIT_FEATURES else 0.0
            if value < minimum or value > 1.0:
                raise ValueError(
                    f"shooting-grammar adapter feature {name} is out of range"
                )
        for indexes in ((0, 1, 2), (5, 6, 7)):
            group_mask = tuple(missing_mask[index] for index in indexes)
            if len(set(group_mask)) != 1:
                raise ValueError(
                    "shooting-grammar adapter probabilities must be all "
                    "present or all missing"
                )
            if not group_mask[0] and abs(
                sum(values[index] for index in indexes) - 1.0
            ) > _PROBABILITY_SUM_TOLERANCE:
                raise ValueError(
                    "shooting-grammar adapter probabilities must sum to one"
                )
        if any(
            value != 0.0
            for value, missing in zip(
                values,
                missing_mask,
                strict=True,
            )
            if missing
        ):
            raise ValueError(
                "missing shooting-grammar values must use zero placeholder"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "missing_mask", missing_mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "source_id": self.source_id,
            "master_time_seconds": self.master_time_seconds,
            "feature_names": list(self.feature_names),
            "values": list(self.values),
            "missing_mask": list(self.missing_mask),
        }

    def named_values(self) -> dict[str, tuple[float, bool]]:
        return {
            name: (value, missing)
            for name, value, missing in zip(
                self.feature_names,
                self.values,
                self.missing_mask,
                strict=True,
            )
        }


def adapt_causal_shooting_grammar_for_stage_edit(
    record: CausalShootingGrammarRecord,
) -> StageEditNamedShootingGrammarFeatures:
    """Return the reviewed named numeric columns and their 1:1 missing mask."""

    if not isinstance(record, CausalShootingGrammarRecord):
        raise TypeError("StageEdit shooting-grammar adapter needs a causal record")
    values, missing = _stage_edit_values_and_mask(record)
    if missing != record.missing_mask:
        raise ValueError("shooting-grammar record missing mask is inconsistent")
    return StageEditNamedShootingGrammarFeatures(
        source_id=record.source_id,
        master_time_seconds=record.master_time_seconds,
        values=values,
        missing_mask=missing,
    )


@dataclass(slots=True)
class _SourceCausalState:
    last_time_seconds: float | None = None
    scalar_state: dict[str, float] = field(default_factory=dict)
    probability_state: dict[str, tuple[float, float, float]] = field(
        default_factory=dict
    )
    last_native_cut_time_seconds: float | None = None
    native_cut_unknown_since_seconds: float | None = None

    def reset_smoothing(self) -> None:
        self.scalar_state.clear()
        self.probability_state.clear()


def _smooth_scalar(
    state: _SourceCausalState,
    name: str,
    current: float | None,
    *,
    alpha: float,
) -> float | None:
    if current is None:
        return None
    previous = state.scalar_state.get(name)
    smoothed = (
        current
        if previous is None
        else alpha * current + (1.0 - alpha) * previous
    )
    result = _stable_float(smoothed)
    state.scalar_state[name] = result
    return result


def _smooth_probabilities(
    state: _SourceCausalState,
    name: str,
    current: tuple[float, float, float] | None,
    *,
    alpha: float,
) -> tuple[float, float, float] | None:
    if current is None:
        return None
    previous = state.probability_state.get(name)
    if previous is None:
        result = tuple(_stable_float(value) for value in current)
    else:
        result = tuple(
            _stable_float(alpha * value + (1.0 - alpha) * old)
            for value, old in zip(current, previous, strict=True)
        )
    typed = (result[0], result[1], result[2])
    state.probability_state[name] = typed
    return typed


def _native_cut_proximity(
    observation: ShootingGrammarObservation,
    state: _SourceCausalState,
    config: CausalShootingGrammarSmoothing,
) -> float | None:
    timestamp = observation.master_time_seconds
    flag = observation.internal_native_cut_since_previous
    if flag is None:
        state.last_native_cut_time_seconds = None
        state.native_cut_unknown_since_seconds = timestamp
        return None
    if flag:
        state.last_native_cut_time_seconds = timestamp
        state.native_cut_unknown_since_seconds = None
        return 1.0
    unknown_since = state.native_cut_unknown_since_seconds
    if unknown_since is not None:
        # No future boundary is consulted.  Once the entire uncertainty is
        # older than the finite proximity window, any possible missed cut has
        # zero proximity and the state can safely become known again.
        if timestamp - unknown_since < config.native_cut_proximity_window_seconds:
            return None
        state.native_cut_unknown_since_seconds = None
        state.last_native_cut_time_seconds = None
    last_cut = state.last_native_cut_time_seconds
    if last_cut is None:
        return 0.0
    age = timestamp - last_cut
    if age < 0.0:
        raise ValueError("native-cut state cannot refer to a future timestamp")
    return _stable_float(
        max(0.0, 1.0 - age / config.native_cut_proximity_window_seconds)
    )


def _build_record(
    observation: ShootingGrammarObservation,
    state: _SourceCausalState,
    config: CausalShootingGrammarSmoothing,
) -> CausalShootingGrammarRecord:
    timestamp = observation.master_time_seconds
    if state.last_time_seconds is not None:
        if timestamp <= state.last_time_seconds:
            raise ValueError("per-source shooting-grammar times must be unique")
        if (
            timestamp - state.last_time_seconds
            > config.maximum_smoothing_gap_seconds
        ):
            state.reset_smoothing()
    # A confirmed native cut is a state discontinuity.  Missing cut-detector
    # coverage also makes smoothing continuity unprovable, so both reset the
    # prior filter before the current observation is incorporated.
    if observation.internal_native_cut_since_previous is not False:
        state.reset_smoothing()

    shot = _smooth_probabilities(
        state,
        "shot_scale",
        _validated_probability_group(
            (
                observation.shot_closeup_probability,
                observation.shot_medium_probability,
                observation.shot_wide_probability,
            ),
            "shot-scale",
        ),
        alpha=config.ema_alpha,
    )
    zoom = _smooth_probabilities(
        state,
        "camera_zoom",
        _validated_probability_group(
            (
                observation.camera_zoom_in_probability,
                observation.camera_zoom_out_probability,
                observation.camera_static_probability,
            ),
            "camera-zoom",
        ),
        alpha=config.ema_alpha,
    )
    pan = _smooth_scalar(
        state,
        "camera_pan_velocity",
        observation.camera_pan_velocity,
        alpha=config.ema_alpha,
    )
    tilt = _smooth_scalar(
        state,
        "camera_tilt_velocity",
        observation.camera_tilt_velocity,
        alpha=config.ema_alpha,
    )
    roll = _smooth_scalar(
        state,
        "camera_roll_degrees_normalized",
        observation.camera_roll_degrees_normalized,
        alpha=config.ema_alpha,
    )
    headroom = _smooth_scalar(
        state,
        "subject_headroom_normalized",
        observation.subject_headroom_normalized,
        alpha=config.ema_alpha,
    )
    lead_room = _smooth_scalar(
        state,
        "subject_lead_room_normalized",
        observation.subject_lead_room_normalized,
        alpha=config.ema_alpha,
    )
    confidence = _smooth_scalar(
        state,
        "shooting_grammar_confidence",
        observation.shooting_grammar_confidence,
        alpha=config.ema_alpha,
    )
    native_cut_proximity = _native_cut_proximity(observation, state, config)
    state.last_time_seconds = timestamp

    return CausalShootingGrammarRecord(
        source_id=observation.source_id,
        master_time_seconds=timestamp,
        shot_closeup_probability=shot[0] if shot is not None else None,
        shot_medium_probability=shot[1] if shot is not None else None,
        shot_wide_probability=shot[2] if shot is not None else None,
        shot_scale_state=_probability_state(
            shot,
            ("closeup", "medium", "wide"),
            tie_tolerance=config.probability_tie_tolerance,
        ),
        camera_pan_velocity=pan,
        camera_pan_direction=_pan_direction(
            pan,
            config.direction_static_threshold,
        ),
        camera_tilt_velocity=tilt,
        camera_tilt_direction=_tilt_direction(
            tilt,
            config.direction_static_threshold,
        ),
        camera_zoom_in_probability=zoom[0] if zoom is not None else None,
        camera_zoom_out_probability=zoom[1] if zoom is not None else None,
        camera_static_probability=zoom[2] if zoom is not None else None,
        camera_zoom_state=_probability_state(
            zoom,
            ("in", "out", "static"),
            tie_tolerance=config.probability_tie_tolerance,
        ),
        camera_roll_degrees_normalized=roll,
        subject_headroom_normalized=headroom,
        subject_lead_room_normalized=lead_room,
        detected_person_count=observation.detected_person_count,
        formation_person_count=observation.formation_person_count,
        native_cut_proximity=native_cut_proximity,
        shooting_grammar_confidence=confidence,
    )


def _temporal_policy(config: CausalShootingGrammarSmoothing) -> dict[str, Any]:
    return {
        "context": "strictly_past_and_current_observations_per_source",
        "future_frames_used": False,
        "future_native_cuts_used": False,
        "input_order": "strict_master_time_then_source_id",
        "missing_observations_imputed": False,
        "native_cut_proximity_basis": (
            "current_or_most_recent_observed_internal_native_cut_only"
        ),
        "native_cut_or_unknown_resets_smoothing": True,
        "smoothing": config.to_dict(),
    }


def _feature_adapter_contract() -> dict[str, Any]:
    return {
        "name": STAGE_EDIT_SHOOTING_GRAMMAR_ADAPTER,
        "feature_names": list(STAGE_EDIT_SHOOTING_GRAMMAR_FEATURE_NAMES),
        "missing_mask_semantics": "true_means_unavailable",
        "missing_numeric_placeholder": 0.0,
        "detected_person_count_normalizer": STAGE_EDIT_PERSON_COUNT_NORMALIZER,
        "formation_pose_coverage_basis": (
            "formation_person_count_divided_by_detected_person_count"
        ),
        "pan_sign_convention": "negative_left_positive_right",
        "tilt_sign_convention": "negative_up_positive_down",
        "probability_groups_sum_to_one_when_observed": True,
        "unknown_classification_sets_missing_mask": False,
        "numeric_ranges": {
            "probability_geometry_count_and_confidence": "[0,1]",
            "signed_camera_velocity_and_roll": "[-1,1]",
        },
    }


def _artifact_unsigned_payload(
    records: Sequence[CausalShootingGrammarRecord],
    provenance: ShootingGrammarProvenance,
    smoothing: CausalShootingGrammarSmoothing,
) -> dict[str, Any]:
    sources = tuple(sorted({record.source_id for record in records}))
    return {
        "schema_version": SHOOTING_GRAMMAR_SCHEMA_VERSION,
        "artifact_kind": SHOOTING_GRAMMAR_ARTIFACT_KIND,
        "analysis_basis": SHOOTING_GRAMMAR_ANALYSIS_BASIS,
        "temporal_policy": _temporal_policy(smoothing),
        "feature_adapter": _feature_adapter_contract(),
        "storage_policy": dict(_STORAGE_POLICY),
        "source_ids": list(sources),
        "source_count": len(sources),
        "record_count": len(records),
        "records": [record.to_dict() for record in records],
        "provenance": provenance.to_dict(artifact_content_sha256=None),
    }


@dataclass(frozen=True, slots=True)
class CausalShootingGrammarArtifact:
    """Self-validating content-addressed shooting-grammar artifact."""

    records: tuple[CausalShootingGrammarRecord, ...]
    provenance: ShootingGrammarProvenance
    smoothing: CausalShootingGrammarSmoothing
    artifact_content_sha256: str
    schema_version: int = SHOOTING_GRAMMAR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records:
            raise ValueError("shooting-grammar artifact requires at least one record")
        if any(
            not isinstance(record, CausalShootingGrammarRecord)
            for record in records
        ):
            raise TypeError("shooting-grammar records have an invalid type")
        if not isinstance(self.provenance, ShootingGrammarProvenance):
            raise TypeError("shooting-grammar provenance has an invalid type")
        if not isinstance(self.smoothing, CausalShootingGrammarSmoothing):
            raise TypeError("shooting-grammar smoothing has an invalid type")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != SHOOTING_GRAMMAR_SCHEMA_VERSION
        ):
            raise ValueError("unsupported shooting-grammar schema version")
        previous_key: tuple[float, str] | None = None
        for record in records:
            key = (record.master_time_seconds, record.source_id)
            if previous_key is not None and key <= previous_key:
                raise ValueError(
                    "shooting-grammar records must be strict master-time/source ordered"
                )
            previous_key = key
            shot = (
                (
                    record.shot_closeup_probability,
                    record.shot_medium_probability,
                    record.shot_wide_probability,
                )
                if record.shot_closeup_probability is not None
                else None
            )
            zoom = (
                (
                    record.camera_zoom_in_probability,
                    record.camera_zoom_out_probability,
                    record.camera_static_probability,
                )
                if record.camera_zoom_in_probability is not None
                else None
            )
            if record.shot_scale_state != _probability_state(
                shot,  # type: ignore[arg-type]
                ("closeup", "medium", "wide"),
                tie_tolerance=self.smoothing.probability_tie_tolerance,
            ):
                raise ValueError("shooting-grammar shot-scale state is inconsistent")
            if record.camera_zoom_state != _probability_state(
                zoom,  # type: ignore[arg-type]
                ("in", "out", "static"),
                tie_tolerance=self.smoothing.probability_tie_tolerance,
            ):
                raise ValueError("shooting-grammar zoom state is inconsistent")
            if record.camera_pan_direction != _pan_direction(
                record.camera_pan_velocity,
                self.smoothing.direction_static_threshold,
            ):
                raise ValueError("shooting-grammar pan state is inconsistent")
            if record.camera_tilt_direction != _tilt_direction(
                record.camera_tilt_velocity,
                self.smoothing.direction_static_threshold,
            ):
                raise ValueError("shooting-grammar tilt state is inconsistent")
        object.__setattr__(self, "records", records)
        digest = str(self.artifact_content_sha256).strip()
        if _SHA256.fullmatch(digest) is None:
            raise ValueError("shooting-grammar artifact content hash is invalid")
        unsigned = _artifact_unsigned_payload(
            records,
            self.provenance,
            self.smoothing,
        )
        if _canonical_sha256(unsigned) != digest:
            raise ValueError("shooting-grammar artifact content hash mismatch")
        object.__setattr__(self, "artifact_content_sha256", digest)

    @classmethod
    def build(
        cls,
        observations: Sequence[ShootingGrammarObservation],
        *,
        provenance: ShootingGrammarProvenance,
        smoothing: CausalShootingGrammarSmoothing = CausalShootingGrammarSmoothing(),
    ) -> "CausalShootingGrammarArtifact":
        items = tuple(observations)
        if not items:
            raise ValueError("shooting-grammar build requires observations")
        if any(not isinstance(item, ShootingGrammarObservation) for item in items):
            raise TypeError("shooting-grammar observations have an invalid type")
        previous_key: tuple[float, str] | None = None
        states: dict[str, _SourceCausalState] = {}
        records: list[CausalShootingGrammarRecord] = []
        for observation in items:
            key = (observation.master_time_seconds, observation.source_id)
            if previous_key is not None and key <= previous_key:
                raise ValueError(
                    "shooting-grammar observations must be strict "
                    "master-time/source ordered"
                )
            previous_key = key
            state = states.setdefault(observation.source_id, _SourceCausalState())
            records.append(_build_record(observation, state, smoothing))
        unsigned = _artifact_unsigned_payload(records, provenance, smoothing)
        return cls(
            records=tuple(records),
            provenance=provenance,
            smoothing=smoothing,
            artifact_content_sha256=_canonical_sha256(unsigned),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = _artifact_unsigned_payload(
            self.records,
            self.provenance,
            self.smoothing,
        )
        if _canonical_sha256(payload) != self.artifact_content_sha256:
            raise ValueError("shooting-grammar artifact mutated after hashing")
        payload["provenance"][
            "artifact_content_sha256"
        ] = self.artifact_content_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CausalShootingGrammarArtifact":
        expected = {
            "schema_version",
            "artifact_kind",
            "analysis_basis",
            "temporal_policy",
            "feature_adapter",
            "storage_policy",
            "source_ids",
            "source_count",
            "record_count",
            "records",
            "provenance",
        }
        if set(payload) != expected:
            raise ValueError("shooting-grammar artifact schema does not match")
        raw_schema_version = payload.get("schema_version")
        if (
            isinstance(raw_schema_version, bool)
            or not isinstance(raw_schema_version, int)
            or raw_schema_version != SHOOTING_GRAMMAR_SCHEMA_VERSION
        ):
            raise ValueError("unsupported shooting-grammar schema version")
        if payload.get("artifact_kind") != SHOOTING_GRAMMAR_ARTIFACT_KIND:
            raise ValueError("unsupported shooting-grammar artifact kind")
        if payload.get("analysis_basis") != SHOOTING_GRAMMAR_ANALYSIS_BASIS:
            raise ValueError("unsupported shooting-grammar analysis basis")
        raw_provenance = payload.get("provenance")
        raw_policy = payload.get("temporal_policy")
        raw_records = payload.get("records")
        if not isinstance(raw_provenance, Mapping):
            raise ValueError("shooting-grammar provenance is missing")
        if not isinstance(raw_policy, Mapping):
            raise ValueError("shooting-grammar temporal policy is missing")
        if not isinstance(raw_records, list):
            raise ValueError("shooting-grammar records must be an array")
        raw_smoothing = raw_policy.get("smoothing")
        if not isinstance(raw_smoothing, Mapping):
            raise ValueError("shooting-grammar smoothing policy is missing")
        smoothing = CausalShootingGrammarSmoothing.from_dict(raw_smoothing)
        if _canonical_json_bytes(dict(raw_policy)) != _canonical_json_bytes(
            _temporal_policy(smoothing)
        ):
            raise ValueError("shooting-grammar temporal policy is unsafe")
        if _canonical_json_bytes(payload.get("feature_adapter")) != (
            _canonical_json_bytes(_feature_adapter_contract())
        ):
            raise ValueError("shooting-grammar feature-adapter contract changed")
        if _canonical_json_bytes(payload.get("storage_policy")) != (
            _canonical_json_bytes(dict(_STORAGE_POLICY))
        ):
            raise ValueError("shooting-grammar storage policy is unsafe")
        for field_name in ("source_count", "record_count"):
            value = payload.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"shooting-grammar {field_name} must be a non-negative integer"
                )
        if not isinstance(payload.get("source_ids"), list):
            raise ValueError("shooting-grammar source_ids must be an array")
        provenance, digest = ShootingGrammarProvenance.from_dict(raw_provenance)
        records = tuple(
            CausalShootingGrammarRecord.from_dict(record)
            if isinstance(record, Mapping)
            else (_ for _ in ()).throw(
                ValueError("shooting-grammar record must be an object")
            )
            for record in raw_records
        )
        artifact = cls(
            records=records,
            provenance=provenance,
            smoothing=smoothing,
            artifact_content_sha256=digest,
            schema_version=raw_schema_version,
        )
        if _canonical_json_bytes(artifact.to_dict()) != _canonical_json_bytes(
            dict(payload)
        ):
            raise ValueError("shooting-grammar artifact derived fields disagree")
        return artifact

    def write_content_addressed(self, root: Path) -> Path:
        """Atomically publish at ``root/<prefix>/<content_sha256>.json``.

        A valid identical destination is reused.  Any different or invalid
        existing file is rejected and never replaced.
        """

        directory = Path(root).expanduser().resolve()
        shard = directory / self.artifact_content_sha256[:2]
        shard.mkdir(parents=True, exist_ok=True)
        destination = shard / f"{self.artifact_content_sha256}.json"
        expected_bytes = (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        if destination.exists() or destination.is_symlink():
            existing = self.load_content_addressed(
                destination,
                expected_content_sha256=self.artifact_content_sha256,
            )
            if existing.to_dict() != self.to_dict():
                raise FileExistsError(
                    f"immutable shooting-grammar artifact differs: {destination}"
                )
            return destination

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.artifact_content_sha256}.",
            suffix=".tmp",
            dir=shard,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(expected_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                existing = self.load_content_addressed(
                    destination,
                    expected_content_sha256=self.artifact_content_sha256,
                )
                if existing.to_dict() != self.to_dict():
                    raise FileExistsError(
                        "raced immutable shooting-grammar artifact differs: "
                        f"{destination}"
                    )
        finally:
            temporary.unlink(missing_ok=True)
        verified = self.load_content_addressed(
            destination,
            expected_content_sha256=self.artifact_content_sha256,
        )
        if verified.to_dict() != self.to_dict():
            raise RuntimeError(
                "published shooting-grammar artifact verification failed"
            )
        return destination

    @classmethod
    def load_content_addressed(
        cls,
        path: Path,
        *,
        expected_content_sha256: str | None = None,
    ) -> "CausalShootingGrammarArtifact":
        requested = Path(path).expanduser()
        if requested.is_symlink():
            raise ValueError(
                "content-addressed shooting-grammar path cannot be a symlink"
            )
        source = requested.resolve()
        if not source.exists() or not source.is_file() or source.is_symlink():
            raise ValueError("content-addressed shooting-grammar path is not a file")
        payload = _load_strict_json(source)
        if not isinstance(payload, Mapping):
            raise ValueError("shooting-grammar artifact root must be an object")
        artifact = cls.from_dict(payload)
        expected_name = f"{artifact.artifact_content_sha256}.json"
        if source.name != expected_name:
            raise ValueError("shooting-grammar filename is not content-addressed")
        if expected_content_sha256 is not None:
            expected = str(expected_content_sha256).strip()
            if _SHA256.fullmatch(expected) is None:
                raise ValueError("expected shooting-grammar hash must be SHA-256")
            if artifact.artifact_content_sha256 != expected:
                raise ValueError("shooting-grammar expected content hash mismatch")
        return artifact


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _load_strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_json_constant,
    )


def build_causal_shooting_grammar_artifact(
    observations: Sequence[ShootingGrammarObservation],
    *,
    provenance: ShootingGrammarProvenance,
    smoothing: CausalShootingGrammarSmoothing = CausalShootingGrammarSmoothing(),
) -> CausalShootingGrammarArtifact:
    return CausalShootingGrammarArtifact.build(
        observations,
        provenance=provenance,
        smoothing=smoothing,
    )


def write_causal_shooting_grammar_artifact(
    artifact: CausalShootingGrammarArtifact,
    root: Path,
) -> Path:
    if not isinstance(artifact, CausalShootingGrammarArtifact):
        raise TypeError("writer needs a causal shooting-grammar artifact")
    return artifact.write_content_addressed(root)


def load_causal_shooting_grammar_artifact(
    path: Path,
    *,
    expected_content_sha256: str | None = None,
) -> CausalShootingGrammarArtifact:
    return CausalShootingGrammarArtifact.load_content_addressed(
        path,
        expected_content_sha256=expected_content_sha256,
    )


__all__ = [
    "CausalShootingGrammarArtifact",
    "CausalShootingGrammarRecord",
    "CausalShootingGrammarSmoothing",
    "MAX_DETECTED_PERSON_COUNT",
    "MAX_FORMATION_PERSON_COUNT",
    "SHOOTING_GRAMMAR_ANALYSIS_BASIS",
    "SHOOTING_GRAMMAR_ARTIFACT_KIND",
    "SHOOTING_GRAMMAR_CONTENT_HASH_BASIS",
    "SHOOTING_GRAMMAR_PRODUCER",
    "SHOOTING_GRAMMAR_SCHEMA_VERSION",
    "STAGE_EDIT_PERSON_COUNT_NORMALIZER",
    "STAGE_EDIT_SHOOTING_GRAMMAR_ADAPTER",
    "STAGE_EDIT_SHOOTING_GRAMMAR_FEATURE_NAMES",
    "ShootingGrammarObservation",
    "ShootingGrammarProvenance",
    "StageEditNamedShootingGrammarFeatures",
    "adapt_causal_shooting_grammar_for_stage_edit",
    "build_causal_shooting_grammar_artifact",
    "load_causal_shooting_grammar_artifact",
    "write_causal_shooting_grammar_artifact",
]
