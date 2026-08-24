"""Hash-bound human-reference calibration for motion-accented close cuts.

The montage grammar profile records *which* source the human editor selected.
This module replays those immutable supervision/snapshot pairs through the
same reference adapter used by StageEdit and measures the exact outgoing and
incoming source observations at every confirmed, non-opening source switch.

Only build, chorus-like, and bridge-like switches involving a close shot are
included in the motion calibration.  The resulting recommendation is a
bounded soft preference.  It is deliberately not a hard gate and contains no
per-section event quota.  Same-token pose-embedding cosine is retained as a
low-coverage audit proxy; it is not described as temporal phase or motion
direction and is not enabled as a runtime threshold.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from .professional_montage_grammar import (
    MEDIUM_MAX_EXCLUSIVE,
    PROFILE_ANALYSIS_BASIS as MONTAGE_PROFILE_ANALYSIS_BASIS,
    PROFILE_SCHEMA_VERSION as MONTAGE_PROFILE_SCHEMA_VERSION,
    ProfessionalMontageGrammarProfile,
)
from .stage_edit_feature_materializer import StageEditFeatureSnapshot
from .stage_edit_reference_adapter import (
    ReferenceTrainingCorpus,
    load_reference_training_corpus,
)
from .stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
)
from .stage_edit_semimarkov_beam import (
    HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
    HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
    HIGHLIGHT_MOTION_CALIBRATION_CONTRACT,
    StageEditHighlightMotionCalibration,
    StageEditSourceContinuity,
    stage_edit_boundary_continuity_score,
)
from .stage_edit_source_pair_training import (
    STAGE_EDIT_SOURCE_PAIR_TRAINING_CONTRACT,
    STAGE_EDIT_SOURCE_PAIR_TRAINING_SCHEMA_VERSION,
    StageEditSourcePairTrainingCorpus,
    materialize_stage_edit_source_pair_training_corpus,
    stage_edit_source_pair_observation_from_candidate,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES,
    SourceCandidateToken,
)


PROFILE_SCHEMA_VERSION = 1
PROFILE_CONTRACT = HIGHLIGHT_MOTION_CALIBRATION_CONTRACT
PROFILE_ANALYSIS_BASIS = (
    "ive_20_human_reference_build_chorus_bridge_close_switch_motion_profile_v1"
)
PROFILE_CONTENT_HASH_ALGORITHM = (
    "sha256_of_canonical_json_with_profile_content_sha256_null"
)
EXPECTED_ARTIST = "IVE"
EXPECTED_SONG_COUNT = 4
EXPECTED_HUMAN_REFERENCE_COUNT = 20
EXPECTED_SOURCE_PAIR_CONTRIBUTING_REFERENCE_COUNT = 17
HIGHLIGHT_ROLES = ("build", "chorus_like", "bridge_like")
QUANTILE_PROBABILITIES = (0.10, 0.25, 0.50, 0.75)
QUANTILE_NAMES = ("p10", "p25", "p50", "p75")
QUANTILE_METHOD = "nearest_rank_empirical"
SOFT_PREFERENCE_THRESHOLD_QUANTILE = 0.10
POSE_PHASE_AUDIT_TARGET_QUANTILE = 0.50
MOTION_RECOMMENDATION_CONTRACT = (
    "both_moving_pose_phase_compatible_continuity_compatible_bounded_soft_v1"
)
SAME_TOKEN_POSE_SIMILARITY_SEMANTICS = (
    "identity_free_same_token_pose_embedding_cosine_proxy_"
    "not_temporal_phase_or_motion_direction"
)

_SHA256_LENGTH = 64
_ROLE_PROBABILITY_NAMES = (
    "intro_probability",
    "verse_probability",
    "prechorus_probability",
    "chorus_probability",
    "bridge_probability",
    "outro_probability",
)
_ROLE_NAMES = (
    "intro",
    "verse_like",
    "build",
    "chorus_like",
    "bridge_like",
    "outro",
)
_SAMPLING_KEYS = {
    "boundary_guard_seconds",
    "max_sources",
    "negative_samples_per_hold",
    "visual_working_width",
}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "contract",
    "analysis_basis",
    "scope",
    "sampling",
    "input_policy",
    "counts",
    "distributions",
    "recommendation",
    "evidence_limitations",
    "provenance",
}


class ProfessionalMotionCutProfileError(ValueError):
    """Raised when calibration evidence or a persisted profile is unsafe."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfessionalMotionCutProfileError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProfessionalMotionCutProfileError(f"{name} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ProfessionalMotionCutProfileError(
            f"{name} schema mismatch; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProfessionalMotionCutProfileError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _finite(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfessionalMotionCutProfileError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ProfessionalMotionCutProfileError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ProfessionalMotionCutProfileError(f"{name} is below {minimum}")
    if maximum is not None and number > maximum:
        raise ProfessionalMotionCutProfileError(f"{name} is above {maximum}")
    return number


def _sha256(value: object, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ProfessionalMotionCutProfileError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _name(value: object, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ProfessionalMotionCutProfileError(f"{name} cannot be blank")
    return result


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProfessionalMotionCutProfileError(
            f"profile is not canonical-JSON serializable: {exc}"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _profile_content_sha256(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    provenance = dict(_mapping(payload.get("provenance"), "profile provenance"))
    provenance["profile_content_sha256"] = None
    payload["provenance"] = provenance
    return _canonical_sha256(payload)


def _reject_constant(value: str) -> None:
    raise ProfessionalMotionCutProfileError(
        f"profile contains non-finite JSON number {value}"
    )


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProfessionalMotionCutProfileError(
                f"profile contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _stable_file_bytes(path: Path, name: str) -> bytes:
    source = Path(path).expanduser()
    if not source.is_absolute():
        raise ProfessionalMotionCutProfileError(f"{name} path must be absolute")
    try:
        before_path = source.lstat()
    except OSError as exc:
        raise ProfessionalMotionCutProfileError(
            f"could not stat {name} {source}: {exc}"
        ) from exc
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise ProfessionalMotionCutProfileError(
            f"{name} must be a regular non-symlink file: {source}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ProfessionalMotionCutProfileError(
            f"could not open {name} {source}: {exc}"
        ) from exc
    try:
        before_open = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = source.lstat()
    except OSError as exc:
        raise ProfessionalMotionCutProfileError(
            f"{name} disappeared while it was read: {source}"
        ) from exc

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if not (
        identity(before_path)
        == identity(before_open)
        == identity(after_open)
        == identity(after_path)
    ):
        raise ProfessionalMotionCutProfileError(f"{name} changed while it was read")
    raw = b"".join(chunks)
    if len(raw) != before_open.st_size:
        raise ProfessionalMotionCutProfileError(f"{name} size changed while read")
    return raw


def _decode_json(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except ProfessionalMotionCutProfileError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProfessionalMotionCutProfileError(
            f"could not decode {name}: {exc}"
        ) from exc
    return dict(_mapping(value, name))


def _verified_json(
    path: Path,
    expected_sha256: object,
    name: str,
) -> tuple[dict[str, Any], bytes]:
    expected = _sha256(expected_sha256, f"{name} expected SHA-256")
    raw = _stable_file_bytes(path, name)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ProfessionalMotionCutProfileError(f"{name} file SHA-256 mismatch")
    return _decode_json(raw, name), raw


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    clean = sorted(_finite(value, "distribution value") for value in values)
    if not clean:
        raise ProfessionalMotionCutProfileError(
            "nearest-rank quantile requires observed values"
        )
    probability = _finite(quantile, "quantile", minimum=0.0, maximum=1.0)
    if probability <= 0.0:
        return clean[0]
    index = max(0, math.ceil(probability * len(clean)) - 1)
    return clean[index]


@dataclass(frozen=True, slots=True)
class EmpiricalMotionDistribution:
    """Nearest-rank distribution with explicit observation support."""

    observed_switch_count: int
    minimum: float
    p10: float
    p25: float
    p50: float
    p75: float
    maximum: float
    minimum_allowed: float
    maximum_allowed: float

    def __post_init__(self) -> None:
        _integer(self.observed_switch_count, "observed_switch_count", minimum=1)
        lower = _finite(self.minimum_allowed, "minimum_allowed")
        upper = _finite(self.maximum_allowed, "maximum_allowed")
        if upper < lower:
            raise ProfessionalMotionCutProfileError(
                "distribution allowed range is inverted"
            )
        values = tuple(
            _finite(
                getattr(self, name),
                name,
                minimum=lower,
                maximum=upper,
            )
            for name in ("minimum", "p10", "p25", "p50", "p75", "maximum")
        )
        if any(right < left for left, right in zip(values, values[1:])):
            raise ProfessionalMotionCutProfileError(
                "empirical distribution values must be monotonic"
            )

    @property
    def quantiles(self) -> tuple[float, float, float, float]:
        return (self.p10, self.p25, self.p50, self.p75)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_switch_count": self.observed_switch_count,
            "quantile_method": QUANTILE_METHOD,
            "quantiles": {
                "p10": self.p10,
                "p25": self.p25,
                "p50": self.p50,
                "p75": self.p75,
            },
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    @classmethod
    def from_values(
        cls,
        values: Sequence[float],
        *,
        minimum_allowed: float,
        maximum_allowed: float,
    ) -> "EmpiricalMotionDistribution":
        clean = tuple(
            _finite(
                value,
                "distribution value",
                minimum=minimum_allowed,
                maximum=maximum_allowed,
            )
            for value in values
        )
        if not clean:
            raise ProfessionalMotionCutProfileError(
                "empirical distribution requires observed values"
            )
        return cls(
            observed_switch_count=len(clean),
            minimum=min(clean),
            p10=_nearest_rank(clean, 0.10),
            p25=_nearest_rank(clean, 0.25),
            p50=_nearest_rank(clean, 0.50),
            p75=_nearest_rank(clean, 0.75),
            maximum=max(clean),
            minimum_allowed=minimum_allowed,
            maximum_allowed=maximum_allowed,
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        name: str,
        minimum_allowed: float,
        maximum_allowed: float,
    ) -> "EmpiricalMotionDistribution":
        _exact_keys(
            value,
            {
                "observed_switch_count",
                "quantile_method",
                "quantiles",
                "minimum",
                "maximum",
            },
            name,
        )
        if value.get("quantile_method") != QUANTILE_METHOD:
            raise ProfessionalMotionCutProfileError(
                f"{name} quantile method is unsupported"
            )
        quantiles = _mapping(value.get("quantiles"), f"{name} quantiles")
        _exact_keys(quantiles, set(QUANTILE_NAMES), f"{name} quantiles")
        return cls(
            observed_switch_count=_integer(
                value.get("observed_switch_count"),
                f"{name} observed_switch_count",
                minimum=1,
            ),
            minimum=_finite(value.get("minimum"), f"{name} minimum"),
            p10=_finite(quantiles.get("p10"), f"{name} p10"),
            p25=_finite(quantiles.get("p25"), f"{name} p25"),
            p50=_finite(quantiles.get("p50"), f"{name} p50"),
            p75=_finite(quantiles.get("p75"), f"{name} p75"),
            maximum=_finite(value.get("maximum"), f"{name} maximum"),
            minimum_allowed=minimum_allowed,
            maximum_allowed=maximum_allowed,
        )


@dataclass(frozen=True, slots=True)
class MotionCutSoftPreferenceRecommendation:
    """Typed non-gating recommendation consumed by StageEdit integration."""

    both_moving_minimum_pose_motion: float
    continuity_compatible_minimum_score: float
    pose_phase_compatible_preferred_cosine: float
    pose_phase_runtime_pair_feature_available: bool = True
    pose_phase_hard_threshold: float | None = None
    sharpness_minimum: float | None = None
    hard_gate: bool = False
    minimum_required_switches_per_section: int | None = None
    threshold_quantile: float = SOFT_PREFERENCE_THRESHOLD_QUANTILE
    pose_phase_target_quantile: float = POSE_PHASE_AUDIT_TARGET_QUANTILE
    contract: str = MOTION_RECOMMENDATION_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != MOTION_RECOMMENDATION_CONTRACT:
            raise ProfessionalMotionCutProfileError(
                "motion recommendation contract mismatch"
            )
        _finite(
            self.both_moving_minimum_pose_motion,
            "both_moving_minimum_pose_motion",
            minimum=0.0,
            maximum=1.0,
        )
        _finite(
            self.continuity_compatible_minimum_score,
            "continuity_compatible_minimum_score",
            minimum=-1.0,
            maximum=0.0,
        )
        _finite(
            self.pose_phase_compatible_preferred_cosine,
            "pose_phase_compatible_preferred_cosine",
            minimum=-1.0,
            maximum=1.0,
        )
        if self.pose_phase_runtime_pair_feature_available is not True:
            raise ProfessionalMotionCutProfileError(
                "pose-phase pair feature must match the current exact-pair runtime"
            )
        if self.pose_phase_hard_threshold is not None:
            raise ProfessionalMotionCutProfileError(
                "low-coverage pose-phase evidence cannot be a hard threshold"
            )
        if self.sharpness_minimum is not None:
            raise ProfessionalMotionCutProfileError(
                "zero-support sharpness cannot supply a recommendation"
            )
        if self.hard_gate is not False:
            raise ProfessionalMotionCutProfileError(
                "motion-cut recommendation must remain a soft preference"
            )
        if self.minimum_required_switches_per_section is not None:
            raise ProfessionalMotionCutProfileError(
                "motion-cut recommendation must not impose a section quota"
            )
        if not math.isclose(
            _finite(self.threshold_quantile, "threshold_quantile"),
            SOFT_PREFERENCE_THRESHOLD_QUANTILE,
            abs_tol=1.0e-12,
        ):
            raise ProfessionalMotionCutProfileError(
                "soft preference threshold must use empirical P10"
            )
        if not math.isclose(
            _finite(self.pose_phase_target_quantile, "pose_phase_target_quantile"),
            POSE_PHASE_AUDIT_TARGET_QUANTILE,
            abs_tol=1.0e-12,
        ):
            raise ProfessionalMotionCutProfileError(
                "pose-phase audit target must use empirical P50"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "mode": "bounded_joint_soft_preference",
            "hard_gate": self.hard_gate,
            "minimum_required_switches_per_section": (
                self.minimum_required_switches_per_section
            ),
            "activation": {
                "threshold_quantile": self.threshold_quantile,
                "both_moving": {
                    "minimum_pose_motion": self.both_moving_minimum_pose_motion,
                    "requires_outgoing_and_incoming_observed": True,
                },
                "continuity_compatible": {
                    "minimum_handcrafted_score": (
                        self.continuity_compatible_minimum_score
                    ),
                    "scorer_contract": HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
                },
                "pose_phase_compatible": {
                    "semantics": SAME_TOKEN_POSE_SIMILARITY_SEMANTICS,
                    "preferred_cosine_target": (
                        self.pose_phase_compatible_preferred_cosine
                    ),
                    "preferred_target_quantile": self.pose_phase_target_quantile,
                    "soft_bonus_qualifier": (
                        self.pose_phase_compatible_preferred_cosine
                    ),
                    "soft_bonus_qualifier_quantile": (
                        self.pose_phase_target_quantile
                    ),
                    "runtime_pair_feature_available": (
                        self.pose_phase_runtime_pair_feature_available
                    ),
                    "default_runtime_threshold_bound": False,
                    "used_for_hard_gate": False,
                    "hard_threshold": self.pose_phase_hard_threshold,
                },
                "sharpness": {
                    "minimum": self.sharpness_minimum,
                    "used_for_activation": False,
                },
                "high_motion_alone_is_sufficient": False,
            },
        }


@dataclass(frozen=True, slots=True)
class _MeasuredMotionEvidence:
    source_pair_positive_count: int
    per_song_source_pair_positive_count: Mapping[str, int]
    build_close_involved_switch_count: int
    chorus_close_involved_switch_count: int
    bridge_close_involved_switch_count: int
    pose_motion_minimum_values: tuple[float, ...]
    continuity_score_values: tuple[float, ...]
    pose_embedding_cosine_values: tuple[float, ...]
    sharpness_minimum_values: tuple[float, ...]
    per_role_observed_support: Mapping[str, Mapping[str, int]]

    @property
    def highlight_close_involved_switch_count(self) -> int:
        return (
            self.build_close_involved_switch_count
            + self.chorus_close_involved_switch_count
            + self.bridge_close_involved_switch_count
        )


def _role_at_token(token: Any) -> str:
    if tuple(DEFAULT_STAGE_EDIT_SCHEMA.section_feature_names[:6]) != (
        _ROLE_PROBABILITY_NAMES
    ):
        raise ProfessionalMotionCutProfileError(
            "StageEdit acoustic-role schema changed"
        )
    values = tuple(token.section_features)
    if len(values) != len(DEFAULT_STAGE_EDIT_SCHEMA.section_feature_names):
        raise ProfessionalMotionCutProfileError(
            "StageEdit section feature width mismatch"
        )
    context_offset = len(DEFAULT_STAGE_EDIT_SCHEMA.music_feature_names)
    missing = tuple(token.missing_mask)[
        context_offset : context_offset + len(values)
    ]
    if len(missing) != len(values) or any(missing[:6]):
        raise ProfessionalMotionCutProfileError(
            "highlight role evidence is missing at a source switch"
        )
    positive = [index for index, value in enumerate(values[:6]) if value > 0.0]
    if len(positive) != 1:
        raise ProfessionalMotionCutProfileError(
            "source switch has no unique acoustic role"
        )
    if any(not 0.0 <= float(value) <= 1.0 for value in values[:6]):
        raise ProfessionalMotionCutProfileError(
            "acoustic role probability lies outside [0, 1]"
        )
    return _ROLE_NAMES[positive[0]]


def _candidate_by_id(token: Any) -> dict[str, SourceCandidateToken]:
    result: dict[str, SourceCandidateToken] = {}
    for candidate in token.source_candidates:
        if not isinstance(candidate, SourceCandidateToken):
            raise ProfessionalMotionCutProfileError(
                "snapshot source candidate type changed"
            )
        source_id = _name(candidate.source_id, "source candidate id")
        if source_id in result:
            raise ProfessionalMotionCutProfileError(
                "snapshot token has duplicate source ids"
            )
        result[source_id] = candidate
    return result


def _named_candidate_value(
    candidate: SourceCandidateToken,
    feature_name: str,
) -> float | None:
    names = tuple(DEFAULT_STAGE_EDIT_SCHEMA.source_feature_names)
    try:
        index = names.index(feature_name)
    except ValueError as exc:
        raise ProfessionalMotionCutProfileError(
            f"StageEdit schema lacks {feature_name}"
        ) from exc
    if len(candidate.features) != len(names) or len(candidate.missing_mask) != len(
        names
    ):
        raise ProfessionalMotionCutProfileError(
            "source feature or missing-mask width mismatch"
        )
    if candidate.missing_mask[index]:
        return None
    return _finite(candidate.features[index], f"source feature {feature_name}")


def _pose_embedding(candidate: SourceCandidateToken) -> tuple[float, ...] | None:
    names = tuple(DEFAULT_STAGE_EDIT_SCHEMA.source_feature_names)
    indices = tuple(names.index(name) for name in STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES)
    if len(candidate.features) != len(names) or len(candidate.missing_mask) != len(
        names
    ):
        raise ProfessionalMotionCutProfileError(
            "source feature or missing-mask width mismatch"
        )
    if any(candidate.missing_mask[index] for index in indices):
        return None
    values = tuple(
        _finite(
            candidate.features[index],
            "pose embedding value",
            minimum=-1.0,
            maximum=1.0,
        )
        for index in indices
    )
    if math.sqrt(sum(value * value for value in values)) <= 1.0e-12:
        return None
    return values


def _cosine_similarity(
    outgoing: SourceCandidateToken,
    incoming: SourceCandidateToken,
) -> float | None:
    left = _pose_embedding(outgoing)
    right = _pose_embedding(incoming)
    if left is None or right is None:
        return None
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    result = sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    return min(1.0, max(-1.0, _finite(result, "pose cosine similarity")))


def _continuity(candidate: SourceCandidateToken) -> StageEditSourceContinuity:
    observation = stage_edit_source_pair_observation_from_candidate(
        candidate,
        schema=DEFAULT_STAGE_EDIT_SCHEMA,
    )
    return StageEditSourceContinuity(**observation.to_dict())


def _measure_motion_cut_pairs(
    corpus: ReferenceTrainingCorpus,
    source_pairs: StageEditSourcePairTrainingCorpus,
) -> _MeasuredMotionEvidence:
    if not isinstance(corpus, ReferenceTrainingCorpus):
        raise ProfessionalMotionCutProfileError(
            "motion calibration requires a typed reference corpus"
        )
    if not isinstance(source_pairs, StageEditSourcePairTrainingCorpus):
        raise ProfessionalMotionCutProfileError(
            "motion calibration requires a typed source-pair corpus"
        )
    if source_pairs.token_schema_fingerprint != DEFAULT_STAGE_EDIT_SCHEMA.fingerprint:
        raise ProfessionalMotionCutProfileError(
            "source-pair token schema fingerprint mismatch"
        )
    examples = {example.reference_id: example for example in corpus.examples}
    if len(examples) != len(corpus.examples):
        raise ProfessionalMotionCutProfileError("reference ids must be unique")
    song_by_group = dict(source_pairs.group_song_ids)
    per_song: dict[str, int] = {}
    role_counts = {role: 0 for role in HIGHLIGHT_ROLES}
    role_support = {
        role: {
            "close_involved_switch_count": 0,
            "pose_motion_pair_observed_switch_count": 0,
            "continuity_observed_switch_count": 0,
            "pose_embedding_cosine_observed_switch_count": 0,
            "sharpness_pair_observed_switch_count": 0,
        }
        for role in HIGHLIGHT_ROLES
    }
    pose_values: list[float] = []
    continuity_values: list[float] = []
    cosine_values: list[float] = []
    sharpness_values: list[float] = []

    for group in source_pairs.groups:
        expected_prefix = f"{group.reference_id}:source-pair:"
        if not group.group_id.startswith(expected_prefix):
            raise ProfessionalMotionCutProfileError(
                "source-pair group id does not bind its reference"
            )
        suffix = group.group_id.removeprefix(expected_prefix)
        if len(suffix) != 6 or not suffix.isdigit():
            raise ProfessionalMotionCutProfileError(
                "source-pair group id lacks an exact token index"
            )
        token_index = int(suffix)
        example = examples.get(group.reference_id)
        if example is None or token_index >= len(example.sequence.tokens):
            raise ProfessionalMotionCutProfileError(
                "source-pair group references an unknown token"
            )
        target = example.targets.targets[token_index]
        if target.cut != 1.0 or target.selected_source_id != (
            group.chosen_incoming_source_id
        ):
            raise ProfessionalMotionCutProfileError(
                "source-pair group is not the exact confirmed positive target"
            )
        song_id = _name(song_by_group.get(group.group_id), "group song id")
        if example.song_id != song_id:
            raise ProfessionalMotionCutProfileError(
                "source-pair song identity mismatch"
            )
        per_song[song_id] = per_song.get(song_id, 0) + 1
        token = example.sequence.tokens[token_index]
        role = _role_at_token(token)
        candidates = _candidate_by_id(token)
        outgoing = candidates.get(group.outgoing_source_id)
        incoming = candidates.get(group.chosen_incoming_source_id)
        if outgoing is None or incoming is None:
            raise ProfessionalMotionCutProfileError(
                "source-pair positive is absent from its exact snapshot token"
            )
        if not (
            outgoing.available
            and outgoing.hard_gate_passed
            and incoming.available
            and incoming.hard_gate_passed
        ):
            raise ProfessionalMotionCutProfileError(
                "source-pair positive failed exact-token availability gates"
            )
        outgoing_scale = _named_candidate_value(outgoing, "shot_scale")
        incoming_scale = _named_candidate_value(incoming, "shot_scale")
        if outgoing_scale is None or incoming_scale is None:
            raise ProfessionalMotionCutProfileError(
                "close involvement is unclassifiable at a source-pair positive"
            )
        close_involved = (
            outgoing_scale >= MEDIUM_MAX_EXCLUSIVE
            or incoming_scale >= MEDIUM_MAX_EXCLUSIVE
        )
        if role not in HIGHLIGHT_ROLES or not close_involved:
            continue

        role_counts[role] += 1
        role_support[role]["close_involved_switch_count"] += 1
        outgoing_pose_motion = _named_candidate_value(outgoing, "pose_motion")
        incoming_pose_motion = _named_candidate_value(incoming, "pose_motion")
        if outgoing_pose_motion is not None and incoming_pose_motion is not None:
            pose_values.append(min(outgoing_pose_motion, incoming_pose_motion))
            role_support[role]["pose_motion_pair_observed_switch_count"] += 1

        continuity_score = stage_edit_boundary_continuity_score(
            _continuity(outgoing),
            _continuity(incoming),
        )
        continuity_values.append(
            _finite(
                continuity_score,
                "resolved handcrafted continuity score",
                minimum=-1.0,
                maximum=0.0,
            )
        )
        role_support[role]["continuity_observed_switch_count"] += 1

        cosine = _cosine_similarity(outgoing, incoming)
        if cosine is not None:
            cosine_values.append(cosine)
            role_support[role][
                "pose_embedding_cosine_observed_switch_count"
            ] += 1

        outgoing_sharpness = _named_candidate_value(outgoing, "sharpness")
        incoming_sharpness = _named_candidate_value(incoming, "sharpness")
        if outgoing_sharpness is not None and incoming_sharpness is not None:
            sharpness_values.append(min(outgoing_sharpness, incoming_sharpness))
            role_support[role]["sharpness_pair_observed_switch_count"] += 1

    if sum(per_song.values()) != source_pairs.group_count:
        raise ProfessionalMotionCutProfileError(
            "source-pair per-song counts do not cover the corpus"
        )
    if not continuity_values or not pose_values or not cosine_values:
        raise ProfessionalMotionCutProfileError(
            "motion calibration lacks required human-reference observations"
        )
    return _MeasuredMotionEvidence(
        source_pair_positive_count=source_pairs.group_count,
        per_song_source_pair_positive_count=dict(sorted(per_song.items())),
        build_close_involved_switch_count=role_counts["build"],
        chorus_close_involved_switch_count=role_counts["chorus_like"],
        bridge_close_involved_switch_count=role_counts["bridge_like"],
        pose_motion_minimum_values=tuple(pose_values),
        continuity_score_values=tuple(continuity_values),
        pose_embedding_cosine_values=tuple(cosine_values),
        sharpness_minimum_values=tuple(sharpness_values),
        per_role_observed_support={
            role: dict(role_support[role]) for role in HIGHLIGHT_ROLES
        },
    )


def _validated_sampling(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    _exact_keys(value, _SAMPLING_KEYS, name)
    boundary_guard = _finite(
        value.get("boundary_guard_seconds"),
        f"{name} boundary_guard_seconds",
        minimum=0.0,
    )
    if boundary_guard <= 0.0:
        raise ProfessionalMotionCutProfileError(
            "boundary_guard_seconds must be positive"
        )
    maximum_sources = _integer(
        value.get("max_sources"), f"{name} max_sources", minimum=2
    )
    negative_samples = _integer(
        value.get("negative_samples_per_hold"),
        f"{name} negative_samples_per_hold",
        minimum=0,
    )
    working_width = _integer(
        value.get("visual_working_width"),
        f"{name} visual_working_width",
        minimum=1,
    )
    return {
        "boundary_guard_seconds": boundary_guard,
        "max_sources": maximum_sources,
        "negative_samples_per_hold": negative_samples,
        "visual_working_width": working_width,
    }


def _load_verified_reference_corpus(
    source_profile: ProfessionalMontageGrammarProfile,
) -> tuple[ReferenceTrainingCorpus, StageEditSourcePairTrainingCorpus, dict[str, Any]]:
    payload = source_profile.to_dict()
    scope = _mapping(payload.get("scope"), "source montage scope")
    if (
        scope.get("artist") != EXPECTED_ARTIST
        or scope.get("song_count") != EXPECTED_SONG_COUNT
        or scope.get("reference_count") != EXPECTED_HUMAN_REFERENCE_COUNT
    ):
        raise ProfessionalMotionCutProfileError(
            "motion calibration requires the exact four-song, 20-reference IVE profile"
        )
    provenance = _mapping(payload.get("provenance"), "source montage provenance")
    if provenance.get("token_schema_fingerprint") != (
        DEFAULT_STAGE_EDIT_SCHEMA.fingerprint
    ):
        raise ProfessionalMotionCutProfileError(
            "source montage token schema fingerprint mismatch"
        )
    paths: list[Path] = []
    snapshots: dict[str, StageEditFeatureSnapshot] = {}
    song_by_reference: dict[str, str] = {}
    expected_supervision_hashes: dict[Path, str] = {}
    sampling_contract: dict[str, Any] | None = None

    raw_songs = _array(provenance.get("songs"), "source montage provenance songs")
    if len(raw_songs) != EXPECTED_SONG_COUNT:
        raise ProfessionalMotionCutProfileError("source montage song count mismatch")
    for raw_song in raw_songs:
        song = _mapping(raw_song, "source montage song provenance")
        song_id = _name(song.get("song_id"), "source montage song id")
        feature_path = Path(_name(song.get("feature_manifest_path"), "feature path"))
        feature_payload, _feature_raw = _verified_json(
            feature_path,
            song.get("feature_manifest_file_sha256"),
            f"{song_id} feature manifest",
        )
        current_sampling = _validated_sampling(
            _mapping(feature_payload.get("sampling"), f"{song_id} sampling"),
            f"{song_id} sampling",
        )
        if sampling_contract is None:
            sampling_contract = current_sampling
        elif current_sampling != sampling_contract:
            raise ProfessionalMotionCutProfileError(
                "source montage songs use different snapshot sampling parameters"
            )
        _verified_json(
            Path(_name(song.get("materialization_manifest_path"), "materialization path")),
            song.get("materialization_manifest_file_sha256"),
            f"{song_id} materialization manifest",
        )
        for raw_reference in _array(song.get("references"), f"{song_id} references"):
            reference = _mapping(raw_reference, f"{song_id} reference provenance")
            reference_id = _name(reference.get("reference_id"), "reference id")
            if reference_id in song_by_reference:
                raise ProfessionalMotionCutProfileError(
                    "reference ids must be unique across songs"
                )
            supervision_path = Path(
                _name(reference.get("supervision_path"), "supervision path")
            )
            _supervision, _supervision_raw = _verified_json(
                supervision_path,
                reference.get("supervision_file_sha256"),
                f"{song_id}:{reference_id} supervision",
            )
            snapshot_path = Path(_name(reference.get("snapshot_path"), "snapshot path"))
            snapshot_payload, _snapshot_raw = _verified_json(
                snapshot_path,
                reference.get("snapshot_file_sha256"),
                f"{song_id}:{reference_id} snapshot",
            )
            try:
                snapshot = StageEditFeatureSnapshot.from_dict(
                    snapshot_payload,
                    schema=DEFAULT_STAGE_EDIT_SCHEMA,
                )
            except (TypeError, ValueError) as exc:
                raise ProfessionalMotionCutProfileError(
                    f"{song_id}:{reference_id} snapshot is invalid: {exc}"
                ) from exc
            if snapshot.snapshot_sha256 != reference.get("snapshot_content_sha256"):
                raise ProfessionalMotionCutProfileError(
                    "snapshot internal content hash mismatch"
                )
            if snapshot.sequence.sequence_id != reference.get("snapshot_sequence_id"):
                raise ProfessionalMotionCutProfileError(
                    "snapshot sequence identity mismatch"
                )
            if len(snapshot.sequence.tokens) != reference.get("token_count"):
                raise ProfessionalMotionCutProfileError("snapshot token count mismatch")
            paths.append(supervision_path)
            snapshots[reference_id] = snapshot
            song_by_reference[reference_id] = song_id
            expected_supervision_hashes[supervision_path] = _sha256(
                reference.get("supervision_file_sha256"),
                "supervision SHA-256",
            )

    if sampling_contract is None or len(paths) != EXPECTED_HUMAN_REFERENCE_COUNT:
        raise ProfessionalMotionCutProfileError(
            "source montage profile does not cover exactly 20 references"
        )
    try:
        corpus = load_reference_training_corpus(
            paths,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
            feature_snapshot_by_reference=snapshots,
            song_id_by_reference=song_by_reference,
            negative_samples_per_hold=sampling_contract[
                "negative_samples_per_hold"
            ],
            boundary_guard_seconds=sampling_contract["boundary_guard_seconds"],
            max_sources=sampling_contract["max_sources"],
        )
    except (TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise ProfessionalMotionCutProfileError(
            f"could not reconstruct exact reference corpus: {exc}"
        ) from exc
    # Close the adapter's second-read window by verifying every supervision
    # again after reconstruction.  Any mutation before/during/after loading
    # fails instead of producing a partly mixed corpus.
    for path, expected in expected_supervision_hashes.items():
        _verified_json(path, expected, f"post-load supervision {path.name}")
    if corpus.statistics.get("reference_count") != EXPECTED_HUMAN_REFERENCE_COUNT:
        raise ProfessionalMotionCutProfileError(
            "reference adapter did not reconstruct all 20 references"
        )
    source_pairs = materialize_stage_edit_source_pair_training_corpus(
        corpus,
        schema=DEFAULT_STAGE_EDIT_SCHEMA,
    )
    if (
        source_pairs.reference_count
        != EXPECTED_SOURCE_PAIR_CONTRIBUTING_REFERENCE_COUNT
    ):
        raise ProfessionalMotionCutProfileError(
            "source-pair contributing reference count changed"
        )
    return corpus, source_pairs, sampling_contract


def _distribution_payload(
    distribution: EmpiricalMotionDistribution,
) -> dict[str, Any]:
    return distribution.to_dict()


def _assemble_profile_payload(
    *,
    source_profile_path: Path,
    source_profile_file_sha256: str,
    source_profile: ProfessionalMontageGrammarProfile,
    corpus: ReferenceTrainingCorpus,
    source_pairs: StageEditSourcePairTrainingCorpus,
    sampling: Mapping[str, Any],
    measured: _MeasuredMotionEvidence,
) -> dict[str, Any]:
    pose = EmpiricalMotionDistribution.from_values(
        measured.pose_motion_minimum_values,
        minimum_allowed=0.0,
        maximum_allowed=1.0,
    )
    continuity = EmpiricalMotionDistribution.from_values(
        measured.continuity_score_values,
        minimum_allowed=-1.0,
        maximum_allowed=0.0,
    )
    cosine = EmpiricalMotionDistribution.from_values(
        measured.pose_embedding_cosine_values,
        minimum_allowed=-1.0,
        maximum_allowed=1.0,
    )
    sharpness_support = len(measured.sharpness_minimum_values)
    if sharpness_support:
        raise ProfessionalMotionCutProfileError(
            "the current profile contract requires zero sharpness pair support"
        )
    recommendation = MotionCutSoftPreferenceRecommendation(
        both_moving_minimum_pose_motion=pose.p10,
        continuity_compatible_minimum_score=continuity.p10,
        pose_phase_compatible_preferred_cosine=cosine.p50,
    )
    highlight_count = measured.highlight_close_involved_switch_count
    payload: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "contract": PROFILE_CONTRACT,
        "analysis_basis": PROFILE_ANALYSIS_BASIS,
        "scope": {
            "artist": EXPECTED_ARTIST,
            "song_count": source_pairs.song_count,
            "human_reference_count": EXPECTED_HUMAN_REFERENCE_COUNT,
            "source_pair_contributing_reference_count": source_pairs.reference_count,
            "highlight_roles": list(HIGHLIGHT_ROLES),
            "close_involved_definition": {
                "source_feature": "shot_scale",
                "outgoing_or_incoming_minimum_inclusive": MEDIUM_MAX_EXCLUSIVE,
                "missing_is_close": False,
                "both_sides_must_be_observed": True,
            },
        },
        "sampling": {
            **dict(sampling),
            "parameters_match_every_source_snapshot_manifest": True,
        },
        "input_policy": {
            "confirmed_nonopening_actual_source_switch_positives_only": True,
            "outgoing_and_incoming_observed_at_same_exact_snapshot_token": True,
            "reference_match_evidence_used_as_model_input": False,
            "selected_or_rejected_source_targets_used_as_model_input": False,
            "raw_frames_or_identity_embeddings_persisted": False,
            "song_specific_runtime_threshold_lookup_present": False,
            "runtime_requires_reference_video": False,
        },
        "counts": {
            "source_pair_positive_count": measured.source_pair_positive_count,
            "per_song_source_pair_positive_count": dict(
                measured.per_song_source_pair_positive_count
            ),
            "build_close_involved_switch_count": (
                measured.build_close_involved_switch_count
            ),
            "chorus_close_involved_switch_count": (
                measured.chorus_close_involved_switch_count
            ),
            "bridge_close_involved_switch_count": (
                measured.bridge_close_involved_switch_count
            ),
            "highlight_close_involved_switch_count": highlight_count,
            "pose_motion_pair_observed_switch_count": (
                pose.observed_switch_count
            ),
            "continuity_observed_switch_count": continuity.observed_switch_count,
            "pose_embedding_cosine_observed_switch_count": (
                cosine.observed_switch_count
            ),
            "sharpness_pair_observed_switch_count": sharpness_support,
            "per_role_observed_support": {
                role: dict(measured.per_role_observed_support[role])
                for role in HIGHLIGHT_ROLES
            },
            "source_pair_exclusions": {
                "opening_selection_excluded_count": (
                    source_pairs.opening_selection_excluded_count
                ),
                "unconfirmed_selection_excluded_count": (
                    source_pairs.unconfirmed_selection_excluded_count
                ),
                "same_source_selection_excluded_count": (
                    source_pairs.same_source_selection_excluded_count
                ),
                "priorless_selection_excluded_count": (
                    source_pairs.priorless_selection_excluded_count
                ),
            },
        },
        "distributions": {
            "pose_motion_minimum": _distribution_payload(pose),
            "handcrafted_continuity_score": _distribution_payload(continuity),
            "pose_embedding_cosine_similarity": {
                **_distribution_payload(cosine),
                "semantics": SAME_TOKEN_POSE_SIMILARITY_SEMANTICS,
                "runtime_pair_feature_available": True,
                "default_runtime_threshold_bound": False,
                "used_for_hard_gate": False,
            },
            "sharpness_minimum": {
                "observed_switch_count": 0,
                "quantile_method": None,
                "quantiles": None,
                "minimum": None,
                "maximum": None,
                "unavailable": True,
            },
        },
        "recommendation": recommendation.to_dict(),
        "evidence_limitations": {
            "motion_direction_observed": False,
            "motion_direction_claimed": False,
            "pose_embedding_cosine_is_temporal_phase_ground_truth": False,
            "pose_embedding_cosine_is_same_token_similarity_proxy_only": True,
            "pose_embedding_cosine_runtime_coverage_fraction": (
                cosine.observed_switch_count / highlight_count
            ),
            "pose_motion_pair_coverage_fraction": (
                pose.observed_switch_count / highlight_count
            ),
            "sharpness_pair_coverage_fraction": 0.0,
            "strict_song_disjoint_evaluation_of_a_source_song_requires_loso_rebuild": (
                True
            ),
            "unseen_ive_song_inference_requires_target_reference_labels": False,
            "held_out_or_human_parity_claimed": False,
        },
        "provenance": {
            "content_hash_algorithm": PROFILE_CONTENT_HASH_ALGORITHM,
            "profile_content_sha256": None,
            "source_montage_profile": {
                "path": os.fspath(source_profile_path),
                "file_sha256": source_profile_file_sha256,
                "content_sha256": source_profile.profile_content_sha256,
                "schema_version": MONTAGE_PROFILE_SCHEMA_VERSION,
                "analysis_basis": MONTAGE_PROFILE_ANALYSIS_BASIS,
            },
            "token_schema_fingerprint": DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
            "reference_corpus_data_id": corpus.data_id,
            "source_pair_training_schema_version": (
                STAGE_EDIT_SOURCE_PAIR_TRAINING_SCHEMA_VERSION
            ),
            "source_pair_training_contract": (
                STAGE_EDIT_SOURCE_PAIR_TRAINING_CONTRACT
            ),
            "source_pair_corpus_sha256": source_pairs.content_sha256,
            "handcrafted_continuity_scorer_contract": (
                HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
            ),
            "handcrafted_continuity_scorer_fingerprint": (
                HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
            ),
        },
    }
    payload["provenance"]["profile_content_sha256"] = _profile_content_sha256(
        payload
    )
    return payload


def build_professional_motion_cut_profile(
    montage_profile_path: Path,
) -> "ProfessionalMotionCutProfile":
    """Rebuild a deterministic 20-reference motion-cut profile.

    The source montage profile is itself rebuilt from all immutable manifests,
    supervisions, and snapshots before any motion statistic is accepted.
    """

    source_path = Path(montage_profile_path).expanduser()
    raw = _stable_file_bytes(source_path, "source montage profile")
    source_file_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        source_profile = ProfessionalMontageGrammarProfile.from_file(
            source_path,
            verify_input_hashes=True,
        )
    except (TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise ProfessionalMotionCutProfileError(
            f"source montage profile failed immutable rebuild: {exc}"
        ) from exc
    source_raw_after = _stable_file_bytes(source_path, "source montage profile")
    if (
        hashlib.sha256(source_raw_after).hexdigest() != source_file_sha256
        or source_profile.file_sha256 != source_file_sha256
    ):
        raise ProfessionalMotionCutProfileError(
            "source montage profile changed across immutable rebuild"
        )
    corpus, source_pairs, sampling = _load_verified_reference_corpus(source_profile)
    measured = _measure_motion_cut_pairs(corpus, source_pairs)
    payload = _assemble_profile_payload(
        source_profile_path=source_path.resolve(),
        source_profile_file_sha256=source_file_sha256,
        source_profile=source_profile,
        corpus=corpus,
        source_pairs=source_pairs,
        sampling=sampling,
        measured=measured,
    )
    return ProfessionalMotionCutProfile.from_mapping(payload)


@dataclass(frozen=True, slots=True)
class ProfessionalMotionCutProfile:
    """Strict typed view of one persisted motion-cut calibration artifact."""

    _payload: Mapping[str, Any]
    source_path: Path | None = None
    file_sha256: str | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ProfessionalMotionCutProfile":
        payload = deepcopy(dict(value))
        cls._validate(payload)
        return cls(payload)

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        expected_file_sha256: str | None = None,
        verify_source_inputs: bool = True,
    ) -> "ProfessionalMotionCutProfile":
        source = Path(path).expanduser()
        raw = _stable_file_bytes(source, "motion-cut profile")
        file_sha = hashlib.sha256(raw).hexdigest()
        if expected_file_sha256 is not None and file_sha != _sha256(
            expected_file_sha256, "expected profile file SHA-256"
        ):
            raise ProfessionalMotionCutProfileError(
                "motion-cut profile file SHA-256 mismatch"
            )
        payload = _decode_json(raw, "motion-cut profile")
        cls._validate(payload)
        result = cls(payload, source_path=source.resolve(), file_sha256=file_sha)
        if verify_source_inputs:
            provenance = _mapping(payload["provenance"], "profile provenance")
            source_identity = _mapping(
                provenance["source_montage_profile"],
                "source montage profile identity",
            )
            rebuilt = build_professional_motion_cut_profile(
                Path(str(source_identity["path"]))
            )
            if rebuilt.to_dict() != payload:
                raise ProfessionalMotionCutProfileError(
                    "motion-cut profile differs from immutable source rebuild"
                )
        return result

    @classmethod
    def _validate(cls, payload: dict[str, Any]) -> None:
        _exact_keys(payload, _TOP_LEVEL_KEYS, "motion-cut profile")
        if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
            raise ProfessionalMotionCutProfileError("profile schema version mismatch")
        if payload.get("contract") != PROFILE_CONTRACT:
            raise ProfessionalMotionCutProfileError("profile contract mismatch")
        if payload.get("analysis_basis") != PROFILE_ANALYSIS_BASIS:
            raise ProfessionalMotionCutProfileError("profile analysis basis mismatch")

        scope = _mapping(payload.get("scope"), "profile scope")
        _exact_keys(
            scope,
            {
                "artist",
                "song_count",
                "human_reference_count",
                "source_pair_contributing_reference_count",
                "highlight_roles",
                "close_involved_definition",
            },
            "profile scope",
        )
        if (
            scope.get("artist") != EXPECTED_ARTIST
            or scope.get("song_count") != EXPECTED_SONG_COUNT
            or scope.get("human_reference_count") != EXPECTED_HUMAN_REFERENCE_COUNT
            or scope.get("source_pair_contributing_reference_count")
            != EXPECTED_SOURCE_PAIR_CONTRIBUTING_REFERENCE_COUNT
            or scope.get("highlight_roles") != list(HIGHLIGHT_ROLES)
        ):
            raise ProfessionalMotionCutProfileError("profile scope mismatch")
        close_definition = _mapping(
            scope.get("close_involved_definition"), "close-involved definition"
        )
        if dict(close_definition) != {
            "source_feature": "shot_scale",
            "outgoing_or_incoming_minimum_inclusive": MEDIUM_MAX_EXCLUSIVE,
            "missing_is_close": False,
            "both_sides_must_be_observed": True,
        }:
            raise ProfessionalMotionCutProfileError(
                "close-involved definition mismatch"
            )

        sampling = _mapping(payload.get("sampling"), "profile sampling")
        _exact_keys(
            sampling,
            _SAMPLING_KEYS | {"parameters_match_every_source_snapshot_manifest"},
            "profile sampling",
        )
        _validated_sampling(
            {key: sampling[key] for key in _SAMPLING_KEYS}, "profile sampling"
        )
        if sampling.get("parameters_match_every_source_snapshot_manifest") is not True:
            raise ProfessionalMotionCutProfileError(
                "sampling parameters are not bound to every snapshot manifest"
            )

        expected_policy = {
            "confirmed_nonopening_actual_source_switch_positives_only": True,
            "outgoing_and_incoming_observed_at_same_exact_snapshot_token": True,
            "reference_match_evidence_used_as_model_input": False,
            "selected_or_rejected_source_targets_used_as_model_input": False,
            "raw_frames_or_identity_embeddings_persisted": False,
            "song_specific_runtime_threshold_lookup_present": False,
            "runtime_requires_reference_video": False,
        }
        if payload.get("input_policy") != expected_policy:
            raise ProfessionalMotionCutProfileError("input leakage policy mismatch")

        counts = _mapping(payload.get("counts"), "profile counts")
        _exact_keys(
            counts,
            {
                "source_pair_positive_count",
                "per_song_source_pair_positive_count",
                "build_close_involved_switch_count",
                "chorus_close_involved_switch_count",
                "bridge_close_involved_switch_count",
                "highlight_close_involved_switch_count",
                "pose_motion_pair_observed_switch_count",
                "continuity_observed_switch_count",
                "pose_embedding_cosine_observed_switch_count",
                "sharpness_pair_observed_switch_count",
                "per_role_observed_support",
                "source_pair_exclusions",
            },
            "profile counts",
        )
        source_pair_count = _integer(
            counts.get("source_pair_positive_count"),
            "source_pair_positive_count",
            minimum=1,
        )
        per_song = _mapping(
            counts.get("per_song_source_pair_positive_count"),
            "per-song source-pair counts",
        )
        if len(per_song) != EXPECTED_SONG_COUNT or sum(
            _integer(value, f"per-song source-pair count {song}", minimum=1)
            for song, value in per_song.items()
        ) != source_pair_count:
            raise ProfessionalMotionCutProfileError(
                "per-song source-pair counts mismatch"
            )
        role_counts = {
            "build": _integer(
                counts.get("build_close_involved_switch_count"),
                "build close-involved count",
                minimum=1,
            ),
            "chorus_like": _integer(
                counts.get("chorus_close_involved_switch_count"),
                "chorus close-involved count",
                minimum=1,
            ),
            "bridge_like": _integer(
                counts.get("bridge_close_involved_switch_count"),
                "bridge close-involved count",
                minimum=1,
            ),
        }
        highlight_count = _integer(
            counts.get("highlight_close_involved_switch_count"),
            "highlight close-involved count",
            minimum=1,
        )
        if sum(role_counts.values()) != highlight_count or highlight_count > (
            source_pair_count
        ):
            raise ProfessionalMotionCutProfileError(
                "highlight close-involved counts mismatch"
            )
        support_names = (
            "pose_motion_pair_observed_switch_count",
            "continuity_observed_switch_count",
            "pose_embedding_cosine_observed_switch_count",
            "sharpness_pair_observed_switch_count",
        )
        supports = {
            name: _integer(counts.get(name), name, minimum=0)
            for name in support_names
        }
        if any(value > highlight_count for value in supports.values()):
            raise ProfessionalMotionCutProfileError(
                "observed support exceeds close-involved switch count"
            )
        if supports["sharpness_pair_observed_switch_count"] != 0:
            raise ProfessionalMotionCutProfileError(
                "current sharpness support must remain explicitly unavailable"
            )
        per_role_support = _mapping(
            counts.get("per_role_observed_support"), "per-role observed support"
        )
        if set(per_role_support) != set(HIGHLIGHT_ROLES):
            raise ProfessionalMotionCutProfileError(
                "per-role observed support roles mismatch"
            )
        role_support_names = {
            "close_involved_switch_count",
            *support_names,
        }
        for role in HIGHLIGHT_ROLES:
            values = _mapping(per_role_support[role], f"{role} support")
            _exact_keys(values, role_support_names, f"{role} support")
            if values.get("close_involved_switch_count") != role_counts[role]:
                raise ProfessionalMotionCutProfileError(
                    f"{role} close-involved support mismatch"
                )
            for support_name in support_names:
                _integer(values.get(support_name), f"{role} {support_name}")
        for support_name in support_names:
            if sum(
                int(_mapping(per_role_support[role], role)[support_name])
                for role in HIGHLIGHT_ROLES
            ) != supports[support_name]:
                raise ProfessionalMotionCutProfileError(
                    f"per-role {support_name} mismatch"
                )
        exclusions = _mapping(counts.get("source_pair_exclusions"), "pair exclusions")
        _exact_keys(
            exclusions,
            {
                "opening_selection_excluded_count",
                "unconfirmed_selection_excluded_count",
                "same_source_selection_excluded_count",
                "priorless_selection_excluded_count",
            },
            "pair exclusions",
        )
        for name, value in exclusions.items():
            _integer(value, name)

        distributions = _mapping(
            payload.get("distributions"), "profile distributions"
        )
        _exact_keys(
            distributions,
            {
                "pose_motion_minimum",
                "handcrafted_continuity_score",
                "pose_embedding_cosine_similarity",
                "sharpness_minimum",
            },
            "profile distributions",
        )
        pose = EmpiricalMotionDistribution.from_mapping(
            _mapping(distributions["pose_motion_minimum"], "pose motion"),
            name="pose motion",
            minimum_allowed=0.0,
            maximum_allowed=1.0,
        )
        continuity = EmpiricalMotionDistribution.from_mapping(
            _mapping(
                distributions["handcrafted_continuity_score"],
                "handcrafted continuity",
            ),
            name="handcrafted continuity",
            minimum_allowed=-1.0,
            maximum_allowed=0.0,
        )
        cosine_mapping = _mapping(
            distributions["pose_embedding_cosine_similarity"],
            "pose cosine similarity",
        )
        _exact_keys(
            cosine_mapping,
            {
                "observed_switch_count",
                "quantile_method",
                "quantiles",
                "minimum",
                "maximum",
                "semantics",
                "runtime_pair_feature_available",
                "default_runtime_threshold_bound",
                "used_for_hard_gate",
            },
            "pose cosine similarity",
        )
        cosine = EmpiricalMotionDistribution.from_mapping(
            {
                key: cosine_mapping[key]
                for key in (
                    "observed_switch_count",
                    "quantile_method",
                    "quantiles",
                    "minimum",
                    "maximum",
                )
            },
            name="pose cosine similarity",
            minimum_allowed=-1.0,
            maximum_allowed=1.0,
        )
        if (
            cosine_mapping.get("semantics") != SAME_TOKEN_POSE_SIMILARITY_SEMANTICS
            or cosine_mapping.get("runtime_pair_feature_available") is not True
            or cosine_mapping.get("default_runtime_threshold_bound") is not False
            or cosine_mapping.get("used_for_hard_gate") is not False
        ):
            raise ProfessionalMotionCutProfileError(
                "pose cosine similarity is not a truthful audit-only proxy"
            )
        sharpness = _mapping(distributions["sharpness_minimum"], "sharpness")
        if dict(sharpness) != {
            "observed_switch_count": 0,
            "quantile_method": None,
            "quantiles": None,
            "minimum": None,
            "maximum": None,
            "unavailable": True,
        }:
            raise ProfessionalMotionCutProfileError(
                "zero-support sharpness must remain unavailable"
            )
        if (
            pose.observed_switch_count
            != supports["pose_motion_pair_observed_switch_count"]
            or continuity.observed_switch_count
            != supports["continuity_observed_switch_count"]
            or cosine.observed_switch_count
            != supports["pose_embedding_cosine_observed_switch_count"]
        ):
            raise ProfessionalMotionCutProfileError(
                "distribution support does not match counts"
            )

        recommendation_payload = _mapping(
            payload.get("recommendation"), "profile recommendation"
        )
        _exact_keys(
            recommendation_payload,
            {
                "contract",
                "mode",
                "hard_gate",
                "minimum_required_switches_per_section",
                "activation",
            },
            "profile recommendation",
        )
        if recommendation_payload.get("mode") != "bounded_joint_soft_preference":
            raise ProfessionalMotionCutProfileError(
                "motion recommendation mode mismatch"
            )
        activation = _mapping(
            recommendation_payload.get("activation"), "recommendation activation"
        )
        _exact_keys(
            activation,
            {
                "threshold_quantile",
                "both_moving",
                "continuity_compatible",
                "pose_phase_compatible",
                "sharpness",
                "high_motion_alone_is_sufficient",
            },
            "recommendation activation",
        )
        both = _mapping(activation["both_moving"], "both-moving recommendation")
        continuity_recommendation = _mapping(
            activation["continuity_compatible"], "continuity recommendation"
        )
        phase = _mapping(
            activation["pose_phase_compatible"], "pose-phase recommendation"
        )
        sharpness_recommendation = _mapping(
            activation["sharpness"], "sharpness recommendation"
        )
        expected_both = {
            "minimum_pose_motion": pose.p10,
            "requires_outgoing_and_incoming_observed": True,
        }
        expected_continuity = {
            "minimum_handcrafted_score": continuity.p10,
            "scorer_contract": HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
        }
        expected_phase = {
            "semantics": SAME_TOKEN_POSE_SIMILARITY_SEMANTICS,
            "preferred_cosine_target": cosine.p50,
            "preferred_target_quantile": POSE_PHASE_AUDIT_TARGET_QUANTILE,
            "soft_bonus_qualifier": cosine.p50,
            "soft_bonus_qualifier_quantile": POSE_PHASE_AUDIT_TARGET_QUANTILE,
            "runtime_pair_feature_available": True,
            "default_runtime_threshold_bound": False,
            "used_for_hard_gate": False,
            "hard_threshold": None,
        }
        expected_sharpness = {"minimum": None, "used_for_activation": False}
        if (
            recommendation_payload.get("contract")
            != MOTION_RECOMMENDATION_CONTRACT
            or recommendation_payload.get("hard_gate") is not False
            or recommendation_payload.get("minimum_required_switches_per_section")
            is not None
            or not math.isclose(
                _finite(activation.get("threshold_quantile"), "threshold quantile"),
                SOFT_PREFERENCE_THRESHOLD_QUANTILE,
                abs_tol=1.0e-12,
            )
            or dict(both) != expected_both
            or dict(continuity_recommendation) != expected_continuity
            or dict(phase) != expected_phase
            or dict(sharpness_recommendation) != expected_sharpness
            or activation.get("high_motion_alone_is_sufficient") is not False
        ):
            raise ProfessionalMotionCutProfileError(
                "motion recommendation is not the calibrated bounded soft policy"
            )

        limitations = _mapping(
            payload.get("evidence_limitations"), "evidence limitations"
        )
        _exact_keys(
            limitations,
            {
                "motion_direction_observed",
                "motion_direction_claimed",
                "pose_embedding_cosine_is_temporal_phase_ground_truth",
                "pose_embedding_cosine_is_same_token_similarity_proxy_only",
                "pose_embedding_cosine_runtime_coverage_fraction",
                "pose_motion_pair_coverage_fraction",
                "sharpness_pair_coverage_fraction",
                "strict_song_disjoint_evaluation_of_a_source_song_requires_loso_rebuild",
                "unseen_ive_song_inference_requires_target_reference_labels",
                "held_out_or_human_parity_claimed",
            },
            "evidence limitations",
        )
        if (
            limitations.get("motion_direction_observed") is not False
            or limitations.get("motion_direction_claimed") is not False
            or limitations.get("pose_embedding_cosine_is_temporal_phase_ground_truth")
            is not False
            or limitations.get("pose_embedding_cosine_is_same_token_similarity_proxy_only")
            is not True
            or limitations.get(
                "strict_song_disjoint_evaluation_of_a_source_song_requires_loso_rebuild"
            )
            is not True
            or limitations.get(
                "unseen_ive_song_inference_requires_target_reference_labels"
            )
            is not False
            or limitations.get("held_out_or_human_parity_claimed") is not False
        ):
            raise ProfessionalMotionCutProfileError(
                "evidence limitations overclaim motion or validation authority"
            )
        expected_coverages = {
            "pose_embedding_cosine_runtime_coverage_fraction": (
                cosine.observed_switch_count / highlight_count
            ),
            "pose_motion_pair_coverage_fraction": (
                pose.observed_switch_count / highlight_count
            ),
            "sharpness_pair_coverage_fraction": 0.0,
        }
        for name, expected in expected_coverages.items():
            if not math.isclose(
                _finite(limitations.get(name), name, minimum=0.0, maximum=1.0),
                expected,
                abs_tol=1.0e-12,
            ):
                raise ProfessionalMotionCutProfileError(f"{name} mismatch")

        provenance = _mapping(payload.get("provenance"), "profile provenance")
        _exact_keys(
            provenance,
            {
                "content_hash_algorithm",
                "profile_content_sha256",
                "source_montage_profile",
                "token_schema_fingerprint",
                "reference_corpus_data_id",
                "source_pair_training_schema_version",
                "source_pair_training_contract",
                "source_pair_corpus_sha256",
                "handcrafted_continuity_scorer_contract",
                "handcrafted_continuity_scorer_fingerprint",
            },
            "profile provenance",
        )
        source_identity = _mapping(
            provenance.get("source_montage_profile"),
            "source montage profile identity",
        )
        _exact_keys(
            source_identity,
            {
                "path",
                "file_sha256",
                "content_sha256",
                "schema_version",
                "analysis_basis",
            },
            "source montage profile identity",
        )
        if not Path(str(source_identity.get("path") or "")).is_absolute():
            raise ProfessionalMotionCutProfileError(
                "source montage profile path must be absolute"
            )
        _sha256(source_identity.get("file_sha256"), "source montage file SHA-256")
        _sha256(
            source_identity.get("content_sha256"), "source montage content SHA-256"
        )
        if (
            source_identity.get("schema_version") != MONTAGE_PROFILE_SCHEMA_VERSION
            or source_identity.get("analysis_basis")
            != MONTAGE_PROFILE_ANALYSIS_BASIS
            or provenance.get("content_hash_algorithm")
            != PROFILE_CONTENT_HASH_ALGORITHM
            or provenance.get("token_schema_fingerprint")
            != DEFAULT_STAGE_EDIT_SCHEMA.fingerprint
            or provenance.get("source_pair_training_schema_version")
            != STAGE_EDIT_SOURCE_PAIR_TRAINING_SCHEMA_VERSION
            or provenance.get("source_pair_training_contract")
            != STAGE_EDIT_SOURCE_PAIR_TRAINING_CONTRACT
            or provenance.get("handcrafted_continuity_scorer_contract")
            != HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
            or provenance.get("handcrafted_continuity_scorer_fingerprint")
            != HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
        ):
            raise ProfessionalMotionCutProfileError("profile provenance mismatch")
        _name(provenance.get("reference_corpus_data_id"), "reference corpus data id")
        _sha256(provenance.get("source_pair_corpus_sha256"), "source-pair corpus SHA")
        profile_hash = _sha256(
            provenance.get("profile_content_sha256"), "profile content SHA-256"
        )
        if profile_hash != _profile_content_sha256(payload):
            raise ProfessionalMotionCutProfileError(
                "profile content SHA-256 mismatch"
            )

    @property
    def profile_content_sha256(self) -> str:
        return str(self._payload["provenance"]["profile_content_sha256"])

    @property
    def recommendation(self) -> MotionCutSoftPreferenceRecommendation:
        distributions = self._payload["distributions"]
        pose = distributions["pose_motion_minimum"]
        continuity = distributions["handcrafted_continuity_score"]
        cosine = distributions["pose_embedding_cosine_similarity"]
        return MotionCutSoftPreferenceRecommendation(
            both_moving_minimum_pose_motion=float(pose["quantiles"]["p10"]),
            continuity_compatible_minimum_score=float(
                continuity["quantiles"]["p10"]
            ),
            pose_phase_compatible_preferred_cosine=float(
                cosine["quantiles"]["p50"]
            ),
        )

    def to_stage_edit_calibration(self) -> StageEditHighlightMotionCalibration:
        """Return the StageEdit-native immutable calibration payload."""

        counts = self._payload["counts"]
        distributions = self._payload["distributions"]

        def quantiles(name: str) -> tuple[float, float, float, float]:
            values = distributions[name]["quantiles"]
            return tuple(float(values[key]) for key in QUANTILE_NAMES)  # type: ignore[return-value]

        return StageEditHighlightMotionCalibration(
            analysis_basis=PROFILE_ANALYSIS_BASIS,
            human_reference_count=int(self._payload["scope"]["human_reference_count"]),
            source_pair_negative_samples=int(
                self._payload["sampling"]["negative_samples_per_hold"]
            ),
            maximum_sources=int(self._payload["sampling"]["max_sources"]),
            confirmed_source_switch_count=int(counts["source_pair_positive_count"]),
            highlight_role_close_involved_switch_count=int(
                counts["highlight_close_involved_switch_count"]
            ),
            build_close_involved_switch_count=int(
                counts["build_close_involved_switch_count"]
            ),
            chorus_close_involved_switch_count=int(
                counts["chorus_close_involved_switch_count"]
            ),
            bridge_close_involved_switch_count=int(
                counts["bridge_close_involved_switch_count"]
            ),
            pose_pair_observed_switch_count=int(
                counts["pose_motion_pair_observed_switch_count"]
            ),
            continuity_observed_switch_count=int(
                counts["continuity_observed_switch_count"]
            ),
            sharpness_pair_observed_switch_count=int(
                counts["sharpness_pair_observed_switch_count"]
            ),
            pose_embedding_cosine_observed_switch_count=int(
                counts["pose_embedding_cosine_observed_switch_count"]
            ),
            pose_motion_minimum_quantiles=quantiles("pose_motion_minimum"),
            continuity_score_quantiles=quantiles(
                "handcrafted_continuity_score"
            ),
            sharpness_minimum_quantiles=None,
            pose_embedding_cosine_quantiles=quantiles(
                "pose_embedding_cosine_similarity"
            ),
            # This calibration remains an independent handcrafted qualifier
            # even when the decoder's primary boundary scorer is learned.
            continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
            continuity_scorer_contract=(
                HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self._payload))

    def write_json(self, path: Path) -> Path:
        destination = Path(path).expanduser()
        if not destination.is_absolute():
            raise ProfessionalMotionCutProfileError(
                "motion-cut profile destination must be absolute"
            )
        if destination.is_symlink():
            raise ProfessionalMotionCutProfileError(
                "motion-cut profile destination must not be a symlink"
            )
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
        restored = ProfessionalMotionCutProfile.from_file(
            destination,
            verify_source_inputs=False,
        )
        if restored.profile_content_sha256 != self.profile_content_sha256:
            raise ProfessionalMotionCutProfileError(
                "written motion-cut profile failed strict round-trip"
            )
        return destination


def load_professional_motion_cut_profile(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
    verify_source_inputs: bool = True,
) -> ProfessionalMotionCutProfile:
    """Load a strictly hash-bound motion-cut calibration profile."""

    return ProfessionalMotionCutProfile.from_file(
        path,
        expected_file_sha256=expected_file_sha256,
        verify_source_inputs=verify_source_inputs,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a hash-bound 20-reference IVE motion-accented close-cut "
            "calibration profile."
        )
    )
    parser.add_argument("--montage-profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = build_professional_motion_cut_profile(args.montage_profile)
    output = profile.write_json(args.output)
    raw = _stable_file_bytes(output, "written motion-cut profile")
    print(
        json.dumps(
            {
                "path": os.fspath(output),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "content_sha256": profile.profile_content_sha256,
                "counts": profile.to_dict()["counts"],
                "recommendation": profile.recommendation.to_dict(),
                "hard_gate": False,
                "minimum_required_switches_per_section": None,
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )
    return 0


__all__ = [
    "EmpiricalMotionDistribution",
    "MOTION_RECOMMENDATION_CONTRACT",
    "MotionCutSoftPreferenceRecommendation",
    "PROFILE_ANALYSIS_BASIS",
    "PROFILE_CONTRACT",
    "PROFILE_SCHEMA_VERSION",
    "ProfessionalMotionCutProfile",
    "ProfessionalMotionCutProfileError",
    "SAME_TOKEN_POSE_SIMILARITY_SEMANTICS",
    "build_professional_motion_cut_profile",
    "load_professional_motion_cut_profile",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
