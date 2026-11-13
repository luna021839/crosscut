"""Temporal pose-motion evidence for dynamic close-cut continuity.

Same-token pose cosine can show that two subjects have a similar pose at a
cut, but it cannot show that either subject is moving or that their movement
continues in the same direction.  This module evaluates a short chronological
pose-embedding window immediately before the outgoing boundary and a short
chronological window immediately after the incoming boundary.

The calculation is deliberately isolated from edit selection.  It exposes a
bounded soft qualifier and contains no quota or hard edit-policy decision.
Every decision threshold is supplied by the caller; this module does not
claim learned or human-reference-calibrated defaults.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any, Sequence


INVALID_CLASSIFICATION = "invalid"
STATIC_POSE_MATCH_CLASSIFICATION = "static_pose_match"
DYNAMIC_POSE_PHASE_MATCH_CLASSIFICATION = "dynamic_pose_phase_match"
DYNAMIC_POSE_PHASE_MISMATCH_CLASSIFICATION = "dynamic_pose_phase_mismatch"
TEMPORAL_POSE_MOTION_SCALAR_EVIDENCE_CONTRACT = (
    "scalar_temporal_pose_motion_boundary_evidence_v2"
)
TEMPORAL_POSE_VELOCITY_FINGERPRINT_CONTRACT = (
    "validated_normalized_temporal_pose_velocity_sha256_v1"
)
_FORBIDDEN_PUBLIC_VECTOR_KEYS = frozenset(
    {
        "pose_model_embedding",
        "normalized_velocity",
        "pose_embedding",
        "embeddings",
    }
)


def _is_forbidden_public_vector_key(key: str) -> bool:
    return bool(
        key in _FORBIDDEN_PUBLIC_VECTOR_KEYS
        or key.endswith("_pose_embedding")
        or key.endswith("_normalized_velocity")
        or key.endswith("_embeddings")
    )


def _content_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _vector_sha256(vector: Sequence[float] | None) -> str | None:
    if vector is None:
        return None
    return _content_sha256(
        {
            "contract": TEMPORAL_POSE_VELOCITY_FINGERPRINT_CONTRACT,
            "feature_dimension": len(vector),
            "coordinates": [float(value) for value in vector],
        }
    )


def require_scalar_stage_edit_pose_payload(
    payload: object,
    *,
    context: str,
) -> None:
    """Fail closed if a public payload contains raw pose-sized arrays."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized_key = str(key)
            if _is_forbidden_public_vector_key(normalized_key):
                raise ValueError(
                    f"{context} contains forbidden raw pose vector key "
                    f"{normalized_key!r}"
                )
            require_scalar_stage_edit_pose_payload(
                value,
                context=context,
            )
        return
    if isinstance(payload, (list, tuple)):
        if payload and all(
            isinstance(value, Real) and not isinstance(value, bool)
            for value in payload
        ) and len(payload) in {60, 120}:
            raise ValueError(
                f"{context} contains a forbidden raw pose-sized numeric array"
            )
        for value in payload:
            require_scalar_stage_edit_pose_payload(value, context=context)


@dataclass(frozen=True, slots=True)
class TemporalPoseWindow:
    """Chronological pose embeddings on one side of an edit boundary.

    Times only need to be local to their source window, but must be finite and
    strictly increasing.  Each embedding must have the same positive feature
    dimension.  At least two samples are required to observe boundary motion.
    """

    times_seconds: Sequence[float]
    embeddings: Sequence[Sequence[float]]


@dataclass(frozen=True, slots=True)
class TemporalPoseMotionThresholds:
    """Caller-supplied evidence thresholds with no production defaults."""

    minimum_boundary_motion_magnitude: float
    minimum_endpoint_pose_cosine: float
    minimum_velocity_direction_cosine: float
    minimum_phase_compatibility: float
    pose_norm_epsilon: float

    def __post_init__(self) -> None:
        values = (
            self.minimum_boundary_motion_magnitude,
            self.minimum_endpoint_pose_cosine,
            self.minimum_velocity_direction_cosine,
            self.minimum_phase_compatibility,
            self.pose_norm_epsilon,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("temporal pose-motion thresholds must be finite numbers")
        if self.minimum_boundary_motion_magnitude <= 0.0:
            raise ValueError("minimum boundary motion magnitude must be positive")
        if not -1.0 <= self.minimum_endpoint_pose_cosine <= 1.0:
            raise ValueError("minimum endpoint pose cosine must be normalized")
        if not -1.0 <= self.minimum_velocity_direction_cosine <= 1.0:
            raise ValueError(
                "minimum velocity direction cosine must be normalized"
            )
        if not 0.0 <= self.minimum_phase_compatibility <= 1.0:
            raise ValueError("minimum phase compatibility must be normalized")
        if self.pose_norm_epsilon <= 0.0:
            raise ValueError("pose norm epsilon must be positive")


@dataclass(frozen=True, slots=True)
class TemporalPoseMotionBoundaryAnalysis:
    """Fail-closed temporal evidence for one candidate source switch."""

    input_valid: bool
    qualifies: bool
    classification: str
    failure_reasons: tuple[str, ...]
    feature_dimension: int | None
    outgoing_sample_count: int
    incoming_sample_count: int
    outgoing_boundary_motion_magnitude: float | None
    incoming_boundary_motion_magnitude: float | None
    outgoing_normalized_velocity: tuple[float, ...] | None
    incoming_normalized_velocity: tuple[float, ...] | None
    endpoint_pose_cosine: float | None
    velocity_direction_cosine: float | None
    phase_compatibility: float | None
    confidence: float
    static_pose_match: bool
    dynamic_pose_phase_match: bool

    def to_dict(self) -> dict[str, object]:
        outgoing_velocity_sha256 = _vector_sha256(
            self.outgoing_normalized_velocity
        )
        incoming_velocity_sha256 = _vector_sha256(
            self.incoming_normalized_velocity
        )
        pair_sha256 = (
            _content_sha256(
                {
                    "contract": TEMPORAL_POSE_VELOCITY_FINGERPRINT_CONTRACT,
                    "feature_dimension": self.feature_dimension,
                    "outgoing_sha256": outgoing_velocity_sha256,
                    "incoming_sha256": incoming_velocity_sha256,
                }
            )
            if outgoing_velocity_sha256 is not None
            or incoming_velocity_sha256 is not None
            else None
        )
        payload: dict[str, object] = {
            "contract": TEMPORAL_POSE_MOTION_SCALAR_EVIDENCE_CONTRACT,
            "input_valid": self.input_valid,
            "qualifies": self.qualifies,
            "classification": self.classification,
            "failure_reasons": list(self.failure_reasons),
            "feature_dimension": self.feature_dimension,
            "outgoing_sample_count": self.outgoing_sample_count,
            "incoming_sample_count": self.incoming_sample_count,
            "outgoing_boundary_motion_magnitude": (
                self.outgoing_boundary_motion_magnitude
            ),
            "incoming_boundary_motion_magnitude": (
                self.incoming_boundary_motion_magnitude
            ),
            "velocity_evidence": {
                "outgoing_observed": (
                    self.outgoing_normalized_velocity is not None
                ),
                "incoming_observed": (
                    self.incoming_normalized_velocity is not None
                ),
                "feature_dimension": self.feature_dimension,
                "trusted_producer_contract": (
                    TEMPORAL_POSE_VELOCITY_FINGERPRINT_CONTRACT
                ),
                "outgoing_content_sha256": outgoing_velocity_sha256,
                "incoming_content_sha256": incoming_velocity_sha256,
                "pair_content_sha256": pair_sha256,
            },
            "endpoint_pose_cosine": self.endpoint_pose_cosine,
            "velocity_direction_cosine": self.velocity_direction_cosine,
            "phase_compatibility": self.phase_compatibility,
            "confidence": self.confidence,
            "static_pose_match": self.static_pose_match,
            "dynamic_pose_phase_match": self.dynamic_pose_phase_match,
        }
        require_scalar_stage_edit_pose_payload(
            payload,
            context="temporal pose-motion boundary analysis",
        )
        return payload


@dataclass(frozen=True, slots=True)
class _ValidatedWindow:
    times_seconds: tuple[float, ...]
    normalized_embeddings: tuple[tuple[float, ...], ...]
    feature_dimension: int


def _sample_count(window: TemporalPoseWindow | None) -> int:
    if window is None:
        return 0
    try:
        return len(window.embeddings)
    except TypeError:
        return 0


def _invalid_analysis(
    *,
    outgoing: TemporalPoseWindow | None,
    incoming: TemporalPoseWindow | None,
    reasons: Sequence[str],
) -> TemporalPoseMotionBoundaryAnalysis:
    return TemporalPoseMotionBoundaryAnalysis(
        input_valid=False,
        qualifies=False,
        classification=INVALID_CLASSIFICATION,
        failure_reasons=tuple(dict.fromkeys(reasons)),
        feature_dimension=None,
        outgoing_sample_count=_sample_count(outgoing),
        incoming_sample_count=_sample_count(incoming),
        outgoing_boundary_motion_magnitude=None,
        incoming_boundary_motion_magnitude=None,
        outgoing_normalized_velocity=None,
        incoming_normalized_velocity=None,
        endpoint_pose_cosine=None,
        velocity_direction_cosine=None,
        phase_compatibility=None,
        confidence=0.0,
        static_pose_match=False,
        dynamic_pose_phase_match=False,
    )


def _validate_window(
    window: TemporalPoseWindow | None,
    *,
    side: str,
    pose_norm_epsilon: float,
) -> tuple[_ValidatedWindow | None, tuple[str, ...]]:
    if window is None:
        return (None, (f"missing_{side}_window",))

    try:
        times = tuple(window.times_seconds)
        raw_embeddings = tuple(window.embeddings)
    except TypeError:
        return (None, (f"invalid_{side}_window_sequence",))

    reasons: list[str] = []
    if len(times) != len(raw_embeddings):
        reasons.append(f"{side}_time_embedding_count_mismatch")
    if len(times) < 2 or len(raw_embeddings) < 2:
        reasons.append(f"insufficient_{side}_samples")
    if reasons:
        return (None, tuple(reasons))

    finite_times: list[float] = []
    for value in times:
        if isinstance(value, bool) or not isinstance(value, Real):
            reasons.append(f"nonfinite_{side}_time")
            break
        number = float(value)
        if not math.isfinite(number):
            reasons.append(f"nonfinite_{side}_time")
            break
        finite_times.append(number)
    if len(finite_times) == len(times) and any(
        right <= left for left, right in zip(finite_times, finite_times[1:])
    ):
        reasons.append(f"nonincreasing_{side}_times")

    feature_dimension: int | None = None
    normalized_embeddings: list[tuple[float, ...]] = []
    for raw_embedding in raw_embeddings:
        try:
            raw_values = tuple(raw_embedding)
        except TypeError:
            reasons.append(f"invalid_{side}_embedding_sequence")
            break
        if feature_dimension is None:
            feature_dimension = len(raw_values)
            if feature_dimension <= 0:
                reasons.append(f"empty_{side}_embedding")
                break
        elif len(raw_values) != feature_dimension:
            reasons.append(f"inconsistent_{side}_embedding_dimension")
            break

        values: list[float] = []
        for value in raw_values:
            if isinstance(value, bool) or not isinstance(value, Real):
                reasons.append(f"nonfinite_{side}_embedding")
                break
            number = float(value)
            if not math.isfinite(number):
                reasons.append(f"nonfinite_{side}_embedding")
                break
            values.append(number)
        if len(values) != len(raw_values):
            break
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= pose_norm_epsilon:
            reasons.append(f"zero_norm_{side}_embedding")
            break
        normalized_embeddings.append(tuple(value / norm for value in values))

    if reasons:
        return (None, tuple(dict.fromkeys(reasons)))
    assert feature_dimension is not None
    return (
        _ValidatedWindow(
            times_seconds=tuple(finite_times),
            normalized_embeddings=tuple(normalized_embeddings),
            feature_dimension=feature_dimension,
        ),
        (),
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def _bounded_cosine(value: float) -> float:
    return min(1.0, max(-1.0, value))


def _boundary_velocity(
    window: _ValidatedWindow,
    *,
    outgoing: bool,
) -> tuple[float, ...]:
    if outgoing:
        left_index = len(window.times_seconds) - 2
        right_index = len(window.times_seconds) - 1
    else:
        left_index = 0
        right_index = 1
    delta_seconds = (
        window.times_seconds[right_index] - window.times_seconds[left_index]
    )
    return tuple(
        (right_value - left_value) / delta_seconds
        for left_value, right_value in zip(
            window.normalized_embeddings[left_index],
            window.normalized_embeddings[right_index],
            strict=True,
        )
    )


def _magnitude(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _unit_vector(
    vector: Sequence[float],
    *,
    epsilon: float,
) -> tuple[float, ...] | None:
    magnitude = _magnitude(vector)
    if magnitude <= epsilon:
        return None
    return tuple(value / magnitude for value in vector)


def _compatibility_from_cosines(
    endpoint_pose_cosine: float,
    velocity_direction_cosine: float,
) -> float:
    """Return the weaker normalized pose-or-direction agreement in [0, 1]."""

    endpoint_agreement = (endpoint_pose_cosine + 1.0) / 2.0
    direction_agreement = (velocity_direction_cosine + 1.0) / 2.0
    return min(1.0, max(0.0, min(endpoint_agreement, direction_agreement)))


def analyze_temporal_pose_motion_boundary(
    *,
    outgoing: TemporalPoseWindow | None,
    incoming: TemporalPoseWindow | None,
    thresholds: TemporalPoseMotionThresholds,
) -> TemporalPoseMotionBoundaryAnalysis:
    """Analyze whether a candidate cut preserves moving pose phase.

    Each embedding is normalized before taking a time derivative.  The result
    is invariant to positive global scaling of either pose-embedding window,
    so camera/encoder amplitude alone is not counted as choreography motion.
    Motion direction is measured from the final outgoing interval and the
    first incoming interval: the samples immediately adjacent to the cut.
    """

    outgoing_validated, outgoing_reasons = _validate_window(
        outgoing,
        side="outgoing",
        pose_norm_epsilon=thresholds.pose_norm_epsilon,
    )
    incoming_validated, incoming_reasons = _validate_window(
        incoming,
        side="incoming",
        pose_norm_epsilon=thresholds.pose_norm_epsilon,
    )
    invalid_reasons = outgoing_reasons + incoming_reasons
    if outgoing_validated is None or incoming_validated is None:
        return _invalid_analysis(
            outgoing=outgoing,
            incoming=incoming,
            reasons=invalid_reasons,
        )
    if outgoing_validated.feature_dimension != incoming_validated.feature_dimension:
        return _invalid_analysis(
            outgoing=outgoing,
            incoming=incoming,
            reasons=("cross_boundary_embedding_dimension_mismatch",),
        )

    outgoing_velocity = _boundary_velocity(outgoing_validated, outgoing=True)
    incoming_velocity = _boundary_velocity(incoming_validated, outgoing=False)
    outgoing_motion = _magnitude(outgoing_velocity)
    incoming_motion = _magnitude(incoming_velocity)
    outgoing_direction = _unit_vector(
        outgoing_velocity,
        epsilon=thresholds.pose_norm_epsilon,
    )
    incoming_direction = _unit_vector(
        incoming_velocity,
        epsilon=thresholds.pose_norm_epsilon,
    )
    endpoint_cosine = _bounded_cosine(
        _dot(
            outgoing_validated.normalized_embeddings[-1],
            incoming_validated.normalized_embeddings[0],
        )
    )

    reasons: list[str] = []
    if outgoing_motion < thresholds.minimum_boundary_motion_magnitude:
        reasons.append("outgoing_boundary_motion_below_threshold")
    if incoming_motion < thresholds.minimum_boundary_motion_magnitude:
        reasons.append("incoming_boundary_motion_below_threshold")
    if outgoing_direction is None or incoming_direction is None:
        direction_cosine = None
        phase_compatibility = None
        reasons.append("boundary_velocity_direction_undefined")
    else:
        direction_cosine = _bounded_cosine(
            _dot(outgoing_direction, incoming_direction)
        )
        phase_compatibility = _compatibility_from_cosines(
            endpoint_cosine,
            direction_cosine,
        )
        if direction_cosine < thresholds.minimum_velocity_direction_cosine:
            reasons.append("velocity_direction_below_threshold")
        if phase_compatibility < thresholds.minimum_phase_compatibility:
            reasons.append("pose_phase_compatibility_below_threshold")
    if endpoint_cosine < thresholds.minimum_endpoint_pose_cosine:
        reasons.append("endpoint_pose_cosine_below_threshold")

    minimum_motion = min(outgoing_motion, incoming_motion)
    motion_support = min(
        1.0,
        minimum_motion / thresholds.minimum_boundary_motion_magnitude,
    )
    confidence = (
        min(motion_support, phase_compatibility)
        if phase_compatibility is not None
        else 0.0
    )
    static_pose_match = (
        endpoint_cosine >= thresholds.minimum_endpoint_pose_cosine
        and (
            outgoing_motion < thresholds.minimum_boundary_motion_magnitude
            or incoming_motion < thresholds.minimum_boundary_motion_magnitude
        )
    )
    qualifies = not reasons
    if qualifies:
        classification = DYNAMIC_POSE_PHASE_MATCH_CLASSIFICATION
    elif static_pose_match:
        classification = STATIC_POSE_MATCH_CLASSIFICATION
    else:
        classification = DYNAMIC_POSE_PHASE_MISMATCH_CLASSIFICATION

    return TemporalPoseMotionBoundaryAnalysis(
        input_valid=True,
        qualifies=qualifies,
        classification=classification,
        failure_reasons=tuple(dict.fromkeys(reasons)),
        feature_dimension=outgoing_validated.feature_dimension,
        outgoing_sample_count=len(outgoing_validated.times_seconds),
        incoming_sample_count=len(incoming_validated.times_seconds),
        outgoing_boundary_motion_magnitude=outgoing_motion,
        incoming_boundary_motion_magnitude=incoming_motion,
        outgoing_normalized_velocity=outgoing_direction,
        incoming_normalized_velocity=incoming_direction,
        endpoint_pose_cosine=endpoint_cosine,
        velocity_direction_cosine=direction_cosine,
        phase_compatibility=phase_compatibility,
        confidence=confidence,
        static_pose_match=static_pose_match,
        dynamic_pose_phase_match=qualifies,
    )


__all__ = [
    "DYNAMIC_POSE_PHASE_MATCH_CLASSIFICATION",
    "DYNAMIC_POSE_PHASE_MISMATCH_CLASSIFICATION",
    "INVALID_CLASSIFICATION",
    "STATIC_POSE_MATCH_CLASSIFICATION",
    "TEMPORAL_POSE_MOTION_SCALAR_EVIDENCE_CONTRACT",
    "TEMPORAL_POSE_VELOCITY_FINGERPRINT_CONTRACT",
    "TemporalPoseMotionBoundaryAnalysis",
    "TemporalPoseMotionThresholds",
    "TemporalPoseWindow",
    "analyze_temporal_pose_motion_boundary",
    "require_scalar_stage_edit_pose_payload",
]
