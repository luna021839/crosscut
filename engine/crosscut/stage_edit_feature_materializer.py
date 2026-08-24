"""Inference-safe feature materialization shared by StageEdit train and serve.

The StageEdit model must see the same numeric columns when it learns from a
finished reference edit and when it edits a new performance.  This module is
the single dependency-free bridge from existing rhythm, section, timeline and
source-local visual analysis into :class:`StageEditToken`.

Reference-attribution evidence is intentionally not representable here.  The
input dataclasses expose only features that can be measured before an editor
chooses a source.  In particular there is no selected label, observed rank,
reverse-match score, homography or reference crop field.  Serialized snapshots
are schema-bound, content-hashed and contain features only -- never targets.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from .section_director import SectionPlan, SectionRole
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_FORMATION_LAYOUT_FEATURE_NAMES,
    STAGE_EDIT_FORMATION_POSE_FEATURE_NAMES,
    STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES,
    SourceCandidateToken,
    StageEditSequence,
    StageEditToken,
    StageEditTokenSchema,
)

if TYPE_CHECKING:  # Avoid importing OpenCV-backed visual analysis at runtime.
    from .rhythm import RhythmAnalysis
    from .timeline import TimelineCandidate
    from .visual_score import FrameVisualDescriptor


FEATURE_MATERIALIZER_SCHEMA_VERSION = 2
FEATURE_MATERIALIZER_CONTRACT = (
    "stage_edit_inference_available_feature_materializer_v2"
)
FEATURE_SNAPSHOT_KIND = "stage_edit_inference_safe_feature_snapshot"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_UPSTREAM_COMPONENTS = frozenset(
    {
        "visual_feature_cache",
        "visual_score",
        "timeline_candidate",
        "rhythm_analysis",
        "section_plan",
        "source_alignment",
        "style_profile",
    }
)


def _finite_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unit_or_none(value: Any) -> float | None:
    number = _finite_or_none(value)
    if number is None:
        return None
    return min(1.0, max(0.0, number))


def _signed_unit_or_none(value: Any) -> float | None:
    number = _finite_or_none(value)
    if number is None:
        return None
    return min(1.0, max(-1.0, number))


def _feature_vector(
    names: Sequence[str],
    observed: Mapping[str, Any],
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    values: list[float] = []
    missing: list[bool] = []
    for name in names:
        number = _finite_or_none(observed.get(name))
        if number is None:
            values.append(0.0)
            missing.append(True)
        else:
            values.append(number)
            missing.append(False)
    return tuple(values), tuple(missing)


@dataclass(frozen=True, slots=True)
class StageEditFeatureProvenance:
    """Strict provenance containing identities, never feature/label payloads."""

    upstream_sha256: tuple[tuple[str, str], ...] = ()
    analyzer_fingerprint_sha256: str | None = None
    contracts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized: list[tuple[str, str]] = []
        seen: set[str] = set()
        for raw_name, raw_digest in self.upstream_sha256:
            name = str(raw_name).strip()
            digest = str(raw_digest).strip()
            if name not in _UPSTREAM_COMPONENTS:
                raise ValueError(f"unsupported StageEdit feature source: {name}")
            if name in seen:
                raise ValueError("StageEdit upstream feature sources must be unique")
            if _SHA256.fullmatch(digest) is None:
                raise ValueError("StageEdit upstream identities must be SHA-256")
            seen.add(name)
            normalized.append((name, digest))
        normalized.sort()
        object.__setattr__(self, "upstream_sha256", tuple(normalized))
        analyzer = self.analyzer_fingerprint_sha256
        if analyzer is not None and _SHA256.fullmatch(str(analyzer)) is None:
            raise ValueError("analyzer fingerprint must be SHA-256 or None")
        object.__setattr__(
            self,
            "analyzer_fingerprint_sha256",
            str(analyzer) if analyzer is not None else None,
        )
        contracts = tuple(sorted({str(value).strip() for value in self.contracts}))
        if any(not value or len(value) > 160 for value in contracts):
            raise ValueError("feature provenance contracts must be bounded strings")
        object.__setattr__(self, "contracts", contracts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer": FEATURE_MATERIALIZER_CONTRACT,
            "input_availability": "available_before_editorial_source_choice",
            "reference_attribution_inputs_present": False,
            "upstream_sha256": dict(self.upstream_sha256),
            "analyzer_fingerprint_sha256": self.analyzer_fingerprint_sha256,
            "contracts": list(self.contracts),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageEditFeatureProvenance":
        expected = {
            "producer",
            "input_availability",
            "reference_attribution_inputs_present",
            "upstream_sha256",
            "analyzer_fingerprint_sha256",
            "contracts",
        }
        if set(payload) != expected:
            raise ValueError("StageEdit feature provenance schema does not match")
        if payload.get("producer") != FEATURE_MATERIALIZER_CONTRACT:
            raise ValueError("StageEdit feature producer is unsupported")
        if payload.get("input_availability") != (
            "available_before_editorial_source_choice"
        ):
            raise ValueError("StageEdit feature availability contract is unsafe")
        if payload.get("reference_attribution_inputs_present") is not False:
            raise ValueError("reference-attribution inputs are prohibited")
        upstream = payload.get("upstream_sha256")
        contracts = payload.get("contracts")
        if not isinstance(upstream, Mapping) or not isinstance(contracts, list):
            raise ValueError("StageEdit feature provenance values are invalid")
        return cls(
            upstream_sha256=tuple(
                (str(name), str(digest)) for name, digest in upstream.items()
            ),
            analyzer_fingerprint_sha256=(
                str(payload["analyzer_fingerprint_sha256"])
                if payload.get("analyzer_fingerprint_sha256") is not None
                else None
            ),
            contracts=tuple(str(value) for value in contracts),
        )


@dataclass(frozen=True, slots=True)
class SourceFeatureObservation:
    """One source's inference-time observations at a decision timestamp.

    Every optional scalar maps to a named ``StageEditTokenSchema`` source
    column.  ``None`` means genuinely unavailable and produces a zero paired
    with a true missing bit.  ``available`` and ``hard_gate_passed`` remain
    distinct because the Transformer exposes both masks.
    """

    source_id: str
    sync_confidence: float | None = None
    sync_residual_normalized: float | None = None
    shot_scale: float | None = None
    face_scale: float | None = None
    face_center_x: float | None = None
    face_center_y: float | None = None
    body_center_x: float | None = None
    body_center_y: float | None = None
    pose_motion: float | None = None
    camera_motion_x: float | None = None
    camera_motion_y: float | None = None
    camera_zoom_velocity: float | None = None
    camera_roll_degrees_normalized: float | None = None
    sharpness: float | None = None
    exposure: float | None = None
    color_temperature_normalized: float | None = None
    occlusion_probability: float | None = None
    automatic_visual_confidence: float | None = None
    pose_model_confidence: float | None = None
    detected_person_count_normalized: float | None = None
    formation_person_count_normalized: float | None = None
    formation_pose_coverage: float | None = None
    pose_model_embedding: tuple[float, ...] = ()
    formation_layout_embedding: tuple[float, ...] = ()
    formation_pose_embedding: tuple[float | None, ...] = ()
    available: bool = True
    hard_gate_passed: bool = True

    def __post_init__(self) -> None:
        source_id = str(self.source_id).strip()
        if not source_id:
            raise ValueError("StageEdit source_id cannot be blank")
        object.__setattr__(self, "source_id", source_id)
        if not isinstance(self.available, bool) or not isinstance(
            self.hard_gate_passed, bool
        ):
            raise TypeError("StageEdit source availability gates must be booleans")
        unit_fields = (
            "sync_confidence",
            "sync_residual_normalized",
            "shot_scale",
            "face_scale",
            "face_center_x",
            "face_center_y",
            "body_center_x",
            "body_center_y",
            "pose_motion",
            "sharpness",
            "exposure",
            "occlusion_probability",
            "automatic_visual_confidence",
            "pose_model_confidence",
            "detected_person_count_normalized",
            "formation_person_count_normalized",
            "formation_pose_coverage",
        )
        signed_fields = (
            "camera_motion_x",
            "camera_motion_y",
            "camera_zoom_velocity",
            "camera_roll_degrees_normalized",
            "color_temperature_normalized",
        )
        for name in unit_fields:
            value = getattr(self, name)
            normalized = _unit_or_none(value)
            if value is not None and normalized is None:
                raise ValueError(f"{name} must be finite when supplied")
            object.__setattr__(self, name, normalized)
        pose = tuple(self.pose_model_embedding)
        if pose and len(pose) != len(STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES):
            raise ValueError("pose model embedding must contain exactly 60 values")
        normalized_pose: list[float] = []
        for value in pose:
            normalized = _signed_unit_or_none(value)
            if normalized is None:
                raise ValueError("pose model embedding must be finite")
            normalized_pose.append(normalized)
        object.__setattr__(self, "pose_model_embedding", tuple(normalized_pose))

        layout = tuple(self.formation_layout_embedding)
        if layout and len(layout) != len(STAGE_EDIT_FORMATION_LAYOUT_FEATURE_NAMES):
            raise ValueError(
                "formation layout embedding must contain exactly 134 values"
            )
        normalized_layout: list[float] = []
        for value in layout:
            normalized = _unit_or_none(value)
            if normalized is None:
                raise ValueError("formation layout embedding must be finite")
            normalized_layout.append(normalized)
        object.__setattr__(
            self, "formation_layout_embedding", tuple(normalized_layout)
        )

        formation_pose = tuple(self.formation_pose_embedding)
        if formation_pose and len(formation_pose) != len(
            STAGE_EDIT_FORMATION_POSE_FEATURE_NAMES
        ):
            raise ValueError(
                "formation pose embedding must contain exactly 144 values"
            )
        normalized_formation_pose: list[float | None] = []
        for index, value in enumerate(formation_pose):
            if value is None:
                normalized_formation_pose.append(None)
                continue
            normalized = (
                _unit_or_none(value)
                if index % 3 == 2
                else _signed_unit_or_none(value)
            )
            if normalized is None:
                raise ValueError("formation pose embedding must be finite")
            normalized_formation_pose.append(normalized)
        object.__setattr__(
            self,
            "formation_pose_embedding",
            tuple(normalized_formation_pose),
        )
        for name in signed_fields:
            value = getattr(self, name)
            normalized = _signed_unit_or_none(value)
            if value is not None and normalized is None:
                raise ValueError(f"{name} must be finite when supplied")
            object.__setattr__(self, name, normalized)

    @property
    def observed_features(self) -> dict[str, float]:
        observed = {
            name: value
            for name in (
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
            )
            if (value := getattr(self, name)) is not None
        }
        # The existing token schema uses this historical column name.  Its
        # materialized semantic is strictly automatic, pre-choice confidence.
        if self.automatic_visual_confidence is not None:
            observed["visual_match_confidence"] = (
                self.automatic_visual_confidence
            )
        for name in (
            "pose_model_confidence",
            "detected_person_count_normalized",
            "formation_person_count_normalized",
            "formation_pose_coverage",
        ):
            if (value := getattr(self, name)) is not None:
                observed[name] = value
        observed.update(
            zip(
                STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES,
                self.pose_model_embedding,
                strict=False,
            )
        )
        observed.update(
            zip(
                STAGE_EDIT_FORMATION_LAYOUT_FEATURE_NAMES,
                self.formation_layout_embedding,
                strict=False,
            )
        )
        observed.update(
            {
                name: value
                for name, value in zip(
                    STAGE_EDIT_FORMATION_POSE_FEATURE_NAMES,
                    self.formation_pose_embedding,
                    strict=False,
                )
                if value is not None
            }
        )
        return observed


@dataclass(frozen=True, slots=True)
class CausalEditDecision:
    """An already-issued edit decision; future/current labels cannot enter."""

    cut_time_seconds: float
    source_id: str | None = None
    face_scale: float | None = None
    face_center_x: float | None = None
    face_center_y: float | None = None
    zoom: float | None = None
    roll_degrees_normalized: float | None = None
    transition_strength: float | None = None

    def __post_init__(self) -> None:
        timestamp = _finite_or_none(self.cut_time_seconds)
        if timestamp is None or timestamp < 0.0:
            raise ValueError("causal edit time must be non-negative and finite")
        object.__setattr__(self, "cut_time_seconds", timestamp)
        if self.source_id is not None:
            source_id = str(self.source_id).strip()
            object.__setattr__(self, "source_id", source_id or None)
        for name in (
            "face_scale",
            "face_center_x",
            "face_center_y",
            "zoom",
            "transition_strength",
        ):
            value = getattr(self, name)
            normalized = _unit_or_none(value)
            if value is not None and normalized is None:
                raise ValueError(f"{name} must be finite when supplied")
            object.__setattr__(self, name, normalized)
        roll = self.roll_degrees_normalized
        normalized_roll = _signed_unit_or_none(roll)
        if roll is not None and normalized_roll is None:
            raise ValueError("roll_degrees_normalized must be finite")
        object.__setattr__(self, "roll_degrees_normalized", normalized_roll)


@dataclass(frozen=True, slots=True)
class StageEditFeatureFrame:
    master_time_seconds: float
    source_observations: tuple[SourceFeatureObservation, ...]
    causal_history: tuple[CausalEditDecision, ...] = ()

    def __post_init__(self) -> None:
        timestamp = _finite_or_none(self.master_time_seconds)
        if timestamp is None or timestamp < 0.0:
            raise ValueError("StageEdit frame time must be non-negative and finite")
        observations = tuple(self.source_observations)
        if not observations:
            raise ValueError("StageEdit feature frame needs at least one source")
        if any(not isinstance(item, SourceFeatureObservation) for item in observations):
            raise TypeError("source observations must use SourceFeatureObservation")
        ids = tuple(item.source_id for item in observations)
        if len(set(ids)) != len(ids):
            raise ValueError("StageEdit source observations must be unique")
        history = tuple(self.causal_history)
        if any(not isinstance(item, CausalEditDecision) for item in history):
            raise TypeError("causal history must use CausalEditDecision")
        if any(
            right.cut_time_seconds < left.cut_time_seconds
            for left, right in zip(history, history[1:])
        ):
            raise ValueError("causal edit history must be time ordered")
        if any(item.cut_time_seconds >= timestamp for item in history):
            raise ValueError("causal history cannot contain current/future decisions")
        object.__setattr__(self, "master_time_seconds", timestamp)
        object.__setattr__(
            self,
            "source_observations",
            tuple(sorted(observations, key=lambda item: item.source_id)),
        )
        object.__setattr__(self, "causal_history", history)


def _bbox_center(
    bbox: Sequence[Any] | None,
) -> tuple[float, float, float] | None:
    if bbox is None or len(bbox) != 4:
        return None
    values = tuple(_finite_or_none(item) for item in bbox)
    if any(value is None for value in values):
        return None
    x, y, width, height = (float(value) for value in values)
    if width <= 0.0 or height <= 0.0:
        return None
    center_x = _unit_or_none(x + width / 2.0)
    center_y = _unit_or_none(y + height / 2.0)
    scale = _unit_or_none(height)
    assert center_x is not None and center_y is not None and scale is not None
    return (center_x, center_y, scale)


def _cosine_motion(
    earlier: Sequence[Any],
    later: Sequence[Any],
) -> float | None:
    if not earlier or len(earlier) != len(later):
        return None
    left = tuple(_finite_or_none(value) for value in earlier)
    right = tuple(_finite_or_none(value) for value in later)
    if any(value is None for value in (*left, *right)):
        return None
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    cosine = min(1.0, max(-1.0, dot / (left_norm * right_norm)))
    return (1.0 - cosine) / 2.0


def _fixed_embedding(
    value: Any,
    *,
    expected_length: int,
    name: str,
    signed: bool,
) -> tuple[float, ...]:
    """Validate a real analyzer embedding without padding or guessing it."""

    if value is None or (isinstance(value, (list, tuple)) and not value):
        return ()
    if not isinstance(value, (list, tuple)) or len(value) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} values")
    output: list[float] = []
    normalizer = _signed_unit_or_none if signed else _unit_or_none
    for item in value:
        normalized = normalizer(item)
        if normalized is None:
            raise ValueError(f"{name} must contain finite numeric values")
        output.append(normalized)
    return tuple(output)


def _formation_person_value(person: Any, name: str) -> Any:
    if isinstance(person, Mapping):
        return person.get(name)
    return getattr(person, name, None)


def _screen_ordered_formation_people(people: Any) -> tuple[Any, ...]:
    if people is None or (isinstance(people, (list, tuple)) and not people):
        return ()
    if not isinstance(people, (list, tuple)) or len(people) > 6:
        raise ValueError("formation_people must contain at most six performers")
    ordered: list[tuple[float, float, Any]] = []
    for person in people:
        centroid = _formation_person_value(person, "centroid_xy")
        if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
            raise ValueError("formation performer requires a screen-space centroid")
        x = _unit_or_none(centroid[0])
        y = _unit_or_none(centroid[1])
        if x is None or y is None:
            raise ValueError("formation performer centroid must be finite")
        ordered.append((x, y, person))
    ordered.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in ordered)


def _fixed_formation_pose_embedding(
    people: Any,
) -> tuple[float | None, ...]:
    """Flatten at most six left-to-right upper-body skeletons.

    A detected performer whose pose estimator failed receives 24 explicit
    missing values.  We never synthesize a neutral skeleton: the paired model
    missing mask is the only representation of unavailable evidence.
    """

    if people is None or (isinstance(people, (list, tuple)) and not people):
        return ()
    output: list[float | None] = []
    for person in people:
        points = _formation_person_value(person, "upper_body_keypoints_xyc")
        if points is None or (isinstance(points, (list, tuple)) and not points):
            output.extend((None,) * (8 * 3))
            continue
        if not isinstance(points, (list, tuple)) or len(points) != 8:
            raise ValueError(
                "formation upper-body pose must contain exactly eight joints"
            )
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 3:
                raise ValueError(
                    "formation upper-body joints require x/y/confidence"
                )
            x = _signed_unit_or_none(point[0])
            y = _signed_unit_or_none(point[1])
            confidence = _unit_or_none(point[2])
            if x is None or y is None or confidence is None:
                raise ValueError("formation upper-body values must be finite")
            output.extend((x, y, confidence))
    output.extend((None,) * (len(STAGE_EDIT_FORMATION_POSE_FEATURE_NAMES) - len(output)))
    return tuple(output)


def _action_formation_observations(
    *,
    pose_embedding: Any,
    pose_backend: Any,
    pose_confidence: Any,
    detected_person_count: Any,
    person_layout_embedding: Any,
    formation_people: Any,
) -> dict[str, Any]:
    """Build identity-free action/formation features shared by train and serve."""

    backend = str(pose_backend or "").strip()
    raw_pose = pose_embedding
    if backend == "structure_hog_fallback":
        # The legacy 96-D HOG vector is not a human pose and must not silently
        # occupy the new skeleton columns.
        pose = ()
        confidence = None
    elif (
        not backend
        or raw_pose is None
        or (isinstance(raw_pose, (list, tuple)) and not raw_pose)
    ):
        pose = ()
        confidence = None
    else:
        pose = _fixed_embedding(
            raw_pose,
            expected_length=len(STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES),
            name="model pose embedding",
            signed=True,
        )
        confidence = _unit_or_none(pose_confidence)
        if pose_confidence is not None and confidence is None:
            raise ValueError("model pose confidence must be finite when supplied")

    layout = _fixed_embedding(
        person_layout_embedding,
        expected_length=len(STAGE_EDIT_FORMATION_LAYOUT_FEATURE_NAMES),
        name="formation layout embedding",
        signed=False,
    )
    ordered_people = _screen_ordered_formation_people(formation_people)
    formation = _fixed_formation_pose_embedding(ordered_people)

    count: int | None
    if detected_person_count is None:
        count = None
    elif (
        isinstance(detected_person_count, bool)
        or not isinstance(detected_person_count, int)
        or detected_person_count < 0
    ):
        raise ValueError("detected_person_count must be a non-negative integer")
    else:
        count = detected_person_count

    people_count = len(ordered_people)
    upper_body_count = (
        sum(
            bool(_formation_person_value(person, "upper_body_keypoints_xyc"))
            for person in ordered_people
        )
        if isinstance(formation_people, (list, tuple))
        else 0
    )
    return {
        "pose_model_confidence": confidence,
        "detected_person_count_normalized": (
            min(6, count) / 6.0 if count is not None else None
        ),
        "formation_person_count_normalized": (
            min(6, people_count) / 6.0
            if isinstance(formation_people, (list, tuple))
            else None
        ),
        "formation_pose_coverage": (
            upper_body_count / people_count if people_count else 0.0
        ),
        "pose_model_embedding": pose,
        "formation_layout_embedding": layout,
        "formation_pose_embedding": formation,
    }


def source_observation_from_timeline_candidate(
    candidate: "TimelineCandidate",
    *,
    descriptor: "FrameVisualDescriptor | None" = None,
    sync_residual_normalized: float | None = None,
    sharpness: float | None = None,
    exposure: float | None = None,
    camera_roll_degrees_normalized: float | None = None,
    available: bool = True,
    hard_gate_passed: bool = True,
) -> SourceFeatureObservation:
    """Adapt existing automatic timeline/descriptor evidence without labels."""

    face = _bbox_center(
        getattr(candidate, "entry_focus_bbox_xywh", None)
        or getattr(candidate, "focus_bbox_xywh", None)
        or (getattr(descriptor, "primary_face_bbox", None) if descriptor else None)
    )
    subject = getattr(candidate, "entry_subject_centroid_xy", None) or getattr(
        candidate, "subject_centroid_xy", None
    )
    if subject is None and descriptor is not None:
        subject = getattr(descriptor, "silhouette_centroid_xy", None)
    subject_values = (
        tuple(_unit_or_none(value) for value in subject)
        if subject is not None and len(subject) == 2
        else (None, None)
    )
    entry_pose = getattr(candidate, "entry_pose_embedding", ())
    exit_pose = getattr(candidate, "exit_pose_embedding", ())
    pose_motion = _cosine_motion(entry_pose, exit_pose)
    motion = getattr(candidate, "entry_motion", None) or getattr(
        candidate, "motion", None
    )
    entry_scale = _finite_or_none(getattr(candidate, "entry_shot_scale", None))
    exit_scale = _finite_or_none(getattr(candidate, "exit_shot_scale", None))
    duration = _finite_or_none(getattr(candidate, "duration_seconds", None))
    zoom_velocity = (
        _signed_unit_or_none((exit_scale - entry_scale) / max(duration, 1e-6))
        if entry_scale is not None and exit_scale is not None and duration is not None
        else None
    )
    color = tuple(getattr(descriptor, "color_embedding", ())) if descriptor else ()
    inferred_exposure = (
        _unit_or_none(color[6]) if exposure is None and len(color) >= 7 else exposure
    )
    color_temperature = (
        _signed_unit_or_none((float(color[2]) - 0.5) * 2.0)
        if len(color) >= 3 and _finite_or_none(color[2]) is not None
        else None
    )
    occlusion = (
        getattr(descriptor, "foreground_obstruction_score", None)
        if descriptor is not None
        else None
    )
    # ``visual_match_confidence`` predates the reference adapter.  Here it is
    # strictly the automatic candidate's composition/boundary confidence; no
    # finished-reference matching evidence is accepted by this API.
    composition = _unit_or_none(getattr(candidate, "composition_score", None))
    boundary = _unit_or_none(getattr(candidate, "boundary_score", None))
    visual_confidence = (
        (composition + boundary) / 2.0
        if composition is not None and boundary is not None
        else composition if composition is not None else boundary
    )
    action_formation = _action_formation_observations(
        pose_embedding=(
            getattr(candidate, "entry_pose_embedding", ())
            or getattr(candidate, "pose_embedding", ())
        ),
        # TimelineCandidate does not preserve a backend string.  A real model
        # pose is unambiguously 60-D; the legacy HOG fallback is 96-D.
        pose_backend=(
            "timeline_model_pose"
            if len(
                getattr(candidate, "entry_pose_embedding", ())
                or getattr(candidate, "pose_embedding", ())
            )
            == len(STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES)
            else "structure_hog_fallback"
        ),
        pose_confidence=(
            None  # TimelineCandidate does not retain analyzer confidence.
        ),
        detected_person_count=(
            getattr(candidate, "entry_detected_person_count", 0)
            or getattr(candidate, "detected_person_count", 0)
        ),
        person_layout_embedding=(
            getattr(candidate, "entry_person_layout_embedding", ())
            or getattr(candidate, "person_layout_embedding", ())
        ),
        formation_people=(
            getattr(candidate, "entry_formation_people", ())
            or getattr(candidate, "formation_people", ())
        ),
    )
    return SourceFeatureObservation(
        source_id=str(getattr(candidate, "source_id")),
        sync_confidence=getattr(candidate, "sync_score", None),
        sync_residual_normalized=sync_residual_normalized,
        shot_scale=(
            entry_scale
            if entry_scale is not None
            else getattr(candidate, "shot_scale", None)
        ),
        face_scale=face[2] if face is not None else None,
        face_center_x=face[0] if face is not None else None,
        face_center_y=face[1] if face is not None else None,
        body_center_x=subject_values[0],
        body_center_y=subject_values[1],
        pose_motion=pose_motion,
        camera_motion_x=motion[0] if motion is not None else None,
        camera_motion_y=motion[1] if motion is not None else None,
        camera_zoom_velocity=zoom_velocity,
        camera_roll_degrees_normalized=camera_roll_degrees_normalized,
        sharpness=sharpness,
        exposure=inferred_exposure,
        color_temperature_normalized=color_temperature,
        occlusion_probability=occlusion,
        automatic_visual_confidence=visual_confidence,
        **action_formation,
        available=available,
        hard_gate_passed=hard_gate_passed,
    )


def source_observation_from_visual_cache_payload(
    source_id: str,
    numeric_features: Mapping[str, Any],
    *,
    following_numeric_features: Mapping[str, Any] | None = None,
    interval_seconds: float | None = None,
    sync_confidence: float | None = None,
    sync_residual_normalized: float | None = None,
    sharpness: float | None = None,
    camera_roll_degrees_normalized: float | None = None,
    available: bool = True,
    hard_gate_passed: bool = True,
) -> SourceFeatureObservation:
    """Adapt numeric output returned by ``PersistentVisualFeatureCache.get``.

    Only documented source-local descriptor fields are read.  Any unrelated
    mapping keys -- including reference-match evidence -- have no route into a
    model column.  A following descriptor optionally supplies pose/zoom motion.
    """

    if not isinstance(numeric_features, Mapping):
        raise TypeError("visual cache numeric_features must be a mapping")
    following = following_numeric_features or {}
    if not isinstance(following, Mapping):
        raise TypeError("following visual cache features must be a mapping")
    face = _bbox_center(numeric_features.get("primary_face_bbox"))
    body = numeric_features.get("silhouette_centroid_xy")
    if not isinstance(body, (list, tuple)) or len(body) != 2:
        body = None
    motion = numeric_features.get("motion")
    if not isinstance(motion, (list, tuple)) or len(motion) != 2:
        motion = None
    color = numeric_features.get("color_embedding")
    if not isinstance(color, (list, tuple)):
        color = ()
    pose = numeric_features.get("pose_embedding")
    next_pose = following.get("pose_embedding")
    pose_motion = (
        _cosine_motion(pose, next_pose)
        if isinstance(pose, (list, tuple))
        and isinstance(next_pose, (list, tuple))
        else None
    )
    duration = _finite_or_none(interval_seconds)
    current_scale = _unit_or_none(numeric_features.get("shot_scale"))
    next_scale = _unit_or_none(following.get("shot_scale"))
    zoom_velocity = (
        _signed_unit_or_none((next_scale - current_scale) / duration)
        if duration is not None
        and duration > 0.0
        and current_scale is not None
        and next_scale is not None
        else None
    )
    quality = _unit_or_none(numeric_features.get("quality_score"))
    composition = _unit_or_none(numeric_features.get("composition_score"))
    visual_confidence = (
        0.40 * quality + 0.60 * composition
        if quality is not None and composition is not None
        else composition if composition is not None else quality
    )
    exposure = _unit_or_none(color[6]) if len(color) >= 7 else None
    temperature = (
        _signed_unit_or_none((float(color[2]) - 0.5) * 2.0)
        if len(color) >= 3 and _finite_or_none(color[2]) is not None
        else None
    )
    action_formation = _action_formation_observations(
        pose_embedding=numeric_features.get("pose_embedding"),
        pose_backend=numeric_features.get("pose_backend"),
        pose_confidence=numeric_features.get("pose_confidence"),
        detected_person_count=numeric_features.get("detected_person_count"),
        person_layout_embedding=numeric_features.get("person_layout_embedding"),
        formation_people=numeric_features.get("formation_people"),
    )
    return SourceFeatureObservation(
        source_id=source_id,
        sync_confidence=sync_confidence,
        sync_residual_normalized=sync_residual_normalized,
        shot_scale=current_scale,
        face_scale=face[2] if face is not None else None,
        face_center_x=face[0] if face is not None else None,
        face_center_y=face[1] if face is not None else None,
        body_center_x=body[0] if body is not None else None,
        body_center_y=body[1] if body is not None else None,
        pose_motion=pose_motion,
        camera_motion_x=motion[0] if motion is not None else None,
        camera_motion_y=motion[1] if motion is not None else None,
        camera_zoom_velocity=zoom_velocity,
        camera_roll_degrees_normalized=camera_roll_degrees_normalized,
        sharpness=sharpness,
        exposure=exposure,
        color_temperature_normalized=temperature,
        occlusion_probability=numeric_features.get(
            "foreground_obstruction_score"
        ),
        automatic_visual_confidence=visual_confidence,
        **action_formation,
        available=available,
        hard_gate_passed=hard_gate_passed,
    )


def materialize_stage_edit_source_candidate(
    observation: SourceFeatureObservation,
    *,
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
) -> SourceCandidateToken:
    values, missing = _feature_vector(
        schema.source_feature_names,
        observation.observed_features,
    )
    return SourceCandidateToken(
        source_id=observation.source_id,
        features=values,
        missing_mask=missing,
        available=observation.available,
        hard_gate_passed=observation.hard_gate_passed,
    )


def _interpolate_onset(rhythm: "RhythmAnalysis", timestamp: float) -> float | None:
    values = rhythm.onset_envelope.values
    if not values:
        return None
    position = max(0.0, timestamp * rhythm.onset_envelope.frame_rate)
    lower = min(len(values) - 1, int(math.floor(position)))
    upper = min(len(values) - 1, lower + 1)
    fraction = position - lower
    value = float(values[lower]) * (1.0 - fraction) + float(values[upper]) * fraction
    return _unit_or_none(value)


def _music_observations(
    timestamp: float,
    rhythm: "RhythmAnalysis | None",
    *,
    performance_start_seconds: float,
    performance_end_seconds: float,
) -> dict[str, float]:
    duration = performance_end_seconds - performance_start_seconds
    progress = min(
        1.0,
        max(0.0, (timestamp - performance_start_seconds) / max(duration, 1e-9)),
    )
    observed: dict[str, float] = {"phrase_progress": progress}
    if rhythm is None:
        return observed
    period = _finite_or_none(rhythm.beat_period_seconds)
    beats = tuple(rhythm.beats)
    if period is None or period <= 0.0 or not beats:
        tempo = _finite_or_none(rhythm.bpm)
        if tempo is not None:
            observed["tempo_normalized"] = min(1.0, max(0.0, (tempo - 60.0) / 140.0))
        onset = _interpolate_onset(rhythm, timestamp)
        if onset is not None:
            observed["onset_strength"] = onset
            observed["spectral_flux"] = onset
        return observed

    preceding = max(
        (beat for beat in beats if beat.time_seconds <= timestamp),
        key=lambda beat: beat.time_seconds,
        default=beats[0],
    )
    phase = ((timestamp - preceding.time_seconds) / period) % 1.0
    angle = 2.0 * math.pi * phase
    observed["beat_phase_sin"] = math.sin(angle)
    observed["beat_phase_cos"] = math.cos(angle)
    bar_phase = ((preceding.index % 4) + phase) / 4.0
    observed["bar_phase_sin"] = math.sin(2.0 * math.pi * bar_phase)
    observed["bar_phase_cos"] = math.cos(2.0 * math.pi * bar_phase)
    nearest = min(beats, key=lambda beat: abs(beat.time_seconds - timestamp))
    proximity = max(0.0, 1.0 - abs(nearest.time_seconds - timestamp) / period)
    observed["downbeat_probability"] = (
        _unit_or_none(nearest.strength) or 0.0
    ) * proximity * (1.0 if nearest.index % 4 == 0 else 0.0)
    onset = _interpolate_onset(rhythm, timestamp)
    if onset is not None:
        observed["onset_strength"] = onset
        observed["spectral_flux"] = onset
    observed["tempo_normalized"] = min(
        1.0, max(0.0, (float(rhythm.bpm) - 60.0) / 140.0)
    )
    following_times = [beat.time_seconds for beat in beats if beat.time_seconds > timestamp]
    next_beat = min(following_times) if following_times else timestamp + period
    observed["seconds_to_next_beat_normalized"] = min(
        1.0, max(0.0, (next_beat - timestamp) / period)
    )
    phrases = [
        item
        for item in rhythm.phrase_candidates
        if item.start_seconds <= timestamp < item.end_seconds
    ]
    if phrases:
        phrase = max(phrases, key=lambda item: (item.beat_count, -item.start_seconds))
        observed["phrase_progress"] = min(
            1.0,
            max(
                0.0,
                (timestamp - phrase.start_seconds)
                / max(phrase.end_seconds - phrase.start_seconds, 1e-9),
            ),
        )
    return observed


def _section_observations(
    timestamp: float,
    section_plan: SectionPlan | None,
) -> dict[str, float]:
    if section_plan is None:
        return {}
    bounded = min(section_plan.end_seconds, max(section_plan.start_seconds, timestamp))
    directive = section_plan.directive_at(bounded)
    role_name = {
        SectionRole.INTRO: "intro_probability",
        SectionRole.VERSE_LIKE: "verse_probability",
        SectionRole.BUILD: "prechorus_probability",
        SectionRole.CHORUS_LIKE: "chorus_probability",
        SectionRole.BRIDGE_LIKE: "bridge_probability",
        SectionRole.OUTRO: "outro_probability",
    }[directive.role]
    probability_names = (
        "intro_probability",
        "verse_probability",
        "prechorus_probability",
        "chorus_probability",
        "bridge_probability",
        "outro_probability",
    )
    observed = {name: 0.0 for name in probability_names}
    observed[role_name] = directive.confidence
    observed["section_progress"] = min(
        1.0,
        max(
            0.0,
            (timestamp - directive.start_seconds)
            / max(directive.end_seconds - directive.start_seconds, 1e-9),
        ),
    )
    previous_energy = (
        section_plan.sections[directive.section_index - 1].energy_score
        if directive.section_index > 0
        else directive.energy_score
    )
    observed["section_energy_delta"] = min(
        1.0, max(-1.0, directive.energy_score - previous_energy)
    )
    return observed


def _recent_edit_observations(
    timestamp: float,
    history: Sequence[CausalEditDecision],
) -> dict[str, float]:
    earlier = tuple(item for item in history if item.cut_time_seconds < timestamp)
    observed: dict[str, float] = {
        "recent_cut_density": min(
            1.0,
            sum(timestamp - 8.0 <= item.cut_time_seconds < timestamp for item in earlier)
            / 8.0,
        )
    }
    if not earlier:
        return observed
    previous = earlier[-1]
    observed["seconds_since_cut_normalized"] = min(
        1.0, max(0.0, (timestamp - previous.cut_time_seconds) / 8.0)
    )
    if previous.source_id is not None:
        run_start = previous.cut_time_seconds
        for decision in reversed(earlier[:-1]):
            if decision.source_id != previous.source_id:
                break
            run_start = decision.cut_time_seconds
        observed["same_source_run_normalized"] = min(
            1.0, max(0.0, (timestamp - run_start) / 8.0)
        )
        observed["source_reuse_count_normalized"] = min(
            1.0,
            sum(item.source_id == previous.source_id for item in earlier[-8:]) / 8.0,
        )
    for source_name, feature_name in (
        ("face_scale", "previous_face_scale"),
        ("face_center_x", "previous_face_center_x"),
        ("face_center_y", "previous_face_center_y"),
        ("zoom", "previous_zoom"),
        ("roll_degrees_normalized", "previous_roll_normalized"),
        ("transition_strength", "previous_transition_strength"),
    ):
        value = getattr(previous, source_name)
        if value is not None:
            observed[feature_name] = value
    return observed


def materialize_stage_edit_token(
    *,
    master_time_seconds: float,
    source_observations: Sequence[SourceFeatureObservation],
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
    rhythm: "RhythmAnalysis | None" = None,
    section_plan: SectionPlan | None = None,
    causal_history: Sequence[CausalEditDecision] = (),
    style_features: Mapping[str, Any] | None = None,
    performance_start_seconds: float = 0.0,
    performance_end_seconds: float | None = None,
) -> StageEditToken:
    """Materialize exactly one train/serve token with explicit missing bits."""

    timestamp = _finite_or_none(master_time_seconds)
    start = _finite_or_none(performance_start_seconds)
    end = _finite_or_none(performance_end_seconds)
    if timestamp is None or timestamp < 0.0:
        raise ValueError("master time must be non-negative and finite")
    if start is None or start < 0.0:
        raise ValueError("performance start must be non-negative and finite")
    if end is None:
        end = max(timestamp, start) + 1.0
    if end <= start:
        raise ValueError("performance feature span must be positive")
    observations = tuple(source_observations)
    if not observations:
        raise ValueError("StageEdit token needs at least one source observation")
    if any(not isinstance(item, SourceFeatureObservation) for item in observations):
        raise TypeError("source observations must use SourceFeatureObservation")
    ids = tuple(item.source_id for item in observations)
    if len(set(ids)) != len(ids):
        raise ValueError("source observations must have unique ids")
    history = tuple(causal_history)
    if any(not isinstance(item, CausalEditDecision) for item in history):
        raise TypeError("causal history must use CausalEditDecision")
    if any(item.cut_time_seconds >= timestamp for item in history):
        raise ValueError("current or future decisions cannot enter StageEdit features")
    if any(
        right.cut_time_seconds < left.cut_time_seconds
        for left, right in zip(history, history[1:])
    ):
        raise ValueError("causal edit history must be time ordered")

    music, music_missing = _feature_vector(
        schema.music_feature_names,
        _music_observations(
            timestamp,
            rhythm,
            performance_start_seconds=start,
            performance_end_seconds=end,
        ),
    )
    section, section_missing = _feature_vector(
        schema.section_feature_names,
        _section_observations(timestamp, section_plan),
    )
    recent, recent_missing = _feature_vector(
        schema.recent_edit_feature_names,
        _recent_edit_observations(timestamp, history),
    )
    style, style_missing = _feature_vector(
        schema.style_feature_names,
        dict(style_features or {}),
    )
    candidates = tuple(
        materialize_stage_edit_source_candidate(item, schema=schema)
        for item in sorted(observations, key=lambda item: item.source_id)
    )
    return StageEditToken(
        master_time_seconds=timestamp,
        music_features=music,
        section_features=section,
        source_candidates=candidates,
        recent_edit_features=recent,
        style_features=style,
        missing_mask=music_missing + section_missing + recent_missing + style_missing,
    )


def _candidate_to_dict(candidate: SourceCandidateToken) -> dict[str, Any]:
    return {
        "source_id": candidate.source_id,
        "features": list(candidate.features),
        "missing_mask": list(candidate.missing_mask),
        "available": candidate.available,
        "hard_gate_passed": candidate.hard_gate_passed,
    }


def _token_to_dict(token: StageEditToken) -> dict[str, Any]:
    return {
        "master_time_seconds": token.master_time_seconds,
        "music_features": list(token.music_features),
        "section_features": list(token.section_features),
        "source_candidates": [
            _candidate_to_dict(candidate) for candidate in token.source_candidates
        ],
        "recent_edit_features": list(token.recent_edit_features),
        "style_features": list(token.style_features),
        "missing_mask": list(token.missing_mask),
    }


def _snapshot_unsigned_payload(
    *,
    sequence: StageEditSequence,
    token_schema_fingerprint: str,
    provenance: StageEditFeatureProvenance,
) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_MATERIALIZER_SCHEMA_VERSION,
        "snapshot_kind": FEATURE_SNAPSHOT_KIND,
        "token_schema_fingerprint": token_schema_fingerprint,
        "sequence_id": sequence.sequence_id,
        "provenance": provenance.to_dict(),
        "tokens": [_token_to_dict(token) for token in sequence.tokens],
        "input_policy": {
            "training_inference_materializer_shared": True,
            "reference_attribution_match_inputs_present": False,
            "selected_labels_present": False,
            "missing_values_are_explicitly_masked": True,
            "source_availability_and_hard_gate_are_distinct": True,
        },
    }


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serialized_numbers(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a numeric array")
    output: list[float] = []
    for item in value:
        number = _finite_or_none(item)
        if number is None:
            raise ValueError(f"{name} contains a non-finite number")
        output.append(number)
    return tuple(output)


def _serialized_bools(value: Any, name: str) -> tuple[bool, ...]:
    if not isinstance(value, list) or any(not isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must be a boolean array")
    return tuple(value)


def _validate_token(token: StageEditToken, schema: StageEditTokenSchema) -> None:
    timestamp = _finite_or_none(token.master_time_seconds)
    if timestamp is None or timestamp < 0.0:
        raise ValueError("StageEdit snapshot token time is invalid")
    expected = (
        len(schema.music_feature_names),
        len(schema.section_feature_names),
        len(schema.recent_edit_feature_names),
        len(schema.style_feature_names),
    )
    actual = (
        len(token.music_features),
        len(token.section_features),
        len(token.recent_edit_features),
        len(token.style_features),
    )
    if actual != expected or len(token.missing_mask) != schema.raw_context_dimension:
        raise ValueError("StageEdit snapshot context dimensions do not match schema")
    if any(not isinstance(value, bool) for value in token.missing_mask):
        raise ValueError("StageEdit snapshot context missing mask must be boolean")
    context = (
        token.music_features
        + token.section_features
        + token.recent_edit_features
        + token.style_features
    )
    if any(_finite_or_none(value) is None for value in context):
        raise ValueError("StageEdit snapshot context values must be finite")
    if not token.source_candidates:
        raise ValueError("StageEdit snapshot token has no source candidates")
    ids: set[str] = set()
    ordered_ids: list[str] = []
    for candidate in token.source_candidates:
        if candidate.source_id in ids or not candidate.source_id.strip():
            raise ValueError("StageEdit snapshot candidate ids are invalid")
        ids.add(candidate.source_id)
        ordered_ids.append(candidate.source_id)
        if (
            len(candidate.features) != schema.raw_source_dimension
            or len(candidate.missing_mask) != schema.raw_source_dimension
        ):
            raise ValueError("StageEdit snapshot source dimensions do not match schema")
        if any(not isinstance(value, bool) for value in candidate.missing_mask):
            raise ValueError("StageEdit snapshot source missing mask must be boolean")
        if any(_finite_or_none(value) is None for value in candidate.features):
            raise ValueError("StageEdit snapshot source values must be finite")
        if not isinstance(candidate.available, bool) or not isinstance(
            candidate.hard_gate_passed, bool
        ):
            raise ValueError("StageEdit snapshot source gates must be booleans")
    if tuple(ordered_ids) != tuple(sorted(ordered_ids)):
        raise ValueError("StageEdit snapshot source order must be canonical")


@dataclass(frozen=True, slots=True)
class StageEditFeatureSnapshot:
    sequence: StageEditSequence
    token_schema_fingerprint: str
    provenance: StageEditFeatureProvenance
    snapshot_sha256: str

    def __post_init__(self) -> None:
        if not self.sequence.sequence_id.strip() or not self.sequence.tokens:
            raise ValueError("StageEdit feature snapshot sequence cannot be empty")
        if _SHA256.fullmatch(self.token_schema_fingerprint) is None:
            raise ValueError("StageEdit token schema fingerprint must be SHA-256")
        if _SHA256.fullmatch(self.snapshot_sha256) is None:
            raise ValueError("StageEdit feature snapshot hash must be SHA-256")
        previous = -math.inf
        for token in self.sequence.tokens:
            timestamp = _finite_or_none(token.master_time_seconds)
            if timestamp is None or timestamp <= previous:
                raise ValueError("snapshot token times must be finite and unique")
            previous = timestamp
        unsigned = _snapshot_unsigned_payload(
            sequence=self.sequence,
            token_schema_fingerprint=self.token_schema_fingerprint,
            provenance=self.provenance,
        )
        if _payload_sha256(unsigned) != self.snapshot_sha256:
            raise ValueError("StageEdit feature snapshot SHA-256 mismatch")

    @classmethod
    def build(
        cls,
        sequence: StageEditSequence,
        *,
        schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
        provenance: StageEditFeatureProvenance = StageEditFeatureProvenance(),
    ) -> "StageEditFeatureSnapshot":
        for token in sequence.tokens:
            _validate_token(token, schema)
        unsigned = _snapshot_unsigned_payload(
            sequence=sequence,
            token_schema_fingerprint=schema.fingerprint,
            provenance=provenance,
        )
        return cls(
            sequence=sequence,
            token_schema_fingerprint=schema.fingerprint,
            provenance=provenance,
            snapshot_sha256=_payload_sha256(unsigned),
        )

    def require_schema(self, schema: StageEditTokenSchema) -> None:
        if self.token_schema_fingerprint != schema.fingerprint:
            raise ValueError("StageEdit feature snapshot token schema mismatch")
        for token in self.sequence.tokens:
            _validate_token(token, schema)

    def token_at(
        self,
        master_time_seconds: float,
        *,
        tolerance_seconds: float = 1e-6,
    ) -> StageEditToken:
        timestamp = _finite_or_none(master_time_seconds)
        if timestamp is None or tolerance_seconds < 0.0:
            raise ValueError("StageEdit snapshot lookup is invalid")
        matches = tuple(
            token
            for token in self.sequence.tokens
            if abs(token.master_time_seconds - timestamp) <= tolerance_seconds
        )
        if len(matches) != 1:
            raise ValueError(
                f"StageEdit feature snapshot has no unique token at {timestamp:.9f}s"
            )
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        payload = _snapshot_unsigned_payload(
            sequence=self.sequence,
            token_schema_fingerprint=self.token_schema_fingerprint,
            provenance=self.provenance,
        )
        if _payload_sha256(payload) != self.snapshot_sha256:
            raise ValueError("StageEdit feature snapshot mutated after hashing")
        payload["snapshot_sha256"] = self.snapshot_sha256
        return payload

    def write_json(self, path: Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination.resolve()

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
    ) -> "StageEditFeatureSnapshot":
        expected = {
            "schema_version",
            "snapshot_kind",
            "token_schema_fingerprint",
            "sequence_id",
            "provenance",
            "tokens",
            "input_policy",
            "snapshot_sha256",
        }
        if set(payload) != expected:
            raise ValueError("StageEdit feature snapshot schema does not match")
        if payload.get("schema_version") != FEATURE_MATERIALIZER_SCHEMA_VERSION:
            raise ValueError("unsupported StageEdit feature snapshot version")
        if payload.get("snapshot_kind") != FEATURE_SNAPSHOT_KIND:
            raise ValueError("unsupported StageEdit feature snapshot kind")
        policy = payload.get("input_policy")
        expected_policy = _snapshot_unsigned_payload(
            sequence=StageEditSequence("policy", ()),
            token_schema_fingerprint=schema.fingerprint,
            provenance=StageEditFeatureProvenance(),
        )["input_policy"]
        if policy != expected_policy:
            raise ValueError("StageEdit feature snapshot input policy is unsafe")
        raw_tokens = payload.get("tokens")
        raw_provenance = payload.get("provenance")
        if not isinstance(raw_tokens, list) or not isinstance(raw_provenance, Mapping):
            raise ValueError("StageEdit feature snapshot payload is invalid")
        tokens: list[StageEditToken] = []
        for raw_token in raw_tokens:
            if not isinstance(raw_token, Mapping) or set(raw_token) != {
                "master_time_seconds",
                "music_features",
                "section_features",
                "source_candidates",
                "recent_edit_features",
                "style_features",
                "missing_mask",
            }:
                raise ValueError("StageEdit snapshot token schema does not match")
            raw_candidates = raw_token["source_candidates"]
            if not isinstance(raw_candidates, list):
                raise ValueError("StageEdit snapshot candidates must be an array")
            candidates: list[SourceCandidateToken] = []
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, Mapping) or set(raw_candidate) != {
                    "source_id",
                    "features",
                    "missing_mask",
                    "available",
                    "hard_gate_passed",
                }:
                    raise ValueError("StageEdit snapshot candidate schema does not match")
                if not isinstance(raw_candidate["available"], bool) or not isinstance(
                    raw_candidate["hard_gate_passed"], bool
                ):
                    raise ValueError("StageEdit snapshot source gates must be booleans")
                if (
                    not isinstance(raw_candidate["source_id"], str)
                    or not raw_candidate["source_id"].strip()
                ):
                    raise ValueError("StageEdit snapshot source id must be a string")
                candidates.append(
                    SourceCandidateToken(
                        source_id=raw_candidate["source_id"],
                        features=_serialized_numbers(
                            raw_candidate["features"], "source features"
                        ),
                        missing_mask=_serialized_bools(
                            raw_candidate["missing_mask"], "source missing mask"
                        ),
                        available=raw_candidate["available"],
                        hard_gate_passed=raw_candidate["hard_gate_passed"],
                    )
                )
            serialized_time = _finite_or_none(raw_token["master_time_seconds"])
            if serialized_time is None:
                raise ValueError("StageEdit snapshot time must be finite")
            tokens.append(
                StageEditToken(
                    master_time_seconds=serialized_time,
                    music_features=_serialized_numbers(
                        raw_token["music_features"], "music features"
                    ),
                    section_features=_serialized_numbers(
                        raw_token["section_features"], "section features"
                    ),
                    source_candidates=tuple(candidates),
                    recent_edit_features=_serialized_numbers(
                        raw_token["recent_edit_features"], "recent edit features"
                    ),
                    style_features=_serialized_numbers(
                        raw_token["style_features"], "style features"
                    ),
                    missing_mask=_serialized_bools(
                        raw_token["missing_mask"], "context missing mask"
                    ),
                )
            )
        raw_sequence_id = payload.get("sequence_id")
        if not isinstance(raw_sequence_id, str) or not raw_sequence_id.strip():
            raise ValueError("StageEdit snapshot sequence id must be a string")
        sequence = StageEditSequence(raw_sequence_id, tuple(tokens))
        provenance = StageEditFeatureProvenance.from_dict(raw_provenance)
        raw_fingerprint = payload.get("token_schema_fingerprint")
        raw_snapshot_sha256 = payload.get("snapshot_sha256")
        if not isinstance(raw_fingerprint, str) or not isinstance(
            raw_snapshot_sha256, str
        ):
            raise ValueError("StageEdit snapshot hashes must be strings")
        snapshot = cls(
            sequence=sequence,
            token_schema_fingerprint=raw_fingerprint,
            provenance=provenance,
            snapshot_sha256=raw_snapshot_sha256,
        )
        snapshot.require_schema(schema)
        unsigned = dict(payload)
        unsigned.pop("snapshot_sha256")
        if _payload_sha256(unsigned) != snapshot.snapshot_sha256:
            raise ValueError("StageEdit feature snapshot SHA-256 mismatch")
        return snapshot


def materialize_stage_edit_feature_snapshot(
    sequence_id: str,
    frames: Sequence[StageEditFeatureFrame],
    *,
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
    rhythm: "RhythmAnalysis | None" = None,
    section_plan: SectionPlan | None = None,
    style_features: Mapping[str, Any] | None = None,
    performance_start_seconds: float = 0.0,
    performance_end_seconds: float | None = None,
    provenance: StageEditFeatureProvenance = StageEditFeatureProvenance(),
) -> StageEditFeatureSnapshot:
    """Materialize a deterministic, target-free feature-only sequence."""

    normalized_id = str(sequence_id).strip()
    if not normalized_id:
        raise ValueError("StageEdit feature sequence id cannot be blank")
    ordered_frames = tuple(frames)
    if not ordered_frames:
        raise ValueError("StageEdit feature materialization needs frames")
    if any(not isinstance(frame, StageEditFeatureFrame) for frame in ordered_frames):
        raise TypeError("feature frames must use StageEditFeatureFrame")
    if any(
        right.master_time_seconds <= left.master_time_seconds
        for left, right in zip(ordered_frames, ordered_frames[1:])
    ):
        raise ValueError("StageEdit feature frames must have unique ordered times")
    if performance_end_seconds is None:
        performance_end_seconds = max(
            ordered_frames[-1].master_time_seconds,
            float(performance_start_seconds),
        ) + 1.0
    sequence = StageEditSequence(
        normalized_id,
        tuple(
            materialize_stage_edit_token(
                master_time_seconds=frame.master_time_seconds,
                source_observations=frame.source_observations,
                schema=schema,
                rhythm=rhythm,
                section_plan=section_plan,
                causal_history=frame.causal_history,
                style_features=style_features,
                performance_start_seconds=performance_start_seconds,
                performance_end_seconds=performance_end_seconds,
            )
            for frame in ordered_frames
        ),
    )
    return StageEditFeatureSnapshot.build(
        sequence,
        schema=schema,
        provenance=provenance,
    )


def subset_stage_edit_token_sources(
    token: StageEditToken,
    source_ids: Sequence[str],
) -> StageEditToken:
    """Return a canonical source subset without changing context features."""

    requested = tuple(sorted({str(value).strip() for value in source_ids}))
    if not requested or any(not value for value in requested):
        raise ValueError("StageEdit source subset cannot be empty")
    by_id = {item.source_id: item for item in token.source_candidates}
    missing = tuple(source_id for source_id in requested if source_id not in by_id)
    if missing:
        raise ValueError(
            "StageEdit feature snapshot is missing source observations: "
            + ", ".join(missing)
        )
    return replace(
        token,
        source_candidates=tuple(by_id[source_id] for source_id in requested),
    )


__all__ = [
    "CausalEditDecision",
    "FEATURE_MATERIALIZER_CONTRACT",
    "FEATURE_MATERIALIZER_SCHEMA_VERSION",
    "FEATURE_SNAPSHOT_KIND",
    "SourceFeatureObservation",
    "StageEditFeatureFrame",
    "StageEditFeatureProvenance",
    "StageEditFeatureSnapshot",
    "materialize_stage_edit_feature_snapshot",
    "materialize_stage_edit_source_candidate",
    "materialize_stage_edit_token",
    "source_observation_from_timeline_candidate",
    "source_observation_from_visual_cache_payload",
    "subset_stage_edit_token_sources",
]
