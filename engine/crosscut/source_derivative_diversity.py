"""Detect same-camera transcodes/crops from synchronized real source frames.

Automatic stage acquisition can return two URLs that are different files but
the same camera feed (a re-encode, clean crop, or lightly reframed derivative).
Treating those files as independent cameras inflates source diversity and lets
an editor evade per-source occupancy limits.  This module resolves that
ambiguity from *content*: checked source clocks map a common canonical schedule
to source-local time, ffprobe supplies the actual decoded-frame PTS, and one
forward decode supplies the requested frames.

Only scalar measurements and decode provenance are serializable.  Frames, ORB
descriptors, and any other image representation are transient and are never
written to the audit result.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from . import reference_match as reference_match_module
from .reference_match import ReferenceMatchConfig, decode_target_frames_sequential
from .source_scalar_cache import PersistentSourceScalarCache, source_content_sha256
from .timewarp import TimeWarpMap
from .visual_feature_cache import python_symbol_dependency_sha256


SOURCE_DERIVATIVE_SCHEMA_VERSION = 1
SOURCE_DERIVATIVE_CACHE_NAMESPACE = "source-derivative-diversity-v1"
SOURCE_DERIVATIVE_CACHE_PAYLOAD_SCHEMA_VERSION = 1
SOURCE_DERIVATIVE_ANALYSIS_BASIS = (
    "canonical_timewarp_actual_pts_multiview_structure_geometry_v1"
)
DERIVATIVE_CONFIRMED = "confirmed_same_camera_derivative"
DERIVATIVE_DISTINCT = "confirmed_distinct_camera"
DERIVATIVE_UNCERTAIN = "uncertain"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _mad(values: Sequence[float]) -> float:
    center = _median(values)
    return _median(tuple(abs(value - center) for value in values))


def _normalized_ncc(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    selected = mask > 0
    if int(np.count_nonzero(selected)) < 256:
        return 0.0
    left_values = left[selected].astype(np.float32)
    right_values = right[selected].astype(np.float32)
    left_values -= float(left_values.mean())
    right_values -= float(right_values.mean())
    denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    if denominator <= 1e-9:
        return 0.0
    return min(1.0, max(-1.0, float(np.dot(left_values, right_values) / denominator)))


@dataclass(frozen=True, slots=True)
class SourceDerivativeConfig:
    sample_fractions: tuple[float, ...] = (
        0.07,
        0.175,
        0.28,
        0.385,
        0.50,
        0.615,
        0.72,
        0.825,
        0.93,
    )
    working_width: int = 480
    orb_nfeatures: int = 1400
    orb_fast_threshold: int = 10
    ratio_test_threshold: float = 0.76
    ransac_reprojection_pixels: float = 3.0
    minimum_decoded_pair_samples: int = 7
    minimum_confirmed_sample_count: int = 6
    minimum_distinct_sample_count: int = 6
    maximum_distinct_matching_samples: int = 1
    minimum_keypoints_per_side: int = 45
    minimum_inliers: int = 30
    minimum_inlier_ratio: float = 0.52
    maximum_normalized_reprojection_error: float = 0.012
    minimum_overlap_fraction: float = 0.42
    minimum_warped_luma_ncc: float = 0.84
    minimum_warped_gradient_ncc: float = 0.70
    maximum_anisotropy_ratio: float = 1.10
    maximum_absolute_rotation_degrees: float = 3.0
    maximum_perspective_strength: float = 0.012
    maximum_log_scale_mad: float = 0.035
    maximum_translation_mad: float = 0.030
    maximum_rotation_mad_degrees: float = 0.75
    maximum_perspective_mad: float = 0.003
    ffprobe_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if len(self.sample_fractions) < 7:
            raise ValueError("derivative detection needs at least seven sample times")
        if any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in self.sample_fractions
        ) or any(
            right <= left
            for left, right in zip(self.sample_fractions, self.sample_fractions[1:])
        ):
            raise ValueError("sample fractions must be strictly increasing in (0, 1)")
        for name in (
            "working_width",
            "orb_nfeatures",
            "orb_fast_threshold",
            "minimum_decoded_pair_samples",
            "minimum_confirmed_sample_count",
            "minimum_distinct_sample_count",
            "maximum_distinct_matching_samples",
            "minimum_keypoints_per_side",
            "minimum_inliers",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.working_width < 160 or self.orb_nfeatures < 100:
            raise ValueError("derivative working image policy is too small")
        if not 3 <= self.minimum_decoded_pair_samples <= len(self.sample_fractions):
            raise ValueError("minimum decoded samples are outside the schedule")
        if not 3 <= self.minimum_confirmed_sample_count <= len(self.sample_fractions):
            raise ValueError("minimum confirmed samples are outside the schedule")
        if not 3 <= self.minimum_distinct_sample_count <= len(self.sample_fractions):
            raise ValueError("minimum distinct samples are outside the schedule")
        for name in (
            "ratio_test_threshold",
            "minimum_inlier_ratio",
            "maximum_normalized_reprojection_error",
            "minimum_overlap_fraction",
            "minimum_warped_luma_ncc",
            "minimum_warped_gradient_ncc",
            "maximum_perspective_strength",
            "maximum_log_scale_mad",
            "maximum_translation_mad",
            "maximum_perspective_mad",
        ):
            number = _finite(getattr(self, name), name)
            if number <= 0.0 or number > 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        for name in (
            "ransac_reprojection_pixels",
            "maximum_anisotropy_ratio",
            "maximum_absolute_rotation_degrees",
            "maximum_rotation_mad_degrees",
            "ffprobe_timeout_seconds",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DecodedSourceFrame:
    """Transient decoded image plus scalar PTS provenance."""

    frame: np.ndarray
    frame_index: int
    requested_source_time_seconds: float
    actual_pts_seconds: float
    normalized_pts_seconds: float
    pts_residual_seconds: float


@dataclass(frozen=True, slots=True)
class PairSampleEvidence:
    sample_index: int
    canonical_time_seconds: float
    left_frame_index: int | None
    right_frame_index: int | None
    left_normalized_pts_seconds: float | None
    right_normalized_pts_seconds: float | None
    left_pts_residual_seconds: float | None
    right_pts_residual_seconds: float | None
    left_keypoint_count: int
    right_keypoint_count: int
    ratio_match_count: int
    homography_inlier_count: int
    homography_inlier_ratio: float
    normalized_reprojection_error: float | None
    overlap_fraction: float | None
    warped_luma_ncc: float | None
    warped_gradient_ncc: float | None
    scale_x: float | None
    scale_y: float | None
    translation_x_normalized: float | None
    translation_y_normalized: float | None
    rotation_degrees: float | None
    perspective_strength: float | None
    disposition: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceDerivativePairEvidence:
    left_source_index: int
    right_source_index: int
    status: str
    reason: str
    decoded_pair_sample_count: int
    matching_sample_count: int
    contradictory_sample_count: int
    ambiguous_sample_count: int
    median_inlier_ratio: float | None
    median_warped_gradient_ncc: float | None
    log_scale_mad: float | None
    translation_mad: float | None
    rotation_mad_degrees: float | None
    perspective_mad: float | None
    samples: tuple[PairSampleEvidence, ...]

    def __post_init__(self) -> None:
        if self.status not in {
            DERIVATIVE_CONFIRMED,
            DERIVATIVE_DISTINCT,
            DERIVATIVE_UNCERTAIN,
        }:
            raise ValueError("invalid source derivative pair status")

    def to_dict(self) -> dict[str, Any]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "samples"
            },
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True, slots=True)
class SourceDerivativeFamilyMetadata:
    """Compact decoder metadata; deliberately outside Transformer tokens."""

    source_count: int
    family_ids: tuple[int, ...]
    audit_sha256: str
    production_ready: bool
    uncertain_pair_count: int
    clock_contract_complete: bool
    family_consistency_conflict_count: int = 0
    analysis_basis: str = SOURCE_DERIVATIVE_ANALYSIS_BASIS

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_count, bool)
            or not isinstance(self.source_count, int)
            or self.source_count < 2
        ):
            raise ValueError("source derivative metadata needs at least two sources")
        if len(self.family_ids) != self.source_count:
            raise ValueError("derivative family ids must match source count")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.family_ids
        ):
            raise ValueError("derivative family ids must be non-negative integers")
        ordered: list[int] = []
        for value in self.family_ids:
            if value not in ordered:
                ordered.append(value)
        remap = {value: index for index, value in enumerate(ordered)}
        if tuple(remap[value] for value in self.family_ids) != self.family_ids:
            raise ValueError("derivative family ids must be first-seen canonical")
        if len(self.audit_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.audit_sha256
        ):
            raise ValueError("source derivative audit SHA-256 is invalid")
        if self.uncertain_pair_count < 0:
            raise ValueError("uncertain pair count cannot be negative")
        if self.family_consistency_conflict_count < 0:
            raise ValueError("family consistency conflict count cannot be negative")
        if not isinstance(self.production_ready, bool) or not isinstance(
            self.clock_contract_complete, bool
        ):
            raise ValueError("source derivative readiness flags must be booleans")
        if self.production_ready and (
            self.uncertain_pair_count
            or self.family_consistency_conflict_count
            or not self.clock_contract_complete
        ):
            raise ValueError("production-ready derivative metadata must be complete")

    @property
    def family_count(self) -> int:
        return len(set(self.family_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": self.analysis_basis,
            "source_count": self.source_count,
            "family_ids": list(self.family_ids),
            "family_count": self.family_count,
            "audit_sha256": self.audit_sha256,
            "production_ready": self.production_ready,
            "uncertain_pair_count": self.uncertain_pair_count,
            "family_consistency_conflict_count": (
                self.family_consistency_conflict_count
            ),
            "clock_contract_complete": self.clock_contract_complete,
            "transformer_token_schema_modified": False,
        }


@dataclass(frozen=True, slots=True)
class SourceDerivativeDiversityResult:
    canonical_sample_times_seconds: tuple[float, ...]
    source_decode_audit: tuple[Mapping[str, Any], ...]
    pairs: tuple[SourceDerivativePairEvidence, ...]
    family_ids: tuple[int, ...]
    family_consistency_conflict_count: int
    clock_contract_complete: bool
    production_ready: bool
    audit_sha256: str
    config: SourceDerivativeConfig

    @property
    def family_count(self) -> int:
        return len(set(self.family_ids))

    @property
    def uncertain_pair_count(self) -> int:
        return sum(pair.status == DERIVATIVE_UNCERTAIN for pair in self.pairs)

    @property
    def confirmed_derivative_pair_count(self) -> int:
        return sum(pair.status == DERIVATIVE_CONFIRMED for pair in self.pairs)

    def decoder_metadata(self) -> SourceDerivativeFamilyMetadata:
        return SourceDerivativeFamilyMetadata(
            source_count=len(self.family_ids),
            family_ids=self.family_ids,
            audit_sha256=self.audit_sha256,
            production_ready=self.production_ready,
            uncertain_pair_count=self.uncertain_pair_count,
            clock_contract_complete=self.clock_contract_complete,
            family_consistency_conflict_count=(
                self.family_consistency_conflict_count
            ),
        )

    def _payload(self, *, include_sha256: bool) -> dict[str, Any]:
        payload = {
            "schema_version": SOURCE_DERIVATIVE_SCHEMA_VERSION,
            "analysis_basis": SOURCE_DERIVATIVE_ANALYSIS_BASIS,
            "source_count": len(self.family_ids),
            "canonical_sample_times_seconds": list(
                self.canonical_sample_times_seconds
            ),
            "source_decode_audit": [dict(item) for item in self.source_decode_audit],
            "pairs": [pair.to_dict() for pair in self.pairs],
            "family_ids": list(self.family_ids),
            "family_count": self.family_count,
            "confirmed_derivative_pair_count": (
                self.confirmed_derivative_pair_count
            ),
            "uncertain_pair_count": self.uncertain_pair_count,
            "family_consistency_conflict_count": (
                self.family_consistency_conflict_count
            ),
            "clock_contract_complete": self.clock_contract_complete,
            "production_ready": self.production_ready,
            "config": self.config.to_dict(),
            "persistence_policy": {
                "raw_frames_persisted": False,
                "descriptors_persisted": False,
                "embeddings_persisted": False,
                "serialized_evidence": "scalar_metrics_and_pts_provenance_only",
            },
            "decoder_policy": {
                "diversity_unit": "derivative_family",
                "same_family_switch_counts_as_source_diversity": False,
                "same_family_occupancy_aggregated": True,
                "transformer_token_schema_modified": False,
            },
        }
        if include_sha256:
            payload["audit_sha256"] = self.audit_sha256
        return payload

    def to_dict(self) -> dict[str, Any]:
        return self._payload(include_sha256=True)


def _required_integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _required_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _optional_finite(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _finite(value, name)


def _exact_mapping(
    value: Any,
    name: str,
    expected_keys: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    result = dict(value)
    expected = set(expected_keys)
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise ValueError(
            f"{name} keys are invalid; missing={missing}; extra={extra}"
        )
    return result


def source_derivative_diversity_from_dict(
    payload: Mapping[str, Any],
) -> SourceDerivativeDiversityResult:
    """Reconstruct one canonical scalar result and verify its audit digest.

    This is deliberately stricter than a convenience JSON loader.  Persistent
    cache records are admitted only when every dataclass field is present, all
    integer/boolean domains are exact, the canonical audit SHA-256 recomputes,
    and serializing the reconstructed result reproduces the input byte model.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("source derivative payload must be an object")
    try:
        raw = json.loads(
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("source derivative payload is not JSON-safe") from exc
    if not isinstance(raw, dict):
        raise ValueError("source derivative payload must normalize to an object")
    if raw.get("schema_version") != SOURCE_DERIVATIVE_SCHEMA_VERSION:
        raise ValueError("source derivative payload schema is unsupported")
    if raw.get("analysis_basis") != SOURCE_DERIVATIVE_ANALYSIS_BASIS:
        raise ValueError("source derivative analysis basis is unsupported")

    config_raw = _exact_mapping(
        raw.get("config"),
        "config",
        SourceDerivativeConfig.__dataclass_fields__,
    )
    fractions_raw = config_raw.get("sample_fractions")
    if not isinstance(fractions_raw, list):
        raise ValueError("config.sample_fractions must be a list")
    config_raw["sample_fractions"] = tuple(
        _finite(value, f"config.sample_fractions[{index}]")
        for index, value in enumerate(fractions_raw)
    )
    config = SourceDerivativeConfig(**config_raw)

    times_raw = raw.get("canonical_sample_times_seconds")
    if not isinstance(times_raw, list):
        raise ValueError("canonical sample times must be a list")
    times = tuple(
        _finite(value, f"canonical_sample_times_seconds[{index}]")
        for index, value in enumerate(times_raw)
    )

    audit_raw = raw.get("source_decode_audit")
    if not isinstance(audit_raw, list) or not all(
        isinstance(item, Mapping) for item in audit_raw
    ):
        raise ValueError("source decode audit must contain objects")
    source_decode_audit = tuple(dict(item) for item in audit_raw)

    pair_rows = raw.get("pairs")
    if not isinstance(pair_rows, list):
        raise ValueError("source derivative pairs must be a list")
    pairs: list[SourceDerivativePairEvidence] = []
    for pair_index, pair_value in enumerate(pair_rows):
        pair_raw = _exact_mapping(
            pair_value,
            f"pairs[{pair_index}]",
            SourceDerivativePairEvidence.__dataclass_fields__,
        )
        samples_raw = pair_raw.pop("samples")
        if not isinstance(samples_raw, list):
            raise ValueError(f"pairs[{pair_index}].samples must be a list")
        samples: list[PairSampleEvidence] = []
        for sample_index, sample_value in enumerate(samples_raw):
            name = f"pairs[{pair_index}].samples[{sample_index}]"
            sample_raw = _exact_mapping(
                sample_value,
                name,
                PairSampleEvidence.__dataclass_fields__,
            )
            for key in (
                "sample_index",
                "left_keypoint_count",
                "right_keypoint_count",
                "ratio_match_count",
                "homography_inlier_count",
            ):
                sample_raw[key] = _required_integer(
                    sample_raw[key], f"{name}.{key}"
                )
            for key in ("left_frame_index", "right_frame_index"):
                if sample_raw[key] is not None:
                    sample_raw[key] = _required_integer(
                        sample_raw[key], f"{name}.{key}"
                    )
            sample_raw["canonical_time_seconds"] = _finite(
                sample_raw["canonical_time_seconds"],
                f"{name}.canonical_time_seconds",
            )
            for key in (
                "left_normalized_pts_seconds",
                "right_normalized_pts_seconds",
                "left_pts_residual_seconds",
                "right_pts_residual_seconds",
                "normalized_reprojection_error",
                "overlap_fraction",
                "warped_luma_ncc",
                "warped_gradient_ncc",
                "scale_x",
                "scale_y",
                "translation_x_normalized",
                "translation_y_normalized",
                "rotation_degrees",
                "perspective_strength",
            ):
                sample_raw[key] = _optional_finite(
                    sample_raw[key], f"{name}.{key}"
                )
            sample_raw["homography_inlier_ratio"] = _finite(
                sample_raw["homography_inlier_ratio"],
                f"{name}.homography_inlier_ratio",
            )
            if not isinstance(sample_raw["disposition"], str) or not sample_raw[
                "disposition"
            ]:
                raise ValueError(f"{name}.disposition must be non-empty")
            samples.append(PairSampleEvidence(**sample_raw))

        for key in (
            "left_source_index",
            "right_source_index",
            "decoded_pair_sample_count",
            "matching_sample_count",
            "contradictory_sample_count",
            "ambiguous_sample_count",
        ):
            pair_raw[key] = _required_integer(
                pair_raw[key], f"pairs[{pair_index}].{key}"
            )
        for key in (
            "median_inlier_ratio",
            "median_warped_gradient_ncc",
            "log_scale_mad",
            "translation_mad",
            "rotation_mad_degrees",
            "perspective_mad",
        ):
            pair_raw[key] = _optional_finite(
                pair_raw[key], f"pairs[{pair_index}].{key}"
            )
        for key in ("status", "reason"):
            if not isinstance(pair_raw[key], str) or not pair_raw[key]:
                raise ValueError(f"pairs[{pair_index}].{key} must be non-empty")
        pairs.append(
            SourceDerivativePairEvidence(samples=tuple(samples), **pair_raw)
        )

    family_ids_raw = raw.get("family_ids")
    if not isinstance(family_ids_raw, list):
        raise ValueError("source derivative family ids must be a list")
    family_ids = tuple(
        _required_integer(value, f"family_ids[{index}]")
        for index, value in enumerate(family_ids_raw)
    )
    audit_sha256 = raw.get("audit_sha256")
    if not isinstance(audit_sha256, str) or len(audit_sha256) != 64:
        raise ValueError("source derivative audit SHA-256 is invalid")
    result = SourceDerivativeDiversityResult(
        canonical_sample_times_seconds=times,
        source_decode_audit=source_decode_audit,
        pairs=tuple(pairs),
        family_ids=family_ids,
        family_consistency_conflict_count=_required_integer(
            raw.get("family_consistency_conflict_count"),
            "family_consistency_conflict_count",
        ),
        clock_contract_complete=_required_boolean(
            raw.get("clock_contract_complete"), "clock_contract_complete"
        ),
        production_ready=_required_boolean(
            raw.get("production_ready"), "production_ready"
        ),
        audit_sha256=audit_sha256,
        config=config,
    )
    result.decoder_metadata()
    encoded = json.dumps(
        result._payload(include_sha256=False),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    expected_sha256 = hashlib.sha256(encoded).hexdigest()
    if not hmac.compare_digest(audit_sha256, expected_sha256):
        raise ValueError("source derivative audit SHA-256 mismatch")
    normalized_result = json.loads(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    if normalized_result != raw:
        raise ValueError("source derivative payload is not canonical")
    return result


def _working_gray(frame: np.ndarray, width: int) -> np.ndarray:
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.size == 0
    ):
        raise ValueError("derivative comparison needs a non-empty uint8 BGR frame")
    height = max(2, int(round(frame.shape[0] * width / frame.shape[1])))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def _geometry_scalars(
    matrix: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float, float, float, float, float, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or abs(matrix[2, 2]) <= 1e-12:
        raise ValueError("invalid homography")
    matrix = matrix / matrix[2, 2]
    center = np.asarray([width * 0.5, height * 0.5, 1.0], dtype=np.float64)

    def project(point: np.ndarray) -> np.ndarray:
        value = matrix @ point
        if abs(value[2]) <= 1e-12:
            raise ValueError("homography projects through infinity")
        return value[:2] / value[2]

    origin = project(center)
    dx = project(center + np.asarray([1.0, 0.0, 0.0])) - origin
    dy = project(center + np.asarray([0.0, 1.0, 0.0])) - origin
    jacobian = np.column_stack((dx, dy))
    singular = np.linalg.svd(jacobian, compute_uv=False)
    if singular[-1] <= 1e-9:
        raise ValueError("degenerate homography")
    scale_x = float(np.linalg.norm(dx))
    scale_y = float(np.linalg.norm(dy))
    anisotropy = float(singular[0] / singular[-1])
    rotation = math.degrees(math.atan2(float(dx[1]), float(dx[0])))
    translation_x = float(origin[0] / width - 0.5)
    translation_y = float(origin[1] / height - 0.5)
    perspective = math.hypot(matrix[2, 0] * width, matrix[2, 1] * height)
    return (
        scale_x,
        scale_y,
        anisotropy,
        rotation,
        translation_x,
        translation_y,
        float(perspective),
    )


def _compare_sample(
    left: DecodedSourceFrame | None,
    right: DecodedSourceFrame | None,
    *,
    sample_index: int,
    canonical_time_seconds: float,
    detector: Any,
    matcher: Any,
    config: SourceDerivativeConfig,
) -> PairSampleEvidence:
    provenance = {
        "sample_index": sample_index,
        "canonical_time_seconds": canonical_time_seconds,
        "left_frame_index": None if left is None else left.frame_index,
        "right_frame_index": None if right is None else right.frame_index,
        "left_normalized_pts_seconds": (
            None if left is None else left.normalized_pts_seconds
        ),
        "right_normalized_pts_seconds": (
            None if right is None else right.normalized_pts_seconds
        ),
        "left_pts_residual_seconds": (
            None if left is None else left.pts_residual_seconds
        ),
        "right_pts_residual_seconds": (
            None if right is None else right.pts_residual_seconds
        ),
    }
    empty = {
        "left_keypoint_count": 0,
        "right_keypoint_count": 0,
        "ratio_match_count": 0,
        "homography_inlier_count": 0,
        "homography_inlier_ratio": 0.0,
        "normalized_reprojection_error": None,
        "overlap_fraction": None,
        "warped_luma_ncc": None,
        "warped_gradient_ncc": None,
        "scale_x": None,
        "scale_y": None,
        "translation_x_normalized": None,
        "translation_y_normalized": None,
        "rotation_degrees": None,
        "perspective_strength": None,
    }
    if left is None or right is None:
        return PairSampleEvidence(**provenance, **empty, disposition="decode_missing")
    try:
        left_gray = _working_gray(left.frame, config.working_width)
        right_gray = _working_gray(right.frame, config.working_width)
        left_points, left_descriptors = detector.detectAndCompute(left_gray, None)
        right_points, right_descriptors = detector.detectAndCompute(right_gray, None)
    except (cv2.error, TypeError, ValueError):
        return PairSampleEvidence(**provenance, **empty, disposition="feature_error")
    left_points = tuple(left_points or ())
    right_points = tuple(right_points or ())
    populated = dict(empty)
    populated["left_keypoint_count"] = len(left_points)
    populated["right_keypoint_count"] = len(right_points)
    if (
        left_descriptors is None
        or right_descriptors is None
        or len(left_descriptors) < 4
        or len(right_descriptors) < 4
    ):
        return PairSampleEvidence(
            **provenance, **populated, disposition="insufficient_structure"
        )
    try:
        pairs = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
    except cv2.error:
        pairs = ()
    good = tuple(
        pair[0]
        for pair in pairs
        if len(pair) == 2
        and pair[0].distance < config.ratio_test_threshold * pair[1].distance
    )
    populated["ratio_match_count"] = len(good)
    if len(good) < 8:
        disposition = (
            "contradictory"
            if min(len(left_points), len(right_points))
            >= config.minimum_keypoints_per_side
            else "insufficient_structure"
        )
        return PairSampleEvidence(**provenance, **populated, disposition=disposition)
    left_xy = np.float32([left_points[item.queryIdx].pt for item in good]).reshape(
        -1, 1, 2
    )
    right_xy = np.float32([right_points[item.trainIdx].pt for item in good]).reshape(
        -1, 1, 2
    )
    try:
        matrix, inlier_mask = cv2.findHomography(
            left_xy,
            right_xy,
            cv2.RANSAC,
            config.ransac_reprojection_pixels,
        )
    except cv2.error:
        matrix, inlier_mask = None, None
    if matrix is None or inlier_mask is None:
        return PairSampleEvidence(
            **provenance, **populated, disposition="contradictory"
        )
    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / len(good)
    populated["homography_inlier_count"] = inlier_count
    populated["homography_inlier_ratio"] = inlier_ratio
    try:
        projected = cv2.perspectiveTransform(left_xy[inliers], matrix).reshape(-1, 2)
        expected = right_xy[inliers].reshape(-1, 2)
        errors = np.linalg.norm(projected - expected, axis=1)
        normalized_error = float(
            np.median(errors) / math.hypot(right_gray.shape[1], right_gray.shape[0])
        )
        (
            scale_x,
            scale_y,
            anisotropy,
            rotation,
            translation_x,
            translation_y,
            perspective,
        ) = _geometry_scalars(matrix, left_gray.shape[1], left_gray.shape[0])
        warped = cv2.warpPerspective(
            left_gray,
            matrix,
            (right_gray.shape[1], right_gray.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask = cv2.warpPerspective(
            np.full(left_gray.shape, 255, dtype=np.uint8),
            matrix,
            (right_gray.shape[1], right_gray.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        overlap = float(np.count_nonzero(mask) / mask.size)
        luma_ncc = _normalized_ncc(warped, right_gray, mask)
        warped_gradient = cv2.magnitude(
            cv2.Sobel(warped, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(warped, cv2.CV_32F, 0, 1, ksize=3),
        )
        right_gradient = cv2.magnitude(
            cv2.Sobel(right_gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(right_gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        gradient_ncc = _normalized_ncc(warped_gradient, right_gradient, mask)
    except (cv2.error, TypeError, ValueError, FloatingPointError):
        return PairSampleEvidence(
            **provenance, **populated, disposition="geometry_error"
        )
    populated.update(
        {
            "normalized_reprojection_error": normalized_error,
            "overlap_fraction": overlap,
            "warped_luma_ncc": luma_ncc,
            "warped_gradient_ncc": gradient_ncc,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "translation_x_normalized": translation_x,
            "translation_y_normalized": translation_y,
            "rotation_degrees": rotation,
            "perspective_strength": perspective,
        }
    )
    matched = (
        inlier_count >= config.minimum_inliers
        and inlier_ratio >= config.minimum_inlier_ratio
        and normalized_error <= config.maximum_normalized_reprojection_error
        and overlap >= config.minimum_overlap_fraction
        and luma_ncc >= config.minimum_warped_luma_ncc
        and gradient_ncc >= config.minimum_warped_gradient_ncc
        and anisotropy <= config.maximum_anisotropy_ratio
        and abs(rotation) <= config.maximum_absolute_rotation_degrees
        and perspective <= config.maximum_perspective_strength
    )
    if matched:
        disposition = "matching"
    elif (
        min(len(left_points), len(right_points)) >= config.minimum_keypoints_per_side
        and (
            inlier_ratio < config.minimum_inlier_ratio * 0.55
            or inlier_count < config.minimum_inliers * 0.45
            or gradient_ncc < config.minimum_warped_gradient_ncc * 0.75
        )
    ):
        disposition = "contradictory"
    else:
        disposition = "ambiguous"
    return PairSampleEvidence(**provenance, **populated, disposition=disposition)


def _classify_pair(
    left_index: int,
    right_index: int,
    samples: tuple[PairSampleEvidence, ...],
    config: SourceDerivativeConfig,
) -> SourceDerivativePairEvidence:
    decoded = tuple(
        item
        for item in samples
        if item.disposition not in {"decode_missing", "feature_error"}
    )
    matching = tuple(item for item in decoded if item.disposition == "matching")
    contradictory = tuple(
        item for item in decoded if item.disposition == "contradictory"
    )
    ambiguous_count = len(decoded) - len(matching) - len(contradictory)

    def optional_median(name: str) -> float | None:
        values = tuple(
            float(value)
            for item in decoded
            if (value := getattr(item, name)) is not None
        )
        return _median(values) if values else None

    stable_samples = matching
    log_scales = tuple(
        math.log(max(1e-9, math.sqrt(item.scale_x * item.scale_y)))
        for item in stable_samples
        if item.scale_x is not None and item.scale_y is not None
    )
    translations_x = tuple(
        float(item.translation_x_normalized)
        for item in stable_samples
        if item.translation_x_normalized is not None
    )
    translations_y = tuple(
        float(item.translation_y_normalized)
        for item in stable_samples
        if item.translation_y_normalized is not None
    )
    rotations = tuple(
        float(item.rotation_degrees)
        for item in stable_samples
        if item.rotation_degrees is not None
    )
    perspectives = tuple(
        float(item.perspective_strength)
        for item in stable_samples
        if item.perspective_strength is not None
    )
    log_scale_mad = _mad(log_scales) if log_scales else None
    translation_mad = (
        max(_mad(translations_x), _mad(translations_y))
        if translations_x and translations_y
        else None
    )
    rotation_mad = _mad(rotations) if rotations else None
    perspective_mad = _mad(perspectives) if perspectives else None
    stable = (
        log_scale_mad is not None
        and translation_mad is not None
        and rotation_mad is not None
        and perspective_mad is not None
        and log_scale_mad <= config.maximum_log_scale_mad
        and translation_mad <= config.maximum_translation_mad
        and rotation_mad <= config.maximum_rotation_mad_degrees
        and perspective_mad <= config.maximum_perspective_mad
    )
    if len(decoded) < config.minimum_decoded_pair_samples:
        status = DERIVATIVE_UNCERTAIN
        reason = "insufficient_actual_pts_frame_coverage"
    elif len(matching) >= config.minimum_confirmed_sample_count and stable:
        status = DERIVATIVE_CONFIRMED
        reason = "multiview_geometry_and_structure_stable"
    elif (
        len(matching) <= config.maximum_distinct_matching_samples
        and len(contradictory) >= config.minimum_distinct_sample_count
    ):
        status = DERIVATIVE_DISTINCT
        reason = "multiview_geometry_contradicts_same_camera"
    else:
        status = DERIVATIVE_UNCERTAIN
        reason = (
            "candidate_derivative_geometry_not_temporally_stable"
            if len(matching) >= config.minimum_confirmed_sample_count
            else "mixed_or_low_confidence_visual_evidence"
        )
    return SourceDerivativePairEvidence(
        left_source_index=left_index,
        right_source_index=right_index,
        status=status,
        reason=reason,
        decoded_pair_sample_count=len(decoded),
        matching_sample_count=len(matching),
        contradictory_sample_count=len(contradictory),
        ambiguous_sample_count=ambiguous_count,
        median_inlier_ratio=optional_median("homography_inlier_ratio"),
        median_warped_gradient_ncc=optional_median("warped_gradient_ncc"),
        log_scale_mad=log_scale_mad,
        translation_mad=translation_mad,
        rotation_mad_degrees=rotation_mad,
        perspective_mad=perspective_mad,
        samples=samples,
    )


def _canonical_family_ids(
    source_count: int,
    pairs: Sequence[SourceDerivativePairEvidence],
) -> tuple[tuple[int, ...], int]:
    parent = list(range(source_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for pair in pairs:
        if pair.status == DERIVATIVE_CONFIRMED:
            union(pair.left_source_index, pair.right_source_index)
    roots = tuple(find(index) for index in range(source_count))
    ordered: list[int] = []
    for root in roots:
        if root not in ordered:
            ordered.append(root)
    mapping = {root: index for index, root in enumerate(ordered)}
    family_ids = tuple(mapping[root] for root in roots)
    conflict_count = sum(
        pair.status == DERIVATIVE_DISTINCT
        and family_ids[pair.left_source_index]
        == family_ids[pair.right_source_index]
        for pair in pairs
    )
    return family_ids, conflict_count


def classify_source_derivative_frames(
    sampled_frames: Sequence[Sequence[DecodedSourceFrame | None]],
    *,
    canonical_sample_times_seconds: Sequence[float],
    source_decode_audit: Sequence[Mapping[str, Any]],
    clock_contract_complete: bool,
    config: SourceDerivativeConfig | None = None,
) -> SourceDerivativeDiversityResult:
    """Classify a transient synchronized frame matrix and return scalar audit."""

    settings = config or SourceDerivativeConfig()
    rows = tuple(tuple(row) for row in sampled_frames)
    times = tuple(
        _finite(value, f"canonical_sample_times_seconds[{index}]")
        for index, value in enumerate(canonical_sample_times_seconds)
    )
    if len(rows) < 2 or len(source_decode_audit) != len(rows):
        raise ValueError("derivative classification needs aligned source rows")
    if len(times) != len(settings.sample_fractions) or any(
        len(row) != len(times) for row in rows
    ):
        raise ValueError("derivative frame matrix does not match sample schedule")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("canonical derivative sample times must increase")
    detector = cv2.ORB_create(
        nfeatures=settings.orb_nfeatures,
        fastThreshold=settings.orb_fast_threshold,
    )
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs: list[SourceDerivativePairEvidence] = []
    for left_index in range(len(rows)):
        for right_index in range(left_index + 1, len(rows)):
            samples = tuple(
                _compare_sample(
                    rows[left_index][sample_index],
                    rows[right_index][sample_index],
                    sample_index=sample_index,
                    canonical_time_seconds=canonical_time,
                    detector=detector,
                    matcher=matcher,
                    config=settings,
                )
                for sample_index, canonical_time in enumerate(times)
            )
            pairs.append(
                _classify_pair(left_index, right_index, samples, settings)
            )
    families, family_conflicts = _canonical_family_ids(len(rows), pairs)
    uncertain_count = sum(pair.status == DERIVATIVE_UNCERTAIN for pair in pairs)
    production_ready = bool(
        clock_contract_complete
        and uncertain_count == 0
        and family_conflicts == 0
    )
    provisional = SourceDerivativeDiversityResult(
        canonical_sample_times_seconds=times,
        source_decode_audit=tuple(dict(item) for item in source_decode_audit),
        pairs=tuple(pairs),
        family_ids=families,
        family_consistency_conflict_count=family_conflicts,
        clock_contract_complete=clock_contract_complete,
        production_ready=production_ready,
        audit_sha256="0" * 64,
        config=settings,
    )
    encoded = json.dumps(
        provisional._payload(include_sha256=False),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return SourceDerivativeDiversityResult(
        canonical_sample_times_seconds=times,
        source_decode_audit=provisional.source_decode_audit,
        pairs=provisional.pairs,
        family_ids=families,
        family_consistency_conflict_count=family_conflicts,
        clock_contract_complete=clock_contract_complete,
        production_ready=production_ready,
        audit_sha256=hashlib.sha256(encoded).hexdigest(),
        config=settings,
    )


def _probe_actual_video_pts(path: Path, *, timeout_seconds: float) -> tuple[float, ...]:
    command = (
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(path),
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"actual PTS probe failed for {path.name}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        raise RuntimeError(
            f"actual PTS probe failed for {path.name}: "
            + (detail[-1] if detail else f"exit {completed.returncode}")
        )
    values: list[float] = []
    for line in completed.stdout.splitlines():
        token = line.strip().split(",", 1)[0]
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    if len(values) < 2 or any(right <= left for left, right in zip(values, values[1:])):
        raise RuntimeError(f"actual video PTS are incomplete or non-monotonic: {path.name}")
    return tuple(values)


def _nearest_pts_index(values: Sequence[float], target: float) -> int:
    position = bisect_left(values, target)
    candidates = tuple(
        index for index in (position - 1, position) if 0 <= index < len(values)
    )
    if not candidates:
        raise ValueError("PTS lookup has no candidates")
    return min(candidates, key=lambda index: (abs(values[index] - target), index))


def _ffprobe_version_identity() -> str:
    try:
        completed = subprocess.run(
            ("ffprobe", "-version"),
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable:{type(exc).__name__}"
    first_line = next(
        (line.strip() for line in completed.stdout.splitlines() if line.strip()),
        "",
    )
    return (
        first_line
        if completed.returncode == 0 and first_line
        else f"unavailable:exit-{completed.returncode}"
    )


def _regular_file_identity(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class PersistentSourceDerivativeDiversityCache:
    """Content-addressed scalar cache for one synchronized source set.

    The shared scalar cache is anchored to the first source, while its analyzer
    identity binds every source digest, source ordering, offset, checked clock,
    common range, configuration, implementation file, and relevant runtime.
    Only the canonical result JSON string is persisted; decoded frames and
    descriptors remain transient.
    """

    def __init__(
        self,
        *,
        source_cache_directory: Path,
        sources: Sequence[Path],
        source_offsets_seconds: Sequence[float],
        source_timewarps: Sequence[TimeWarpMap | None],
        common_master_start: float,
        common_master_end: float,
        config: SourceDerivativeConfig | None = None,
    ) -> None:
        settings = config or SourceDerivativeConfig()
        normalized_sources = tuple(
            Path(value).expanduser().resolve(strict=True) for value in sources
        )
        offsets = tuple(
            _finite(value, f"source_offsets_seconds[{index}]")
            for index, value in enumerate(source_offsets_seconds)
        )
        clocks = tuple(source_timewarps)
        if len(normalized_sources) < 2 or not (
            len(normalized_sources) == len(offsets) == len(clocks)
        ):
            raise ValueError("derivative cache inputs must align")
        start = _finite(common_master_start, "common_master_start")
        end = _finite(common_master_end, "common_master_end")
        if start < 0.0 or end <= start:
            raise ValueError("derivative cache common master range is invalid")
        for index, clock in enumerate(clocks):
            if clock is not None and not isinstance(clock, TimeWarpMap):
                raise TypeError(
                    "derivative cache clocks must contain TimeWarpMap or None"
                )
            if clock is not None and not math.isclose(
                clock.global_offset_seconds,
                offsets[index],
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("derivative cache clock and offset disagree")

        source_hashes = tuple(
            source_content_sha256(path) for path in normalized_sources
        )
        self._source_paths = normalized_sources
        self._source_stat_identities = tuple(
            _regular_file_identity(path) for path in normalized_sources
        )
        self._disabled_by_source_change = False
        self._semantic_invalid_count = 0
        module_path = Path(__file__).resolve(strict=True)
        reference_module_file = reference_match_module.__file__
        if reference_module_file is None:
            raise ValueError("reference-match implementation path is unavailable")
        reference_module_path = Path(reference_module_file).resolve(strict=True)
        analyzer = {
            "analysis_basis": SOURCE_DERIVATIVE_ANALYSIS_BASIS,
            "payload_schema_version": (
                SOURCE_DERIVATIVE_CACHE_PAYLOAD_SCHEMA_VERSION
            ),
            "ordered_source_content": [
                {"source_ordinal": index, "sha256": digest}
                for index, digest in enumerate(source_hashes)
            ],
            "source_offsets_seconds": list(offsets),
            "source_timewarps": [
                None
                if clock is None
                else json.loads(
                    json.dumps(
                        clock.to_dict(),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
                for clock in clocks
            ],
            "common_master_range": {
                "start_seconds": start,
                "end_seconds": end,
            },
            "config_json": json.dumps(
                settings.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "implementation_sha256": {
                "source_derivative_analysis_ast": (
                    python_symbol_dependency_sha256(
                        module_path,
                        root_symbols=("analyze_source_derivative_diversity",),
                    )
                ),
                reference_module_path.name: source_content_sha256(
                    reference_module_path
                ),
            },
            "runtime": {
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "opencv_version": cv2.__version__,
                "opencv_build_sha256": hashlib.sha256(
                    cv2.getBuildInformation().encode("utf-8")
                ).hexdigest(),
                "ffprobe_version": _ffprobe_version_identity(),
            },
            "persistence_policy": "canonical_scalar_result_json_only",
        }
        self._cache = PersistentSourceScalarCache(
            source_cache_directory=source_cache_directory,
            source_path=normalized_sources[0],
            namespace=SOURCE_DERIVATIVE_CACHE_NAMESPACE,
            analyzer=analyzer,
        )

    def _sources_unchanged(self) -> bool:
        if self._disabled_by_source_change:
            return False
        try:
            current = tuple(
                _regular_file_identity(path) for path in self._source_paths
            )
        except OSError:
            self._disabled_by_source_change = True
            return False
        if current != self._source_stat_identities:
            self._disabled_by_source_change = True
            return False
        return True

    def get(self) -> SourceDerivativeDiversityResult | None:
        if not self._sources_unchanged():
            return None
        payload = self._cache.get()
        if payload is None:
            return None
        try:
            if set(payload) != {"payload_schema_version", "result_json"}:
                raise ValueError("derivative cache payload keys are invalid")
            if (
                payload["payload_schema_version"]
                != SOURCE_DERIVATIVE_CACHE_PAYLOAD_SCHEMA_VERSION
            ):
                raise ValueError("derivative cache payload schema is unsupported")
            encoded = payload["result_json"]
            if not isinstance(encoded, str):
                raise ValueError("derivative cache result JSON must be a string")
            decoded = json.loads(encoded)
            if not isinstance(decoded, Mapping):
                raise ValueError("derivative cache result must be an object")
            return source_derivative_diversity_from_dict(decoded)
        except (json.JSONDecodeError, TypeError, ValueError):
            self._semantic_invalid_count += 1
            return None

    def put(self, result: SourceDerivativeDiversityResult) -> bool:
        if not isinstance(result, SourceDerivativeDiversityResult):
            raise TypeError("derivative cache needs a diversity result")
        if not self._sources_unchanged():
            return False
        canonical = source_derivative_diversity_from_dict(result.to_dict())
        return self._cache.put(
            {
                "payload_schema_version": (
                    SOURCE_DERIVATIVE_CACHE_PAYLOAD_SCHEMA_VERSION
                ),
                "result_json": json.dumps(
                    canonical.to_dict(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = self._cache.diagnostics()
        diagnostics.update(
            {
                "payload_schema_version": (
                    SOURCE_DERIVATIVE_CACHE_PAYLOAD_SCHEMA_VERSION
                ),
                "semantic_invalid_record_count": self._semantic_invalid_count,
                "all_sources_unchanged": not self._disabled_by_source_change,
                "decoded_frames_persisted": False,
                "descriptors_persisted": False,
            }
        )
        return diagnostics


def analyze_source_derivative_diversity(
    sources: Sequence[Path],
    *,
    source_offsets_seconds: Sequence[float],
    source_timewarps: Sequence[TimeWarpMap | None],
    common_master_start: float,
    common_master_end: float,
    config: SourceDerivativeConfig | None = None,
) -> SourceDerivativeDiversityResult:
    """Decode actual-PTS synchronized source frames and resolve camera families."""

    settings = config or SourceDerivativeConfig()
    normalized_sources = tuple(Path(value).expanduser().resolve() for value in sources)
    offsets = tuple(
        _finite(value, f"source_offsets_seconds[{index}]")
        for index, value in enumerate(source_offsets_seconds)
    )
    clocks = tuple(source_timewarps)
    if len(normalized_sources) < 2 or not (
        len(offsets) == len(clocks) == len(normalized_sources)
    ):
        raise ValueError("derivative source paths, offsets, and clocks must align")
    start = _finite(common_master_start, "common_master_start")
    end = _finite(common_master_end, "common_master_end")
    if start < 0.0 or end <= start:
        raise ValueError("derivative common master range is invalid")
    for index, clock in enumerate(clocks):
        if clock is not None and not isinstance(clock, TimeWarpMap):
            raise TypeError("derivative source clocks must contain TimeWarpMap or None")
        if clock is not None and not math.isclose(
            clock.global_offset_seconds,
            offsets[index],
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("derivative timewarp and scalar offset disagree")
    clock_contract_complete = all(
        clock is not None and not clock.scalar_fallback for clock in clocks
    )
    canonical_times = tuple(
        start + fraction * (end - start) for fraction in settings.sample_fractions
    )
    rows: list[tuple[DecodedSourceFrame | None, ...]] = []
    source_audit: list[dict[str, Any]] = []
    decode_config = ReferenceMatchConfig(working_width=settings.working_width)
    for source_index, (path, offset, clock) in enumerate(
        zip(normalized_sources, offsets, clocks, strict=True)
    ):
        try:
            raw_pts = _probe_actual_video_pts(
                path, timeout_seconds=settings.ffprobe_timeout_seconds
            )
            pts_origin = raw_pts[0]
            normalized_pts = tuple(value - pts_origin for value in raw_pts)
            positive_deltas = tuple(
                right - left
                for left, right in zip(normalized_pts, normalized_pts[1:])
                if right > left
            )
            median_step = _median(positive_deltas)
            maximum_residual = max(0.050, 1.5 * median_step)
            requested_source_times = tuple(
                (
                    clock.master_to_source(canonical_time)
                    if clock is not None and not clock.scalar_fallback
                    else canonical_time + offset
                )
                for canonical_time in canonical_times
            )
            indices = tuple(
                _nearest_pts_index(normalized_pts, source_time)
                if source_time >= 0.0
                else -1
                for source_time in requested_source_times
            )
            batch = decode_target_frames_sequential(
                path,
                (index for index in indices if index >= 0),
                config=decode_config,
            )
            row: list[DecodedSourceFrame | None] = []
            accepted_count = 0
            maximum_observed_residual = 0.0
            for source_time, frame_index in zip(
                requested_source_times, indices, strict=True
            ):
                if frame_index < 0 or frame_index not in batch.frames:
                    row.append(None)
                    continue
                residual = abs(normalized_pts[frame_index] - source_time)
                maximum_observed_residual = max(maximum_observed_residual, residual)
                if residual > maximum_residual:
                    row.append(None)
                    continue
                accepted_count += 1
                row.append(
                    DecodedSourceFrame(
                        frame=batch.frames[frame_index],
                        frame_index=frame_index,
                        requested_source_time_seconds=source_time,
                        actual_pts_seconds=raw_pts[frame_index],
                        normalized_pts_seconds=normalized_pts[frame_index],
                        pts_residual_seconds=residual,
                    )
                )
            rows.append(tuple(row))
            source_audit.append(
                {
                    "source_index": source_index,
                    "clock_basis": (
                        "checked_local_monotonic_timewarp"
                        if clock is not None and not clock.scalar_fallback
                        else "scalar_fallback_not_production_complete"
                    ),
                    "pts_basis": "ffprobe_best_effort_timestamp_time",
                    "pts_frame_count": len(raw_pts),
                    "pts_origin_seconds": pts_origin,
                    "median_pts_step_seconds": median_step,
                    "requested_sample_count": len(canonical_times),
                    "decoded_sample_count": accepted_count,
                    "maximum_pts_residual_seconds": maximum_observed_residual,
                    "decode_policy": "single_forward_decode_exact_selected_frame_indices",
                    "raw_frames_persisted": False,
                }
            )
        except (cv2.error, OSError, RuntimeError, TypeError, ValueError) as exc:
            rows.append(tuple(None for _ in canonical_times))
            source_audit.append(
                {
                    "source_index": source_index,
                    "clock_basis": (
                        "checked_local_monotonic_timewarp"
                        if clock is not None and not clock.scalar_fallback
                        else "scalar_fallback_not_production_complete"
                    ),
                    "pts_basis": "ffprobe_best_effort_timestamp_time",
                    "requested_sample_count": len(canonical_times),
                    "decoded_sample_count": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_frames_persisted": False,
                }
            )
    return classify_source_derivative_frames(
        rows,
        canonical_sample_times_seconds=canonical_times,
        source_decode_audit=source_audit,
        clock_contract_complete=clock_contract_complete,
        config=settings,
    )


__all__ = [
    "DERIVATIVE_CONFIRMED",
    "DERIVATIVE_DISTINCT",
    "DERIVATIVE_UNCERTAIN",
    "DecodedSourceFrame",
    "PairSampleEvidence",
    "SOURCE_DERIVATIVE_ANALYSIS_BASIS",
    "SOURCE_DERIVATIVE_CACHE_NAMESPACE",
    "PersistentSourceDerivativeDiversityCache",
    "SourceDerivativeConfig",
    "SourceDerivativeDiversityResult",
    "SourceDerivativeFamilyMetadata",
    "SourceDerivativePairEvidence",
    "analyze_source_derivative_diversity",
    "classify_source_derivative_frames",
    "source_derivative_diversity_from_dict",
]
