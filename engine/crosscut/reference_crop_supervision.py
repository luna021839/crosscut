"""Recover renderer-native crop supervision from attributed reference shots.

This module is intentionally downstream of reference source attribution and
independent from the production timeline.  It re-decodes several interior
frames from an attributed detector-owned shot, estimates the *source to
reference* camera transform at every sample, and emits a target only when the
same transform is stable over time and improves the pixels it actually warps.

Nothing is silently made safe.  A zoom-out, excessive pan/roll, unstable
estimate, weak feature footprint, perspective change, or non-improving warp
loss-masks the whole crop target.  When an otherwise bounded affine needs only
additional overscan to satisfy the production renderer's rotation/edge
contract, the target is explicitly projected to the minimum renderer-safe zoom
and that exact projected warp must pass the pixel review again.  Pixels,
descriptors, and keypoints are transient and are never serialized.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .reference_evidence_builder import ReferenceSourceClock
from .render import EditorialTransformKeyframe


SCHEMA_VERSION = 1
ANALYSIS_BASIS = (
    "attributed_detected_shot_multiframe_source_to_reference_crop_supervision_v1"
)
CONTENT_HASH_BASIS = (
    "sha256_canonical_json_with_provenance.artifact_content_sha256_null"
)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _artifact_content_sha256(payload: Mapping[str, Any]) -> str:
    identity = deepcopy(dict(payload))
    provenance = identity.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("crop supervision provenance must be an object")
    normalized = deepcopy(dict(provenance))
    normalized["artifact_content_sha256"] = None
    identity["provenance"] = normalized
    return _canonical_json_sha256(identity)


@dataclass(frozen=True, slots=True)
class ReferenceCropSupervisionConfig:
    """Conservative geometry, renderer, and temporal acceptance gates."""

    sample_fractions: tuple[float, ...] = (0.25, 0.50, 0.75)
    minimum_sample_count: int = 3
    working_width: int = 640
    orb_nfeatures: int = 2400
    orb_fast_threshold: int = 8
    ratio_test_threshold: float = 0.77
    minimum_ratio_matches: int = 28
    minimum_mutual_matches: int = 18
    minimum_affine_inliers: int = 14
    minimum_affine_inlier_ratio: float = 0.52
    ransac_reprojection_threshold_pixels: float = 3.0
    maximum_median_reprojection_error_pixels: float = 2.5
    maximum_p90_reprojection_error_pixels: float = 5.0
    minimum_inlier_x_span: float = 0.24
    minimum_inlier_y_span: float = 0.18
    minimum_inlier_coverage: float = 0.060
    minimum_homography_inlier_ratio: float = 0.48
    maximum_homography_median_reprojection_error_pixels: float = 2.8
    maximum_homography_p90_reprojection_error_pixels: float = 5.6
    maximum_perspective_distortion: float = 0.075
    minimum_zoom: float = 1.0
    maximum_zoom: float = 1.18
    maximum_center_shift: float = 0.040
    maximum_roll_degrees: float = 1.60
    project_minimum_renderer_safe_zoom: bool = True
    minimum_warp_valid_fraction: float = 0.84
    minimum_warp_structure_ncc: float = 0.34
    minimum_warp_structure_ncc_improvement: float = 0.020
    maximum_temporal_zoom_deviation: float = 0.018
    maximum_temporal_center_deviation: float = 0.012
    maximum_temporal_roll_deviation_degrees: float = 0.45
    ransac_seed: int = 90417

    def __post_init__(self) -> None:
        fractions = tuple(
            _finite(value, "sample fraction") for value in self.sample_fractions
        )
        if len(fractions) < 3 or len(fractions) != len(set(fractions)):
            raise ValueError("crop supervision needs at least three unique samples")
        if any(not 0.0 < value < 1.0 for value in fractions):
            raise ValueError("sample fractions must lie strictly inside the shot")
        if tuple(sorted(fractions)) != fractions:
            raise ValueError("sample fractions must be strictly increasing")
        object.__setattr__(self, "sample_fractions", fractions)
        for name in (
            "minimum_sample_count",
            "working_width",
            "orb_nfeatures",
            "minimum_ratio_matches",
            "minimum_mutual_matches",
            "minimum_affine_inliers",
        ):
            _positive_integer(getattr(self, name), name)
        if self.minimum_sample_count < 3:
            raise ValueError("minimum_sample_count must be at least three")
        if self.minimum_sample_count > len(fractions):
            raise ValueError("minimum_sample_count exceeds configured samples")
        if self.working_width < 160 or self.orb_nfeatures < 100:
            raise ValueError("crop supervision working geometry is too small")
        if (
            isinstance(self.orb_fast_threshold, bool)
            or not isinstance(self.orb_fast_threshold, int)
            or self.orb_fast_threshold < 0
        ):
            raise ValueError("orb_fast_threshold must be a non-negative integer")
        if (
            isinstance(self.ransac_seed, bool)
            or not isinstance(self.ransac_seed, int)
            or self.ransac_seed < 0
        ):
            raise ValueError("ransac_seed must be a non-negative integer")
        if not isinstance(self.project_minimum_renderer_safe_zoom, bool):
            raise TypeError(
                "project_minimum_renderer_safe_zoom must be a boolean"
            )
        if not 0.0 < _finite(self.ratio_test_threshold, "ratio threshold") < 1.0:
            raise ValueError("ratio_test_threshold must lie in (0, 1)")
        normalized = (
            "minimum_affine_inlier_ratio",
            "minimum_inlier_x_span",
            "minimum_inlier_y_span",
            "minimum_inlier_coverage",
            "minimum_homography_inlier_ratio",
            "maximum_perspective_distortion",
            "maximum_center_shift",
            "minimum_warp_valid_fraction",
            "minimum_warp_structure_ncc",
            "minimum_warp_structure_ncc_improvement",
            "maximum_temporal_zoom_deviation",
            "maximum_temporal_center_deviation",
        )
        for name in normalized:
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be normalized")
        positive = (
            "ransac_reprojection_threshold_pixels",
            "maximum_median_reprojection_error_pixels",
            "maximum_p90_reprojection_error_pixels",
            "maximum_homography_median_reprojection_error_pixels",
            "maximum_homography_p90_reprojection_error_pixels",
            "minimum_zoom",
            "maximum_zoom",
            "maximum_roll_degrees",
            "maximum_temporal_roll_deviation_degrees",
        )
        for name in positive:
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_zoom != 1.0 or not 1.0 < self.maximum_zoom <= 1.18:
            raise ValueError("renderer zoom supervision must remain in [1, 1.18]")
        if self.maximum_center_shift > 0.040:
            raise ValueError("renderer center supervision exceeds the four-percent cap")
        if self.maximum_roll_degrees > 1.60:
            raise ValueError("renderer roll supervision exceeds the subtle-roll cap")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: list(value) if name == "sample_fractions" else value
            for name, value in (
                (field_name, getattr(self, field_name))
                for field_name in self.__dataclass_fields__
            )
        }


@dataclass(frozen=True, slots=True)
class ReferenceCropSupervisionRequest:
    """One provenance-bearing attributed shot and its two media clocks."""

    attributed_shot: Mapping[str, Any]
    attribution_path: Path
    reference_media_path: Path
    source_media_path: Path
    reference_clock: ReferenceSourceClock
    source_clock: ReferenceSourceClock
    source_id: str
    expected_attribution_file_sha256: str | None = None
    expected_reference_media_sha256: str | None = None
    expected_source_media_sha256: str | None = None
    caller_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.attributed_shot, Mapping):
            raise TypeError("attributed_shot must be an object")
        source_id = str(self.source_id).strip()
        if not source_id:
            raise ValueError("source_id cannot be blank")
        if not isinstance(self.reference_clock, ReferenceSourceClock) or not isinstance(
            self.source_clock, ReferenceSourceClock
        ):
            raise TypeError("reference and source clocks must be ReferenceSourceClock values")
        paths = {
            "attribution_path": Path(self.attribution_path).expanduser(),
            "reference_media_path": Path(self.reference_media_path).expanduser(),
            "source_media_path": Path(self.source_media_path).expanduser(),
        }
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{name} is not a file: {path}")
            object.__setattr__(self, name, path.resolve())
        if paths["reference_media_path"].resolve() == paths[
            "source_media_path"
        ].resolve():
            raise ValueError("reference and source media must be distinct files")
        for name in (
            "expected_attribution_file_sha256",
            "expected_reference_media_sha256",
            "expected_source_media_sha256",
        ):
            value = getattr(self, name)
            if value is not None and not _is_sha256(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        try:
            serialized_caller = json.loads(
                json.dumps(
                    dict(self.caller_provenance),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("caller_provenance must be finite JSON") from exc
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "attributed_shot", deepcopy(dict(self.attributed_shot)))
        object.__setattr__(self, "caller_provenance", serialized_caller)


@dataclass(frozen=True, slots=True)
class _MediaInfo:
    path: Path
    frame_rate: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.frame_rate

    def to_dict(self, *, sha256: str) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "media_sha256": sha256,
            "byte_size": self.path.stat().st_size,
            "frame_rate": self.frame_rate,
            "declared_frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "declared_duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True, slots=True)
class _DecodedFrame:
    requested_seconds: float
    frame_index: int
    decoded_seconds: float
    frame: np.ndarray


@dataclass(frozen=True, slots=True)
class _EstimatedTransform:
    affine_source_to_reference: np.ndarray
    zoom: float
    center_x: float
    center_y: float
    roll_degrees: float
    diagnostics: Mapping[str, Any]

    def renderer_dict(self) -> dict[str, float]:
        return {
            "zoom": self.zoom,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "roll_degrees": self.roll_degrees,
        }


@dataclass(frozen=True, slots=True)
class ReferenceCropSupervisionArtifact:
    """Validated serializable supervision plus an overwrite-refusing writer."""

    payload: dict[str, Any]

    @property
    def eligible(self) -> bool:
        return bool(self.payload["supervision"]["eligible"])

    @property
    def target(self) -> Mapping[str, float] | None:
        value = self.payload["supervision"]["renderer_target"]
        return value if isinstance(value, Mapping) else None

    def write(self, output_path: Path) -> Path:
        destination = Path(output_path).expanduser()
        resolved = destination.resolve()
        protected = {
            Path(str(self.payload["provenance"][name]["path"])).resolve()
            for name in ("attribution", "reference_media", "source_media")
        }
        if resolved in protected:
            raise ValueError("crop supervision output cannot overwrite an input artifact")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("x", encoding="utf-8") as handle:
                json.dump(
                    self.payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                handle.write("\n")
        except FileExistsError as exc:
            raise FileExistsError(
                f"crop supervision output already exists: {destination}"
            ) from exc
        return resolved


def _media_info(path: Path) -> _MediaInfo:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open media: {path}")
        frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
        height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    finally:
        capture.release()
    if (
        not math.isfinite(frame_rate)
        or frame_rate <= 0.0
        or frame_count <= 0
        or width <= 1
        or height <= 1
    ):
        raise ValueError(f"media metadata is invalid: {path}")
    return _MediaInfo(path, frame_rate, frame_count, width, height)


def _decode_frame(info: _MediaInfo, seconds: float) -> _DecodedFrame:
    requested = _finite(seconds, "requested media seconds")
    if requested < 0.0 or requested >= info.duration_seconds:
        raise ValueError(
            f"requested media time {requested:.6f}s is outside {info.path}"
        )
    index = min(info.frame_count - 1, max(0, int(round(requested * info.frame_rate))))
    # A fresh decoder per sample prevents a prior sequential read or cached
    # representative frame from masquerading as shot-interior supervision.
    capture = cv2.VideoCapture(str(info.path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot reopen media: {info.path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"failed to decode frame {index} from {info.path}")
    return _DecodedFrame(requested, index, index / info.frame_rate, frame)


def _media_time_is_decodable(info: _MediaInfo, seconds: float) -> bool:
    """Return whether a clock request lies in the declared video-frame domain."""

    requested = _finite(seconds, "requested media seconds")
    return 0.0 <= requested < info.duration_seconds


def _canonical_output(frame: np.ndarray, width: int) -> np.ndarray:
    """Center-crop to 16:9 then scale, matching the group renderer's basis."""

    source_height, source_width = frame.shape[:2]
    target_aspect = 16.0 / 9.0
    aspect = source_width / source_height
    if aspect > target_aspect:
        crop_width = int(round(source_height * target_aspect))
        left = max(0, (source_width - crop_width) // 2)
        cropped = frame[:, left : left + crop_width]
    else:
        crop_height = int(round(source_width / target_aspect))
        top = max(0, (source_height - crop_height) // 2)
        cropped = frame[top : top + crop_height, :]
    height = max(2, int(round(width / target_aspect)))
    interpolation = cv2.INTER_AREA if cropped.shape[1] > width else cv2.INTER_LINEAR
    return cv2.resize(cropped, (width, height), interpolation=interpolation)


def _ratio_matches(
    left: np.ndarray,
    right: np.ndarray,
    threshold: float,
) -> tuple[cv2.DMatch, ...]:
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(
        left, right, k=2
    )
    accepted = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in (pair,)
        if first.distance < threshold * second.distance
    ]
    return tuple(
        sorted(accepted, key=lambda item: (item.distance, item.queryIdx, item.trainIdx))
    )


def _mutual_matches(
    source_descriptors: np.ndarray,
    reference_descriptors: np.ndarray,
    threshold: float,
) -> tuple[cv2.DMatch, ...]:
    forward = _ratio_matches(source_descriptors, reference_descriptors, threshold)
    reverse = _ratio_matches(reference_descriptors, source_descriptors, threshold)
    reverse_pairs = {(item.trainIdx, item.queryIdx) for item in reverse}
    return tuple(
        item for item in forward if (item.queryIdx, item.trainIdx) in reverse_pairs
    )


def _percentile(values: np.ndarray, fraction: float) -> float:
    return float(np.percentile(values.astype(np.float64), fraction * 100.0))


def _perspective_distortion(
    homography: np.ndarray,
    width: int,
    height: int,
) -> float:
    matrix = homography.astype(np.float64)
    if abs(float(matrix[2, 2])) <= 1e-12:
        return math.inf
    matrix /= float(matrix[2, 2])
    denominators = tuple(
        float(matrix[2, 0] * x + matrix[2, 1] * y + 1.0)
        for x, y in ((0, 0), (width, 0), (0, height), (width, height))
    )
    if any(not math.isfinite(value) or value <= 1e-6 for value in denominators):
        return math.inf
    return max(denominators) / min(denominators) - 1.0


def _renderer_values_from_affine(
    affine: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    linear = affine[:, :2].astype(np.float64)
    determinant = float(np.linalg.det(linear))
    if determinant <= 0.0:
        raise ValueError("reflected_or_singular_affine")
    zoom = math.sqrt(
        max(0.0, float(linear[0, 0] ** 2 + linear[1, 0] ** 2))
    )
    roll = math.degrees(math.atan2(float(linear[1, 0]), float(linear[0, 0])))
    input_center = np.asarray([width / 2.0, height / 2.0, 1.0])
    target_center = affine @ input_center
    target_x = float(target_center[0]) / width
    target_y = float(target_center[1]) / height
    center_x = 0.5 - (target_x - 0.5) / zoom
    center_y = 0.5 - (target_y - 0.5) / zoom
    return (zoom, center_x, center_y, roll)


def _affine_from_renderer_values(
    *,
    zoom: float,
    center_x: float,
    center_y: float,
    roll_degrees: float,
    width: int,
    height: int,
) -> np.ndarray:
    radians = math.radians(roll_degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    linear = zoom * np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    input_center = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
    target_center = np.asarray(
        [
            width * (zoom * (0.5 - center_x) + 0.5),
            height * (zoom * (0.5 - center_y) + 0.5),
        ],
        dtype=np.float64,
    )
    translation = target_center - linear @ input_center
    return np.column_stack((linear, translation)).astype(np.float64)


def _minimum_renderer_safe_target(
    *,
    zoom: float,
    center_x: float,
    center_y: float,
    roll_degrees: float,
    settings: ReferenceCropSupervisionConfig,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    """Project only zoom to the minimum production-renderer-safe value.

    Pan and roll are never changed.  The monotonic bisection is accepted only
    when the production validator accepts a value within the configured crop
    ceiling.  Callers must still review the pixels warped by the projected
    target; this helper establishes geometry safety, not supervision quality.
    """

    raw = {
        "zoom": zoom,
        "center_x": center_x,
        "center_y": center_y,
        "roll_degrees": roll_degrees,
    }
    try:
        EditorialTransformKeyframe(
            0.0, zoom, center_x, center_y, roll_degrees
        )
    except ValueError as raw_error:
        diagnostics: dict[str, Any] = {
            "applied": False,
            "basis": (
                "minimum_zoom_bisection_against_"
                "EditorialTransformKeyframe_v1"
            ),
            "raw_target": raw,
            "raw_renderer_rejection": str(raw_error),
            "pan_and_roll_modified": False,
        }
        if not settings.project_minimum_renderer_safe_zoom:
            diagnostics["rejection_reason"] = "projection_disabled"
            return (None, diagnostics)
        try:
            EditorialTransformKeyframe(
                0.0,
                settings.maximum_zoom,
                center_x,
                center_y,
                roll_degrees,
            )
        except ValueError as maximum_error:
            diagnostics["rejection_reason"] = "no_safe_zoom_within_bound"
            diagnostics["maximum_zoom_renderer_rejection"] = str(maximum_error)
            return (None, diagnostics)
        low = zoom
        high = settings.maximum_zoom
        for _ in range(56):
            midpoint = (low + high) / 2.0
            try:
                EditorialTransformKeyframe(
                    0.0, midpoint, center_x, center_y, roll_degrees
                )
            except ValueError:
                low = midpoint
            else:
                high = midpoint
        projected = {
            "zoom": high,
            "center_x": center_x,
            "center_y": center_y,
            "roll_degrees": roll_degrees,
        }
        EditorialTransformKeyframe(0.0, **projected)
        diagnostics.update(
            {
                "applied": True,
                "projected_target": projected,
                "additional_zoom": high - zoom,
                "pixel_warp_revalidation_required": True,
            }
        )
        return (projected, diagnostics)
    return (
        raw,
        {
            "applied": False,
            "basis": (
                "minimum_zoom_bisection_against_"
                "EditorialTransformKeyframe_v1"
            ),
            "raw_target": raw,
            "raw_renderer_rejection": None,
            "pan_and_roll_modified": False,
            "pixel_warp_revalidation_required": True,
        },
    )


def _gradient_structure(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (0, 0), 0.75)
    x_gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    y_gradient = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(x_gradient, y_gradient)


def _masked_ncc(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    selected_left = left[mask]
    selected_right = right[mask]
    if selected_left.size < 256:
        return -1.0
    selected_left = selected_left - float(selected_left.mean())
    selected_right = selected_right - float(selected_right.mean())
    denominator = math.sqrt(
        float(np.dot(selected_left, selected_left))
        * float(np.dot(selected_right, selected_right))
    )
    if denominator <= 1e-12:
        return -1.0
    return float(np.dot(selected_left, selected_right) / denominator)


def _warp_review(
    reference: np.ndarray,
    source: np.ndarray,
    affine: np.ndarray,
) -> dict[str, float]:
    height, width = reference.shape[:2]
    warped = cv2.warpAffine(
        source,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    support = cv2.warpAffine(
        np.full(source.shape[:2], 255, dtype=np.uint8),
        affine,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    support = cv2.erode(support, np.ones((5, 5), dtype=np.uint8)) > 0
    valid_fraction = float(support.mean())
    reference_structure = _gradient_structure(reference)
    source_structure = _gradient_structure(source)
    warped_structure = _gradient_structure(warped)
    baseline_ncc = _masked_ncc(reference_structure, source_structure, support)
    warped_ncc = _masked_ncc(reference_structure, warped_structure, support)
    return {
        "valid_fraction": valid_fraction,
        "baseline_structure_ncc": baseline_ncc,
        "warped_structure_ncc": warped_ncc,
        "structure_ncc_improvement": warped_ncc - baseline_ncc,
    }


def _estimate_sample(
    reference: np.ndarray,
    source: np.ndarray,
    settings: ReferenceCropSupervisionConfig,
) -> tuple[_EstimatedTransform | None, str | None, dict[str, Any]]:
    if reference.shape != source.shape:
        raise ValueError("canonical reference/source frames must be equal-sized")
    height, width = reference.shape[:2]
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(
        nfeatures=settings.orb_nfeatures,
        fastThreshold=settings.orb_fast_threshold,
        edgeThreshold=19,
    )
    reference_keypoints, reference_descriptors = orb.detectAndCompute(
        reference_gray, None
    )
    source_keypoints, source_descriptors = orb.detectAndCompute(source_gray, None)
    diagnostics: dict[str, Any] = {
        "frame_size": [width, height],
        "match_direction": "source_output_to_reference_output",
        "reference_keypoint_count": len(reference_keypoints),
        "source_keypoint_count": len(source_keypoints),
    }

    def reject(reason: str) -> tuple[None, str, dict[str, Any]]:
        diagnostics["decision"] = reason
        return (None, reason, diagnostics)

    if reference_descriptors is None or source_descriptors is None:
        return reject("descriptors_unavailable")
    ratio = _ratio_matches(
        source_descriptors, reference_descriptors, settings.ratio_test_threshold
    )
    diagnostics["ratio_match_count"] = len(ratio)
    if len(ratio) < settings.minimum_ratio_matches:
        return reject("insufficient_ratio_matches")
    matches = _mutual_matches(
        source_descriptors, reference_descriptors, settings.ratio_test_threshold
    )
    diagnostics["mutual_match_count"] = len(matches)
    if len(matches) < settings.minimum_mutual_matches:
        return reject("insufficient_mutual_matches")
    source_points = np.float32(
        [source_keypoints[item.queryIdx].pt for item in matches]
    )
    reference_points = np.float32(
        [reference_keypoints[item.trainIdx].pt for item in matches]
    )
    cv2.setRNGSeed(settings.ransac_seed)
    affine, affine_mask = cv2.estimateAffinePartial2D(
        source_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=settings.ransac_reprojection_threshold_pixels,
        maxIters=5000,
        confidence=0.997,
        refineIters=25,
    )
    if affine is None or affine_mask is None:
        return reject("affine_estimation_failed")
    affine_inliers = affine_mask.reshape(-1).astype(bool)
    affine_inlier_count = int(affine_inliers.sum())
    affine_inlier_ratio = affine_inlier_count / len(matches)
    diagnostics["affine_inlier_count"] = affine_inlier_count
    diagnostics["affine_inlier_ratio"] = affine_inlier_ratio
    if affine_inlier_count < settings.minimum_affine_inliers:
        return reject("insufficient_affine_inliers")
    if affine_inlier_ratio < settings.minimum_affine_inlier_ratio:
        return reject("affine_inlier_ratio_too_low")
    inlier_source = source_points[affine_inliers]
    inlier_reference = reference_points[affine_inliers]
    affine_projection = cv2.transform(
        inlier_source.reshape(-1, 1, 2), affine
    ).reshape(-1, 2)
    affine_errors = np.linalg.norm(affine_projection - inlier_reference, axis=1)
    median_error = float(np.median(affine_errors))
    p90_error = _percentile(affine_errors, 0.90)
    diagnostics["affine_median_reprojection_error_pixels"] = median_error
    diagnostics["affine_p90_reprojection_error_pixels"] = p90_error
    if median_error > settings.maximum_median_reprojection_error_pixels:
        return reject("affine_median_reprojection_error_too_high")
    if p90_error > settings.maximum_p90_reprojection_error_pixels:
        return reject("affine_p90_reprojection_error_too_high")
    x_span = float(np.ptp(inlier_source[:, 0])) / width
    y_span = float(np.ptp(inlier_source[:, 1])) / height
    coverage = x_span * y_span
    diagnostics.update(
        {
            "inlier_x_span": x_span,
            "inlier_y_span": y_span,
            "inlier_coverage": coverage,
        }
    )
    if x_span < settings.minimum_inlier_x_span:
        return reject("inlier_horizontal_coverage_too_small")
    if y_span < settings.minimum_inlier_y_span:
        return reject("inlier_vertical_coverage_too_small")
    if coverage < settings.minimum_inlier_coverage:
        return reject("inlier_area_coverage_too_small")

    cv2.setRNGSeed(settings.ransac_seed)
    homography, homography_mask = cv2.findHomography(
        source_points,
        reference_points,
        cv2.RANSAC,
        settings.ransac_reprojection_threshold_pixels,
        maxIters=5000,
        confidence=0.997,
    )
    if homography is None or homography_mask is None:
        return reject("perspective_validation_unavailable")
    homography_inliers = homography_mask.reshape(-1).astype(bool)
    homography_inlier_count = int(homography_inliers.sum())
    homography_inlier_ratio = homography_inlier_count / len(matches)
    diagnostics["homography_inlier_count"] = homography_inlier_count
    diagnostics["homography_inlier_ratio"] = homography_inlier_ratio
    if homography_inlier_ratio < settings.minimum_homography_inlier_ratio:
        return reject("homography_inlier_ratio_too_low")
    homography_source = source_points[homography_inliers]
    homography_x_span = float(np.ptp(homography_source[:, 0])) / width
    homography_y_span = float(np.ptp(homography_source[:, 1])) / height
    homography_coverage = homography_x_span * homography_y_span
    diagnostics.update(
        {
            "homography_inlier_x_span": homography_x_span,
            "homography_inlier_y_span": homography_y_span,
            "homography_inlier_coverage": homography_coverage,
        }
    )
    if homography_x_span < settings.minimum_inlier_x_span:
        return reject("homography_horizontal_coverage_too_small")
    if homography_y_span < settings.minimum_inlier_y_span:
        return reject("homography_vertical_coverage_too_small")
    if homography_coverage < settings.minimum_inlier_coverage:
        return reject("homography_area_coverage_too_small")
    homography_projection = cv2.perspectiveTransform(
        homography_source.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    homography_errors = np.linalg.norm(
        homography_projection - reference_points[homography_inliers], axis=1
    )
    homography_median = float(np.median(homography_errors))
    homography_p90 = _percentile(homography_errors, 0.90)
    perspective = _perspective_distortion(homography, width, height)
    diagnostics.update(
        {
            "homography_median_reprojection_error_pixels": homography_median,
            "homography_p90_reprojection_error_pixels": homography_p90,
            "perspective_distortion": perspective,
        }
    )
    if homography_median > settings.maximum_homography_median_reprojection_error_pixels:
        return reject("homography_median_reprojection_error_too_high")
    if homography_p90 > settings.maximum_homography_p90_reprojection_error_pixels:
        return reject("homography_p90_reprojection_error_too_high")
    if not math.isfinite(perspective):
        return reject("invalid_perspective_geometry")
    if perspective > settings.maximum_perspective_distortion:
        return reject("perspective_distortion_too_high")

    try:
        zoom, center_x, center_y, roll = _renderer_values_from_affine(
            affine, width=width, height=height
        )
    except ValueError as exc:
        return reject(str(exc))
    diagnostics["affine_matrix_source_to_reference"] = affine.tolist()
    diagnostics["estimated_renderer_transform"] = {
        "zoom": zoom,
        "center_x": center_x,
        "center_y": center_y,
        "roll_degrees": roll,
    }
    if zoom < settings.minimum_zoom - 1e-3:
        return reject("requires_zoom_out")
    if zoom > settings.maximum_zoom + 1e-9:
        return reject("scale_exceeds_renderer_zoom_bound")
    if math.hypot(center_x - 0.5, center_y - 0.5) > settings.maximum_center_shift + 1e-9:
        return reject("translation_exceeds_renderer_center_bound")
    if abs(roll) > settings.maximum_roll_degrees + 1e-9:
        return reject("rotation_exceeds_renderer_roll_bound")
    renderer_target, renderer_projection = _minimum_renderer_safe_target(
        zoom=zoom,
        center_x=center_x,
        center_y=center_y,
        roll_degrees=roll,
        settings=settings,
    )
    diagnostics["renderer_safe_zoom_projection"] = renderer_projection
    if renderer_target is None:
        return reject("renderer_transform_safety_rejection")
    renderer_affine = _affine_from_renderer_values(
        zoom=renderer_target["zoom"],
        center_x=renderer_target["center_x"],
        center_y=renderer_target["center_y"],
        roll_degrees=renderer_target["roll_degrees"],
        width=width,
        height=height,
    )
    warp_review = _warp_review(reference, source, renderer_affine)
    diagnostics["actual_warp_review"] = warp_review
    if warp_review["valid_fraction"] < settings.minimum_warp_valid_fraction:
        return reject("actual_warp_valid_coverage_too_small")
    if warp_review["warped_structure_ncc"] < settings.minimum_warp_structure_ncc:
        return reject("actual_warp_structure_match_too_low")
    if (
        warp_review["structure_ncc_improvement"]
        < settings.minimum_warp_structure_ncc_improvement
    ):
        return reject("actual_warp_improvement_too_small")
    diagnostics["decision"] = "all_per_frame_crop_supervision_gates_passed"
    return (
        _EstimatedTransform(
            renderer_affine,
            renderer_target["zoom"],
            renderer_target["center_x"],
            renderer_target["center_y"],
            renderer_target["roll_degrees"],
            diagnostics,
        ),
        None,
        diagnostics,
    )


def _validated_attributed_shot(
    request: ReferenceCropSupervisionRequest,
    attribution_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], int, float, float]:
    shot = deepcopy(dict(request.attributed_shot))
    if shot.get("accepted") is not True:
        raise ValueError("crop supervision requires an accepted attributed shot")
    if str(shot.get("attributed_video_id") or "") != request.source_id:
        raise ValueError("attributed shot source does not match requested source_id")
    shot_index = shot.get("shot_index")
    if isinstance(shot_index, bool) or not isinstance(shot_index, int) or shot_index < 0:
        raise ValueError("attributed shot_index must be a non-negative integer")
    start = _finite(shot.get("reference_start_seconds"), "shot start")
    end = _finite(shot.get("reference_end_seconds"), "shot end")
    if start < 0.0 or end <= start:
        raise ValueError("attributed shot must have a positive master-time span")
    canonical = shot.get("canonical_detected_shot")
    if not isinstance(canonical, Mapping):
        raise ValueError("crop supervision requires detector-owned canonical shot metadata")
    if (
        canonical.get("schema_version") != 1
        or canonical.get("detector_shot_index") != shot_index
        or canonical.get("source_run_merge_does_not_define_shot_partition") is not True
    ):
        raise ValueError("canonical detected-shot provenance is inconsistent")
    raw_shots = attribution_payload.get("shots")
    if not isinstance(raw_shots, list) or shot_index >= len(raw_shots):
        raise ValueError("attribution artifact does not contain the requested shot")
    artifact_shot = raw_shots[shot_index]
    if not isinstance(artifact_shot, Mapping) or _canonical_json_bytes(
        artifact_shot
    ) != _canonical_json_bytes(shot):
        raise ValueError("attributed shot differs from its provenance artifact")
    return (shot, shot_index, start, end)


def _temporal_review(
    transforms: Sequence[_EstimatedTransform],
    settings: ReferenceCropSupervisionConfig,
) -> tuple[dict[str, Any], str | None, dict[str, float] | None]:
    zoom = float(median(item.zoom for item in transforms))
    center_x = float(median(item.center_x for item in transforms))
    center_y = float(median(item.center_y for item in transforms))
    roll = float(median(item.roll_degrees for item in transforms))
    zoom_deviation = max(abs(item.zoom - zoom) for item in transforms)
    center_deviation = max(
        math.hypot(item.center_x - center_x, item.center_y - center_y)
        for item in transforms
    )
    roll_deviation = max(abs(item.roll_degrees - roll) for item in transforms)
    diagnostics = {
        "aggregation": "componentwise_median_of_independently_decoded_samples",
        "sample_count": len(transforms),
        "maximum_zoom_deviation": zoom_deviation,
        "maximum_center_deviation": center_deviation,
        "maximum_roll_deviation_degrees": roll_deviation,
        "median_renderer_transform": {
            "zoom": zoom,
            "center_x": center_x,
            "center_y": center_y,
            "roll_degrees": roll,
        },
    }
    if zoom_deviation > settings.maximum_temporal_zoom_deviation:
        return (diagnostics, "temporal_zoom_instability", None)
    if center_deviation > settings.maximum_temporal_center_deviation:
        return (diagnostics, "temporal_center_instability", None)
    if roll_deviation > settings.maximum_temporal_roll_deviation_degrees:
        return (diagnostics, "temporal_roll_instability", None)
    try:
        EditorialTransformKeyframe(0.0, zoom, center_x, center_y, roll)
    except ValueError:
        return (diagnostics, "aggregate_renderer_transform_safety_rejection", None)
    return (
        diagnostics,
        None,
        {
            "zoom": zoom,
            "center_x": center_x,
            "center_y": center_y,
            "roll_degrees": roll,
        },
    )


def _aggregate_actual_warp_review(
    target: Mapping[str, float],
    frame_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    settings: ReferenceCropSupervisionConfig,
) -> tuple[dict[str, Any], str | None]:
    reviews: list[dict[str, float]] = []
    for reference, source in frame_pairs:
        height, width = reference.shape[:2]
        affine = _affine_from_renderer_values(
            zoom=float(target["zoom"]),
            center_x=float(target["center_x"]),
            center_y=float(target["center_y"]),
            roll_degrees=float(target["roll_degrees"]),
            width=width,
            height=height,
        )
        reviews.append(_warp_review(reference, source, affine))
    minimum_valid = min(item["valid_fraction"] for item in reviews)
    median_ncc = float(median(item["warped_structure_ncc"] for item in reviews))
    median_improvement = float(
        median(item["structure_ncc_improvement"] for item in reviews)
    )
    minimum_improvement = min(item["structure_ncc_improvement"] for item in reviews)
    diagnostics: dict[str, Any] = {
        "basis": "actual_renderer_reconstruction_warp_on_every_interior_frame",
        "per_sample": reviews,
        "minimum_valid_fraction": minimum_valid,
        "median_warped_structure_ncc": median_ncc,
        "median_structure_ncc_improvement": median_improvement,
        "minimum_structure_ncc_improvement": minimum_improvement,
    }
    if minimum_valid < settings.minimum_warp_valid_fraction:
        return (diagnostics, "aggregate_warp_valid_coverage_too_small")
    if median_ncc < settings.minimum_warp_structure_ncc:
        return (diagnostics, "aggregate_warp_structure_match_too_low")
    if minimum_improvement < settings.minimum_warp_structure_ncc_improvement:
        return (diagnostics, "aggregate_warp_improvement_too_small")
    return (diagnostics, None)


def build_reference_crop_supervision(
    request: ReferenceCropSupervisionRequest,
    *,
    config: ReferenceCropSupervisionConfig | None = None,
) -> ReferenceCropSupervisionArtifact:
    """Build one fail-closed crop target from a detector-owned attributed shot."""

    if not isinstance(request, ReferenceCropSupervisionRequest):
        raise TypeError("request must be ReferenceCropSupervisionRequest")
    settings = config or ReferenceCropSupervisionConfig()
    initial_hashes = {
        "attribution": _file_sha256(request.attribution_path),
        "reference_media": _file_sha256(request.reference_media_path),
        "source_media": _file_sha256(request.source_media_path),
    }
    expected_hashes = {
        "attribution": request.expected_attribution_file_sha256,
        "reference_media": request.expected_reference_media_sha256,
        "source_media": request.expected_source_media_sha256,
    }
    for name, expected in expected_hashes.items():
        if expected is not None and expected != initial_hashes[name]:
            raise ValueError(f"{name} SHA-256 does not match the caller expectation")
    raw_attribution = request.attribution_path.read_bytes()
    if hashlib.sha256(raw_attribution).hexdigest() != initial_hashes["attribution"]:
        raise ValueError("attribution artifact changed while being read")
    attribution_payload = json.loads(raw_attribution)
    if not isinstance(attribution_payload, Mapping):
        raise ValueError("attribution artifact root must be an object")
    shot, shot_index, start, end = _validated_attributed_shot(
        request, attribution_payload
    )
    reference_info = _media_info(request.reference_media_path)
    source_info = _media_info(request.source_media_path)
    master_times = tuple(start + (end - start) * item for item in settings.sample_fractions)
    sample_payloads: list[dict[str, Any]] = []
    transforms: list[_EstimatedTransform] = []
    frame_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    reference_indices: list[int] = []
    source_indices: list[int] = []
    media_domain_rejection_count = 0
    for sample_index, master_seconds in enumerate(master_times):
        reference_seconds = request.reference_clock.master_to_source(master_seconds)
        source_seconds = request.source_clock.master_to_source(master_seconds)
        reference_in_domain = _media_time_is_decodable(
            reference_info, reference_seconds
        )
        source_in_domain = _media_time_is_decodable(source_info, source_seconds)
        if not reference_in_domain or not source_in_domain:
            media_domain_rejection_count += 1
            rejection = (
                "reference_clock_outside_media_domain"
                if not reference_in_domain
                else "source_clock_outside_media_domain"
            )
            sample_payloads.append(
                {
                    "sample_index": sample_index,
                    "shot_fraction": settings.sample_fractions[sample_index],
                    "master_seconds": master_seconds,
                    "reference_media_seconds_requested": reference_seconds,
                    "reference_frame_index": None,
                    "reference_frame_seconds_decoded": None,
                    "source_media_seconds_requested": source_seconds,
                    "source_frame_index": None,
                    "source_frame_seconds_decoded": None,
                    "accepted": False,
                    "rejection_reason": rejection,
                    "diagnostics": {
                        "decision": rejection,
                        "reference_in_declared_frame_domain": reference_in_domain,
                        "source_in_declared_frame_domain": source_in_domain,
                        "reference_declared_duration_seconds": (
                            reference_info.duration_seconds
                        ),
                        "source_declared_duration_seconds": (
                            source_info.duration_seconds
                        ),
                        "decode_attempted": False,
                        "clock_values_clamped": False,
                    },
                }
            )
            continue
        reference_decoded = _decode_frame(reference_info, reference_seconds)
        source_decoded = _decode_frame(source_info, source_seconds)
        reference_frame = _canonical_output(
            reference_decoded.frame, settings.working_width
        )
        source_frame = _canonical_output(source_decoded.frame, settings.working_width)
        transform, rejection, diagnostics = _estimate_sample(
            reference_frame, source_frame, settings
        )
        reference_indices.append(reference_decoded.frame_index)
        source_indices.append(source_decoded.frame_index)
        frame_pairs.append((reference_frame, source_frame))
        if transform is not None:
            transforms.append(transform)
        sample_payloads.append(
            {
                "sample_index": sample_index,
                "shot_fraction": settings.sample_fractions[sample_index],
                "master_seconds": master_seconds,
                "reference_media_seconds_requested": reference_seconds,
                "reference_frame_index": reference_decoded.frame_index,
                "reference_frame_seconds_decoded": reference_decoded.decoded_seconds,
                "source_media_seconds_requested": source_seconds,
                "source_frame_index": source_decoded.frame_index,
                "source_frame_seconds_decoded": source_decoded.decoded_seconds,
                "accepted": transform is not None,
                "rejection_reason": rejection,
                "diagnostics": diagnostics,
            }
        )

    rejection_reason: str | None = None
    target: dict[str, float] | None = None
    temporal_diagnostics: dict[str, Any]
    aggregate_warp_diagnostics: dict[str, Any] | None = None
    if media_domain_rejection_count:
        rejection_reason = "one_or_more_interior_sample_times_outside_media_domain"
        temporal_diagnostics = {
            "sample_count": len(master_times),
            "media_domain_rejection_count": media_domain_rejection_count,
            "decode_was_not_clamped_or_substituted": True,
        }
    elif len(set(reference_indices)) < settings.minimum_sample_count or len(
        set(source_indices)
    ) < settings.minimum_sample_count:
        rejection_reason = "insufficient_distinct_interior_decoded_frames"
        temporal_diagnostics = {
            "sample_count": len(master_times),
            "distinct_reference_frame_count": len(set(reference_indices)),
            "distinct_source_frame_count": len(set(source_indices)),
        }
    elif len(transforms) != len(master_times):
        rejection_reason = "one_or_more_interior_sample_gates_failed"
        temporal_diagnostics = {
            "sample_count": len(master_times),
            "accepted_sample_count": len(transforms),
            "required_all_configured_samples": True,
        }
    else:
        temporal_diagnostics, rejection_reason, target = _temporal_review(
            transforms, settings
        )
        if target is not None:
            aggregate_warp_diagnostics, aggregate_rejection = (
                _aggregate_actual_warp_review(target, frame_pairs, settings)
            )
            if aggregate_rejection is not None:
                rejection_reason = aggregate_rejection
                target = None

    final_hashes = {
        "attribution": _file_sha256(request.attribution_path),
        "reference_media": _file_sha256(request.reference_media_path),
        "source_media": _file_sha256(request.source_media_path),
    }
    if final_hashes != initial_hashes:
        raise ValueError("an input artifact changed during crop supervision decoding")
    eligible = target is not None and rejection_reason is None
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_basis": ANALYSIS_BASIS,
        "direction": "source_media_output_to_reference_edit_output",
        "shot": {
            "shot_index": shot_index,
            "source_id": request.source_id,
            "master_start_seconds": start,
            "master_end_seconds": end,
            "attributed_shot_content_sha256": _canonical_json_sha256(shot),
            "canonical_detected_shot": deepcopy(shot["canonical_detected_shot"]),
        },
        "configuration": settings.to_dict(),
        "time_mapping": {
            "master_clock_semantics": "shared_performance_clock",
            "reference_master_to_media": request.reference_clock.to_dict(),
            "source_master_to_media": request.source_clock.to_dict(),
        },
        "samples": sample_payloads,
        "temporal_stability_review": temporal_diagnostics,
        "aggregate_actual_warp_review": aggregate_warp_diagnostics,
        "supervision": {
            "eligible": eligible,
            "renderer_coordinate_system": (
                "EditorialTransformKeyframe_zoom_center_x_center_y_roll_degrees"
            ),
            "renderer_target": target,
            "rejection_reason": rejection_reason,
            "loss_mask": {
                "crop_transform_loss_masked": not eligible,
                "reason": rejection_reason,
                "unsafe_or_failed_values_clamped": False,
            },
        },
        "provenance": {
            "content_hash_basis": CONTENT_HASH_BASIS,
            "artifact_content_sha256": None,
            "attribution": {
                "path": str(request.attribution_path),
                "file_sha256": initial_hashes["attribution"],
                "canonical_json_sha256": _canonical_json_sha256(
                    attribution_payload
                ),
            },
            "reference_media": reference_info.to_dict(
                sha256=initial_hashes["reference_media"]
            ),
            "source_media": source_info.to_dict(
                sha256=initial_hashes["source_media"]
            ),
            "caller": deepcopy(dict(request.caller_provenance)),
            "input_hashes_rechecked_after_decode": True,
            "storage_policy": {
                "pixels_serialized": False,
                "descriptors_serialized": False,
                "matched_keypoints_serialized": False,
                "identity_or_appearance_embeddings_computed": False,
            },
        },
    }
    payload["provenance"]["artifact_content_sha256"] = _artifact_content_sha256(
        payload
    )
    validate_reference_crop_supervision_payload(payload)
    return ReferenceCropSupervisionArtifact(payload)


def validate_reference_crop_supervision_payload(
    value: Mapping[str, Any],
    *,
    verify_input_files: bool = False,
) -> None:
    """Fail closed on schema, self-hash, renderer target, and optional inputs."""

    if not isinstance(value, Mapping):
        raise TypeError("crop supervision payload must be an object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported crop supervision schema version")
    if value.get("analysis_basis") != ANALYSIS_BASIS:
        raise ValueError("unsupported crop supervision analysis basis")
    if value.get("direction") != "source_media_output_to_reference_edit_output":
        raise ValueError("crop supervision direction is invalid")
    samples = value.get("samples")
    if not isinstance(samples, list) or len(samples) < 3:
        raise ValueError("crop supervision must retain at least three sample audits")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("crop supervision provenance is missing")
    if provenance.get("content_hash_basis") != CONTENT_HASH_BASIS:
        raise ValueError("crop supervision content-hash basis is invalid")
    content_hash = provenance.get("artifact_content_sha256")
    if not _is_sha256(content_hash) or content_hash != _artifact_content_sha256(value):
        raise ValueError("crop supervision artifact content hash mismatch")
    supervision = value.get("supervision")
    if not isinstance(supervision, Mapping):
        raise ValueError("crop supervision decision is missing")
    eligible = supervision.get("eligible")
    target = supervision.get("renderer_target")
    rejection = supervision.get("rejection_reason")
    loss_mask = supervision.get("loss_mask")
    if not isinstance(eligible, bool) or not isinstance(loss_mask, Mapping):
        raise ValueError("crop supervision decision schema is invalid")
    if loss_mask.get("unsafe_or_failed_values_clamped") is not False:
        raise ValueError("crop supervision must never serialize clamped failures")
    if eligible:
        if not isinstance(target, Mapping) or rejection is not None:
            raise ValueError("eligible crop supervision needs exactly one target")
        EditorialTransformKeyframe(
            0.0,
            _finite(target.get("zoom"), "target zoom"),
            _finite(target.get("center_x"), "target center_x"),
            _finite(target.get("center_y"), "target center_y"),
            _finite(target.get("roll_degrees"), "target roll"),
        )
        if loss_mask.get("crop_transform_loss_masked") is not False:
            raise ValueError("eligible crop target cannot be loss-masked")
    else:
        if target is not None or not isinstance(rejection, str) or not rejection:
            raise ValueError("rejected crop supervision must be explicit and target-free")
        if loss_mask.get("crop_transform_loss_masked") is not True:
            raise ValueError("rejected crop supervision must be loss-masked")
    for name, digest_name in (
        ("attribution", "file_sha256"),
        ("reference_media", "media_sha256"),
        ("source_media", "media_sha256"),
    ):
        item = provenance.get(name)
        if not isinstance(item, Mapping) or not _is_sha256(item.get(digest_name)):
            raise ValueError(f"{name} provenance is invalid")
        path = Path(str(item.get("path") or ""))
        if not path.is_absolute():
            raise ValueError(f"{name} provenance path must be absolute")
        if verify_input_files:
            if not path.is_file() or _file_sha256(path) != item[digest_name]:
                raise ValueError(f"{name} provenance no longer matches its input")


def load_reference_crop_supervision(
    path: Path,
    *,
    verify_input_files: bool = True,
) -> ReferenceCropSupervisionArtifact:
    source = Path(path).expanduser()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("crop supervision JSON root must be an object")
    validate_reference_crop_supervision_payload(
        payload, verify_input_files=verify_input_files
    )
    return ReferenceCropSupervisionArtifact(deepcopy(dict(payload)))


__all__ = [
    "ANALYSIS_BASIS",
    "CONTENT_HASH_BASIS",
    "SCHEMA_VERSION",
    "ReferenceCropSupervisionArtifact",
    "ReferenceCropSupervisionConfig",
    "ReferenceCropSupervisionRequest",
    "build_reference_crop_supervision",
    "load_reference_crop_supervision",
    "validate_reference_crop_supervision_payload",
]
