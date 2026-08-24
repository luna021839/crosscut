"""Conservative output-space affine evidence for one cross-source hard cut.

The matcher estimates how the incoming rendered frame would have to move to
meet the outgoing frame.  ORB correspondences are filtered bidirectionally,
then a RANSAC partial affine supplies scale/rotation/translation while a
separate homography checks that perspective distortion is not being hidden by
the simpler model.  The result is evidence only: a transform becomes eligible
solely when it fits the renderer's zoom/pan/roll limits and improves supplied
non-identifying subject geometry without cropping that subject.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .render import EditorialTransformKeyframe, minimum_roll_safe_zoom


ANALYSIS_BASIS = "orb_mutual_ratio_ransac_partial_affine_subject_gate_v1"


def _finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _normalized_box(
    value: Sequence[float] | None,
    name: str,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if len(value) != 4:
        raise ValueError(f"{name} needs four values")
    x, y, width, height = tuple(_finite(item, name) for item in value)
    if not (
        0.0 <= x <= 1.0
        and 0.0 <= y <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
        and x + width <= 1.000001
        and y + height <= 1.000001
    ):
        raise ValueError(f"{name} must fit normalized frame bounds")
    return (x, y, width, height)


@dataclass(frozen=True, slots=True)
class BoundaryAffineMatchConfig:
    working_width: int = 640
    orb_nfeatures: int = 1800
    orb_fast_threshold: int = 10
    ratio_test_threshold: float = 0.76
    minimum_ratio_matches: int = 24
    minimum_mutual_matches: int = 16
    minimum_affine_inliers: int = 13
    minimum_affine_inlier_ratio: float = 0.52
    ransac_reprojection_threshold_pixels: float = 3.0
    maximum_median_reprojection_error_pixels: float = 2.4
    maximum_p90_reprojection_error_pixels: float = 4.8
    minimum_inlier_x_span: float = 0.22
    minimum_inlier_y_span: float = 0.18
    minimum_inlier_coverage: float = 0.055
    maximum_perspective_distortion: float = 0.075
    minimum_determinant: float = 0.94
    maximum_determinant: float = 1.40
    maximum_zoom: float = 1.18
    maximum_center_shift: float = 0.025
    maximum_roll_degrees: float = 1.60
    minimum_subject_visible_fraction: float = 0.97
    minimum_baseline_geometry_mismatch: float = 0.035
    minimum_geometry_improvement: float = 0.020
    minimum_geometry_improvement_fraction: float = 0.16
    ransac_seed: int = 74117

    def __post_init__(self) -> None:
        integer_fields = (
            "working_width",
            "orb_nfeatures",
            "orb_fast_threshold",
            "minimum_ratio_matches",
            "minimum_mutual_matches",
            "minimum_affine_inliers",
            "ransac_seed",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.working_width < 160 or self.orb_nfeatures < 100:
            raise ValueError("affine working size and feature count are too small")
        if self.orb_fast_threshold < 0 or self.ransac_seed < 0:
            raise ValueError("ORB threshold and RANSAC seed must be non-negative")
        if min(
            self.minimum_ratio_matches,
            self.minimum_mutual_matches,
            self.minimum_affine_inliers,
        ) < 4:
            raise ValueError("affine matching gates need at least four matches")
        normalized_fields = (
            "minimum_affine_inlier_ratio",
            "minimum_inlier_x_span",
            "minimum_inlier_y_span",
            "minimum_inlier_coverage",
            "maximum_perspective_distortion",
            "maximum_center_shift",
            "minimum_subject_visible_fraction",
            "minimum_baseline_geometry_mismatch",
            "minimum_geometry_improvement",
            "minimum_geometry_improvement_fraction",
        )
        for name in normalized_fields:
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be normalized")
        ratio = _finite(self.ratio_test_threshold, "ratio_test_threshold")
        if not 0.0 < ratio < 1.0:
            raise ValueError("ratio_test_threshold must be in (0, 1)")
        for name in (
            "ransac_reprojection_threshold_pixels",
            "maximum_median_reprojection_error_pixels",
            "maximum_p90_reprojection_error_pixels",
            "minimum_determinant",
            "maximum_determinant",
            "maximum_zoom",
            "maximum_roll_degrees",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 1.0 < self.maximum_zoom <= 1.18:
            raise ValueError("maximum_zoom must fit EditorialTransformKeyframe")
        if self.maximum_center_shift > 0.04:
            raise ValueError("maximum_center_shift exceeds renderer safety")
        if self.maximum_roll_degrees > 1.60:
            raise ValueError("maximum_roll_degrees exceeds renderer safety")
        if self.maximum_determinant <= self.minimum_determinant:
            raise ValueError("determinant bounds are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": ANALYSIS_BASIS,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            },
        }


@dataclass(frozen=True, slots=True)
class BoundaryAffineTransformEvidence:
    zoom: float
    center_x: float
    center_y: float
    roll_degrees: float

    def __post_init__(self) -> None:
        # This is the exact production schema/safety validator.  Keeping the
        # check here prevents a serialized eligible result from being stronger
        # than the renderer can actually consume.
        EditorialTransformKeyframe(
            0.0,
            self.zoom,
            self.center_x,
            self.center_y,
            self.roll_degrees,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "zoom": self.zoom,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "roll_degrees": self.roll_degrees,
        }


@dataclass(frozen=True, slots=True)
class BoundaryAffineMatchResult:
    eligible: bool
    rejection_reason: str | None
    transform: BoundaryAffineTransformEvidence | None
    diagnostics: Mapping[str, Any]
    analysis_basis: str = ANALYSIS_BASIS

    def __post_init__(self) -> None:
        if self.eligible != (self.transform is not None):
            raise ValueError("eligible affine result needs exactly one transform")
        if self.eligible and self.rejection_reason is not None:
            raise ValueError("eligible affine result cannot have a rejection reason")
        if not self.eligible and not self.rejection_reason:
            raise ValueError("rejected affine result needs an explicit reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": self.analysis_basis,
            "eligible": self.eligible,
            "rejection_reason": self.rejection_reason,
            "transform": self.transform.to_dict() if self.transform else None,
            "diagnostics": dict(self.diagnostics),
            "storage_policy": {
                "pixels_serialized": False,
                "descriptors_serialized": False,
                "matched_keypoints_serialized": False,
                "identity_or_appearance_embeddings_computed": False,
            },
        }


def _reject(reason: str, diagnostics: Mapping[str, Any]) -> BoundaryAffineMatchResult:
    return BoundaryAffineMatchResult(False, reason, None, dict(diagnostics))


def _working_gray(frame: np.ndarray, width: int) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("boundary affine matching expects a BGR image")
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError("boundary affine matching expects a non-empty frame")
    if frame.shape[1] <= width:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height = max(2, int(round(frame.shape[0] * width / frame.shape[1])))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def _ratio_matches(
    descriptors_left: np.ndarray,
    descriptors_right: np.ndarray,
    threshold: float,
) -> tuple[cv2.DMatch, ...]:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(descriptors_left, descriptors_right, k=2)
    accepted = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in (pair,)
        if first.distance < threshold * second.distance
    ]
    return tuple(
        sorted(
            accepted,
            key=lambda item: (item.distance, item.queryIdx, item.trainIdx),
        )
    )


def _mutual_matches(
    incoming_descriptors: np.ndarray,
    outgoing_descriptors: np.ndarray,
    threshold: float,
) -> tuple[cv2.DMatch, ...]:
    forward = _ratio_matches(incoming_descriptors, outgoing_descriptors, threshold)
    reverse = _ratio_matches(outgoing_descriptors, incoming_descriptors, threshold)
    reverse_pairs = {(item.trainIdx, item.queryIdx) for item in reverse}
    return tuple(
        item
        for item in forward
        if (item.queryIdx, item.trainIdx) in reverse_pairs
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
    denominators = [
        float(matrix[2, 0] * x + matrix[2, 1] * y + 1.0)
        for x, y in ((0, 0), (width, 0), (0, height), (width, height))
    ]
    if any(value <= 1e-6 or not math.isfinite(value) for value in denominators):
        return math.inf
    return max(denominators) / min(denominators) - 1.0


def _apply_editorial_point(
    point: tuple[float, float],
    transform: BoundaryAffineTransformEvidence,
    aspect: float,
) -> tuple[float, float]:
    x_value, y_value = point
    dx = x_value - 0.5
    dy = y_value - 0.5
    radians = math.radians(transform.roll_degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    rotated_x = 0.5 + cosine * dx - sine * dy / aspect
    rotated_y = 0.5 + sine * aspect * dx + cosine * dy
    return (
        transform.zoom * (rotated_x - transform.center_x) + 0.5,
        transform.zoom * (rotated_y - transform.center_y) + 0.5,
    )


def _transform_box(
    box: tuple[float, float, float, float],
    transform: BoundaryAffineTransformEvidence,
    aspect: float,
) -> tuple[tuple[float, float, float, float], float]:
    x, y, width, height = box
    points = tuple(
        _apply_editorial_point(point, transform, aspect)
        for point in (
            (x, y),
            (x + width, y),
            (x, y + height),
            (x + width, y + height),
        )
    )
    left = min(point[0] for point in points)
    right = max(point[0] for point in points)
    top = min(point[1] for point in points)
    bottom = max(point[1] for point in points)
    transformed = (left, top, right - left, bottom - top)
    area = max(1e-12, transformed[2] * transformed[3])
    clipped_width = max(0.0, min(1.0, right) - max(0.0, left))
    clipped_height = max(0.0, min(1.0, bottom) - max(0.0, top))
    visibility = _unit(clipped_width * clipped_height / area)
    return (transformed, visibility)


def _geometry_mismatch(
    outgoing: tuple[float, float, float, float],
    incoming: tuple[float, float, float, float],
) -> float:
    outgoing_center = (
        outgoing[0] + outgoing[2] / 2.0,
        outgoing[1] + outgoing[3] / 2.0,
    )
    incoming_center = (
        incoming[0] + incoming[2] / 2.0,
        incoming[1] + incoming[3] / 2.0,
    )
    center = _unit(
        math.hypot(
            outgoing_center[0] - incoming_center[0],
            outgoing_center[1] - incoming_center[1],
        )
        / 0.35
    )
    height = _unit(abs(math.log(outgoing[3] / incoming[3])) / math.log(1.8))
    width = _unit(abs(math.log(outgoing[2] / incoming[2])) / math.log(1.8))
    return 0.58 * center + 0.24 * height + 0.18 * width


def _candidate_center(
    affine: np.ndarray,
    width: int,
    height: int,
    zoom: float,
) -> tuple[float, float]:
    input_center = np.asarray([width / 2.0, height / 2.0, 1.0])
    target = affine @ input_center
    target_x = float(target[0]) / width
    target_y = float(target[1]) / height
    return (
        0.5 - (target_x - 0.5) / zoom,
        0.5 - (target_y - 0.5) / zoom,
    )


def match_boundary_affine(
    outgoing_frame: np.ndarray,
    incoming_frame: np.ndarray,
    *,
    outgoing_subject_box: Sequence[float] | None = None,
    incoming_subject_box: Sequence[float] | None = None,
    config: BoundaryAffineMatchConfig | None = None,
) -> BoundaryAffineMatchResult:
    """Estimate bounded evidence that maps incoming output into outgoing output."""

    settings = config or BoundaryAffineMatchConfig()
    if (
        not isinstance(outgoing_frame, np.ndarray)
        or not isinstance(incoming_frame, np.ndarray)
        or outgoing_frame.shape != incoming_frame.shape
    ):
        raise ValueError("boundary frames must be equal-sized BGR images")
    outgoing_box = _normalized_box(outgoing_subject_box, "outgoing subject box")
    incoming_box = _normalized_box(incoming_subject_box, "incoming subject box")
    outgoing_gray = _working_gray(outgoing_frame, settings.working_width)
    incoming_gray = _working_gray(incoming_frame, settings.working_width)
    height, width = outgoing_gray.shape
    diagnostics: dict[str, Any] = {
        "frame_size": [width, height],
        "subject_geometry_supplied": (
            outgoing_box is not None and incoming_box is not None
        ),
        "match_direction": "incoming_output_to_outgoing_output",
    }

    orb = cv2.ORB_create(
        nfeatures=settings.orb_nfeatures,
        fastThreshold=settings.orb_fast_threshold,
        edgeThreshold=19,
    )
    outgoing_keypoints, outgoing_descriptors = orb.detectAndCompute(
        outgoing_gray, None
    )
    incoming_keypoints, incoming_descriptors = orb.detectAndCompute(
        incoming_gray, None
    )
    diagnostics["outgoing_keypoint_count"] = len(outgoing_keypoints)
    diagnostics["incoming_keypoint_count"] = len(incoming_keypoints)
    if outgoing_descriptors is None or incoming_descriptors is None:
        return _reject("descriptors_unavailable", diagnostics)

    forward = _ratio_matches(
        incoming_descriptors,
        outgoing_descriptors,
        settings.ratio_test_threshold,
    )
    diagnostics["ratio_match_count"] = len(forward)
    if len(forward) < settings.minimum_ratio_matches:
        return _reject("insufficient_ratio_matches", diagnostics)
    matches = _mutual_matches(
        incoming_descriptors,
        outgoing_descriptors,
        settings.ratio_test_threshold,
    )
    diagnostics["mutual_match_count"] = len(matches)
    if len(matches) < settings.minimum_mutual_matches:
        return _reject("insufficient_mutual_matches", diagnostics)

    incoming_points = np.float32(
        [incoming_keypoints[item.queryIdx].pt for item in matches]
    )
    outgoing_points = np.float32(
        [outgoing_keypoints[item.trainIdx].pt for item in matches]
    )
    cv2.setRNGSeed(settings.ransac_seed)
    affine, affine_mask = cv2.estimateAffinePartial2D(
        incoming_points,
        outgoing_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=settings.ransac_reprojection_threshold_pixels,
        maxIters=4000,
        confidence=0.995,
        refineIters=20,
    )
    if affine is None or affine_mask is None:
        return _reject("affine_estimation_failed", diagnostics)
    inlier_mask = affine_mask.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / len(matches)
    diagnostics["affine_inlier_count"] = inlier_count
    diagnostics["affine_inlier_ratio"] = inlier_ratio
    if inlier_count < settings.minimum_affine_inliers:
        return _reject("insufficient_affine_inliers", diagnostics)
    if inlier_ratio < settings.minimum_affine_inlier_ratio:
        return _reject("affine_inlier_ratio_too_low", diagnostics)

    inlier_incoming = incoming_points[inlier_mask]
    inlier_outgoing = outgoing_points[inlier_mask]
    projected = cv2.transform(inlier_incoming.reshape(-1, 1, 2), affine).reshape(-1, 2)
    errors = np.linalg.norm(projected - inlier_outgoing, axis=1)
    median_error = float(np.median(errors))
    p90_error = _percentile(errors, 0.90)
    diagnostics["median_reprojection_error_pixels"] = median_error
    diagnostics["p90_reprojection_error_pixels"] = p90_error
    if median_error > settings.maximum_median_reprojection_error_pixels:
        return _reject("median_reprojection_error_too_high", diagnostics)
    if p90_error > settings.maximum_p90_reprojection_error_pixels:
        return _reject("p90_reprojection_error_too_high", diagnostics)

    x_span = float(np.ptp(inlier_incoming[:, 0])) / width
    y_span = float(np.ptp(inlier_incoming[:, 1])) / height
    coverage = x_span * y_span
    diagnostics["inlier_x_span"] = x_span
    diagnostics["inlier_y_span"] = y_span
    diagnostics["inlier_coverage"] = coverage
    if x_span < settings.minimum_inlier_x_span:
        return _reject("inlier_horizontal_coverage_too_small", diagnostics)
    if y_span < settings.minimum_inlier_y_span:
        return _reject("inlier_vertical_coverage_too_small", diagnostics)
    if coverage < settings.minimum_inlier_coverage:
        return _reject("inlier_area_coverage_too_small", diagnostics)

    cv2.setRNGSeed(settings.ransac_seed)
    homography, homography_mask = cv2.findHomography(
        incoming_points,
        outgoing_points,
        cv2.RANSAC,
        settings.ransac_reprojection_threshold_pixels,
        maxIters=4000,
        confidence=0.995,
    )
    if homography is None or homography_mask is None:
        return _reject("perspective_validation_unavailable", diagnostics)
    homography_inliers = int(homography_mask.reshape(-1).sum())
    perspective = _perspective_distortion(homography, width, height)
    diagnostics["homography_inlier_count"] = homography_inliers
    diagnostics["perspective_distortion"] = perspective
    if not math.isfinite(perspective):
        return _reject("invalid_perspective_geometry", diagnostics)
    if perspective > settings.maximum_perspective_distortion:
        return _reject("perspective_distortion_too_high", diagnostics)

    linear = affine[:, :2].astype(np.float64)
    determinant = float(np.linalg.det(linear))
    scale = math.sqrt(max(0.0, float(linear[0, 0] ** 2 + linear[1, 0] ** 2)))
    roll_degrees = math.degrees(math.atan2(float(linear[1, 0]), float(linear[0, 0])))
    diagnostics["affine_matrix_incoming_to_outgoing"] = affine.tolist()
    diagnostics["determinant"] = determinant
    diagnostics["estimated_scale"] = scale
    diagnostics["estimated_roll_degrees"] = roll_degrees
    if determinant <= 0.0:
        return _reject("reflected_or_singular_affine", diagnostics)
    if not settings.minimum_determinant <= determinant <= settings.maximum_determinant:
        return _reject("determinant_outside_safe_range", diagnostics)
    if scale < 1.0 - 1e-3:
        return _reject("requires_zoom_out", diagnostics)
    if scale > settings.maximum_zoom + 1e-9:
        return _reject("scale_exceeds_zoom_bound", diagnostics)
    if abs(roll_degrees) > settings.maximum_roll_degrees + 1e-9:
        return _reject("rotation_exceeds_roll_bound", diagnostics)

    zoom = max(1.0, scale)
    if abs(roll_degrees) > 1e-9:
        zoom = max(zoom, minimum_roll_safe_zoom(roll_degrees))
    diagnostics["roll_safe_zoom"] = zoom
    if zoom > settings.maximum_zoom + 1e-9:
        return _reject("roll_overscan_exceeds_zoom_bound", diagnostics)
    center_x, center_y = _candidate_center(affine, width, height, zoom)
    center_shift = math.hypot(center_x - 0.5, center_y - 0.5)
    diagnostics["estimated_center_xy"] = [center_x, center_y]
    diagnostics["center_shift"] = center_shift
    if center_shift > settings.maximum_center_shift + 1e-9:
        return _reject("translation_exceeds_center_shift_bound", diagnostics)

    try:
        transform = BoundaryAffineTransformEvidence(
            zoom=zoom,
            center_x=center_x,
            center_y=center_y,
            roll_degrees=roll_degrees,
        )
    except ValueError:
        return _reject("renderer_transform_safety_rejection", diagnostics)

    if outgoing_box is None or incoming_box is None:
        return _reject("subject_geometry_unavailable", diagnostics)
    baseline_mismatch = _geometry_mismatch(outgoing_box, incoming_box)
    transformed_box, visibility = _transform_box(
        incoming_box,
        transform,
        outgoing_frame.shape[1] / outgoing_frame.shape[0],
    )
    transformed_mismatch = _geometry_mismatch(outgoing_box, transformed_box)
    improvement = baseline_mismatch - transformed_mismatch
    improvement_fraction = improvement / max(1e-9, baseline_mismatch)
    diagnostics["subject_review"] = {
        "outgoing_box_xywh": list(outgoing_box),
        "incoming_box_xywh": list(incoming_box),
        "transformed_incoming_box_xywh": list(transformed_box),
        "transformed_visible_fraction": visibility,
        "baseline_geometry_mismatch": baseline_mismatch,
        "transformed_geometry_mismatch": transformed_mismatch,
        "geometry_improvement": improvement,
        "geometry_improvement_fraction": improvement_fraction,
    }
    if visibility < settings.minimum_subject_visible_fraction:
        return _reject("transformed_subject_visibility_too_low", diagnostics)
    if baseline_mismatch < settings.minimum_baseline_geometry_mismatch:
        return _reject("subject_geometry_already_matched", diagnostics)
    if improvement < settings.minimum_geometry_improvement:
        return _reject("subject_geometry_improvement_too_small", diagnostics)
    if improvement_fraction < settings.minimum_geometry_improvement_fraction:
        return _reject("subject_geometry_improvement_fraction_too_small", diagnostics)

    diagnostics["decision"] = "bounded_affine_improves_visible_subject_geometry"
    return BoundaryAffineMatchResult(True, None, transform, diagnostics)


__all__ = [
    "ANALYSIS_BASIS",
    "BoundaryAffineMatchConfig",
    "BoundaryAffineMatchResult",
    "BoundaryAffineTransformEvidence",
    "match_boundary_affine",
]
