"""Offline visual matching for finished stage edits and synchronized sources.

This module promotes the conservative reference-analysis matcher into reusable
production code.  It intentionally does not decide shot boundaries, align
audio, or write datasets.  Callers provide synchronized target frame indices;
the decoder reads their union once in ascending source order, and the matcher
returns manifest-ready visual evidence for each requested reference frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


ANALYSIS_BASIS = "orb_homography_structure_color_reference_match_v1"


def _finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


@dataclass(frozen=True, slots=True)
class ReferenceMatchConfig:
    """Exact feature/scoring policy calibrated on the IVE reference set."""

    working_width: int = 384
    orb_nfeatures: int = 900
    orb_fast_threshold: int = 12
    ratio_test_threshold: float = 0.76
    minimum_homography_matches: int = 8
    ransac_reprojection_threshold: float = 4.0
    structure_width: int = 64
    structure_height: int = 36
    hsv_hue_bins: int = 24
    hsv_saturation_bins: int = 16
    inlier_strength_denominator: float = 55.0
    good_match_strength_denominator: float = 90.0
    inlier_ratio_strength_denominator: float = 0.72
    inlier_weight: float = 0.50
    good_match_weight: float = 0.12
    geometric_quality_weight: float = 0.10
    structure_weight: float = 0.18
    color_weight: float = 0.10
    temporal_jitter_frames: tuple[int, ...] = (-2, 0, 2)

    def __post_init__(self) -> None:
        for name in (
            "working_width",
            "orb_nfeatures",
            "minimum_homography_matches",
            "structure_width",
            "structure_height",
            "hsv_hue_bins",
            "hsv_saturation_bins",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.working_width < 32:
            raise ValueError("working_width must be at least 32 pixels")
        if self.minimum_homography_matches < 4:
            raise ValueError("minimum_homography_matches must be at least four")
        if (
            isinstance(self.orb_fast_threshold, bool)
            or not isinstance(self.orb_fast_threshold, int)
            or self.orb_fast_threshold < 0
        ):
            raise ValueError("orb_fast_threshold must be a non-negative integer")
        ratio = _finite(self.ratio_test_threshold, "ratio_test_threshold")
        if not 0.0 < ratio < 1.0:
            raise ValueError("ratio_test_threshold must be between zero and one")
        for name in (
            "ransac_reprojection_threshold",
            "inlier_strength_denominator",
            "good_match_strength_denominator",
            "inlier_ratio_strength_denominator",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be greater than zero")
        weights = (
            self.inlier_weight,
            self.good_match_weight,
            self.geometric_quality_weight,
            self.structure_weight,
            self.color_weight,
        )
        if any(_finite(value, "match weight") < 0.0 for value in weights):
            raise ValueError("match weights must be non-negative")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("match weights must sum to one")
        if not self.temporal_jitter_frames:
            raise ValueError("temporal_jitter_frames cannot be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.temporal_jitter_frames
        ):
            raise ValueError("temporal_jitter_frames must contain integers")
        if len(set(self.temporal_jitter_frames)) != len(
            self.temporal_jitter_frames
        ):
            raise ValueError("temporal_jitter_frames cannot contain duplicates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": ANALYSIS_BASIS,
            "working_width": self.working_width,
            "orb": {
                "nfeatures": self.orb_nfeatures,
                "fast_threshold": self.orb_fast_threshold,
                "ratio_test_threshold": self.ratio_test_threshold,
                "minimum_homography_matches": self.minimum_homography_matches,
                "ransac_reprojection_threshold": (
                    self.ransac_reprojection_threshold
                ),
            },
            "structure_ncc_size": [
                self.structure_width,
                self.structure_height,
            ],
            "hsv_histogram_bins": [
                self.hsv_hue_bins,
                self.hsv_saturation_bins,
            ],
            "score_weights": {
                "homography_inliers": self.inlier_weight,
                "ratio_matches": self.good_match_weight,
                "homography_inlier_ratio": self.geometric_quality_weight,
                "structure_ncc": self.structure_weight,
                "hsv_color_similarity": self.color_weight,
            },
            "strength_denominators": {
                "homography_inliers": self.inlier_strength_denominator,
                "ratio_matches": self.good_match_strength_denominator,
                "homography_inlier_ratio": (
                    self.inlier_ratio_strength_denominator
                ),
            },
            "temporal_jitter_frames": list(self.temporal_jitter_frames),
        }


@dataclass(frozen=True, slots=True)
class ReferenceAcceptanceConfig:
    minimum_score: float = 0.40
    minimum_homography_inliers: int = 10
    minimum_inlier_ratio: float = 0.25
    minimum_top_two_margin: float = 0.055
    high_confidence_score: float = 0.68

    def __post_init__(self) -> None:
        for name in (
            "minimum_score",
            "minimum_inlier_ratio",
            "minimum_top_two_margin",
            "high_confidence_score",
        ):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if (
            isinstance(self.minimum_homography_inliers, bool)
            or not isinstance(self.minimum_homography_inliers, int)
            or self.minimum_homography_inliers < 1
        ):
            raise ValueError("minimum_homography_inliers must be positive")
        if self.high_confidence_score < self.minimum_score:
            raise ValueError("high_confidence_score cannot be below minimum_score")

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "unknown_on_low_confidence": True,
            "minimum_score": self.minimum_score,
            "minimum_homography_inliers": self.minimum_homography_inliers,
            "minimum_inlier_ratio": self.minimum_inlier_ratio,
            "minimum_top_two_margin_below_high_confidence": (
                self.minimum_top_two_margin
            ),
            "high_confidence_score": self.high_confidence_score,
        }


@dataclass(frozen=True, slots=True)
class FrameFeatures:
    """Transient non-serialized features for one resized BGR frame."""

    structure: NDArray[np.float32]
    hsv_histogram: NDArray[np.float32]
    keypoints: tuple[Any, ...]
    descriptors: NDArray[np.uint8] | None


@dataclass(frozen=True, slots=True)
class FrameMatch:
    score: float
    orb_good_matches: int
    orb_homography_inliers: int
    orb_inlier_ratio: float
    structure_ncc: float
    color_similarity: float
    homography_reference_to_source: (
        tuple[tuple[float, float, float], ...] | None
    ) = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "orb_good_matches": self.orb_good_matches,
            "orb_homography_inliers": self.orb_homography_inliers,
            "orb_inlier_ratio": self.orb_inlier_ratio,
            "structure_ncc": self.structure_ncc,
            "color_similarity": self.color_similarity,
            "homography_reference_to_source": (
                [list(row) for row in self.homography_reference_to_source]
                if self.homography_reference_to_source is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class TimedFrameMatch(FrameMatch):
    source_frame_index: int = 0
    source_time_seconds: float = 0.0
    time_adjustment_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **FrameMatch.to_dict(self),
            "source_frame_index": self.source_frame_index,
            "source_time_seconds": self.source_time_seconds,
            "time_adjustment_seconds": self.time_adjustment_seconds,
        }


@dataclass(frozen=True, slots=True)
class ReferenceMatchDecision:
    accepted: bool
    reason: str
    top_two_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "decision_reason": self.reason,
            "top_two_margin": self.top_two_margin,
        }


@dataclass(frozen=True, slots=True)
class DecodedTargetFrames:
    """One sequential decode pass over the requested source-frame indices."""

    frame_rate: float
    declared_frame_count: int
    requested_indices: tuple[int, ...]
    frames: Mapping[int, NDArray[np.uint8]]

    @property
    def decoded_indices(self) -> tuple[int, ...]:
        return tuple(index for index in self.requested_indices if index in self.frames)

    @property
    def missing_indices(self) -> tuple[int, ...]:
        return tuple(index for index in self.requested_indices if index not in self.frames)


def _working_frame(
    frame: NDArray[np.uint8],
    working_width: int,
) -> NDArray[np.uint8]:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("reference matching requires a BGR image")
    height, width = frame.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("reference matching requires a non-empty image")
    target_height = max(2, int(round(height * working_width / width)))
    if width == working_width and height == target_height:
        return frame
    return cv2.resize(
        frame,
        (working_width, target_height),
        # Keep the exact reference-analysis policy even for an unusually small
        # input. Production broadcast sources are always downscaled here.
        interpolation=cv2.INTER_AREA,
    )


class OfflineReferenceMatcher:
    """Reusable ORB/NCC/HSV matcher with one cached ORB extractor."""

    def __init__(self, config: ReferenceMatchConfig | None = None) -> None:
        self.config = config or ReferenceMatchConfig()
        self._orb = cv2.ORB_create(
            nfeatures=self.config.orb_nfeatures,
            fastThreshold=self.config.orb_fast_threshold,
        )
        self._descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def extract(self, frame: NDArray[np.uint8]) -> FrameFeatures:
        working = _working_frame(frame, self.config.working_width)
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        structure = cv2.resize(
            gray,
            (self.config.structure_width, self.config.structure_height),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        structure -= float(structure.mean())
        norm = float(np.linalg.norm(structure))
        if norm > 1e-6:
            structure /= norm
        hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [self.config.hsv_hue_bins, self.config.hsv_saturation_bins],
            [0, 180, 0, 256],
        ).astype(np.float32)
        cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        return FrameFeatures(
            structure=structure,
            hsv_histogram=histogram,
            keypoints=tuple(keypoints or ()),
            descriptors=descriptors,
        )

    def compare(self, reference: FrameFeatures, source: FrameFeatures) -> FrameMatch:
        if reference.structure.shape != source.structure.shape:
            raise ValueError("structure feature shapes must match")
        if reference.hsv_histogram.shape != source.hsv_histogram.shape:
            raise ValueError("HSV histogram feature shapes must match")
        structure_ncc = float(np.sum(reference.structure * source.structure))
        structure_ncc = min(1.0, max(-1.0, structure_ncc))
        color_similarity = 1.0 - float(
            cv2.compareHist(
                reference.hsv_histogram,
                source.hsv_histogram,
                cv2.HISTCMP_BHATTACHARYYA,
            )
        )
        color_similarity = min(1.0, max(0.0, color_similarity))

        good: list[Any] = []
        left = reference.descriptors
        right = source.descriptors
        if (
            left is not None
            and right is not None
            and len(left) >= 4
            and len(right) >= 4
        ):
            try:
                pairs = self._descriptor_matcher.knnMatch(left, right, k=2)
            except cv2.error:
                pairs = ()
            good = [
                pair[0]
                for pair in pairs
                if len(pair) == 2
                and pair[0].distance
                < self.config.ratio_test_threshold * pair[1].distance
            ]

        inliers = 0
        inlier_ratio = 0.0
        homography: NDArray[np.float64] | None = None
        if len(good) >= self.config.minimum_homography_matches:
            reference_points = np.float32(
                [reference.keypoints[item.queryIdx].pt for item in good]
            ).reshape(-1, 1, 2)
            source_points = np.float32(
                [source.keypoints[item.trainIdx].pt for item in good]
            ).reshape(-1, 1, 2)
            try:
                matrix, mask = cv2.findHomography(
                    reference_points,
                    source_points,
                    cv2.RANSAC,
                    self.config.ransac_reprojection_threshold,
                )
            except cv2.error:
                matrix, mask = None, None
            if matrix is not None and mask is not None and matrix.shape == (3, 3):
                inliers = int(mask.ravel().sum())
                inlier_ratio = inliers / len(good)
                homography = np.asarray(matrix, dtype=np.float64)

        inlier_strength = min(
            1.0,
            inliers / self.config.inlier_strength_denominator,
        )
        good_strength = min(
            1.0,
            len(good) / self.config.good_match_strength_denominator,
        )
        geometric_quality = min(
            1.0,
            inlier_ratio / self.config.inlier_ratio_strength_denominator,
        )
        combined = (
            self.config.inlier_weight * inlier_strength
            + self.config.good_match_weight * good_strength
            + self.config.geometric_quality_weight * geometric_quality
            + self.config.structure_weight * max(0.0, structure_ncc)
            + self.config.color_weight * max(0.0, color_similarity)
        )
        serialized_homography = (
            tuple(
                tuple(float(value) for value in row)
                for row in homography.tolist()
            )
            if homography is not None
            else None
        )
        return FrameMatch(
            score=min(1.0, max(0.0, combined)),
            orb_good_matches=len(good),
            orb_homography_inliers=inliers,
            orb_inlier_ratio=inlier_ratio,
            structure_ncc=structure_ncc,
            color_similarity=color_similarity,
            homography_reference_to_source=serialized_homography,
        )


def extract_frame_features(
    frame: NDArray[np.uint8],
    *,
    config: ReferenceMatchConfig | None = None,
) -> FrameFeatures:
    return OfflineReferenceMatcher(config).extract(frame)


def match_frame_features(
    reference: FrameFeatures,
    source: FrameFeatures,
    *,
    config: ReferenceMatchConfig | None = None,
) -> FrameMatch:
    return OfflineReferenceMatcher(config).compare(reference, source)


def _requested_indices(indices: Iterable[int]) -> tuple[int, ...]:
    requested: set[int] = set()
    for value in indices:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("target frame indices must be integers")
        if value >= 0:
            requested.add(value)
    return tuple(sorted(requested))


def decode_target_frames_sequential(
    path: Path,
    target_indices: Iterable[int],
    *,
    config: ReferenceMatchConfig | None = None,
) -> DecodedTargetFrames:
    """Decode requested indices in one forward-only pass with no random seeks."""

    settings = config or ReferenceMatchConfig()
    requested = _requested_indices(target_indices)
    if not requested:
        return DecodedTargetFrames(
            frame_rate=0.0,
            declared_frame_count=0,
            requested_indices=(),
            frames={},
        )
    capture = cv2.VideoCapture(str(Path(path)))
    if not capture.isOpened():
        raise ValueError(f"cannot decode reference-match source: {path}")
    frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
    declared_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    declared_count = (
        int(round(declared_count_value))
        if math.isfinite(declared_count_value) and declared_count_value > 0.0
        else 0
    )
    if not math.isfinite(frame_rate) or frame_rate <= 0.0:
        capture.release()
        raise ValueError(f"invalid source frame rate: {path}")

    frames: dict[int, NDArray[np.uint8]] = {}
    target_set = set(requested)
    final_index = requested[-1]
    index = 0
    try:
        while index <= final_index:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index in target_set:
                frames[index] = _working_frame(frame, settings.working_width)
                if len(frames) == len(requested):
                    break
            index += 1
    finally:
        capture.release()
    return DecodedTargetFrames(
        frame_rate=frame_rate,
        declared_frame_count=max(0, declared_count),
        requested_indices=requested,
        frames=frames,
    )


def match_aligned_target_frames(
    source_path: Path,
    target_indices_by_item: Sequence[Sequence[int]],
    reference_features: Sequence[FrameFeatures],
    *,
    expected_source_times: Sequence[float] | None = None,
    config: ReferenceMatchConfig | None = None,
    matcher: OfflineReferenceMatcher | None = None,
) -> tuple[TimedFrameMatch | None, ...]:
    """Choose the strongest synchronized target frame for every reference item.

    All distinct source indices are decoded once.  ``target_indices_by_item``
    normally contains the configured ``(-2, 0, +2)`` neighborhood for each
    audio-aligned shot midpoint.
    """

    if len(target_indices_by_item) != len(reference_features):
        raise ValueError("target-index groups must match reference feature count")
    if expected_source_times is not None and len(expected_source_times) != len(
        reference_features
    ):
        raise ValueError("expected source times must match reference feature count")
    normalized_expected_times: tuple[float, ...] | None = None
    if expected_source_times is not None:
        normalized_expected_times = tuple(
            _finite(value, "expected source time")
            for value in expected_source_times
        )
        if any(value < 0.0 for value in normalized_expected_times):
            raise ValueError("expected source times cannot be negative")
    if matcher is not None and config is not None and matcher.config != config:
        raise ValueError("matcher and config cannot specify different policies")
    active_matcher = matcher or OfflineReferenceMatcher(config)
    normalized_groups = tuple(
        _requested_indices(group) for group in target_indices_by_item
    )
    batch = decode_target_frames_sequential(
        source_path,
        (
            index
            for group in normalized_groups
            for index in group
        ),
        config=active_matcher.config,
    )
    if batch.requested_indices and batch.frame_rate <= 0.0:
        raise ValueError("decoded target frames do not have a valid frame rate")
    features_by_index = {
        index: active_matcher.extract(frame)
        for index, frame in batch.frames.items()
    }
    output: list[TimedFrameMatch | None] = []
    for item_index, (reference, candidate_indices) in enumerate(
        zip(reference_features, normalized_groups, strict=True)
    ):
        best_index: int | None = None
        best_match: FrameMatch | None = None
        # Repeated/held frames can produce bit-identical scores for every
        # jitter candidate. Prefer the audio-aligned time when supplied;
        # otherwise prefer the group's temporal center. This prevents the
        # sorted (-2, 0, +2) neighborhood from introducing a systematic -2
        # frame bias.
        group_center = (
            (
                candidate_indices[(len(candidate_indices) - 1) // 2]
                + candidate_indices[len(candidate_indices) // 2]
            )
            / 2.0
            if candidate_indices
            else 0.0
        )
        expected_time = (
            normalized_expected_times[item_index]
            if normalized_expected_times is not None
            else None
        )

        def temporal_distance(frame_index: int) -> float:
            if expected_time is not None:
                return abs(frame_index / batch.frame_rate - expected_time)
            return abs(frame_index - group_center)

        for frame_index in candidate_indices:
            features = features_by_index.get(frame_index)
            if features is None:
                continue
            candidate = active_matcher.compare(reference, features)
            score_better = (
                best_match is None
                or candidate.score > best_match.score + 1e-12
            )
            score_tied_and_closer = (
                best_match is not None
                and best_index is not None
                and math.isclose(
                    candidate.score,
                    best_match.score,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and temporal_distance(frame_index)
                < temporal_distance(best_index) - 1e-12
            )
            if score_better or score_tied_and_closer:
                best_index = frame_index
                best_match = candidate
        if best_index is None or best_match is None:
            output.append(None)
            continue
        source_time = best_index / batch.frame_rate
        expected = expected_time
        output.append(
            TimedFrameMatch(
                score=best_match.score,
                orb_good_matches=best_match.orb_good_matches,
                orb_homography_inliers=(
                    best_match.orb_homography_inliers
                ),
                orb_inlier_ratio=best_match.orb_inlier_ratio,
                structure_ncc=best_match.structure_ncc,
                color_similarity=best_match.color_similarity,
                homography_reference_to_source=(
                    best_match.homography_reference_to_source
                ),
                source_frame_index=best_index,
                source_time_seconds=source_time,
                time_adjustment_seconds=(
                    source_time - expected if expected is not None else None
                ),
            )
        )
    return tuple(output)


def decide_reference_match(
    best: FrameMatch | TimedFrameMatch | None,
    runner_up_score: float = 0.0,
    *,
    config: ReferenceAcceptanceConfig | None = None,
) -> ReferenceMatchDecision:
    """Apply the scratch matcher's conservative unknown-on-doubt policy."""

    settings = config or ReferenceAcceptanceConfig()
    if best is None:
        return ReferenceMatchDecision(False, "no_candidate", 0.0)
    second = _finite(runner_up_score, "runner_up_score")
    margin = best.score - second
    if best.score < settings.minimum_score:
        return ReferenceMatchDecision(False, "combined_score_below_0_40", margin)
    if best.orb_homography_inliers < settings.minimum_homography_inliers:
        return ReferenceMatchDecision(
            False,
            "fewer_than_10_homography_inliers",
            margin,
        )
    if best.orb_inlier_ratio < settings.minimum_inlier_ratio:
        return ReferenceMatchDecision(
            False,
            "homography_inlier_ratio_below_0_25",
            margin,
        )
    if (
        margin < settings.minimum_top_two_margin
        and best.score < settings.high_confidence_score
    ):
        return ReferenceMatchDecision(
            False,
            "top_two_margin_below_0_055",
            margin,
        )
    return ReferenceMatchDecision(
        True,
        "conservative_orb_homography_match",
        margin,
    )


__all__ = [
    "ANALYSIS_BASIS",
    "DecodedTargetFrames",
    "FrameFeatures",
    "FrameMatch",
    "OfflineReferenceMatcher",
    "ReferenceAcceptanceConfig",
    "ReferenceMatchConfig",
    "ReferenceMatchDecision",
    "TimedFrameMatch",
    "decode_target_frames_sequential",
    "decide_reference_match",
    "extract_frame_features",
    "match_aligned_target_frames",
    "match_frame_features",
]
