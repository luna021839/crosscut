"""Frame scoring for the group cross-edit draft.

The engine does not identify IVE members in broadcast footage. It combines
technical image evidence with optional on-device human pose and aggregate
silhouette models. Entry and exit descriptors remain separate so a cut compares
the actual outgoing and incoming frames.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .face_identity import SFACE_BACKEND, extract_sface_embedding
from .pose import SubjectGeometryDescriptor, describe_subject_geometry
from .timeline import (
    ShotType,
    TimelineCandidate,
    TimelineConfig,
    TimelineResult,
    face_geometry_minimax_mismatch,
    select_timeline,
)


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True, slots=True)
class VisualDiversityConfig:
    """Safe preference strength for the metadata source-coverage order."""

    preferred_source_bonus: float = 0.72
    maximum_preferred_score_sacrifice: float = 0.58

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.preferred_source_bonus)
            or self.preferred_source_bonus < 0.0
        ):
            raise ValueError("preferred source bonus must be finite and non-negative")
        if (
            not math.isfinite(self.maximum_preferred_score_sacrifice)
            or self.maximum_preferred_score_sacrifice < 0.0
        ):
            raise ValueError(
                "maximum preferred score sacrifice must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class FrameVisualDescriptor:
    quality_score: float
    composition_score: float
    shot_scale: float
    shot_type: ShotType
    face_count: int
    structure_embedding: tuple[float, ...]
    motion: tuple[float, float] | None = None
    # Normalized x/y/width/height for the only safely dominant face. Multiple
    # similarly-sized detections deliberately remain ``None``: basic
    # face-match cross-edit must fall back to a raw cut rather than crop an
    # arbitrary group member.
    primary_face_bbox: tuple[float, float, float, float] | None = None
    color_embedding: tuple[float, ...] = ()
    face_detector_backend: str = "none"
    primary_face_landmarks_xy: tuple[tuple[float, float], ...] | None = None
    primary_face_head_roll_radians: float | None = None
    primary_face_eye_scale: float | None = None
    primary_face_geometry: tuple[float, ...] | None = None
    pose_embedding: tuple[float, ...] = ()
    pose_confidence: float = 0.0
    detected_person_count: int = 0
    pose_backend: str = "structure_hog_fallback"
    silhouette_embedding: tuple[float, ...] = ()
    silhouette_bbox_xywh: tuple[float, float, float, float] | None = None
    silhouette_centroid_xy: tuple[float, float] | None = None
    silhouette_area_ratio: float = 0.0
    silhouette_confidence: float = 0.0
    silhouette_backend: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_score": self.quality_score,
            "composition_score": self.composition_score,
            "shot_scale": self.shot_scale,
            "shot_type": self.shot_type.value,
            "face_count": self.face_count,
            "primary_face_bbox_xywh": (
                list(self.primary_face_bbox)
                if self.primary_face_bbox is not None
                else None
            ),
            "color_embedding": list(self.color_embedding),
            "face_detector_backend": self.face_detector_backend,
            "primary_face_landmarks_xy": (
                [list(point) for point in self.primary_face_landmarks_xy]
                if self.primary_face_landmarks_xy is not None
                else None
            ),
            "primary_face_head_roll_radians": self.primary_face_head_roll_radians,
            "primary_face_eye_scale": self.primary_face_eye_scale,
            "primary_face_geometry": (
                list(self.primary_face_geometry)
                if self.primary_face_geometry is not None
                else None
            ),
            "primary_face_head_pose_proxies": (
                {
                    "yaw_nose_horizontal": self.primary_face_geometry[6],
                    "pitch_nose_vertical": self.primary_face_geometry[7],
                    "mouth_vertical": self.primary_face_geometry[8],
                    "mouth_asymmetry": self.primary_face_geometry[9],
                }
                if self.primary_face_geometry is not None
                else None
            ),
            "pose_backend": self.pose_backend,
            "pose_confidence": self.pose_confidence,
            "pose_available": self.pose_backend != "structure_hog_fallback",
            "detected_person_count": self.detected_person_count,
            "silhouette_backend": self.silhouette_backend,
            "silhouette_confidence": self.silhouette_confidence,
            "silhouette_bbox_xywh": (
                list(self.silhouette_bbox_xywh)
                if self.silhouette_bbox_xywh is not None
                else None
            ),
            "silhouette_centroid_xy": (
                list(self.silhouette_centroid_xy)
                if self.silhouette_centroid_xy is not None
                else None
            ),
            "silhouette_area_ratio": self.silhouette_area_ratio,
            "motion": list(self.motion) if self.motion is not None else None,
        }


@dataclass(frozen=True, slots=True)
class VisualTimelineChoice:
    segment_index: int
    source_index: int
    canonical_master_start: float
    canonical_master_end: float
    source_local_start: float
    source_local_end: float
    quality_score: float
    composition_score: float
    boundary_score: float
    shot_scale: float
    shot_type: ShotType
    shot_id: str
    face_samples_xywh: tuple[
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
    ] = (None, None, None)
    face_match_candidate_failure: str | None = None
    face_detector_backends: tuple[str, str, str] = ("none", "none", "none")
    face_landmark_samples_xy: tuple[
        tuple[tuple[float, float], ...] | None,
        tuple[tuple[float, float], ...] | None,
        tuple[tuple[float, float], ...] | None,
    ] = (None, None, None)
    face_head_roll_samples_radians: tuple[
        float | None, float | None, float | None
    ] = (None, None, None)
    face_eye_scale_samples: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )
    entry_face_geometry_window: tuple[tuple[float, ...], ...] = ()
    exit_face_geometry_window: tuple[tuple[float, ...], ...] = ()
    # Dissolves need different evidence from hard cuts: both sources are shown
    # *after* the nominal boundary.  These windows sample cut, handle midpoint,
    # and handle end and must never be substituted with the pre-cut minimax
    # windows above.
    dissolve_entry_face_geometry_window: tuple[tuple[float, ...], ...] = ()
    dissolve_exit_face_geometry_window: tuple[tuple[float, ...], ...] = ()
    dissolve_entry_face_box_window: tuple[
        tuple[float, float, float, float], ...
    ] = ()
    dissolve_exit_face_box_window: tuple[
        tuple[float, float, float, float], ...
    ] = ()
    # Raw appearance/body vectors are transient safety evidence. ``to_dict``
    # intentionally records only backend/count/confidence metadata; biometric
    # embeddings are never persisted to the analysis manifest.
    dissolve_entry_face_identity_window: tuple[tuple[float, ...], ...] = ()
    dissolve_exit_face_identity_window: tuple[tuple[float, ...], ...] = ()
    dissolve_entry_pose_embedding_window: tuple[tuple[float, ...], ...] = ()
    dissolve_exit_pose_embedding_window: tuple[tuple[float, ...], ...] = ()
    dissolve_entry_pose_backend_window: tuple[str, ...] = ()
    dissolve_exit_pose_backend_window: tuple[str, ...] = ()
    dissolve_entry_pose_confidence_window: tuple[float, ...] = ()
    dissolve_exit_pose_confidence_window: tuple[float, ...] = ()
    dissolve_entry_silhouette_embedding_window: tuple[
        tuple[float, ...], ...
    ] = ()
    dissolve_exit_silhouette_embedding_window: tuple[
        tuple[float, ...], ...
    ] = ()
    dissolve_entry_silhouette_backend_window: tuple[str, ...] = ()
    dissolve_exit_silhouette_backend_window: tuple[str, ...] = ()
    dissolve_entry_silhouette_confidence_window: tuple[float, ...] = ()
    dissolve_exit_silhouette_confidence_window: tuple[float, ...] = ()
    face_transition_minimax_mismatch_from_previous: float | None = None
    pose_backends: tuple[str, str, str] = (
        "structure_hog_fallback",
        "structure_hog_fallback",
        "structure_hog_fallback",
    )
    pose_confidences: tuple[float, float, float] = (0.0, 0.0, 0.0)
    detected_person_counts: tuple[int, int, int] = (0, 0, 0)
    silhouette_samples_xywh: tuple[
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
    ] = (None, None, None)
    silhouette_centroid_samples_xy: tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
    ] = (None, None, None)
    silhouette_backends: tuple[str, str, str] = ("none", "none", "none")
    silhouette_confidences: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pose_transition_basis_from_previous: str | None = None
    pose_transition_mismatch_from_previous: float | None = None
    silhouette_transition_mismatch_from_previous: float | None = None

    @property
    def duration_seconds(self) -> float:
        return self.canonical_master_end - self.canonical_master_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "source_index": self.source_index,
            "canonical_master_start": self.canonical_master_start,
            "canonical_master_end": self.canonical_master_end,
            "source_local_start": self.source_local_start,
            "source_local_end": self.source_local_end,
            "duration_seconds": self.duration_seconds,
            "quality_score": self.quality_score,
            "composition_score": self.composition_score,
            "boundary_score": self.boundary_score,
            "shot_scale": self.shot_scale,
            "shot_type": self.shot_type.value,
            "shot_id": self.shot_id,
            "face_samples_xywh": [
                list(sample) if sample is not None else None
                for sample in self.face_samples_xywh
            ],
            "face_match_candidate_failure": self.face_match_candidate_failure,
            "face_detector_backends": list(self.face_detector_backends),
            "face_landmark_samples_xy": [
                [list(point) for point in sample] if sample is not None else None
                for sample in self.face_landmark_samples_xy
            ],
            "face_head_roll_samples_radians": list(
                self.face_head_roll_samples_radians
            ),
            "face_eye_scale_samples": list(self.face_eye_scale_samples),
            "entry_face_geometry_window": [
                list(geometry) for geometry in self.entry_face_geometry_window
            ],
            "exit_face_geometry_window": [
                list(geometry) for geometry in self.exit_face_geometry_window
            ],
            "dissolve_entry_face_geometry_window": [
                list(geometry)
                for geometry in self.dissolve_entry_face_geometry_window
            ],
            "dissolve_exit_face_geometry_window": [
                list(geometry)
                for geometry in self.dissolve_exit_face_geometry_window
            ],
            "dissolve_entry_face_box_window": [
                list(box) for box in self.dissolve_entry_face_box_window
            ],
            "dissolve_exit_face_box_window": [
                list(box) for box in self.dissolve_exit_face_box_window
            ],
            "dissolve_entry_evidence": {
                "appearance_compatibility_backend": SFACE_BACKEND,
                "appearance_embedding_count": sum(
                    bool(embedding)
                    for embedding in self.dissolve_entry_face_identity_window
                ),
                "pose_backends": list(self.dissolve_entry_pose_backend_window),
                "pose_confidences": list(
                    self.dissolve_entry_pose_confidence_window
                ),
                "silhouette_backends": list(
                    self.dissolve_entry_silhouette_backend_window
                ),
                "silhouette_confidences": list(
                    self.dissolve_entry_silhouette_confidence_window
                ),
                "raw_embeddings_persisted": False,
            },
            "dissolve_exit_evidence": {
                "appearance_compatibility_backend": SFACE_BACKEND,
                "appearance_embedding_count": sum(
                    bool(embedding)
                    for embedding in self.dissolve_exit_face_identity_window
                ),
                "pose_backends": list(self.dissolve_exit_pose_backend_window),
                "pose_confidences": list(
                    self.dissolve_exit_pose_confidence_window
                ),
                "silhouette_backends": list(
                    self.dissolve_exit_silhouette_backend_window
                ),
                "silhouette_confidences": list(
                    self.dissolve_exit_silhouette_confidence_window
                ),
                "raw_embeddings_persisted": False,
            },
            "face_transition_minimax_mismatch_from_previous": (
                self.face_transition_minimax_mismatch_from_previous
            ),
            "pose_backends": list(self.pose_backends),
            "pose_confidences": list(self.pose_confidences),
            "detected_person_counts": list(self.detected_person_counts),
            "silhouette_samples_xywh": [
                list(sample) if sample is not None else None
                for sample in self.silhouette_samples_xywh
            ],
            "silhouette_centroid_samples_xy": [
                list(sample) if sample is not None else None
                for sample in self.silhouette_centroid_samples_xy
            ],
            "silhouette_backends": list(self.silhouette_backends),
            "silhouette_confidences": list(self.silhouette_confidences),
            "pose_transition_basis_from_previous": (
                self.pose_transition_basis_from_previous
            ),
            "pose_transition_mismatch_from_previous": (
                self.pose_transition_mismatch_from_previous
            ),
            "silhouette_transition_mismatch_from_previous": (
                self.silhouette_transition_mismatch_from_previous
            ),
        }


@dataclass(frozen=True, slots=True)
class VisualTimelinePlan:
    choices: tuple[VisualTimelineChoice, ...]
    optimizer_score: float
    analysis_basis: str = (
        "frame_quality_pose_silhouette_motion_color_face_minimax_coverage_v6"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": self.analysis_basis,
            "optimizer_score": self.optimizer_score,
            "face_transition_policy": {
                "detector": "opencv_zoo_yunet_2023mar",
                "window": "three_consecutive_synchronized_frame_pairs",
                "objective": "worst_pair_mismatch_penalty",
                "missing_or_ambiguous_face": "neutral",
                "selection_effect": "penalty_only_never_eligibility_gate",
                "provenance": "independent_modern_adaptation_of_cross_cutting_eye_minimax",
            },
            "dissolve_safety_window": {
                "detector": "opencv_zoo_yunet_2023mar",
                "appearance_model": SFACE_BACKEND,
                "sample_count": 3,
                "handle_frames": _DISSOLVE_HANDLE_FRAMES,
                "timing": "cut_handle_midpoint_handle_end_on_both_sources",
                "scene_cut_or_ambiguous_face": "empty_never_blend",
                "future_body_requirement": (
                    "reliable_pose_or_aggregate_silhouette_at_all_three_samples"
                ),
                "missing_future_body": "tier_b_subject_aligned_hard_cut",
                "purpose": "eligibility_gate_not_timeline_reward",
                "inference_scope": (
                    "post_dp_selected_source_change_boundaries_with_safe_face_candidates"
                ),
                "interpretation": (
                    "appearance_compatibility_gate_not_named_identity_or_authentication"
                ),
                "raw_appearance_embeddings_persisted": False,
            },
            "body_transition_policy": {
                "pose": "opencv_zoo_mediapipe_33_keypoints_when_one_subject_is_dominant",
                "silhouette": "opencv_zoo_pphumanseg_aggregate_human_layout",
                "pose_missing": "deterministic_structure_hog_fallback",
                "silhouette_missing": "neutral_not_an_eligibility_gate",
                "selection_effect": "penalty_only_never_eligibility_gate",
                "dissolve_effect": (
                    "future_model_evidence_required_for_tier_a_only"
                ),
            },
            "choices": [choice.to_dict() for choice in self.choices],
        }


_FACE_CASCADE: cv2.CascadeClassifier | None = None
_YUNET_DETECTORS: dict[tuple[int, int], Any] = {}
_YUNET_DISABLED = False
_YUNET_INTEGRITY_VALID: bool | None = None
YUNET_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_EXPECTED_SHA256 = (
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
)
_YUNET_MODEL = Path(__file__).resolve().parent / "models" / YUNET_MODEL_NAME


def yunet_model_asset_status(
    *,
    model_path: Path | None = None,
    verify_load: bool = False,
) -> dict[str, Any]:
    """Report exact YuNet integrity and optional FaceDetectorYN loadability."""

    path = model_path or _YUNET_MODEL
    actual_digest: str | None = None
    if path.is_file():
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            actual_digest = digest.hexdigest()
        except OSError:
            actual_digest = None
    digest_valid = actual_digest == YUNET_EXPECTED_SHA256
    loadable: bool | None = None
    if verify_load and digest_valid:
        try:
            cv2.FaceDetectorYN.create(
                str(path),
                "",
                (320, 180),
                0.76,
                0.3,
                5000,
            )
            loadable = True
        except (cv2.error, AttributeError):
            loadable = False
    return {
        "name": YUNET_MODEL_NAME,
        "path": str(path),
        "present": path.is_file(),
        "digest_valid": digest_valid,
        "loadable": loadable,
        "expected_sha256": YUNET_EXPECTED_SHA256,
        "actual_sha256": actual_digest,
    }


@dataclass(frozen=True, slots=True)
class _DetectedFace:
    bbox_xywh: tuple[float, float, float, float]
    landmarks_xy: tuple[tuple[float, float], ...] | None = None
    confidence: float | None = None


def _face_cascade() -> cv2.CascadeClassifier | None:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        classifier = cv2.CascadeClassifier(str(path))
        _FACE_CASCADE = classifier
    return None if _FACE_CASCADE.empty() else _FACE_CASCADE


def _yunet_detector(width: int, height: int) -> Any | None:
    """Load the bundled MIT-licensed OpenCV Zoo YuNet detector when possible."""

    global _YUNET_DISABLED, _YUNET_INTEGRITY_VALID
    if _YUNET_DISABLED or not _YUNET_MODEL.is_file() or not hasattr(
        cv2, "FaceDetectorYN"
    ):
        return None
    if _YUNET_INTEGRITY_VALID is None:
        _YUNET_INTEGRITY_VALID = bool(
            yunet_model_asset_status()["digest_valid"]
        )
    if not _YUNET_INTEGRITY_VALID:
        _YUNET_DISABLED = True
        return None
    size = (width, height)
    if size not in _YUNET_DETECTORS:
        try:
            _YUNET_DETECTORS[size] = cv2.FaceDetectorYN.create(
                str(_YUNET_MODEL),
                "",
                size,
                0.76,
                0.3,
                5000,
            )
        except (cv2.error, AttributeError):
            _YUNET_DISABLED = True
            return None
    return _YUNET_DETECTORS[size]


def _detect_faces(
    working: np.ndarray,
    gray: np.ndarray,
) -> tuple[tuple[_DetectedFace, ...], str]:
    height, width = gray.shape
    detector = _yunet_detector(width, height)
    if detector is not None:
        try:
            _result, detections = detector.detect(working)
            if detections is None:
                return ((), "yunet")
            faces = tuple(
                _DetectedFace(
                    bbox_xywh=tuple(float(value) for value in detection[:4]),
                    landmarks_xy=tuple(
                        (float(detection[index]), float(detection[index + 1]))
                        for index in range(4, 14, 2)
                    ),
                    confidence=float(detection[-1]),
                )
                for detection in detections
                if float(detection[-1]) >= 0.76
            )
            return (faces, "yunet")
        except cv2.error:
            # A malformed frame/model mismatch is not a reason to abort the
            # complete edit. Haar remains a local, dependency-free fallback.
            pass
    cascade = _face_cascade()
    if cascade is None or min(gray.shape) < 48:
        return ((), "none")
    boxes = cascade.detectMultiScale(
        gray,
        scaleFactor=1.16,
        minNeighbors=4,
        minSize=(18, 18),
    )
    return (
        tuple(
            _DetectedFace(tuple(float(value) for value in box))
            for box in boxes
        ),
        "haar",
    )


def _working_frame(frame: np.ndarray, maximum_width: int = 480) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
        raise ValueError("frame must be a non-empty grayscale or BGR image")
    height, width = frame.shape[:2]
    if width > maximum_width:
        scale = maximum_width / width
        frame = cv2.resize(
            frame,
            (maximum_width, max(2, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.shape[2] != 3:
        raise ValueError("frame must use 1, 3, or 4 channels")
    return frame


def _structure_embedding(gray: np.ndarray) -> tuple[float, ...]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    rows, columns, bins = 4, 6, 4
    embedding: list[float] = []
    height, width = gray.shape
    for row in range(rows):
        y0, y1 = row * height // rows, (row + 1) * height // rows
        for column in range(columns):
            x0, x1 = column * width // columns, (column + 1) * width // columns
            cell_magnitude = magnitude[y0:y1, x0:x1]
            cell_angle = angle[y0:y1, x0:x1]
            histogram = np.zeros(bins, dtype=np.float64)
            indices = np.floor(cell_angle * bins / (2 * math.pi)).astype(np.int32) % bins
            for bin_index in range(bins):
                histogram[bin_index] = float(cell_magnitude[indices == bin_index].sum())
            total = float(histogram.sum())
            if total > 1e-9:
                histogram /= total
            embedding.extend(float(value) for value in histogram)
    return tuple(embedding)


def _color_embedding(frame: np.ndarray, gray: np.ndarray) -> tuple[float, ...]:
    """Compact global color/exposure signature used only at cut boundaries."""

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    means = lab.reshape(-1, 3).mean(axis=0) / 255.0
    deviations = lab.reshape(-1, 3).std(axis=0) / 128.0
    return tuple(
        float(min(1.0, max(0.0, value)))
        for value in (
            *means,
            *deviations,
            float(gray.mean()) / 255.0,
            float(gray.std()) / 128.0,
        )
    )


def _motion(gray: np.ndarray, following: np.ndarray | None) -> tuple[float, float] | None:
    if following is None:
        return None
    following_gray = cv2.cvtColor(_working_frame(following), cv2.COLOR_BGR2GRAY)
    if following_gray.shape != gray.shape:
        following_gray = cv2.resize(following_gray, (gray.shape[1], gray.shape[0]))
    flow = cv2.calcOpticalFlowFarneback(
        gray,
        following_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.1,
        0,
    )
    height, width = gray.shape
    center = flow[height // 6 : height * 5 // 6, width // 6 : width * 5 // 6]
    if center.size == 0:
        return None
    x = float(np.median(center[:, :, 0])) / max(1.0, width * 0.03)
    y = float(np.median(center[:, :, 1])) / max(1.0, height * 0.03)
    return (max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y)))


def _dominant_face(faces: Sequence[_DetectedFace]) -> _DetectedFace | None:
    """Return one unambiguous dominant face without guessing in group shots."""

    if not faces:
        return None
    ordered = sorted(
        faces,
        key=lambda face: face.bbox_xywh[2] * face.bbox_xywh[3],
        reverse=True,
    )
    largest = ordered[0]
    if len(ordered) > 1:
        largest_area = float(largest.bbox_xywh[2] * largest.bbox_xywh[3])
        runner_up_area = float(
            ordered[1].bbox_xywh[2] * ordered[1].bbox_xywh[3]
        )
        # A group shot often contains several valid faces. Only a clearly
        # dominant detection is safe enough to drive a crop.
        if largest_area < runner_up_area * 1.8:
            return None
    return largest


def _normalized_face_metadata(
    face: _DetectedFace | None,
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[
    tuple[float, float, float, float] | None,
    tuple[tuple[float, float], ...] | None,
    float | None,
    float | None,
    tuple[float, ...] | None,
]:
    """Normalize YuNet's five points and derive stable 2D head geometry.

    The compact geometry vector is intentionally detector-agnostic at the DP
    boundary: image-left/right eye endpoints, eye scale, roll, and nose/mouth
    offsets in the eye-aligned coordinate system. Haar detections retain the
    crop bbox but return no geometry, making the transition term neutral.
    """

    if face is None:
        return (None, None, None, None, None)
    x, y, width, height = face.bbox_xywh
    bbox = (
        _unit(x / frame_width),
        _unit(y / frame_height),
        _unit(width / frame_width),
        _unit(height / frame_height),
    )
    if face.landmarks_xy is None or len(face.landmarks_xy) != 5:
        return (bbox, None, None, None, None)
    normalized_landmarks = tuple(
        (_unit(point_x / frame_width), _unit(point_y / frame_height))
        for point_x, point_y in face.landmarks_xy
    )
    # OpenCV YuNet returns two eyes, nose, and two mouth corners. Canonicalize
    # eye direction by image x so roll sign is stable across detector versions.
    eye_left_px, eye_right_px = sorted(face.landmarks_xy[:2], key=lambda point: point[0])
    delta_x = eye_right_px[0] - eye_left_px[0]
    delta_y = eye_right_px[1] - eye_left_px[1]
    eye_distance_px = math.hypot(delta_x, delta_y)
    if eye_distance_px < 2.0:
        return (bbox, normalized_landmarks, None, None, None)
    roll = math.atan2(delta_y, delta_x)
    eye_scale = eye_distance_px / frame_height
    cosine = delta_x / eye_distance_px
    sine = delta_y / eye_distance_px
    eye_mid_x = (eye_left_px[0] + eye_right_px[0]) / 2.0
    eye_mid_y = (eye_left_px[1] + eye_right_px[1]) / 2.0

    def local_offset(point: tuple[float, float]) -> tuple[float, float]:
        point_x = point[0] - eye_mid_x
        point_y = point[1] - eye_mid_y
        return (
            (point_x * cosine + point_y * sine) / eye_distance_px,
            (-point_x * sine + point_y * cosine) / eye_distance_px,
        )

    nose = local_offset(face.landmarks_xy[2])
    mouth_corners = sorted(
        (local_offset(face.landmarks_xy[3]), local_offset(face.landmarks_xy[4])),
        key=lambda point: point[0],
    )
    mouth_vertical = (mouth_corners[0][1] + mouth_corners[1][1]) / 2.0
    mouth_asymmetry = mouth_corners[1][1] - mouth_corners[0][1]
    normalized_eye_left, normalized_eye_right = sorted(
        normalized_landmarks[:2], key=lambda point: point[0]
    )
    geometry = (
        normalized_eye_left[0],
        normalized_eye_left[1],
        normalized_eye_right[0],
        normalized_eye_right[1],
        eye_scale,
        roll,
        # Five 2D points cannot provide a full 3D head pose, but these stable
        # eye-aligned quantities act as lightweight yaw/pitch/asymmetry proxies.
        nose[0],
        nose[1],
        mouth_vertical,
        mouth_asymmetry,
    )
    return (
        bbox,
        normalized_landmarks,
        roll,
        eye_scale,
        geometry,
    )


def describe_frame(
    frame: np.ndarray,
    *,
    following_frame: np.ndarray | None = None,
    include_subject_geometry: bool = True,
) -> FrameVisualDescriptor:
    working = _working_frame(frame)
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    structure_embedding = _structure_embedding(gray)
    subject = (
        describe_subject_geometry(working)
        if include_subject_geometry
        else SubjectGeometryDescriptor()
    )
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness = 1.0 - math.exp(-laplacian_variance / 180.0)
    mean = float(gray.mean())
    contrast = float(gray.std())
    clipped = float(np.mean((gray < 8) | (gray > 247)))
    exposure = math.exp(-((mean - 118.0) / 105.0) ** 2) * (1.0 - min(0.8, clipped))
    contrast_score = min(1.0, contrast / 58.0)
    quality = _unit(0.45 * sharpness + 0.32 * exposure + 0.23 * contrast_score)

    edges = cv2.Canny(gray, 50, 140).astype(np.float32) / 255.0
    moments = cv2.moments(edges)
    if moments["m00"] > 1e-6:
        center_x = moments["m10"] / moments["m00"] / gray.shape[1]
        center_y = moments["m01"] / moments["m00"] / gray.shape[0]
        center_distance = math.hypot(center_x - 0.5, center_y - 0.48) / 0.72
        composition = 1.0 - _unit(center_distance)
    else:
        composition = 0.35

    if subject.silhouette_centroid_xy is not None:
        subject_x, subject_y = subject.silhouette_centroid_xy
        subject_center_score = 1.0 - _unit(
            math.hypot(subject_x - 0.5, subject_y - 0.50) / 0.72
        )
        composition = _unit(0.72 * composition + 0.28 * subject_center_score)

    faces, face_detector_backend = _detect_faces(working, gray)
    face_count = len(faces)
    dominant_face = _dominant_face(faces)
    (
        primary_face_bbox,
        primary_face_landmarks,
        primary_face_roll,
        primary_face_eye_scale,
        primary_face_geometry,
    ) = _normalized_face_metadata(
        dominant_face,
        frame_width=gray.shape[1],
        frame_height=gray.shape[0],
    )
    if face_count:
        largest = max(
            faces,
            key=lambda face: face.bbox_xywh[2] * face.bbox_xywh[3],
        )
        x, y, width, height = largest.bbox_xywh
        face_scale = height / gray.shape[0]
        face_center_x = (x + width / 2) / gray.shape[1]
        face_center_y = (y + height / 2) / gray.shape[0]
        face_center_score = 1.0 - _unit(
            math.hypot(face_center_x - 0.5, face_center_y - 0.42) / 0.68
        )
        composition = _unit(0.55 * composition + 0.45 * face_center_score)
        shot_scale = _unit(face_scale / 0.42)
    else:
        shot_scale = 0.12

    if face_count >= 4 or shot_scale < 0.22:
        shot_type = ShotType.GROUP_WIDE
    elif shot_scale < 0.52:
        shot_type = ShotType.MEDIUM
    else:
        shot_type = ShotType.CLOSE_UP
    return FrameVisualDescriptor(
        quality_score=quality,
        composition_score=composition,
        shot_scale=shot_scale,
        shot_type=shot_type,
        face_count=face_count,
        structure_embedding=structure_embedding,
        motion=_motion(gray, following_frame),
        primary_face_bbox=primary_face_bbox,
        color_embedding=_color_embedding(working, gray),
        face_detector_backend=face_detector_backend,
        primary_face_landmarks_xy=primary_face_landmarks,
        primary_face_head_roll_radians=primary_face_roll,
        primary_face_eye_scale=primary_face_eye_scale,
        primary_face_geometry=primary_face_geometry,
        pose_embedding=(
            subject.pose_embedding
            if subject.pose_embedding
            else structure_embedding
        ),
        pose_confidence=subject.pose_confidence,
        detected_person_count=subject.detected_person_count,
        pose_backend=(
            subject.pose_backend
            if subject.pose_embedding
            else "structure_hog_fallback"
        ),
        silhouette_embedding=subject.silhouette_embedding,
        silhouette_bbox_xywh=subject.silhouette_bbox_xywh,
        silhouette_centroid_xy=subject.silhouette_centroid_xy,
        silhouette_area_ratio=subject.silhouette_area_ratio,
        silhouette_confidence=subject.silhouette_confidence,
        silhouette_backend=subject.silhouette_backend,
    )


def _face_match_candidate_failure(
    descriptors: tuple[
        FrameVisualDescriptor,
        FrameVisualDescriptor,
        FrameVisualDescriptor,
    ],
    *,
    has_internal_camera_cut: bool,
) -> str | None:
    """Gate face-driven crops without claiming identity recognition."""

    if has_internal_camera_cut:
        return "internal_camera_cut"
    boxes = tuple(item.primary_face_bbox for item in descriptors)
    if any(box is None for box in boxes):
        if any(item.face_count == 0 for item in descriptors):
            return "face_not_detected"
        return "ambiguous_multiple_faces"
    safe_boxes = tuple(box for box in boxes if box is not None)
    if any(box[3] < 0.035 for box in safe_boxes):
        return "face_too_small"
    centers = tuple(
        (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
        for box in safe_boxes
    )
    maximum_center_span = max(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for left in centers
        for right in centers
    )
    if maximum_center_span > 0.28:
        return "unstable_face_position"
    heights = tuple(box[3] for box in safe_boxes)
    if max(heights) / min(heights) > 1.8:
        return "unstable_face_scale"
    return None


@dataclass(frozen=True, slots=True)
class _CandidateFaceMetadata:
    boxes: tuple[
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
    ]
    failure: str | None
    backends: tuple[str, str, str]
    landmarks: tuple[
        tuple[tuple[float, float], ...] | None,
        tuple[tuple[float, float], ...] | None,
        tuple[tuple[float, float], ...] | None,
    ]
    rolls: tuple[float | None, float | None, float | None]
    eye_scales: tuple[float | None, float | None, float | None]


@dataclass(frozen=True, slots=True)
class _DissolveEvidence:
    """Transient cut/mid/end evidence for one selected source handle.

    These vectors never enter timeline scoring and the appearance/body
    embeddings are intentionally omitted from the serialized analysis.  An
    empty field is a safety failure, never permission to blend.
    """

    face_geometry: tuple[tuple[float, ...], ...] = ()
    face_boxes: tuple[tuple[float, float, float, float], ...] = ()
    face_identity_embeddings: tuple[tuple[float, ...], ...] = ()
    pose_embeddings: tuple[tuple[float, ...], ...] = ()
    pose_backends: tuple[str, ...] = ()
    pose_confidences: tuple[float, ...] = ()
    silhouette_embeddings: tuple[tuple[float, ...], ...] = ()
    silhouette_backends: tuple[str, ...] = ()
    silhouette_confidences: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class _CandidateSubjectMetadata:
    pose_backends: tuple[str, str, str]
    pose_confidences: tuple[float, float, float]
    person_counts: tuple[int, int, int]
    silhouette_boxes: tuple[
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
    ]
    silhouette_centroids: tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
    ]
    silhouette_backends: tuple[str, str, str]
    silhouette_confidences: tuple[float, float, float]


_BOUNDARY_WINDOW_FRAMES = 3
_BOUNDARY_FRAME_STEP_SECONDS = 1.0 / 29.97
_DISSOLVE_HANDLE_FRAMES = 11
_DISSOLVE_HANDLE_SECONDS = _DISSOLVE_HANDLE_FRAMES / 29.97


class _VideoSampler:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ValueError(f"cannot open video for visual scoring: {path}")
        # A full song with nine sources can touch thousands of frames. Keep a
        # small per-source LRU: adjacent segments reuse boundary frames without
        # retaining the complete decoded performance in RAM.
        self.cache: OrderedDict[int, np.ndarray | None] = OrderedDict()
        self.descriptor_cache: OrderedDict[
            int, FrameVisualDescriptor | None
        ] = OrderedDict()
        self.transition_descriptor_cache: OrderedDict[
            int, FrameVisualDescriptor | None
        ] = OrderedDict()
        self.sface_cache: OrderedDict[int, tuple[float, ...]] = OrderedDict()

    def frame_at(self, seconds: float) -> np.ndarray | None:
        key = int(round(seconds * 1000.0))
        if key not in self.cache:
            self.capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
            success, frame = self.capture.read()
            self.cache[key] = _working_frame(frame) if success else None
            while len(self.cache) > 32:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]

    def descriptor_at(self, seconds: float) -> FrameVisualDescriptor | None:
        key = int(round(seconds * 1000.0))
        if key not in self.descriptor_cache:
            frame = self.frame_at(seconds)
            self.descriptor_cache[key] = (
                describe_frame(frame, include_subject_geometry=False)
                if frame is not None
                else None
            )
            while len(self.descriptor_cache) > 64:
                self.descriptor_cache.popitem(last=False)
        else:
            self.descriptor_cache.move_to_end(key)
        return self.descriptor_cache[key]

    def transition_descriptor_at(
        self, seconds: float
    ) -> FrameVisualDescriptor | None:
        """Describe one selected handle frame with full body evidence."""

        key = int(round(seconds * 1000.0))
        if key not in self.transition_descriptor_cache:
            frame = self.frame_at(seconds)
            self.transition_descriptor_cache[key] = (
                describe_frame(frame, include_subject_geometry=True)
                if frame is not None
                else None
            )
            while len(self.transition_descriptor_cache) > 16:
                self.transition_descriptor_cache.popitem(last=False)
        else:
            self.transition_descriptor_cache.move_to_end(key)
        return self.transition_descriptor_cache[key]

    def sface_embedding_at(
        self,
        seconds: float,
        descriptor: FrameVisualDescriptor,
    ) -> tuple[float, ...]:
        """Embed an already-unambiguous YuNet face in one selected frame."""

        key = int(round(seconds * 1000.0))
        if key not in self.sface_cache:
            frame = self.frame_at(seconds)
            bbox = descriptor.primary_face_bbox
            landmarks = descriptor.primary_face_landmarks_xy
            embedding: tuple[float, ...] = ()
            if (
                frame is not None
                and descriptor.face_detector_backend == "yunet"
                and bbox is not None
                and landmarks is not None
                and len(landmarks) == 5
            ):
                embedding = extract_sface_embedding(
                    frame,
                    bbox_xywh=bbox,
                    landmarks_xy=landmarks,
                )
            self.sface_cache[key] = embedding
            while len(self.sface_cache) > 16:
                self.sface_cache.popitem(last=False)
        else:
            self.sface_cache.move_to_end(key)
        return self.sface_cache[key]

    def close(self) -> None:
        self.capture.release()


def _face_geometry_boundary_window(
    sampler: _VideoSampler,
    *,
    boundary_seconds: float,
    incoming: bool,
    scene_cut_times: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """Sample three synchronized frames around one possible hard cut.

    Outgoing samples end one frame before the cut; incoming samples end at the
    cut. Paired entries therefore follow the original consecutive-frame eye
    comparison idea (outgoing ``i`` versus incoming ``i+1``).
    """

    step = _BOUNDARY_FRAME_STEP_SECONDS
    last_offset = 0 if incoming else -1
    offsets = tuple(
        last_offset - (_BOUNDARY_WINDOW_FRAMES - 1) + index
        for index in range(_BOUNDARY_WINDOW_FRAMES)
    )
    times = tuple(boundary_seconds + offset * step for offset in offsets)
    if times[0] < 0.0:
        return ()
    if any(times[0] - 1e-6 <= float(cut) <= times[-1] + 1e-6 for cut in scene_cut_times):
        return ()
    descriptors = tuple(sampler.descriptor_at(time_value) for time_value in times)
    if any(
        descriptor is None or descriptor.primary_face_geometry is None
        for descriptor in descriptors
    ):
        return ()
    geometries = tuple(
        descriptor.primary_face_geometry
        for descriptor in descriptors
        if descriptor is not None and descriptor.primary_face_geometry is not None
    )
    # Guard against a detector switching between people inside the tiny window.
    eye_centers = tuple(
        (
            (geometry[0] + geometry[2]) / 2.0,
            (geometry[1] + geometry[3]) / 2.0,
        )
        for geometry in geometries
    )
    if max(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for left in eye_centers
        for right in eye_centers
    ) > 0.16:
        return ()
    scales = tuple(geometry[4] for geometry in geometries)
    if max(scales) / min(scales) > 1.5:
        return ()
    rolls = tuple(geometry[5] for geometry in geometries)
    if max(rolls) - min(rolls) > math.radians(28.0):
        return ()
    return geometries


def _dissolve_evidence_window(
    sampler: _VideoSampler,
    *,
    boundary_seconds: float,
    scene_cut_times: Sequence[float],
) -> _DissolveEvidence:
    """Sample actual cut/mid/end frames from one selected dissolve handle."""

    times = (
        boundary_seconds,
        boundary_seconds + _DISSOLVE_HANDLE_SECONDS / 2.0,
        boundary_seconds + _DISSOLVE_HANDLE_SECONDS,
    )
    half_frame = _BOUNDARY_FRAME_STEP_SECONDS / 2.0
    if any(
        times[0] - half_frame <= float(cut) <= times[-1] + half_frame
        for cut in scene_cut_times
    ):
        return _DissolveEvidence()
    descriptors = tuple(
        sampler.transition_descriptor_at(time_value) for time_value in times
    )
    if any(
        descriptor is None
        or descriptor.face_detector_backend != "yunet"
        or descriptor.primary_face_geometry is None
        or descriptor.primary_face_bbox is None
        or descriptor.primary_face_landmarks_xy is None
        or len(descriptor.primary_face_landmarks_xy) != 5
        for descriptor in descriptors
    ):
        return _DissolveEvidence()
    safe_descriptors = tuple(
        descriptor for descriptor in descriptors if descriptor is not None
    )
    geometries = tuple(
        descriptor.primary_face_geometry
        for descriptor in safe_descriptors
        if descriptor.primary_face_geometry is not None
    )
    boxes = tuple(
        descriptor.primary_face_bbox
        for descriptor in safe_descriptors
        if descriptor.primary_face_bbox is not None
    )
    centers = tuple(
        (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0) for box in boxes
    )
    if max(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for left in centers
        for right in centers
    ) > 0.16:
        return _DissolveEvidence()
    scales = tuple(geometry[4] for geometry in geometries)
    if min(scales) <= 0.0 or max(scales) / min(scales) > 1.5:
        return _DissolveEvidence()
    rolls = tuple(geometry[5] for geometry in geometries)
    if max(
        abs((left - right + math.pi) % (2.0 * math.pi) - math.pi)
        for left in rolls
        for right in rolls
    ) > math.radians(28.0):
        return _DissolveEvidence()

    identity_embeddings = tuple(
        sampler.sface_embedding_at(time_value, descriptor)
        for time_value, descriptor in zip(times, safe_descriptors, strict=True)
    )
    return _DissolveEvidence(
        face_geometry=geometries,
        face_boxes=boxes,
        face_identity_embeddings=identity_embeddings,
        pose_embeddings=tuple(
            tuple(descriptor.pose_embedding) for descriptor in safe_descriptors
        ),
        pose_backends=tuple(
            descriptor.pose_backend for descriptor in safe_descriptors
        ),
        pose_confidences=tuple(
            float(descriptor.pose_confidence) for descriptor in safe_descriptors
        ),
        silhouette_embeddings=tuple(
            tuple(descriptor.silhouette_embedding)
            for descriptor in safe_descriptors
        ),
        silhouette_backends=tuple(
            descriptor.silhouette_backend for descriptor in safe_descriptors
        ),
        silhouette_confidences=tuple(
            float(descriptor.silhouette_confidence)
            for descriptor in safe_descriptors
        ),
    )


def _embedding_rms_distance(
    left: tuple[float, ...], right: tuple[float, ...]
) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    squared = sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))
    return min(1.0, math.sqrt(squared / len(left)))


def _candidate_can_enter_dissolve_postpass(
    metadata: _CandidateFaceMetadata | None,
) -> bool:
    """Cheap prefilter before opening sources for selected-boundary evidence."""

    return bool(
        metadata is not None
        and metadata.failure is None
        and all(backend == "yunet" for backend in metadata.backends)
        and all(
            landmarks is not None and len(landmarks) == 5
            for landmarks in metadata.landmarks
        )
    )


def _selected_dissolve_evidence(
    *,
    sources: Sequence[Path],
    optimized: TimelineResult,
    audio_offsets: Sequence[float],
    scene_cut_times: Sequence[Sequence[float]],
    face_metadata_by_candidate: dict[str, _CandidateFaceMetadata],
) -> tuple[dict[str, _DissolveEvidence], dict[str, _DissolveEvidence]]:
    """Run expensive safety models only on DP-selected switch boundaries.

    The first result is entry evidence keyed by the incoming candidate id; the
    second is exit evidence keyed by the outgoing candidate id.  Candidate
    scoring stays independent of SFace and future body inference.
    """

    entry: dict[str, _DissolveEvidence] = {}
    exit_: dict[str, _DissolveEvidence] = {}
    boundaries = tuple(zip(optimized.candidates, optimized.candidates[1:]))
    eligible_boundaries = tuple(
        (outgoing, incoming)
        for outgoing, incoming in boundaries
        if outgoing.source_id != incoming.source_id
        and incoming.duration_seconds + 1e-9 >= _DISSOLVE_HANDLE_SECONDS
        and _candidate_can_enter_dissolve_postpass(
            face_metadata_by_candidate.get(outgoing.candidate_id)
        )
        and _candidate_can_enter_dissolve_postpass(
            face_metadata_by_candidate.get(incoming.candidate_id)
        )
    )
    if not eligible_boundaries:
        return (entry, exit_)

    samplers = [_VideoSampler(Path(path)) for path in sources]
    try:
        for outgoing, incoming in eligible_boundaries:
            outgoing_source = int(outgoing.source_id)
            incoming_source = int(incoming.source_id)
            exit_[outgoing.candidate_id] = _dissolve_evidence_window(
                samplers[outgoing_source],
                boundary_seconds=(
                    outgoing.end_seconds + float(audio_offsets[outgoing_source])
                ),
                scene_cut_times=scene_cut_times[outgoing_source],
            )
            entry[incoming.candidate_id] = _dissolve_evidence_window(
                samplers[incoming_source],
                boundary_seconds=(
                    incoming.start_seconds + float(audio_offsets[incoming_source])
                ),
                scene_cut_times=scene_cut_times[incoming_source],
            )
    finally:
        for sampler in samplers:
            sampler.close()
    return (entry, exit_)


def build_visual_timeline(
    *,
    sources: Sequence[Path],
    canonical_spans: Sequence[tuple[float, float]],
    audio_offsets: Sequence[float],
    sync_scores: Sequence[float],
    scene_cut_times: Sequence[Sequence[float]],
    preferred_source_indices: Sequence[int] | None = None,
    diversity_config: VisualDiversityConfig | None = None,
) -> VisualTimelinePlan:
    """Score every synchronized source option and globally select the edit."""

    source_count = len(sources)
    if source_count < 2:
        raise ValueError("visual timeline needs at least two sources")
    if not (
        len(audio_offsets)
        == len(sync_scores)
        == len(scene_cut_times)
        == source_count
    ):
        raise ValueError("per-source visual inputs must have matching lengths")
    if not canonical_spans:
        raise ValueError("canonical_spans cannot be empty")
    preferred_indices = (
        tuple(preferred_source_indices)
        if preferred_source_indices is not None
        else None
    )
    preferred_source_ids: tuple[str | None, ...] | None = None
    if preferred_indices is not None:
        if len(preferred_indices) != len(canonical_spans):
            raise ValueError("preferred source assignments must match canonical spans")
        if any(
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
            or source_index >= source_count
            for source_index in preferred_indices
        ):
            raise ValueError("preferred source assignment contains an invalid index")
        # Coverage is meaningful only when there are enough edit spans to show
        # every source.  The metadata draft provides a full first-use cycle.
        if len(canonical_spans) >= source_count:
            if set(preferred_indices) != set(range(source_count)):
                raise ValueError(
                    "preferred source assignments must cover every source before reuse"
                )
            preferred_source_ids = tuple(str(index) for index in preferred_indices)
    diversity = diversity_config or VisualDiversityConfig()
    samplers = [_VideoSampler(Path(path)) for path in sources]
    candidates_by_segment: list[list[TimelineCandidate]] = []
    face_metadata_by_candidate: dict[str, _CandidateFaceMetadata] = {}
    subject_metadata_by_candidate: dict[str, _CandidateSubjectMetadata] = {}
    try:
        for segment_index, (master_start, master_end) in enumerate(canonical_spans):
            if master_end <= master_start:
                raise ValueError("canonical spans must have positive duration")
            duration = master_end - master_start
            candidates: list[TimelineCandidate] = []
            for source_index, sampler in enumerate(samplers):
                offset = float(audio_offsets[source_index])
                source_start = master_start + offset
                source_end = master_end + offset
                inset = min(0.05, duration * 0.12)
                entry_time = source_start + inset
                exit_time = source_end - inset
                middle_time = (source_start + source_end) / 2.0
                entry_frame = sampler.frame_at(entry_time)
                middle_frame = sampler.frame_at(middle_time)
                exit_frame = sampler.frame_at(exit_time)
                if entry_frame is None or middle_frame is None or exit_frame is None:
                    continue
                following = sampler.frame_at(min(source_end - 0.001, entry_time + 0.08))
                entry = describe_frame(entry_frame, following_frame=following)
                middle = describe_frame(middle_frame)
                exit_descriptor = describe_frame(exit_frame)
                cuts = tuple(float(value) for value in scene_cut_times[source_index])
                tolerance = min(0.14, max(0.06, duration * 0.12))
                near_boundary = any(
                    abs(cut - source_start) <= tolerance
                    or abs(cut - source_end) <= tolerance
                    for cut in cuts
                )
                boundary_score = 0.92 if near_boundary else 0.52
                shot_number = bisect_right(cuts, middle_time)
                has_internal_camera_cut = any(
                    source_start + tolerance < cut < source_end - tolerance
                    for cut in cuts
                )
                candidate_id = f"{source_index}:{segment_index}"
                face_samples = (
                    entry.primary_face_bbox,
                    middle.primary_face_bbox,
                    exit_descriptor.primary_face_bbox,
                )
                face_metadata_by_candidate[candidate_id] = _CandidateFaceMetadata(
                    boxes=face_samples,
                    failure=_face_match_candidate_failure(
                        (entry, middle, exit_descriptor),
                        has_internal_camera_cut=has_internal_camera_cut,
                    ),
                    backends=(
                        entry.face_detector_backend,
                        middle.face_detector_backend,
                        exit_descriptor.face_detector_backend,
                    ),
                    landmarks=(
                        entry.primary_face_landmarks_xy,
                        middle.primary_face_landmarks_xy,
                        exit_descriptor.primary_face_landmarks_xy,
                    ),
                    rolls=(
                        entry.primary_face_head_roll_radians,
                        middle.primary_face_head_roll_radians,
                        exit_descriptor.primary_face_head_roll_radians,
                    ),
                    eye_scales=(
                        entry.primary_face_eye_scale,
                        middle.primary_face_eye_scale,
                        exit_descriptor.primary_face_eye_scale,
                    ),
                )
                subject_metadata_by_candidate[candidate_id] = _CandidateSubjectMetadata(
                    pose_backends=(
                        entry.pose_backend,
                        middle.pose_backend,
                        exit_descriptor.pose_backend,
                    ),
                    pose_confidences=(
                        entry.pose_confidence,
                        middle.pose_confidence,
                        exit_descriptor.pose_confidence,
                    ),
                    person_counts=(
                        entry.detected_person_count,
                        middle.detected_person_count,
                        exit_descriptor.detected_person_count,
                    ),
                    silhouette_boxes=(
                        entry.silhouette_bbox_xywh,
                        middle.silhouette_bbox_xywh,
                        exit_descriptor.silhouette_bbox_xywh,
                    ),
                    silhouette_centroids=(
                        entry.silhouette_centroid_xy,
                        middle.silhouette_centroid_xy,
                        exit_descriptor.silhouette_centroid_xy,
                    ),
                    silhouette_backends=(
                        entry.silhouette_backend,
                        middle.silhouette_backend,
                        exit_descriptor.silhouette_backend,
                    ),
                    silhouette_confidences=(
                        entry.silhouette_confidence,
                        middle.silhouette_confidence,
                        exit_descriptor.silhouette_confidence,
                    ),
                )
                if duration >= 4.0 * _BOUNDARY_FRAME_STEP_SECONDS:
                    entry_face_window = _face_geometry_boundary_window(
                        sampler,
                        boundary_seconds=source_start,
                        incoming=True,
                        scene_cut_times=cuts,
                    )
                    exit_face_window = _face_geometry_boundary_window(
                        sampler,
                        boundary_seconds=source_end,
                        incoming=False,
                        scene_cut_times=cuts,
                    )
                else:
                    entry_face_window = ()
                    exit_face_window = ()
                quality = (entry.quality_score + middle.quality_score + exit_descriptor.quality_score) / 3.0
                composition = (
                    entry.composition_score
                    + middle.composition_score
                    + exit_descriptor.composition_score
                ) / 3.0
                candidates.append(
                    TimelineCandidate(
                        segment_index=segment_index,
                        source_id=str(source_index),
                        start_seconds=master_start,
                        end_seconds=master_end,
                        quality_score=quality,
                        sync_score=_unit(sync_scores[source_index]),
                        composition_score=composition,
                        boundary_score=boundary_score,
                        shot_scale=middle.shot_scale,
                        shot_type=middle.shot_type,
                        is_group_wide=middle.shot_type is ShotType.GROUP_WIDE,
                        pose_embedding=(
                            middle.pose_embedding
                            if middle.pose_backend != "structure_hog_fallback"
                            else ()
                        ),
                        entry_pose_embedding=(
                            entry.pose_embedding
                            if entry.pose_backend != "structure_hog_fallback"
                            else ()
                        ),
                        exit_pose_embedding=(
                            exit_descriptor.pose_embedding
                            if exit_descriptor.pose_backend
                            != "structure_hog_fallback"
                            else ()
                        ),
                        structure_embedding=middle.structure_embedding,
                        entry_structure_embedding=entry.structure_embedding,
                        exit_structure_embedding=exit_descriptor.structure_embedding,
                        silhouette_embedding=middle.silhouette_embedding,
                        entry_silhouette_embedding=entry.silhouette_embedding,
                        exit_silhouette_embedding=exit_descriptor.silhouette_embedding,
                        color_embedding=middle.color_embedding,
                        entry_color_embedding=entry.color_embedding,
                        exit_color_embedding=exit_descriptor.color_embedding,
                        entry_face_geometry_window=entry_face_window,
                        exit_face_geometry_window=exit_face_window,
                        motion=entry.motion,
                        candidate_id=candidate_id,
                        shot_id=f"{source_index}:shot-{shot_number}",
                    )
                )
            if not candidates:
                raise ValueError(f"no source frame is readable for segment {segment_index}")
            candidates_by_segment.append(candidates)
    finally:
        for sampler in samplers:
            sampler.close()

    optimized: TimelineResult = select_timeline(
        candidates_by_segment,
        config=TimelineConfig(
            quality_weight=1.15,
            sync_weight=1.25,
            composition_weight=0.55,
            boundary_weight=0.62,
            continuity_bonus=0.14,
            cut_penalty=0.10,
            early_cut_penalty=0.18,
            long_hold_penalty=0.22,
            pose_transition_weight=0.68,
            silhouette_transition_weight=0.72,
            motion_transition_weight=0.32,
            scale_transition_weight=0.42,
            color_transition_weight=0.26,
            face_geometry_transition_weight=0.32,
            minimum_hold_segments=1,
            preferred_max_hold_segments=4,
            # Rhythm spans are edit opportunities.  Track source hold separately
            # from shot hold so a broadcaster's internal camera cut cannot reset
            # the cross-edit cadence.  The cap is adaptive: it applies only when
            # another source's local emission + transition score is within the
            # configured sacrifice, so a broken/poor camera is never selected
            # solely to satisfy rotation.
            preferred_max_source_hold_segments=3,
            maximum_same_source_hold_segments=4,
            source_hold_penalty=0.34,
            maximum_switch_score_sacrifice=0.58,
            preferred_source_bonus=diversity.preferred_source_bonus,
            preferred_source_max_score_sacrifice=(
                diversity.maximum_preferred_score_sacrifice
            ),
        ),
        preferred_source_ids=preferred_source_ids,
    )
    dissolve_entry_evidence, dissolve_exit_evidence = (
        _selected_dissolve_evidence(
            sources=sources,
            optimized=optimized,
            audio_offsets=audio_offsets,
            scene_cut_times=scene_cut_times,
            face_metadata_by_candidate=face_metadata_by_candidate,
        )
    )
    choices: list[VisualTimelineChoice] = []
    for optimized_index, choice in enumerate(optimized.candidates):
        source_index = int(choice.source_id)
        offset = float(audio_offsets[source_index])
        face_metadata = face_metadata_by_candidate.get(
            choice.candidate_id,
            _CandidateFaceMetadata(
                boxes=(None, None, None),
                failure="face_metadata_unavailable",
                backends=("none", "none", "none"),
                landmarks=(None, None, None),
                rolls=(None, None, None),
                eye_scales=(None, None, None),
            ),
        )
        entry_evidence = dissolve_entry_evidence.get(
            choice.candidate_id, _DissolveEvidence()
        )
        exit_evidence = dissolve_exit_evidence.get(
            choice.candidate_id, _DissolveEvidence()
        )
        subject_metadata = subject_metadata_by_candidate.get(
            choice.candidate_id,
            _CandidateSubjectMetadata(
                pose_backends=(
                    "structure_hog_fallback",
                    "structure_hog_fallback",
                    "structure_hog_fallback",
                ),
                pose_confidences=(0.0, 0.0, 0.0),
                person_counts=(0, 0, 0),
                silhouette_boxes=(None, None, None),
                silhouette_centroids=(None, None, None),
                silhouette_backends=("none", "none", "none"),
                silhouette_confidences=(0.0, 0.0, 0.0),
            ),
        )
        if optimized_index == 0:
            face_transition_mismatch = None
            pose_transition_basis = None
            pose_transition_mismatch = None
            silhouette_transition_mismatch = None
        else:
            previous = optimized.candidates[optimized_index - 1]
            previous_subject_metadata = subject_metadata_by_candidate.get(
                previous.candidate_id
            )
            mismatch = face_geometry_minimax_mismatch(
                previous.exit_face_geometry_window,
                choice.entry_face_geometry_window,
            )
            face_transition_mismatch = (
                mismatch
                if previous.exit_face_geometry_window
                and choice.entry_face_geometry_window
                else None
            )
            previous_pose = previous.exit_pose_embedding or previous.pose_embedding
            current_pose = choice.entry_pose_embedding or choice.pose_embedding
            actual_pose_pair = (
                previous_subject_metadata is not None
                and previous_subject_metadata.pose_backends[2]
                != "structure_hog_fallback"
                and subject_metadata.pose_backends[0] != "structure_hog_fallback"
                and bool(previous_pose)
                and bool(current_pose)
                and len(previous_pose) == len(current_pose)
            )
            if actual_pose_pair:
                pose_transition_basis = "opencv_zoo_mediapipe_pose"
            else:
                pose_transition_basis = "structure_hog_fallback"
                previous_pose = (
                    previous.exit_structure_embedding
                    or previous.structure_embedding
                )
                current_pose = (
                    choice.entry_structure_embedding or choice.structure_embedding
                )
            pose_transition_mismatch = _embedding_rms_distance(
                previous_pose,
                current_pose,
            )
            silhouette_transition_mismatch = _embedding_rms_distance(
                previous.exit_silhouette_embedding
                or previous.silhouette_embedding,
                choice.entry_silhouette_embedding or choice.silhouette_embedding,
            )
        choices.append(
            VisualTimelineChoice(
                segment_index=choice.segment_index,
                source_index=source_index,
                canonical_master_start=choice.start_seconds,
                canonical_master_end=choice.end_seconds,
                source_local_start=choice.start_seconds + offset,
                source_local_end=choice.end_seconds + offset,
                quality_score=choice.quality_score,
                composition_score=choice.composition_score,
                boundary_score=choice.boundary_score,
                shot_scale=choice.shot_scale,
                shot_type=choice.shot_type,
                shot_id=choice.shot_id,
                face_samples_xywh=face_metadata.boxes,
                face_match_candidate_failure=face_metadata.failure,
                face_detector_backends=face_metadata.backends,
                face_landmark_samples_xy=face_metadata.landmarks,
                face_head_roll_samples_radians=face_metadata.rolls,
                face_eye_scale_samples=face_metadata.eye_scales,
                entry_face_geometry_window=choice.entry_face_geometry_window,
                exit_face_geometry_window=choice.exit_face_geometry_window,
                dissolve_entry_face_geometry_window=(
                    entry_evidence.face_geometry
                ),
                dissolve_exit_face_geometry_window=(
                    exit_evidence.face_geometry
                ),
                dissolve_entry_face_box_window=(
                    entry_evidence.face_boxes
                ),
                dissolve_exit_face_box_window=(
                    exit_evidence.face_boxes
                ),
                dissolve_entry_face_identity_window=(
                    entry_evidence.face_identity_embeddings
                ),
                dissolve_exit_face_identity_window=(
                    exit_evidence.face_identity_embeddings
                ),
                dissolve_entry_pose_embedding_window=(
                    entry_evidence.pose_embeddings
                ),
                dissolve_exit_pose_embedding_window=(
                    exit_evidence.pose_embeddings
                ),
                dissolve_entry_pose_backend_window=(
                    entry_evidence.pose_backends
                ),
                dissolve_exit_pose_backend_window=(
                    exit_evidence.pose_backends
                ),
                dissolve_entry_pose_confidence_window=(
                    entry_evidence.pose_confidences
                ),
                dissolve_exit_pose_confidence_window=(
                    exit_evidence.pose_confidences
                ),
                dissolve_entry_silhouette_embedding_window=(
                    entry_evidence.silhouette_embeddings
                ),
                dissolve_exit_silhouette_embedding_window=(
                    exit_evidence.silhouette_embeddings
                ),
                dissolve_entry_silhouette_backend_window=(
                    entry_evidence.silhouette_backends
                ),
                dissolve_exit_silhouette_backend_window=(
                    exit_evidence.silhouette_backends
                ),
                dissolve_entry_silhouette_confidence_window=(
                    entry_evidence.silhouette_confidences
                ),
                dissolve_exit_silhouette_confidence_window=(
                    exit_evidence.silhouette_confidences
                ),
                face_transition_minimax_mismatch_from_previous=(
                    face_transition_mismatch
                ),
                pose_backends=subject_metadata.pose_backends,
                pose_confidences=subject_metadata.pose_confidences,
                detected_person_counts=subject_metadata.person_counts,
                silhouette_samples_xywh=subject_metadata.silhouette_boxes,
                silhouette_centroid_samples_xy=(
                    subject_metadata.silhouette_centroids
                ),
                silhouette_backends=subject_metadata.silhouette_backends,
                silhouette_confidences=subject_metadata.silhouette_confidences,
                pose_transition_basis_from_previous=pose_transition_basis,
                pose_transition_mismatch_from_previous=(
                    pose_transition_mismatch
                ),
                silhouette_transition_mismatch_from_previous=(
                    silhouette_transition_mismatch
                ),
            )
        )
    return VisualTimelinePlan(
        choices=tuple(choices),
        optimizer_score=optimized.total_score,
    )


__all__ = [
    "FrameVisualDescriptor",
    "VisualDiversityConfig",
    "VisualTimelineChoice",
    "VisualTimelinePlan",
    "YUNET_EXPECTED_SHA256",
    "YUNET_MODEL_NAME",
    "build_visual_timeline",
    "describe_frame",
    "yunet_model_asset_status",
]
