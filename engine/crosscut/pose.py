"""Optional on-device human pose and silhouette descriptors.

The inference pre/post-processing is an original, compact adaptation of the
Apache-2.0 OpenCV Zoo MediaPipe person detector, MediaPipe pose estimator, and
PP-HumanSeg examples.  The exact upstream revisions, asset digests, and license
are recorded next to the bundled models in ``models/PROVENANCE.pose.md``.

All model use is best effort.  A missing, corrupt, or unsupported model returns
an empty descriptor so the editor can deterministically retain its existing
gradient-structure fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

import cv2
import numpy as np


_MODEL_DIRECTORY = Path(__file__).resolve().parent / "models"
PERSON_DETECTOR_MODEL = "person_detection_mediapipe_2023mar.onnx"
POSE_ESTIMATOR_MODEL = "pose_estimation_mediapipe_2023mar.onnx"
HUMAN_SEGMENTER_MODEL = "human_segmentation_pphumanseg_2023mar.onnx"

_EXPECTED_SHA256 = {
    PERSON_DETECTOR_MODEL: (
        "47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f"
    ),
    POSE_ESTIMATOR_MODEL: (
        "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f"
    ),
    HUMAN_SEGMENTER_MODEL: (
        "552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24"
    ),
}

POSE_BACKEND = "opencv_zoo_mediapipe_pose_2023mar"
SILHOUETTE_BACKEND = "opencv_zoo_pphumanseg_2023mar"


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


@dataclass(frozen=True, slots=True)
class ModelAssetStatus:
    name: str
    path: str
    present: bool
    digest_valid: bool
    loadable: bool | None
    expected_sha256: str
    actual_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "present": self.present,
            "digest_valid": self.digest_valid,
            "loadable": self.loadable,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
        }


def model_asset_status(
    *, model_directory: Path | None = None, verify_load: bool = False
) -> tuple[ModelAssetStatus, ...]:
    """Report exact bundled model integrity and optional OpenCV loadability."""

    directory = model_directory or _MODEL_DIRECTORY
    results: list[ModelAssetStatus] = []
    for name, expected_digest in _EXPECTED_SHA256.items():
        path = directory / name
        actual_digest = _sha256(path) if path.is_file() else None
        digest_valid = actual_digest == expected_digest
        loadable: bool | None = None
        if verify_load and digest_valid:
            try:
                cv2.dnn.readNet(str(path))
                loadable = True
            except cv2.error:
                loadable = False
        results.append(
            ModelAssetStatus(
                name=name,
                path=str(path),
                present=path.is_file(),
                digest_valid=digest_valid,
                loadable=loadable,
                expected_sha256=expected_digest,
                actual_sha256=actual_digest,
            )
        )
    return tuple(results)


@dataclass(frozen=True, slots=True)
class SubjectGeometryDescriptor:
    """Pose of one unambiguous lead plus the aggregate on-screen silhouette."""

    pose_embedding: tuple[float, ...] = ()
    pose_confidence: float = 0.0
    detected_person_count: int = 0
    person_layout_embedding: tuple[float, ...] = ()
    formation_people: tuple["FormationPersonDescriptor", ...] = ()
    pose_backend: str = "none"
    silhouette_embedding: tuple[float, ...] = ()
    silhouette_bbox_xywh: tuple[float, float, float, float] | None = None
    silhouette_centroid_xy: tuple[float, float] | None = None
    silhouette_area_ratio: float = 0.0
    silhouette_confidence: float = 0.0
    silhouette_backend: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pose_backend": self.pose_backend,
            "pose_confidence": self.pose_confidence,
            "detected_person_count": self.detected_person_count,
            "person_layout_available": bool(self.person_layout_embedding),
            "formation_person_count": len(self.formation_people),
            "formation_upper_body_count": sum(
                bool(person.upper_body_keypoints_xyc)
                for person in self.formation_people
            ),
            "pose_available": bool(self.pose_embedding),
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
            "silhouette_available": bool(self.silhouette_embedding),
        }


@dataclass(frozen=True, slots=True)
class FormationPersonDescriptor:
    """Non-identifying screen-space geometry for one detected performer.

    Descriptors are ordered left-to-right by the caller.  The optional upper
    body vector contains only shoulders, elbows, wrists, and hips normalized
    around the torso; it deliberately excludes face landmarks and appearance
    features.  This lets the timeline compare choreography without persisting
    identity evidence.
    """

    bbox_xywh: tuple[float, float, float, float]
    centroid_xy: tuple[float, float]
    detection_confidence: float
    upper_body_keypoints_xyc: tuple[tuple[float, float, float], ...] = field(
        default=(), repr=False
    )

    def __post_init__(self) -> None:
        left, top, width, height = self.bbox_xywh
        if any(not math.isfinite(float(value)) for value in self.bbox_xywh):
            raise ValueError("formation bbox values must be finite")
        if (
            left < 0.0
            or top < 0.0
            or width <= 0.0
            or height <= 0.0
            or left + width > 1.000001
            or top + height > 1.000001
        ):
            raise ValueError("formation bbox must fit normalized image bounds")
        if len(self.centroid_xy) != 2 or any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in self.centroid_xy
        ):
            raise ValueError("formation centroid must contain two unit values")
        if (
            not math.isfinite(float(self.detection_confidence))
            or not 0.0 <= float(self.detection_confidence) <= 1.0
        ):
            raise ValueError("formation detection confidence must be normalized")
        if self.upper_body_keypoints_xyc and len(
            self.upper_body_keypoints_xyc
        ) != len(_FORMATION_UPPER_BODY_JOINTS):
            raise ValueError("formation upper-body descriptor has an invalid size")
        for point in self.upper_body_keypoints_xyc:
            if len(point) != 3:
                raise ValueError("formation upper-body points require x/y/confidence")
            x, y, confidence = point
            if (
                not math.isfinite(float(x))
                or not math.isfinite(float(y))
                or not math.isfinite(float(confidence))
                or not -1.0 <= float(x) <= 1.0
                or not -1.0 <= float(y) <= 1.0
                or not 0.0 <= float(confidence) <= 1.0
            ):
                raise ValueError("formation upper-body values are outside bounds")


def _pose_detector_anchors() -> np.ndarray:
    """Generate the exact 2,254 anchors embedded by OpenCV Zoo's wrapper."""

    anchors: list[tuple[float, float]] = []
    for grid_size, copies in ((28, 2), (14, 2), (7, 6)):
        for y in range(grid_size):
            for x in range(grid_size):
                anchor = ((x + 0.5) / grid_size, (y + 0.5) / grid_size)
                anchors.extend(anchor for _ in range(copies))
    return np.asarray(anchors, dtype=np.float64)


class _MediaPipePersonDetector:
    def __init__(
        self,
        model_path: Path,
        *,
        score_threshold: float = 0.44,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
    ) -> None:
        self._net = cv2.dnn.readNet(str(model_path))
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._anchors = _pose_detector_anchors()

    @staticmethod
    def _preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        ratio = min(224.0 / height, 224.0 / width)
        resized_width = max(1, int(width * ratio))
        resized_height = max(1, int(height * ratio))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (resized_width, resized_height)).astype(np.float32)
        rgb = (rgb / 255.0 - 0.5) * 2.0
        pad_width = 224 - resized_width
        pad_height = 224 - resized_height
        left = pad_width // 2
        top = pad_height // 2
        rgb = cv2.copyMakeBorder(
            rgb,
            top,
            pad_height - top,
            left,
            pad_width - left,
            cv2.BORDER_CONSTANT,
            value=(0.0, 0.0, 0.0),
        )
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        return blob, np.asarray((left / ratio, top / ratio), dtype=np.float64)

    def infer(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        blob, pad_bias = self._preprocess(image)
        self._net.setInput(blob)
        outputs = self._net.forward(self._net.getUnconnectedOutLayersNames())
        box_output = next(item for item in outputs if item.shape[-1] == 12)
        score_output = next(item for item in outputs if item.shape[-1] == 1)

        raw_scores = np.clip(score_output[0, :, 0].astype(np.float64), -100, 100)
        scores = 1.0 / (1.0 + np.exp(-raw_scores))
        deltas = box_output[0].astype(np.float64)
        scale = float(max(width, height))
        centers = deltas[:, :2] / 224.0 + self._anchors
        sizes = deltas[:, 2:4] / 224.0
        xy1 = (centers - sizes / 2.0) * scale - pad_bias
        xy2 = (centers + sizes / 2.0) * scale - pad_bias
        xyxy = np.concatenate((xy1, xy2), axis=1)
        xywh = np.concatenate((xy1, np.maximum(0.0, xy2 - xy1)), axis=1)
        kept = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            scores.tolist(),
            self._score_threshold,
            self._nms_threshold,
            top_k=self._top_k,
        )
        if len(kept) == 0:
            return np.empty((0, 13), dtype=np.float64)
        indices = np.asarray(kept, dtype=np.int64).reshape(-1)
        landmarks = deltas[indices, 4:].reshape(-1, 4, 2) / 224.0
        landmarks += self._anchors[indices, np.newaxis, :]
        landmarks = landmarks * scale - pad_bias
        return np.c_[
            xyxy[indices],
            landmarks.reshape(-1, 8),
            scores[indices],
        ]


@dataclass(frozen=True, slots=True)
class _PoseObservation:
    screen_landmarks: np.ndarray
    world_landmarks: np.ndarray
    confidence: float


@dataclass(frozen=True, slots=True)
class PersonDetection:
    """One normalized full-person candidate from the bundled detector."""

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float


class _MediaPipePoseEstimator:
    def __init__(self, model_path: Path, *, confidence_threshold: float = 0.50) -> None:
        self._net = cv2.dnn.readNet(str(model_path))
        self._confidence_threshold = confidence_threshold

    @staticmethod
    def _preprocess(
        image: np.ndarray, person: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray] | None:
        keypoints = person[4:12].reshape(4, 2).astype(np.float64)
        hip_center = keypoints[0].copy()
        full_body_point = keypoints[1].copy()
        radius = float(np.linalg.norm(hip_center - full_body_point))
        if not math.isfinite(radius) or radius < 2.0:
            return None

        full_bbox = np.asarray(
            (np.floor(hip_center - radius), np.ceil(hip_center + radius)),
            dtype=np.int32,
        )
        image_height, image_width = image.shape[:2]
        clipped_bbox = full_bbox.copy()
        clipped_bbox[:, 0] = np.clip(clipped_bbox[:, 0], 0, image_width)
        clipped_bbox[:, 1] = np.clip(clipped_bbox[:, 1], 0, image_height)
        if np.any(clipped_bbox[1] <= clipped_bbox[0]):
            return None

        crop = image[
            clipped_bbox[0, 1] : clipped_bbox[1, 1],
            clipped_bbox[0, 0] : clipped_bbox[1, 0],
        ]
        left, top = clipped_bbox[0] - full_bbox[0]
        right, bottom = full_bbox[1] - clipped_bbox[1]
        crop = cv2.copyMakeBorder(
            crop,
            int(top),
            int(bottom),
            int(left),
            int(right),
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        pad_bias = clipped_bbox[0] - np.asarray((left, top))
        local_hip = hip_center - pad_bias
        local_full = full_body_point - pad_bias
        radians = math.pi / 2.0 - math.atan2(
            -(local_full[1] - local_hip[1]),
            local_full[0] - local_hip[0],
        )
        radians -= 2.0 * math.pi * math.floor((radians + math.pi) / (2.0 * math.pi))
        angle = math.degrees(radians)
        rotation = cv2.getRotationMatrix2D(tuple(local_hip), angle, 1.0)
        rotated = cv2.warpAffine(crop, rotation, (crop.shape[1], crop.shape[0]))
        blob = cv2.resize(rotated, (256, 256), interpolation=cv2.INTER_AREA)
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        roi_bbox = np.asarray(((0, 0), (crop.shape[1], crop.shape[0])), dtype=np.int32)
        return blob[np.newaxis, ...], roi_bbox, angle, rotation, pad_bias

    def infer(self, image: np.ndarray, person: np.ndarray) -> _PoseObservation | None:
        prepared = self._preprocess(image, person)
        if prepared is None:
            return None
        blob, roi_bbox, angle, rotation, pad_bias = prepared
        self._net.setInput(blob)
        outputs = self._net.forward(self._net.getUnconnectedOutLayersNames())
        landmark_blob = next(item for item in outputs if item.shape == (1, 195))
        world_blob = next(item for item in outputs if item.shape == (1, 117))
        confidence_blob = next(item for item in outputs if item.shape == (1, 1))
        confidence = float(confidence_blob[0, 0])
        if confidence < self._confidence_threshold:
            return None

        screen = landmark_blob[0].reshape(39, 5).astype(np.float64)
        world = world_blob[0].reshape(39, 3).astype(np.float64)
        screen[:, 3:] = 1.0 / (1.0 + np.exp(-np.clip(screen[:, 3:], -100, 100)))

        roi_size = (roi_bbox[1] - roi_bbox[0]).astype(np.float64)
        scale = roi_size / 256.0
        screen[:, :2] = (screen[:, :2] - 128.0) * scale
        screen[:, 2] *= float(max(scale))
        coordinate_rotation = cv2.getRotationMatrix2D((0, 0), angle, 1.0)[:, :2]
        rotated_xy = np.dot(screen[:, :2], coordinate_rotation)
        rotated_world_xy = np.dot(world[:, :2], coordinate_rotation)

        rotation_component = np.asarray(
            (
                (rotation[0, 0], rotation[1, 0]),
                (rotation[0, 1], rotation[1, 1]),
            )
        )
        translation = rotation[:, 2]
        inverse_translation = -rotation_component @ translation
        inverse_rotation = np.c_[rotation_component, inverse_translation]
        roi_center = np.append(np.sum(roi_bbox, axis=0) / 2.0, 1.0)
        original_center = np.asarray(
            (np.dot(roi_center, inverse_rotation[0]), np.dot(roi_center, inverse_rotation[1]))
        )
        screen[:, :2] = rotated_xy + original_center + pad_bias
        world[:, :2] = rotated_world_xy
        return _PoseObservation(
            screen_landmarks=screen[:33],
            world_landmarks=world[:33],
            confidence=confidence,
        )


class _PPHumanSegmenter:
    def __init__(self, model_path: Path) -> None:
        self._net = cv2.dnn.readNet(str(model_path))

    def foreground_probability(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (192, 192)).astype(np.float32) / 255.0
        normalized = (resized - 0.5) / 0.5
        self._net.setInput(cv2.dnn.blobFromImage(normalized))
        output = self._net.forward()
        if output.shape != (1, 2, 192, 192):
            raise ValueError(f"unexpected PP-HumanSeg output shape: {output.shape}")
        return output[0, 1].astype(np.float32)


_POSE_JOINTS = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32)
_FORMATION_UPPER_BODY_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24)
_MAXIMUM_FORMATION_PEOPLE = 6


def _pose_embedding(observation: _PoseObservation) -> tuple[tuple[float, ...], float]:
    """Create a translation/scale-normalized 3D skeleton with visibility gates."""

    screen = observation.screen_landmarks
    reliability = np.sqrt(np.clip(screen[:, 3], 0, 1) * np.clip(screen[:, 4], 0, 1))
    hips = (23, 24)
    shoulders = (11, 12)
    if min(reliability[index] for index in (*hips, *shoulders)) < 0.28:
        return (), 0.0
    hip_center = (screen[23, :3] + screen[24, :3]) / 2.0
    shoulder_center = (screen[11, :3] + screen[12, :3]) / 2.0
    torso = float(np.linalg.norm(shoulder_center[:2] - hip_center[:2]))
    shoulder_width = float(np.linalg.norm(screen[11, :2] - screen[12, :2]))
    hip_width = float(np.linalg.norm(screen[23, :2] - screen[24, :2]))
    scale = max(torso * 1.7, shoulder_width, hip_width * 1.35, 1.0)
    visible_count = sum(reliability[index] >= 0.45 for index in _POSE_JOINTS)
    if visible_count < 8:
        return (), 0.0

    values: list[float] = []
    for index in _POSE_JOINTS:
        joint_confidence = _unit(reliability[index])
        relative = np.clip((screen[index, :3] - hip_center) / scale, -2.5, 2.5) / 2.5
        values.extend(float(value * joint_confidence) for value in relative)
        values.append(joint_confidence)
    reliability_score = float(np.mean([reliability[index] for index in _POSE_JOINTS]))
    return tuple(values), _unit(observation.confidence * reliability_score)


def _upper_body_shape(
    observation: _PoseObservation,
) -> tuple[tuple[float, float, float], ...]:
    """Return a torso-normalized, non-facial upper-body skeleton."""

    screen = observation.screen_landmarks
    reliability = np.sqrt(np.clip(screen[:, 3], 0, 1) * np.clip(screen[:, 4], 0, 1))
    shoulders = (11, 12)
    hips = (23, 24)
    if min(reliability[index] for index in (*shoulders, *hips)) < 0.28:
        return ()
    shoulder_center = (screen[11, :2] + screen[12, :2]) / 2.0
    hip_center = (screen[23, :2] + screen[24, :2]) / 2.0
    torso_center = (shoulder_center + hip_center) / 2.0
    torso_height = float(np.linalg.norm(shoulder_center - hip_center))
    shoulder_width = float(np.linalg.norm(screen[11, :2] - screen[12, :2]))
    scale = max(torso_height * 1.65, shoulder_width, 1.0)
    visible_count = sum(
        reliability[index] >= 0.35 for index in _FORMATION_UPPER_BODY_JOINTS
    )
    if visible_count < 5:
        return ()

    points: list[tuple[float, float, float]] = []
    for index in _FORMATION_UPPER_BODY_JOINTS:
        relative = np.clip((screen[index, :2] - torso_center) / scale, -1.0, 1.0)
        points.append(
            (
                float(relative[0]),
                float(relative[1]),
                _unit(reliability[index] * observation.confidence),
            )
        )
    return tuple(points)


def _silhouette_descriptor(
    foreground_probability: np.ndarray,
) -> tuple[
    tuple[float, ...],
    tuple[float, float, float, float] | None,
    tuple[float, float] | None,
    float,
    float,
]:
    """Encode aggregate on-screen human layout, retaining multiple people."""

    probability = np.asarray(foreground_probability, dtype=np.float32)
    if probability.ndim != 2 or probability.size == 0:
        raise ValueError("foreground probability must be a non-empty 2D array")
    binary = (probability >= 0.52).astype(np.uint8)
    component_count, labels, statistics, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    minimum_component_area = max(6, int(round(binary.size * 0.0008)))
    kept = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, component_count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if area < minimum_component_area:
            continue
        component = labels == label
        if float(probability[component].mean()) < 0.60:
            continue
        kept[component] = 1
    ys, xs = np.nonzero(kept)
    if len(xs) == 0:
        return (), None, None, 0.0, 0.0

    height, width = kept.shape
    area_ratio = float(kept.mean())
    foreground_confidence = float(probability[kept.astype(bool)].mean())
    if area_ratio < 0.0025 or foreground_confidence < 0.62:
        return (), None, None, area_ratio, foreground_confidence

    left = float(xs.min()) / width
    top = float(ys.min()) / height
    right = float(xs.max() + 1) / width
    bottom = float(ys.max() + 1) / height
    weights = probability * kept
    weight_sum = float(weights.sum())
    grid_x = np.arange(width, dtype=np.float32) + 0.5
    grid_y = np.arange(height, dtype=np.float32) + 0.5
    center_x = float((weights.sum(axis=0) * grid_x).sum() / weight_sum / width)
    center_y = float((weights.sum(axis=1) * grid_y).sum() / weight_sum / height)

    # A coarse probability map captures group formation and full-body outline.
    # Repeating geometry gives centroid/scale enough influence under RMS distance
    # without coupling timeline.py to this model-specific representation.
    masked_probability = probability * kept
    coarse = cv2.resize(masked_probability, (12, 8), interpolation=cv2.INTER_AREA)
    coarse = np.sqrt(np.clip(coarse, 0.0, 1.0))
    geometry = (
        _unit(center_x),
        _unit(center_y),
        _unit(right - left),
        _unit(bottom - top),
        _unit(math.sqrt(area_ratio)),
    )
    embedding = tuple(float(value) for value in coarse.reshape(-1)) + geometry * 5
    return (
        embedding,
        (left, top, right - left, bottom - top),
        (center_x, center_y),
        area_ratio,
        _unit(foreground_confidence),
    )


def _person_layout_embedding(
    persons: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[float, ...]:
    """Encode the left-to-right on-screen formation of all detected dancers."""

    boxes: list[tuple[float, float, float, float, float]] = []
    for person in persons:
        left = _unit(float(person[0]) / frame_width)
        top = _unit(float(person[1]) / frame_height)
        right = _unit(float(person[2]) / frame_width)
        bottom = _unit(float(person[3]) / frame_height)
        confidence = _unit(float(person[-1]))
        if right - left < 0.03 or bottom - top < 0.08 or confidence < 0.44:
            continue
        boxes.append((left, top, right, bottom, confidence))
    if not boxes:
        return ()
    boxes.sort(key=lambda item: ((item[0] + item[2]) / 2.0, item[1], -item[4]))

    occupancy = np.zeros((72, 128), dtype=np.float32)
    for left, top, right, bottom, confidence in boxes:
        x1 = max(0, min(127, int(math.floor(left * 128))))
        y1 = max(0, min(71, int(math.floor(top * 72))))
        x2 = max(x1 + 1, min(128, int(math.ceil(right * 128))))
        y2 = max(y1 + 1, min(72, int(math.ceil(bottom * 72))))
        occupancy[y1:y2, x1:x2] = np.maximum(
            occupancy[y1:y2, x1:x2], confidence
        )
    coarse = cv2.resize(occupancy, (12, 8), interpolation=cv2.INTER_AREA)

    ordered_geometry: list[float] = []
    for left, top, right, bottom, confidence in boxes[:6]:
        ordered_geometry.extend(
            (
                (left + right) / 2.0,
                (top + bottom) / 2.0,
                right - left,
                bottom - top,
                confidence,
            )
        )
    ordered_geometry.extend((0.0,) * (6 * 5 - len(ordered_geometry)))
    count_value = min(1.0, len(boxes) / 6.0)
    return (
        tuple(float(value) for value in coarse.reshape(-1))
        + tuple(ordered_geometry)
        + (count_value,) * 8
    )


def _formation_people_descriptor(
    persons: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    pose_estimator: _MediaPipePoseEstimator | None = None,
    image: np.ndarray | None = None,
) -> tuple[
    tuple[FormationPersonDescriptor, ...],
    dict[int, _PoseObservation],
]:
    """Describe at most six strong detections in stable left-to-right order."""

    valid: list[
        tuple[
            int,
            np.ndarray,
            tuple[float, float, float, float],
            tuple[float, float],
            float,
            float,
        ]
    ] = []
    for index, person in enumerate(persons):
        left = _unit(float(person[0]) / frame_width)
        top = _unit(float(person[1]) / frame_height)
        right = _unit(float(person[2]) / frame_width)
        bottom = _unit(float(person[3]) / frame_height)
        confidence = _unit(float(person[-1]))
        width = right - left
        height = bottom - top
        if width < 0.03 or height < 0.08 or confidence < 0.44:
            continue
        bbox = (left, top, width, height)
        centroid = (left + width / 2.0, top + height / 2.0)
        # Prefer real foreground people over a small detector duplicate when
        # more than the six IVE members survive NMS.
        rank = confidence * math.sqrt(width * height)
        valid.append((index, person, bbox, centroid, confidence, rank))

    selected = sorted(
        valid,
        key=lambda item: (item[5], item[4], item[2][2] * item[2][3]),
        reverse=True,
    )[:_MAXIMUM_FORMATION_PEOPLE]
    selected.sort(key=lambda item: (item[3][0], item[3][1], -item[4]))

    observations: dict[int, _PoseObservation] = {}
    descriptors: list[FormationPersonDescriptor] = []
    for index, person, bbox, centroid, confidence, _rank in selected:
        observation = None
        if pose_estimator is not None and image is not None:
            try:
                observation = pose_estimator.infer(image, person)
            except (cv2.error, ValueError, FloatingPointError):
                observation = None
        if observation is not None:
            observations[index] = observation
        descriptors.append(
            FormationPersonDescriptor(
                bbox_xywh=bbox,
                centroid_xy=centroid,
                detection_confidence=confidence,
                upper_body_keypoints_xyc=(
                    _upper_body_shape(observation)
                    if observation is not None
                    else ()
                ),
            )
        )
    return tuple(descriptors), observations


def _dominant_person(persons: np.ndarray) -> np.ndarray | None:
    if len(persons) == 0:
        return None
    ranked: list[tuple[float, float, np.ndarray]] = []
    for person in persons:
        landmarks = person[4:12].reshape(4, 2)
        radius = float(np.linalg.norm(landmarks[0] - landmarks[1]))
        if not math.isfinite(radius) or radius < 2.0:
            continue
        score = _unit(person[-1])
        ranked.append((radius, score, person))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0] * (0.65 + 0.35 * item[1]), item[1]), reverse=True)
    if len(ranked) > 1:
        largest_radius, largest_score, _person = ranked[0]
        second_radius, second_score, _second_person = ranked[1]
        # When two performers have comparable screen size, choosing one would
        # make the pose identity jump between sources. Aggregate silhouette is
        # the safe descriptor for that group shot.
        if second_radius >= largest_radius * 0.78 and second_score >= largest_score * 0.78:
            return None
    return ranked[0][2]


class SubjectGeometryAnalyzer:
    """Lazy, independent pose and silhouette inference with safe degradation."""

    def __init__(self, model_directory: Path | None = None) -> None:
        self._directory = model_directory or _MODEL_DIRECTORY
        self._pose_detector: _MediaPipePersonDetector | None = None
        self._pose_estimator: _MediaPipePoseEstimator | None = None
        self._segmenter: _PPHumanSegmenter | None = None
        self._pose_load_attempted = False
        self._segmenter_load_attempted = False
        self._lock = Lock()

    def _valid_model(self, name: str) -> bool:
        path = self._directory / name
        return path.is_file() and _sha256(path) == _EXPECTED_SHA256[name]

    def _load_pose(self) -> None:
        if self._pose_load_attempted:
            return
        self._pose_load_attempted = True
        if not self._valid_model(PERSON_DETECTOR_MODEL) or not self._valid_model(
            POSE_ESTIMATOR_MODEL
        ):
            return
        try:
            self._pose_detector = _MediaPipePersonDetector(
                self._directory / PERSON_DETECTOR_MODEL
            )
            self._pose_estimator = _MediaPipePoseEstimator(
                self._directory / POSE_ESTIMATOR_MODEL
            )
        except cv2.error:
            self._pose_detector = None
            self._pose_estimator = None

    def _load_segmenter(self) -> None:
        if self._segmenter_load_attempted:
            return
        self._segmenter_load_attempted = True
        if not self._valid_model(HUMAN_SEGMENTER_MODEL):
            return
        try:
            self._segmenter = _PPHumanSegmenter(
                self._directory / HUMAN_SEGMENTER_MODEL
            )
        except cv2.error:
            self._segmenter = None

    def describe(
        self,
        image: np.ndarray,
        *,
        include_formation_pose: bool = False,
    ) -> SubjectGeometryDescriptor:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("subject geometry expects a non-empty BGR image")
        with self._lock:
            self._load_pose()
            self._load_segmenter()
            silhouette_embedding: tuple[float, ...] = ()
            silhouette_bbox = None
            silhouette_centroid = None
            silhouette_area = 0.0
            silhouette_confidence = 0.0
            silhouette_backend = "none"
            if self._segmenter is not None:
                try:
                    (
                        silhouette_embedding,
                        silhouette_bbox,
                        silhouette_centroid,
                        silhouette_area,
                        silhouette_confidence,
                    ) = _silhouette_descriptor(
                        self._segmenter.foreground_probability(image)
                    )
                    silhouette_backend = SILHOUETTE_BACKEND
                except (cv2.error, ValueError, FloatingPointError):
                    pass

            pose_embedding: tuple[float, ...] = ()
            pose_confidence = 0.0
            person_count = 0
            person_layout_embedding: tuple[float, ...] = ()
            formation_people: tuple[FormationPersonDescriptor, ...] = ()
            pose_backend = "none"
            if self._pose_detector is not None and self._pose_estimator is not None:
                try:
                    persons = self._pose_detector.infer(image)
                    person_count = len(persons)
                    person_layout_embedding = _person_layout_embedding(
                        persons,
                        frame_width=image.shape[1],
                        frame_height=image.shape[0],
                    )
                    formation_people, formation_observations = (
                        _formation_people_descriptor(
                            persons,
                            frame_width=image.shape[1],
                            frame_height=image.shape[0],
                            pose_estimator=(
                                self._pose_estimator
                                if include_formation_pose
                                else None
                            ),
                            image=image if include_formation_pose else None,
                        )
                    )
                    dominant = _dominant_person(persons)
                    if dominant is not None:
                        dominant_index = next(
                            (
                                index
                                for index, person in enumerate(persons)
                                if np.array_equal(person, dominant)
                            ),
                            -1,
                        )
                        observation = formation_observations.get(dominant_index)
                        if observation is None:
                            observation = self._pose_estimator.infer(image, dominant)
                        if observation is not None:
                            pose_embedding, pose_confidence = _pose_embedding(observation)
                            if pose_embedding:
                                detector_confidence = _unit(dominant[-1])
                                pose_confidence = _unit(
                                    math.sqrt(detector_confidence * pose_confidence)
                                )
                                pose_backend = POSE_BACKEND
                except (cv2.error, ValueError, FloatingPointError):
                    pass

            return SubjectGeometryDescriptor(
                pose_embedding=pose_embedding,
                pose_confidence=pose_confidence,
                detected_person_count=person_count,
                person_layout_embedding=person_layout_embedding,
                formation_people=formation_people,
                pose_backend=pose_backend,
                silhouette_embedding=silhouette_embedding,
                silhouette_bbox_xywh=silhouette_bbox,
                silhouette_centroid_xy=silhouette_centroid,
                silhouette_area_ratio=silhouette_area,
                silhouette_confidence=silhouette_confidence,
                silhouette_backend=silhouette_backend,
            )


_DEFAULT_ANALYZER: SubjectGeometryAnalyzer | None = None


def describe_subject_geometry(
    image: np.ndarray,
    *,
    include_formation_pose: bool = False,
) -> SubjectGeometryDescriptor:
    global _DEFAULT_ANALYZER
    if _DEFAULT_ANALYZER is None:
        _DEFAULT_ANALYZER = SubjectGeometryAnalyzer()
    return _DEFAULT_ANALYZER.describe(
        image,
        include_formation_pose=include_formation_pose,
    )


def detect_people(image: np.ndarray) -> tuple[PersonDetection, ...]:
    """Return clipped person boxes without selecting an identity.

    Automatic member anchoring applies its own multi-frame ambiguity and
    stability gates. This low-level function deliberately exposes every valid
    detector result instead of choosing a performer.
    """

    global _DEFAULT_ANALYZER
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("person detection expects a non-empty BGR image")
    if _DEFAULT_ANALYZER is None:
        _DEFAULT_ANALYZER = SubjectGeometryAnalyzer()
    analyzer = _DEFAULT_ANALYZER
    with analyzer._lock:
        analyzer._load_pose()
        if analyzer._pose_detector is None:
            return ()
        try:
            raw = analyzer._pose_detector.infer(image)
        except (cv2.error, ValueError, FloatingPointError):
            return ()
    height, width = image.shape[:2]
    detections: list[PersonDetection] = []
    for person in raw:
        left, top, right, bottom = (
            float(person[0]) / width,
            float(person[1]) / height,
            float(person[2]) / width,
            float(person[3]) / height,
        )
        left = max(0.0, min(1.0, left))
        top = max(0.0, min(1.0, top))
        right = max(0.0, min(1.0, right))
        bottom = max(0.0, min(1.0, bottom))
        confidence = _unit(float(person[-1]))
        if right - left < 0.03 or bottom - top < 0.08 or confidence < 0.44:
            continue
        detections.append(
            PersonDetection(
                bbox_xyxy=(left, top, right, bottom),
                confidence=confidence,
            )
        )
    return tuple(
        sorted(
            detections,
            key=lambda item: (
                -item.confidence,
                item.bbox_xyxy,
            ),
        )
    )


__all__ = [
    "FormationPersonDescriptor",
    "HUMAN_SEGMENTER_MODEL",
    "ModelAssetStatus",
    "PersonDetection",
    "PERSON_DETECTOR_MODEL",
    "POSE_BACKEND",
    "POSE_ESTIMATOR_MODEL",
    "SILHOUETTE_BACKEND",
    "SubjectGeometryAnalyzer",
    "SubjectGeometryDescriptor",
    "detect_people",
    "describe_subject_geometry",
    "model_asset_status",
]
