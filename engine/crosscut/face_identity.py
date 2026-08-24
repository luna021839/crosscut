"""Local SFace appearance embeddings for transition safety.

This module does not identify a named person and must not be described as an
authentication system. It only asks whether two already unambiguous dominant
YuNet face tracks have sufficiently similar SFace appearance embeddings. The
global selector uses that evidence to avoid a conspicuous different-face jump
between two close-ups; the transition post-pass uses a stricter three-frame
window before permitting a short dissolve. Missing model/detection evidence
returns an empty embedding, so a close-up source switch is rejected and a
dissolve is downgraded to a hard cut.

The bundled OpenCV Zoo SFace model and its official example code are Apache
License 2.0. The compact wrapper below independently follows OpenCV's public
``FaceRecognizerSF.alignCrop`` / ``feature`` API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


SFACE_MODEL = "face_recognition_sface_2021dec.onnx"
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
SFACE_BACKEND = "opencv_zoo_sface_2021dec"
SFACE_EMBEDDING_SIZE = 128


@dataclass(frozen=True, slots=True)
class SFaceAssetStatus:
    name: str
    path: str
    expected_sha256: str
    actual_sha256: str | None
    present: bool
    digest_valid: bool
    loadable: bool | None
    embedding_size: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "present": self.present,
            "digest_valid": self.digest_valid,
            "loadable": self.loadable,
            "embedding_size": self.embedding_size,
        }


def _default_model_path() -> Path:
    return Path(__file__).resolve().parent / "models" / SFACE_MODEL


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_normalized_face(
    bbox_xywh: Sequence[float],
    landmarks_xy: Sequence[Sequence[float]],
) -> None:
    if len(bbox_xywh) != 4 or len(landmarks_xy) != 5:
        raise ValueError("SFace needs one bbox and five YuNet landmarks")
    values = tuple(float(value) for value in bbox_xywh) + tuple(
        float(value) for point in landmarks_xy for value in point
    )
    if len(values) != 14 or any(not math.isfinite(value) for value in values):
        raise ValueError("SFace geometry must be finite")
    x, y, width, height = (float(value) for value in bbox_xywh)
    if (
        x < 0.0
        or y < 0.0
        or width <= 0.0
        or height <= 0.0
        or x + width > 1.000001
        or y + height > 1.000001
        or any(
            not 0.0 <= float(coordinate) <= 1.0
            for point in landmarks_xy
            for coordinate in point
        )
    ):
        raise ValueError("SFace geometry must be normalized inside the frame")


class SFaceEmbedder:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = (model_path or _default_model_path()).resolve()
        self._recognizer = cv2.FaceRecognizerSF.create(
            str(self.model_path),
            "",
        )

    def embedding(
        self,
        frame: np.ndarray,
        *,
        bbox_xywh: Sequence[float],
        landmarks_xy: Sequence[Sequence[float]],
    ) -> tuple[float, ...]:
        if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim != 3:
            raise ValueError("SFace frame must be a non-empty BGR image")
        if frame.shape[2] != 3:
            raise ValueError("SFace frame must have three BGR channels")
        _validate_normalized_face(bbox_xywh, landmarks_xy)
        height, width = frame.shape[:2]
        detection = np.empty((1, 15), dtype=np.float32)
        x, y, box_width, box_height = (float(value) for value in bbox_xywh)
        detection[0, :4] = (
            x * width,
            y * height,
            box_width * width,
            box_height * height,
        )
        for index, point in enumerate(landmarks_xy):
            detection[0, 4 + index * 2] = float(point[0]) * width
            detection[0, 5 + index * 2] = float(point[1]) * height
        detection[0, 14] = 1.0
        aligned = self._recognizer.alignCrop(frame, detection)
        feature = np.asarray(self._recognizer.feature(aligned), dtype=np.float64).reshape(-1)
        if feature.size != SFACE_EMBEDDING_SIZE or not np.all(np.isfinite(feature)):
            raise RuntimeError("SFace returned an invalid embedding")
        norm = float(np.linalg.norm(feature))
        if norm <= 1e-12:
            raise RuntimeError("SFace returned an empty embedding")
        return tuple(float(value) for value in feature / norm)


def _smoke_frame() -> tuple[
    np.ndarray,
    tuple[float, float, float, float],
    tuple[tuple[float, float], ...],
]:
    y, x = np.mgrid[0:160, 0:160]
    frame = np.stack(
        (
            (40 + x * 0.7).clip(0, 255),
            (35 + y * 0.8).clip(0, 255),
            (70 + (x + y) * 0.45).clip(0, 255),
        ),
        axis=2,
    ).astype(np.uint8)
    bbox = (0.20, 0.12, 0.60, 0.76)
    landmarks = (
        (0.38, 0.39),
        (0.62, 0.39),
        (0.50, 0.53),
        (0.40, 0.67),
        (0.60, 0.67),
    )
    return (frame, bbox, landmarks)


def sface_model_asset_status(
    *,
    model_path: Path | None = None,
    verify_load: bool = False,
) -> SFaceAssetStatus:
    path = (model_path or _default_model_path()).resolve()
    present = path.is_file()
    try:
        actual = _sha256(path) if present else None
    except OSError:
        actual = None
    valid = actual == SFACE_SHA256
    loadable: bool | None = None
    embedding_size: int | None = None
    if verify_load and valid:
        try:
            frame, bbox, landmarks = _smoke_frame()
            embedding = SFaceEmbedder(path).embedding(
                frame,
                bbox_xywh=bbox,
                landmarks_xy=landmarks,
            )
            embedding_size = len(embedding)
            loadable = embedding_size == SFACE_EMBEDDING_SIZE
        except (AttributeError, cv2.error, OSError, RuntimeError, ValueError):
            loadable = False
    return SFaceAssetStatus(
        name=SFACE_MODEL,
        path=str(path),
        expected_sha256=SFACE_SHA256,
        actual_sha256=actual,
        present=present,
        digest_valid=valid,
        loadable=loadable,
        embedding_size=embedding_size,
    )


_DEFAULT_EMBEDDER: SFaceEmbedder | None = None
_SFACE_DISABLED = False


def extract_sface_embedding(
    frame: np.ndarray,
    *,
    bbox_xywh: Sequence[float],
    landmarks_xy: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Return a normalized local appearance embedding, or empty on any miss."""

    global _DEFAULT_EMBEDDER, _SFACE_DISABLED
    if _SFACE_DISABLED:
        return ()
    if _DEFAULT_EMBEDDER is None:
        status = sface_model_asset_status()
        if not status.digest_valid:
            _SFACE_DISABLED = True
            return ()
        try:
            _DEFAULT_EMBEDDER = SFaceEmbedder()
        except (AttributeError, cv2.error, OSError):
            _SFACE_DISABLED = True
            return ()
    try:
        return _DEFAULT_EMBEDDER.embedding(
            frame,
            bbox_xywh=bbox_xywh,
            landmarks_xy=landmarks_xy,
        )
    except (AttributeError, cv2.error, RuntimeError, ValueError):
        return ()


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        return None
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    value = float(np.dot(left_array, right_array) / (left_norm * right_norm))
    return max(-1.0, min(1.0, value))


__all__ = [
    "SFACE_BACKEND",
    "SFACE_EMBEDDING_SIZE",
    "SFACE_MODEL",
    "SFACE_SHA256",
    "SFaceAssetStatus",
    "SFaceEmbedder",
    "cosine_similarity",
    "extract_sface_embedding",
    "sface_model_asset_status",
]
