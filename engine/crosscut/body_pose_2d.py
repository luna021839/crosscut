"""Independent RTMPose Body7 evidence for exact-frame boundary evaluation.

The model is an optional, explicitly supplied external asset. This module does
not download it, alter legacy MediaPipe descriptors, select a member identity,
or authorize a cut. Its separate fingerprint lets a consumer add evidence at
requested boundaries without invalidating unrelated audio/scene/face caches.

Pre/postprocessing follows OpenMMLab's exported RTMPose pipeline: RGB input,
1.25 box padding, zero-roll top-down affine and SimCC split ratio 2. The generic
Python demo omits BGR-to-RGB; the actual export's ``to_rgb: true`` is retained.
Model/algorithm: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
Apache-2.0; the externally acquired model must retain its accompanying license.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from .pose import FormationPersonDescriptor


RTMPOSE_BODY7_MODEL_SHA256 = "5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c"
_BODY_JOINTS = (5, 6, 7, 8, 9, 10, 11, 12)
_TORSO = (0, 1, 6, 7)


def _valid_box(box: tuple[float, float, float, float]) -> bool:
    return bool(
        len(box) == 4
        and all(not isinstance(v, bool) and isinstance(v, (float, int))
                and math.isfinite(v) for v in box)
        and box[0] >= 0 and box[1] >= 0 and box[2] > 0 and box[3] > 0
        and box[0] + box[2] <= 1.000001 and box[1] + box[3] <= 1.000001
    )


def pose_input_geometry(frame_size_wh, box):
    """The exact padded crop and source-pixel affine used by this adapter.

    The padded support can cross a frame edge. Do not clamp it or use the
    tight detector rectangle as the claimed pose input/arm visibility extent.
    """
    if (not isinstance(frame_size_wh,tuple) or len(frame_size_wh)!=2
            or any(type(v) is not int or v<1 for v in frame_size_wh) or not _valid_box(box)):
        raise ValueError('pose input geometry requires a valid detector ROI and actual pixel raster')
    width, height = frame_size_wh
    x, y, bw, bh = np.asarray(box) * np.asarray((width, height, width, height))
    center = np.asarray((x + bw / 2, y + bh / 2))
    scale = np.asarray((bw, bh)) * 1.25
    scale = np.asarray((max(scale[0], scale[1] * .75),
                        max(scale[1], scale[0] / .75)))
    factor = 192 / scale[0]
    affine = np.asarray(((factor, 0, 96 - factor * center[0]),
                         (0, factor, 128 - factor * center[1])), np.float32)
    crop_xywh = tuple(map(float, ((center[0]-scale[0]/2)/width,
        (center[1]-scale[1]/2)/height,scale[0]/width,scale[1]/height)))
    return center, scale, affine, crop_xywh


def _prepare_input(image: np.ndarray, box: tuple[float, float, float, float]):
    """Return RGB NCHW float32 plus the exact inverse-coordinate parameters."""
    height, width = image.shape[:2]
    center,scale,affine,_ = pose_input_geometry((width,height),box)
    crop = cv2.warpAffine(image, affine, (192, 256), flags=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = (rgb - np.asarray((123.675, 116.28, 103.53), np.float32)) / np.asarray(
        (58.395, 57.12, 57.375), np.float32
    )
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None]), center, scale


def _decode_outputs(outputs, center, scale):
    if (
        len(outputs) != 2
        or [out.shape for out in outputs] != [(1, 17, 384), (1, 17, 512)]
        or any(not np.isfinite(out).all() for out in outputs)
    ):
        raise ValueError("RTMPose SimCC output contract differs")
    sx, sy = (out[0] for out in outputs)
    positions = np.stack((sx.argmax(1), sy.argmax(1)), axis=-1) / 2
    scores = np.minimum(sx.max(1), sy.max(1))
    points = positions / np.asarray((192, 256)) * scale + center - scale / 2
    return points, scores


@dataclass(frozen=True, slots=True)
class BodyPose2DEvidence:
    reason: str
    bbox_xywh: tuple[float, float, float, float]
    frame_size_wh: tuple[int, int]
    # Only non-facial joints are retained. Coordinates are not clamped: an
    # off-screen estimate must remain distinguishable from a visible joint.
    upper_body_xyc: tuple[tuple[float, float, float], ...] = ()
    face_anchor_consistent: bool | None = None
    pose_crop_xywh: tuple[float,float,float,float] | None = None
    source_pixel_to_pose_input_affine: tuple = ()
    # Preserve the actual un-clipped 17-joint output/confidence for ownership.
    # This is not a head-extent/visibility or named-identity measurement.
    raw_keypoints_xyc: tuple = ()

    def as_formation_person(self, detection_confidence: float) -> FormationPersonDescriptor | None:
        """Expose the eight-joint basis only when its normalization torso is visible."""
        if self.reason != "observed" or len(self.upper_body_xyc) != 8:
            return None
        points = np.asarray(self.upper_body_xyc)
        if (
            min(points[list(_TORSO), 2]) < .3
            or sum(points[:, 2] >= .3) < 5
            or np.any(points[list(_TORSO), :2] < 0)
            or np.any(points[list(_TORSO), :2] >= 1)
            or self.face_anchor_consistent is False
        ):
            return None
        pixels = points[:, :2] * np.asarray(self.frame_size_wh)
        shoulder = (pixels[0] + pixels[1]) / 2
        hip = (pixels[6] + pixels[7]) / 2
        center = (shoulder + hip) / 2
        scale = max(float(np.linalg.norm(shoulder - hip)) * 1.65,
                    float(np.linalg.norm(pixels[0] - pixels[1])), 1.)
        normalized = np.clip((pixels - center) / scale, -1, 1)
        values = tuple((float(x), float(y), float(confidence))
                       for (x, y), confidence in zip(normalized, points[:, 2]))
        bx, by, bw, bh = self.bbox_xywh
        return FormationPersonDescriptor(self.bbox_xywh, (bx + bw / 2, by + bh / 2),
                                         detection_confidence, values)


def pose_face_anchor_consistency(evidence, face_bbox_xywh):
    """Pure constraint view over a raw pose; never a second network forward.

    Face analysis variants do not alter the top-down body tensor. They must
    not lend one another their constraint result or modify cached raw joints.
    This is a nose/face ROI check, not named or cross-camera identity.
    """
    if face_bbox_xywh is None:
        return None
    if not _valid_box(face_bbox_xywh):
        raise ValueError('pose face constraint must fit normalized image bounds')
    if evidence is None or evidence.reason != 'observed':
        return None
    raw=evidence.raw_keypoints_xyc
    if (len(raw)!=17 or any(len(point)!=3 or any(not math.isfinite(v) for v in point) for point in raw)):
        raise ValueError('pose face constraint requires the original finite raw17 model output')
    x,y,confidence=raw[0]
    fx,fy,fw,fh=face_bbox_xywh
    distance=math.hypot((x-(fx+fw/2))/fw,(y-(fy+fh/2))/fh)
    return bool(confidence>=.3 and distance<=.8)


class BodyPose2DAnalyzer:
    """Lazy, locked inference on a caller-selected body; never an identity tracker."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self._net = None
        self._load_reason = None
        self._lock = Lock()

    @staticmethod
    def fingerprint() -> dict[str, str]:
        return {
            "contract": "rtmpose_body7_exact_roi_v1",
            "model_sha256": RTMPOSE_BODY7_MODEL_SHA256,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "opencv_version": cv2.__version__, "numpy_version": np.__version__,
            "backend": "opencv_cpu", "feature_basis": "eight_nonfacial_screen_xyc",
        }

    def _load(self) -> str | None:
        if self._load_reason is not None:
            return self._load_reason
        if self._net is not None:
            return None
        if self.model_path is None or not self.model_path.is_file():
            return "pose_model_unavailable"  # allow an external SSD to return
        try:
            digest = hashlib.sha256()
            with self.model_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != RTMPOSE_BODY7_MODEL_SHA256:
                self._load_reason = "pose_model_digest_mismatch"
                return self._load_reason
            self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except (OSError, cv2.error):
            return "pose_model_unavailable"
        return None

    def describe(self, image: np.ndarray, *, bbox_xywh: tuple[float, float, float, float],
                 face_bbox_xywh: tuple[float, float, float, float] | None = None) -> BodyPose2DEvidence:
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
                or not image.size or image.dtype != np.uint8):
            raise ValueError("body pose expects a non-empty uint8 BGR image")
        if not _valid_box(bbox_xywh) or (face_bbox_xywh is not None and not _valid_box(face_bbox_xywh)):
            raise ValueError("body and optional face boxes must fit normalized image bounds")
        height, width = image.shape[:2]
        base = {"bbox_xywh": bbox_xywh, "frame_size_wh": (width, height)}
        with self._lock:
            missing = self._load()
            if missing:
                return BodyPose2DEvidence(missing, **base)
            try:
                tensor, center, scale = _prepare_input(image, bbox_xywh)
                _,_,affine,crop = pose_input_geometry((width,height),bbox_xywh)
                base.update(pose_crop_xywh=crop,
                    source_pixel_to_pose_input_affine=tuple(tuple(map(float,row)) for row in affine))
                self._net.setInput(tensor)
                outputs = self._net.forward(self._net.getUnconnectedOutLayersNames())
                points, scores = _decode_outputs(outputs, center, scale)
            except (cv2.error, ValueError, FloatingPointError):
                return BodyPose2DEvidence("pose_inference_failed", **base)
        values = tuple((float(points[i, 0] / width), float(points[i, 1] / height),
                        float(np.clip(scores[i], 0, 1))) for i in _BODY_JOINTS)
        raw = BodyPose2DEvidence("observed", upper_body_xyc=values,
            raw_keypoints_xyc=tuple((float(x/width),float(y/height),float(confidence))
                for (x,y),confidence in zip(points,scores,strict=True)),**base)
        return replace(raw,face_anchor_consistent=pose_face_anchor_consistency(raw,face_bbox_xywh))
