"""Explicit, pinned YOLO11 COCO person rectangles from original BGR pixels.

Opt-in adapter only: no model download, Torch import, default-detector change,
cast selection or ownership/pose/identity claim. One infer call runs one model
forward, so the native observation session can charge/cache it honestly.
Pre/postprocessing is original code for the documented YOLO detection tensor.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

from .onnx_cpu_inference import OnnxCpuNetwork

CONTRACT = 'native_yolo11_coco_person_rectangles_onnx_cpu_v1'
MODEL_NAME = 'yolo11l_native_people'
PINNED_YOLO11L_ONNX_SHA256 = 'd7da2dd3cd23bffb3b2b24dee0e4575a8aa50ccdb2b3510aa9f8a57a0b232562'


@dataclass(frozen=True, slots=True)
class PersonDetectorConfig:
    maximum_side: int = 1280
    confidence: float = .35
    nms_iou: float = .55

    def __post_init__(self):
        if (type(self.maximum_side) is not int or not 32 <= self.maximum_side <= 1280
                or self.maximum_side % 32):
            raise ValueError('person detector size must be a bounded multiple of 32')
        for value in (self.confidence, self.nms_iou):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError('person detector thresholds must be explicit finite fractions')


@dataclass(frozen=True, slots=True)
class NativeLetterbox:
    native_width: int
    native_height: int
    input_width: int
    input_height: int
    scale: float
    left: int
    top: int


def native_letterbox(pixels, *, maximum_side):
    PersonDetectorConfig(maximum_side=maximum_side)
    if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
            or pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 1):
        raise ValueError('person detector requires nonempty original uint8 BGR pixels')
    height, width = pixels.shape[:2]
    scale = maximum_side / max(height, width)
    new_width, new_height = round(width * scale), round(height * scale)
    if min(new_width, new_height) < 1:
        raise ValueError('unsupported source aspect ratio')
    dw, dh = (maximum_side - new_width) % 32, (maximum_side - new_height) % 32
    left, top = round(dw / 2 - .1), round(dh / 2 - .1)
    right, bottom = round(dw / 2 + .1), round(dh / 2 + .1)
    resized = cv2.resize(pixels, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= np.float32(255.)
    transform = NativeLetterbox(width, height, padded.shape[1], padded.shape[0], scale, left, top)
    return tensor, transform


def person_rows(prediction, transform, *, config):
    """Decode the explicit 80-class XYWH+scores tensor, preserving native axes."""
    anchors = sum((transform.input_width // s) * (transform.input_height // s) for s in (8, 16, 32))
    if (not isinstance(prediction, np.ndarray) or prediction.shape != (1, 84, anchors)
            or prediction.dtype != np.float32 or not np.isfinite(prediction).all()):
        raise ValueError('person detector returned a different/nonfinite YOLO11 tensor')
    values = prediction[0].T
    scores = values[:, 4:]
    if np.any(scores < 0) or np.any(scores > 1) or np.any(values[:, 2:4] < 0):
        raise ValueError('invalid detector probabilities or box extents')
    # Match single-label COCO inference: a different winning class does not
    # become a person merely because class zero also has a nonzero score.
    labels = scores.argmax(axis=1)
    selected = values[(labels == 0) & (scores[:, 0] > config.confidence)]
    if not len(selected):
        return np.empty((0, 5), dtype=np.float32)
    boxes = selected[:, :4].copy()
    boxes[:, :2] -= boxes[:, 2:] / 2
    ids = np.asarray(cv2.dnn.NMSBoxes(boxes.tolist(), selected[:, 4].tolist(),
                     config.confidence, config.nms_iou)).reshape(-1)
    if not len(ids):
        return np.empty((0, 5), dtype=np.float32)
    boxes = boxes[ids]
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, (0, 2)] = (boxes[:, (0, 2)] - transform.left) / transform.scale
    boxes[:, (1, 3)] = (boxes[:, (1, 3)] - transform.top) / transform.scale
    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, transform.native_width)
    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, transform.native_height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    # Padding-only boxes have no native-pixel support; not a semantic cast filter.
    return np.ascontiguousarray(np.column_stack((boxes[valid], selected[ids[valid], 4])), dtype=np.float32)


class NativePersonOnnxDetector:
    def __init__(self, path, *, expected_sha256, config=None):
        self.config = config if config is not None else PersonDetectorConfig()
        if not isinstance(self.config, PersonDetectorConfig):
            raise ValueError('explicit person detector configuration required')
        self.network = OnnxCpuNetwork(Path(path), expected_sha256=expected_sha256,
                                      output_names=('output0',), intra_op_threads=4)

    def infer(self, pixels):
        tensor, transform = native_letterbox(pixels, maximum_side=self.config.maximum_side)
        self.network.setInput(tensor)
        return person_rows(self.network.forward(), transform, config=self.config)

    def fingerprint(self):
        return dict(contract=CONTRACT, model_backend=self.network.fingerprint(),
            maximum_side=self.config.maximum_side, stride=32, input_color='BGR_to_RGB',
            pixel_scale='float32_divide_255', padding_value=114,
            confidence=self.config.confidence, nms_iou=self.config.nms_iou,
            coco_class=0, single_label=True, raw_person_count_not_truncated=True,
            inference_calls_per_observation=1, cast_identity_or_pose_certified=False,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def register_native_person_model(session, model_path, *, config=None):
    """Explicit normal-session integration, retaining its budget/cache ownership.

    The source owner still opts in and chooses its editorial cohort separately.
    The YOLO call is never labelled NanoDet or charged as a free cached result.
    A changed/corrupt pinned asset fails rather than silently selecting another
    backend. No person list, existing registration, or production default is
    replaced by this helper.
    """
    session.budget.check()
    if MODEL_NAME in session.models:
        raise ValueError('native person detector already registered in this session')
    path = Path(model_path)
    detector = NativePersonOnnxDetector(path, expected_sha256=PINNED_YOLO11L_ONNX_SHA256,
                                        config=config)
    session.budget.check()
    session.register_model(MODEL_NAME, path, detector.infer,
        adapter_fingerprint=detector.fingerprint(),
        expected_asset_sha256=PINNED_YOLO11L_ONNX_SHA256)
    return MODEL_NAME
