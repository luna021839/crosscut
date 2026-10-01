"""Optional native semantic-part measurements, not an admission certificate.

The FP32 SCHP LIP export predicts hair/face/arm pixels rather than inflating a
face rectangle into an invented whole head. Predictions and uncertainty stay
separate from owned subjects, complete extent and motion. No remote Python,
automatic download, training, part imputation or persistent cache rewrite.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

from .group_joint_observations import _file_identity


SCHP_LIP_REVISION = '7c0028fe34dcd8bd534ffa291a9086cdc856e4a9'
SCHP_LIP_GRAPH_SHA256 = '41c9d6c753a387f794bf3cbd9aed5e20c81a934fdc91731475bd46138937f84c'
SCHP_LIP_DATA_SHA256 = 'be931b91c76ed7707b5426abbccee98620a122b777c7cbd1f771a122803666e7'
SCHP_LIP_MODEL_NAME = 'schp_lip_native_semantic_parts_fp32'
SCHP_LIP_LABELS = ('background', 'hat', 'hair', 'glove', 'sunglasses',
    'upper_clothes', 'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf',
    'skirt', 'face', 'left_arm', 'right_arm', 'left_leg', 'right_leg',
    'left_shoe', 'right_shoe')
HEAD_LABELS = (1, 2, 4, 13)
ARM_LABELS = (14, 15)


def _valid_native_roi(box):
    return (isinstance(box, (tuple, list)) and len(box) == 4
        and all(type(v) in (int, float) and math.isfinite(v) for v in box)
        and box[0] >= 0 and box[1] >= 0 and min(box[2:]) > 0
        and box[0]+box[2] <= 1.000001 and box[1]+box[3] <= 1.000001)


def prepare_schp_native_roi(pixels, bbox_xywh):
    """Publisher's PIL bilinear RGB/473/BGR-indexed normalization contract.

    The integer native crop bounds are explicit. No out-of-native padding is
    invented, and a caller must not treat this ROI as full person support.
    """
    from PIL import Image
    if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
            or pixels.ndim != 3 or pixels.shape[2] != 3 or not _valid_native_roi(bbox_xywh)):
        raise ValueError('semantic parts need actual BGR8 pixels and a valid native ROI')
    height, width = pixels.shape[:2]
    x, y, w, h = bbox_xywh
    left, top = math.floor(x*width), math.floor(y*height)
    right, bottom = min(width, math.ceil((x+w)*width)), min(height, math.ceil((y+h)*height))
    if right <= left or bottom <= top:
        raise ValueError('semantic ROI has no native pixels')
    image = Image.fromarray(cv2.cvtColor(pixels[top:bottom, left:right], cv2.COLOR_BGR2RGB))
    image = image.resize((473, 473), resample=Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float32)/255.
    values = (values-np.asarray((.406, .456, .485), np.float32))/np.asarray((.225, .224, .229), np.float32)
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None]), (left, top, right-left, bottom-top)


@dataclass(frozen=True, slots=True)
class NativeSemanticParts:
    frame_size_wh: tuple[int, int]
    requested_roi_xywh: tuple[float, float, float, float]
    crop_native_xywh_pixels: tuple[int, int, int, int]
    label_map: np.ndarray
    winning_probability: np.ndarray

    def __post_init__(self):
        if (not isinstance(self.frame_size_wh, tuple) or len(self.frame_size_wh) != 2
                or any(type(v) is not int or v <= 0 for v in self.frame_size_wh)
                or not _valid_native_roi(self.requested_roi_xywh)
                or not isinstance(self.crop_native_xywh_pixels, tuple)
                or len(self.crop_native_xywh_pixels) != 4
                or any(type(v) is not int for v in self.crop_native_xywh_pixels)
                or min(self.crop_native_xywh_pixels[:2]) < 0
                or min(self.crop_native_xywh_pixels[2:]) <= 0
                or self.crop_native_xywh_pixels[0]+self.crop_native_xywh_pixels[2] > self.frame_size_wh[0]
                or self.crop_native_xywh_pixels[1]+self.crop_native_xywh_pixels[3] > self.frame_size_wh[1]
                or not isinstance(self.label_map, np.ndarray) or self.label_map.dtype != np.uint8
                or self.label_map.shape != (473, 473) or np.any(self.label_map >= 20)
                or not isinstance(self.winning_probability, np.ndarray)
                or self.winning_probability.dtype != np.float32
                or self.winning_probability.shape != self.label_map.shape
                or not np.isfinite(self.winning_probability).all()
                or np.any(self.winning_probability < 0) or np.any(self.winning_probability > 1)):
            raise ValueError('semantic prediction must retain its exact native support and finite part raster')
        for name in ('label_map', 'winning_probability'):
            values = getattr(self, name).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        width, height = self.frame_size_wh
        x, y, w, h = self.requested_roi_xywh
        left, top = math.floor(x*width), math.floor(y*height)
        expected = (left, top, min(width, math.ceil((x+w)*width))-left,
            min(height, math.ceil((y+h)*height))-top)
        if expected != self.crop_native_xywh_pixels:
            raise ValueError('semantic prediction crop differs from its requested native ROI')

    def to_dict(self):
        return dict(contract='native_schp_lip_semantic_part_prediction_not_extent_certificate_v1',
            frame_size_wh=self.frame_size_wh, requested_roi_xywh=self.requested_roi_xywh,
            crop_native_xywh_pixels=self.crop_native_xywh_pixels, prediction_raster_size_wh=(473, 473),
            label_counts={name: int(np.count_nonzero(self.label_map == n)) for n, name in enumerate(SCHP_LIP_LABELS)},
            label_map_sha256=hashlib.sha256(memoryview(self.label_map)).hexdigest(),
            winning_probability_sha256=hashlib.sha256(memoryview(self.winning_probability)).hexdigest(),
            model_revision=SCHP_LIP_REVISION, model_graph_sha256=SCHP_LIP_GRAPH_SHA256,
            model_external_data_sha256=SCHP_LIP_DATA_SHA256,
            probability_is_calibrated_confidence=False, subject_ownership_certified=False,
            whole_head_extent_certified=False, arm_visibility_certified=False,
            head_or_arm_motion_certified=False, production_admitted=False)


class SCHPNativePartAnalyzer:
    """Pinned standard ONNX only; the export's remote Python is never run."""
    def __init__(self, model_path):
        import onnxruntime as ort
        self.model_path = Path(model_path)
        if self.model_path.name != 'schp-lip-20.onnx':
            raise ValueError('semantic part graph must keep its bound external-data filename')
        self.data_path = self.model_path.with_name('schp-lip-20.onnx.data')
        self.identities = {}
        for path, expected in ((self.model_path, SCHP_LIP_GRAPH_SHA256), (self.data_path, SCHP_LIP_DATA_SHA256)):
            before = _file_identity(path)
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != expected or _file_identity(path) != before:
                raise ValueError('semantic graph/external weights differ from the pinned asset; preserved')
            self.identities[path] = before
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.runtime = ort.InferenceSession(str(self.model_path), options, providers=['CPUExecutionProvider'])
        if ([(v.name, v.shape[1:], v.type) for v in self.runtime.get_inputs()] !=
                [('pixel_values', [3, 473, 473], 'tensor(float)')]
                or [(v.name, v.shape[1:], v.type) for v in self.runtime.get_outputs()] !=
                [('logits', [20, 473, 473], 'tensor(float)')]):
            raise ValueError('semantic ONNX input/output differs from the pinned publisher contract')
        self.assert_assets()

    def assert_assets(self):
        if any(_file_identity(path) != identity for path, identity in self.identities.items()):
            raise ValueError('semantic model graph/external weight identity changed')

    def describe(self, pixels, *, bbox_xywh):
        self.assert_assets()
        values, bounds = prepare_schp_native_roi(pixels, bbox_xywh)
        logits = self.runtime.run(['logits'], {'pixel_values': values})[0]
        self.assert_assets()
        if logits.shape != (1, 20, 473, 473) or not np.isfinite(logits).all():
            raise ValueError('semantic logits changed shape or contain nonfinite values')
        logits = logits[0]
        labels = logits.argmax(axis=0).astype(np.uint8)
        # Retain the actual winning softmax value, not a binary success score.
        normalizer = np.exp(logits-logits.max(axis=0, keepdims=True)).sum(axis=0)
        probability = (1./normalizer).astype(np.float32)
        return NativeSemanticParts((pixels.shape[1], pixels.shape[0]), tuple(bbox_xywh), bounds, labels, probability)


def register_joint_native_part_model(session, model_path):
    import onnxruntime as ort
    import PIL
    session.budget.check()
    analyzer = SCHPNativePartAnalyzer(model_path)
    session.budget.check()
    session.register_model(SCHP_LIP_MODEL_NAME, analyzer.model_path, analyzer.describe,
        asset_dependencies=(analyzer.data_path,),
        adapter_fingerprint={'contract': 'source_native_semantic_parts_only_not_role_or_visibility_v1',
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'external_weights_sha256': SCHP_LIP_DATA_SHA256, 'publisher_revision': SCHP_LIP_REVISION,
            'onnxruntime_version': ort.__version__, 'pillow_version': PIL.__version__,
            'providers': ['CPUExecutionProvider'], 'threads': 4, 'labels': SCHP_LIP_LABELS,
            'resize': 'PIL_RGB_bilinear_473', 'mean': [.406, .456, .485], 'std': [.225, .224, .229]})
    analyzer.assert_assets()
    return analyzer


def observed_head_components(parts, *, face_boxes_xywh):
    """Retain ALL predicted head components touching each measured face.

    Touching/fragmented people, competing anchors, missing hair and low model
    probability are data, not permission to choose the largest component or
    declare complete whole-head extent. The caller still owns correspondence.
    """
    boxes = tuple(face_boxes_xywh)
    if not isinstance(parts, NativeSemanticParts) or any(not _valid_native_roi(box) for box in boxes):
        raise ValueError('semantic head support needs actual part predictions and native face anchors')
    mask = np.isin(parts.label_map, HEAD_LABELS).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    x, y, w, h = parts.crop_native_xywh_pixels
    width, height = parts.frame_size_wh
    hypotheses = []
    for n in range(1, count):
        left, top, bw, bh, area = map(int, stats[n])
        native_box = ((x+left*w/473)/width, (y+top*h/473)/height,
            bw*w/473/width, bh*h/473/height)
        anchors = []
        for index, box in enumerate(boxes):
            rx0 = max(0, math.floor((box[0]*width-x)*473/w))
            ry0 = max(0, math.floor((box[1]*height-y)*473/h))
            rx1 = min(473, math.ceil(((box[0]+box[2])*width-x)*473/w))
            ry1 = min(473, math.ceil(((box[1]+box[3])*height-y)*473/h))
            if rx0 < rx1 and ry0 < ry1 and np.any(labels[ry0:ry1, rx0:rx1] == n):
                anchors.append(index)
        if anchors:
            values = parts.winning_probability[labels == n]
            hypotheses.append(dict(component_id=n, face_anchor_indices=tuple(anchors),
                predicted_native_box_xywh=native_box, predicted_pixels=area,
                minimum_winning_probability=float(values.min()), mean_winning_probability=float(values.mean()),
                multiple_face_anchors=len(anchors) != 1, complete_extent_verified=False,
                subject_identity_verified=False))
    return tuple(hypotheses)
