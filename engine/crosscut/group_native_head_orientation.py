"""Counted tilted-face detector alternatives on the ORIGINAL source raster.

These rotate only the detector's analysis input, never a render/source sample.
Every detection maps back to original source coordinates and keeps its own
variant/ROI. There is no largest-face pick, duplicate-person merge, physical
extent claim, pose imputation or semantic identity admission.
"""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

from .group_joint_native_observer import NativeGroupHead
from .visual_score import YUNET_EXPECTED_SHA256, _working_frame


SUPPORTED_ANALYSIS_ROLLS = (-60., -30., 30., 60.)


def _angles(angles):
    if (not isinstance(angles, tuple) or not angles or len(angles) > 4
            or len(set(angles)) != len(angles) or any(type(a) not in (int, float)
                or not math.isfinite(a) or a not in SUPPORTED_ANALYSIS_ROLLS for a in angles)):
        raise ValueError('native head orientation needs at most four distinct frozen analysis rolls')
    return tuple(float(a) for a in angles)


def _model_name(angle):
    return f'yunet_native_analysis_roll_{"minus" if angle < 0 else "plus"}_{int(abs(angle))}'


@dataclass(frozen=True, slots=True)
class OrientedNativeHeadAlternative:
    model_name: str
    analysis_roll_degrees: float
    working_size_wh: tuple[int, int]
    source_to_analysis_pixel_affine: tuple[tuple[float, float, float], ...]
    head: NativeGroupHead
    inverse_mapped_face_roi_corners_xy: tuple[tuple[float, float], ...]

    def to_dict(self):
        return dict(**asdict(self), detector_roi_not_physical_whole_head_extent=True,
            source_coordinates_not_clamped=True, analysis_variant_is_not_another_person=True,
            candidate_admitted=False, named_or_semantic_identity_verified=False)


def inverse_mapped_head(row, *, working_size_wh, analysis_affine,
        analysis_roll_degrees, observation_id):
    """Replay detector pixels through the exact inverse analysis transform.

    Normalized coordinates are relative to the original (unrotated) working
    raster. Its aspect ratio matches the original source. Off-raster corners
    remain off-raster; no clipping is used to fabricate valid facial coverage.
    """
    width, height = working_size_wh
    matrix = np.asarray(analysis_affine, dtype=float)
    values = np.asarray(row, dtype=float)
    if (type(width) is not int or type(height) is not int or min(width, height) < 1
            or matrix.shape != (2, 3) or not np.isfinite(matrix).all()
            or values.shape != (15,) or not np.isfinite(values).all()
            or min(values[2:4]) <= 0 or not 0 <= values[14] <= 1
            or analysis_roll_degrees not in SUPPORTED_ANALYSIS_ROLLS
            or type(observation_id) is not int or observation_id < 0
            or abs(np.linalg.det(matrix[:, :2])) < 1e-12):
        raise ValueError('oriented native head needs finite detector/affine/source pixel geometry')
    expected = cv2.getRotationMatrix2D(((width-1)/2,(height-1)/2),float(analysis_roll_degrees),1.)
    if not np.allclose(matrix,expected,rtol=0.,atol=1e-12):
        raise ValueError('oriented native head declared roll/center does not replay its analysis affine')
    inverse = cv2.invertAffineTransform(matrix)
    def unrotate(points):
        return (np.column_stack((points,np.ones(len(points))))@inverse.T)/[width,height]
    x,y,w,h = values[:4]
    corners = unrotate(np.asarray(((x,y),(x+w,y),(x+w,y+h),(x,y+h))))
    points = unrotate(values[4:14].reshape(5,2))
    low, high = corners.min(0), corners.max(0)
    head = NativeGroupHead(observation_id,tuple(float(v) for v in (*low,*(high-low))),
        tuple(tuple(float(v) for v in point) for point in points),float(values[14]))
    return OrientedNativeHeadAlternative(_model_name(analysis_roll_degrees),float(analysis_roll_degrees),
        (width,height),tuple(tuple(float(v) for v in line) for line in matrix),head,
        tuple(tuple(float(v) for v in point) for point in corners))


def register_native_head_orientation_models(session, model_path, *, angles=SUPPORTED_ANALYSIS_ROLLS):
    angles = _angles(angles); path = Path(model_path)
    session.budget.check()
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle,'sha256').hexdigest()
    if digest != YUNET_EXPECTED_SHA256:
        raise ValueError('native head orientation differs from the pinned YuNet model')
    names = []
    for angle in angles:
        detector = cv2.FaceDetectorYN.create(str(path),'',(960,540),.76,.3,5000)
        name = _model_name(angle); names.append(name)
        def observe(pixels, *, _angle=angle, _detector=detector):
            working = _working_frame(pixels,maximum_width=960)
            height,width = working.shape[:2]
            affine = cv2.getRotationMatrix2D(((width-1)/2,(height-1)/2),_angle,1.)
            oriented = cv2.warpAffine(working,affine,(width,height),flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            _detector.setInputSize((width,height)); _,rows = _detector.detect(oriented)
            if rows is None:
                return ()
            if len(rows) > 24:
                return None  # explicit unavailable, never convenient truncation
            return tuple(inverse_mapped_head(row,working_size_wh=(width,height),
                analysis_affine=affine,analysis_roll_degrees=_angle,observation_id=index)
                for index,row in enumerate(rows))
        # One registered call is exactly one actual detector invocation. The
        # ensemble is never hidden inside a single model-budget debit.
        session.register_model(name,path,observe,adapter_fingerprint=dict(
            contract='original_source_inverse_mapped_analysis_roll_face_alternatives_v1',
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            opencv_version=cv2.__version__,angle=angle,maximum_width=960,
            confidence=.76,nms=.3,top_k=5000,interpolation='linear',border='constant_zero',
            center_basis='working_raster_pixel_centers'))
    return tuple(names)


def observe_native_head_orientation_alternatives(session, *, request_source_index,
        native_frame_index, angles=SUPPORTED_ANALYSIS_ROLLS):
    """All declared variants, raw pixel and registered adapter bindings.

    Missing/ambiguous observations remain explicit. Native model-cache reuse
    is allowed only by the existing worker's exact source/pixel/adapter keys.
    A separate consumer must associate a subject and verify physical extent.
    """
    angles = _angles(angles)
    identity = session._assert_source_identity(request_source_index)
    pixels = session.native_pixels(request_source_index,native_frame_index)
    values = []
    for angle in angles:
        name = _model_name(angle)
        alternatives = session.observe_native(request_source_index,native_frame_index,model=name)
        values.append(dict(model_name=name,analysis_roll_degrees=angle,
            model_content_sha256=session.models[name][2],adapter_sha256=session.model_adapters[name],
            unavailable=alternatives is None,
            alternatives=[] if alternatives is None else [v.to_dict() for v in alternatives]))
    clock = session.domain.clocks_by_request[request_source_index]
    return dict(contract='bound_original_native_head_analysis_orientation_alternatives_v1',
        request_source_index=request_source_index,native_frame_index=native_frame_index,
        source_file_identity=identity,source_clock=clock.to_dict(),
        native_pixel_sha256=None if pixels is None else hashlib.sha256(memoryview(pixels)).hexdigest(),
        variants=values,source_pixels_or_clock_or_render_modified=False,
        hypotheses_deduplicated_or_subject_selected=False,source_occlusion_inferred=False,
        physical_whole_head_extent_verified=False,semantic_subject_verified=False,
        candidate_admitted=False,quality_approved=False)
