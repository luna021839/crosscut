"""Optional pinned RTMW-133 observations, separate from legacy body caches.

No downloads, detector-based skeleton selection, named identity or inferred
visibility. SimCC peaks are RAW scores, often above one, NOT probabilities.
Graph metadata is authoritative: this export's SDK preprocess JSON incorrectly
retains192x256, whereas its ONNX input and postprocess both specify288x384.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import cv2
import numpy as np
from crosscut.body_pose_2d import pose_input_geometry
from .grid_perception import file_identity

MODEL_SHA256='bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054'
MODEL_URL='https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip'


@dataclass(frozen=True)
class WholeBodyPose:
    reason: str
    wholebody_xyc: np.ndarray
    crop_xywh: tuple
    source_to_input_affine: np.ndarray

    @property
    def raw_keypoints_xyc(self):
        # Same COCO body17 order; retain133 separately, never relabel hand
        # landmarks as body joints or silently clip raw scores into [0,1].
        return self.wholebody_xyc[:17]


def prepare(image, box):
    if (not isinstance(image,np.ndarray) or image.ndim!=3 or image.shape[2]!=3
            or image.dtype!=np.uint8 or not image.size):
        raise ValueError('RTMW expects actual uint8 BGR pixels')
    h,w=image.shape[:2]
    center,scale,affine,crop=pose_input_geometry((w,h),tuple(box))
    affine=affine*1.5
    raster=cv2.warpAffine(image,affine,(288,384),flags=cv2.INTER_LINEAR)
    rgb=cv2.cvtColor(raster,cv2.COLOR_BGR2RGB).astype(np.float32)
    value=(rgb-np.array([123.675,116.28,103.53],np.float32))/np.array([58.395,57.12,57.375],np.float32)
    return np.ascontiguousarray(value.transpose(2,0,1)[None]),center,scale,affine,crop


def decode(outputs, center, scale, *, raster_wh):
    if (len(outputs)!=2 or [v.shape for v in outputs]!=[(1,133,576),(1,133,768)]
            or any(not np.isfinite(v).all() for v in outputs)):
        raise ValueError('RTMW requires the actual finite133-joint SimCC outputs')
    sx,sy=(v[0] for v in outputs)
    xy=np.stack((sx.argmax(1),sy.argmax(1)),axis=-1)/2.
    positions=(xy/[288.,384.]*scale+center-scale/2.)/np.asarray(raster_wh)
    scores=np.minimum(sx.max(1),sy.max(1))
    # Negative maxima carry no positive keypoint evidence; coordinates remain
    # the actual decoded argmax, including off-raster model estimates.
    raw=np.column_stack((positions,scores)).astype(np.float32)
    raw.setflags(write=False)
    return raw


class WholeBodySensor:
    """One lazy CPU session; warm supplementary cache need not load weights."""
    def __init__(self,model_path):
        self.path=Path(model_path);self.runtime=None;self.identity=None;self.calls=0

    def _load(self):
        if self.runtime is None:
            import onnxruntime as ort
            self.identity=file_identity(self.path)
            with self.path.open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=MODEL_SHA256:
                    raise ValueError('RTMW asset differs from the pinned official model')
            options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
            options.log_severity_level=3
            self.runtime=ort.InferenceSession(str(self.path),sess_options=options,providers=['CPUExecutionProvider'])
            inputs=self.runtime.get_inputs()
            if len(inputs)!=1 or inputs[0].name!='input' or inputs[0].shape[1:]!=[3,384,288]:
                raise ValueError('RTMW graph input differs from pinned source geometry')
        if self.identity!=file_identity(self.path):raise OSError('RTMW model changed during observation')

    def describe(self,image,*,bbox_xywh):
        tensor,center,scale,affine,crop=prepare(image,bbox_xywh)
        self._load()
        outputs=self.runtime.run(None,{'input':tensor});self.calls+=1
        if self.identity!=file_identity(self.path):raise OSError('RTMW model changed during inference')
        raw=decode(outputs,center,scale,raster_wh=(image.shape[1],image.shape[0]))
        return WholeBodyPose('observed',raw,crop,affine)
