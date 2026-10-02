"""Optional pinned ArcFace descriptor for source-local cross-camera trials.

This does not identify a named person or turn descriptor similarity into
semantic identity. No model is downloaded here; the caller supplies its
verified local asset. Existing SFace descriptors are never reinterpreted.
"""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import cv2
import numpy as np

from .face_identity import SFACE_SHA256, _validate_normalized_face


ARCFACE_R50_SHA256 = '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43'
ARCFACE_MODEL_NAME = 'arcface_r50_native_face'
SFACE_MODEL_NAME = 'sface_owned_face'
ARCFACE_TARGET = np.asarray(((38.2946,51.6963),(73.5318,51.5014),(56.0252,71.7366),
    (41.5493,92.3655),(70.7299,92.2041)),dtype=np.float64)
ARCFACE_TARGET.setflags(write=False)


def native_descriptor_schema(model_name):
    """Pinned spaces are incompatible even when a cosine looks plausible."""
    if model_name == SFACE_MODEL_NAME:
        return 128, SFACE_SHA256
    if model_name == ARCFACE_MODEL_NAME:
        return 512, ARCFACE_R50_SHA256
    raise ValueError('unknown native face descriptor space')


@dataclass(frozen=True, slots=True)
class NativeFaceDescriptorSpace:
    model_name: str
    dimension: int
    model_sha256: str
    adapter_sha256: str

    def __post_init__(self):
        dimension, digest = native_descriptor_schema(self.model_name)
        if (type(self.dimension) is not int or self.dimension != dimension
                or self.model_sha256 != digest or not isinstance(self.adapter_sha256, str)
                or len(self.adapter_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.adapter_sha256)):
            raise ValueError('native descriptor space needs pinned model, dimension and adapter')

    def to_dict(self):
        return asdict(self)


def native_descriptor_space(session, model_name):
    """Resolve the live registered asset, not a caller's display model label."""
    dimension, digest = native_descriptor_schema(model_name)
    if model_name not in session.models:
        raise ValueError('native descriptor backend is not registered in this worker')
    session._assert_model_assets(model_name)
    if session.models[model_name][2] != digest:
        raise ValueError('native descriptor backend differs from its pinned asset')
    return NativeFaceDescriptorSpace(model_name, dimension, digest, session.model_adapters[model_name])


def face_similarity_affine(source_points):
    """Deterministic proper 2-D similarity using all five observed points."""
    points=np.asarray(source_points,dtype=np.float64)
    if points.shape!=(5,2) or not np.isfinite(points).all():
        raise ValueError('face alignment needs five finite observed source landmarks')
    center=points.mean(0);target_center=ARCFACE_TARGET.mean(0)
    a=points-center;b=ARCFACE_TARGET-target_center
    variance=float(np.sum(a*a)/len(a))
    if variance<1e-6 or np.linalg.matrix_rank(a)<2:
        raise ValueError('face alignment landmarks are degenerate')
    u,s,vt=np.linalg.svd(b.T@a/len(a))
    signs=np.asarray((1.,1. if np.linalg.det(u@vt)>0 else -1.))
    rotation=(u*signs)@vt;scale=float(np.dot(s,signs)/variance)
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('face alignment requires a proper finite similarity')
    linear=scale*rotation;translation=target_center-linear@center
    return np.column_stack((linear,translation))


def arcface_input(frame, *, bbox_xywh, landmarks_xy):
    if (not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.ndim!=3
            or frame.shape[2]!=3 or not frame.size):
        raise ValueError('native face descriptor needs the actual uint8 BGR source raster')
    _validate_normalized_face(bbox_xywh,landmarks_xy)
    height,width=frame.shape[:2]
    matrix=face_similarity_affine(np.asarray(landmarks_xy)*[width,height])
    aligned=cv2.warpAffine(frame,matrix,(112,112),flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    # Pinned w600k_r50's official ArcFace preprocessing: RGB, 127.5/127.5.
    return np.ascontiguousarray(((aligned[:,:,::-1].astype(np.float32)-127.5)/127.5).transpose(2,0,1)[None])


class NativeArcFaceR50:
    def __init__(self,model_path):
        self.path=Path(model_path)
        with self.path.open('rb') as handle:
            # Hash and load the SAME bytes. Loading the path after hashing
            # permits a replacement between the two operations to run an
            # unpinned model under the expected descriptor-space identity.
            # Bound the read even when an invalid file grows while being read.
            model_bytes=handle.read(192*1024*1024+1)
        if len(model_bytes)>192*1024*1024:
            raise ValueError('native ArcFace descriptor exceeds its pinned asset size budget')
        digest=hashlib.sha256(model_bytes).hexdigest()
        if digest!=ARCFACE_R50_SHA256:
            raise ValueError('native ArcFace descriptor differs from pinned official R50 asset')
        self.asset_sha256=digest
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(model_bytes,sess_options=options,providers=['CPUExecutionProvider'])
        inputs=self.session.get_inputs();outputs=self.session.get_outputs()
        if (len(inputs)!=1 or len(outputs)!=1 or inputs[0].type!='tensor(float)'
                or len(inputs[0].shape)!=4 or tuple(inputs[0].shape[1:])!=(3,112,112)
                or len(outputs[0].shape)!=2 or outputs[0].shape[1]!=512):
            raise ValueError('native ArcFace pinned model IO schema differs')
        self.input_name=inputs[0].name;self.output_name=outputs[0].name

    def embedding(self,frame,*,bbox_xywh,landmarks_xy):
        blob=arcface_input(frame,bbox_xywh=bbox_xywh,landmarks_xy=landmarks_xy)
        result=self.session.run([self.output_name],{self.input_name:blob})[0]
        vector=np.asarray(result,dtype=np.float64).reshape(-1)
        if vector.size!=512 or not np.isfinite(vector).all():
            raise ValueError('native ArcFace returned invalid descriptor coordinates')
        length=float(np.linalg.norm(vector))
        if length<=1e-12:
            raise ValueError('native ArcFace returned an empty descriptor')
        return tuple(float(v) for v in vector/length)


def register_native_arcface_model(session,model_path):
    """Exactly one ORT inference per counted source/ROI invocation."""
    session.budget.check();model=NativeArcFaceR50(model_path);session.budget.check()
    import onnxruntime as ort
    session.register_model(ARCFACE_MODEL_NAME,model.path,model.embedding,
        expected_asset_sha256=model.asset_sha256,adapter_fingerprint=dict(
        contract='source_native_arcface_r50_512d_similarity_not_semantic_identity_v1',
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_sha256=ARCFACE_R50_SHA256,onnxruntime_version=ort.__version__,opencv_version=cv2.__version__,
        execution_provider='CPUExecutionProvider',input_size=(112,112),input_color='RGB',
        input_mean=127.5,input_std=127.5,alignment='five_observed_points_proper_similarity_svd',
        embedding_dimension=512,source_coordinates_clamped=False))
    return ARCFACE_MODEL_NAME
