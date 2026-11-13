"""Lazy pinned whole-frame COCO17 sensor; no face-conditioned inference.

CoreML's unsupported control-flow/NMS operators explicitly retain CPU support.
The cache location is caller-owned; do not compile a heavy default system
cache or download/convert/install another model as a fallback.
"""
import hashlib
from pathlib import Path

from .multi_person_pose import MODEL_SHA256,tensor,decode


class Sensor:
    def __init__(self,model,*,backend='coreml',coreml_cache=None,minimum_score=.1):
        import onnxruntime as ort
        path=Path(model)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=MODEL_SHA256:
            raise ValueError('pinned original RTMO body7-L ONNX required')
        if backend not in ('cpu','coreml') or not 0<=minimum_score<=1:
            raise ValueError('explicit original-pose backend and score threshold required')
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.add_free_dimension_override_by_name('batch',1)
        providers=['CPUExecutionProvider'];provider_options=None
        if backend=='coreml':
            if coreml_cache is None or 'CoreMLExecutionProvider' not in ort.get_available_providers():
                raise OSError('explicit available CoreML provider and model cache required')
            cache=Path(coreml_cache).resolve(strict=True)
            provider_options=dict(ModelFormat='MLProgram',MLComputeUnits='ALL',
                RequireStaticInputShapes='1',EnableOnSubgraphs='0',ModelCacheDirectory=str(cache))
            providers=[('CoreMLExecutionProvider',provider_options),'CPUExecutionProvider']
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=providers)
        if [v.name for v in self.session.get_inputs()]!=['input'] or [v.name for v in self.session.get_outputs()]!=['dets','keypoints']:
            raise ValueError('original RTMO input/output family differs')
        self.minimum_score=minimum_score;self.forwards=0
        self.receipt=dict(weights_sha256=MODEL_SHA256,configured_backend=backend,
            actual_session_providers=self.session.get_providers(),CoreML_options=provider_options,
            onnxruntime_version=ort.__version__,CPU_control_flow_fallback_explicit=backend=='coreml',
            named_identity_certified=False)

    def describe(self,image):
        """One normalization in the shared decoder, never a second inverse."""
        prepared,ratio=tensor(image)
        outputs=self.session.run(['dets','keypoints'],{'input':prepared});self.forwards+=1
        height,width=image.shape[:2]
        return decode(outputs,ratio=ratio,raster=(width,height),minimum_score=self.minimum_score)

    def close(self):
        self.session=None
