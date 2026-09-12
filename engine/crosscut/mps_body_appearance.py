"""Optional Apple GPU adapter; not enabled in active production jobs.

CPU and MPS evidence namespaces must remain separate. The adapter preserves
the pretrained model, input preprocessing and CPU-normalized output contract.
PyTorch 2.1's MPS InstanceNorm requires normal tensor version counters, so
model construction and GPU forward use no_grad rather than inference tensors.
"""
import hashlib
from pathlib import Path
from .body_appearance import BodyAppearanceEncoder


class MPSBodyAppearanceEncoder(BodyAppearanceEncoder):
    def _load(self):
        if self._model is not None:return self._model
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError('explicit MPS body encoder requires an available Apple GPU')
        with torch.inference_mode(False):
            native=super()._load().to('mps')
            class CPUContractForward(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.native=native
                def forward(self,values):
                    with torch.inference_mode(False),torch.no_grad():
                        return self.native(values.clone().to('mps')).to('cpu')
            self._model=CPUContractForward()
        return self._model

    def fingerprint(self):
        import torch
        return {**super().fingerprint(),'execution_device':'mps',
            'torch_version':torch.__version__,
            'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cpu_cache_interchangeable':False}
