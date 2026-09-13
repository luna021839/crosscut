"""Explicit CPU execution adapter for existing OpenCV model preprocessing.

This does not select a backend globally or alter production defaults. Callers
must bind this backend identity into numeric caches before opting into it.
OpenCV and ORT outputs need not be bitwise equal even with identical weights.
"""
from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CONTRACT = "opencv_named_float32_forward_with_explicit_onnxruntime_cpu_v1"


class OnnxCpuBackendUnavailable(RuntimeError):
    pass


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


class OnnxCpuNetwork:
    """Small named-forward interface; geometry/decoding remain with the caller."""

    def __init__(
        self,
        model_path: Path,
        *,
        expected_sha256: str,
        output_names: Sequence[str],
        intra_op_threads: int = 2,
    ) -> None:
        if (isinstance(intra_op_threads, bool) or not isinstance(intra_op_threads, int)
                or intra_op_threads < 1):
            raise ValueError("ONNX CPU thread count must be a positive integer")
        if isinstance(output_names,str):
            raise ValueError("ONNX CPU output names must be a sequence, not one string")
        names = tuple(output_names)
        if not names or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):
            raise ValueError("ONNX CPU output names must be nonempty and unique")
        if (not isinstance(expected_sha256,str) or len(expected_sha256)!=64
                or any(c not in "0123456789abcdef" for c in expected_sha256)):
            raise ValueError("ONNX CPU model needs an expected SHA-256")
        path = Path(model_path).resolve(strict=True)
        before = _file_identity(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
        if digest.hexdigest()!=expected_sha256 or _file_identity(path)!=before:
            raise ValueError("ONNX CPU model differs from its immutable binding")
        try:
            runtime = import_module("onnxruntime")
        except ImportError as error:
            raise OnnxCpuBackendUnavailable("Install the optional ONNX Runtime CPU dependency") from error
        if "CPUExecutionProvider" not in runtime.get_available_providers():
            raise OnnxCpuBackendUnavailable("ONNX Runtime CPU provider is unavailable")
        options = runtime.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        session = runtime.InferenceSession(str(path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        if _file_identity(path)!=before:
            raise ValueError("ONNX CPU model changed while loading")
        if session.get_providers()!=["CPUExecutionProvider"]:
            raise ValueError("ONNX CPU session selected a different backend")
        inputs = session.get_inputs()
        if len(inputs)!=1 or inputs[0].type!="tensor(float)":
            raise ValueError("ONNX CPU adapter requires one float32 model input")
        if set(names)!={value.name for value in session.get_outputs()}:
            raise ValueError("ONNX CPU outputs differ from the caller's model contract")
        self._session = session
        self._input_name = inputs[0].name
        self._input_shape = tuple(inputs[0].shape)
        self._output_names = names
        self._value: np.ndarray | None = None
        self._fingerprint = {
            "contract": CONTRACT,
            "backend": "onnxruntime_cpu",
            "runtime_version": str(runtime.__version__),
            "numpy_version": np.__version__,
            "intra_op_threads": intra_op_threads,
            "inter_op_threads": 1,
            "model_sha256": expected_sha256,
            "output_names": list(names),
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "bitwise_opencv_cache_compatibility": False,
        }

    def fingerprint(self) -> dict[str, Any]:
        from copy import deepcopy
        return deepcopy(self._fingerprint)

    def getUnconnectedOutLayersNames(self) -> tuple[str, ...]:
        return self._output_names

    def setInput(self, value: np.ndarray) -> None:
        # Never keep the previous tensor after a failed new input validation.
        self._value = None
        if not isinstance(value,np.ndarray) or value.dtype!=np.float32:
            raise ValueError("ONNX CPU input must be a float32 array")
        if len(value.shape)!=len(self._input_shape) or any(
            isinstance(wanted,int) and wanted>=0 and wanted!=actual
            for wanted,actual in zip(self._input_shape,value.shape)
        ):
            raise ValueError("ONNX CPU input shape differs from the model")
        if not np.isfinite(value).all():
            raise ValueError("ONNX CPU input must be finite")
        # Match setInput ownership: subsequent caller mutation cannot alter it.
        self._value = np.array(value, dtype=np.float32, order="C", copy=True)

    def forward(self, names: Sequence[str] | None = None) -> Any:
        if self._value is None:
            raise ValueError("ONNX CPU forward requires a valid input")
        if names is None and len(self._output_names)!=1:
            raise ValueError("Multi-output ONNX CPU models require explicit output names")
        if isinstance(names,str):
            raise ValueError("ONNX CPU forward names must be a sequence")
        selected = self._output_names if names is None else tuple(names)
        if not selected or len(set(selected))!=len(selected) or any(n not in self._output_names for n in selected):
            raise ValueError("Unknown or duplicate ONNX CPU output name")
        result = self._session.run(list(selected), {self._input_name: self._value})
        if len(result)!=len(selected) or any(not isinstance(v,np.ndarray) or not np.isfinite(v).all() for v in result):
            raise ValueError("ONNX CPU returned invalid model outputs")
        return result[0] if names is None else tuple(result)
