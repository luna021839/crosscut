"""Float32 Apple GPU executor for the pinned MIT TransNetV2 weights.

This is the same pretrained shot detector, not a learned editing model. Weight
names and shapes are checked before execution. The first window is compared
with TensorFlow; near-threshold windows retain TensorFlow's exact decision.
No decoder, frame, threshold, padding or source clock is changed here.

Architecture reference: soCzech/TransNetV2, commit
85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed (licenses/TransNetV2-MIT.txt).
"""
from __future__ import annotations

from typing import Any
import numpy as np

EXECUTOR_CONTRACT = "transnetv2_pinned_float32_mlx_reference_calibrated_v1"
CALIBRATION_LOGIT_ATOL = 1e-4
REFERENCE_THRESHOLD_MARGIN = 1e-3


def expected_weight_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for stage in range(1, 4):
        filters = 16 * 2 ** (stage - 1)
        incoming = 3 if stage == 1 else filters * 2
        for block in range(1, 3):
            key = f"SDDCNN_{stage}/DDCNN_{block}"
            channels = incoming if block == 1 else filters * 4
            for dilation in (1, 2, 4, 8):
                conv = f"{key}/Conv3D_{dilation}"
                shapes[conv + "/conv_spatial/kernel"] = (1, 3, 3, channels, filters * 2)
                shapes[conv + "/conv_temporal/kernel"] = (3, 1, 1, filters * 2, filters)
            for name in ("gamma", "beta", "moving_mean", "moving_variance"):
                shapes[f"{key}/bn/{name}"] = (filters * 4,)
    for name, incoming, outgoing in (
        ("dense", 4864, 1024), ("dense_1", 1024, 1), ("dense_2", 1024, 1),
        ("FrameSimilarity/dense_3", 448, 128),
        ("FrameSimilarity/dense_4", 101, 128),
        ("ColorHistograms/dense_5", 101, 128),
    ):
        shapes[name + "/kernel"] = (incoming, outgoing)
        shapes[name + "/bias"] = (outgoing,)
    return shapes


class MLXScenePort:
    def __init__(self, model: Any) -> None:
        import mlx.core as mx
        if not mx.metal.is_available():
            raise RuntimeError("TransNetV2 MLX executor needs an Apple GPU")
        weights = {}
        for variable in model.variables:
            name = variable.name.removeprefix("TransNet/").removesuffix(":0")
            if name in weights:
                raise ValueError("duplicate TransNetV2 weight name")
            weights[name] = variable.numpy()
        expected = expected_weight_shapes()
        if set(weights) != set(expected):
            raise ValueError("TransNetV2 weight names differ from the pinned architecture")
        for name, shape in expected.items():
            value = weights[name]
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"invalid TransNetV2 weight: {name}")
        self.mx = mx
        with mx.stream(mx.gpu):
            self.weights = {name: mx.array(value) for name, value in weights.items()}
            self.convolutions = {name: mx.transpose(value, (4, 0, 1, 2, 3))
                                 for name, value in self.weights.items() if value.ndim == 5}
            mx.eval(self.weights, self.convolutions)

    def _dense(self, values, name: str):
        return values @ self.weights[name + "/kernel"] + self.weights[name + "/bias"]

    def _similarities(self, values):
        mx = self.mx
        values = values * mx.rsqrt(mx.maximum(mx.sum(values * values, axis=-1, keepdims=True), 1e-12))
        similarities = values @ mx.swapaxes(values, -1, -2)
        count = values.shape[1]
        indices = mx.arange(count)[:, None] + mx.arange(-50, 51)[None, :]
        valid = (indices >= 0) & (indices < count)
        return mx.take_along_axis(similarities, mx.clip(indices, 0, count - 1)[None], axis=2) * valid

    def __call__(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if frames.shape != (1, 100, 27, 48, 3) or frames.dtype != np.uint8:
            raise ValueError("TransNetV2 MLX input must be one RGB uint8 100-frame window")
        with self.mx.stream(self.mx.gpu):
            return self._predict(frames)

    def _predict(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mx = self.mx
        values = mx.array(frames, dtype=mx.float32) / 255.0
        features = []
        for stage in range(1, 4):
            shortcut = None
            for block in range(1, 3):
                key = f"SDDCNN_{stage}/DDCNN_{block}"
                parts = []
                for dilation in (1, 2, 4, 8):
                    conv = f"{key}/Conv3D_{dilation}"
                    part = mx.conv3d(values, self.convolutions[conv + "/conv_spatial/kernel"], padding=(0, 1, 1))
                    part = mx.conv3d(part, self.convolutions[conv + "/conv_temporal/kernel"],
                                     padding=(dilation, 0, 0), dilation=(dilation, 1, 1))
                    parts.append(part)
                values = mx.concatenate(parts, axis=-1)
                weights = self.weights
                values = ((values - weights[key + "/bn/moving_mean"])
                          * mx.rsqrt(weights[key + "/bn/moving_variance"] + 0.001)
                          * weights[key + "/bn/gamma"] + weights[key + "/bn/beta"])
                if block == 1:
                    values = mx.maximum(values, 0)
                    shortcut = values
            values = mx.maximum(values, 0) + shortcut
            height, width = values.shape[2] // 2 * 2, values.shape[3] // 2 * 2
            values = (values[:, :, :height:2, :width:2] + values[:, :, 1:height:2, :width:2]
                      + values[:, :, :height:2, 1:width:2] + values[:, :, 1:height:2, 1:width:2]) * 0.25
            features.append(mx.mean(values, axis=(2, 3)))
        frame_similarities = self._similarities(self._dense(mx.concatenate(features, axis=-1), "FrameSimilarity/dense_3"))
        frame_features = mx.maximum(self._dense(frame_similarities, "FrameSimilarity/dense_4"), 0)
        bins = frames.astype(np.int32) // 32
        bins = (bins[..., 0] * 64 + bins[..., 1] * 8 + bins[..., 2]).reshape(100, -1)
        histograms = np.stack([np.bincount(row, minlength=512) for row in bins]).astype(np.float32)[None]
        color_features = mx.maximum(self._dense(self._similarities(mx.array(histograms)), "ColorHistograms/dense_5"), 0)
        values = mx.concatenate([color_features, frame_features, values.reshape(1, 100, -1)], axis=-1)
        values = mx.maximum(self._dense(values, "dense"), 0)
        single, many = self._dense(values, "dense_1"), self._dense(values, "dense_2")
        mx.eval(single, many)
        return np.array(single), np.array(many)


class CalibratedMLXSceneModel:
    def __init__(self, reference: Any, *, port: Any = None) -> None:
        self.reference = reference
        self.port = MLXScenePort(reference) if port is None else port
        self.calibrated = False
        self.disabled_reason: str | None = None
        self.accelerated_windows = 0
        self.reference_windows = 0
        self.calibration_max_logit_error: float | None = None

    def _reference(self, values):
        self.reference_windows += 1
        return self.reference(values)

    def __call__(self, values):
        if self.disabled_reason is not None:
            return self._reference(values)
        array = np.asarray(values)
        if (array.shape != (1, 100, 27, 48, 3) or not np.isfinite(array).all()
                or np.any(array < 0) or np.any(array > 255) or np.any(array != np.floor(array))):
            raise ValueError("invalid TransNetV2 integer-valued frame window")
        try:
            single, many = self.port(array.astype(np.uint8))
            if (single.shape != (1, 100, 1) or many.shape != single.shape
                    or not np.isfinite(single).all() or not np.isfinite(many).all()):
                raise ValueError("invalid MLX TransNetV2 output")
        except (RuntimeError, ValueError) as exc:
            self.disabled_reason = f"{type(exc).__name__}: {exc}"
            return self._reference(values)
        if not self.calibrated:
            reference_single, heads = self._reference(values)
            reference_many = heads["many_hot"]
            error = max(float(np.max(np.abs(np.asarray(reference_single) - single))),
                        float(np.max(np.abs(np.asarray(reference_many) - many))))
            self.calibration_max_logit_error = error
            if not np.isfinite(error) or error > CALIBRATION_LOGIT_ATOL:
                self.disabled_reason = "reference_calibration_mismatch"
            else:
                self.calibrated = True
            return reference_single, heads
        if np.any(np.abs(single) <= REFERENCE_THRESHOLD_MARGIN) or np.any(np.abs(many) <= REFERENCE_THRESHOLD_MARGIN):
            return self._reference(values)
        self.accelerated_windows += 1
        return single, {"many_hot": many}

    def diagnostics(self) -> dict[str, Any]:
        return {"executor_contract": EXECUTOR_CONTRACT, "calibrated": self.calibrated,
                "counter_scope": "cached_executor_lifetime",
                "disabled_reason": self.disabled_reason,
                "calibration_max_logit_error": self.calibration_max_logit_error,
                "accelerated_windows": self.accelerated_windows, "reference_windows": self.reference_windows}
