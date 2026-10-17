"""Differentiable risk of played v3 camera curves on real observed clocks.

Each count retains its actual decoder time branch and prefix of value knots.
The hinge sum below is exactly the renderer's clamped linear interpolation,
including its constant tail. No target positions/times are interpolated or
invented. Count probabilities remain differentiable; count CE is kept separate.
"""
from __future__ import annotations

from typing import Any

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - optional outside Apple Silicon.
    mx = None


DECODED_CURVE_RISK_CONTRACT = "expected_count_renderer_linear_curve_original_observation_normalized_L1_v1"


def decoded_curve_risk(
    values: Any,
    times_by_count: Any,
    count_logits: Any,
    observed_times: Any,
    observed_values: Any,
    observed_mask: Any = None,
) -> tuple[Any, Any]:
    """Return expected risk and all eight conditional risks per trajectory."""
    if mx is None:
        raise RuntimeError("Decoded Dynamic Crop curve loss requires MLX")
    prefix = values.shape[:-2]
    if (values.shape[-2:] != (8, 4)
            or times_by_count.shape != (*prefix, 8, 8)
            or count_logits.shape != (*prefix, 8)
            or observed_times.shape[:-1] != prefix
            or observed_times.shape[-1] < 1
            or observed_values.shape != (*observed_times.shape, 4)):
        raise ValueError("One source-bound trajectory and original observation clock required")
    if observed_mask is None:
        observed_mask = mx.ones(observed_times.shape, dtype=mx.bool_)
    if observed_mask.shape != observed_times.shape:
        raise ValueError("Original observation mask shape differs")
    scales = mx.array([.18, .04, .04, 1.60], dtype=values.dtype)
    weights = observed_mask.astype(values.dtype)
    denominator = mx.maximum(weights.sum(axis=-1) * 4, 1.)
    risks = []
    for count in range(1, 9):
        predicted = mx.broadcast_to(values[..., :1, :], observed_values.shape)
        if count > 1:
            times = times_by_count[..., count - 1, :count]
            elapsed = observed_times[..., :, None] - times[..., None, :-1]
            gaps = times[..., 1:] - times[..., :-1]
            fraction = mx.minimum(1., mx.maximum(0., elapsed / gaps[..., None, :]))
            deltas = values[..., 1:count, :] - values[..., :count - 1, :]
            predicted = predicted + (fraction[..., None] * deltas[..., None, :, :]).sum(axis=-2)
        residual = mx.abs((predicted - observed_values) / scales)
        risks.append((residual * weights[..., None]).sum(axis=(-2, -1)) / denominator)
    conditional = mx.stack(risks, axis=-1)
    expected = (mx.softmax(count_logits, axis=-1) * conditional).sum(axis=-1)
    return expected, conditional
