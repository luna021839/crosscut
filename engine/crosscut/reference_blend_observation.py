"""Measure a convex two-source pixel hypothesis without inventing cut labels.

The caller supplies independently aligned source pixels and their common valid
support. A fit is not source identity, geometry proof, or dissolve supervision.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def measure_aligned_reference_blend(
    reference: np.ndarray,
    outgoing: np.ndarray,
    incoming: np.ndarray,
    valid_support: np.ndarray,
) -> dict[str, Any]:
    """Least-squares alpha for R ≈ (1-alpha) A + alpha B on shared support."""
    images = (reference, outgoing, incoming)
    if any(
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        for image in images
    ):
        raise ValueError("blend observation requires uint8 three-channel pixels")
    if any(image.shape != reference.shape for image in images):
        raise ValueError("blend observation pixels must share one aligned canvas")
    if (
        not isinstance(valid_support, np.ndarray)
        or valid_support.dtype != np.bool_
        or valid_support.shape != reference.shape[:2]
    ):
        raise ValueError("blend support must be a boolean mask on the aligned canvas")
    count = int(np.count_nonzero(valid_support))
    result: dict[str, Any] = {
        "contract": "aligned_convex_two_source_pixel_observation_v1",
        "support_pixel_count": count,
        "support_fraction": count / valid_support.size if valid_support.size else 0.0,
        "pixel_color_space": "encoded_bgr_u8_divided_by_255",
        "source_identity_inferred": False,
        "transition_kind_authorized": False,
    }
    if count == 0:
        return {**result, "status": "no_shared_support"}
    target, left, right = (
        image[valid_support].astype(np.float64) / 255.0 for image in images
    )
    direction = right - left
    energy = float(np.mean(direction * direction))
    result["source_separation_mse"] = energy
    if energy <= np.finfo(np.float64).eps:
        return {**result, "status": "indistinguishable_sources"}
    unbounded_alpha = float(np.mean((target - left) * direction) / energy)
    alpha = float(np.clip(unbounded_alpha, 0.0, 1.0))
    reconstructed = left + alpha * direction
    left_mse = float(np.mean((target - left) ** 2))
    right_mse = float(np.mean((target - right) ** 2))
    mixture_mse = float(np.mean((target - reconstructed) ** 2))
    best_endpoint = min(left_mse, right_mse)
    return {
        **result,
        "status": "measured",
        "incoming_alpha": alpha,
        "unbounded_incoming_alpha": unbounded_alpha,
        "outgoing_mse": left_mse,
        "incoming_mse": right_mse,
        "mixture_mse": mixture_mse,
        "relative_mse_improvement_over_best_endpoint": (
            (best_endpoint - mixture_mse) / best_endpoint if best_endpoint > 0.0 else 0.0
        ),
    }
