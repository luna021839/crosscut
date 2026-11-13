"""Observed no-op camera targets for dynamic supervision, not crop positives.

An already matching source cannot improve by the positive-crop delta. Without
this separate hypothesis, local frame search can prefer a worse wrong-phase
image solely because a small warp improves that worse image.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

import numpy as np

from .reference_crop_supervision import _EstimatedTransform, _warp_review


IDENTITY_OBSERVATION_POLICY = {
    "contract": "independently_pixel_verified_noop_camera_observation_v1",
    "minimum_identity_structure_ncc": 0.95,
    "maximum_estimated_corner_displacement_pixels": 1.0,
    "required_prior_rejection": "actual_warp_improvement_too_small",
}


def validate_identity_observation(diagnostics: Mapping[str, Any], keyframe: Mapping[str, Any], policy: Any) -> None:
    """Bind serialized no-op claims to their declared policy and exact target."""
    evidence = diagnostics.get("identity_observation")
    if evidence is None:
        return
    if policy != IDENTITY_OBSERVATION_POLICY or not isinstance(evidence, Mapping):
        raise ValueError("identity observation lacks its declared policy")
    if any(evidence.get(k) != v for k,v in IDENTITY_OBSERVATION_POLICY.items()):
        raise ValueError("identity observation policy differs")
    if (evidence.get("identity_pixels_remeasured") is not True
        or evidence.get("positive_crop_improvement_gate_relaxed") is not False
        or diagnostics.get("decision") != "verified_noop_source_reference_match"):
        raise ValueError("identity observation lacks independent pixel measurement")
    shift=evidence.get("maximum_estimated_corner_displacement_pixels_observed")
    review=diagnostics.get("actual_warp_review", {})
    ncc=review.get("warped_structure_ncc")
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (shift,ncc)):
        raise ValueError("identity observation has invalid measurements")
    if not 0 <= shift <= IDENTITY_OBSERVATION_POLICY["maximum_estimated_corner_displacement_pixels"]:
        raise ValueError("identity observation has excessive corner displacement")
    if ncc < IDENTITY_OBSERVATION_POLICY["minimum_identity_structure_ncc"] or review.get("valid_fraction") != 1.:
        raise ValueError("identity observation pixels do not pass")
    if tuple(keyframe.get(k) for k in ("zoom","center_x","center_y","roll_degrees")) != (1.,.5,.5,0.):
        raise ValueError("identity observation is bound to a nonidentity target")


def recover_verified_identity_sample(
    reference: np.ndarray,
    source: np.ndarray,
    rejection: str | None,
    diagnostics: Mapping[str, Any],
) -> tuple[_EstimatedTransform, dict[str, Any]] | None:
    """Test the real identity pixels after every geometric gate has passed.

    Never rescue decode, correspondence, perspective or renderer failures.
    No rejected affine is clamped. The identity is a separate measured model,
    allowed only when the estimated affine moves every image corner <= 1 px.
    """
    if rejection != IDENTITY_OBSERVATION_POLICY["required_prior_rejection"]:
        return None
    if reference.shape != source.shape or reference.ndim != 3:
        return None
    try:
        affine = np.asarray(diagnostics["affine_matrix_source_to_reference"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        return None
    height, width = reference.shape[:2]
    corners = np.asarray(((0., 0., 1.), (width-1., 0., 1.),
                          (0., height-1., 1.), (width-1., height-1., 1.)))
    maximum_shift = float(np.linalg.norm(corners @ affine.T - corners[:, :2], axis=1).max())
    if maximum_shift > IDENTITY_OBSERVATION_POLICY["maximum_estimated_corner_displacement_pixels"]:
        return None
    identity = np.asarray(((1., 0., 0.), (0., 1., 0.)))
    review = _warp_review(reference, source, identity)
    ncc = review["warped_structure_ncc"]
    if not math.isfinite(ncc) or ncc < IDENTITY_OBSERVATION_POLICY["minimum_identity_structure_ncc"]:
        return None
    if review["valid_fraction"] != 1.0:
        return None
    result = deepcopy(dict(diagnostics))
    result["identity_observation"] = {
        **IDENTITY_OBSERVATION_POLICY,
        "maximum_estimated_corner_displacement_pixels_observed": maximum_shift,
        "positive_crop_rejection": rejection,
        "positive_crop_estimated_renderer_transform": result.get("estimated_renderer_transform"),
        "positive_crop_actual_warp_review": result.get("actual_warp_review"),
        "positive_crop_renderer_safe_zoom_projection": result.get("renderer_safe_zoom_projection"),
        "identity_pixels_remeasured": True,
        "positive_crop_improvement_gate_relaxed": False,
    }
    result["estimated_renderer_transform"] = dict(zoom=1.,center_x=.5,center_y=.5,roll_degrees=0.)
    result["renderer_safe_zoom_projection"] = {
        "applied": False, "basis": "independently_measured_identity",
        "raw_target": result["estimated_renderer_transform"],
    }
    result["actual_warp_review"] = review
    result["decision"] = "verified_noop_source_reference_match"
    return _EstimatedTransform(identity,1.,.5,.5,0.,result), result
